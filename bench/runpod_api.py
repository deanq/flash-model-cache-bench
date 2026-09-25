"""Thin Runpod API client for what the Flash CLI does not cover.

- invoking deployed endpoints and reading job timing (/run, /status, /health)
- attaching a Model Store cached model (GraphQL `modelReferences`) -- Flash
  1.20 has no field for it, and the REST API does not expose it either
- creating/attaching/deleting a Global Store volume (GraphQL
  `globalStoreBucket*` and saveEndpoint `volumes`) -- same story
- deleting the network volume and listing leftovers during teardown

Lifecycle (deploy/undeploy) stays with the `flash` CLI.
"""
from __future__ import annotations

import os
import time
import tomllib
from pathlib import Path

import httpx

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
SERVERLESS = "https://api.runpod.ai/v2"
# Cloudflare in front of rest.runpod.io rejects requests without a User-Agent (403/1010).
USER_AGENT = "flash-model-cache-bench/0.1"
TIMEOUT = httpx.Timeout(60.0, connect=15.0)
PAUSED_POLL_S = 2.0


class RunpodError(RuntimeError):
    pass


def api_key() -> str:
    """RUNPOD_API_KEY, else the key `flash login` / runpodctl stored."""
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.is_file():
        data = tomllib.loads(cfg.read_text())
        key = data.get("api_key") or data.get("default", {}).get("api_key")
        if key:
            return key
    raise RunpodError("no Runpod API key: set RUNPOD_API_KEY in .env or run `uv run flash login`")


class Runpod:
    def __init__(self, key: str | None = None, client: httpx.Client | None = None):
        self._key = key or api_key()
        self._http = client or httpx.Client(timeout=TIMEOUT, headers={
            "Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT})

    # ------------------------------------------------------------------ jobs
    def run(self, endpoint_id: str, payload: dict, *, paused_retry_s: float = 90.0) -> str:
        """Submit a job. Right after workersMax is restored the endpoint can still
        answer 409 ENDPOINT_PAUSED for a few seconds; retry through that window."""
        deadline = time.monotonic() + paused_retry_s
        while True:
            r = self._http.post(f"{SERVERLESS}/{endpoint_id}/run", json={"input": payload})
            if r.status_code == 409 and "ENDPOINT_PAUSED" in r.text and time.monotonic() < deadline:
                time.sleep(PAUSED_POLL_S)
                continue
            _raise_for(r, f"submit job to {endpoint_id}")
            return r.json()["id"]

    def status(self, endpoint_id: str, job_id: str) -> dict:
        r = self._http.get(f"{SERVERLESS}/{endpoint_id}/status/{job_id}")
        _raise_for(r, f"status of job {job_id} on {endpoint_id}")
        return r.json()

    def cancel(self, endpoint_id: str, job_id: str) -> None:
        self._http.post(f"{SERVERLESS}/{endpoint_id}/cancel/{job_id}")

    def health(self, endpoint_id: str) -> dict:
        r = self._http.get(f"{SERVERLESS}/{endpoint_id}/health")
        _raise_for(r, f"health of {endpoint_id}")
        return r.json()

    # ------------------------------------------------------------- graphql
    def graphql(self, query: str, variables: dict | None = None) -> dict:
        r = self._http.post(GRAPHQL, json={"query": query, "variables": variables or {}})
        _raise_for(r, "graphql")
        body = r.json()
        if body.get("errors"):
            raise RunpodError(f"graphql errors: {body['errors']}")
        return body["data"]

    def endpoint(self, endpoint_id: str) -> dict:
        eps = self.graphql(_ENDPOINTS_QUERY)["myself"]["endpoints"]
        ep = next((e for e in eps if e["id"] == endpoint_id), None)
        if not ep:
            raise RunpodError(f"endpoint {endpoint_id} not found")
        return ep

    def update_endpoint(self, endpoint_id: str, **changes) -> dict:
        """saveEndpoint is an upsert: send the full current config back with only
        `changes` applied. modelReferences and attached global volumes are carried
        over so a later update (e.g. a workersMax toggle) does not detach them."""
        ep = self.endpoint(endpoint_id)
        payload = {k: ep[k] for k in _SAVE_FIELDS if ep.get(k) is not None}
        if ep.get("modelReferences"):
            payload["modelReferences"] = ep["modelReferences"]
        global_volumes = _global_volume_inputs(ep.get("volumes"))
        if global_volumes:
            payload["volumes"] = global_volumes
        payload.update(changes)
        return self.graphql(_SAVE_MUTATION, {"input": payload})["saveEndpoint"]

    def attach_global_volume(self, endpoint_id: str, volume_id: str) -> dict:
        """Attach a Global Store volume (mounts at /runpod-volume). Flash 1.20 and
        REST v2 cannot; this is the console's saveEndpoint `volumes` input."""
        self.update_endpoint(endpoint_id, volumes=[{"volumeId": volume_id, "volumeType": OBJECT_STORE_VOLUME}])
        attached = [v["volumeId"] for v in self.endpoint(endpoint_id).get("volumes") or []
                    if v.get("volumeType") == OBJECT_STORE_VOLUME]
        if attached != [volume_id]:
            raise RunpodError(f"global volume {volume_id} not attached to {endpoint_id}: {attached}")
        return {"id": endpoint_id, "globalVolume": volume_id}

    def set_model_references(self, endpoint_id: str, refs: list[str]) -> dict:
        """Attach Model Store cached model(s)."""
        saved = self.update_endpoint(endpoint_id, modelReferences=refs)
        # The API normalizes "org/name" to "https://huggingface.co/Org/Name:<commit>".
        persisted = [r.lower() for r in saved.get("modelReferences") or []]
        if len(persisted) != len(refs) or not all(ref.split(":")[0] in got for ref, got in zip(refs, persisted)):
            raise RunpodError(f"saveEndpoint did not persist modelReferences={refs}: {saved}")
        return saved

    def set_workers_max(self, endpoint_id: str, n: int) -> dict:
        return self.update_endpoint(endpoint_id, workersMax=n)

    def list_endpoints(self) -> list[dict]:
        return self.graphql("query { myself { endpoints { id name } } }")["myself"]["endpoints"]

    # --------------------------------------------------------- global store
    def create_global_store(self, name: str) -> str:
        data = self.graphql(_GS_CREATE, {"input": {"name": name}})
        return data["globalStoreBucketCreate"]["id"]

    def list_global_stores(self) -> list[dict]:
        return self.graphql("query { myself { globalStoreBuckets { id name } } }")["myself"]["globalStoreBuckets"]

    def delete_global_store(self, bucket_id: str) -> None:
        if not self.graphql(_GS_DELETE, {"input": {"id": bucket_id}})["globalStoreBucketDelete"]:
            raise RunpodError(f"globalStoreBucketDelete returned false for {bucket_id}")

    # -------------------------------------------------------------- volumes
    def list_network_volumes(self) -> list[dict]:
        r = self._http.get(f"{REST}/networkvolumes")
        _raise_for(r, "list network volumes")
        return r.json()

    def delete_network_volume(self, volume_id: str) -> None:
        r = self._http.delete(f"{REST}/networkvolumes/{volume_id}")
        if r.status_code == 404:
            return
        _raise_for(r, f"delete network volume {volume_id}")


def model_reference(repo_id: str, revision: str | None = None) -> str:
    """Console format: org/name[:revision], lowercased."""
    ref = repo_id if not revision else f"{repo_id}:{revision}"
    return ref.lower()


def _global_volume_inputs(volumes) -> list[dict]:
    return [{"volumeId": v["volumeId"], "volumeType": OBJECT_STORE_VOLUME}
            for v in volumes or [] if v.get("volumeType") == OBJECT_STORE_VOLUME]


def _raise_for(r: httpx.Response, what: str) -> None:
    if r.is_success:
        return
    raise RunpodError(f"{what}: HTTP {r.status_code}: {r.text[:500]}")


OBJECT_STORE_VOLUME = "OBJECT_STORE_VOLUME"   # Global Store volume type in EndpointInput.volumes

_SAVE_FIELDS = ("id", "name", "templateId", "gpuIds", "gpuCount", "workersMin", "workersMax", "idleTimeout",
                "locations", "networkVolumeId", "scalerType", "scalerValue", "flashBootType",
                "executionTimeoutMs", "type", "allowedCudaVersions", "minCudaVersion", "flashEnvironmentId")

_ENDPOINTS_QUERY = """
query {
  myself { endpoints {
    id name templateId gpuIds gpuCount workersMin workersMax idleTimeout locations
    networkVolumeId scalerType scalerValue flashBootType executionTimeoutMs type
    allowedCudaVersions minCudaVersion flashEnvironmentId modelReferences
    volumes { volumeId volumeType mountPath }
  } }
}"""

_SAVE_MUTATION = """
mutation saveEndpoint($input: EndpointInput!) {
  saveEndpoint(input: $input) { id name modelReferences workersMax idleTimeout }
}"""

_GS_CREATE = """
mutation globalStoreBucketCreate($input: GlobalStoreBucketCreateInput!) {
  globalStoreBucketCreate(input: $input) { id name }
}"""

_GS_DELETE = """
mutation globalStoreBucketDelete($input: GlobalStoreBucketDeleteInput!) {
  globalStoreBucketDelete(input: $input)
}"""
