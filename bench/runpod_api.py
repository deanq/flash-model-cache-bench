"""Thin Runpod API client for what `rp flash` does not cover.

- invoking deployed endpoints and reading job timing (/run, /status, /health)
- toggling workersMax to force cold starts (REST v2 PATCH, as `rp endpoint scale` does)
- reading back a deployed environment's endpoints and their attachments
- deleting the storage `rp flash app delete` leaves behind (network volume,
  global volume) and listing leftovers during teardown

Lifecycle (deploy, Model Store and Global Store attachment, undeploy) is
`rp flash` (runpod>=2.1.0): `model=` and `GlobalVolume` mounts replaced the
console GraphQL calls this harness needed with Flash 1.20.
"""
from __future__ import annotations

import asyncio
import os
import time
import tomllib
from pathlib import Path

import httpx

REST = "https://rest.runpod.io/v1"
REST_V2 = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"
SERVERLESS = "https://api.runpod.ai/v2"
# Cloudflare in front of rest.runpod.io rejects requests without a User-Agent (403/1010).
USER_AGENT = "flash-model-cache-bench/0.2"
TIMEOUT = httpx.Timeout(60.0, connect=15.0)
PAUSED_POLL_S = 2.0


class RunpodError(RuntimeError):
    pass


def api_key() -> str:
    """RUNPOD_API_KEY, else the key `rp login` / runpodctl stored."""
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.is_file():
        data = tomllib.loads(cfg.read_text())
        key = data.get("api_key") or data.get("default", {}).get("api_key")
        if key:
            return key
    raise RunpodError("no Runpod API key: set RUNPOD_API_KEY in .env or run `uv run rp login`")


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
        """Read-only view of a deployed endpoint, used to verify what `rp flash deploy` attached."""
        eps = self.graphql(_ENDPOINTS_QUERY)["myself"]["endpoints"]
        ep = next((e for e in eps if e["id"] == endpoint_id), None)
        if not ep:
            raise RunpodError(f"endpoint {endpoint_id} not found")
        return ep

    def set_workers_max(self, endpoint_id: str, n: int) -> dict:
        """REST v2 partial update, like `rp endpoint scale --max-workers`: only the
        workers block changes, so modelReferences, volumes and the Flash
        environment binding are left alone (a GraphQL saveEndpoint is a full upsert)."""
        url = f"{REST_V2}/serverless/{endpoint_id}"
        current = self._http.get(url)
        _raise_for(current, f"read endpoint {endpoint_id}")
        workers = {**(current.json().get("workers") or {}), "max": n}
        r = self._http.patch(url, json={"workers": workers})
        _raise_for(r, f"set workersMax={n} on {endpoint_id}")
        return r.json()

    def list_endpoints(self) -> list[dict]:
        return self.graphql("query { myself { endpoints { id name } } }")["myself"]["endpoints"]

    # ----------------------------------------------------------- flash apps
    def flash_environment(self, app: str, env: str) -> dict | None:
        """The deployed environment with its endpoints ({id, name}), or None."""
        return self._apps(lambda c: c.get_environment_by_name(app, env))

    def flash_app_exists(self, app: str) -> bool:
        return self._apps(lambda c: c.get_app_by_name(app)) is not None

    def _apps(self, call):
        from runpod.apps.api import AppsApiClient

        async def go():
            return await call(AppsApiClient(api_key=self._key))
        try:
            return asyncio.run(go())
        except Exception as e:  # noqa: BLE001 - SDK raises several types; callers handle one
            raise RunpodError(f"flash app API: {type(e).__name__}: {e}") from e

    # --------------------------------------------------------- global store
    def list_global_stores(self) -> list[dict]:
        """Global volumes (REST v2), the API `runpod.GlobalVolume` creates them with."""
        from runpod.apps.api import AppsApiClient

        async def go():
            return await AppsApiClient(api_key=self._key).list_global_volumes()
        try:
            return asyncio.run(go())
        except Exception as e:  # noqa: BLE001
            raise RunpodError(f"list global volumes: {type(e).__name__}: {e}") from e

    def delete_global_store(self, volume_id: str) -> None:
        r = self._http.delete(f"{REST_V2}/global-volumes/{volume_id}")
        if r.status_code == 404:
            return
        _raise_for(r, f"delete global volume {volume_id}")

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


def _raise_for(r: httpx.Response, what: str) -> None:
    if r.is_success:
        return
    raise RunpodError(f"{what}: HTTP {r.status_code}: {r.text[:500]}")


_ENDPOINTS_QUERY = """
query {
  myself { endpoints {
    id name templateId gpuIds gpuCount workersMin workersMax idleTimeout locations
    networkVolumeId scalerType scalerValue flashBootType executionTimeoutMs type
    allowedCudaVersions minCudaVersion flashEnvironmentId modelReferences
    volumes { volumeId volumeType mountPath }
  } }
}"""
