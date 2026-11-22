"""Live Runpod adapter for mscache.driver.Client: public REST v2 + GraphQL only (customer path).

Endpoint creation is REST v2 (pinned DC, one GPU pool, workers 0/0) followed by a GraphQL
saveEndpoint full upsert that sets modelReferences; the upsert echoes every field it must keep.
"""
from __future__ import annotations

import time
from pathlib import Path

from bench.runpod_api import REST_V2, Runpod, RunpodError, _ENDPOINTS_QUERY, _raise_for
from mscache import spec

ENDPOINTS_QUERY = _ENDPOINTS_QUERY.replace("modelReferences", "modelReferences instanceIds")
PODS_QUERY = ("query { myself { endpoints { id pods { id machineId desiredStatus createdAt lastStartedAt "
              "machine { gpuDisplayName location } } } } }")
SAVE = "mutation saveEndpoint($input: EndpointInput!) { saveEndpoint(input: $input) { id modelReferences } }"
SAVE_FIELDS = ("id", "name", "templateId", "gpuIds", "gpuCount", "workersMin", "workersMax", "idleTimeout",
               "locations", "networkVolumeId", "scalerType", "scalerValue", "flashBootType",
               "executionTimeoutMs", "type", "allowedCudaVersions", "minCudaVersion", "instanceIds")
HANDLER = Path(__file__).with_name("probe_handler.py")
DISK_GB = 20               # Model Store mounts the copy from the host; container disk does not hold it
DELETE_RETRIES = 12        # workers still terminating make delete 400 "Try again"
DELETE_RETRY_S = 10


class LiveClient:
    def __init__(self, rp: Runpod | None = None):
        self.rp = rp or Runpod()

    def _body(self, name: str, sc: spec.Scenario) -> dict:
        gpu = spec.GPUS[sc.gpu]
        pools = {"pools": [gpu.pool], **({"excludedTypes": list(gpu.excluded)} if gpu.excluded else {})}
        return {"name": name, "type": "QUEUE", "image": "python:3.12-slim",
                "entrypoint": ["/bin/bash", "-c"],
                "cmd": ['pip install -q --no-cache-dir --root-user-action=ignore runpod && exec python -u -c "$PROBE_SRC"'],
                "env": {"PROBE_SRC": HANDLER.read_text()}, "disk": DISK_GB, "dataCenterIds": [sc.dc],
                "workers": {"min": 0, "max": 0, "idleTimeout": spec.IDLE_TIMEOUT_S},
                "scaling": {"type": "QUEUE_DELAY", "queueDelay": 1}, "timeout": 600_000,   # caps a hung job; staging is before exec
                "flashboot": "OFF", "gpu": pools}

    def create(self, name: str, sc: spec.Scenario) -> dict:
        r = self.rp._http.post(f"{REST_V2}/serverless", json=self._body(name, sc))
        _raise_for(r, f"create endpoint {name}")
        ep = r.json()["id"]
        try:
            cur = next(e for e in self.rp.graphql(ENDPOINTS_QUERY)["myself"]["endpoints"] if e["id"] == ep)
            payload = {k: cur[k] for k in SAVE_FIELDS if cur.get(k) not in (None, "", [])}
            payload["modelReferences"] = [sc.model.lower()]
            refs = self.rp.graphql(SAVE, {"input": payload})["saveEndpoint"]["modelReferences"]
            if not refs:
                raise RunpodError(f"modelReferences not saved on {ep} for {sc.model}")
        except Exception:
            self.delete(ep)   # never leave a half-made endpoint behind
            raise
        return {"id": ep, "model_ref": refs[0]}

    def delete(self, ep: str) -> None:
        for _ in range(DELETE_RETRIES):
            r = self.rp._http.delete(f"{REST_V2}/serverless/{ep}")
            if r.is_success or r.status_code == 404:
                return
            time.sleep(DELETE_RETRY_S)
        _raise_for(r, f"delete endpoint {ep}")

    def set_workers_max(self, ep: str, n: int) -> None:
        self.rp.set_workers_max(ep, n)

    def submit(self, ep: str, payload: dict) -> str:
        return self.rp.run(ep, payload)

    def status(self, ep: str, job_id: str) -> dict:
        return self.rp.status(ep, job_id)

    def cancel(self, ep: str, job_id: str) -> None:
        self.rp.cancel(ep, job_id)

    def pods(self, ep: str) -> dict[str, dict]:
        eps = self.rp.graphql(PODS_QUERY)["myself"]["endpoints"]
        return {p["id"]: p for e in eps if e["id"] == ep for p in e["pods"] or []}

    def leftovers(self) -> list[dict]:
        return [e for e in self.rp.list_endpoints() if e["name"].startswith(spec.NAME_PREFIX)]
