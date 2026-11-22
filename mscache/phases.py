"""Per-load records: every phase as a UTC timestamp, joinable with host logs on pod/machine/time.

Phases: submit -> pod_created -> pod_last_started (lastStartedAt, "resumed") -> container_start
(pid 1 start, in-container) -> handler_ready -> job_start -> job_done_seen (first poll that saw it).
Model staging happens before the container starts (the copy is mounted, not downloaded in-container),
so staging_s = container_start - pod_created (or submit, for a restarted pod that kept its id). It includes the ai-api wait (0 or ~45 s), so the
outcome here is an estimate; the main Mac's host-log join (peer/origin per file) is final.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath

# cached loads stage in 2-5 s, or 45-60 s when the ai-api wait lands here; cold pulls ran 130 s+
CACHED_STAGING_MAX_S = 75


def iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def epoch(ts) -> float | None:
    if ts in (None, ""):
        return None
    if isinstance(ts, (int, float)) or str(ts).isdigit():
        return float(ts) / 1000
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def _in_container(out: dict) -> dict[str, float | None]:
    now = out.get("now")
    if now is None:
        return {"container_start": None, "handler_ready": None, "job_start": None}
    age = out.get("container_age_s")
    handler = out.get("handler_age_s")
    return {"container_start": now - age if age is not None else None,
            "handler_ready": now - handler if handler is not None else None,
            "job_start": now}


def _outcome(status: str | None, staging_s: float | None, no_worker: bool) -> str:
    if no_worker:
        return "no_worker"
    if status != "COMPLETED":
        return "failed"
    if staging_s is None:
        return "unknown"
    return "cached" if staging_s <= CACHED_STAGING_MAX_S else "pull"


def load_record(*, submitted: float, status: dict, pod: dict, done_seen: float, no_worker: bool = False) -> dict:
    out = status.get("output") if isinstance(status.get("output"), dict) else {}
    t = {"submit": submitted,
         "pod_created": epoch(pod.get("createdAt")),
         "pod_last_started": epoch(pod.get("lastStartedAt")),
         **_in_container(out),
         "job_done_seen": done_seen}
    # a restarted pod keeps its old createdAt; its staging cannot start before this load's submit
    reused = t["pod_created"] is not None and t["pod_created"] < submitted
    anchor = max(t["pod_created"], submitted) if t["pod_created"] is not None else None
    staging = (round(t["container_start"] - anchor, 1)
               if t["container_start"] is not None and anchor is not None else None)
    snapshot = out.get("snapshot")
    machine = pod.get("machine") or {}
    return {
        "kind": "load",
        "status": status.get("status"),
        "job_id": status.get("id"),
        "pod_id": pod.get("id") or status.get("workerId"),
        "machine_id": pod.get("machineId"),
        "gpu_name": machine.get("gpuDisplayName"),
        "location": machine.get("location"),
        "revision": PurePosixPath(snapshot).name if snapshot else None,
        "n_files": out.get("n_files"),
        "staged_bytes": out.get("staged_bytes"),
        "first_job_on_worker": out.get("first_job_on_worker"),
        "delay_ms": status.get("delayTime"),
        "exec_ms": status.get("executionTime"),
        "submitted": submitted,
        "phases_utc": {k: iso(v) for k, v in t.items()},
        "offsets_s": {k: round(v - submitted, 1) for k, v in t.items() if k != "submit" and v is not None},
        "pod_reused": reused,
        "staging_s": staging,
        "outcome": _outcome(status.get("status"), staging, no_worker),
        "error": status.get("error"),
    }
