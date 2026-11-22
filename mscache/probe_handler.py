"""MSCACHE worker: reports where/when the Model Store copy was staged. No GPU work.

Runs from env PROBE_SRC inside python:3.12-slim. Returns machine identity, container/handler ages,
the snapshot path (its last component is the commit), file count and staged bytes, then holds the
worker for `hold_s` so the host reports the copy DEPLOYED before teardown.
"""
import glob
import os
import socket
import time

import runpod

HUB = "/runpod-volume/huggingface-cache/hub"
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
PROCESS_START = time.time()
first_job = True


def _safe_env() -> dict:
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith("RUNPOD_") and not any(m in k for m in SECRET_MARKERS)}


def _container_age_s() -> float:
    with open("/proc/1/stat") as f:   # pid 1 start (clock ticks since boot) vs uptime
        start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/uptime") as f:
        uptime = float(f.read().split()[0])
    return uptime - start_ticks / os.sysconf("SC_CLK_TCK")


def _snapshot() -> tuple[str | None, list[os.stat_result]]:
    snaps = glob.glob(f"{HUB}/models--*/snapshots/*")
    if not snaps:
        return None, []
    files = [os.path.join(r, n) for r, _, ns in os.walk(snaps[0]) for n in ns]
    return snaps[0], [os.stat(p) for p in files]   # stat follows the blob symlinks


def handler(job):
    global first_job
    is_first, first_job = first_job, False
    snap, stats = _snapshot()
    out = {"hostname": socket.gethostname(), "env": _safe_env(), "first_job_on_worker": is_first,
           "container_age_s": round(_container_age_s(), 1), "handler_age_s": round(time.time() - PROCESS_START, 1),
           "now": time.time(), "snapshot": snap, "n_files": len(stats),
           "staged_bytes": sum(s.st_size for s in stats),
           "staged_mtime_min": min((s.st_mtime for s in stats), default=None),
           "staged_mtime_max": max((s.st_mtime for s in stats), default=None),
           "hub_listing": sorted(os.listdir(HUB)) if os.path.isdir(HUB) else None}
    time.sleep(float(job["input"].get("hold_s", 0)))
    return out


runpod.serverless.start({"handler": handler})
