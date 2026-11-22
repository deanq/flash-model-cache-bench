"""Execute one Scenario on one fresh endpoint; emit one JSON record per load, pod-only pod and event.

The endpoint is deleted in `finally`, whatever happens. A load with no worker (or a stall) ends the
scenario: later steps would measure nothing and keep a paid endpoint around.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from mscache import phases, spec

POLL_S = 5
DONE = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
GONE = {"EXITED", "TERMINATED"}


class Client(Protocol):
    def create(self, name: str, scenario: spec.Scenario) -> dict: ...
    def delete(self, ep: str) -> None: ...
    def set_workers_max(self, ep: str, n: int) -> None: ...
    def submit(self, ep: str, payload: dict) -> str: ...
    def status(self, ep: str, job_id: str) -> dict: ...
    def cancel(self, ep: str, job_id: str) -> None: ...
    def pods(self, ep: str) -> dict[str, dict]: ...


@dataclass
class ScenarioResult:
    ok: bool
    reason: str = ""
    loads: list[dict] = field(default_factory=list)


class _Stop(Exception):
    """A load got no worker or stalled; stop the scenario cleanly."""


class Driver:
    def __init__(self, client: Client, *, run_id: str, sink: Callable[[dict], None],
                 now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.rp, self.run_id, self.sink, self.now, self.sleep = client, run_id, sink, now, sleep

    # ------------------------------------------------------------ plumbing
    def _emit(self, rec: dict) -> dict:
        rec = {**self._ctx, **rec}
        self.sink(rec)
        return rec

    def _event(self, event: str, **kw) -> None:
        self._emit({"kind": "event", "event": event, "t": self.now(), "t_utc": phases.iso(self.now()), **kw})

    # ---------------------------------------------------------------- run
    def run(self, sc: spec.Scenario) -> ScenarioResult:
        self._ctx = {"run_id": self.run_id, "scenario": sc.sid, "arm": sc.arm, "dc": sc.dc,
                     "gpu": sc.gpu, "model": sc.model}
        created = self.rp.create(sc.endpoint_name(self.run_id), sc)
        ep = created["id"]
        self._ctx |= {"endpoint_id": ep, "model_ref": created.get("model_ref")}
        self._event("endpoint_created")
        result = ScenarioResult(ok=True)
        try:
            for i, step in enumerate(sc.steps):
                self._step(ep, i, step, result)
        except _Stop as stop:
            result.ok, result.reason = False, str(stop)
        finally:
            self._cleanup(ep)
        return result

    def _cleanup(self, ep: str) -> None:
        try:
            self.rp.set_workers_max(ep, 0)
        finally:
            self.rp.delete(ep)
            self._event("endpoint_deleted")

    def _step(self, ep: str, i: int, step: spec.Step, result: ScenarioResult) -> None:
        match step:
            case spec.Deploy():
                result.loads.append(self._deploy(ep, i, step, result.loads))
            case spec.Teardown():
                self.rp.set_workers_max(ep, 0)
                self._wait_no_live_pods(ep, "teardown")
            case spec.AwaitScaleDown():
                self._wait_no_live_pods(ep, "scale_down")
            case spec.Idle(seconds=s):
                self.sleep(s)

    def _wait_no_live_pods(self, ep: str, what: str) -> None:
        deadline = self.now() + spec.TEARDOWN_TIMEOUT_S
        while self.now() < deadline:
            if all(p["desiredStatus"] in GONE for p in self.rp.pods(ep).values()):
                self._event(f"{what}_done")
                return
            self.sleep(POLL_S)
        self._event(f"{what}_timeout", live=[k for k, p in self.rp.pods(ep).items() if p["desiredStatus"] not in GONE])

    # ------------------------------------------------------------- deploy
    def _deploy(self, ep: str, i: int, d: spec.Deploy, prev: list[dict]) -> dict:
        stale = set(self.rp.pods(ep))          # pods listed from earlier rounds are not this load's
        if d.scale_up:
            self.rp.set_workers_max(ep, 1)
        submitted = self.now()
        jid = self.rp.submit(ep, {"hold_s": d.hold_s})
        status, seen, failure = self._poll(ep, jid, submitted, stale)
        pod = seen.get(status.get("workerId") or "", {})
        rec = phases.load_record(submitted=submitted, status={"id": jid, **status}, pod=pod,
                                 done_seen=self.now(), no_worker=failure == "no_worker")
        last = next((r["machine_id"] for r in reversed(prev) if r.get("machine_id")), None)
        rec |= {"role": d.role, "step": i, "hold_s": d.hold_s, "failure": failure,
                "same_machine_as_prev": (rec["machine_id"] == last) if last and rec["machine_id"] else None}
        rec = self._emit(rec)
        for pid, p in seen.items():
            if pid != rec["pod_id"]:
                self._emit({"kind": "pod_only", "step": i, "pod_id": pid, "machine_id": p.get("machineId"),
                            "pod_created_utc": phases.iso(phases.epoch(p.get("createdAt"))),
                            "pod_last_started_utc": phases.iso(phases.epoch(p.get("lastStartedAt")))})
        if failure:
            raise _Stop(f"{failure} at step {i} ({rec['outcome']})")
        return rec

    def _poll(self, ep: str, jid: str, submitted: float, stale: set[str]) -> tuple[dict, dict, str | None]:
        """(final status, this load's pods by id, failure or None)."""
        seen: dict[str, dict] = {}
        last_live = None
        while True:
            self.sleep(POLL_S)
            seen |= {k: v for k, v in self.rp.pods(ep).items() if k not in stale}
            status = self.rp.status(ep, jid)
            if status.get("status") in DONE:
                return status, seen, None if status["status"] == "COMPLETED" else "job_" + status["status"].lower()
            t = self.now()
            if any(p["desiredStatus"] not in GONE for p in seen.values()) or (seen and last_live is None):
                last_live = t   # stall clock starts when the first pod appears
            failure = self._failure(t, submitted, seen, last_live)
            if failure:
                self.rp.cancel(ep, jid)
                return {**status, "status": "CANCELLED"}, seen, failure

    @staticmethod
    def _failure(t: float, submitted: float, seen: dict, last_live: float | None) -> str | None:
        if not seen and t - submitted > spec.ABANDON_S:
            return "no_worker"
        if seen and last_live is not None and t - last_live > spec.STALL_S:
            return "stall"
        if t - submitted > spec.ROUND_TIMEOUT_S:
            return "timeout"
        return None
