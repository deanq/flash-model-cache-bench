"""Execute one Scenario on one fresh endpoint; emit one JSON record per load, pod-only pod and event.

The endpoint is deleted in `finally`, whatever happens. A load with no worker (or a stall) ends the
scenario: later steps would measure nothing and keep a paid endpoint around.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from mscache import phases, spec

POLL_S = 5
INTERRUPT_POLL_S = 1   # S3 cuts at pod_created + 10 s; a 5 s poll would blur that
DONE = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
GONE = {"EXITED", "TERMINATED"}
LIVE_WORKERS = ("initializing", "running", "idle", "ready")


class Client(Protocol):
    def create(self, name: str, scenario: spec.Scenario) -> dict: ...
    def delete(self, ep: str) -> None: ...
    def set_workers_max(self, ep: str, n: int) -> None: ...
    def submit(self, ep: str, payload: dict) -> str: ...
    def status(self, ep: str, job_id: str) -> dict: ...
    def cancel(self, ep: str, job_id: str) -> None: ...
    def pods(self, ep: str) -> dict[str, dict]: ...
    def workers(self, ep: str) -> dict[str, int]: ...


@dataclass
class ScenarioResult:
    ok: bool
    reason: str = ""
    loads: list[dict] = field(default_factory=list)


class _Stop(Exception):
    """A load got no worker or stalled; stop the scenario cleanly."""


class Driver:
    def __init__(self, client: Client, *, run_id: str, sink: Callable[[dict], None],
                 now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 signals: dict[str, threading.Event] | None = None):
        self.rp, self.run_id, self.sink, self.now, self.sleep = client, run_id, sink, now, sleep
        # shared by every Driver of one run, so S4/S5 arms on separate endpoints can line up
        self.signals = {} if signals is None else signals

    def _signal(self, tag: str) -> threading.Event:
        return self.signals.setdefault(tag, threading.Event())

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
        self._sc, self._n_eps, self._ep = sc, 0, None
        result = ScenarioResult(ok=True)
        try:
            self._create()
            for i, step in enumerate(sc.steps):
                self._step(self._ep, i, step, result)
        except _Stop as stop:
            result.ok, result.reason = False, str(stop)
        finally:
            self._cleanup()
        return result

    def _create(self) -> None:
        """A rejected create (S8: bad revision, unreadable repo) is a result, not a crash."""
        try:
            created = self.rp.create(self._sc.endpoint_name(self.run_id, self._n_eps), self._sc)
        except Exception as e:   # noqa: BLE001 - recorded with its message and returned as the result
            self._event("create_failed", error=str(e), error_type=type(e).__name__)
            raise _Stop(f"create_failed: {e}") from e
        self._ep, self._n_eps = created["id"], self._n_eps + 1
        self._ctx |= {"endpoint_id": self._ep, "model_ref": created.get("model_ref")}
        self._event("endpoint_created")

    def _cleanup(self) -> None:
        ep, self._ep = self._ep, None
        if ep is None:   # a failed Recreate already deleted the old one
            return
        try:
            self.rp.set_workers_max(ep, 0)
        finally:
            self.rp.delete(ep)
            self._event("endpoint_deleted")

    def _step(self, ep: str, i: int, step: spec.Step, result: ScenarioResult) -> None:
        match step:
            case spec.Recreate():
                self._cleanup()
                self._create()
            case spec.Deploy():
                result.loads.append(self._deploy(ep, i, step, result.loads))
            case spec.Teardown():
                self.rp.set_workers_max(ep, 0)
                self._wait_no_live_pods(ep, "teardown")
            case spec.AwaitScaleDown():
                self._wait_no_live_pods(ep, "scale_down")
            case spec.Interrupt():
                self._interrupt(ep, i, step)
            case spec.Signal(tag=tag):
                self._signal(tag).set()
                self._event("signal", tag=tag)
            case spec.Await(tag=tag):
                self._await(tag)
            case spec.Idle(seconds=s):
                self.sleep(s)

    def _await(self, tag: str) -> None:
        ev, deadline = self._signal(tag), self.now() + spec.AWAIT_TIMEOUT_S
        while not ev.is_set():
            if self.now() > deadline:
                raise _Stop(f"await_timeout on {tag}")
            self.sleep(POLL_S)
        self._event("await_done", tag=tag)

    # ---------------------------------------------------------- interrupt
    def _interrupt(self, ep: str, i: int, step: spec.Interrupt) -> None:
        stale = set(self.rp.pods(ep))
        self.rp.set_workers_max(ep, 1)
        submitted = self.now()
        jid = self.rp.submit(ep, {"hold_s": spec.MEASURE_HOLD_S})
        pod = self._first_new_pod(ep, jid, submitted, stale)
        created = phases.epoch(pod.get("createdAt")) or self.now()
        cut_at = max(created, submitted) + step.after_s
        while self.now() < cut_at:
            self.sleep(min(INTERRUPT_POLL_S, cut_at - self.now()))
        self.rp.set_workers_max(ep, 0)
        self.rp.cancel(ep, jid)
        cut = self.now()
        started = phases.epoch((self.rp.pods(ep).get(pod["id"]) or pod).get("lastStartedAt"))
        self._emit({"kind": "interrupt", "step": i, "after_s": step.after_s, "job_id": jid,
                    "pod_id": pod["id"], "machine_id": pod.get("machineId"),
                    "submitted_utc": phases.iso(submitted), "pod_created_utc": phases.iso(created),
                    "teardown_utc": phases.iso(cut), "pod_last_started_utc": phases.iso(started),
                    "teardown_after_pod_s": round(cut - created, 1),
                    # True: the copy was already staged, so this round is an S2 round, not S3
                    "container_started_before_teardown": started is not None and started <= cut})
        self._wait_no_live_pods(ep, "teardown")

    def _first_new_pod(self, ep: str, jid: str, submitted: float, stale: set[str]) -> dict:
        while True:
            new = [p for k, p in self.rp.pods(ep).items() if k not in stale]
            if new:
                return min(new, key=lambda p: phases.epoch(p.get("createdAt")) or 0)
            if self.now() - submitted > spec.ABANDON_S:
                self.rp.cancel(ep, jid)
                self._event("interrupt_no_worker")
                raise _Stop("no_worker before interrupt")
            self.sleep(INTERRUPT_POLL_S)

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
        status, seen, failure = self._poll(ep, jid, submitted, stale, d.signal)
        wid = status.get("workerId") or ""
        # a worker stopped by idle timeout restarts under its old pod id, so it is in `stale`
        pod = seen.get(wid) or (self.rp.pods(ep).get(wid, {}) if wid in stale else {})
        rec = phases.load_record(submitted=submitted, status={"id": jid, **status}, pod=pod,
                                 done_seen=self.now(), no_worker=failure == "no_worker")
        last = next((r["machine_id"] for r in reversed(prev) if r.get("machine_id")), None)
        rec |= {"role": d.role, "step": i, "hold_s": d.hold_s, "failure": failure,
                # resumed = an earlier worker restarted (same pod id), not a new pod placed for this load
                "resumed": bool(rec["pod_reused"] or rec["pod_id"] in {r["pod_id"] for r in prev if r.get("pod_id")}),
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

    def _poll(self, ep: str, jid: str, submitted: float, stale: set[str],
              signal: str | None = None) -> tuple[dict, dict, str | None]:
        """(final status, this load's pods by id, failure or None). `signal` is set once the job runs."""
        seen: dict[str, dict] = {}
        clocks = {"live": None, "throttled": None}   # first-seen times; live None = nothing placed yet
        while True:
            self.sleep(POLL_S)
            seen |= {k: v for k, v in self.rp.pods(ep).items() if k not in stale}
            status = self.rp.status(ep, jid)
            if signal and status.get("status") in ("IN_PROGRESS", "COMPLETED") and not self._signal(signal).is_set():
                self._signal(signal).set()
                self._event("signal", tag=signal)
            if status.get("status") in DONE:
                return status, seen, None if status["status"] == "COMPLETED" else "job_" + status["status"].lower()
            t = self.now()
            self._tick(clocks, t, seen, self.rp.workers(ep))
            failure = self._failure(t, submitted, clocks)
            if failure:
                self.rp.cancel(ep, jid)
                return {**status, "status": "CANCELLED"}, seen, failure

    @staticmethod
    def _tick(clocks: dict, t: float, seen: dict, workers: dict) -> None:
        """A staging or GPU-waiting pod is listed EXITED; /health is the truth on whether it is alive."""
        live = sum(workers.get(k, 0) for k in LIVE_WORKERS) or any(p["desiredStatus"] not in GONE for p in seen.values())
        placed = bool(seen or live or workers.get("throttled"))
        if live or (placed and clocks["live"] is None):
            clocks["live"] = t   # stall clock starts when the worker first shows up
        if workers.get("throttled") and not live:
            clocks["throttled"] = clocks["throttled"] or t
        else:
            clocks["throttled"] = None

    @staticmethod
    def _failure(t: float, submitted: float, clocks: dict) -> str | None:
        if clocks["live"] is None and t - submitted > spec.ABANDON_S:
            return "no_worker"
        if clocks["throttled"] is not None and t - clocks["throttled"] > spec.THROTTLE_S:
            return "throttled"
        if clocks["live"] is not None and clocks["throttled"] is None and t - clocks["live"] > spec.STALL_S:
            return "stall"
        if t - submitted > spec.ROUND_TIMEOUT_S:
            return "timeout"
        return None
