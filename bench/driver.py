"""Trial scheduling: prime, then N rounds of (force cold -> 1 cold + W warm) per strategy.

Cold is *forced*, not waited for: idle timeout proved unreliable (workers stayed
up minutes past a 5 s idleTimeout), so each round pauses every endpoint
(workersMax=0), waits until /health reports no workers, then restores
workersMax=1. The handler's own `cold_process` flag is the ground truth -- a
cold trial served by an already-loaded process is recorded as invalid and
retried once.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from bench.config import Config
from bench.deploy import log
from bench.runpod_api import Runpod, RunpodError

POLL_S = 0.5
DRAIN_TIMEOUT_S = 180
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")


def waves(strategies: list[str]) -> list[list[str]]:
    """Group strategies for concurrent execution so no two volume-backed ones overlap."""
    first = [s for s in strategies if s != "volumecache"]
    rest = [s for s in strategies if s == "volumecache"]
    if "volume" not in first:
        return [strategies]
    return [w for w in (first, rest) if w]


@dataclass
class Trial:
    strategy: str
    phase: str                      # prime | cold | warm
    round: int
    seq: int = 0                    # warm request index within a round
    job_id: str | None = None
    status: str | None = None
    e2e_s: float | None = None      # job accepted -> terminal status (client-observed)
    submit_wait_s: float | None = None  # time the endpoint answered 409 paused after un-pausing
    delay_ms: int | None = None     # platform: queue + worker start
    exec_ms: int | None = None      # platform: handler execution
    valid: bool = True
    invalid_reason: str | None = None
    output: dict = field(default_factory=dict)
    error: str | None = None
    started_at: float = 0.0


class Recorder:
    """Append-only JSONL, one line per finished trial, safe across threads."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self.trials: list[Trial] = []

    def add(self, t: Trial) -> None:
        with self._lock:
            self.trials.append(t)
            with open(self._path, "a") as fh:
                fh.write(json.dumps(asdict(t)) + "\n")
        took = f"{t.e2e_s:.1f}s" if t.e2e_s is not None else "-"
        flag = "" if t.valid else f"  INVALID: {t.invalid_reason}"
        log(f"{t.strategy:<11} {t.phase:<5} r{t.round}.{t.seq} {t.status} e2e={took} "
            f"delay={t.delay_ms}ms exec={t.exec_ms}ms{flag}")


def run_job(rp: Runpod, endpoint_id: str, payload: dict, timeout_s: float, trial: Trial) -> Trial:
    trial.started_at = time.time()
    t0 = time.monotonic()
    try:
        trial.job_id = rp.run(endpoint_id, payload)
        # Restart the clock once the job is accepted: right after force_cold the
        # endpoint may reject submissions (409 paused) for a while, and that is
        # an artifact of the benchmark, not part of the cold start.
        trial.submit_wait_s = time.monotonic() - t0
        t0 = time.monotonic()
        while True:
            st = rp.status(endpoint_id, trial.job_id)
            if st.get("status") in TERMINAL:
                break
            if time.monotonic() - t0 > timeout_s:
                rp.cancel(endpoint_id, trial.job_id)
                st = {"status": "CLIENT_TIMEOUT", "error": f"no result after {timeout_s:.0f}s"}
                break
            time.sleep(POLL_S)
    except RunpodError as e:
        st = {"status": "CLIENT_ERROR", "error": str(e)}
    trial.e2e_s = time.monotonic() - t0
    trial.status = st.get("status")
    trial.delay_ms, trial.exec_ms = st.get("delayTime"), st.get("executionTime")
    trial.output = st.get("output") if isinstance(st.get("output"), dict) else {}
    trial.error = st.get("error") or trial.output.get("error")
    if trial.status != "COMPLETED" or not trial.output.get("ok"):
        trial.valid, trial.invalid_reason = False, trial.error or f"status {trial.status}"
    return trial


def force_cold(rp: Runpod, endpoint_ids: list[str]) -> None:
    for eid in endpoint_ids:
        rp.set_workers_max(eid, 0)
    deadline = time.monotonic() + DRAIN_TIMEOUT_S
    while True:
        busy = {eid: w for eid in endpoint_ids if sum((w := rp.health(eid).get("workers", {})).values())}
        if not busy:
            break
        if time.monotonic() > deadline:
            raise RunpodError(f"workers did not drain within {DRAIN_TIMEOUT_S}s: {busy}")
        time.sleep(2)
    for eid in endpoint_ids:
        rp.set_workers_max(eid, 1)


class Driver:
    def __init__(self, cfg: Config, rp: Runpod, endpoints: dict[str, str], recorder: Recorder):
        self.cfg, self.rp, self.endpoints, self.rec = cfg, rp, endpoints, recorder
        self.payload = {"max_new_tokens": cfg.max_new_tokens}

    def _each(self, strategies: list[str], fn) -> None:
        if self.cfg.parallel and len(strategies) > 1:
            with ThreadPoolExecutor(max_workers=len(strategies)) as pool:
                list(pool.map(fn, strategies))
        else:
            for s in strategies:
                fn(s)

    def prime(self) -> list[str]:
        """First request per strategy: populates caches (and Model Store staging).
        Returns the strategies that succeeded; the rest are reported and skipped."""
        log("prime: first request per strategy (downloads / populates caches)")
        ok: list[str] = []

        def one(s: str) -> None:
            t = run_job(self.rp, self.endpoints[s], self.payload, self.cfg.prime_timeout_s, Trial(s, "prime", 0))
            self.rec.add(t)
            if t.valid:
                ok.append(s)
            else:
                log(f"prime failed for {s}; excluding it from cold rounds: {t.error}")

        self._each(list(self.cfg.strategies), one)
        return [s for s in self.cfg.strategies if s in ok]

    def _cold_then_warm(self, s: str, rnd: int) -> None:
        eid = self.endpoints[s]
        for attempt in (1, 2):
            t = run_job(self.rp, eid, self.payload, self.cfg.trial_timeout_s, Trial(s, "cold", rnd))
            if t.valid and not t.output.get("cold_process"):
                t.valid, t.invalid_reason = False, "served by an already-loaded worker (not cold)"
            self.rec.add(t)
            if t.valid or t.output.get("cold_process") is not False or attempt == 2:
                break
            force_cold(self.rp, [eid])
        if not t.valid:
            return
        for i in range(self.cfg.warm_per_cold):
            w = run_job(self.rp, eid, self.payload, self.cfg.trial_timeout_s, Trial(s, "warm", rnd, seq=i + 1))
            if w.valid and w.output.get("cold_process"):
                w.valid, w.invalid_reason = False, "landed on a fresh worker (not warm)"
            self.rec.add(w)

    def rounds(self, strategies: list[str]) -> None:
        for rnd in range(1, self.cfg.cold_trials + 1):
            log(f"round {rnd}/{self.cfg.cold_trials}: forcing scale-to-zero on {strategies}")
            force_cold(self.rp, [self.endpoints[s] for s in strategies])
            for wave in (waves(strategies) if self.cfg.parallel else [strategies]):
                self._each(wave, lambda s: self._cold_then_warm(s, rnd))
