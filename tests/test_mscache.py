"""MSCACHE harness: scenario specs, phase records, cost, and the driver against a fake Runpod."""
import json

import pytest

from mscache import cost, phases, spec
from mscache.driver import Driver, ScenarioResult

T0 = 1_700_000_000.0


# ------------------------------------------------------------------ spec
def test_s2_arm_is_seed_then_repeats_of_teardown_gap_redeploy():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=600, repeats=3)
    kinds = [type(s).__name__ for s in sc.steps]
    assert kinds == ["Deploy"] + ["Teardown", "Idle", "Deploy"] * 3
    deploys = [s for s in sc.steps if isinstance(s, spec.Deploy)]
    assert deploys[0].role == "seed" and all(d.role == "measure" for d in deploys[1:])
    assert all(d.hold_s == spec.READY_HOLD_S for d in deploys)   # copy fully on disk before teardown
    assert [s.seconds for s in sc.steps if isinstance(s, spec.Idle)] == [600] * 3
    assert sc.sid == "S2" and sc.arm == "gap600"


def test_s9_keeps_workers_max_and_waits_for_idle_scale_down():
    sc = spec.s9(dc="CA-MTL-1", gpu="a40", model="m/b", repeats=3)
    kinds = [type(s).__name__ for s in sc.steps]
    assert kinds == ["Deploy"] + ["AwaitScaleDown", "Idle", "Deploy"] * 3
    deploys = [s for s in sc.steps if isinstance(s, spec.Deploy)]
    assert deploys[0].scale_up and not any(d.scale_up for d in deploys[1:])


def test_scenario_rejects_unknown_gpu_and_bad_repo():
    with pytest.raises(ValueError, match="gpu"):
        spec.s9(dc="CA-MTL-1", gpu="tpu", model="m/b")
    with pytest.raises(ValueError, match="repo"):
        spec.s9(dc="CA-MTL-1", gpu="a40", model="not a repo")


def test_endpoint_name_is_prefixed_and_short():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60)
    name = sc.endpoint_name("r1121a")
    assert name.startswith(spec.NAME_PREFIX) and "camtl1" in name and len(name) <= 60


# ---------------------------------------------------------------- phases
def _pod(created=T0 + 1, started=T0 + 23, machine="7c66abc", pid="pod1"):
    return {"id": pid, "machineId": machine, "desiredStatus": "RUNNING",
            "createdAt": phases.iso(created), "lastStartedAt": phases.iso(started),
            "machine": {"gpuDisplayName": "A40", "location": "CA-MTL-1"}}


def _output(container_start=T0 + 25, handler_start=T0 + 45, now=T0 + 46):
    return {"now": now, "container_age_s": now - container_start, "handler_age_s": now - handler_start,
            "snapshot": "/runpod-volume/huggingface-cache/hub/models--m--a/snapshots/abc123",
            "n_files": 10, "staged_bytes": 27_000_000_000, "first_job_on_worker": True}


def test_load_record_has_joinable_utc_phases_and_offsets():
    rec = phases.load_record(submitted=T0, status={"status": "COMPLETED", "delayTime": 46000, "executionTime": 121000,
                                                   "workerId": "pod1", "output": _output()},
                             pod=_pod(), done_seen=T0 + 170)
    assert rec["pod_id"] == "pod1" and rec["machine_id"] == "7c66abc" and rec["revision"] == "abc123"
    assert rec["staged_bytes"] == 27_000_000_000
    assert rec["phases_utc"]["submit"].endswith("Z") and rec["phases_utc"]["container_start"].endswith("Z")
    assert rec["offsets_s"] == {"pod_created": 1.0, "pod_last_started": 23.0, "container_start": 25.0,
                                "handler_ready": 45.0, "job_start": 46.0, "job_done_seen": 170.0}
    assert rec["staging_s"] == 24.0          # pod created -> container start
    assert rec["outcome"] == "cached"        # client-side estimate; host-log join is final


@pytest.mark.parametrize("staging,want", [(4, "cached"), (58, "cached"), (139, "pull")])
def test_outcome_estimate_from_staging(staging, want):
    out = _output(container_start=T0 + 1 + staging, handler_start=T0 + 2 + staging, now=T0 + 3 + staging)
    rec = phases.load_record(submitted=T0, status={"status": "COMPLETED", "workerId": "pod1", "output": out},
                             pod=_pod(), done_seen=T0 + 200)
    assert rec["outcome"] == want


def test_no_worker_and_failed_outcomes():
    nw = phases.load_record(submitted=T0, status={"status": "CANCELLED"}, pod={}, done_seen=T0 + 300, no_worker=True)
    assert nw["outcome"] == "no_worker" and nw["pod_id"] is None
    failed = phases.load_record(submitted=T0, status={"status": "FAILED", "workerId": "pod1", "error": "boom"},
                                pod=_pod(), done_seen=T0 + 60)
    assert failed["outcome"] == "failed" and failed["error"] == "boom"


# ------------------------------------------------------------------ cost
def test_cost_counts_worker_seconds_per_deploy_at_pool_price():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=3)
    est = cost.estimate(sc, size_gb=30)
    per_s = spec.GPUS["a40"].usd_per_h / 3600
    assert est.loads == 4
    assert est.likely_usd == pytest.approx(per_s * est.likely_worker_s, rel=1e-6)
    assert est.worst_usd > est.likely_usd and est.wall_s > 3 * 60


# ---------------------------------------------------------------- driver
class FakeClock:
    def __init__(self):
        self.t = T0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeRunpod:
    """One endpoint. A submitted job gets a pod `place_after` s later and completes `run_s` after that."""

    def __init__(self, clock, place_after=5.0, run_s=60.0, machines=("7c66abc",), never_place=False):
        self.clock, self.place_after, self.run_s, self.never_place = clock, place_after, run_s, never_place
        self.machines = list(machines)
        self.extra_machine = None
        self.workers_max = 0
        self.jobs, self.pods_, self.deleted, self.created = {}, {}, [], []
        self.n = 0

    def create(self, name, scenario):
        self.created.append(name)
        return {"id": "ep1", "model_ref": f"huggingface.co/{scenario.model}@abc123"}

    def delete(self, ep):
        self.deleted.append(ep)

    def set_workers_max(self, ep, n):
        self.workers_max = n
        if n == 0:
            for p in self.pods_.values():
                p["desiredStatus"] = "EXITED"

    def submit(self, ep, payload):
        self.n += 1
        jid = f"j{self.n}"
        self.jobs[jid] = {"t": self.clock.now(), "payload": payload, "status": "IN_QUEUE"}
        return jid

    def cancel(self, ep, jid):
        self.jobs[jid]["status"] = "CANCELLED"

    def _advance(self):
        for jid, j in self.jobs.items():
            if j["status"] in ("COMPLETED", "CANCELLED") or self.never_place or self.workers_max == 0:
                continue
            t_place = j["t"] + self.place_after
            if self.clock.now() >= t_place and "pod" not in j:
                if self.reuse_pods and self.pods_:   # a stopped worker restarts under its old pod id
                    pid = next(iter(self.pods_))
                    self.pods_[pid] |= {"desiredStatus": "RUNNING", "lastStartedAt": phases.iso(t_place + 2)}
                    j["pod"] = pid
                    continue
                pid = f"pod{len(self.pods_) + 1}"
                machine = self.machines[min(len(self.pods_), len(self.machines) - 1)]
                self.pods_[pid] = _pod(created=t_place, started=t_place + 2, machine=machine, pid=pid)
                if self.exited_while_staging or self.throttled:
                    self.pods_[pid]["desiredStatus"] = "EXITED"
                j["pod"] = pid
                if self.extra_machine:   # e.g. 7vmo: an extra pod placed for the same job, killed later
                    self.pods_[f"extra{pid}"] = _pod(created=t_place, machine=self.extra_machine, pid=f"extra{pid}")
            if "pod" in j and self.clock.now() >= t_place + self.run_s and not self.throttled:
                j["status"] = "COMPLETED"
        for j in self.jobs.values():   # idle timeout removes the worker after its job
            done_t = j["t"] + self.place_after + self.run_s
            if j["status"] == "COMPLETED" and self.clock.now() >= done_t + spec.IDLE_TIMEOUT_S:
                self.pods_[j["pod"]]["desiredStatus"] = "EXITED"

    def status(self, ep, jid):
        self._advance()
        j = self.jobs[jid]
        st = {"id": jid, "status": j["status"]}
        if j["status"] == "COMPLETED":
            now = j["t"] + self.place_after + self.run_s
            st.update(workerId=j["pod"], delayTime=30000, executionTime=1000,
                      output=_output(container_start=j["t"] + self.place_after + 3,
                                     handler_start=j["t"] + self.place_after + 20, now=now))
        return st

    def pods(self, ep):
        self._advance()
        return {k: dict(v) for k, v in self.pods_.items() if v["desiredStatus"] != "GONE"}

    # live API (11-22 pilot): a pod staging a model or waiting for a GPU is listed desiredStatus EXITED;
    # only /health tells it is initializing or throttled
    exited_while_staging = False
    throttled = False
    reuse_pods = False

    def workers(self, ep):
        self._advance()
        busy = sum(1 for j in self.jobs.values() if "pod" in j and j["status"] not in ("COMPLETED", "CANCELLED"))
        if self.throttled:
            return {"initializing": 0, "running": 0, "idle": 0, "ready": 0, "throttled": busy}
        return {"initializing": busy, "running": 0, "idle": 0, "ready": 0, "throttled": 0}


def _run(sc, **fake_kw):
    clock = FakeClock()
    rp = FakeRunpod(clock, **fake_kw)
    lines = []
    result = Driver(rp, run_id="t1", sink=lambda rec: lines.append(json.loads(json.dumps(rec))),
                    now=clock.now, sleep=clock.sleep).run(sc)
    return rp, lines, result, clock


def test_driver_runs_s2_records_every_load_and_always_deletes():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=2)
    rp, lines, result, _ = _run(sc)
    loads = [r for r in lines if r["kind"] == "load"]
    assert [r["role"] for r in loads] == ["seed", "measure", "measure"]
    assert all(r["outcome"] == "cached" and r["endpoint_id"] == "ep1" for r in loads)
    assert all(r["scenario"] == "S2" and r["arm"] == "gap60" and r["dc"] == "CA-MTL-1" for r in loads)
    assert loads[1]["same_machine_as_prev"] is True
    assert loads[0]["model_ref"].endswith("@abc123")
    assert rp.deleted == ["ep1"] and isinstance(result, ScenarioResult) and result.ok


def test_driver_holds_job_for_ready_plus_two_minutes():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=1)
    rp, _, _, _ = _run(sc)
    assert all(j["payload"]["hold_s"] == spec.READY_HOLD_S for j in rp.jobs.values())


def test_idle_gap_is_measured_from_teardown_done():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=600, repeats=1)
    _, lines, _, _ = _run(sc)
    gone = next(r for r in lines if r["kind"] == "event" and r["event"] == "teardown_done")
    second = [r for r in lines if r["kind"] == "load"][1]
    assert second["submitted"] - gone["t"] >= 600


def test_no_worker_abandons_cancels_and_stops_scenario():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=2)
    rp, lines, result, clock = _run(sc, never_place=True)
    loads = [r for r in lines if r["kind"] == "load"]
    assert len(loads) == 1 and loads[0]["outcome"] == "no_worker"
    assert all(j["status"] == "CANCELLED" for j in rp.jobs.values())
    assert rp.deleted == ["ep1"] and not result.ok and "no_worker" in result.reason
    assert clock.now() - T0 < spec.ABANDON_S + 60


def test_driver_deletes_endpoint_even_when_a_step_raises():
    sc = spec.s9(dc="CA-MTL-1", gpu="a40", model="m/a", repeats=1)
    clock = FakeClock()
    rp = FakeRunpod(clock)

    def boom(ep, jid):
        raise RuntimeError("api down")
    rp.status = boom
    with pytest.raises(RuntimeError, match="api down"):
        Driver(rp, run_id="t1", sink=lambda r: None, now=clock.now, sleep=clock.sleep).run(sc)
    assert rp.deleted == ["ep1"]


def test_pod_only_pods_are_recorded_as_taint():
    sc = spec.s9(dc="CA-MTL-1", gpu="a40", model="m/a", repeats=1)
    clock = FakeClock()
    rp = FakeRunpod(clock)
    rp.extra_machine = "25e7zzz"
    lines = []
    Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    taint = [r for r in lines if r["kind"] == "pod_only"]
    assert taint and taint[0]["pod_id"].startswith("extra") and taint[0]["machine_id"] == "25e7zzz"


def test_long_staging_listed_exited_is_not_a_stall_while_health_says_initializing():
    sc = spec.s9(dc="CA-MTL-1", gpu="a40", model="m/a", repeats=1)
    clock = FakeClock()
    rp = FakeRunpod(clock, run_s=spec.STALL_S + 300)
    rp.exited_while_staging = True
    lines = []
    result = Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    loads = [r for r in lines if r["kind"] == "load"]
    assert loads[0]["outcome"] != "failed" and loads[0]["failure"] is None, loads[0]
    assert result.ok


def test_restarted_worker_with_reused_pod_id_is_joined_and_classified():
    sc = spec.s9(dc="CA-MTL-1", gpu="a40", model="m/a", repeats=1)
    clock = FakeClock()
    rp = FakeRunpod(clock)
    rp.reuse_pods = True
    lines = []
    Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    measure = [r for r in lines if r["kind"] == "load"][1]
    assert measure["machine_id"] == "7c66abc" and measure["pod_reused"] is True
    assert measure["outcome"] == "cached", measure   # staging anchored on submit, not the old createdAt


def test_throttled_worker_ends_load_as_throttled_after_limit():
    sc = spec.s9(dc="US-TX-3", gpu="l40s", model="m/a", repeats=1)
    clock = FakeClock()
    rp = FakeRunpod(clock)
    rp.throttled = True
    lines = []
    result = Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    load = next(r for r in lines if r["kind"] == "load")
    assert load["failure"] == "throttled" and not result.ok
    assert spec.THROTTLE_S <= clock.now() - T0 < spec.THROTTLE_S + 120
    assert rp.deleted == ["ep1"]


# ------------------------------------------------------------ plan/report
from mscache import plan, report  # noqa: E402


def test_pilot_uses_a_distinct_model_per_scenario_so_arms_cannot_taint_each_other():
    scs = plan.pilot("CA-MTL-1", "a40")
    assert [(s.sid, s.arm) for s in scs] == [("S2", "gap60"), ("S2", "gap600"), ("S9", "")]
    assert len({s.model for s in scs}) == len(scs)
    assert all(s.dc == "CA-MTL-1" and s.gpu == "a40" for s in scs)


def test_plan_total_states_likely_worst_and_wall():
    scs = plan.pilot("CA-MTL-1", "a40")
    tot = plan.total(scs)
    assert tot["likely_usd"] < tot["worst_usd"] and tot["loads"] == 12
    assert tot["wall_s"] == max(cost.estimate(s, size_gb=plan.SIZE_GB[s.model]).wall_s for s in scs)


def test_report_summarizes_measure_loads_median_and_worst():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=3)
    _, lines, _, _ = _run(sc)
    text = report.render(lines)
    assert "S2 gap60" in text and "measure loads: 3" in text
    assert "median" in text and "worst" in text and "ep1" in text
    assert "outcomes: cached=3" in text
