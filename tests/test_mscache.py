"""MSCACHE harness: scenario specs, phase records, cost, and the driver against a fake Runpod."""
import json
import threading

import pytest

from mscache import cost, phases, spec
from mscache.api import LiveClient
from mscache.driver import Driver, ScenarioResult

T0 = 1_700_000_000.0


# ------------------------------------------------------------------ spec
def test_s2_arm_is_seed_then_repeats_of_teardown_gap_redeploy():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=600, repeats=3)
    kinds = [type(s).__name__ for s in sc.steps]
    # a fresh endpoint per redeploy: ai-api resumes a stopped worker (same pod, same machine) before placing a new one
    assert kinds == ["Deploy"] + ["Teardown", "Recreate", "Idle", "Deploy"] * 3
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
    assert sc.endpoint_name("r1121a", 2).endswith("-2") and sc.endpoint_name("r1121a", 2) != name


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


def test_ai_api_floor_wait_is_flagged_so_cache_numbers_exclude_it():
    # aj82koxw08zcbp, 11-22: pod created at +0.8 s, container at +49.7 s, no transfer (local copy)
    rec = phases.load_record(submitted=T0, status={"status": "COMPLETED", "output": _output(
        container_start=T0 + 49.7, handler_start=T0 + 71, now=T0 + 72)}, pod=_pod(created=T0 + 0.8), done_seen=T0 + 200)
    assert rec["outcome"] == "cached" and rec["floor_wait"] is True
    fast = phases.load_record(submitted=T0, status={"status": "COMPLETED", "output": _output(
        container_start=T0 + 3.9, handler_start=T0 + 25, now=T0 + 26)}, pod=_pod(created=T0 + 1), done_seen=T0 + 150)
    assert fast["floor_wait"] is False
    # kzlvbq4ht3vm26, TX-3 11-22: pod placed at +238 s, staged 23.6 s after the floor had ended
    late = phases.load_record(submitted=T0, status={"status": "COMPLETED", "output": _output(
        container_start=T0 + 261.7, handler_start=T0 + 280, now=T0 + 281)}, pod=_pod(created=T0 + 238.1),
        done_seen=T0 + 400)
    assert late["outcome"] == "cached" and late["floor_wait"] is False


def test_staging_is_labeled_by_anchor_and_submit_to_container_is_recorded():
    # 64q5on3hbd3zjf, MO-1 11-22: new pod at +24.9 s, container at +48.3 s
    rec = phases.load_record(submitted=T0, status={"status": "COMPLETED", "output": _output(
        container_start=T0 + 48.3, handler_start=T0 + 60, now=T0 + 61)}, pod=_pod(created=T0 + 24.9),
        done_seen=T0 + 200)
    assert rec["staging_s"] == 23.4 and rec["staging_from"] == "pod_created"
    assert rec["submit_to_container_s"] == 48.3 and rec["floor_wait"] is True
    restarted = phases.load_record(submitted=T0, status={"status": "COMPLETED", "output": _output(
        container_start=T0 + 2.0, handler_start=T0 + 10, now=T0 + 11)}, pod=_pod(created=T0 - 600),
        done_seen=T0 + 100)
    assert restarted["staging_from"] == "submit" and restarted["staging_s"] == 2.0
    text = report.render([rec | {"role": "measure", "scenario": "S2", "arm": "gap60", "endpoint_id": "e",
                                 "model": "m/a", "same_machine_as_prev": None, "resumed": False}])
    assert "23.4 (from pod_created) | 48.3 |" in text and "submit->container start: median 48s" in text


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

    def __init__(self, clock, place_after=5.0, run_s=60.0, machines=("7c66abc",), never_place=False,
                 start_after=2.0):
        self.clock, self.place_after, self.run_s, self.never_place = clock, place_after, run_s, never_place
        self.start_after = start_after   # pod placed -> container start (lastStartedAt)
        self.machines = list(machines)
        self.extra_machine = None
        self.workers_max = 0
        self.jobs, self.pods_, self.deleted, self.created = {}, {}, [], []
        self.n = 0

    def create(self, name, scenario):
        self.created.append(name)
        self.ep = f"ep{len(self.created)}"
        return {"id": self.ep, "model_ref": f"huggingface.co/{scenario.model}@abc123"}

    def delete(self, ep):
        self.deleted.append(ep)
        for p in self.pods_.values():
            if p["ep"] == ep:
                p["desiredStatus"] = "GONE"

    def set_workers_max(self, ep, n):
        self.workers_max = n
        if n == 0:
            for p in self.pods_.values():
                p["desiredStatus"] = "EXITED"

    def submit(self, ep, payload):
        self.n += 1
        jid = f"j{self.n}"
        self.jobs[jid] = {"t": self.clock.now(), "payload": payload, "status": "IN_QUEUE", "ep": ep}
        return jid

    def cancel(self, ep, jid):
        self.jobs[jid]["status"] = "CANCELLED"

    def _advance(self):
        for jid, j in self.jobs.items():
            if j["status"] in ("COMPLETED", "CANCELLED") or self.never_place or self.workers_max == 0:
                continue
            t_place = j["t"] + self.place_after
            if self.clock.now() >= t_place and "pod" not in j:
                mine = [k for k, p in self.pods_.items() if p["ep"] == j["ep"]]
                if self.reuse_pods and mine:   # ai-api resumes a stopped worker of THIS endpoint, same pod id
                    pid = mine[0]
                    self.pods_[pid] |= {"desiredStatus": "RUNNING", "lastStartedAt": phases.iso(t_place + 2)}
                    j["pod"] = pid
                    continue
                pid = f"pod{len(self.pods_) + 1}"
                machine = self.machines[min(len(self.pods_), len(self.machines) - 1)]
                self.pods_[pid] = _pod(created=t_place, started=t_place + self.start_after, machine=machine,
                                       pid=pid) | {"ep": j["ep"], "start_t": t_place + self.start_after}
                if self.exited_while_staging or self.throttled:
                    self.pods_[pid]["desiredStatus"] = "EXITED"
                j["pod"] = pid
                if self.extra_machine:   # e.g. 7vmo: an extra pod placed for the same job, killed later
                    self.pods_[f"extra{pid}"] = _pod(created=t_place, machine=self.extra_machine,
                                                     pid=f"extra{pid}") | {"ep": j["ep"]}
            if "pod" in j and self.clock.now() >= t_place + self.run_s and not self.throttled:
                j["status"] = "COMPLETED"
        for j in self.jobs.values():   # idle timeout removes the worker after its job
            done_t = j["t"] + self.place_after + self.run_s
            if j["status"] == "COMPLETED" and self.clock.now() >= done_t + spec.IDLE_TIMEOUT_S:
                self.pods_[j["pod"]]["desiredStatus"] = "EXITED"

    def status(self, ep, jid):
        self._advance()
        j = self.jobs[jid]
        running = j["status"] == "IN_QUEUE" and "pod" in j
        st = {"id": jid, "status": "IN_PROGRESS" if running else j["status"]}
        if j["status"] == "COMPLETED":
            now = j["t"] + self.place_after + self.run_s
            st.update(workerId=j["pod"], delayTime=30000, executionTime=1000,
                      output=_output(container_start=j["t"] + self.place_after + 3,
                                     handler_start=j["t"] + self.place_after + 20, now=now))
        return st

    def pods(self, ep):
        self._advance()
        out = {k: dict(v) for k, v in self.pods_.items() if v["desiredStatus"] != "GONE" and v["ep"] == ep}
        for p in out.values():   # lastStartedAt appears when the container starts
            if self.clock.now() < p.get("start_t", 0):
                p["lastStartedAt"] = None
        return out

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
    assert all(r["outcome"] == "cached" for r in loads)
    assert [r["endpoint_id"] for r in loads] == ["ep1", "ep2", "ep3"]   # fresh endpoint per redeploy
    assert [r["resumed"] for r in loads] == [False, False, False]
    assert all(r["scenario"] == "S2" and r["arm"] == "gap60" and r["dc"] == "CA-MTL-1" for r in loads)
    assert loads[1]["same_machine_as_prev"] is True
    assert loads[0]["model_ref"].endswith("@abc123")
    assert rp.deleted == ["ep1", "ep2", "ep3"] and isinstance(result, ScenarioResult) and result.ok
    assert len(set(rp.created)) == 3


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


def test_late_placement_like_tx3_470s_is_not_abandoned():
    sc = spec.s9(dc="US-TX-3", gpu="l40s", model="m/a", repeats=1)
    _, lines, result, _ = _run(sc, place_after=470.0)
    load = next(r for r in lines if r["kind"] == "load")
    assert load["failure"] is None and result.ok, load


def test_s2_redeploy_cannot_resume_the_old_worker_even_when_the_platform_would():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=2)
    clock = FakeClock()
    rp = FakeRunpod(clock)
    rp.reuse_pods = True
    lines = []
    Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    loads = [r for r in lines if r["kind"] == "load"]
    assert len({r["pod_id"] for r in loads}) == 3 and not any(r["resumed"] for r in loads)
    events = [r["event"] for r in lines if r["kind"] == "event"]
    assert events.count("endpoint_created") == 3 and events.count("endpoint_deleted") == 3


def test_s9_resume_is_flagged_on_the_load():
    sc = spec.s9(dc="CA-MTL-1", gpu="a40", model="m/a", repeats=1)
    clock = FakeClock()
    rp = FakeRunpod(clock)
    rp.reuse_pods = True
    lines = []
    Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    seed, measure = [r for r in lines if r["kind"] == "load"]
    assert seed["resumed"] is False and measure["resumed"] is True


def test_recreate_failure_still_leaves_no_endpoint_behind():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=1)
    clock = FakeClock()
    rp = FakeRunpod(clock)
    real = rp.create

    def create(name, scenario):
        if rp.created:
            raise RuntimeError("create failed")
        return real(name, scenario)
    rp.create = create
    lines = []
    result = Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    assert not result.ok and "create failed" in result.reason
    assert rp.deleted == ["ep1"]
    assert any(r.get("event") == "create_failed" and "create failed" in r["error"] for r in lines)


# ------------------------------------------------------- full-matrix specs
def _kinds(sc):
    return [type(s).__name__ for s in sc.steps]


def test_s1_is_one_cold_load_on_a_model_new_to_the_dc():
    sc = spec.s1(dc="CA-MTL-1", gpu="a40", model="m/a")
    assert _kinds(sc) == ["Deploy"] and sc.deploys[0].role == "cold" and sc.sid == "S1"


def test_s3_tears_down_after_pod_created_then_redeploys_on_a_fresh_endpoint():
    sc = spec.s3(dc="CA-MTL-1", gpu="a40", model="m/big", after_s=30)
    assert _kinds(sc) == ["Interrupt", "Recreate", "Deploy"] and sc.arm == "t30"
    assert sc.steps[0].after_s == 30 and sc.deploys[0].role == "measure"


def test_s7_is_a_cold_load_then_a_fresh_redeploy_per_size_class():
    sc = spec.s7(dc="CA-MTL-1", gpu="a40", model="m/a", size="large")
    assert _kinds(sc) == ["Deploy", "Teardown", "Recreate", "Idle", "Deploy"] and sc.arm == "large"
    assert [d.role for d in sc.deploys] == ["cold", "measure"]
    with pytest.raises(ValueError, match="size"):
        spec.s7(dc="CA-MTL-1", gpu="a40", model="m/a", size="huge")


def test_s10_redeploys_immediately_on_a_fresh_endpoint():
    sc = spec.s10(dc="CA-MTL-1", gpu="a40", model="m/a")
    assert _kinds(sc) == ["Deploy"] + ["Teardown", "Recreate", "Idle", "Deploy"] * 3
    assert {s.seconds for s in sc.steps if isinstance(s, spec.Idle)} == {0}


def test_s8_takes_a_bad_revision_or_unreadable_repo():
    bad_rev = spec.s8(dc="CA-MTL-1", gpu="a40", model="m/a:0000000000000000000000000000000000000000", arm="badrev")
    gated = spec.s8(dc="CA-MTL-1", gpu="a40", model="meta-llama/Llama-3.1-8B", arm="gated")
    assert _kinds(bad_rev) == ["Deploy"] and bad_rev.deploys[0].role == "failure" and gated.arm == "gated"
    with pytest.raises(ValueError, match="repo"):
        spec.s8(dc="CA-MTL-1", gpu="a40", model="m/a:", arm="badrev")


def test_s4_holder_signals_ready_and_peers_wait_for_the_copy_to_settle():
    holder, *peers = spec.s4(dc="US-TX-3", gpu="l40s", model="m/a", peers=3)
    assert holder.arm == "holder" and [p.arm for p in peers] == ["peer1", "peer2", "peer3"]
    assert holder.deploys[0].signal == "S4:m/a" and holder.deploys[0].hold_s >= spec.S4_HOLD_S
    for p in peers:
        assert _kinds(p) == ["Await", "Idle", "Deploy"] and p.steps[0].tag == "S4:m/a"
        assert p.steps[1].seconds == spec.READY_HOLD_S and p.model == "m/a"


def test_s5_arms_and_models():
    scs = spec.s5(dc="US-TX-3", gpu="l40s", cold=["m/c1", "m/c2"], warm="m/w")
    warm, *cold = scs
    assert warm.arm == "warm" and [c.arm for c in cold] == ["cold1", "cold2"]
    assert all(c.deploys[0].role == "cold" for c in cold)
    assert len({s.model for s in scs}) == 3
    assert [c.arm for c in spec.s5(dc="US-TX-3", gpu="l40s", cold=["m/c1", "m/c2", "m/c3"])] == \
        ["cold1", "cold2", "cold3"]


# ------------------------------------------------------ full-matrix driver
def test_interrupt_records_whether_the_container_started_before_teardown():
    sc = spec.s3(dc="CA-MTL-1", gpu="a40", model="m/big", after_s=10)
    rp, lines, result, _ = _run(sc, start_after=200.0)
    cut = next(r for r in lines if r["kind"] == "interrupt")
    assert cut["container_started_before_teardown"] is False and cut["after_s"] == 10
    assert 10 <= cut["teardown_after_pod_s"] < 10 + 3
    assert cut["pod_id"] and cut["machine_id"] and cut["teardown_utc"].endswith("Z")
    load = next(r for r in lines if r["kind"] == "load")
    assert load["endpoint_id"] == "ep2" and result.ok and rp.deleted == ["ep1", "ep2"]


def test_interrupt_after_container_start_is_flagged_so_the_round_counts_as_s2():
    sc = spec.s3(dc="CA-MTL-1", gpu="a40", model="m/big", after_s=60)
    _, lines, _, _ = _run(sc, start_after=2.0)
    cut = next(r for r in lines if r["kind"] == "interrupt")
    assert cut["container_started_before_teardown"] is True


def test_s3_measure_records_whether_it_landed_on_the_interrupted_pods_machine():
    # placement ignores the cache (DD 11-25 join), so every S3 redeploy must say where it landed
    sc = spec.s3(dc="CA-MTL-1", gpu="a40", model="m/big", after_s=10)
    _, lines, _, _ = _run(sc, start_after=200.0, machines=("7c66abc", "9zzz111"))
    load = next(r for r in lines if r["kind"] == "load")
    assert load["machine_id"] == "9zzz111" and load["same_machine_as_prev"] is False
    _, lines, _, _ = _run(sc, start_after=200.0)
    load = next(r for r in lines if r["kind"] == "load")
    assert load["same_machine_as_prev"] is True


def test_interrupt_with_no_worker_stops_and_cleans_up():
    sc = spec.s3(dc="US-TX-3", gpu="l40s", model="m/big", after_s=10)
    rp, lines, result, _ = _run(sc, never_place=True)
    assert not result.ok and "no_worker" in result.reason and rp.deleted == ["ep1"]


def test_create_rejected_is_a_result_not_a_crash():
    sc = spec.s8(dc="CA-MTL-1", gpu="a40", model="m/a:0000000000000000000000000000000000000000", arm="badrev")
    clock = FakeClock()
    rp = FakeRunpod(clock)

    def create(name, scenario):
        raise RuntimeError("modelReferences not saved")
    rp.create = create
    lines = []
    result = Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep).run(sc)
    assert not result.ok and "modelReferences not saved" in result.reason and rp.deleted == []
    ev = next(r for r in lines if r.get("event") == "create_failed")
    assert ev["error"] == "modelReferences not saved"


def test_await_times_out_cleanly_when_the_signal_never_comes():
    _, peer, *_ = spec.s4(dc="US-TX-3", gpu="l40s", model="m/a", peers=1)
    rp, lines, result, clock = _run(peer)
    assert not result.ok and "await_timeout" in result.reason and rp.deleted == ["ep1"]
    assert clock.now() - T0 >= spec.AWAIT_TIMEOUT_S


def test_deploy_signal_fires_once_the_job_is_running_and_releases_the_waiter():
    holder, peer = spec.s4(dc="US-TX-3", gpu="l40s", model="m/a", peers=1)
    clock = FakeClock()
    signals = {}
    rp = FakeRunpod(clock, run_s=spec.S4_HOLD_S)
    Driver(rp, run_id="t1", sink=lambda r: None, now=clock.now, sleep=clock.sleep, signals=signals).run(holder)
    assert signals["S4:m/a"].is_set()
    rp2 = FakeRunpod(clock)
    lines = []
    result = Driver(rp2, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep,
                    signals=signals).run(peer)
    assert result.ok and [r["role"] for r in lines if r["kind"] == "load"] == ["measure"]


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
    assert "outcomes: cached (est.)=3" in text


def test_report_keeps_ai_api_floor_loads_out_of_staging_and_counts_resumes():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=3)
    _, lines, _, _ = _run(sc)
    loads = [r for r in lines if r["kind"] == "load"]
    measure = [r for r in loads if r["role"] == "measure"]
    measure[0] |= {"floor_wait": True, "staging_s": 49.0}
    for r in measure[1:]:
        r["staging_s"] = 4.0
    measure[2]["resumed"] = True
    text = report.render(lines)
    assert "staging_s (excl. ai-api floor): median 4s, worst 4s" in text
    assert "cached behind ai-api floor: 1" in text
    assert "new pod=2 resumed (same pod id)=1" in text
    assert ", ".join(dict.fromkeys(r["endpoint_id"] for r in loads)) in text
    assert len({r["endpoint_id"] for r in loads}) == 4
    assert "cached (est.) (ai-api floor)" in text


# ------------------------------------------------------------- matrix plan
def _scs(lanes):
    return [s for lane in lanes for batch in lane for s in batch]


@pytest.mark.parametrize("dc", plan.LRU_DCS)
def test_matrix_covers_the_required_scenarios_per_lru_dc(dc):
    w1, w2 = _scs(plan.matrix(dc, "a40", window=1)), _scs(plan.matrix(dc, "a40", window=2))
    keys = {f"{s.sid}{s.arm}" for s in w1}
    assert {"S1", "S2gap60", "S2gap600", "S2gap3600", "S9", "S10", "S7small", "S7medium", "S7large",
            "S3t10", "S3t30", "S3t60"} <= keys
    assert any(s.sid == "S4" for s in w2) == (dc in plan.PEER_LIVE)
    assert any(s.sid == "S8" for s in w2) == (dc == "CA-MTL-1")


@pytest.mark.parametrize("window", [1, 2])
def test_matrix_models_are_distinct_within_a_window_except_s4_arms(window):
    scs = [s for s in _scs(plan.matrix("CA-MTL-1", "a40", window=window)) if s.sid != "S4"]
    models = [s.model.split(":")[0] for s in scs if s.sid != "S8"]
    assert len(models) == len(set(models))
    pilot_models = {s.model for s in plan.pilot("CA-MTL-1", "a40")}
    assert not pilot_models & set(models)          # pilot DCs already hold those copies


def test_matrix_core_lane_is_serial_and_every_model_has_a_size():
    lanes = plan.matrix("US-MO-1", "a100", window=1)
    assert all(len(batch) == 1 for batch in lanes[1])
    for s in _scs(lanes) + _scs(plan.matrix("US-MO-1", "a100", window=2)) + _scs(plan.control("EUR-IS-2", "4090")):
        assert plan.size_gb(s) > 0


def test_s3_models_are_100_gb_plus_and_s7_sizes_match_their_class():
    assert all(plan.SIZE_GB[m] >= 100 for m in plan.M["S3"].values())
    s7 = plan.M["S7"]
    assert plan.SIZE_GB[s7["small"]] < 10 and 20 <= plan.SIZE_GB[s7["medium"]] <= 40
    assert plan.SIZE_GB[s7["large"]] >= 100


def test_lanes_total_wall_is_longest_lane_of_serial_batches():
    lanes = plan.matrix("CA-MTL-1", "a40", window=1)
    tot = plan.lanes_total(lanes)
    core = sum(plan.total(b)["wall_s"] for b in lanes[1])
    assert tot["wall_s"] == max(core, plan.total(lanes[0][0])["wall_s"])
    assert tot["likely_usd"] < tot["worst_usd"]
    assert plan.lanes_total([])["loads"] == 0


def test_cli_only_filters_the_matrix_and_drops_empty_batches():
    import argparse
    from mscache.__main__ import _lanes, _scenarios
    args = argparse.Namespace(dc="CA-MTL-1", gpu="a40", window=1, control=False, only="S3t10,S2gap3600")
    lanes = _lanes(args)
    assert [[[s.sid + s.arm for s in b] for b in lane] for lane in lanes] == [[["S2gap3600"]], [["S3t10"]]]
    pilot = _lanes(argparse.Namespace(dc="CA-MTL-1", gpu="a40", window=None, control=False, only=None, serial=True))
    assert len(pilot) == 1 and len(_scenarios(pilot)) == 3


def test_cli_execute_runs_lanes_in_parallel_and_batches_in_order():
    from mscache.__main__ import _execute
    order = []

    class Rec:
        def run(self, s):
            order.append(s.sid + s.arm)
            return ScenarioResult(ok=True)
    lanes = plan.matrix("US-TX-3", "l40s", window=2)
    out = _execute(lanes, Rec)
    assert len(out) == len(_scs(lanes)) and all(r.ok for _, r in out)
    s4 = [k for k in order if k.startswith("S4")]
    assert order.index(s4[-1]) < order.index("S5warm")   # S5 batch starts after the S4 batch is done


# ------------------------------------------- 11-24 fixes before the v4.9.0 run
@pytest.mark.parametrize("staging,want", [(4, "cached"), (139, "pull")])
def test_cutoff_outcome_is_marked_as_an_estimate_for_the_host_join(staging, want):
    # 73y4 16:54 (11-22): a 24 s origin pull read as cached by the cutoff; only the host join can tell
    out = _output(container_start=T0 + 1 + staging, handler_start=T0 + 2 + staging, now=T0 + 3 + staging)
    rec = phases.load_record(submitted=T0, status={"status": "COMPLETED", "workerId": "pod1", "output": out},
                             pod=_pod(), done_seen=T0 + 200)
    assert rec["outcome"] == want and rec["outcome_estimated"] is True
    assert rec["outcome_rule"] == f"staging_s <= {phases.CACHED_STAGING_MAX_S} s"


def test_failure_outcomes_are_observed_not_estimated():
    nw = phases.load_record(submitted=T0, status={"status": "CANCELLED"}, pod={}, done_seen=T0 + 300, no_worker=True)
    assert nw["outcome_estimated"] is False and nw["outcome_rule"] is None


def test_report_labels_cutoff_outcomes_as_estimates():
    sc = spec.s2(dc="CA-MTL-1", gpu="a40", model="m/a", gap_s=60, repeats=1)
    _, lines, _, _ = _run(sc)
    text = report.render(lines)
    assert "| cached (est.) |" in text and "outcomes: cached (est.)=1" in text


def test_leftovers_are_scoped_to_the_runs_dc():
    class Rp:
        def list_endpoints(self):
            return [{"id": "a", "name": "msc-camtl1-s2gap600-a40-11241200"},
                    {"id": "b", "name": "msc-mo1-s4holder-l40s-11241200"},
                    {"id": "c", "name": "other-endpoint"}]
    live = LiveClient(Rp())
    assert [e["id"] for e in live.leftovers("CA-MTL-1")] == ["a"]
    assert [e["id"] for e in live.leftovers("US-MO-1")] == ["b"]
    assert [e["id"] for e in live.leftovers()] == ["a", "b"]


def test_dc_prefix_is_the_start_of_every_endpoint_name_in_that_dc():
    sc = spec.s1(dc="US-MO-1", gpu="l40s", model="m/a")
    assert spec.dc_prefix("US-MO-1") == "msc-mo1-"
    assert sc.endpoint_name("r1").startswith(spec.dc_prefix("US-MO-1"))


HOME, BUSY = "S5home:m/w", "S5:m/w"


def test_s5_seeds_warm_first_then_cold_pulls_and_warm_measure_co_land_on_its_machine():
    warm, *cold = spec.s5(dc="CA-MTL-1", gpu="a40", cold=["m/c1", "m/c2"], warm="m/w")
    assert _kinds(warm) == ["Deploy", "Teardown", "Recreate", "Await", "Idle", "Deploy"]
    seed, measure = warm.deploys
    assert (seed.role, seed.pin, seed.co_land, seed.tries) == ("seed", HOME, None, 1)
    assert warm.steps[3].tag == BUSY
    assert (measure.role, measure.co_land, measure.tries) == ("measure", BUSY, spec.S5_CO_LAND_TRIES)
    for c in cold:
        assert _kinds(c) == ["Await", "Deploy"] and c.steps[0].tag == HOME
        d = c.deploys[0]
        assert (d.role, d.co_land, d.busy, d.tries) == ("cold", HOME, BUSY, spec.S5_CO_LAND_TRIES)
    # 11-25 MO-1: a cold 72B pull and the queued warm load both hit the 30 min round timeout
    assert all(d.timeout_s == spec.S5_ROUND_TIMEOUT_S > spec.ROUND_TIMEOUT_S
               for sc in (warm, *cold) for d in sc.deploys)


def test_cost_counts_the_worst_case_aborted_co_land_tries():
    warm, *_ = spec.s5(dc="CA-MTL-1", gpu="a40", cold=["m/c1"], warm="m/w")
    one_try = spec.Scenario(warm.sid, warm.arm, warm.dc, warm.gpu, warm.model,
                            tuple(spec.Deploy(s.role, s.hold_s, co_land=s.co_land) if isinstance(s, spec.Deploy)
                                  else s for s in warm.steps))
    three, one = cost.estimate(warm, size_gb=64), cost.estimate(one_try, size_gb=64)
    extra = spec.S5_CO_LAND_TRIES - 1
    assert three.likely_worker_s - one.likely_worker_s == pytest.approx(extra * cost.ABORTED_TRY_S)
    assert three.worst_worker_s - one.worst_worker_s == pytest.approx(extra * cost.ABORTED_TRY_S)
    assert three.wall_s - one.wall_s == pytest.approx(extra * (cost.ABORTED_TRY_S + cost.TEARDOWN_WALL_S))
    assert three.loads == one.loads + extra


def _driver(clock, rp, lines, signals, busy):
    return Driver(rp, run_id="t1", sink=lines.append, now=clock.now, sleep=clock.sleep, signals=signals,
                  busy=busy)


def _set(tag):
    ev = threading.Event()
    ev.set()
    return {tag: ev}


def test_warm_seed_pins_its_machine_and_signals_once_its_copy_is_staged():
    seed = spec.Scenario("S5", "warm", "CA-MTL-1", "a40", "m/w", (spec.Deploy("seed", 60, pin=HOME),))
    clock, signals, busy, lines = FakeClock(), {}, {}, []
    result = _driver(clock, FakeRunpod(clock, machines=("mH",)), lines, signals, busy).run(seed)
    assert result.ok and signals[HOME].is_set()
    # the pin outlives the seed load: cold pulls target this machine after the seed is torn down
    assert set(busy[HOME].values()) == {"mH"}


def _cold_run(machines):
    _, *cold = spec.s5(dc="CA-MTL-1", gpu="a40", cold=["m/c1"], warm="m/w")
    clock, signals, busy, lines = FakeClock(), _set(HOME), {HOME: {"seedpod": "mH"}}, []
    rp = FakeRunpod(clock, machines=machines)
    result = _driver(clock, rp, lines, signals, busy).run(cold[0])
    loads = [r for r in lines if r["kind"] == "load"]
    marks = [(r["event"], r["machine_id"]) for r in lines if r.get("event", "").startswith("busy")]
    return rp, loads, marks, result, signals, busy


def test_cold_pull_redeploys_until_it_lands_on_the_seed_machine_and_only_then_marks_it_busy():
    rp, loads, marks, result, signals, busy = _cold_run(("mA", "mH"))
    assert result.ok
    assert [(r["machine_id"], r["co_landed"], r["failure"]) for r in loads] == [
        ("mA", False, "off_target"), ("mH", True, None)]
    assert marks == [("busy_on", "mH"), ("busy_off", "mH")]
    assert signals[BUSY].is_set() and busy[BUSY] == {}
    assert len(rp.created) == 2 and sorted(rp.deleted) == ["ep1", "ep2"]


def test_cold_pull_off_the_seed_machine_on_its_last_try_runs_but_never_marks_busy():
    rp, loads, marks, result, signals, _ = _cold_run(("mA",))
    assert result.ok
    assert [(r["co_landed"], r["failure"]) for r in loads] == [
        (False, "off_target"), (False, "off_target"), (False, None)]
    assert marks == [] and BUSY not in signals


def _warm_run(machines, cold_busy):
    warm, *_ = spec.s5(dc="CA-MTL-1", gpu="a40", cold=["m/c1"], warm="m/w")
    clock, lines = FakeClock(), []
    rp = FakeRunpod(clock, machines=machines)
    result = _driver(clock, rp, lines, _set(BUSY), {BUSY: dict(cold_busy)}).run(warm)
    return rp, [r for r in lines if r["kind"] == "load"], result


def test_warm_measure_redeploys_until_it_lands_where_a_cold_pull_is_staging():
    rp, loads, result = _warm_run(("mH", "mA", "mH"), {"coldpod": "mH"})
    assert result.ok
    assert [(r["role"], r["machine_id"], r["co_landed"], r["attempt"]) for r in loads] == [
        ("seed", "mH", None, 1), ("measure", "mA", False, 1), ("measure", "mH", True, 2)]
    assert [r["failure"] for r in loads] == [None, "off_target", None]
    assert len(rp.created) == 3 and sorted(rp.deleted) == ["ep1", "ep2", "ep3"]


def test_warm_measure_gives_up_after_three_tries_and_records_not_co_landed():
    rp, loads, result = _warm_run(("mH", "mA"), {"coldpod": "mH2"})
    assert result.ok
    assert [(r["role"], r["co_landed"], r["failure"]) for r in loads] == [
        ("seed", None, None), ("measure", False, "off_target"), ("measure", False, "off_target"),
        ("measure", False, None)]
    assert len(rp.created) == len(rp.deleted) == 4


def test_deploy_timeout_override_lets_a_long_queued_load_finish():
    def sc(**kw):
        return spec.Scenario("S5", "cold1", "CA-MTL-1", "a40", "m/c1", (spec.Deploy("cold", 60, **kw),))
    run_s = spec.ROUND_TIMEOUT_S + 300
    _, _, short, _ = _run(sc(), run_s=run_s)
    _, _, long_, _ = _run(sc(timeout_s=spec.S5_ROUND_TIMEOUT_S), run_s=run_s)
    assert not short.ok and "timeout" in short.reason
    assert long_.ok
