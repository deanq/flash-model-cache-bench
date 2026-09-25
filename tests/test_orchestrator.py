import json

import pytest

from bench import config, deploy, driver, report
from bench.runpod_api import RunpodError, model_reference


# ---------------------------------------------------------------- config
def test_config_defaults_and_empty_values():
    cfg = config.load({"MODEL": "", "COLD_TRIALS": ""}, run_id="r1")
    assert cfg.model == "Qwen/Qwen2.5-3B-Instruct" and cfg.cold_trials == 3
    assert cfg.strategies == config.STRATEGIES and cfg.volume_name == "hfbench-r1"
    assert cfg.deploy_env()["BENCH_VOLUME_NAME"] == "hfbench-r1"


@pytest.mark.parametrize("env,match", [
    ({"MODEL": "not a repo"}, "repo id"),
    ({"DATACENTER": "MARS-1"}, "DATACENTER"),
    ({"STRATEGIES": "nocache,tape"}, "unknown"),
    ({"COLD_TRIALS": "0"}, "out of range"),
    ({"PARALLEL": "maybe"}, "boolean"),
])
def test_config_rejects_bad_values(env, match):
    with pytest.raises(config.ConfigError, match=match):
        config.load(env, run_id="r")


def test_model_reference_matches_console_format():
    assert model_reference("Qwen/Qwen2.5-3B-Instruct") == "qwen/qwen2.5-3b-instruct"
    assert model_reference("A/B", "abc") == "a/b:abc"


# ---------------------------------------------------------------- deploy
def test_endpoints_from_manifest_keys_by_strategy():
    manifest = {"resources": {"hfbench-volume": {"endpoint_id": "e1"}, "hfbench-nocache": {"endpoint_id": "e2"},
                              "other": {"endpoint_id": "x"}, "hfbench-modelstore": {}}}
    assert deploy.endpoints_from_manifest(manifest) == {"volume": "e1", "nocache": "e2"}


class FakeRp:
    def __init__(self, endpoints=(), volumes=()):
        self.endpoints, self.volumes = list(endpoints), list(volumes)
        self.workers_max: dict[str, int] = {}
        self.health_seq: dict[str, list[dict]] = {}
        self.jobs: dict[str, list[dict]] = {}

    def list_endpoints(self):
        return self.endpoints

    def list_network_volumes(self):
        return self.volumes

    def set_workers_max(self, eid, n):
        self.workers_max[eid] = n

    def health(self, eid):
        seq = self.health_seq.get(eid) or [{}]
        return {"workers": seq.pop(0) if len(seq) > 1 else seq[0]}

    def run(self, eid, payload):
        return f"job-{eid}-{len(self.jobs.setdefault(eid, []))}"

    def status(self, eid, job):
        return self.jobs[eid].pop(0)

    def cancel(self, eid, job):
        pass


def test_verify_clean_reports_leftover_endpoints_and_volume():
    state = deploy.State("r", "app", "env", "hfbench-r", endpoints={"volume": "e1"}, volume_id="v1")
    rp = FakeRp(endpoints=[{"id": "e1", "name": "x"}, {"id": "z", "name": "hfbench-nocache"}, {"id": "o", "name": "mine"}],
                volumes=[{"id": "v1", "name": "hfbench-r"}, {"id": "v2", "name": "keep-me"}])
    left = deploy.verify_clean(state, rp)
    assert len(left) == 3 and not any("keep-me" in x or "mine" in x for x in left)


# ---------------------------------------------------------------- driver
def completed(output: dict, delay=1000, exec_=2000):
    return {"status": "COMPLETED", "delayTime": delay, "executionTime": exec_, "output": {"ok": True, **output}}


def test_run_job_marks_handler_error_invalid(monkeypatch):
    monkeypatch.setattr(driver, "POLL_S", 0)
    rp = FakeRp()
    rp.jobs["e"] = [{"status": "IN_PROGRESS"}, {"status": "COMPLETED", "output": {"ok": False, "error": "boom"}}]
    t = driver.run_job(rp, "e", {}, 10, driver.Trial("volume", "cold", 1))
    assert not t.valid and t.invalid_reason == "boom"


def test_run_job_client_error_is_recorded_not_raised(monkeypatch):
    rp = FakeRp()
    monkeypatch.setattr(rp, "run", lambda *a: (_ for _ in ()).throw(RunpodError("HTTP 500")))
    t = driver.run_job(rp, "e", {}, 10, driver.Trial("volume", "cold", 1))
    assert t.status == "CLIENT_ERROR" and not t.valid


def test_force_cold_pauses_waits_for_drain_then_restores(monkeypatch):
    monkeypatch.setattr(driver.time, "sleep", lambda s: None)
    rp = FakeRp()
    rp.health_seq["e"] = [{"idle": 1}, {"idle": 0}]
    driver.force_cold(rp, ["e"])
    assert rp.workers_max == {"e": 1}


def test_cold_trial_on_warm_process_is_invalid_and_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "POLL_S", 0)
    monkeypatch.setattr(driver.time, "sleep", lambda s: None)
    cfg = config.load({"WARM_PER_COLD": "1", "STRATEGIES": "volume"}, run_id="r")
    rp = FakeRp()
    rp.jobs["e"] = [completed({"cold_process": False}), completed({"cold_process": True}),
                    completed({"cold_process": False})]
    rec = driver.Recorder(str(tmp_path / "r.jsonl"))
    driver.Driver(cfg, rp, {"volume": "e"}, rec)._cold_then_warm("volume", 1)
    phases = [(t.phase, t.valid) for t in rec.trials]
    assert phases == [("cold", False), ("cold", True), ("warm", True)]
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 3


# ---------------------------------------------------------------- report
def trial(strategy, phase, e2e, *, acquire=None, load=1.0, hydrate=None, valid=True, worker="w", rnd=1):
    cold = None
    if phase in ("cold", "prime"):
        acq = {"hydrate_s": hydrate} if hydrate is not None else {}
        cold = {"import_s": 5.0, "acquire_s": acquire, "acquire": acq, "tokenizer_load_s": 0.0,
                "model_load_s": load, "bytes": 2 ** 30, "hub_online": False}
    return {"strategy": strategy, "phase": phase, "round": rnd, "e2e_s": e2e, "delay_ms": 1000, "exec_ms": 2000,
            "valid": valid, "error": None if valid else "x",
            "output": {"cold": cold, "worker_id": worker, "generate": {"generate_s": 1.0, "tokens_per_s": 50.0},
                       "gpu": "RTX", "versions": {"torch": "2"}}}


def test_summary_and_verdict_compare_storage_time():
    trials = [trial("volume", "cold", 20, acquire=0.1, load=6.0), trial("volume", "cold", 22, acquire=0.1, load=6.0),
              trial("volumecache", "cold", 15, acquire=2.0, load=1.0, hydrate=1.9),
              trial("volumecache", "cold", 99, valid=False), trial("volume", "warm", 1.5)]
    s = report.summarize(trials, ["volume", "volumecache"])
    assert s["volume"]["cold_n"] == 2 and s["volumecache"]["cold_invalid"] == 1
    assert s["volume"]["storage"].median == pytest.approx(6.1)
    verdict = report.in_place_vs_cached(s, "volume", "volumecache", "network volume")
    assert verdict.startswith("**network volume: volumecache**") and "2.03×" in verdict
    assert report.verdicts(s) == [verdict]
    assert report.ranking(s) == ["volumecache", "volume"]


def test_stat_is_median_first_and_robust_to_outliers():
    x = report.stat([10, 11, 118])
    assert x.median == 11 and x.fmt() == "11.0s [10.0–118.0]"
    assert report.stat([]).fmt() == "—" and report.stat([2]).fmt() == "2.0s"


def test_platform_delay_outlier_is_flagged():
    trials = [trial(s, "cold", 20) for s in ("nocache", "volume")]
    slow = trial("modelstore", "cold", 80)
    slow["delay_ms"] = 70000
    s = report.summarize(trials + [slow], ["nocache", "volume", "modelstore"])
    obs = report._delay_outliers(s, ["nocache", "volume", "modelstore"])
    assert len(obs) == 1 and "modelstore" in obs[0]


def test_render_includes_all_sections(tmp_path):
    meta = {"run_id": "r", "model": "A/B", "gpu": "ADA_24", "datacenter": "EU-RO-1", "cold_trials": 1,
            "warm_per_cold": 1, "parallel": True, "evict_page_cache": True, "strategies": ["nocache", "modelstore"]}
    trials = [trial("nocache", "prime", 30, acquire=10), trial("nocache", "cold", 30, acquire=10),
              trial("modelstore", "prime", 0, valid=False)]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(t) for t in trials))
    md = open(report.write(str(tmp_path))).read()
    for heading in ("## Answer", "## Cold start", "## Warm requests", "## First-ever request", "## Caveats"):
        assert heading in md
    assert "| modelstore | failed |" in md


def test_waves_keep_shared_storage_strategies_apart():
    assert driver.waves(["nocache", "volume", "volumecache", "modelstore"]) == [
        ["nocache", "volume", "modelstore"], ["volumecache"]]
    assert driver.waves(list(config.STRATEGIES)) == [
        ["nocache", "volume", "modelstore", "globalstore"], ["volumecache", "gsvolumecache"]]
    assert driver.waves(["volumecache", "modelstore"]) == [["volumecache", "modelstore"]]
    assert driver.waves(["gsvolumecache"]) == [["gsvolumecache"]]


def test_hub_column_is_empty_when_hub_never_used():
    s = report.summarize([trial("volume", "cold", 20, acquire=0.1)], ["volume"])
    assert s["volume"]["hub"].n == 0


def test_update_endpoint_carries_global_volume_and_model_refs():
    from bench.runpod_api import Runpod, OBJECT_STORE_VOLUME
    rp = Runpod.__new__(Runpod)
    sent = {}
    ep = {"id": "e", "name": "n", "workersMax": 1, "modelReferences": ["m"],
          "volumes": [{"volumeId": "g", "volumeType": OBJECT_STORE_VOLUME, "mountPath": "/runpod-volume"},
                      {"volumeId": "nv", "volumeType": "NETWORK_VOLUME", "mountPath": "/runpod-volume"}]}
    rp.endpoint = lambda eid: ep
    rp.graphql = lambda q, v: sent.update(v["input"]) or {"saveEndpoint": {"id": "e"}}
    rp.update_endpoint("e", workersMax=0)
    assert sent["workersMax"] == 0 and sent["modelReferences"] == ["m"]
    assert sent["volumes"] == [{"volumeId": "g", "volumeType": OBJECT_STORE_VOLUME}]


def test_verify_clean_reports_leftover_global_store():
    state = deploy.State("r", "app", "env", "hfbench-r", global_store_name="hfbench-r", global_store_id="g1")
    rp = FakeRp()
    rp.list_global_stores = lambda: [{"id": "g1", "name": "hfbench-r"}, {"id": "g2", "name": "someone-else"}]
    assert deploy.verify_clean(state, rp) == ["global store hfbench-r (g1)"]


def test_old_state_files_without_global_store_fields_still_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "old").mkdir(parents=True)
    (tmp_path / "runs" / "old" / "state.json").write_text(json.dumps(
        {"run_id": "old", "app": "a", "env": "e", "volume_name": "v", "endpoints": {}, "volume_id": None,
         "torn_down": True}))
    assert deploy.State.load("old").global_store_id is None


def test_config_knows_global_store_strategies():
    assert config.load({"STRATEGIES": "nocache"}, run_id="r").uses_global_store is False
    cfg = config.load({"STRATEGIES": "globalstore"}, run_id="r")
    assert cfg.uses_global_store and cfg.global_store_name == "hfbench-r"
