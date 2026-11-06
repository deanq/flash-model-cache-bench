import json

import pytest

from bench import config, deploy, driver, report
from bench.runpod_api import RunpodError


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


def test_each_run_is_its_own_app_and_deploy_env_carries_selection():
    cfg = config.load({"STRATEGIES": "nocache,volume"}, run_id="r1")
    assert cfg.app_name == "hfbench-r1" and cfg.uses_network_volume and not cfg.uses_global_store
    env = cfg.deploy_env()
    assert env["BENCH_APP"] == "hfbench-r1" and env["BENCH_STRATEGIES"] == "nocache,volume"
    assert env["BENCH_GLOBAL_STORE_NAME"] == "hfbench-r1"


@pytest.mark.parametrize("run_id", ["a b", "x/y", "-lead", "z" * 41])
def test_config_rejects_run_ids_unsafe_as_resource_names(run_id):
    with pytest.raises(config.ConfigError, match="RUN_ID"):
        config.load({}, run_id=run_id)


# ---------------------------------------------------------------- deploy
def test_endpoints_from_environment_keys_by_strategy():
    environment = {"endpoints": [{"id": "e1", "name": "hfbench-volume"}, {"id": "e2", "name": "hfbench-nocache"},
                                 {"id": "x", "name": "other"}]}
    assert deploy.endpoints_from_environment(environment) == {"volume": "e1", "nocache": "e2"}
    assert deploy.endpoints_from_environment(None) == {}


class FakeRp:
    def __init__(self, endpoints=(), volumes=(), apps=()):
        self.endpoints, self.volumes, self.apps = list(endpoints), list(volumes), set(apps)
        self.workers_max: dict[str, int] = {}
        self.health_seq: dict[str, list[dict]] = {}
        self.jobs: dict[str, list[dict]] = {}

    def list_endpoints(self):
        return self.endpoints

    def list_network_volumes(self):
        return self.volumes

    def list_global_stores(self):
        return []

    def flash_app_exists(self, app):
        return app in self.apps

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


def test_verify_clean_reports_only_this_runs_leftovers():
    """The account is shared: same-named endpoints of other apps/runs are not ours."""
    state = deploy.State("r", "hfbench-r", "env", "hfbench-r", endpoints={"volume": "e1"}, volume_id="v1")
    rp = FakeRp(endpoints=[{"id": "e1", "name": "hfbench-volume"}, {"id": "z", "name": "hfbench-nocache"},
                           {"id": "o", "name": "mine"}],
                volumes=[{"id": "v1", "name": "hfbench-r"}, {"id": "v2", "name": "keep-me"}],
                apps={"hfbench-r", "hfbench-other-run"})
    assert deploy.verify_clean(state, rp) == [
        "flash app hfbench-r", "endpoint hfbench-volume (e1)", "network volume hfbench-r (v1)"]


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


def test_prime_never_runs_strategies_sharing_storage_at_once(tmp_path, monkeypatch):
    """globalstore and gshydrate populate the same Global Store folder; the store has
    no locking (last write wins), so their first requests must not overlap."""
    cfg = config.load({"STRATEGIES": "nocache,globalstore,gshydrate", "PARALLEL": "1"}, run_id="r")
    rec = driver.Recorder(str(tmp_path / "r.jsonl"))
    d = driver.Driver(cfg, FakeRp(), {"nocache": "a", "globalstore": "b", "gshydrate": "c"}, rec)
    batches = []
    monkeypatch.setattr(d, "_each", lambda strategies, fn: batches.append(list(strategies)))
    d.prime()
    assert batches == [["nocache", "globalstore"], ["gshydrate"]]


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
        ["nocache", "volume", "modelstore", "globalstore"], ["volumecache", "gsvolumecache"], ["gshydrate"]]
    assert driver.waves(["volumecache", "modelstore"]) == [["volumecache", "modelstore"]]
    assert driver.waves(["gsvolumecache"]) == [["gsvolumecache"]]
    assert driver.waves(["nocache", "globalstore", "gshydrate"]) == [["nocache", "globalstore"], ["gshydrate"]]


def test_hub_column_is_empty_when_hub_never_used():
    s = report.summarize([trial("volume", "cold", 20, acquire=0.1)], ["volume"])
    assert s["volume"]["hub"].n == 0


def test_set_workers_max_patches_only_the_workers_block():
    """A partial REST update cannot drop modelReferences, volumes or the Flash
    environment binding, which a GraphQL saveEndpoint upsert would have to resend."""
    import httpx
    from bench.runpod_api import Runpod

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.method == "GET":
            return httpx.Response(200, json={"id": "e", "workers": {"min": 0, "max": 1, "idleTimeout": 5}})
        return httpx.Response(200, json={"id": "e"})

    rp = Runpod(key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    rp.set_workers_max("e", 0)
    assert [(m, p) for m, p, _ in seen] == [("GET", "/v2/serverless/e"), ("PATCH", "/v2/serverless/e")]
    assert json.loads(seen[1][2]) == {"workers": {"min": 0, "max": 0, "idleTimeout": 5}}


def test_set_workers_max_raises_on_api_error():
    import httpx
    from bench.runpod_api import Runpod

    rp = Runpod(key="k", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(404, text="not found"))))
    with pytest.raises(RunpodError, match="read endpoint e: HTTP 404"):
        rp.set_workers_max("e", 0)


def test_verify_clean_reports_leftover_global_store():
    state = deploy.State("r", "app", "env", "hfbench-r", global_store_name="hfbench-r", global_store_id="g1")
    rp = FakeRp()
    rp.list_global_stores = lambda: [{"id": "g1", "name": "hfbench-r"}, {"id": "g2", "name": "someone-else"}]
    assert deploy.verify_clean(state, rp) == ["global volume hfbench-r (g1)"]


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


def test_verify_attachments_fails_when_deploy_did_not_attach_storage():
    cfg = config.load({"STRATEGIES": "modelstore,globalstore"}, run_id="r")
    state = deploy.State("r", "hfbench-r", "env", "hfbench-r", endpoints={"modelstore": "m", "globalstore": "g"},
                         global_store_name="hfbench-r", global_store_id="gv")
    rp = FakeRp()
    eps = {"m": {"modelReferences": ["https://huggingface.co/Qwen/X:abc"]},
           "g": {"volumes": [{"volumeId": "gv", "volumeType": "OBJECT_STORE_VOLUME"}]}}
    rp.endpoint = lambda eid: eps[eid]
    assert deploy.verify_attachments(cfg, state, rp)["modelstore"]["modelReferences"]
    eps["g"] = {"volumes": []}
    with pytest.raises(RunpodError, match="globalstore .*not attached"):
        deploy.verify_attachments(cfg, state, rp)


# ---------------------------------------------------------------- worker app
def _load_bench_app(monkeypatch, **env):
    import importlib
    import sys

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("bench_app", None)
    return importlib.import_module("bench_app")


def test_bench_app_registers_only_selected_strategies_with_their_storage(monkeypatch):
    app_mod = _load_bench_app(monkeypatch, BENCH_RUN_ID="t1", BENCH_APP="hfbench-t1",
                              BENCH_STRATEGIES="nocache,volumecache,modelstore,gshydrate")
    specs = {name: h.spec for name, h in app_mod.app.resources.items()}
    assert app_mod.app.name == "hfbench-t1"
    assert sorted(specs) == ["hfbench-gshydrate", "hfbench-modelstore", "hfbench-nocache", "hfbench-volumecache"]
    manifest = {n: s.to_manifest() for n, s in specs.items()}
    assert manifest["hfbench-volumecache"]["mounts"] == [
        {"kind": "network", "reference": "hfbench-t1", "path": "/runpod-volume"}]
    assert manifest["hfbench-gshydrate"]["mounts"] == [
        {"kind": "global", "reference": "hfbench-t1", "path": "/runpod-volume"}]
    assert "mounts" not in manifest["hfbench-nocache"]
    assert manifest["hfbench-modelstore"]["model"] == "Qwen/Qwen2.5-3B-Instruct"
    nocache = manifest["hfbench-nocache"]
    assert (nocache["workersMin"], nocache["workersMax"], nocache["idleTimeout"]) == (0, 1, 5)
    assert nocache["executionTimeoutMs"] == 30 * 60 * 1000 and nocache["locations"] == "EU-RO-1"
    assert nocache["dependencies"] == app_mod.DEPENDENCIES
    # the worker imports the module too: it must see the same app and selection
    env = nocache["env"]
    assert env["BENCH_APP"] == "hfbench-t1" and env["BENCH_STRATEGIES"] == "nocache,volumecache,modelstore,gshydrate"
    # unselected strategies stay plain functions (not deployed); selected ones keep their names
    assert not hasattr(app_mod.hfbench_volume, "spec") and app_mod.hfbench_nocache.spec.name == "hfbench-nocache"
