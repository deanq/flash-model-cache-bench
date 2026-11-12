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
def trial(strategy, phase, e2e, *, acquire=None, load=1.0, hydrate=None, prefetch=None, valid=True, worker="w",
          rnd=1):
    cold = None
    if phase in ("cold", "prime"):
        acq = {k: v for k, v in (("hydrate_s", hydrate), ("prefetch_s", prefetch)) if v is not None}
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


def test_shared_storage_read_rate_uses_the_step_that_reads_the_storage():
    """MiB/s off shared storage: the model load for in-place strategies, the hydrate or
    prefetch for strategies that read it up front; nothing for a Hub download."""
    trials = [trial("volume", "cold", 20, acquire=0.1, load=4.0),
              trial("volumecache", "cold", 15, acquire=2.0, load=1.0, hydrate=1.0),
              trial("volumeprefetch", "cold", 15, acquire=2.1, load=1.5, prefetch=0.5),
              trial("nocache", "cold", 30, acquire=10, load=1.0)]
    s = report.summarize(trials, ["volume", "volumecache", "volumeprefetch", "nocache"])
    assert s["volume"]["read_mibps"].median == pytest.approx(256)
    assert s["volumecache"]["read_mibps"].median == pytest.approx(1024)
    assert s["volumeprefetch"]["read_mibps"].median == pytest.approx(2048)
    assert s["volumeprefetch"]["hydrate"].median == pytest.approx(0.5)
    assert s["nocache"]["read_mibps"].n == 0


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
        ["nocache", "volume", "modelstore", "globalstore"], ["volumecache", "gsvolumecache"],
        ["gshydrate", "volumeprefetch"]]
    assert driver.waves(["volume", "volumeprefetch"]) == [["volume"], ["volumeprefetch"]]
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
                              BENCH_STRATEGIES="nocache,volumecache,modelstore,gshydrate,volumeprefetch")
    specs = {name: h.spec for name, h in app_mod.app.resources.items()}
    assert app_mod.app.name == "hfbench-t1"
    assert sorted(specs) == ["hfbench-gshydrate", "hfbench-modelstore", "hfbench-nocache", "hfbench-volumecache",
                             "hfbench-volumeprefetch"]
    manifest = {n: s.to_manifest() for n, s in specs.items()}
    assert manifest["hfbench-volumecache"]["mounts"] == [
        {"kind": "network", "reference": "hfbench-t1", "path": "/runpod-volume"}]
    assert manifest["hfbench-volumeprefetch"]["mounts"] == manifest["hfbench-volumecache"]["mounts"]
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
    assert env["BENCH_APP"] == "hfbench-t1" and env["BENCH_STRATEGIES"] == "nocache,volumecache,modelstore,gshydrate,volumeprefetch"
    # unselected strategies stay plain functions (not deployed); selected ones keep their names
    assert not hasattr(app_mod.hfbench_volume, "spec") and app_mod.hfbench_nocache.spec.name == "hfbench-nocache"


def test_usage_reports_wall_clock_handler_and_billed_gpu_minutes_for_this_run_only():
    meta = {"started_at": 1000.0, "finished_at": 1000.0 + 90 * 60, "endpoints": {"volume": "e1", "nocache": "e2"}}
    trials = [trial("volume", "cold", 20), trial("nocache", "warm", 5)]       # exec_ms 2000 each
    billing = [{"endpointId": "e1", "timeBilledMs": 120_000, "amount": 0.5},
               {"endpointId": "e2", "timeBilledMs": 60_000, "amount": 0.25},
               {"endpointId": "e2", "timeBilledMs": 60_000, "amount": 0.25},   # second day bucket
               {"endpointId": "other-run", "timeBilledMs": 10 ** 9, "amount": 99.0}]
    u = report.usage(meta, trials, billing)
    assert u["wall_clock_min"] == pytest.approx(90)
    assert u["handler_gpu_min"] == pytest.approx(4000 / 60000)
    assert u["billed_gpu_min"] == pytest.approx(4.0) and u["billed_usd"] == pytest.approx(1.0)
    assert u["billed_gpu_min_by_strategy"] == {"volume": pytest.approx(2.0), "nocache": pytest.approx(2.0)}


def test_usage_without_billing_leaves_billed_fields_empty():
    u = report.usage({"started_at": 0.0, "endpoints": {}}, [], None)
    assert u["wall_clock_min"] is None and u["billed_gpu_min"] is None and u["handler_gpu_min"] == 0


def test_summary_json_carries_run_usage_next_to_strategies(tmp_path):
    meta = {"run_id": "r", "model": "A/B", "gpu": "AMPERE_80", "datacenter": "EU-RO-1", "cold_trials": 1,
            "warm_per_cold": 0, "parallel": True, "evict_page_cache": True, "strategies": ["nocache"],
            "started_at": 0.0, "finished_at": 600.0, "endpoints": {"nocache": "e2"}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "results.jsonl").write_text(json.dumps(trial("nocache", "cold", 30, acquire=10)))
    (tmp_path / "billing.json").write_text(json.dumps([{"endpointId": "e2", "timeBilledMs": 60_000, "amount": 0.1}]))
    md = open(report.write(str(tmp_path))).read()
    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["usage"]["billed_gpu_min"] == pytest.approx(1.0) and data["strategies"]["nocache"]["cold_n"] == 1
    assert "## Time and cost" in md and "billed GPU time: 1.0 GPU-min" in md


def test_billing_rows_cover_the_run_window_in_daily_buckets():
    import httpx
    from bench.runpod_api import Runpod

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"endpointId": "e1", "timeBilledMs": 1, "amount": 0.0}])

    rp = Runpod(key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    rows = rp.endpoint_billing(start=1793772060.0, end=1793772967.0)   # 2026-11-04 UTC
    assert rows == [{"endpointId": "e1", "timeBilledMs": 1, "amount": 0.0}]
    q = seen[0].url.params
    assert seen[0].url.path == "/v1/billing/endpoints" and q["bucketSize"] == "day"
    assert (q["startTime"], q["endTime"]) == ("2026-11-04T00:00:00Z", "2026-11-05T00:00:00Z")


def test_cmd_billing_writes_billing_json_and_rerenders(tmp_path, monkeypatch):
    from bench import __main__ as cli

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    meta = {"run_id": "r1", "model": "A/B", "gpu": "AMPERE_80", "datacenter": "EU-RO-1", "cold_trials": 1,
            "warm_per_cold": 0, "parallel": True, "evict_page_cache": True, "strategies": ["nocache"],
            "started_at": 10.0, "finished_at": 70.0, "endpoints": {"nocache": "e2"}}
    (run_dir / "meta.json").write_text(json.dumps(meta))
    (run_dir / "results.jsonl").write_text(json.dumps(trial("nocache", "cold", 30, acquire=10)))

    class FakeBilling:
        def endpoint_billing(self, start, end):
            assert (start, end) == (10.0, 70.0)
            return [{"endpointId": "e2", "timeBilledMs": 120_000, "amount": 0.2}]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Runpod", FakeBilling)
    assert cli.main(["billing", "r1"]) == 0
    assert json.loads((run_dir / "billing.json").read_text())[0]["endpointId"] == "e2"
    assert json.loads((run_dir / "summary.json").read_text())["usage"]["billed_gpu_min"] == pytest.approx(2.0)


def test_status_polls_survive_transient_connection_resets(monkeypatch):
    """A 40-min run died on one `Connection reset by peer` while polling. Reads are
    idempotent, so retry them; a persistent outage still surfaces as RunpodError."""
    import httpx
    from bench import runpod_api
    from bench.runpod_api import Runpod

    monkeypatch.setattr(runpod_api, "READ_RETRY_BASE_S", 0)
    calls = []

    def flaky(request):
        calls.append(request.url.path)
        if len(calls) <= 2:
            raise httpx.ReadError("[Errno 54] Connection reset by peer")
        return httpx.Response(200, json={"status": "COMPLETED"})

    rp = Runpod(key="k", client=httpx.Client(transport=httpx.MockTransport(flaky)))
    assert rp.status("e", "j")["status"] == "COMPLETED" and len(calls) == 3

    down = Runpod(key="k", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("unreachable")))))
    with pytest.raises(RunpodError, match="health of e: ConnectError"):
        down.health("e")


def _makeup_driver(tmp_path, monkeypatch, jobs, makeup="2"):
    monkeypatch.setattr(driver, "POLL_S", 0)
    monkeypatch.setattr(driver.time, "sleep", lambda s: None)
    cfg = config.load({"COLD_TRIALS": "2", "WARM_PER_COLD": "0", "STRATEGIES": "nocache,volume",
                       "PARALLEL": "0", "MAKEUP_ROUNDS": makeup}, run_id="r")
    rp = FakeRp()
    rp.jobs.update(jobs)
    rec = driver.Recorder(str(tmp_path / "r.jsonl"))
    driver.Driver(cfg, rp, {"nocache": "a", "volume": "b"}, rec).rounds(["nocache", "volume"])
    return rec


def test_makeup_rounds_top_up_only_strategies_short_of_valid_cold_trials(tmp_path, monkeypatch):
    """A capacity timeout costs a trial the strategy didn't lose; re-run only the short ones."""
    timeout = {"status": "FAILED", "error": "no worker"}
    rec = _makeup_driver(tmp_path, monkeypatch, {
        "a": [completed({"cold_process": True})] * 2,
        "b": [timeout, completed({"cold_process": True}), completed({"cold_process": True})]})
    cold = [(t.strategy, t.round, t.valid) for t in rec.trials if t.phase == "cold"]
    assert cold == [("nocache", 1, True), ("volume", 1, False), ("nocache", 2, True), ("volume", 2, True),
                    ("volume", 3, True)]


def test_makeup_rounds_are_capped(tmp_path, monkeypatch):
    timeout = {"status": "FAILED", "error": "no worker"}
    rec = _makeup_driver(tmp_path, monkeypatch, {"a": [completed({"cold_process": True})] * 2,
                                                 "b": [timeout] * 10}, makeup="1")
    assert [t.round for t in rec.trials if t.strategy == "volume"] == [1, 2, 3]


def test_usage_reports_not_yet_billed_instead_of_zero_when_no_rows_match():
    """Billing lags hours; a run absent from the rows is unbilled-so-far, not free."""
    meta = {"started_at": 0.0, "finished_at": 60.0, "endpoints": {"volume": "e1"}}
    u = report.usage(meta, [], [{"endpointId": "someone-else", "timeBilledMs": 5, "amount": 1.0}])
    assert u["billed_gpu_min"] is None and u["billed_usd"] is None
    assert "not fetched yet" in "\n".join(report._usage_lines(u)) or "not in the billing" in "\n".join(report._usage_lines(u))


def test_render_has_overall_cold_request_table_with_best_case():
    """Model Store stages before the container starts: its cost is in delayTime, not
    storage time, so the report must compare whole cold requests too."""
    meta = {"run_id": "r", "model": "A/B", "gpu": "H100", "datacenter": "US-CA-2", "cold_trials": 2,
            "warm_per_cold": 0, "parallel": True, "evict_page_cache": True, "strategies": ["volumecache", "modelstore"]}
    trials = [trial("volumecache", "cold", 25, acquire=2), trial("volumecache", "cold", 99, acquire=2),
              trial("modelstore", "cold", 70, acquire=0), trial("modelstore", "cold", 80, acquire=0)]
    trials[0]["delay_ms"], trials[1]["delay_ms"] = 14000, 90000
    trials[2]["delay_ms"], trials[3]["delay_ms"] = 59000, 66000
    md = report.render(meta, trials)
    assert "## Overall cold request" in md
    section = md.split("## Overall cold request")[1].split("\n## ")[0]
    assert "| volumecache | 62.0s [25.0–99.0] | 52.0s [14.0–90.0] |" in section
    assert "| 25.0s | 14.0s |" in section          # best case: min e2e, min delay
    assert section.index("| volumecache |") < section.index("| modelstore |")   # ranked by best case


def test_combined_run_sums_usage_of_its_source_runs(tmp_path):
    runs = tmp_path / "runs"
    base = {"model": "A/B", "gpu": "H100", "datacenter": "US-CA-2", "cold_trials": 1, "warm_per_cold": 0,
            "parallel": True, "evict_page_cache": True, "strategies": ["nocache"]}
    for rid, (start, end, eid, ms) in {"main": (0.0, 600.0, "e1", 60_000), "topup": (1000.0, 1300.0, "e2", 30_000)}.items():
        d = runs / rid
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({**base, "run_id": rid, "started_at": start, "finished_at": end,
                                                 "endpoints": {"nocache": eid}}))
        (d / "results.jsonl").write_text(json.dumps(trial("nocache", "cold", 30, acquire=10)))
        (d / "billing.json").write_text(json.dumps([{"endpointId": eid, "timeBilledMs": ms, "amount": 0.1}]))
    comb = runs / "comb"
    comb.mkdir()
    (comb / "meta.json").write_text(json.dumps({**base, "run_id": "comb", "started_at": 0.0, "finished_at": 600.0,
                                                "endpoints": {"nocache": "e1"}, "combined_from": ["main", "topup"]}))
    (comb / "results.jsonl").write_text(json.dumps(trial("nocache", "cold", 30, acquire=10)))
    md = open(report.write(str(comb))).read()
    u = json.loads((comb / "summary.json").read_text())["usage"]
    assert u["wall_clock_min"] == pytest.approx(15.0) and u["billed_gpu_min"] == pytest.approx(1.5)
    assert u["billed_usd"] == pytest.approx(0.2) and sorted(u["runs"]) == ["main", "topup"]
    assert "combined from `main`, `topup` (sums over those runs)" in md


def test_render_shows_combined_note_in_header():
    """Merged runs measured on different days must say so where readers look first."""
    meta = {"run_id": "r", "model": "A/B", "gpu": "H100", "datacenter": "US-CA-2", "cold_trials": 1,
            "warm_per_cold": 0, "parallel": True, "evict_page_cache": True, "strategies": ["volumecache"],
            "combined_note": "Global Store rows measured 2026-11-12"}
    md = report.render(meta, [trial("volumecache", "cold", 25, acquire=2)])
    header = md.split("## Answer")[0]
    assert "- **Combined:** Global Store rows measured 2026-11-12" in header


def test_render_omits_combined_line_for_single_runs():
    meta = {"run_id": "r", "model": "A/B", "gpu": "H100", "datacenter": "US-CA-2", "cold_trials": 1,
            "warm_per_cold": 0, "parallel": True, "evict_page_cache": True, "strategies": ["volumecache"]}
    assert "**Combined:**" not in report.render(meta, [trial("volumecache", "cold", 25, acquire=2)])
