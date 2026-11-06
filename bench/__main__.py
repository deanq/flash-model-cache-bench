"""CLI: python -m bench {preflight,run,report,clean}

`run` is the whole pipeline: preflight -> deploy (`rp flash deploy`) -> verify
attachments -> prime -> cold/warm rounds -> report -> teardown. Teardown runs in `finally` (and on
SIGTERM), so a failed or interrupted run still deletes what it created.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

from bench import config as config_mod
from bench import deploy, report
from bench.driver import Driver, Recorder
from bench.runpod_api import Runpod, RunpodError, api_key


def preflight(cfg: config_mod.Config, rp: Runpod) -> None:
    api_key()
    if rp.flash_app_exists(cfg.app_name):
        raise SystemExit(f"app {cfg.app_name!r} already exists — RUN_ID={cfg.run_id!r} was used before. "
                         f"Pick another RUN_ID, or run `make clean` if that run was not torn down.")
    dirty = [p for p in glob.glob("runs/*/state.json") if not json.load(open(p)).get("torn_down")]
    if dirty:
        deploy.log(f"warning: runs with unfinished teardown: {dirty} (`make clean` retries them)")
    deploy.log("preflight ok")


def cmd_run(args) -> int:
    cfg = config_mod.load()
    rp = Runpod()
    preflight(cfg, rp)
    os.makedirs(cfg.run_dir, exist_ok=True)
    meta = {**cfg.to_dict(), "started_at": time.time()}
    _write_json(os.path.join(cfg.run_dir, "meta.json"), meta)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    state = None
    exit_code = 0
    try:
        state = deploy.deploy(cfg, rp)
        meta["endpoints"] = state.endpoints
        meta["app"], meta["volume_id"], meta["global_store_id"] = state.app, state.volume_id, state.global_store_id
        meta["attachments"] = deploy.verify_attachments(cfg, state, rp)
        _write_json(os.path.join(cfg.run_dir, "meta.json"), meta)
        driver = Driver(cfg, rp, state.endpoints, Recorder(os.path.join(cfg.run_dir, "results.jsonl")))
        primed = driver.prime()
        if primed:
            driver.rounds(primed)
        meta["finished_at"] = time.time()
        _write_json(os.path.join(cfg.run_dir, "meta.json"), meta)
    except KeyboardInterrupt:
        deploy.log("interrupted — tearing down")
        exit_code = 130
    except (RunpodError, OSError, ValueError, subprocess.CalledProcessError) as e:
        deploy.log(f"run failed: {e}")
        exit_code = 1
    finally:
        exit_code = _finish(cfg, rp, state, exit_code)
    return exit_code


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def _finish(cfg, rp: Runpod, state, exit_code: int) -> int:
    if os.path.exists(os.path.join(cfg.run_dir, "results.jsonl")):
        path = report.write(cfg.run_dir)
        deploy.log(f"report: {path}")
    if cfg.keep:
        deploy.log("KEEP=1: leaving resources up. Run `make clean` when done — they bill while workers run "
                   "and the volume bills per GB-month.")
        return exit_code
    state = state or _load_state(cfg.run_id) or deploy.State(
        cfg.run_id, cfg.app_name, cfg.env_name, cfg.volume_name,
        global_store_name=cfg.global_store_name if cfg.uses_global_store else "")
    leftovers = deploy.teardown(state, rp)
    if leftovers:
        deploy.log(f"TEARDOWN INCOMPLETE — still present: {leftovers}. Re-run `make clean`.")
        return exit_code or 2
    deploy.log(f"teardown verified: app {state.app}, endpoints {sorted(state.endpoints.values())}, "
               f"network volume {state.volume_id}, global volume {state.global_store_id} are gone")
    return exit_code


def _load_state(run_id: str):
    try:
        return deploy.State.load(run_id)
    except FileNotFoundError:
        return None


def cmd_report(args) -> int:
    run_id = args.run_id or _latest_run()
    path = report.write(os.path.join("runs", run_id))
    print(open(path).read())
    return 0


def cmd_clean(args) -> int:
    """Tear down every run whose state says it is not torn down. State is written before
    anything is created, so every run that may have left resources has one."""
    rp = Runpod()
    states = [deploy.State.load(os.path.basename(os.path.dirname(p))) for p in glob.glob("runs/*/state.json")]
    pending = [s for s in states if not s.torn_down]
    if not pending:
        deploy.log("nothing to clean: every run in runs/ is torn down")
    code = 0
    for s in pending:
        left = deploy.teardown(s, rp)
        deploy.log(f"{s.run_id}: " + ("clean" if not left else f"still present: {left}"))
        code = code or (2 if left else 0)
    return code


def cmd_preflight(args) -> int:
    preflight(config_mod.load(), Runpod())
    return 0


def _latest_run() -> str:
    runs = sorted(d for d in os.listdir("runs") if os.path.isfile(os.path.join("runs", d, "meta.json")))
    if not runs:
        raise SystemExit("no runs/ yet — `make run` first")
    return runs[-1]


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("preflight").set_defaults(fn=cmd_preflight)
    sub.add_parser("clean").set_defaults(fn=cmd_clean)
    r = sub.add_parser("report")
    r.add_argument("run_id", nargs="?")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except config_mod.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
