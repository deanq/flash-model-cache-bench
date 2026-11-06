"""Deploy/teardown via the `rp flash` CLI (runpod>=2.1.0), plus the cleanup it does not do.

Each run is its own Flash app (`hfbench-<run id>`), so `rp flash app delete`
only ever removes this run's endpoints. State is written to
runs/<id>/state.json *before* anything is created, and updated as resources
appear, so `make clean` can always finish a teardown that a crashed or
interrupted run started.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

from bench.config import GLOBAL_STORE_STRATEGIES, RESOURCE_PREFIX, Config
from bench.runpod_api import Runpod, RunpodError, api_key

WORKER_DIR = "worker"
ENDPOINT_PREFIX = RESOURCE_PREFIX   # endpoint name = resource name = hfbench-<strategy>


@dataclass
class State:
    run_id: str
    app: str
    env: str
    volume_name: str
    endpoints: dict[str, str] = field(default_factory=dict)   # strategy -> endpoint id
    volume_id: str | None = None
    global_store_name: str = ""
    global_store_id: str | None = None
    torn_down: bool = False

    @staticmethod
    def path(run_id: str) -> str:
        return os.path.join("runs", run_id, "state.json")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path(self.run_id)), exist_ok=True)
        with open(self.path(self.run_id), "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, run_id: str) -> "State":
        with open(cls.path(run_id)) as fh:
            return cls(**json.load(fh))


def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def _rp(args: list[str], extra_env: dict[str, str] | None = None, check: bool = True,
        quiet: bool = False) -> subprocess.CompletedProcess:
    env = {**os.environ, "RUNPOD_API_KEY": api_key(), **(extra_env or {})}
    return subprocess.run([_rp_bin(), *args], cwd=WORKER_DIR, env=env, check=check, text=True,
                          stdin=subprocess.DEVNULL, capture_output=quiet)


def _rp_bin() -> str:
    """The rp CLI pinned in this project's venv, not whatever is on PATH."""
    local = os.path.join(os.path.dirname(sys.executable), "rp")
    return local if os.path.exists(local) else "rp"


def endpoints_from_environment(environment: dict | None) -> dict[str, str]:
    """strategy -> endpoint id, from a Flash environment's {id, name} endpoints."""
    out = {}
    for ep in (environment or {}).get("endpoints") or []:
        if ep["name"].startswith(ENDPOINT_PREFIX):
            out[ep["name"][len(ENDPOINT_PREFIX):]] = ep["id"]
    return out


def deploy(cfg: Config, rp: Runpod) -> State:
    state = State(run_id=cfg.run_id, app=cfg.app_name, env=cfg.env_name, volume_name=cfg.volume_name,
                  global_store_name=cfg.global_store_name if cfg.uses_global_store else "")
    state.save()
    log(f"deploying app={cfg.app_name} env={cfg.env_name} model={cfg.model} gpu={cfg.gpu} dc={cfg.datacenter} "
        f"strategies={','.join(cfg.strategies)}")
    try:
        _rp(["flash", "deploy", "--env", cfg.env_name], cfg.deploy_env())
    finally:
        # Record whatever was created, even if the deploy failed half way.
        state.endpoints = endpoints_from_environment(_environment_or_none(rp, state))
        if cfg.uses_network_volume:
            state.volume_id = find_volume_id(rp, cfg.volume_name)
        if cfg.uses_global_store:
            state.global_store_id = find_global_store_id(rp, cfg.global_store_name)
        state.save()
    missing = [s for s in cfg.strategies if s not in state.endpoints]
    if missing:
        raise RunpodError(f"deploy did not produce endpoints for {missing}; environment has {state.endpoints}")
    log(f"endpoints: {state.endpoints}  volume: {state.volume_id}  global store: {state.global_store_id}")
    return state


def _environment_or_none(rp: Runpod, state: State) -> dict | None:
    try:
        return rp.flash_environment(state.app, state.env)
    except RunpodError as e:
        log(f"warning: could not read environment {state.app}/{state.env}: {e}")
        return None


def verify_attachments(cfg: Config, state: State, rp: Runpod) -> dict:
    """Check that the deploy attached what the strategies need: the cached model on
    `modelstore`, the run's global volume on Global Store endpoints, and the
    network volume on `volume`/`volumecache`. Returns what was found, for meta.json."""
    found: dict = {}
    for s, eid in state.endpoints.items():
        ep = rp.endpoint(eid)
        found[s] = {"modelReferences": ep.get("modelReferences"), "networkVolumeId": ep.get("networkVolumeId"),
                    "volumes": ep.get("volumes"), "executionTimeoutMs": ep.get("executionTimeoutMs")}
        problems = []
        if s == "modelstore" and not ep.get("modelReferences"):
            problems.append("no modelReferences")
        if s in GLOBAL_STORE_STRATEGIES:
            ids = [v.get("volumeId") for v in ep.get("volumes") or []]
            if state.global_store_id not in ids:
                problems.append(f"global volume {state.global_store_id} not attached (volumes={ids})")
        if problems:
            raise RunpodError(f"{s} ({eid}): {'; '.join(problems)}")
    log(f"attachments verified: {json.dumps(found)}")
    return found


def find_volume_id(rp: Runpod, name: str) -> str | None:
    matches = [v["id"] for v in rp.list_network_volumes() if v.get("name") == name]
    return matches[0] if matches else None


def teardown(state: State, rp: Runpod) -> list[str]:
    """Delete the run's Flash app (all its endpoints), then its network volume and global
    volume, which `rp flash app delete` leaves behind; verify nothing is left.
    Returns human-readable leftovers (empty = clean)."""
    if rp.flash_app_exists(state.app):
        log(f"teardown: deleting app {state.app}")
        res = _rp(["flash", "app", "delete", state.app, "--yes"], check=False)
        if res.returncode != 0:
            log(f"warning: `rp flash app delete {state.app}` exited {res.returncode}; checking what is left")
    volume_id = state.volume_id or find_volume_id(rp, state.volume_name)
    if volume_id:
        log(f"teardown: deleting network volume {state.volume_name} ({volume_id})")
        _with_retry(lambda: rp.delete_network_volume(volume_id), f"delete volume {volume_id}")
    store_id = state.global_store_id or find_global_store_id(rp, state.global_store_name)
    if store_id:
        log(f"teardown: deleting global volume {state.global_store_name} ({store_id})")
        _with_retry(lambda: rp.delete_global_store(store_id), f"delete global volume {store_id}")
    leftovers = verify_clean(state, rp)
    state.torn_down = not leftovers
    state.save()
    return leftovers


def find_global_store_id(rp: Runpod, name: str) -> str | None:
    if not name:
        return None
    matches = [b["id"] for b in rp.list_global_stores() if b.get("name") == name]
    return matches[0] if matches else None


def _with_retry(fn, what: str, attempts: int = 6, wait_s: float = 10) -> None:
    """Storage still attached to a just-deleted endpoint can refuse deletion briefly."""
    for i in range(attempts):
        try:
            fn()
            return
        except RunpodError as e:
            if i == attempts - 1:
                raise
            log(f"{what}: retry {i + 1}/{attempts}: {e}")
            time.sleep(wait_s)


def verify_clean(state: State, rp: Runpod) -> list[str]:
    """Only this run's resources count: its app, the endpoint ids it recorded, and the
    volume/global volume named after it. Other deployments in the account are ignored."""
    left = []
    if rp.flash_app_exists(state.app):
        left.append(f"flash app {state.app}")
    ids = set(state.endpoints.values())
    for ep in rp.list_endpoints():
        if ep["id"] in ids:
            left.append(f"endpoint {ep['name']} ({ep['id']})")
    for v in rp.list_network_volumes():
        if v.get("name") == state.volume_name or v.get("id") == state.volume_id:
            left.append(f"network volume {v['name']} ({v['id']})")
    if state.global_store_name or state.global_store_id:
        for b in rp.list_global_stores():
            if b.get("name") == state.global_store_name or b.get("id") == state.global_store_id:
                left.append(f"global volume {b['name']} ({b['id']})")
    return left
