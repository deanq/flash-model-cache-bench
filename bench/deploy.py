"""Deploy/teardown via the flash CLI, plus the pieces Flash cannot do yet.

State is written to runs/<id>/state.json *before* anything is created, and
updated as resources appear, so `make clean` can always finish a teardown that
a crashed or interrupted run started.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

from bench.config import APP_NAME, GLOBAL_STORE_STRATEGIES, Config
from bench.runpod_api import Runpod, RunpodError, model_reference

WORKER_DIR = "worker"
MANIFEST = os.path.join(WORKER_DIR, ".flash", "flash_manifest.json")
ENDPOINT_PREFIX = "hfbench-"


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


def _flash(args: list[str], extra_env: dict[str, str] | None = None, check: bool = True,
           stdin: str | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    cmd = [_flash_bin(), *args]
    return subprocess.run(cmd, cwd=WORKER_DIR, env={**os.environ, **(extra_env or {})}, check=check,
                          text=True, input=stdin, capture_output=quiet)


def _flash_bin() -> str:
    """The flash CLI pinned in this project's venv, not whatever is on PATH."""
    local = os.path.join(os.path.dirname(sys.executable), "flash")
    return local if os.path.exists(local) else "flash"


def endpoints_from_manifest(manifest: dict) -> dict[str, str]:
    out = {}
    for name, res in manifest.get("resources", {}).items():
        if name.startswith(ENDPOINT_PREFIX) and res.get("endpoint_id"):
            out[name[len(ENDPOINT_PREFIX):]] = res["endpoint_id"]
    return out


def deploy(cfg: Config, rp: Runpod) -> State:
    state = State(run_id=cfg.run_id, app=APP_NAME, env=cfg.env_name, volume_name=cfg.volume_name,
                  global_store_name=cfg.global_store_name if cfg.uses_global_store else "")
    state.save()
    if cfg.uses_global_store:
        state.global_store_id = rp.create_global_store(cfg.global_store_name)
        state.save()
        log(f"created global store {cfg.global_store_name} ({state.global_store_id})")
    log(f"deploying app={APP_NAME} env={cfg.env_name} model={cfg.model} gpu={cfg.gpu} dc={cfg.datacenter}")
    # Flash caches provisioned resources in worker/.flash/resources.pkl and tries
    # to *update* them on the next deploy; after a teardown those ids are gone
    # ("Endpoint not found"). Every run provisions from scratch.
    shutil.rmtree(os.path.join(WORKER_DIR, ".flash"), ignore_errors=True)
    _flash(["deploy", "--app", APP_NAME, "--env", cfg.env_name], cfg.deploy_env())
    with open(MANIFEST) as fh:
        state.endpoints = endpoints_from_manifest(json.load(fh))
    state.volume_id = find_volume_id(rp, cfg.volume_name)
    state.save()
    missing = [s for s in cfg.strategies if s not in state.endpoints]
    if missing:
        raise RunpodError(f"deploy did not produce endpoints for {missing}; manifest has {state.endpoints}")
    log(f"endpoints: {state.endpoints}  volume: {state.volume_id}")
    return state


def attach_cached_model(cfg: Config, state: State, rp: Runpod) -> str | None:
    """Model Store needs modelReferences on the endpoint. Flash 1.20 has no field
    for it, so it is set with the same GraphQL mutation the console uses."""
    eid = state.endpoints.get("modelstore")
    if eid is None:
        return None
    saved = rp.set_model_references(eid, [model_reference(cfg.model)])
    ref = saved["modelReferences"][0]
    log(f"modelstore: attached cached model {ref}")
    return ref


def attach_global_store(cfg: Config, state: State, rp: Runpod) -> None:
    """Global Store endpoints get the run's global volume after deploy."""
    for s in GLOBAL_STORE_STRATEGIES:
        if s in cfg.strategies and s in state.endpoints:
            rp.attach_global_volume(state.endpoints[s], state.global_store_id)
            log(f"{s}: attached global store {state.global_store_id}")


def find_volume_id(rp: Runpod, name: str) -> str | None:
    matches = [v["id"] for v in rp.list_network_volumes() if v.get("name") == name]
    return matches[0] if matches else None


def teardown(state: State, rp: Runpod) -> list[str]:
    """Delete the Flash app (all its endpoints), the run's network volume and global store, then
    verify nothing is left. Returns human-readable leftovers (empty = clean)."""
    log(f"teardown: deleting app {state.app}")
    res = _flash(["app", "delete", state.app], check=False, stdin="y\n")
    if res.returncode != 0:
        log(f"warning: `flash app delete {state.app}` exited {res.returncode}; checking what is left")
    volume_id = state.volume_id or find_volume_id(rp, state.volume_name)
    if volume_id:
        log(f"teardown: deleting network volume {state.volume_name} ({volume_id})")
        _with_retry(lambda: rp.delete_network_volume(volume_id), f"delete volume {volume_id}")
    store_id = state.global_store_id or find_global_store_id(rp, state.global_store_name)
    if store_id:
        log(f"teardown: deleting global store {state.global_store_name} ({store_id})")
        _with_retry(lambda: rp.delete_global_store(store_id), f"delete global store {store_id}")
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
    left = []
    ids = set(state.endpoints.values())
    for ep in rp.list_endpoints():
        if ep["id"] in ids or ep["name"].startswith(ENDPOINT_PREFIX):
            left.append(f"endpoint {ep['name']} ({ep['id']})")
    for v in rp.list_network_volumes():
        if v.get("name") == state.volume_name or v.get("id") == state.volume_id:
            left.append(f"network volume {v['name']} ({v['id']})")
    if state.global_store_name or state.global_store_id:
        for b in rp.list_global_stores():
            if b.get("name") == state.global_store_name or b.get("id") == state.global_store_id:
                left.append(f"global store {b['name']} ({b['id']})")
    return left
