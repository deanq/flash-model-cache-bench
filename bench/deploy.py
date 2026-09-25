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
from dataclasses import asdict, dataclass, field

from bench.config import APP_NAME, Config
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
    state = State(run_id=cfg.run_id, app=APP_NAME, env=cfg.env_name, volume_name=cfg.volume_name)
    state.save()
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


def find_volume_id(rp: Runpod, name: str) -> str | None:
    matches = [v["id"] for v in rp.list_network_volumes() if v.get("name") == name]
    return matches[0] if matches else None


def teardown(state: State, rp: Runpod) -> list[str]:
    """Delete the Flash app (all its endpoints) and the run's network volume, then
    verify nothing is left. Returns human-readable leftovers (empty = clean)."""
    log(f"teardown: deleting app {state.app}")
    res = _flash(["app", "delete", state.app], check=False, stdin="y\n")
    if res.returncode != 0:
        log(f"warning: `flash app delete {state.app}` exited {res.returncode}; checking what is left")
    volume_id = state.volume_id or find_volume_id(rp, state.volume_name)
    if volume_id:
        log(f"teardown: deleting network volume {state.volume_name} ({volume_id})")
        _delete_volume_with_retry(rp, volume_id)
    leftovers = verify_clean(state, rp)
    state.torn_down = not leftovers
    state.save()
    return leftovers


def _delete_volume_with_retry(rp: Runpod, volume_id: str, attempts: int = 6) -> None:
    """A volume still attached to a just-deleted endpoint can refuse deletion briefly."""
    import time
    for i in range(attempts):
        try:
            rp.delete_network_volume(volume_id)
            return
        except RunpodError as e:
            if i == attempts - 1:
                raise
            log(f"volume delete retry {i + 1}/{attempts}: {e}")
            time.sleep(10)


def verify_clean(state: State, rp: Runpod) -> list[str]:
    left = []
    ids = set(state.endpoints.values())
    for ep in rp.list_endpoints():
        if ep["id"] in ids or ep["name"].startswith(ENDPOINT_PREFIX):
            left.append(f"endpoint {ep['name']} ({ep['id']})")
    for v in rp.list_network_volumes():
        if v.get("name") == state.volume_name or v.get("id") == state.volume_id:
            left.append(f"network volume {v['name']} ({v['id']})")
    return left
