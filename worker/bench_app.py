"""Flash app (runpod.apps): one queue-based GPU endpoint per model-acquisition strategy.

All strategies run the same handler code on the same GPU class in the same
datacenter; only the storage path differs. Deploy-time configuration comes from
BENCH_* environment variables set by the orchestrator (bench/deploy.py) -- the
defaults below make a bare `rp flash deploy` in this directory work too.

Only the strategies in BENCH_STRATEGIES are registered, so a run never creates
storage it does not use: the network volume exists only if `volume` or
`volumecache` is selected, the global volume only if a Global Store strategy is.

workers=(0, 1) and a short idle timeout are deliberate: the driver forces a
scale-to-zero between trials, and a single worker per endpoint makes "is this
request cold?" unambiguous (the handler also reports it).
"""
import os

from runpod import App, GlobalVolume, Model, NetworkVolume

import runtime

ENDPOINT_PREFIX = "hfbench-"
ALL_STRATEGIES = ("nocache", "volume", "volumecache", "modelstore", "globalstore", "gsvolumecache", "gshydrate",
                  "volumeprefetch", "volumecache864")
NETWORK_VOLUME_STRATEGIES = ("volume", "volumecache", "volumeprefetch", "volumecache864")
GLOBAL_STORE_STRATEGIES = ("globalstore", "gsvolumecache", "gshydrate")
DEPENDENCIES = ["transformers>=4.56", "accelerate>=1.0", "safetensors"]

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-3B-Instruct")
RUN_ID = os.environ.get("BENCH_RUN_ID", "local")
APP_NAME = os.environ.get("BENCH_APP", f"{ENDPOINT_PREFIX}{RUN_ID}")
DATACENTER = os.environ.get("BENCH_DC", "EU-RO-1")
VOLUME_NAME = os.environ.get("BENCH_VOLUME_NAME", f"{ENDPOINT_PREFIX}{RUN_ID}")
GLOBAL_STORE_NAME = os.environ.get("BENCH_GLOBAL_STORE_NAME", f"{ENDPOINT_PREFIX}{RUN_ID}")
VOLUME_GB = int(os.environ.get("BENCH_VOLUME_GB", "50"))
IDLE_TIMEOUT_S = int(os.environ.get("BENCH_IDLE_TIMEOUT_S", "5"))
EXECUTION_TIMEOUT_MS = 30 * 60 * 1000   # first population downloads the whole model
SELECTED = tuple(s.strip() for s in os.environ.get("BENCH_STRATEGIES", ",".join(ALL_STRATEGIES)).split(",")
                 if s.strip())

app = App(APP_NAME)

# Volumes are references; `rp flash deploy` creates them on first use (and
# `rp flash app delete` leaves them behind -- bench/deploy.py deletes them).
_network_volume = NetworkVolume(VOLUME_NAME, size=VOLUME_GB, datacenter=DATACENTER)
_global_store = GlobalVolume(GLOBAL_STORE_NAME)


def _gpus() -> list[str]:
    """BENCH_GPU is a comma list of GpuGroup/GpuType names or shorthands (e.g. ADA_24, 4090)."""
    return [g.strip() for g in os.environ.get("BENCH_GPU", "ADA_24").split(",") if g.strip()]


def _env(strategy: str) -> dict:
    env = {"BENCH_STRATEGY": strategy, "BENCH_MODEL": MODEL, "BENCH_EVICT": os.environ.get("BENCH_EVICT", "1"),
           "BENCH_NAMESPACE": f"{ENDPOINT_PREFIX}{RUN_ID}", "HF_HUB_DISABLE_TELEMETRY": "1",
           # The worker imports this module too; give it the same app identity and selection.
           "BENCH_APP": APP_NAME, "BENCH_RUN_ID": RUN_ID, "BENCH_STRATEGIES": ",".join(SELECTED)}
    if os.environ.get("HF_TOKEN"):      # only needed for gated/private models
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return env


def _mounts(strategy: str) -> dict | None:
    """An endpoint takes one volume at /runpod-volume, so network-volume and
    Global Store strategies get separate endpoints."""
    if strategy in NETWORK_VOLUME_STRATEGIES:
        return {"/runpod-volume": _network_volume}
    if strategy in GLOBAL_STORE_STRATEGIES:
        return {"/runpod-volume": _global_store}
    return None


def endpoint_config(strategy: str) -> dict:
    return {"name": f"{ENDPOINT_PREFIX}{strategy}", "gpu": _gpus(), "datacenter": DATACENTER, "workers": (0, 1),
            "idle_timeout": IDLE_TIMEOUT_S, "execution_timeout_ms": EXECUTION_TIMEOUT_MS,
            "mounts": _mounts(strategy), "env": _env(strategy), "dependencies": DEPENDENCIES,
            # Model Store: the platform stages the weights before the worker starts.
            "model": Model(MODEL) if strategy == "modelstore" else None}


def bench_endpoint(strategy: str):
    """`@app.queue(...)` for selected strategies; unselected ones stay plain functions
    so they are neither deployed nor create storage."""
    def decorate(fn):
        return app.queue(**endpoint_config(strategy))(fn) if strategy in SELECTED else fn
    return decorate


@bench_endpoint("nocache")
async def hfbench_nocache(**payload) -> dict:
    return runtime.handle("nocache", payload)


@bench_endpoint("volume")
async def hfbench_volume(**payload) -> dict:
    return runtime.handle("volume", payload)


@bench_endpoint("volumecache")
async def hfbench_volumecache(**payload) -> dict:
    return runtime.handle("volumecache", payload)


@bench_endpoint("modelstore")
async def hfbench_modelstore(**payload) -> dict:
    return runtime.handle("modelstore", payload)


@bench_endpoint("globalstore")
async def hfbench_globalstore(**payload) -> dict:
    return runtime.handle("globalstore", payload)


@bench_endpoint("gsvolumecache")
async def hfbench_gsvolumecache(**payload) -> dict:
    return runtime.handle("gsvolumecache", payload)


@bench_endpoint("gshydrate")
async def hfbench_gshydrate(**payload) -> dict:
    return runtime.handle("gshydrate", payload)


@bench_endpoint("volumeprefetch")
async def hfbench_volumeprefetch(**payload) -> dict:
    return runtime.handle("volumeprefetch", payload)


@bench_endpoint("volumecache864")
async def hfbench_volumecache864(**payload) -> dict:
    return runtime.handle("volumecache864", payload)
