"""Flash app: one queue-based GPU endpoint per model-acquisition strategy.

All four run the same handler code on the same GPU class in the same
datacenter; only the storage path differs. Deploy-time configuration comes from
BENCH_* environment variables set by the orchestrator (bench/cli.py) -- the
defaults below make a bare `flash build` in this directory work too.

workers=(0, 1) and a short idle timeout are deliberate: the driver forces a
scale-to-zero between trials, and a single worker per endpoint makes "is this
request cold?" unambiguous (the handler also reports it).
"""
import os

from runpod_flash import DataCenter, Endpoint, GpuGroup, GpuType
from runpod_flash.core.resources.network_volume import NetworkVolume

import runtime

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-3B-Instruct")
RUN_ID = os.environ.get("BENCH_RUN_ID", "local")
DATACENTER = DataCenter(os.environ.get("BENCH_DC", "EU-RO-1"))
VOLUME_NAME = os.environ.get("BENCH_VOLUME_NAME", f"hfbench-{RUN_ID}")
VOLUME_GB = int(os.environ.get("BENCH_VOLUME_GB", "50"))
IDLE_TIMEOUT_S = int(os.environ.get("BENCH_IDLE_TIMEOUT_S", "5"))
EXECUTION_TIMEOUT_MS = 30 * 60 * 1000   # first population downloads the whole model


def _gpus() -> list:
    """BENCH_GPU is a comma list of GpuGroup or GpuType names (e.g. ADA_24)."""
    out = []
    for name in os.environ.get("BENCH_GPU", "ADA_24").split(","):
        name = name.strip()
        out.append(GpuGroup[name] if name in GpuGroup.__members__ else GpuType[name])
    return out


def _env(strategy: str) -> dict:
    env = {"BENCH_STRATEGY": strategy, "BENCH_MODEL": MODEL, "BENCH_EVICT": os.environ.get("BENCH_EVICT", "1"),
           "BENCH_NAMESPACE": f"hfbench-{RUN_ID}", "HF_HUB_DISABLE_TELEMETRY": "1"}
    if os.environ.get("HF_TOKEN"):      # only needed for gated/private models
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return env


_volume = NetworkVolume(name=VOLUME_NAME, size=VOLUME_GB, datacenter=DATACENTER)


def _config(strategy: str, volume) -> dict:
    return {"name": f"hfbench-{strategy}", "gpu": _gpus(), "datacenter": DATACENTER, "workers": (0, 1),
            "idle_timeout": IDLE_TIMEOUT_S, "execution_timeout_ms": EXECUTION_TIMEOUT_MS,
            "volume": volume, "env": _env(strategy)}


# `flash build` reads dependencies=[...] from the source AST, so each decorator
# repeats the literal list instead of sharing a constant. Keep the four in sync.
@Endpoint(**_config("nocache", None), dependencies=["transformers>=4.56", "accelerate>=1.0", "safetensors"])
async def hfbench_nocache(**payload) -> dict:
    return runtime.handle("nocache", payload)


@Endpoint(**_config("volume", _volume), dependencies=["transformers>=4.56", "accelerate>=1.0", "safetensors"])
async def hfbench_volume(**payload) -> dict:
    return runtime.handle("volume", payload)


@Endpoint(**_config("volumecache", _volume), dependencies=["transformers>=4.56", "accelerate>=1.0", "safetensors"])
async def hfbench_volumecache(**payload) -> dict:
    return runtime.handle("volumecache", payload)


# The cached model is attached after deploy (bench/runpod_api.py): Flash has no
# modelReferences field yet.
@Endpoint(**_config("modelstore", None), dependencies=["transformers>=4.56", "accelerate>=1.0", "safetensors"])
async def hfbench_modelstore(**payload) -> dict:
    return runtime.handle("modelstore", payload)
