"""Run configuration: environment variables (set via .env or `make run VAR=...`),
validated once at startup so a typo fails before anything is deployed."""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass

APP_NAME = "hf-cache-bench"
STRATEGIES = ("nocache", "volume", "volumecache", "modelstore")
DATACENTERS = ("US-CA-2", "US-IL-1", "US-KS-2", "US-MO-1", "US-MO-2", "US-NC-2", "US-NE-1", "US-WA-1",
               "EU-CZ-1", "EU-RO-1", "EUR-NO-1")   # runpod_flash.DataCenter (network-volume capable)
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    run_id: str
    model: str
    gpu: str
    datacenter: str
    volume_gb: int
    strategies: tuple[str, ...]
    cold_trials: int
    warm_per_cold: int
    max_new_tokens: int
    parallel: bool
    evict_page_cache: bool
    prime_timeout_s: int
    trial_timeout_s: int
    keep: bool                   # skip teardown (debugging only)

    @property
    def env_name(self) -> str:
        return f"run-{self.run_id}"

    @property
    def volume_name(self) -> str:
        return f"hfbench-{self.run_id}"

    @property
    def run_dir(self) -> str:
        return os.path.join("runs", self.run_id)

    def deploy_env(self) -> dict[str, str]:
        """BENCH_* variables read by worker/bench_app.py at `flash deploy` time."""
        env = {"BENCH_RUN_ID": self.run_id, "BENCH_MODEL": self.model, "BENCH_GPU": self.gpu,
               "BENCH_DC": self.datacenter, "BENCH_VOLUME_NAME": self.volume_name,
               "BENCH_VOLUME_GB": str(self.volume_gb), "BENCH_EVICT": "1" if self.evict_page_cache else "0"}
        return env

    def to_dict(self) -> dict:
        return asdict(self)


def _str(env, name: str, default: str) -> str:
    """Unset and empty both mean "default" (make exports empty vars)."""
    return env.get(name) or default


def _int(env, name: str, default: int, lo: int, hi: int) -> int:
    raw = _str(env, name, str(default))
    try:
        val = int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not an integer") from None
    if not lo <= val <= hi:
        raise ConfigError(f"{name}={val} out of range [{lo}, {hi}]")
    return val


def _bool(env, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use 1/0)")


def load(env=None, *, run_id: str | None = None) -> Config:
    env = os.environ if env is None else env
    model = _str(env, "MODEL", "Qwen/Qwen2.5-3B-Instruct")
    if not _REPO_ID.match(model):
        raise ConfigError(f"MODEL={model!r} is not a Hugging Face repo id (org/name)")
    dc = _str(env, "DATACENTER", "EU-RO-1")
    if dc not in DATACENTERS:
        raise ConfigError(f"DATACENTER={dc!r} not one of {DATACENTERS}")
    strategies = tuple(s.strip() for s in _str(env, "STRATEGIES", ",".join(STRATEGIES)).split(",") if s.strip())
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown or not strategies:
        raise ConfigError(f"STRATEGIES has unknown/empty entries {unknown}; choose from {STRATEGIES}")
    return Config(
        run_id=run_id or env.get("RUN_ID") or time.strftime("%Y%m%d-%H%M%S"),
        model=model,
        gpu=_str(env, "GPU", "ADA_24"),
        datacenter=dc,
        volume_gb=_int(env, "VOLUME_GB", 50, 10, 4000),
        strategies=strategies,
        cold_trials=_int(env, "COLD_TRIALS", 3, 1, 50),
        warm_per_cold=_int(env, "WARM_PER_COLD", 2, 0, 50),
        max_new_tokens=_int(env, "MAX_NEW_TOKENS", 128, 1, 4096),
        parallel=_bool(env, "PARALLEL", True),
        evict_page_cache=_bool(env, "EVICT_PAGE_CACHE", True),
        prime_timeout_s=_int(env, "PRIME_TIMEOUT_S", 45 * 60, 60, 4 * 3600),
        trial_timeout_s=_int(env, "TRIAL_TIMEOUT_S", 20 * 60, 60, 4 * 3600),
        keep=_bool(env, "KEEP", False),
    )
