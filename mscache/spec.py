"""Scenario specs: one variable per scenario, expressed as an ordered list of steps.

A scenario runs on ONE endpoint pinned to one DC and one GPU pool, with ONE worker.
Each Deploy step is one load (one record). Steps are data; mscache.driver executes them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NAME_PREFIX = "msc-"
# handler ready + 2 min: the host reports the copy DEPLOYED before teardown, so LRU keeps it
READY_HOLD_S = 120
MEASURE_HOLD_S = 10
# no pod and no /health worker this long = no capacity. TX-3 placed a pod 470 s after submit on 11-22,
# and the program rule is "no workers for 15 min = stop"
ABANDON_S = 900
# a pod staging a large model is not RUNNING yet; only all-pods-exited this long counts as a stall
STALL_S = 600
# worker placed but waiting for a free GPU on its machine; the program rule is "no workers for 15 min = stop"
THROTTLE_S = 900
ROUND_TIMEOUT_S = 30 * 60
TEARDOWN_TIMEOUT_S = 600
SETTLE_S = 60
IDLE_TIMEOUT_S = 5
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


@dataclass(frozen=True)
class Gpu:
    pool: str
    usd_per_h: float           # serverless price from GET /v2/catalog/gpus, 2026-11-21
    excluded: tuple[str, ...] = ()


# narrow pools only: one GPU type per scenario keeps placement a constant, not a variable
GPUS = {
    "a40": Gpu("AMPERE_48", 1.22),
    "l40s": Gpu("ADA_48_PRO", 1.89, ("NVIDIA L40", "NVIDIA RTX 6000 Ada Generation",
                                     "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb")),
    "4090": Gpu("ADA_24", 1.34),
    "a100": Gpu("AMPERE_80", 2.99),
    "h100": Gpu("ADA_80_PRO", 5.89),
    "l4": Gpu("AMPERE_24", 0.75),
}


@dataclass(frozen=True)
class Deploy:
    role: str                  # "seed" (warms the copy, not measured) or "measure"
    hold_s: int
    scale_up: bool = True      # False: workersMax stays put, the job itself triggers scale-from-zero


@dataclass(frozen=True)
class Teardown:
    """workersMax 0, then wait until no pod is live (the copy's last pod has left)."""


@dataclass(frozen=True)
class AwaitScaleDown:
    """Leave workersMax alone; wait until idle timeout has removed every pod."""


@dataclass(frozen=True)
class Recreate:
    """Delete the endpoint and create a fresh one (same DC, GPU pool, model). ai-api resumes a stopped
    worker (same pod id, same machine) before placing a new one, so only a new endpoint guarantees
    the next load is a NEW worker and measures LRU retention, not a resume."""


@dataclass(frozen=True)
class Idle:
    seconds: int


Step = Deploy | Teardown | AwaitScaleDown | Recreate | Idle


@dataclass(frozen=True)
class Scenario:
    sid: str
    arm: str
    dc: str
    gpu: str
    model: str
    steps: tuple[Step, ...]

    def __post_init__(self):
        if self.gpu not in GPUS:
            raise ValueError(f"unknown gpu {self.gpu!r}; one of {sorted(GPUS)}")
        if not _REPO.match(self.model):
            raise ValueError(f"model must be an HF repo id like org/name, got {self.model!r}")

    def endpoint_name(self, run_id: str, n: int = 0) -> str:
        """n > 0 names the n-th recreated endpoint of the scenario."""
        dc = self.dc.lower().replace("us-", "").replace("-", "")
        suffix = f"-{n}" if n else ""
        return f"{NAME_PREFIX}{dc}-{self.sid.lower()}{self.arm}-{self.gpu}-{run_id}"[:60 - len(suffix)] + suffix

    @property
    def deploys(self) -> list[Deploy]:
        return [s for s in self.steps if isinstance(s, Deploy)]


def fresh_redeploy(gap_s: int, hold_s: int = READY_HOLD_S) -> list[Step]:
    """Teardown, new endpoint, gap, deploy: the redeploy is a new worker (for S2, S3, S10)."""
    return [Teardown(), Recreate(), Idle(gap_s), Deploy("measure", hold_s)]


def s2(*, dc: str, gpu: str, model: str, gap_s: int, repeats: int = 3) -> Scenario:
    """Warm reuse after teardown: seed, then [teardown, new endpoint, idle gap, redeploy] x repeats."""
    steps: list[Step] = [Deploy("seed", READY_HOLD_S)]
    for _ in range(repeats):
        steps += fresh_redeploy(gap_s)
    return Scenario("S2", f"gap{gap_s}", dc, gpu, model, tuple(steps))


def s9(*, dc: str, gpu: str, model: str, repeats: int = 3) -> Scenario:
    """Scale-from-zero on a cached model: workersMax stays 1, idle timeout removes the worker."""
    steps: list[Step] = [Deploy("seed", READY_HOLD_S)]
    for _ in range(repeats):
        steps += [AwaitScaleDown(), Idle(SETTLE_S), Deploy("measure", MEASURE_HOLD_S, scale_up=False)]
    return Scenario("S9", "", dc, gpu, model, tuple(steps))
