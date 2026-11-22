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
# a placed worker shows up as a pod within seconds even while the model is pulling; none after this = no capacity
ABANDON_S = 300
# a pod staging a large model is not RUNNING yet; only all-pods-exited this long counts as a stall
STALL_S = 600
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
class Idle:
    seconds: int


Step = Deploy | Teardown | AwaitScaleDown | Idle


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

    def endpoint_name(self, run_id: str) -> str:
        dc = self.dc.lower().replace("us-", "").replace("-", "")
        return f"{NAME_PREFIX}{dc}-{self.sid.lower()}{self.arm}-{self.gpu}-{run_id}"[:60]

    @property
    def deploys(self) -> list[Deploy]:
        return [s for s in self.steps if isinstance(s, Deploy)]


def s2(*, dc: str, gpu: str, model: str, gap_s: int, repeats: int = 3) -> Scenario:
    """Warm reuse after teardown: seed, then [teardown, idle gap, redeploy] x repeats."""
    steps: list[Step] = [Deploy("seed", READY_HOLD_S)]
    for _ in range(repeats):
        steps += [Teardown(), Idle(gap_s), Deploy("measure", READY_HOLD_S)]
    return Scenario("S2", f"gap{gap_s}", dc, gpu, model, tuple(steps))


def s9(*, dc: str, gpu: str, model: str, repeats: int = 3) -> Scenario:
    """Scale-from-zero on a cached model: workersMax stays 1, idle timeout removes the worker."""
    steps: list[Step] = [Deploy("seed", READY_HOLD_S)]
    for _ in range(repeats):
        steps += [AwaitScaleDown(), Idle(SETTLE_S), Deploy("measure", MEASURE_HOLD_S, scale_up=False)]
    return Scenario("S9", "", dc, gpu, model, tuple(steps))
