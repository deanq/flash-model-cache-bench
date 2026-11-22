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
# S4 holder keeps its worker while the peers place (ready + 2 min settle + ~1-3 min placement)
S4_HOLD_S = 300
# S5+: the cold pulls get this head start, so the warm deploy lands while they hold the download slots
S5_WARM_DELAY_S = 30
AWAIT_TIMEOUT_S = ROUND_TIMEOUT_S
SIZES = ("small", "medium", "large")   # <10 GB, 20-40 GB, 100+ GB
# org/name, optionally :revision (S8 points one at a commit that does not exist)
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+(:[\w.-]+)?$")


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
    role: str                  # "seed" (warms, not measured), "measure", "cold" (first pull) or "failure"
    hold_s: int
    scale_up: bool = True      # False: workersMax stays put, the job itself triggers scale-from-zero
    signal: str | None = None  # set this tag once the job is running (handler up, copy on the host)


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
class Interrupt:
    """S3: submit, then tear down `after_s` after pod_created (the copy is staged before the container
    starts, so container start cannot anchor a mid-download teardown). Records whether the container
    had started by then; if it had, the round is an S2 round, not S3."""
    after_s: int


@dataclass(frozen=True)
class Signal:
    """Set a tag other scenarios of the same run can Await."""
    tag: str


@dataclass(frozen=True)
class Await:
    """Wait (endpoint at 0 workers, $0) until another scenario sets the tag; AWAIT_TIMEOUT_S ends it."""
    tag: str


@dataclass(frozen=True)
class Idle:
    seconds: int


Step = Deploy | Teardown | AwaitScaleDown | Recreate | Interrupt | Signal | Await | Idle


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


def s1(*, dc: str, gpu: str, model: str) -> Scenario:
    """Cold first pull of a model never seen in the DC (seeds of other scenarios are S1 samples too)."""
    return Scenario("S1", "", dc, gpu, model, (Deploy("cold", READY_HOLD_S),))


def s3(*, dc: str, gpu: str, model: str, after_s: int) -> Scenario:
    """Short-round teardown at pod_created + after_s on a 100+ GB model, then an immediate redeploy on a
    new endpoint: does it wait on the first download, or pull the same bytes twice?"""
    steps = (Interrupt(after_s), Recreate(), Deploy("measure", READY_HOLD_S))
    return Scenario("S3", f"t{after_s}", dc, gpu, model, steps)


def s7(*, dc: str, gpu: str, model: str, size: str) -> Scenario:
    """Size spread: cold pull, then one cached redeploy on a new endpoint (S2 covers cached repeats)."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    steps = (Deploy("cold", READY_HOLD_S), *fresh_redeploy(60))
    return Scenario("S7", size, dc, gpu, model, steps)


def s10(*, dc: str, gpu: str, model: str, repeats: int = 3) -> Scenario:
    """Teardown then immediate redeploy (no gap) on a new endpoint: does the new worker start cleanly?"""
    steps: list[Step] = [Deploy("seed", READY_HOLD_S)]
    for _ in range(repeats):
        steps += fresh_redeploy(0)
    return Scenario("S10", "", dc, gpu, model, tuple(steps))


def s8(*, dc: str, gpu: str, model: str, arm: str) -> Scenario:
    """Failure path: an unreadable (gated/private) repo or a revision that does not exist.
    Measures time to failure and the error shown; a rejected create is a result too."""
    return Scenario("S8", arm, dc, gpu, model, (Deploy("failure", MEASURE_HOLD_S),))


def s4(*, dc: str, gpu: str, model: str, peers: int = 3) -> list[Scenario]:
    """Peer vs origin: a holder pulls the model; once it is running and settled, `peers` endpoints with
    the same model deploy at once. Peers landing on other machines are peer-pull candidates (the main
    Mac's host-log join says peer or origin). Run in peer-live DCs only."""
    tag = f"S4:{model}"
    holder = Scenario("S4", "holder", dc, gpu, model, (Deploy("seed", S4_HOLD_S, signal=tag),))
    rest = [Scenario("S4", f"peer{n}", dc, gpu, model, (Await(tag), Idle(READY_HOLD_S), Deploy("measure", READY_HOLD_S)))
            for n in range(1, peers + 1)]
    return [holder, *rest]


def s5(*, dc: str, gpu: str, cold: list[str], warm: str | None = None) -> list[Scenario]:
    """Slot contention: cold pulls of different large models at once (they co-land, as in the pilot).
    S5+ (warm given): a cached model redeploys while the cold pulls hold the host's 4 download slots."""
    if warm is None:
        return [Scenario("S5", f"cold{n}", dc, gpu, m, (Deploy("cold", READY_HOLD_S),))
                for n, m in enumerate(cold, 1)]
    tag = f"S5:{warm}"
    warm_sc = Scenario("S5", "warm", dc, gpu, warm,
                       (Deploy("seed", READY_HOLD_S), Teardown(), Recreate(), Signal(tag),
                        Idle(S5_WARM_DELAY_S), Deploy("measure", READY_HOLD_S)))
    return [warm_sc, *(Scenario("S5", f"cold{n}", dc, gpu, m, (Await(tag), Deploy("cold", READY_HOLD_S)))
                       for n, m in enumerate(cold, 1))]
