"""Cost and wall-clock estimate per scenario, from billed worker-seconds at the pool's serverless price.

Billing on 11-21 came in at about half of the live estimates, so `likely` is the planning number and
`worst` (cold pull on every load, slow origin, ai-api wait on every load) is the cap to state up front.
"""
from __future__ import annotations

from dataclasses import dataclass

from mscache import spec

CACHED_STAGING_S = 45       # ai-api wait lands on most scale-ups
HANDLER_START_S = 20        # pip install runpod + import in python:3.12-slim
ORIGIN_GBPS = 0.4           # slowest observed pulls (CA-MTL-1 peer p50 was ~0.04 GB/s; origin ~0.2-0.5)
WORST_GBPS = 0.05
TEARDOWN_WALL_S = 40        # workersMax 0 until no live pod
INTERRUPT_PLACE_S = 10      # S3: pod placement before the after_s clock starts


@dataclass(frozen=True)
class Estimate:
    loads: int
    likely_worker_s: float
    worst_worker_s: float
    likely_usd: float
    worst_usd: float
    wall_s: float


def _per_load_s(d: spec.Deploy, staging_s: float) -> float:
    return staging_s + HANDLER_START_S + d.hold_s + spec.IDLE_TIMEOUT_S


def estimate(sc: spec.Scenario, *, size_gb: float) -> Estimate:
    usd_s = spec.GPUS[sc.gpu].usd_per_h / 3600
    cold_s = size_gb / ORIGIN_GBPS
    worst_cold_s = size_gb / WORST_GBPS
    likely = worst = 0.0
    for i, d in enumerate(sc.deploys):
        likely += _per_load_s(d, cold_s if i == 0 else CACHED_STAGING_S)
        worst += _per_load_s(d, worst_cold_s)
    # S3: billed from placement to the cut; the redeploy after it is costed as a cold load above
    cut = sum(s.after_s + INTERRUPT_PLACE_S for s in sc.steps if isinstance(s, spec.Interrupt))
    likely, worst = likely + cut, worst + cut
    gaps = sum(s.seconds for s in sc.steps if isinstance(s, spec.Idle)) + cut
    teardowns = sum(isinstance(s, (spec.Teardown, spec.AwaitScaleDown)) for s in sc.steps) * TEARDOWN_WALL_S
    return Estimate(loads=len(sc.deploys), likely_worker_s=likely, worst_worker_s=worst,
                    likely_usd=likely * usd_s, worst_usd=worst * usd_s, wall_s=likely + gaps + teardowns)
