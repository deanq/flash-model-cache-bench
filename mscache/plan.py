"""Run plans. Every scenario in a plan gets its own model so parallel arms never hold each other's copy.

Models are public, ungated and rarely deployed (HF downloads under ~100k/month on 2026-11-21), so a seed
load is likely a true first pull in the DC; the seed doubles as an S1 sample (host-log join decides).
"""
from __future__ import annotations

from mscache import cost, spec

# repo -> size in GB at the main revision (HF API, 2026-11-21)
SIZE_GB = {"allenai/OLMo-2-1124-13B-Instruct": 27.4,
           "tiiuae/Falcon3-10B-Instruct": 20.6,
           "ibm-granite/granite-3.3-8b-instruct": 16.3}
PILOT_S2 = (("allenai/OLMo-2-1124-13B-Instruct", 60), ("tiiuae/Falcon3-10B-Instruct", 600))
PILOT_S9 = "ibm-granite/granite-3.3-8b-instruct"


def pilot(dc: str, gpu: str) -> list[spec.Scenario]:
    arms = [spec.s2(dc=dc, gpu=gpu, model=m, gap_s=g) for m, g in PILOT_S2]
    return arms + [spec.s9(dc=dc, gpu=gpu, model=PILOT_S9)]


def total(scs: list[spec.Scenario]) -> dict:
    """Scenarios run in parallel on separate endpoints: cost adds up, wall time is the longest arm."""
    ests = [cost.estimate(s, size_gb=SIZE_GB[s.model]) for s in scs]
    return {"loads": sum(e.loads for e in ests),
            "likely_usd": sum(e.likely_usd for e in ests),
            "worst_usd": sum(e.worst_usd for e in ests),
            "wall_s": max(e.wall_s for e in ests)}
