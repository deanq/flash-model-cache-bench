"""Run plans. Every scenario in a plan gets its own model so parallel arms never hold each other's copy.

Models are public, ungated and rarely deployed, so a seed load is likely a true first pull in the DC;
the seed doubles as an S1 sample (host-log join decides).

A plan is lanes -> batches -> scenarios: lanes run in parallel, a lane's batches run one after another,
and a batch's scenarios run in parallel (S4/S5 arms that must overlap). The 11-22 pilot showed parallel
scenarios co-land on one machine and share its 4 download slots, so the matrix serializes everything
except the 60-min S2 arm, whose idle gaps would otherwise set the wall time.
"""
from __future__ import annotations

from mscache import cost, spec

# repo -> size in GB at the main revision (HF API, all files, 2026-11-21/22)
SIZE_GB = {"allenai/OLMo-2-1124-13B-Instruct": 27.4,
           "tiiuae/Falcon3-10B-Instruct": 20.6,
           "ibm-granite/granite-3.3-8b-instruct": 16.3,
           "Qwen/Qwen1.5-32B-Chat": 65.0,
           "tiiuae/Falcon3-7B-Instruct": 14.9,
           "allenai/OLMo-2-1124-7B-Instruct": 14.6,
           "ibm-granite/granite-3.1-8b-instruct": 16.3,
           "tiiuae/Falcon3-3B-Instruct": 6.5,
           "allenai/OLMo-2-0425-1B-Instruct": 3.0,
           "ibm-granite/granite-3.1-2b-instruct": 5.1,
           "Qwen/Qwen2.5-14B": 29.6,
           "Qwen/Qwen2-72B": 145.4,
           "NousResearch/Hermes-3-Llama-3.1-70B": 141.1,
           "Qwen/Qwen2-Math-72B-Instruct": 145.4,
           "allenai/Llama-3.1-Tulu-3-70B-SFT": 141.1,
           "01-ai/Yi-1.5-34B-Chat": 68.8,
           "Qwen/Qwen1.5-72B-Chat": 144.6,
           "abacusai/Smaug-72B-v0.1": 144.6,
           "Qwen/Qwen2.5-72B": 145.4,
           "Qwen/Qwen2.5-72B-Instruct": 145.4,
           "Qwen/Qwen2-72B-Instruct": 145.4,
           "Qwen/Qwen2.5-Math-72B-Instruct": 145.4,
           "allenai/OLMo-2-0325-32B-Instruct": 64.5,
           "meta-llama/Llama-3.1-8B": 32.1}
FAILURE_SIZE_GB = 1.0      # S8 loads fail before (or instead of) a pull
PILOT_S2 = (("allenai/OLMo-2-1124-13B-Instruct", 60), ("tiiuae/Falcon3-10B-Instruct", 600))
PILOT_S9 = "ibm-granite/granite-3.3-8b-instruct"

# full matrix, the same models in every DC (each DC's cache is its own); none was used in the pilot
M = {"S1": "Qwen/Qwen1.5-32B-Chat",
     "S2": {60: "tiiuae/Falcon3-7B-Instruct", 600: "allenai/OLMo-2-1124-7B-Instruct",
            3600: "ibm-granite/granite-3.1-8b-instruct"},
     "S9": "tiiuae/Falcon3-3B-Instruct",
     "S10": "allenai/OLMo-2-0425-1B-Instruct",
     "S7": {"small": "ibm-granite/granite-3.1-2b-instruct", "medium": "Qwen/Qwen2.5-14B",
            "large": "Qwen/Qwen2-72B"},
     "S3": {10: "NousResearch/Hermes-3-Llama-3.1-70B", 30: "Qwen/Qwen2-Math-72B-Instruct",
            60: "allenai/Llama-3.1-Tulu-3-70B-SFT"},
     "S4": "01-ai/Yi-1.5-34B-Chat",
     # 11-25: the 11-22/11-25 cold set (Qwen1.5-72B-Chat, Smaug-72B, Qwen2.5-72B) is now cached by LRU on
     # MTL and MO-1 machines, so a rerun there would not be cold; this set is new to every DC
     "S5_cold": ("Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2-72B-Instruct", "Qwen/Qwen2.5-Math-72B-Instruct"),
     "S5_warm": "allenai/OLMo-2-0325-32B-Instruct",
     "S8_gated": "meta-llama/Llama-3.1-8B",
     "S8_badrev": "allenai/OLMo-2-0425-1B-Instruct:" + "0" * 40}
LRU_DCS = ("CA-MTL-1", "EU-SE-1", "US-NC-1", "US-TX-3", "US-MO-1")
PEER_LIVE = frozenset({"US-TX-3", "CA-MTL-1", "US-MO-1"})   # S4/S5 run only here
CONTROL_DCS = ("EUR-IS-2", "US-NE-1")                       # LRU off; main Mac's pick, fallback

Batch = list[spec.Scenario]
Lane = list[Batch]


def size_gb(sc: spec.Scenario) -> float:
    return FAILURE_SIZE_GB if sc.sid == "S8" else SIZE_GB[sc.model]


def pilot(dc: str, gpu: str) -> list[spec.Scenario]:
    arms = [spec.s2(dc=dc, gpu=gpu, model=m, gap_s=g) for m, g in PILOT_S2]
    return arms + [spec.s9(dc=dc, gpu=gpu, model=PILOT_S9)]


def total(scs: list[spec.Scenario]) -> dict:
    """Scenarios run in parallel on separate endpoints: cost adds up, wall time is the longest arm."""
    ests = [cost.estimate(s, size_gb=size_gb(s)) for s in scs]
    return {"loads": sum(e.loads for e in ests),
            "likely_usd": sum(e.likely_usd for e in ests),
            "worst_usd": sum(e.worst_usd for e in ests),
            "wall_s": max(e.wall_s for e in ests)}


def _core(dc: str, gpu: str) -> Lane:
    """One scenario at a time: S2 1/10 min, S9, S10, S1, S7 x3 sizes, S3 x3 cut points."""
    kw = {"dc": dc, "gpu": gpu}
    scs = [spec.s2(**kw, model=M["S2"][60], gap_s=60), spec.s2(**kw, model=M["S2"][600], gap_s=600),
           spec.s9(**kw, model=M["S9"]), spec.s10(**kw, model=M["S10"]), spec.s1(**kw, model=M["S1"]),
           *(spec.s7(**kw, model=m, size=size) for size, m in M["S7"].items()),
           *(spec.s3(**kw, model=m, after_s=t) for t, m in M["S3"].items())]
    return [[s] for s in scs]


def _contention(dc: str, gpu: str) -> Lane:
    """S4 (holder + 3 peers) then S5+ (warm redeploy during 3 cold pulls); arms of each overlap."""
    kw = {"dc": dc, "gpu": gpu}
    return [spec.s4(**kw, model=M["S4"]), spec.s5(**kw, cold=list(M["S5_cold"]), warm=M["S5_warm"])]


def matrix(dc: str, gpu: str, *, window: int) -> list[Lane]:
    """Window 1: S2 60-min arm beside the serial core. Window 2: S4/S5 (peer-live DCs) and S8 (CA-MTL-1)."""
    if window == 1:
        return [[[spec.s2(dc=dc, gpu=gpu, model=M["S2"][3600], gap_s=3600)]], _core(dc, gpu)]
    lanes: list[Lane] = [_contention(dc, gpu)] if dc in PEER_LIVE else []
    if dc == "CA-MTL-1":   # once
        lanes.append([[spec.s8(dc=dc, gpu=gpu, model=M["S8_gated"], arm="gated")],
                      [spec.s8(dc=dc, gpu=gpu, model=M["S8_badrev"], arm="badrev")]])
    return lanes


def control(dc: str, gpu: str) -> list[Lane]:
    """LRU-off comparison for S2 and S9, same models and gaps as the LRU DCs (1 and 10 min)."""
    kw = {"dc": dc, "gpu": gpu}
    return [[[spec.s2(**kw, model=M["S2"][60], gap_s=60)], [spec.s2(**kw, model=M["S2"][600], gap_s=600)],
             [spec.s9(**kw, model=M["S9"])]]]


def lanes_total(lanes: list[Lane]) -> dict:
    """Cost adds up; wall = longest lane, a lane = its batches back to back, a batch = its longest arm."""
    scs = [s for lane in lanes for batch in lane for s in batch]
    if not scs:
        return {"loads": 0, "likely_usd": 0.0, "worst_usd": 0.0, "wall_s": 0.0}
    walls = [sum(total(batch)["wall_s"] for batch in lane) for lane in lanes]
    return total(scs) | {"wall_s": max(walls)}
