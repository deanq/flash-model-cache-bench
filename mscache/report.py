"""Per-run report for DQ -> main Mac: per-scenario summary plus the per-load join table."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

JOIN_PHASES = ("submit", "pod_created", "pod_last_started", "container_start", "handler_ready", "job_start")


def _stat(xs: list[float]) -> str:
    return f"median {median(xs):.0f}s, worst {max(xs):.0f}s" if xs else "n/a"


def _scenario(key: tuple[str, str], recs: list[dict]) -> list[str]:
    loads = [r for r in recs if r["kind"] == "load"]
    measure = [r for r in loads if r["role"] == "measure"]
    ready = [r["offsets_s"]["job_start"] for r in measure if "job_start" in r["offsets_s"]]
    staging = [r["staging_s"] for r in measure if r["staging_s"] is not None]
    outcomes = Counter(r["outcome"] for r in measure)
    same = Counter(r["same_machine_as_prev"] for r in measure)
    taint = sum(r["kind"] == "pod_only" for r in recs)
    head = f"## {key[0]} {key[1]}".rstrip()
    ep = loads[0]["endpoint_id"] if loads else "?"
    model = loads[0]["model"] if loads else "?"
    return [f"{head}  (ep {ep}, model {model})",
            f"- measure loads: {len(measure)}; submit->job start: {_stat(ready)}; staging: {_stat(staging)}",
            "- outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())),
            f"- same machine as previous load: yes={same[True]} no={same[False]}; pod-only pods (taint): {taint}"]


def _join_rows(recs: list[dict]) -> list[str]:
    cols = ["scenario", "role", "endpoint", "pod", "machine", "model", "revision", "GB", "outcome", *JOIN_PHASES]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in (r for r in recs if r["kind"] == "load"):
        gb = f"{r['staged_bytes'] / 1e9:.1f}" if r.get("staged_bytes") else ""
        cells = [f"{r['scenario']} {r['arm']}".strip(), r["role"], r["endpoint_id"], r["pod_id"], r["machine_id"],
                 r["model"], r["revision"], gb, r["outcome"], *(r["phases_utc"].get(p) for p in JOIN_PHASES)]
        rows.append("| " + " | ".join(str(c or "") for c in cells) + " |")
    return rows


def render(recs: list[dict]) -> str:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        by[(r["scenario"], r["arm"])].append(r)
    out = ["# MSCACHE run report", "",
           "Outcomes are in-container estimates from staging time; the host-log join is final.", ""]
    for key, rs in by.items():
        out += _scenario(key, rs) + [""]
    return "\n".join(out + ["## Per-load join table (UTC)", ""] + _join_rows(recs)) + "\n"
