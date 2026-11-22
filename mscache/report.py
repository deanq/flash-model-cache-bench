"""Per-run report for DQ -> main Mac: per-scenario summary plus the per-load join table."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

JOIN_PHASES = ("submit", "pod_created", "pod_last_started", "container_start", "handler_ready", "job_start")
ANCHOR_NOTE = ("staging_s = container start minus pod_created (new pod) or minus submit (restarted pod), "
               "so placement wait is not in it; submit_to_container_s includes placement and the ai-api wait.")


def _stat(xs: list[float]) -> str:
    return f"median {median(xs):.0f}s, worst {max(xs):.0f}s" if xs else "n/a"


def _staging_from(r: dict) -> str:
    # records before the staging_from field: a restarted pod (pod_reused) was anchored on submit
    return r.get("staging_from") or ("submit" if r.get("pod_reused") else "pod_created")


def _submit_to_container(r: dict) -> float | None:
    return r.get("submit_to_container_s", r["offsets_s"].get("container_start"))


def _scenario(key: tuple[str, str], recs: list[dict]) -> list[str]:
    loads = [r for r in recs if r["kind"] == "load"]
    measure = [r for r in loads if r["role"] == "measure"]
    ready = [r["offsets_s"]["job_start"] for r in measure if "job_start" in r["offsets_s"]]
    # a cached load that waited out ai-api's scale-from-zero floor measures ai-api, not the copy
    floor = [r for r in measure if r.get("floor_wait")]
    staging = [r["staging_s"] for r in measure if r["staging_s"] is not None and not r.get("floor_wait")]
    to_container = [x for x in map(_submit_to_container, measure) if x is not None]
    outcomes = Counter(r["outcome"] for r in measure)
    same = Counter(r["same_machine_as_prev"] for r in measure)
    resumed = Counter(bool(r.get("resumed")) for r in measure)
    taint = sum(r["kind"] == "pod_only" for r in recs)
    head = f"## {key[0]} {key[1]}".rstrip()
    eps = ", ".join(dict.fromkeys(r["endpoint_id"] for r in loads)) or "?"
    model = loads[0]["model"] if loads else "?"
    return [f"{head}  (ep {eps}, model {model})",
            f"- measure loads: {len(measure)}; submit->container start: {_stat(to_container)}; "
            f"submit->job start: {_stat(ready)}",
            f"- staging_s (excl. ai-api floor): {_stat(staging)}; cached behind ai-api floor: {len(floor)}",
            "- outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())),
            f"- worker: new pod={resumed[False]} resumed (same pod id)={resumed[True]}",
            f"- same machine as previous load: yes={same[True]} no={same[False]}; pod-only pods (taint): {taint}"]


def _join_rows(recs: list[dict]) -> list[str]:
    cols = ["scenario", "role", "endpoint", "pod", "worker", "machine", "model", "revision", "GB", "outcome",
            "staging_s", "submit_to_container_s", *JOIN_PHASES]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in (r for r in recs if r["kind"] == "load"):
        gb = f"{r['staged_bytes'] / 1e9:.1f}" if r.get("staged_bytes") else ""
        worker = "resumed" if r.get("resumed") else "new"
        outcome = f"{r['outcome']} (ai-api floor)" if r.get("floor_wait") else r["outcome"]
        staging = f"{r['staging_s']} (from {_staging_from(r)})" if r.get("staging_s") is not None else ""
        cells = [f"{r['scenario']} {r['arm']}".strip(), r["role"], r["endpoint_id"], r["pod_id"], worker,
                 r["machine_id"], r["model"], r["revision"], gb, outcome, staging, _submit_to_container(r),
                 *(r["phases_utc"].get(p) for p in JOIN_PHASES)]
        rows.append("| " + " | ".join(str(c or "") for c in cells) + " |")
    return rows


def render(recs: list[dict]) -> str:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        by[(r["scenario"], r["arm"])].append(r)
    out = ["# MSCACHE run report", "",
           "Outcomes are in-container estimates from staging time; the host-log join is final.", ANCHOR_NOTE, ""]
    for key, rs in by.items():
        out += _scenario(key, rs) + [""]
    return "\n".join(out + ["## Per-load join table (UTC)", ""] + _join_rows(recs)) + "\n"
