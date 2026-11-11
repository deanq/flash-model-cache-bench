"""Turn runs/<id>/results.jsonl into runs/<id>/report.md (+ summary.json).

Pure functions over trial dicts so the math is unit-testable without Runpod.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass

LABELS = {
    "nocache": "no cache (HF download every cold start)",
    "volume": "network volume, loaded in place",
    "volumecache": "VolumeCache (volume -> container disk)",
    "modelstore": "Runpod Model Store (cached model)",
    "globalstore": "Runpod Global Store (global volume), loaded in place",
    "gsvolumecache": "VolumeCache (Global Store -> container disk)",
    "gshydrate": "parallel range hydrate (Global Store -> container disk)",
    "volumeprefetch": "network volume, parallel page-cache prefetch, loaded in place",
}
# (in place, VolumeCache, storage name): the question "is loading straight off
# shared storage slower than copying it to container disk first?" per backend.
IN_PLACE_VS_CACHED = (("volume", "volumecache", "network volume"),
                      ("volumeprefetch", "volumecache", "network volume (prefetched)"),
                      ("globalstore", "gsvolumecache", "Global Store"),
                      ("globalstore", "gshydrate", "Global Store (parallel hydrate)"))
VOLUMECACHE_STRATEGIES = ("volumecache", "gsvolumecache")
# Strategies whose model load reads shared storage directly (no up-front copy/prefetch).
LOAD_READS_SHARED_STORAGE = ("volume", "globalstore", "modelstore")
MIB = 1024 ** 2
GIB = 1024 ** 3
DELAY_OUTLIER_FACTOR = 2.0


@dataclass(frozen=True)
class Stat:
    """Median-first: cold starts have scheduler outliers (a 118 s placement wait
    next to 10 s ones) that drag a mean far from the typical case."""
    n: int
    median: float | None
    mean: float | None
    lo: float | None
    hi: float | None

    def fmt(self, unit: str = "s", digits: int = 1) -> str:
        if not self.n or self.median is None:
            return "—"
        if self.n == 1 or self.lo == self.hi:
            return f"{self.median:.{digits}f}{unit}"
        return f"{self.median:.{digits}f}{unit} [{self.lo:.{digits}f}–{self.hi:.{digits}f}]"


def stat(values) -> Stat:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return Stat(0, None, None, None, None)
    return Stat(len(xs), statistics.median(xs), statistics.fmean(xs), min(xs), max(xs))


def load_trials(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _cold(t: dict) -> dict:
    return (t.get("output") or {}).get("cold") or {}


def _gen(t: dict) -> dict:
    return (t.get("output") or {}).get("generate") or {}


def _acq(t: dict, key: str) -> float | None:
    return (_cold(t).get("acquire") or {}).get(key)


def _hub_s(t: dict) -> float | None:
    """Time spent talking to the Hub (metadata relink and/or byte download); None if it never did."""
    parts = [x for x in (_acq(t, "hub_s"), _acq(t, "download_s")) if x is not None]
    return sum(parts) if parts else None


def _upfront_read_s(t: dict) -> float | None:
    """Time spent reading shared storage before load: VolumeCache/gshydrate hydrate or prefetch."""
    return _acq(t, "hydrate_s") if _acq(t, "hydrate_s") is not None else _acq(t, "prefetch_s")


def read_mibps(t: dict) -> float | None:
    """Throughput off shared storage, from the step that reads it: the model load for
    in-place strategies, the hydrate/prefetch otherwise. None for Hub downloads."""
    nbytes = _cold(t).get("bytes")
    secs = _upfront_read_s(t)
    if secs is None and t["strategy"] in LOAD_READS_SHARED_STORAGE:
        secs = _cold(t).get("model_load_s")
    if not nbytes or not secs:
        return None
    return nbytes / MIB / secs


def storage_s(t: dict) -> float | None:
    """Time the storage strategy is responsible for: getting bytes local + loading them."""
    c = _cold(t)
    if c.get("acquire_s") is None or c.get("model_load_s") is None:
        return None
    return c["acquire_s"] + c.get("tokenizer_load_s", 0.0) + c["model_load_s"]


def summarize(trials: list[dict], strategies: list[str]) -> dict:
    out = {}
    for s in strategies:
        mine = [t for t in trials if t["strategy"] == s]
        cold = [t for t in mine if t["phase"] == "cold" and t["valid"]]
        warm = [t for t in mine if t["phase"] == "warm" and t["valid"]]
        prime = next((t for t in mine if t["phase"] == "prime"), None)
        out[s] = {
            "prime": _prime_summary(prime),
            "cold_n": len(cold),
            "cold_invalid": sum(1 for t in mine if t["phase"] == "cold" and not t["valid"]),
            "cold_e2e": stat(t["e2e_s"] for t in cold),
            "cold_delay": stat((t["delay_ms"] or 0) / 1000 for t in cold),
            "cold_exec": stat((t["exec_ms"] or 0) / 1000 for t in cold),
            "import": stat(_cold(t).get("import_s") for t in cold),
            "acquire": stat(_cold(t).get("acquire_s") for t in cold),
            "hydrate": stat(_upfront_read_s(t) for t in cold),
            "read_mibps": stat(read_mibps(t) for t in cold),
            "hub": stat(_hub_s(t) for t in cold),
            "load": stat((_cold(t).get("tokenizer_load_s") or 0) + (_cold(t).get("model_load_s") or 0)
                         for t in cold),
            "storage": stat(storage_s(t) for t in cold),
            "first_generate": stat(_gen(t).get("generate_s") for t in cold),
            "hub_online_cold": sum(1 for t in cold if _cold(t).get("hub_online")),
            "warm_e2e": stat(t["e2e_s"] for t in warm),
            "warm_tps": stat(_gen(t).get("tokens_per_s") for t in warm),
            "bytes": next((_cold(t).get("bytes") for t in cold + ([prime] if prime else []) if _cold(t).get("bytes")), None),
            "workers": sorted({(t.get("output") or {}).get("worker_id") for t in cold} - {None}),
            "errors": [t.get("error") for t in mine if not t["valid"] and t.get("error")],
        }
    return out


def _prime_summary(t: dict | None) -> dict:
    if t is None:
        return {"ok": False, "error": "not run"}
    return {"ok": t["valid"], "e2e_s": t["e2e_s"], "delay_s": (t["delay_ms"] or 0) / 1000,
            "acquire_s": _cold(t).get("acquire_s"), "acquire": _cold(t).get("acquire"),
            "error": t.get("error")}


def environment(trials: list[dict]) -> dict:
    first = next((t["output"] for t in trials if (t.get("output") or {}).get("versions")), {})
    return {"gpu": first.get("gpu"), "versions": first.get("versions"), "dc": first.get("dc")}


def ranking(summary: dict) -> list[str]:
    ranked = [s for s in summary if summary[s]["cold_e2e"].n]
    return sorted(ranked, key=lambda s: summary[s]["cold_e2e"].median)


def in_place_vs_cached(summary: dict, in_place: str, cached: str, storage: str) -> str | None:
    """Is loading straight off `storage` slower than VolumeCache-copying it to container disk first?"""
    v, vc = summary.get(in_place), summary.get(cached)
    if not v or not vc or not v["storage"].n or not vc["storage"].n:
        return None
    a, b = v["storage"].median, vc["storage"].median
    faster, slower = (in_place, cached) if a < b else (cached, in_place)
    ratio = max(a, b) / min(a, b)
    e2e = ""
    if v["cold_e2e"].n and vc["cold_e2e"].n:
        e2e = (f" Client-observed cold request: {in_place} {v['cold_e2e'].median:.1f}s vs {cached} "
               f"{vc['cold_e2e'].median:.1f}s (medians; includes scheduler placement, which the strategy does "
               f"not control).")
    return (f"**{storage}: {faster}** gets the model onto the GPU faster. Median storage time (acquire + load): "
            f"`{in_place}` {a:.1f}s (loading straight off the mount: {v['load'].fmt()}) vs `{cached}` {b:.1f}s "
            f"(hydrate {vc['hydrate'].fmt()} + load from container disk {vc['load'].fmt()}) — "
            f"{slower} is {ratio:.2f}× slower.{e2e}")


def verdicts(summary: dict) -> list[str]:
    return [v for v in (in_place_vs_cached(summary, *pair) for pair in IN_PLACE_VS_CACHED) if v]


def render(meta: dict, trials: list[dict]) -> str:
    strategies = meta["strategies"]
    summary = summarize(trials, strategies)
    env = environment(trials)
    size = next((summary[s]["bytes"] for s in strategies if summary[s]["bytes"]), None)
    lines = [f"# Model-loading strategies on Runpod Flash — run `{meta['run_id']}`", ""]
    lines += [f"- **Model:** `{meta['model']}`" + (f" ({size / GIB:.2f} GiB of weights/tokenizer)" if size else ""),
              f"- **GPU:** {env.get('gpu') or meta['gpu']} · **Datacenter:** {meta['datacenter']}",
              f"- **Trials:** {meta['cold_trials']} forced-cold per strategy, {meta['warm_per_cold']} warm after each; "
              f"strategies run {'concurrently in waves' if meta['parallel'] else 'one at a time'}; "
              f"page cache evicted before storage reads: {'yes' if meta['evict_page_cache'] else 'no'}",
              f"- **Versions:** " + ", ".join(f"{k} {v}" for k, v in (env.get("versions") or {}).items() if v), ""]

    order = ranking(summary)
    lines += ["## Answer", ""]
    for verdict in verdicts(summary):
        lines += [verdict, ""]
    if order:
        lines += ["Cold-request ranking (client-observed, median): " +
                  " < ".join(f"{s} {summary[s]['cold_e2e'].median:.1f}s" for s in order), ""]

    lines += ["## Cold start (forced scale-to-zero, first request on a new worker)", "",
              "| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate/prefetch | ↳ Hub | load (tok+model) | **storage total** | shared-storage read | first generate | n |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in strategies:
        x = summary[s]
        lines.append(f"| {s} | {x['cold_e2e'].fmt()} | {x['cold_delay'].fmt()} | {x['cold_exec'].fmt()} | "
                     f"{x['import'].fmt()} | {x['acquire'].fmt(digits=2)} | {x['hydrate'].fmt(digits=2)} | "
                     f"{x['hub'].fmt(digits=2)} | {x['load'].fmt(digits=2)} | **{x['storage'].fmt(digits=2)}** | "
                     f"{x['read_mibps'].fmt(unit=' MiB/s', digits=0)} | {x['first_generate'].fmt(digits=2)} | {x['cold_n']}"
                     + (f" (+{x['cold_invalid']} invalid)" if x["cold_invalid"] else "") + " |")
    lines += ["", "Cells are **median [min–max]** over valid cold trials.", "",
              "- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, "
              "container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own "
              "un-pause window).",
              "- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. "
              "Model Store's staging/placement shows up here, not in the handler.",
              "- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.",
              "- *shared-storage read*: model bytes ÷ the step that reads the volume/store — the model load for "
              "in-place strategies (`volume`, `globalstore`, `modelstore`), the hydrate or prefetch otherwise.",
              ""]

    lines += ["## Warm requests (model already in GPU memory)", "",
              "| strategy | warm request e2e | tokens/s | n |", "|---|---|---|---|"]
    for s in strategies:
        x = summary[s]
        lines.append(f"| {s} | {x['warm_e2e'].fmt()} | {x['warm_tps'].fmt(unit='', digits=1)} | {x['warm_e2e'].n} |")
    lines += ["", "Warm throughput should be identical across strategies — once weights are in VRAM, storage is out "
              "of the picture. A gap here points at GPU/host variance, not the caching strategy.", ""]

    lines += ["## First-ever request (cache population)", "",
              "| strategy | e2e | platform delay | acquire | detail |", "|---|---|---|---|---|"]
    for s in strategies:
        p = summary[s]["prime"]
        if not p["ok"]:
            lines.append(f"| {s} | failed | | | {p.get('error')} |")
            continue
        detail = ", ".join(f"{k} {v:.1f}s" for k, v in (p.get("acquire") or {}).items())
        acq = f"{p['acquire_s']:.1f}s" if p.get("acquire_s") is not None else "—"
        lines.append(f"| {s} | {p['e2e_s']:.1f}s | {p['delay_s']:.1f}s | {acq} | {detail} |")
    lines += ["", "One-time per endpoint (nocache pays it on *every* cold start — compare with its cold row).", ""]

    lines += ["## Observations", ""] + _observations(summary, strategies) + [""]
    lines += ["## Strategies", ""] + [f"- **{s}** — {LABELS[s]}" for s in strategies] + [""]
    lines += _caveats(meta)
    return "\n".join(lines) + "\n"


def _observations(summary: dict, strategies: list[str]) -> list[str]:
    obs = []
    for s in strategies:
        x = summary[s]
        if x["cold_n"] and s in ("volume", "globalstore", *VOLUMECACHE_STRATEGIES) and x["hub_online_cold"]:
            obs.append(f"- `{s}` needed the Hugging Face Hub on {x['hub_online_cold']}/{x['cold_n']} cold starts "
                       f"({x['hub'].fmt(digits=2)} each)."
                       + (" VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no "
                          "snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). "
                          "A Hub outage would therefore fall back to a full download." if s in VOLUMECACHE_STRATEGIES else ""))
        if x["cold_invalid"]:
            obs.append(f"- `{s}`: {x['cold_invalid']} cold trial(s) invalid: {sorted(set(x['errors']))[:3]}")
        if len(x["workers"]) < x["cold_n"]:
            obs.append(f"- `{s}`: {x['cold_n']} cold trials ran on {len(x['workers'])} distinct worker id(s) — "
                       "FlashBoot restarted the same worker; its process was still fresh (verified per trial).")
    obs += _delay_outliers(summary, strategies)
    return obs or ["- Nothing unusual flagged automatically."]


def _delay_outliers(summary: dict, strategies: list[str]) -> list[str]:
    """Flag a strategy whose platform delay (scheduling + worker start) is far above the rest."""
    delays = {s: summary[s]["cold_delay"].median for s in strategies if summary[s]["cold_delay"].n}
    out = []
    for s, d in delays.items():
        others = [x for k, x in delays.items() if k != s]
        if others and d > DELAY_OUTLIER_FACTOR * statistics.median(others):
            out.append(f"- `{s}` waited {summary[s]['cold_delay'].fmt()} for a worker on cold starts vs a median of "
                       f"{statistics.median(others):.1f}s for the other strategies. That platform delay, not "
                       f"loading, dominates its cold start"
                       + (" — consistent with the documented behaviour of holding workers until a host that has "
                          "the cached model is available (the delay is not billed as worker time)."
                          if s == "modelstore" else "."))
    return out


def _caveats(meta: dict) -> list[str]:
    return ["## Caveats", "",
            "- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. "
            "Look at the [min–max] range before reading small differences.",
            "- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, "
            "but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.",
            "- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.",
            "- `GPU` names a class, not a machine: hosts in the same class differ (see the warm tokens/s range), "
            "which shows up in import, load and generate times alike.",
            ("- Cold rounds ran strategies concurrently in waves; strategies sharing storage (`volume`/`volumecache`/`volumeprefetch`, "
             "`globalstore`/`gsvolumecache`/`gshydrate`) were never in the same wave. (The prime phase runs all at once.)"
             if meta["parallel"] else "- Strategies ran one at a time (PARALLEL=0)."),
            "- Model Store is declared with `model=runpod.Model(...)` on the endpoint (sets `modelReferences`).",
            *(_GLOBAL_STORE_CAVEATS if any(s in meta["strategies"] for s in ("globalstore", "gsvolumecache", "gshydrate")) else []),
            ""]


_GLOBAL_STORE_CAVEATS = [
    "- Global Store (beta) is declared as a `runpod.GlobalVolume` mount. Its endpoints are pinned to the same "
    "datacenter as the rest for comparability, although a global volume would let them run anywhere.",
    "- Global Store is object storage behind a FUSE client (geesefs), which keeps its own memory cache that "
    "`posix_fadvise` cannot evict.",
    "- `globalstore` keeps a plain directory of real files, not an HF cache: HF snapshot symlinks written by one "
    "worker come back as 0-byte regular files on the next worker's mount.",
    "- `gsvolumecache` uses stock VolumeCache (runpod>=2.1.0 copies with copyfile; older releases' copy2 failed "
    "with EPERM on chmod/utime there and silently left every large file out of the mirror).",
]


def usage(meta: dict, trials: list[dict], billing: list[dict] | None) -> dict:
    """Wall clock, handler time, and (once fetched with `bench billing`) Runpod-billed
    GPU time and cost. Billing rows are filtered to this run's endpoint ids, so other
    workloads in the account never leak in."""
    started, finished = meta.get("started_at"), meta.get("finished_at")
    out = {"wall_clock_min": (finished - started) / 60 if started and finished else None,
           "handler_gpu_min": sum((t.get("exec_ms") or 0) for t in trials) / 60000,
           "billed_gpu_min": None, "billed_usd": None, "billed_gpu_min_by_strategy": None}
    if billing is None:
        return out
    by_id = {eid: s for s, eid in (meta.get("endpoints") or {}).items()}
    mine = [row for row in billing if row.get("endpointId") in by_id]
    if not mine:                       # billing lags usage by hours: unbilled so far, not free
        return out
    per: dict[str, float] = {}
    for row in mine:
        s = by_id[row["endpointId"]]
        per[s] = per.get(s, 0.0) + row.get("timeBilledMs", 0) / 60000
    out.update(billed_gpu_min=sum(per.values()), billed_usd=sum(row.get("amount", 0.0) for row in mine),
               billed_gpu_min_by_strategy=per)
    return out


def _usage_lines(u: dict) -> list[str]:
    def num(v, fmt):
        return "—" if v is None else format(v, fmt)
    billed = (f"billed GPU time: {num(u['billed_gpu_min'], '.1f')} GPU-min, ${num(u['billed_usd'], '.2f')} "
              f"(Runpod billing API, this run's endpoints only)" if u["billed_gpu_min"] is not None
              else "billed GPU time: not in the billing API yet (`make billing RUN=<id>` later; billing can lag usage by several hours)")
    lines = ["## Time and cost", "",
             f"- wall clock (deploy → teardown): {num(u['wall_clock_min'], '.1f')} min",
             f"- handler time (sum of `executionTime`): {u['handler_gpu_min']:.1f} GPU-min",
             f"- {billed}"]
    if u["billed_gpu_min_by_strategy"]:
        lines.append("- billed by strategy: " + ", ".join(
            f"{s} {m:.1f}" for s, m in sorted(u["billed_gpu_min_by_strategy"].items())))
    return lines + [""]


def _read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def write(run_dir: str) -> str:
    meta = _read_json(os.path.join(run_dir, "meta.json"))
    trials = load_trials(os.path.join(run_dir, "results.jsonl"))
    run_usage = usage(meta, trials, _read_json(os.path.join(run_dir, "billing.json")))
    md = render(meta, trials) + "\n".join(_usage_lines(run_usage))
    path = os.path.join(run_dir, "report.md")
    with open(path, "w") as fh:
        fh.write(md)
    summary = {"usage": run_usage, "strategies": summarize(trials, meta["strategies"])}
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.__dict__)
    return path
