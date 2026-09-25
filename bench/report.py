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
}
GIB = 1024 ** 3


@dataclass(frozen=True)
class Stat:
    n: int
    mean: float | None
    stdev: float | None
    lo: float | None
    hi: float | None

    def fmt(self, unit: str = "s", digits: int = 1) -> str:
        if not self.n or self.mean is None:
            return "—"
        spread = f" ± {self.stdev:.{digits}f}" if self.stdev else ""
        return f"{self.mean:.{digits}f}{spread}{unit}"


def stat(values) -> Stat:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return Stat(0, None, None, None, None)
    return Stat(len(xs), statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else None, min(xs), max(xs))


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
            "hydrate": stat(_acq(t, "hydrate_s") for t in cold),
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
    return sorted(ranked, key=lambda s: summary[s]["cold_e2e"].mean)


def volume_vs_volumecache(summary: dict) -> str | None:
    """The headline question: is loading in place off the volume slower than
    copying to container disk first?"""
    v, vc = summary.get("volume"), summary.get("volumecache")
    if not v or not vc or not v["storage"].n or not vc["storage"].n:
        return None
    a, b = v["storage"].mean, vc["storage"].mean
    faster, slower = ("volume", "volumecache") if a < b else ("volumecache", "volume")
    ratio = max(a, b) / min(a, b)
    e2e = ""
    if v["cold_e2e"].n and vc["cold_e2e"].n:
        e2e = (f" End to end (client-observed cold request) that is {v['cold_e2e'].mean:.1f}s vs "
               f"{vc['cold_e2e'].mean:.1f}s.")
    return (f"**{faster}** gets the model onto the GPU faster: storage time (acquire + load) "
            f"{a:.1f}s for volume vs {b:.1f}s for VolumeCache (hydrate {vc['hydrate'].fmt()}, "
            f"then load {vc['load'].fmt()} from container disk) — {slower} is {ratio:.2f}× slower.{e2e}")


def render(meta: dict, trials: list[dict]) -> str:
    strategies = meta["strategies"]
    summary = summarize(trials, strategies)
    env = environment(trials)
    size = next((summary[s]["bytes"] for s in strategies if summary[s]["bytes"]), None)
    lines = [f"# Model-loading strategies on Runpod Flash — run `{meta['run_id']}`", ""]
    lines += [f"- **Model:** `{meta['model']}`" + (f" ({size / GIB:.2f} GiB of weights/tokenizer)" if size else ""),
              f"- **GPU:** {env.get('gpu') or meta['gpu']} · **Datacenter:** {meta['datacenter']}",
              f"- **Trials:** {meta['cold_trials']} forced-cold per strategy, {meta['warm_per_cold']} warm after each; "
              f"strategies run {'concurrently' if meta['parallel'] else 'one at a time'}; "
              f"page cache evicted before storage reads: {'yes' if meta['evict_page_cache'] else 'no'}",
              f"- **Versions:** " + ", ".join(f"{k} {v}" for k, v in (env.get("versions") or {}).items() if v), ""]

    verdict = volume_vs_volumecache(summary)
    order = ranking(summary)
    lines += ["## Answer", ""]
    if verdict:
        lines += [verdict, ""]
    if order:
        lines += ["Cold-request ranking (client-observed, mean): " +
                  " < ".join(f"{s} {summary[s]['cold_e2e'].mean:.1f}s" for s in order), ""]

    lines += ["## Cold start (forced scale-to-zero, first request on a new worker)", "",
              "| strategy | cold request e2e | platform delay | handler exec | import | acquire | ↳ hydrate | ↳ Hub | load (tok+model) | **storage total** | first generate | n |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in strategies:
        x = summary[s]
        lines.append(f"| {s} | {x['cold_e2e'].fmt()} | {x['cold_delay'].fmt()} | {x['cold_exec'].fmt()} | "
                     f"{x['import'].fmt()} | {x['acquire'].fmt(digits=2)} | {x['hydrate'].fmt(digits=2)} | "
                     f"{x['hub'].fmt(digits=2)} | {x['load'].fmt(digits=2)} | **{x['storage'].fmt(digits=2)}** | "
                     f"{x['first_generate'].fmt(digits=2)} | {x['cold_n']}"
                     + (f" (+{x['cold_invalid']} invalid)" if x["cold_invalid"] else "") + " |")
    lines += ["", "- *cold request e2e*: job accepted → result as seen by the client (includes queueing, worker start, "
              "container/Python boot, the handler, and 0.5s polling granularity; excludes the benchmark's own "
              "un-pause window).",
              "- *platform delay*: Runpod `delayTime` — queue wait + worker start before the handler runs. "
              "Model Store's staging/placement shows up here, not in the handler.",
              "- *storage total* = acquire + tokenizer load + model load: the part of the cold start the strategy controls.",
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
        if x["cold_n"] and s in ("volumecache", "volume") and x["hub_online_cold"]:
            obs.append(f"- `{s}` needed the Hugging Face Hub on {x['hub_online_cold']}/{x['cold_n']} cold starts "
                       f"({x['hub'].fmt(digits=2)} each)."
                       + (" VolumeCache skips symlinks and `refs/`, so a hydrated HF cache has blobs but no "
                          "snapshot; `huggingface_hub` re-links it via Hub metadata calls (no weight bytes). "
                          "A Hub outage would therefore fall back to a full download." if s == "volumecache" else ""))
        if x["cold_invalid"]:
            obs.append(f"- `{s}`: {x['cold_invalid']} cold trial(s) invalid: {sorted(set(x['errors']))[:3]}")
        if len(x["workers"]) < x["cold_n"]:
            obs.append(f"- `{s}`: {x['cold_n']} cold trials ran on {len(x['workers'])} distinct worker id(s) — "
                       "FlashBoot restarted the same worker; its process was still fresh (verified per trial).")
    return obs or ["- Nothing unusual flagged automatically."]


def _caveats(meta: dict) -> list[str]:
    return ["## Caveats", "",
            "- Each cold trial lands wherever the scheduler puts it; host disk, NIC and volume-cluster load vary. "
            "Look at ± spread before reading small differences.",
            "- `posix_fadvise(DONTNEED)` evicts the kernel page cache for files read off the volume/Model Store, "
            "but cannot reach a FUSE client's own userspace cache on the host — a same-host restart may still read warm.",
            "- nocache downloads straight from the Hugging Face CDN; its numbers move with CDN/WAN conditions.",
            ("- Cold rounds ran strategies concurrently in waves; `volume` and `volumecache` were never in the "
             "same wave, so they did not contend for the network volume. (The prime phase runs all four at once.)"
             if meta["parallel"] else "- Strategies ran one at a time (PARALLEL=0)."),
            "- Model Store is attached with the console's GraphQL `modelReferences` field; Flash 1.20 has no "
            "parameter for it.", ""]


def write(run_dir: str) -> str:
    with open(os.path.join(run_dir, "meta.json")) as fh:
        meta = json.load(fh)
    trials = load_trials(os.path.join(run_dir, "results.jsonl"))
    md = render(meta, trials)
    path = os.path.join(run_dir, "report.md")
    with open(path, "w") as fh:
        fh.write(md)
    summary = summarize(trials, meta["strategies"])
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.__dict__)
    return path
