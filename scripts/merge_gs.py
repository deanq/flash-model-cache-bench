"""Merge a Global Store run into an earlier H100 run; the anchor strategy's rows
from the GS run are dropped but reported as a drift check in the header."""
import datetime as dt
import json
import os
import sys

from bench import report

ANCHOR = "volumecache"


def read(path):
    with open(path) as fh:
        return json.load(fh)


def rows(run_dir):
    with open(os.path.join(run_dir, "results.jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def medians(trials, strategy):
    s = report.summarize(trials, [strategy])[strategy]
    return s["storage"].median, s["cold_e2e"].median, s["cold_n"]


def main(runs_root, base_id, gs_id, out_id):
    base_dir, gs_dir = os.path.join(runs_root, base_id), os.path.join(runs_root, gs_id)
    base_meta, gs_meta = read(os.path.join(base_dir, "meta.json")), read(os.path.join(gs_dir, "meta.json"))
    base_rows, gs_rows = rows(base_dir), rows(gs_dir)
    for key in ("model", "gpu", "datacenter", "cold_trials", "warm_per_cold"):
        if base_meta[key] != gs_meta[key]:
            sys.exit(f"refusing to merge: {key} differs ({base_meta[key]!r} vs {gs_meta[key]!r})")

    then_storage, then_e2e, then_n = medians(base_rows, ANCHOR)
    now_storage, now_e2e, now_n = medians(gs_rows, ANCHOR)
    gs_strategies = [s for s in gs_meta["strategies"] if s != ANCHOR]
    day = lambda ts: dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()
    note = (f"{', '.join(f'`{s}`' for s in gs_strategies)} measured {day(gs_meta['started_at'])} in run `{gs_id}`; "
            f"the rest measured {day(base_meta['started_at'])}. Drift check: `{ANCHOR}` re-run alongside Global Store "
            f"had median storage {now_storage:.1f}s / cold e2e {now_e2e:.1f}s (n={now_n}) vs "
            f"{then_storage:.1f}s / {then_e2e:.1f}s (n={then_n}) here; its re-run rows are excluded from the tables.")
    if base_meta.get("combined_note"):
        note = base_meta["combined_note"] + ". " + note

    meta = dict(base_meta)
    meta.update(run_id=out_id, combined_from=(base_meta.get("combined_from") or [base_id]) + [gs_id],
                combined_note=note, strategies=base_meta["strategies"] + gs_strategies,
                endpoints={**base_meta["endpoints"], **{s: e for s, e in gs_meta["endpoints"].items() if s != ANCHOR}},
                global_store_id=gs_meta.get("global_store_id"),
                drift_check={"strategy": ANCHOR, "base": {"storage_s": then_storage, "cold_e2e_s": then_e2e, "n": then_n},
                             "global_store_run": {"storage_s": now_storage, "cold_e2e_s": now_e2e, "n": now_n}})
    out_dir = os.path.join(runs_root, out_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as fh:
        for r in base_rows + [r for r in gs_rows if r["strategy"] != ANCHOR]:
            fh.write(json.dumps(r) + "\n")
    print(report.write(out_dir))
    print(note)


if __name__ == "__main__":
    main(*sys.argv[1:5])
