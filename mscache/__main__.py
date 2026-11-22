"""MSCACHE CLI.

  python -m mscache plan  --dc CA-MTL-1 --gpu a40 [--window 1|2 | --control]   cost + wall estimate, $0
  python -m mscache smoke --dc CA-MTL-1 --gpu a40 [--window 1|2 | --control]   create+delete each endpoint, $0
  python -m mscache run   --dc CA-MTL-1 --gpu a40 [--window 1|2 | --control] --spend   PAID
  python -m mscache report runs/mscache/<run_id>            re-render a report
  python -m mscache leftovers                               list msc-* endpoints (should be none)

No --window/--control = the pilot (S2 60 s/600 s + S9). A plan is lanes -> batches -> scenarios:
lanes run in parallel, batches in a lane one after another, scenarios in a batch in parallel.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mscache import cost, plan, report, spec
from mscache.api import LiveClient
from mscache.driver import Driver, ScenarioResult

RUNS = Path("runs/mscache")


def _key(s: spec.Scenario) -> str:
    return f"{s.sid}{s.arm}"


def _lanes(args) -> list[plan.Lane]:
    if args.control:
        lanes = plan.control(args.dc, args.gpu)
    elif args.window:
        lanes = plan.matrix(args.dc, args.gpu, window=args.window)
    else:
        scs = plan.pilot(args.dc, args.gpu)
        # --serial: scenarios co-landed on one machine in the 11-22 pilot and shared its download slots
        lanes = [[[s] for s in scs]] if getattr(args, "serial", False) else [[[s]] for s in scs]
    only = set(args.only.split(",")) if getattr(args, "only", None) else None
    if only:
        lanes = [[[s for s in b if _key(s) in only] for b in lane] for lane in lanes]
        lanes = [[b for b in lane if b] for lane in lanes]
        lanes = [lane for lane in lanes if lane]
        if not lanes:
            sys.exit(f"--only {args.only!r} matches no scenario")
    return lanes


def _scenarios(lanes: list[plan.Lane]) -> list[spec.Scenario]:
    return [s for lane in lanes for batch in lane for s in batch]


def _plan(args) -> int:
    lanes = _lanes(args)
    for s in _scenarios(lanes):
        e = cost.estimate(s, size_gb=plan.size_gb(s))
        print(f"{_key(s):14} {s.model:58} loads={e.loads} likely=${e.likely_usd:.2f} "
              f"worst=${e.worst_usd:.2f} wall={e.wall_s / 60:.0f}min")
    t = plan.lanes_total(lanes)
    print(f"TOTAL loads={t['loads']} likely=${t['likely_usd']:.2f} worst=${t['worst_usd']:.2f} "
          f"wall={t['wall_s'] / 60:.0f}min ({len(lanes)} lanes)")
    return 0


def _smoke(args) -> int:
    """$0: every endpoint of the plan at 0 workers; checks each model ref resolves (or is rejected)."""
    live = LiveClient()
    for s in _scenarios(_lanes(args)):
        try:
            made = live.create(s.endpoint_name("smoke"), s)
        except Exception as e:   # noqa: BLE001 - S8 refs are expected to fail; print and go on
            print(f"REJECTED {_key(s)} {s.model}: {e}")
            continue
        live.delete(made["id"])
        print(f"ok {_key(s)} ep={made['id']} ref={made['model_ref']}")
    return _leftovers(args)


def _execute(lanes: list[plan.Lane], new_driver) -> list[tuple[spec.Scenario, ScenarioResult]]:
    def batch(b: plan.Batch) -> list[tuple[spec.Scenario, ScenarioResult]]:
        with ThreadPoolExecutor(len(b)) as pool:
            return list(zip(b, pool.map(lambda s: new_driver().run(s), b)))

    def lane(ln: plan.Lane) -> list[tuple[spec.Scenario, ScenarioResult]]:
        return [r for b in ln for r in batch(b)]

    with ThreadPoolExecutor(len(lanes)) as pool:
        return [r for rs in pool.map(lane, lanes) for r in rs]


def _run(args) -> int:
    if not args.spend:
        sys.exit("refusing to spend without --spend (run `plan` first and get Dean's go)")
    lanes = _lanes(args)
    # parallel per-DC runs start in the same minute; the DC keeps their dirs apart
    run_id = time.strftime("%m%d%H%M", time.gmtime())
    tag = "control" if args.control else f"w{args.window}" if args.window else "pilot"
    out = RUNS / f"{run_id}-{args.dc}-{args.gpu}-{tag}"
    out.mkdir(parents=True)
    lock, path = threading.Lock(), out / "records.jsonl"

    def sink(rec: dict) -> None:
        with lock, path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["kind"] in ("load", "event", "interrupt"):
            print(f"{time.strftime('%H:%M:%S')} {rec['scenario']}{rec['arm']} "
                  f"{rec.get('event') or rec.get('outcome') or rec['kind']} "
                  f"{rec.get('machine_id') or ''} {rec.get('offsets_s', '')}", flush=True)

    live, signals = LiveClient(), {}   # one signal table per run: S4/S5 arms line up through it
    results = _execute(lanes, lambda: Driver(live, run_id=run_id, sink=sink, signals=signals))
    for s, r in results:
        print(f"RESULT {_key(s)} ok={r.ok} {r.reason}")
    recs = [json.loads(line) for line in path.open()]
    (out / "report.md").write_text(report.render(recs))
    print(f"report: {out / 'report.md'}")
    return _leftovers(args)


def _report(args) -> int:
    recs = [json.loads(line) for line in (Path(args.run_dir) / "records.jsonl").open()]
    print(report.render(recs))
    return 0


def _leftovers(_args) -> int:
    left = LiveClient().leftovers()
    print("leftovers:", left or "none")
    return 1 if left else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mscache")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "smoke", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--dc", required=True)
        sp.add_argument("--gpu", required=True, choices=sorted(spec.GPUS))
        which = sp.add_mutually_exclusive_group()
        which.add_argument("--window", type=int, choices=(1, 2), help="full matrix window")
        which.add_argument("--control", action="store_true", help="LRU-off control DC: S2 + S9")
        sp.add_argument("--only", help="comma list of scenario keys, e.g. S2gap600,S9")
        if name == "run":
            sp.add_argument("--spend", action="store_true")
            sp.add_argument("--serial", action="store_true", help="pilot only: one scenario at a time")
    sub.add_parser("report").add_argument("run_dir")
    sub.add_parser("leftovers")
    args = p.parse_args(argv)
    return {"plan": _plan, "smoke": _smoke, "run": _run, "report": _report, "leftovers": _leftovers}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
