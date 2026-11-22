"""MSCACHE CLI.

  python -m mscache plan  --dc CA-MTL-1 --gpu a40           cost + wall estimate, $0
  python -m mscache smoke --dc CA-MTL-1 --gpu a40           create+delete each endpoint at 0 workers, $0
  python -m mscache run   --dc CA-MTL-1 --gpu a40 --spend   the pilot (PAID): scenarios in parallel
  python -m mscache report runs/mscache/<run_id>            re-render a report
  python -m mscache leftovers                               list msc-* endpoints (should be none)
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
from mscache.driver import Driver

RUNS = Path("runs/mscache")


def _plan(args) -> int:
    scs = plan.pilot(args.dc, args.gpu)
    for s in scs:
        e = cost.estimate(s, size_gb=plan.SIZE_GB[s.model])
        print(f"{s.sid} {s.arm:8} {s.model:40} loads={e.loads} likely=${e.likely_usd:.2f} "
              f"worst=${e.worst_usd:.2f} wall={e.wall_s / 60:.0f}min")
    t = plan.total(scs)
    print(f"TOTAL loads={t['loads']} likely=${t['likely_usd']:.2f} worst=${t['worst_usd']:.2f} "
          f"wall={t['wall_s'] / 60:.0f}min (parallel)")
    return 0


def _smoke(args) -> int:
    live = LiveClient()
    for s in plan.pilot(args.dc, args.gpu):
        made = live.create(s.endpoint_name("smoke"), s)
        live.delete(made["id"])
        print(f"ok {s.sid}{s.arm} ep={made['id']} ref={made['model_ref']}")
    return _leftovers(args)


def _run(args) -> int:
    if not args.spend:
        sys.exit("refusing to spend without --spend (run `plan` first and get Dean's go)")
    run_id = time.strftime("%m%d%H%M", time.gmtime())
    out = RUNS / run_id
    out.mkdir(parents=True)
    lock, path = threading.Lock(), out / "records.jsonl"

    def sink(rec: dict) -> None:
        with lock, path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["kind"] in ("load", "event"):
            print(f"{time.strftime('%H:%M:%S')} {rec['scenario']}{rec['arm']} {rec.get('event') or rec['outcome']} "
                  f"{rec.get('machine_id') or ''} {rec.get('offsets_s', '')}", flush=True)

    live = LiveClient()
    scs = plan.pilot(args.dc, args.gpu)
    with ThreadPoolExecutor(len(scs)) as pool:
        results = list(pool.map(lambda s: Driver(live, run_id=run_id, sink=sink).run(s), scs))
    for s, r in zip(scs, results):
        print(f"RESULT {s.sid}{s.arm} ok={r.ok} {r.reason}")
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
        if name == "run":
            sp.add_argument("--spend", action="store_true")
    sub.add_parser("report").add_argument("run_dir")
    sub.add_parser("leftovers")
    args = p.parse_args(argv)
    return {"plan": _plan, "smoke": _smoke, "run": _run, "report": _report, "leftovers": _leftovers}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
