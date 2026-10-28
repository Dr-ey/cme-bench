"""Progress of a config, decided from result markers (not from logs); logs give timing and ETA.

    python scripts/status.py --config configs/catsdogs.yaml
    python scripts/status.py --config configs/catsdogs.yaml --by bg --failed
    python scripts/status.py --config configs/catsdogs.yaml --watch 120 --export status.csv

States: done | stale (finished with another config hash) | failed | running (heartbeat fresh) |
dead (heartbeat older than orchestrator.dead_after_s) | blocked (dependencies missing) | pending.
Works on any copy of runs/ + logs/ (e.g. after merging several machines).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config, resolve  # noqa: E402
from cmebench.store import read_json  # noqa: E402
from cmebench.tasks import Task, all_tasks  # noqa: E402

STATES = ["done", "running", "pending", "blocked", "failed", "dead", "stale"]


def state(cfg, t: Task, now: float) -> str:
    m = read_json(t.marker(cfg))
    if m:
        return "done" if m.get("hash") == t.expected_hash(cfg) else "stale"
    hb = read_json(t.heartbeat(cfg))
    if hb and hb.get("hash") == t.expected_hash(cfg):
        if now - float(hb.get("beat", 0)) < float(cfg["orchestrator"]["dead_after_s"]):
            return "running"
        if not t.failed_marker(cfg).exists():
            return "dead"
    if t.failed_marker(cfg).exists():
        return "failed"
    return "pending" if t.ready(cfg) else "blocked"


def timings(cfg):
    """Median seconds per (stage, arch, method) from 'done' events + last activity per machine."""
    dur, last = defaultdict(list), {}
    for f in resolve(cfg, "logs").glob("*/*.jsonl"):
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("dataset") != cfg["dataset"]["name"]:
                continue
            last[r["machine"]] = max(last.get(r["machine"], 0), r["t"])
            if r["event"] == "done":
                dur[(r["stage"], r["arch"], r["method"])].append(r["seconds"])
    return {k: median(v) for k, v in dur.items()}, last


def estimate(t: Task, med) -> float:
    if (t.stage, t.arch, t.method) in med:
        return med[(t.stage, t.arch, t.method)]
    same = [v for (s, _, m), v in med.items() if s == t.stage and m == t.method]
    if same:
        return median(same)
    return 60.0 if t.stage == "eval" else 600.0  # unknown yet: rough guess per 20-image shard


def fmt_h(sec: float) -> str:
    return f"{sec / 3600:6.1f} h"


def report(cfg, a) -> None:
    now = time.time()
    tasks = all_tasks(cfg, a.stages.split(","))
    states = {t.id: state(cfg, t, now) for t in tasks}
    med, last = timings(cfg)
    print(f"== {cfg['dataset']['name']}  {time.strftime('%Y-%m-%d %H:%M:%S')}  tasks: {len(tasks)}")
    print("   " + "  ".join(f"{s}={c}" for s, c in Counter(states.values()).most_common()))

    key = {"arch": lambda t: t.arch, "method": lambda t: t.method,
           "bg": lambda t: f"{t.gen_bg or 'none'}>{t.eval_bg or '-'}"}[a.by]
    rows = defaultdict(Counter)
    for t in tasks:
        rows[(t.stage, key(t))][states[t.id]] += 1
    print(f"\n{'stage':6s} {a.by:22s} {'done%':>6s} " + " ".join(f"{s:>8s}" for s in STATES))
    for (stage, k), c in sorted(rows.items()):
        tot = sum(c.values())
        print(f"{stage:6s} {k:22s} {100 * c['done'] / tot:5.1f}% " + " ".join(f"{c[s]:8d}" for s in STATES))

    remaining = [t for t in tasks if states[t.id] != "done"]
    left = sum(estimate(t, med) for t in remaining)
    active = [m for m, ts in last.items() if now - ts < 3600] or ["(none)"]
    print(f"\nremaining work ≈ {fmt_h(left)} of GPU time; active machines (last hour): {', '.join(active)}")
    if len(active) > 1:
        print(f"wall-clock ETA with {len(active)} machines ≈ {fmt_h(left / len(active))}")

    bad = [t for t in tasks if states[t.id] in ("failed", "dead", "stale")]
    if bad and a.failed:
        print("\nproblem tasks:")
        for t in bad[:50]:
            err = read_json(t.failed_marker(cfg)) or {}
            print(f"  {states[t.id]:6s} {t.id}  {str(err.get('error', ''))[:120]}")
    if a.export:
        with open(a.export, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task", "stage", "arch", "method", "gen_bg", "eval_bg", "shard", "state", "est_seconds"])
            for t in tasks:
                w.writerow([t.id, t.stage, t.arch, t.method, t.gen_bg, t.eval_bg, t.shard, states[t.id],
                            round(estimate(t, med))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--stages", default="gen,eval")
    ap.add_argument("--by", choices=["arch", "method", "bg"], default="method")
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("--export")
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    while True:
        report(cfg, a)
        if not a.watch:
            break
        time.sleep(a.watch)
        print("\n" + "-" * 100)


if __name__ == "__main__":
    main()
