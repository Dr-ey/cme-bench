"""Orchestrator: schedule every unfinished task of a config, in priority order, on this machine.

    python scripts/run.py --config configs/catsdogs.yaml
    python scripts/run.py --config configs/catsdogs.yaml --machine-id A --part 0/2     # machine A of 2
    python scripts/run.py --config configs/catsdogs.yaml --part 0/2 --sub 1/3          # its local worker 1 of 3
    python scripts/run.py --config configs/catsdogs.yaml --only method=cme,gradcam bg=mean --dry-run
    python scripts/run.py --config configs/catsdogs.yaml --stages extra                 # fidelity_T, sanity, ...

Work is split between machines by (arch, image shard), so an evaluation task and the maps it needs
are produced on the same machine. Results are one file per task: merge machines by copying runs/
(rsync -a runs/ other:cme-bench/runs/  |  robocopy runs \\\\other\\cme-bench\\runs /E).
Each batch of tasks of one architecture runs in a fresh worker process (model loaded once per batch);
failed tasks are retried `orchestrator.retries` times. Safe to stop (Ctrl+C) and restart at any time.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config  # noqa: E402
from cmebench.store import read_json  # noqa: E402
from cmebench.tasks import Task, all_tasks  # noqa: E402


def part_of(task: Task, k: int, n: int, i: int = 0, w: int = 1) -> bool:
    """Machine part k of n, then local worker i of w inside it (w may differ between machines)."""
    h = int(hashlib.sha1(f"{task.arch}:{task.shard}".encode()).hexdigest()[:8], 16)
    return h % n == k and (h // n) % w == i


def matches(task: Task, only: dict) -> bool:
    for key, vals in only.items():
        if key == "bg":
            if not ({task.gen_bg, task.eval_bg} & set(vals)) and not (task.gen_bg is None and task.eval_bg is None):
                return False
        elif str(getattr(task, key)) not in vals:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--stages", default="gen,eval")
    ap.add_argument("--machine-id", default=os.environ.get("CMEBENCH_MACHINE", "local"))
    ap.add_argument("--part", default="0/1", help="k/N: this machine takes part k of N")
    ap.add_argument("--sub", default="0/1", help="i/W: local parallel worker i of W within this part")
    ap.add_argument("--only", nargs="*", default=[], help="filters key=v1,v2 (arch, method, stage, bg)")
    ap.add_argument("--batch", type=int, default=40, help="tasks per worker process")
    ap.add_argument("--wait", action="store_true", help="when blocked, wait for other machines instead of exiting")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    k, n = (int(v) for v in a.part.split("/"))
    i, w = (int(v) for v in a.sub.split("/"))
    only = {kv.split("=")[0]: kv.split("=")[1].split(",") for kv in a.only}
    stages = a.stages.split(",")
    if "extra" in stages:
        from cmebench.extra import run_extra_stages  # noqa: WPS433
        return run_extra_stages(cfg, a)

    tasks = [t for t in all_tasks(cfg, stages) if part_of(t, k, n, i, w) and matches(t, only)]
    retries = int(cfg["orchestrator"]["retries"])
    attempts: Counter = Counter()
    py = sys.executable
    while True:
        pending = [t for t in tasks if not t.is_done(cfg) and attempts[t.id] <= retries]
        if a.dry_run:
            by = Counter((t.stage, t.method) for t in pending)
            print(f"{len(pending)} pending of {len(tasks)} tasks (part {k}/{n})")
            for (stage, m), c in sorted(by.items()):
                print(f"  {stage:5s} {m:16s} {c}")
            return 0
        if not pending:
            print("[run] all tasks done")
            return 0
        ready = [t for t in pending if t.ready(cfg)]
        if not ready:
            msg = f"[run] {len(pending)} tasks blocked by dependencies"
            if a.wait:
                print(msg + " — waiting 120 s")
                time.sleep(120)
                continue
            print(msg + " (run the other parts, sync runs/, then restart)")
            return 1
        arch = ready[0].arch
        batch = [t for t in ready if t.arch == arch][: a.batch]
        for t in batch:
            attempts[t.id] += 1
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\n".join(t.id for t in batch))
            task_file = f.name
        print(f"[run] {time.strftime('%H:%M:%S')} worker: {arch} x {len(batch)} tasks "
              f"({len(pending)} pending)", flush=True)
        cmd = [py, "-m", "cmebench.worker", "--config", a.config, "--task-file", task_file,
               "--machine-id", a.machine_id]
        if a.set:
            cmd += ["--set", *a.set]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
        subprocess.run(cmd, cwd=ROOT, env=env)
        os.unlink(task_file)
        for t in batch:
            if not t.is_done(cfg):
                err = read_json(t.failed_marker(cfg)) or {}
                print(f"[run]   not done: {t.id} {str(err.get('error', ''))[:160]}")


if __name__ == "__main__":
    sys.exit(main())
