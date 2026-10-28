"""Run everything the article needs, in priority order; resumable (restart = continue).

    python scripts/run_all.py [--machine-id A --part 0/2] [--from-step N]

Steps (each is a scripts/run.py call; finished work is skipped automatically):
  1  catsdogs   main comparison (12 methods x 4 x 4 backgrounds; EP and Adaptive last by priority)
  2  catsdogs   extra: fidelity_T
  3  catsdogs   extra: sanity
  4  catsdogs   extra: adaptive  -> then cme_adaptive maps + metrics
  5  catsdogs   extra: complementarity
  6  oxford     main maps (backgrounds mean, blur) + extra: oxford localization, shortcut
  7  imagenetv2 main comparison
  8  tables and figures for all datasets
With --part k/N only step 1, 6 (main) and 7 are split between machines; extra stages run on part 0.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args, log) -> int:
    cmd = [PY, *args]
    msg = f"\n[run_all] {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(args)}"
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-id", default="local")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--from-step", type=int, default=1)
    a = ap.parse_args()
    k = int(a.part.split("/")[0])
    common = ["--machine-id", a.machine_id, "--part", a.part]
    main_run = lambda cfg, *extra: ["scripts/run.py", "--config", cfg, *common, *extra]  # noqa: E731
    extra = lambda cfg, stage: ["scripts/run.py", "--config", cfg, "--machine-id", a.machine_id,  # noqa: E731
                                "--stages", "extra", "--only", f"stage={stage}"]
    steps = [
        (1, [main_run("configs/catsdogs.yaml", "--only", "method=" + ",".join(
            ["cme", "gradcam", "gradcam++", "guided", "guided_gradcam", "cam", "random", "rise", "scorecam",
             "libragrad", "mp", "ep"]))], True),
        (2, [extra("configs/catsdogs.yaml", "fidelity_T")], False),
        (3, [extra("configs/catsdogs.yaml", "sanity")], False),
        (4, [extra("configs/catsdogs.yaml", "adaptive"),
             main_run("configs/catsdogs.yaml", "--only", "method=cme_adaptive")], False),
        (5, [extra("configs/catsdogs.yaml", "complementarity")], False),
        (6, [main_run("configs/oxford.yaml"), extra("configs/oxford.yaml", "oxford"),
             extra("configs/oxford.yaml", "shortcut")], True),
        (7, [main_run("configs/imagenetv2.yaml")], True),
        (8, [["scripts/60_stats.py", "--config", c] for c in ("configs/catsdogs.yaml", "configs/oxford.yaml",
                                                              "configs/imagenetv2.yaml")] +
            [["scripts/65_figures.py", "--config", c] for c in ("configs/catsdogs.yaml", "configs/oxford.yaml",
                                                                "configs/imagenetv2.yaml")], False),
    ]
    logp = ROOT / "logs" / f"run_all_{a.machine_id}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    with open(logp, "a", encoding="utf-8") as log:
        for n, cmds, split in steps:
            if n < a.from_step or (not split and k != 0):
                continue
            for c in cmds:
                rc = run(c, log)
                log.write(f"[run_all] step {n} exit {rc}\n")
                log.flush()
    print("[run_all] finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
