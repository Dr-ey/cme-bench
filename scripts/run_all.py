"""Run everything the article needs, in priority order; resumable (restart = continue).

    python scripts/run_all.py [--machine-id A --part 0/2] [--workers 3] [--from-step N]
    python scripts/run_all.py --workers 3 --first gen_bg=mean,none eval_bg=mean,blur   # article minimum first

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
--workers W runs W orchestrators in parallel on this GPU for the split steps (one CME image at a time
leaves the GPU mostly idle); the step ends when all W are finished. Their logs go to logs/<id>.w<i>/.
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


def run_parallel(argss, log) -> int:
    procs = []
    for args in argss:
        msg = f"\n[run_all] {time.strftime('%Y-%m-%d %H:%M:%S')} & {' '.join(args)}"
        print(msg, flush=True)
        log.write(msg + "\n")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT), CMEBENCH_GPU_SHARE=str(len(argss)))
        procs.append(subprocess.Popen([PY, *args], cwd=ROOT, env=env))
        time.sleep(20)  # stagger model loading
    log.flush()
    return max(p.wait() for p in procs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-id", default="local")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--workers", type=int, default=1, help="parallel orchestrators on this GPU (split steps)")
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--first", nargs="*", default=[],
                    help="run.py filters for a priority subset of step 1, e.g. gen_bg=mean,none eval_bg=mean,blur; "
                         "the rest of step 1 then runs after step 5")
    a = ap.parse_args()
    k = int(a.part.split("/")[0])
    common = ["--machine-id", a.machine_id, "--part", a.part]
    main_run = lambda cfg, *extra: ["scripts/run.py", "--config", cfg, *common, *extra]  # noqa: E731
    whole = lambda cfg, *extra: ["scripts/run.py", "--config", cfg,  # noqa: E731
                                 "--machine-id", a.machine_id, *extra]  # non-split step: all parts on part 0
    extra = lambda cfg, stage: ["scripts/run.py", "--config", cfg, "--machine-id", a.machine_id,  # noqa: E731
                                "--stages", "extra", "--only", f"stage={stage}"]
    main_methods = "method=" + ",".join(["cme", "gradcam", "gradcam++", "guided", "guided_gradcam", "cam", "random",
                                         "rise", "scorecam", "libragrad", "mp", "ep"])
    steps = [
        (1, [main_run("configs/catsdogs.yaml", "--only", main_methods, *a.first)], True),
        (2, [extra("configs/catsdogs.yaml", "fidelity_T")], False),
        (3, [extra("configs/catsdogs.yaml", "sanity")], False),
        (4, [extra("configs/catsdogs.yaml", "adaptive"),
             whole("configs/catsdogs.yaml", "--only", "method=cme_adaptive")], False),
        (5, [extra("configs/catsdogs.yaml", "complementarity")], False),
        # with --first: the rest of the catsdogs main comparison (other backgrounds) after the extra stages
        (5, [main_run("configs/catsdogs.yaml", "--only", main_methods)] if a.first else [], True),
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
                if split and a.workers > 1 and c[0] == "scripts/run.py" and "--part" in c:
                    rc = run_parallel([[*c, "--sub", f"{i}/{a.workers}", "--machine-id", f"{a.machine_id}.w{i}"]
                                       for i in range(a.workers)], log)
                else:
                    rc = run(c, log)
                log.write(f"[run_all] step {n} exit {rc}\n")
                log.flush()
    print("[run_all] finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
