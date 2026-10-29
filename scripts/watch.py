"""Console dashboard: progress bars per dataset / architecture / method, the running tasks and the GPU,
redrawn every N seconds (Ctrl+C to quit). Read-only, like status.py, and uses the same task states.

    python scripts/watch.py                                   # catsdogs, oxford, imagenetv2; every 60 s
    python scripts/watch.py --configs configs/catsdogs.yaml --interval 30 --by arch

Bars are weighted by estimated GPU time (median 'done' seconds from logs), so a finished ViT/EP shard
counts more than a finished Grad-CAM evaluation; the task counts are shown next to them.
The extra stages (fidelity_T, sanity, adaptive, ...) are not tasks and are not shown; see logs/run_all_*.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from status import estimate, state, timings  # noqa: E402
from cmebench.config import load_config  # noqa: E402
from cmebench.data.datasets import load_items  # noqa: E402
from cmebench.store import map_path, read_json  # noqa: E402
from cmebench.tasks import all_tasks, shard_slice  # noqa: E402

DEFAULT = ["configs/catsdogs.yaml", "configs/oxford.yaml", "configs/imagenetv2.yaml"]
GREEN, YELLOW, RED, DIM, BOLD, RESET = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[2m", "\x1b[1m", "\x1b[0m"


def setup_console() -> None:
    if os.name == "nt":
        try:
            import ctypes  # noqa: WPS433
            k = ctypes.windll.kernel32
            k.SetConsoleOutputCP(65001)
            h = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:  # noqa: BLE001 — cosmetic only
            pass
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def bar(frac: float, width: int, failed: float = 0.0) -> str:
    frac = min(max(frac, 0.0), 1.0)
    cells = frac * width
    full = int(cells)
    eighths = int((cells - full) * 8) if full < width else 0
    part = "▏▎▍▌▋▊▉"[eighths - 1] if eighths else ""
    s = "█" * full + part
    color = GREEN if frac >= 1.0 else (RED if failed else YELLOW)
    return f"{color}{s}{RESET}{DIM}{'·' * (width - len(s))}{RESET}"


def fmt_h(sec: float) -> str:
    return f"{sec / 3600:5.1f} h" if sec >= 3600 else f"{sec / 60:4.0f} min"


def gpu_line() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                              "power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return "GPU: nvidia-smi not available"
    lines = []
    for ln in out:
        name, util, used, total, power, temp = [v.strip() for v in ln.split(",")]
        t = int(float(temp))
        tc = RED if t >= 85 else (YELLOW if t >= 80 else "")
        lines.append(f"GPU {name}: load {bar(float(util) / 100, 20)} {util:>3s}%   "
                     f"VRAM {float(used) / 1024:.1f}/{float(total) / 1024:.1f} GB   "
                     f"{float(power):.0f} W   {tc}{t} °C{RESET}")
    return "\n".join(lines)


class View:
    def __init__(self, path: str) -> None:
        self.cfg = load_config(path, [])
        self.name = self.cfg["dataset"]["name"]
        self.tasks = all_tasks(self.cfg, ["gen", "eval"])
        self.items = load_items(self.cfg)

    def snapshot(self):
        now = time.time()
        med, last = timings(self.cfg)
        st = {t.id: state(self.cfg, t, now) for t in self.tasks}
        est = {t.id: estimate(t, med) for t in self.tasks}
        return st, est, last

    def shard_progress(self, t) -> str:
        if t.stage != "gen":
            return ""
        items = self.items[shard_slice(self.cfg, t.shard)]
        n = sum(map_path(self.cfg, t.arch, t.method, t.gen_bg, it.image_id).exists() for it in items)
        return f"{n:2d}/{len(items)} img"


def group_rows(view: View, st, est, key) -> list:
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])  # done_s, total_s, done_n, total_n, failed_n
    for t in view.tasks:
        a = agg[key(t)]
        a[1] += est[t.id]
        a[3] += 1
        if st[t.id] == "done":
            a[0] += est[t.id]
            a[2] += 1
        elif st[t.id] in ("failed", "dead"):
            a[4] += 1
    return sorted(agg.items())


def draw(views, a, history) -> str:
    out = [f"{BOLD}cme-bench  {time.strftime('%Y-%m-%d %H:%M:%S')}{RESET}   "
           f"{DIM}(refresh {a.interval} s, Ctrl+C to quit){RESET}", gpu_line(), ""]
    machines = {}
    for v in views:
        st, est, last = v.snapshot()
        for m, ts in last.items():
            machines[m] = max(machines.get(m, 0), ts)
        done_s = sum(est[i] for i, s in st.items() if s == "done")
        tot_s = sum(est.values())
        n_done = sum(s == "done" for s in st.values())
        n_fail = sum(s in ("failed", "dead") for s in st.values())
        left = tot_s - done_s
        # wall-clock rate from the previous refreshes of this dataset
        hist = history.setdefault(v.name, [])
        hist.append((time.time(), done_s))
        del hist[:-60]
        rate = ""
        if len(hist) >= 2 and hist[-1][1] > hist[0][1]:
            speed = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])  # GPU-s per wall-s
            rate = f"   ETA ≈ {fmt_h(left / speed)} wall-clock (x{speed:.1f} over the last {len(hist)} refreshes)"
        fail = f"   {RED}failed {n_fail}{RESET}" if n_fail else ""
        out.append(f"{BOLD}{v.name:11s}{RESET} {bar(done_s / tot_s if tot_s else 0, 40, n_fail)} "
                   f"{100 * done_s / tot_s if tot_s else 0:5.1f}%  tasks {n_done}/{len(st)}  "
                   f"left ≈ {fmt_h(left)} GPU{fail}{rate}")
        if done_s == 0 and not any(s == "running" for s in st.values()):
            continue  # not started yet: one line is enough
        key = {"arch": lambda t: t.arch, "method": lambda t: t.method,
               "bg": lambda t: t.gen_bg or "none"}[a.by]
        for k, (ds, ts, dn, tn, fn) in group_rows(v, st, est, key):
            f = f" {RED}✗{fn}{RESET}" if fn else ""
            out.append(f"  {k:15s} {bar(ds / ts if ts else 0, 30, fn)} {100 * ds / ts if ts else 0:5.1f}%  "
                       f"{dn:5d}/{tn:<5d} {DIM}{fmt_h(ts - ds)} left{RESET}{f}")
        running = [t for t in v.tasks if st[t.id] == "running"]
        if running:
            out.append(f"  {DIM}running:{RESET}")
            for t in running:
                hb = read_json(t.heartbeat(v.cfg)) or {}
                machines[hb.get("machine", "?")] = float(hb.get("beat", 0))
                el = time.time() - float(hb.get("started", time.time()))
                out.append(f"    {hb.get('machine', '?'):11s} {t.id:44s} {v.shard_progress(t):10s} "
                           f"{fmt_h(el)} / ~{fmt_h(est[t.id])}")
        out.append("")
    now = time.time()
    act = [m for m, ts in sorted(machines.items()) if now - ts < 900]
    out.append(f"{DIM}active workers: {', '.join(act) or '(none)'}{RESET}")
    return "\n".join(out)


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=DEFAULT)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--by", choices=["arch", "method", "bg"], default="method")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    views = []
    for c in a.configs:
        try:
            views.append(View(c))
        except Exception as e:  # noqa: BLE001 — e.g. dataset not downloaded yet
            print(f"skip {c}: {e!r}")
    history: dict = {}
    try:
        while True:
            screen = draw(views, a, history)
            print(("" if a.once else "\x1b[2J\x1b[H") + screen, flush=True)
            if a.once:
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print(RESET)


if __name__ == "__main__":
    main()
