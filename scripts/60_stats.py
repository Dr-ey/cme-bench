"""Tables for the article / reviewer response from runs/<dataset>/metrics.

    python scripts/60_stats.py --config configs/catsdogs.yaml [--reference cme libragrad]

Writes runs/<dataset>/tables/{summary, ceiling, ranking_stability, cme_fidelity, paired_<ref>}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config  # noqa: E402
from cmebench.stats import write_all  # noqa: E402
from cmebench.store import run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--reference", nargs="+", default=["cme"])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    for p in write_all(run_dir(cfg), references=a.reference):
        print("wrote", p)


if __name__ == "__main__":
    main()
