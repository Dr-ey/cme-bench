"""Table 2 of the article: canonical CME against every competing method on Gap, one Holm family per
(architecture, evaluation background).

    python scripts/61_table2.py --config configs/catsdogs.yaml [--gen-bg mean] [--eval-bg mean blur]

CME and the background-dependent methods are taken with generation background --gen-bg; background-free
methods have none. ΔGap = competitor − CME (positive: CME better), 95% bootstrap CI, matched-pairs
rank-biserial r, two-sided paired Wilcoxon, Holm over the competitors of one architecture and background.
Writes runs/<dataset>/tables/table2_all.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config  # noqa: E402
from cmebench.stats import bootstrap_ci, holm, load_metrics, rank_biserial  # noqa: E402
from cmebench.store import run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gen-bg", default="mean")
    ap.add_argument("--eval-bg", nargs="+", default=["mean", "blur"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = load_metrics(run_dir(cfg))
    d = d[d.EvalBg.isin(a.eval_bg) & d.GenBg.isin([a.gen_bg, "none"])]
    rows = []
    for (arch, ebg), s in d.groupby(["Arch", "EvalBg"]):
        r = s[s.Method == "cme"].set_index("ImageId")["Gap"]
        fam = []
        for m, o in s[~s.Method.isin(["cme", "random", "cme_adaptive"])].groupby("Method"):
            o = o.set_index("ImageId")["Gap"]
            c = r.index.intersection(o.index)
            diff = (o.loc[c] - r.loc[c]).to_numpy()
            lo, hi = bootstrap_ci(diff)
            fam.append(dict(Arch=arch, EvalBg=ebg, Method=m, N=len(c), CME=r.loc[c].mean(), Comp=o.loc[c].mean(),
                            dGap=diff.mean(), lo=lo, hi=hi, r=rank_biserial(diff), p=float(wilcoxon(diff).pvalue)))
        for f, ph in zip(fam, holm([f["p"] for f in fam])):
            f["p_holm"] = float(ph)
        rows += fam
    t = pd.DataFrame(rows)
    t["rank"] = t.groupby(["Arch", "EvalBg"]).Comp.rank()
    out = run_dir(cfg) / "tables" / "table2_all.csv"
    t.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
