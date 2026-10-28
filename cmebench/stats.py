"""Statistics for the article and the response to the reviewer (M9).

All tests are paired over images; Gap = DeletionAUC - InsertionAUC (lower is better).
* Wilcoxon signed-rank (two-sided, scipy default zero handling), n = images;
* Holm correction within each (dataset, arch, gen_bg, eval_bg, reference) family;
* effect sizes: mean paired difference with 95 % bootstrap CI and matched-pairs rank-biserial r
  (positive r / difference = the reference (CME) is better);
* ceiling share: images where the Random map gives Deletion > 0.9 and Insertion > 0.9 (Fig. 5);
* ranking stability across evaluation backgrounds: Kendall tau of method rankings.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

KEYS = ["Dataset", "Arch", "GenBg", "EvalBg"]


def load_metrics(run_dir: Path) -> pd.DataFrame:
    files = glob.glob(str(run_dir / "metrics" / "**" / "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no metric CSVs under {run_dir}")
    d = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    d["Gap"] = d["DeletionAUC"] - d["InsertionAUC"]
    return d


def holm(p: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p), dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(diff: np.ndarray, b: int = 5000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, (b, diff.size))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["DeletionAUC", "InsertionAUC", "Gap", "Flip", "ConfidenceDrop", "IoUvsCME"]
    return d.groupby(KEYS + ["Method"])[cols].mean().reset_index()


def paired_tests(d: pd.DataFrame, reference: str = "cme", metric: str = "Gap") -> pd.DataFrame:
    """Reference method vs every other non-random method, per (dataset, arch, gen_bg, eval_bg).

    For background-independent methods (GenBg='none') the comparison uses the reference generated with
    the same background as the evaluation (diagonal); for background-dependent ones, the same GenBg."""
    out = []
    ref = d[d.Method == reference]
    for (ds, arch, ebg), sub in d.groupby(["Dataset", "Arch", "EvalBg"]):
        for gbg in sorted(set(sub.GenBg) - {"none"}):
            r = ref[(ref.Dataset == ds) & (ref.Arch == arch) & (ref.EvalBg == ebg) & (ref.GenBg == gbg)]
            if r.empty:
                continue
            r = r.set_index("ImageId")[metric]
            others = sub[(sub.Method != reference) & (sub.Method != "random") &
                         ((sub.GenBg == gbg) | ((sub.GenBg == "none") & (gbg == ebg)))]
            fam = []
            for m, o in others.groupby("Method"):
                o = o.set_index("ImageId")[metric]
                common = r.index.intersection(o.index)
                if len(common) < 5:
                    continue
                diff = (o.loc[common] - r.loc[common]).to_numpy()  # >0 -> reference better (Gap lower)
                p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                lo, hi = bootstrap_ci(diff)
                fam.append({"Dataset": ds, "Arch": arch, "GenBg": gbg, "EvalBg": ebg, "Reference": reference,
                            "Method": m, "N": len(common), "RefMean": float(r.loc[common].mean()),
                            "MethodMean": float(o.loc[common].mean()), "MeanDiff": float(diff.mean()),
                            "CI_lo": lo, "CI_hi": hi, "RankBiserial": rank_biserial(diff), "p": p})
            if fam:
                for row, ph in zip(fam, holm([f["p"] for f in fam])):
                    row["p_holm"] = float(ph)
                out += fam
    return pd.DataFrame(out)


def ceiling(d: pd.DataFrame, thr: float = 0.9) -> pd.DataFrame:
    r = d[d.Method == "random"]
    return (r.assign(Ceiling=(r.DeletionAUC > thr) & (r.InsertionAUC > thr))
             .groupby(["Dataset", "Arch", "EvalBg"])["Ceiling"].mean().reset_index())


def fidelity(d: pd.DataFrame) -> pd.DataFrame:
    """CME 'sufficient evidence' table: preservation, residual logit error, retained area, link to Gap."""
    c = d[(d.Method == "cme") & (d.GenBg == d.EvalBg)]
    rows = []
    for (ds, arch, bg), s in c.groupby(["Dataset", "Arch", "GenBg"]):
        row = {"Dataset": ds, "Arch": arch, "Bg": bg, "N": len(s),
               "PredPreservedRate": s.PredPreserved.mean(), "LogitMSE_median": s.LogitMSE.median(),
               "LogitMSE_q75": s.LogitMSE.quantile(0.75), "DeltaMargin_median": s.DeltaMargin.median(),
               "RetainedMass_mean": s.RetainedMass.mean(), "Area05_mean": s.Area05.mean(),
               "Area90_mean": s.Area90.mean(), "Gap_mean": s.Gap.mean()}
        for col in ("LogitMSE", "RetainedMass", "Area90"):
            rho, p = spearmanr(s[col], s.Gap)
            row[f"rho_{col}_Gap"], row[f"p_{col}_Gap"] = float(rho), float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_stability(d: pd.DataFrame, reference_bg: str = "mean") -> pd.DataFrame:
    """Kendall tau between method rankings (mean Gap) under reference_bg and each other eval bg.
    Uses the diagonal: bg-dependent maps generated with the evaluation background."""
    diag = d[(d.GenBg == d.EvalBg) | (d.GenBg == "none")]
    means = diag.groupby(["Dataset", "Arch", "EvalBg", "Method"]).Gap.mean().reset_index()
    out = []
    for (ds, arch), s in means.groupby(["Dataset", "Arch"]):
        base = s[s.EvalBg == reference_bg].set_index("Method").Gap
        for ebg, t in s.groupby("EvalBg"):
            if ebg == reference_bg:
                continue
            t = t.set_index("Method").Gap
            common = base.index.intersection(t.index)
            tau, p = kendalltau(base.loc[common], t.loc[common])
            out.append({"Dataset": ds, "Arch": arch, "EvalBg": ebg, "KendallTau_vs_" + reference_bg: tau,
                        "p": p, "NMethods": len(common)})
    return pd.DataFrame(out)


def write_all(run_dir: Path, out_dir: Optional[Path] = None, references: List[str] = ("cme",)) -> List[Path]:
    d = load_metrics(run_dir)
    out_dir = out_dir or run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in [("summary", summary(d)), ("ceiling", ceiling(d)), ("ranking_stability", ranking_stability(d))]:
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        written.append(p)
    if "PredPreserved" in d.columns:
        p = out_dir / "cme_fidelity.csv"
        fidelity(d).to_csv(p, index=False)
        written.append(p)
    for ref in references:
        p = out_dir / f"paired_{ref}.csv"
        paired_tests(d, ref).to_csv(p, index=False)
        written.append(p)
    return written
