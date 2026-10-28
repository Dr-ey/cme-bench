"""Article figures from the benchmark results (style of the article's Fig. 3/7/8 heatmaps).

    python scripts/65_figures.py --config configs/catsdogs.yaml [--bg mean]

Writes runs/<dataset>/figures/:
  heat_{Deletion,Insertion,Gap,Flip,ConfidenceDrop,IoU}_<bg>.png  method x architecture, best non-random outlined
  rho_Gap_ConfDrop_<bg>.png, rho_Del_ConfDrop_<bg>.png            per-image Spearman (article Fig. 8)
  bg_matrix_<arch>.png                                             Gap of each method: generation bg x evaluation bg
  rank_stability.png                                               Kendall tau of method rankings vs 'mean'
  fidelity_T.png, sanity_cascading.png, localization.png, shortcut.png (when those stages exist)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config  # noqa: E402
from cmebench.stats import load_metrics, ranking_stability  # noqa: E402
from cmebench.store import run_dir  # noqa: E402

ARCH_LBL = {"vit": "ViT-B/16", "resnet": "ResNet50", "mixer": "MLP-Mixer-B/16", "greed": "GREED",
            "vgg19": "VGG19", "mobilenet": "MobileNetV3"}
METH_LBL = {"cme": "CME", "cme_adaptive": "Adaptive CME", "cam": "CAM", "gradcam": "Grad-CAM",
            "gradcam++": "Grad-CAM++", "guided": "Guided BP", "guided_gradcam": "Guided Grad-CAM",
            "rise": "RISE", "mp": "Meaningful Pert.", "ep": "Extremal Pert.", "scorecam": "Score-CAM",
            "libragrad": "LibraGrad", "random": "Random"}
LOWER_BETTER = {"DeletionAUC": True, "InsertionAUC": False, "Gap": True, "Flip": False,
                "ConfidenceDrop": False, "IoUvsCME": False}


def heatmap(mat: pd.DataFrame, title: str, path: Path, lower_better: bool, fmt: str = "{:.3f}",
            outline: bool = True, cbar: str = "") -> None:
    rows = [m for m in METH_LBL if m in mat.index]
    cols = [a for a in ARCH_LBL if a in mat.columns]
    mat = mat.reindex(index=rows, columns=cols)
    v = mat.to_numpy(dtype=float)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 3, 0.45 * len(rows) + 2), dpi=200)
    im = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
                continue
            r, g, b, _ = cmap((v[i, j] - vmin) / max(vmax - vmin, 1e-12))
            ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "black")
    if "random" in rows:
        ax.axhline(rows.index("random") - 0.5, color="black", linewidth=1.4)
    if outline:
        nr = [i for i, m in enumerate(rows) if m != "random"]
        for j in range(v.shape[1]):
            col = v[nr, j]
            if np.all(np.isnan(col)):
                continue
            i = nr[int(np.nanargmin(col) if lower_better else np.nanargmax(col))]
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor="red", linewidth=2, zorder=5))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([ARCH_LBL[c] for c in cols], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([METH_LBL[r] for r in rows])
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).set_label(cbar)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def diag(d: pd.DataFrame, bg: str) -> pd.DataFrame:
    """Rows for evaluation background `bg`: bg-dependent maps generated with the same bg."""
    return d[(d.EvalBg == bg) & ((d.GenBg == bg) | (d.GenBg == "none"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--bg", nargs="+", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    rd = run_dir(cfg)
    out = rd / "figures"
    d = load_metrics(rd)
    ds = cfg["dataset"]["name"]
    for bg in a.bg or cfg["backgrounds"]:
        s = diag(d, bg)
        for col, lower in LOWER_BETTER.items():
            m = s.groupby(["Method", "Arch"])[col].mean().unstack("Arch")
            heatmap(m, f"{ds}: mean {col} (background: {bg})", out / f"heat_{col}_{bg}.png", lower,
                    outline=col != "IoUvsCME", cbar=col)
        for col, name in (("Gap", "Gap"), ("DeletionAUC", "Del")):
            rho = s.groupby(["Method", "Arch"]).apply(
                lambda g: spearmanr(g[col], g.ConfidenceDrop).correlation).unstack("Arch")
            heatmap(rho, f"{ds}: Spearman ρ({name}, ConfidenceDrop) (background: {bg})",
                    out / f"rho_{name}_ConfDrop_{bg}.png", True, fmt="{:+.2f}", outline=col == "Gap", cbar="ρ")
    for arch, s in d[d.GenBg != "none"].groupby("Arch"):
        piv = s.groupby(["Method", "GenBg", "EvalBg"]).Gap.mean().reset_index()
        methods = sorted(piv.Method.unique(), key=list(METH_LBL).index)
        bgs = cfg["backgrounds"]
        fig, axes = plt.subplots(1, len(methods), figsize=(2.6 * len(methods), 2.8), dpi=200, squeeze=False)
        for ax, m in zip(axes[0], methods):
            mat = piv[piv.Method == m].pivot(index="GenBg", columns="EvalBg", values="Gap").reindex(index=bgs, columns=bgs)
            ax.imshow(mat.to_numpy(float), cmap="viridis")
            for i in range(len(bgs)):
                for j in range(len(bgs)):
                    ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
            ax.set_title(METH_LBL[m], fontsize=9)
            ax.set_xticks(range(len(bgs)))
            ax.set_xticklabels(bgs, rotation=45, fontsize=7)
            ax.set_yticks(range(len(bgs)))
            ax.set_yticklabels(bgs, fontsize=7)
        axes[0][0].set_ylabel("generation bg")
        fig.suptitle(f"{ARCH_LBL.get(arch, arch)}: Gap, generation × evaluation background", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"bg_matrix_{arch}.png", bbox_inches="tight")
        plt.close(fig)
    rs = ranking_stability(d)
    if len(rs):
        tau = [c for c in rs.columns if c.startswith("KendallTau")][0]
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=200)
        for ebg, g in rs.groupby("EvalBg"):
            ax.plot([ARCH_LBL.get(x, x) for x in g.Arch], g[tau], marker="o", label=f"mean vs {ebg}")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_ylabel("Kendall τ of method ranking (Gap)")
        ax.set_title(f"{ds}: ranking stability across evaluation backgrounds", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "rank_stability.png", bbox_inches="tight")
        plt.close(fig)
    _extra_figures(rd, out)
    print("figures in", out)


def _extra_figures(rd: Path, out: Path) -> None:
    f = sorted(glob.glob(str(rd / "fidelity_T" / "*.csv")))
    if f:
        d = pd.concat(pd.read_csv(x) for x in f)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), dpi=200)
        for arch, s in d.groupby("Arch"):
            g = s.groupby("T").agg(Area=("Area05", "mean"), Gap=("Gap", "mean"), Keep=("PredPreserved", "mean"))
            for ax, col in zip(axes, ("Area", "Gap", "Keep")):
                ax.plot(g.index, g[col], marker="o", label=ARCH_LBL.get(arch, arch))
        for ax, t in zip(axes, ("area with α > 0.5", "Gap (bg mean)", "prediction preserved")):
            ax.set_xscale("log")
            ax.set_xlabel("iterations T")
            ax.set_title(t)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fidelity_T.png", bbox_inches="tight")
        plt.close(fig)
    f = sorted(glob.glob(str(rd / "sanity" / "cascading_*.csv")))
    if f:
        d = pd.concat(pd.read_csv(x) for x in f)
        archs = sorted(d.Arch.unique())
        fig, axes = plt.subplots(1, len(archs), figsize=(4 * len(archs), 3.2), dpi=200, squeeze=False)
        for ax, arch in zip(axes[0], archs):
            s = d[d.Arch == arch]
            labels = s.drop_duplicates("Step").sort_values("Step").Randomized.tolist()
            for m, g in s.groupby("Method"):
                gg = g.groupby("Step").Spearman.mean()
                ax.plot(gg.index, gg.values, marker="o", label=METH_LBL.get(m, m))
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
            ax.set_ylim(-0.2, 1.05)
            ax.set_title(f"{ARCH_LBL.get(arch, arch)}: cascading randomization")
            ax.set_ylabel("Spearman ρ with original map")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "sanity_cascading.png", bbox_inches="tight")
        plt.close(fig)
    f = sorted(glob.glob(str(rd / "localization" / "*.csv")))
    if f:
        d = pd.concat(pd.read_csv(x) for x in f)
        d = d[d.GenBg.isin(["none", "mean"])]
        heatmap(d.groupby(["Method", "Arch"]).EnergyIn.mean().unstack("Arch"),
                "Oxford-IIIT Pet: share of saliency on the pet (energy pointing game)", out / "localization.png",
                False, fmt="{:.2f}", cbar="EnergyIn")
    p = rd / "shortcut"
    f = sorted(glob.glob(str(p / "*.csv")))
    f = [x for x in f if not x.endswith("heads.csv")]
    if f:
        d = pd.concat(pd.read_csv(x) for x in f)
        g = d.groupby(["Arch", "Method", "Head"]).BackgroundMass.mean().unstack("Head")
        g["delta"] = g["biased"] - g["clean"]
        heatmap(g["delta"].unstack("Arch"), "Shortcut: BackgroundMass(biased) − BackgroundMass(clean)",
                out / "shortcut.png", False, fmt="{:+.3f}", cbar="Δ background mass")


if __name__ == "__main__":
    main()
