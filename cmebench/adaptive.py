"""Adaptive CME with a held-out validation and a fallback rule (reviewer M7-M8, major comment 2).

Search space (as in the article, exp_001_param_search.py):
  coarse: w_class in {0.01, 0.05, 0.2} x w_sharp in {0.2, 0.5, 0.8} x smoothness {square, abs}
          x preservation {logits, margin} = 36 configs (canonical wc0.05_ws0.5_square_logits included),
          fast optimizer with early stopping; w_blur (w_reg) = 1 - w_class;
  fine:   w_class x {0.5, 1, 2} (clipped to [0.005, 0.5]), w_sharp + {-0.15, 0, +0.15} (clipped to
          [0.05, 0.95]) around the coarse winner, full 1000-iteration optimization.
Protocol (new):
  * pool of 100 images (50 + 50, seed 42, disjoint from eval-200; the article's 16 images are its prefix);
  * split: selection half (50) and validation half (50), stratified;
  * coarse on 32 images of the selection half, fine on all 50; ranking: valid (decision kept on
    >= 95 % of images) first, then lower mean Gap;
  * the selection criterion uses the evaluation background `select_bg` (default 'blur'), NOT the
    background used for the headline table — so Adaptive CME is not tuned on the reported metric;
  * validation: winner vs canonical on the 50 held-out images (paired Wilcoxon on Gap with select_bg);
    the winner is kept only if p < 0.05 AND mean gain >= min_gain; otherwise canonical is used.
Writes runs/catsdogs/adaptive/<arch>_{coarse,fine,validation}.csv and <arch>_selected.json; the
method 'cme_adaptive' then generates eval-200 maps with the selected configuration.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wilcoxon

from cmebench import backgrounds as bgm
from cmebench.config import resolve
from cmebench.core import cme as cme_core
from cmebench.core import metrics as mc
from cmebench.core.cme_fast import cgd_importance_fast
from cmebench.core.degradation import DegradationConfig
from cmebench.data.datasets import Item, load_image, load_items
from cmebench.env import get_device
from cmebench.models.registry import build_model, norm_for
from cmebench.store import run_dir, write_json_atomic

COARSE_W_CLASS = [0.01, 0.05, 0.2]
COARSE_W_SHARP = [0.2, 0.5, 0.8]
ERR_METHODS = ["square", "abs"]
LOSS_MODES = ["logits", "margin"]
FINE_WC_MULT = [0.5, 1.0, 2.0]
FINE_WS_SHIFT = [-0.15, 0.0, 0.15]
WC_CLIP = (0.005, 0.5)
WS_CLIP = (0.05, 0.95)
CANONICAL = (0.05, 0.5, "square", "logits")


def config_id(c: Tuple) -> str:
    wc, ws, err, loss = c
    return f"wc{wc:g}_ws{ws:g}_{err}_{loss}"


def coarse_grid() -> List[Tuple]:
    return [(wc, ws, e, l) for wc in COARSE_W_CLASS for ws in COARSE_W_SHARP for e in ERR_METHODS for l in LOSS_MODES]


def fine_grid(c: Tuple) -> List[Tuple]:
    wc0, ws0, err, loss = c
    pts = {(round(float(np.clip(wc0 * m, *WC_CLIP)), 6), round(float(np.clip(ws0 + d, *WS_CLIP)), 6), err, loss)
           for m in FINE_WC_MULT for d in FINE_WS_SHIFT}
    return sorted(pts)


def pool_items(cfg: Dict, n_per_class: int = 50) -> List[Item]:
    """Same deterministic procedure as the article's tuning set, extended to n_per_class per class."""
    data = resolve(cfg, "data")
    exclude = {i.rel_path for i in load_items(cfg)}
    out: List[Item] = []
    for label, name in enumerate(["Cat", "Dog"]):
        files = sorted((data / "PetImages" / name).glob("*.jpg"))
        random.Random(int(cfg["seed"])).shuffle(files)
        chosen = []
        for p in files:
            rel = f"PetImages/{name}/{p.name}"
            if len(chosen) >= n_per_class:
                break
            if rel in exclude:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:  # noqa: BLE001
                continue
            chosen.append(rel)
        out += [Item(f"{name}_{Path(r).stem}", r, label) for r in sorted(chosen)]
    return out


def evaluate(cfg: Dict, model, arch: str, items: List[Item], c: Tuple, screening: bool, select_bg: str) -> pd.DataFrame:
    wc, ws, err, loss = c
    dcfg = DegradationConfig(iterations=int(cfg["cme"]["iterations"]), lr=float(cfg["cme"]["lr"]), weight_class=wc,
                             weight_blur=round(1.0 - wc, 6), weight_sharpness=ws, err_method=err,
                             init_alpha_logit=float(cfg["cme"]["init_alpha_logit"]))
    mean, std = norm_for("catsdogs", arch)
    dev = next(model.parameters()).device
    rows = []
    for it in items:
        x = load_image(cfg, it, mean, std).to(dev)
        if screening:
            a2d, mask, area, res = cgd_importance_fast(model, x, dev, dcfg, loss, float(cfg["cme"]["max_area"]),
                                                       early_stop=cme_core.EarlyStopConfig())
        else:
            a2d, mask, area, res = cme_core.cgd_importance(model, x, dev, dcfg, loss, float(cfg["cme"]["max_area"]))
        target = int(res["target_class"])
        base = bgm.make_background(x, select_bg, mean, std, it.image_id, "eval")
        di = mc.deletion_insertion(model, x, a2d, target, base, int(cfg["evaluation"]["n_steps"]))
        rows.append({"Arch": arch, "ConfigId": config_id(c), "WClass": wc, "WSharp": ws, "Err": err, "Loss": loss,
                     "Screening": int(screening), "ImageId": it.image_id,
                     "Kept": int(int(res["final_logits"].argmax(1)) == target),
                     "Gap": di["deletion_auc"] - di["insertion_auc"], "Iterations": res.get("iterations_run")})
    return pd.DataFrame(rows)


def _rank(df: pd.DataFrame, keep_rate: float) -> pd.DataFrame:
    agg = df.groupby(["ConfigId", "WClass", "WSharp", "Err", "Loss"]).agg(Gap=("Gap", "mean"), Keep=("Kept", "mean")).reset_index()
    agg["Valid"] = (agg.Keep >= keep_rate).astype(int)
    return agg.sort_values(["Valid", "Gap", "Keep"], ascending=[False, True, False]).reset_index(drop=True)


def _stage(cfg, model, arch, items, grid, screening, select_bg, path: Path) -> pd.DataFrame:
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    have = set(done.ConfigId) if len(done) else set()
    for c in grid:
        if config_id(c) in have:
            continue
        t0 = time.time()
        df = evaluate(cfg, model, arch, items, c, screening, select_bg)
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"[adaptive] {arch} {config_id(c):28s} Gap={df.Gap.mean():+.4f} keep={df.Kept.mean():.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
    return pd.read_csv(path)


def run(cfg: Dict, archs: List[str]) -> None:
    prm = cfg.get("adaptive", {})
    select_bg = prm.get("select_bg", "blur")
    keep_rate = float(prm.get("keep_rate", 0.95))
    min_gain = float(prm.get("min_gain", 0.02))
    n_coarse = int(prm.get("n_coarse", 32))
    pool = pool_items(cfg, int(prm.get("n_per_class", 50)))
    rng = random.Random(int(cfg["seed"]))
    sel, val = [], []
    for label in (0, 1):
        part = [i for i in pool if i.label == label]
        rng.shuffle(part)
        sel += part[: len(part) // 2]
        val += part[len(part) // 2:]
    coarse_items = [i for i in sel if i.label == 0][: n_coarse // 2] + [i for i in sel if i.label == 1][: n_coarse // 2]
    out = run_dir(cfg) / "adaptive"
    out.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    for arch in archs:
        if (out / f"{arch}_selected.json").exists():
            print(f"[adaptive] have {arch}_selected.json")
            continue
        model = build_model("catsdogs", arch, dev, resolve(cfg, "weights"))
        coarse = _rank(_stage(cfg, model, arch, coarse_items, coarse_grid(), True, select_bg,
                              out / f"{arch}_coarse.csv"), keep_rate)
        best_c = tuple(coarse.loc[0, ["WClass", "WSharp", "Err", "Loss"]])
        fine = _rank(_stage(cfg, model, arch, sel, fine_grid(best_c), False, select_bg, out / f"{arch}_fine.csv"), keep_rate)
        winner = tuple(fine.loc[0, ["WClass", "WSharp", "Err", "Loss"]])
        winner = (float(winner[0]), float(winner[1]), str(winner[2]), str(winner[3]))
        vpath = out / f"{arch}_validation.csv"
        vw = _stage(cfg, model, arch, val, [winner, CANONICAL], False, select_bg, vpath)
        a = vw[vw.ConfigId == config_id(CANONICAL)].set_index("ImageId").Gap
        b = vw[vw.ConfigId == config_id(winner)].set_index("ImageId").Gap
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).to_numpy()  # > 0: winner better
        p = float(wilcoxon(diff).pvalue) if winner != CANONICAL and np.any(diff != 0) else 1.0
        keep_winner = winner != CANONICAL and p < 0.05 and diff.mean() >= min_gain
        chosen = winner if keep_winner else CANONICAL
        write_json_atomic(out / f"{arch}_selected.json", {
            "arch": arch, "select_bg": select_bg, "coarse_best": config_id(best_c), "winner": config_id(winner),
            "validation_gain_mean": float(diff.mean()), "validation_p": p, "n_validation": int(len(common)),
            "fallback_to_canonical": not keep_winner, "selected": dict(zip(["w_class", "w_sharp", "err_method",
                                                                              "loss_mode"], chosen)),
            "rule": f"keep winner iff p < 0.05 and mean gain >= {min_gain}",
        })
        print(f"[adaptive] {arch}: winner {config_id(winner)} gain {diff.mean():+.4f} p={p:.3g} -> "
              f"{'WINNER' if keep_winner else 'CANONICAL'}", flush=True)
