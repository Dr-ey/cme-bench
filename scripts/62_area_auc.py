"""Deletion/Insertion AUC restricted to the first part of the curve (default 0-20% of the area).

    python scripts/62_area_auc.py --config configs/catsdogs.yaml [--area 0.2] [--eval-bg mean] [--archs vit mixer]

Uses the evaluation step schedule (n_steps), so the points are the first points of the full curves.
Maps: background methods generated with --gen-bg, the others background-free; Random = one smooth map
(the first of the evaluator's random_k). AUCs are normalised by the area, so they stay in [0, 1].
Writes runs/<dataset>/tables/area_auc_<eval_bg>_<area>.csv (one row per image and method; resumable per arch).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench import backgrounds as bgm  # noqa: E402
from cmebench.config import load_config  # noqa: E402
from cmebench.core import metrics as mc  # noqa: E402
from cmebench.data.datasets import load_image  # noqa: E402
from cmebench.models.registry import norm_for  # noqa: E402
from cmebench.store import load_map, map_path, run_dir  # noqa: E402
from cmebench.tasks import RANDOM  # noqa: E402
from cmebench.worker import Worker  # noqa: E402


def curves(model, x, importance, target, baseline, fractions):
    _, _, h, w = x.shape
    order = torch.as_tensor(np.argsort(importance.reshape(-1))[::-1].copy(), device=x.device)
    out = {}
    for mode in ("deletion", "insertion"):
        frames = []
        for f in fractions:
            k = int(round(f * h * w))
            keep = torch.full((h * w,), 1.0 if mode == "deletion" else 0.0, device=x.device)
            if k > 0:
                keep[order[:k]] = 0.0 if mode == "deletion" else 1.0
            keep = keep.view(1, 1, h, w)
            frames.append(x * keep + baseline * (1.0 - keep))
        out[mode] = mc._target_prob(model, torch.cat(frames), target)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--area", type=float, default=0.2)
    ap.add_argument("--eval-bg", default="mean")
    ap.add_argument("--gen-bg", default="mean")
    ap.add_argument("--archs", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config)
    n_steps = int(cfg["evaluation"]["n_steps"])
    k_max = int(round(a.area * n_steps))
    fractions = np.linspace(0.0, 1.0, n_steps + 1)[: k_max + 1]
    out = run_dir(cfg) / "tables" / f"area_auc_{a.eval_bg}_{int(round(a.area * 100))}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["Arch"])
    w = Worker(cfg, "area_auc")
    archs = a.archs or list(cfg["dataset"]["archs"])
    with torch.no_grad():
        for arch in archs:
            if arch in set(done.Arch):
                continue
            methods = [m for m, spec in cfg["methods"].items()
                       if m != "cme_adaptive" and (spec["archs"] is None or arch in spec["archs"])]
            model = w.model(arch)
            mean, std = norm_for(cfg["dataset"]["name"], arch)
            rows = []
            for item in w.items:
                x = load_image(cfg, item, mean, std).to(w.device)
                baseline = bgm.make_background(x, a.eval_bg, mean, std, item.image_id, "eval")
                target = int(load_map(map_path(cfg, arch, "cme", a.gen_bg, item.image_id))["target_class"])
                for m in methods:
                    if m == RANDOM:
                        g = torch.Generator().manual_seed(bgm.image_seed(item.image_id, 7))
                        sal = mc.random_smooth_map(tuple(x.shape[-2:]), g)
                    else:
                        p = map_path(cfg, arch, m, a.gen_bg if cfg["methods"][m]["bg"] else None, item.image_id)
                        if not p.exists():
                            continue
                        sal = np.asarray(load_map(p)["saliency"], dtype=np.float32)
                    c = curves(model, x, sal, target, baseline, fractions)
                    d = float(np.trapz(c["deletion"], fractions)) / fractions[-1]
                    i = float(np.trapz(c["insertion"], fractions)) / fractions[-1]
                    rows.append({"Arch": arch, "Method": m, "ImageId": item.image_id, "DelAUC": d, "InsAUC": i,
                                 "Gap": d - i, "DelEnd": float(c["deletion"][-1]), "InsEnd": float(c["insertion"][-1])})
                print(f"[area_auc] {arch} {item.image_id}", flush=True)
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            done.to_csv(out, index=False)
            print(f"[area_auc] wrote {arch} -> {out}", flush=True)


if __name__ == "__main__":
    main()
