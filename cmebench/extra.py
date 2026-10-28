"""Additional stages required by the review. Run via `scripts/run.py --stages extra --only stage=<name>`.

fidelity_T       CME with T = 4000 and snapshots at {100,250,500,1000,2000,4000} on 50 images per arch
                 (one optimization trajectory; Adam has no schedule, so the T=1000 snapshot equals the
                 canonical run). Per snapshot: preservation, logit error, retained area, Del/Ins/Gap.
complementarity  Uses existing maps (gen/eval bg 'mean'): removes CME-only, competitor-only and shared
                 equal-area regions; evaluates a rank-averaged CME+competitor map.
sanity           cascading / label randomization / class sensitivity (cmebench/sanity.py).
oxford           localization vs trimaps of maps produced with configs/oxford.yaml (cmebench/localization.py).
shortcut         controlled background shortcut on Oxford composites (cmebench/shortcut.py).
adaptive         Adaptive CME with cross-validated tuning (cmebench/adaptive.py).
Outputs: runs/<dataset>/<stage>/<arch>.csv ; resumable per image.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from cmebench import backgrounds as bgm
from cmebench.config import resolve
from cmebench.core import metrics as mc
from cmebench.data.datasets import Item, load_image, load_items
from cmebench.env import get_device, set_determinism
from cmebench.methods.base import Context
from cmebench.methods.legacy import fidelity_fields, run_cme
from cmebench.models.registry import build_model, norm_for
from cmebench.store import load_map, map_path, run_dir

STAGES = ["fidelity_T", "complementarity", "sanity", "oxford", "shortcut", "adaptive"]
SNAPSHOTS = [100, 250, 500, 1000, 2000, 4000]


def _balanced(items: List[Item], n: int) -> List[Item]:
    cats = [i for i in items if i.label == 0][: n // 2]
    dogs = [i for i in items if i.label == 1][: n - n // 2]
    return cats + dogs if cats and dogs else items[:n]


def _append(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def _done_ids(path: Path) -> set:
    return set(pd.read_csv(path).ImageId) if path.exists() else set()


def fidelity_T(cfg: Dict, archs: List[str], n_images: int = 50) -> None:
    dev = get_device()
    items = _balanced(load_items(cfg), n_images)
    for arch in archs:
        out = run_dir(cfg) / "fidelity_T" / f"{arch}.csv"
        done = _done_ids(out)
        model = build_model(cfg["dataset"]["name"], arch, dev, resolve(cfg, "weights"))
        mean, std = norm_for(cfg["dataset"]["name"], arch)
        c4 = dict(cfg, cme=dict(cfg["cme"], iterations=max(SNAPSHOTS), fast=False))
        for it in items:
            if it.image_id in done:
                continue
            t0 = time.time()
            x = load_image(cfg, it, mean, std).to(dev)
            ctx = Context(arch, cfg["dataset"]["name"], it.image_id, tuple(mean), tuple(std), dev, {}, c4, "mean")
            _, _, _, res = run_cme(model, x, None, ctx, snapshot_steps=SNAPSHOTS)
            rows = []
            for t in SNAPSHOTS:
                snap = res["snapshots"][t]
                a2d = snap["alpha"].mean(dim=1).squeeze(0).numpy().astype(np.float32)
                fid = fidelity_fields({"target_logits": res["target_logits"], "final_logits": snap["logits"],
                                       "target_class": res["target_class"], "iterations_run": t}, a2d)
                di = mc.deletion_insertion(model, x, a2d, int(res["target_class"]), None,
                                           int(cfg["evaluation"]["n_steps"]))
                rows.append({"Arch": arch, "ImageId": it.image_id, "T": t, **fid,
                             "DeletionAUC": di["deletion_auc"], "InsertionAUC": di["insertion_auc"],
                             "Gap": di["deletion_auc"] - di["insertion_auc"]})
            _append(out, rows)
            print(f"[fidelity_T] {arch} {it.image_id} {time.time() - t0:.0f}s", flush=True)


def complementarity(cfg: Dict, archs: List[str], competitor: str = "gradcam") -> None:
    dev = get_device()
    items = load_items(cfg)
    for arch in archs:
        out = run_dir(cfg) / "complementarity" / f"{arch}_{competitor}.csv"
        done = _done_ids(out)
        model = build_model(cfg["dataset"]["name"], arch, dev, resolve(cfg, "weights"))
        mean, std = norm_for(cfg["dataset"]["name"], arch)
        for it in items:
            if it.image_id in done:
                continue
            p_c = map_path(cfg, arch, "cme", "mean", it.image_id)
            p_g = map_path(cfg, arch, competitor, None, it.image_id)
            if not (p_c.exists() and p_g.exists()):
                continue
            c, g = load_map(p_c), load_map(p_g)
            x = load_image(cfg, it, mean, std).to(dev)
            base = bgm.make_background(x, "mean", mean, std, it.image_id, "eval")
            target = int(c["target_class"])
            cm = np.asarray(c["binary_mask"], bool)
            gm = mc.binarize_top_area(np.asarray(g["saliency"], np.float32), float(cm.mean()))
            row = {"Arch": arch, "ImageId": it.image_id, "Competitor": competitor,
                   "IoU": mc.iou(cm, gm), "AreaCME": float(cm.mean())}
            for name, region in (("cme_only", cm & ~gm), ("comp_only", gm & ~cm), ("both", cm & gm),
                                 ("union", cm | gm)):
                r = mc.removal_metrics(model, x, np.zeros(cm.shape, np.float32), target, 0.0, base, remove_mask=region)
                row[f"Area_{name}"] = float(region.mean())
                row[f"ConfDrop_{name}"] = r["confidence_drop"]
                row[f"Flip_{name}"] = r["flip"]
            rank = lambda s: pd.Series(np.asarray(s, np.float64).ravel()).rank(pct=True).to_numpy().reshape(cm.shape)  # noqa: E731
            combo = ((rank(c["saliency"]) + rank(g["saliency"])) / 2).astype(np.float32)
            for name, smap in (("cme", c["saliency"]), ("comp", g["saliency"]), ("combined", combo)):
                di = mc.deletion_insertion(model, x, np.asarray(smap, np.float32), target, base,
                                           int(cfg["evaluation"]["n_steps"]))
                row[f"Gap_{name}"] = di["deletion_auc"] - di["insertion_auc"]
            _append(out, [row])


def run_extra_stages(cfg: Dict, args) -> int:
    if cfg.get("determinism", True):
        set_determinism(int(cfg["seed"]))
    only = {kv.split("=")[0]: kv.split("=")[1].split(",") for kv in getattr(args, "only", [])}
    stages = only.get("stage", ["fidelity_T", "complementarity"])
    archs = only.get("arch", cfg["dataset"]["archs"])
    for st in stages:
        if st == "fidelity_T":
            fidelity_T(cfg, archs)
        elif st == "complementarity":
            for comp in only.get("competitor", ["gradcam", "guided_gradcam"]):
                complementarity(cfg, archs, comp)
        elif st == "sanity":
            from cmebench import sanity

            sanity.run(cfg, only.get("arch", ["resnet", "vgg19", "vit", "mixer"]),
                       only.get("test", ["cascading", "labels", "class"]))
        elif st == "oxford":  # needs maps of configs/oxford.yaml (run.py --config configs/oxford.yaml first)
            from cmebench import localization

            localization.run(cfg, archs)
        elif st == "shortcut":  # run with configs/oxford.yaml
            from cmebench import shortcut

            shortcut.run(cfg, only.get("arch", ["resnet", "vit"]))
        elif st == "adaptive":  # configs/catsdogs.yaml; afterwards run.py generates 'cme_adaptive' maps
            from cmebench import adaptive

            adaptive.run(cfg, archs)
        else:
            print(f"unknown stage {st!r}; available: {', '.join(STAGES)}")
            return 2
    return 0
