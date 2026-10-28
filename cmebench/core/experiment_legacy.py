# Ported from cat_vs_dog/exp_001_experiment.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""exp_001 — live 200-image experiment driver for the orchestrator (cell 8).

Recomputes CME live with the chosen DegradationConfig + loss_mode + area cap, then scores every
method through the single shared evaluator (exp_001_metrics_common). Maximum reuse: this module
only orchestrates exp_001_cgd, exp_001_gradcam_common, exp_001_metrics_common.

Methods: CME, CAM (ResNet50 only), Grad-CAM, Grad-CAM++, Guided Grad-CAM, Random.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import ImageFile

import cmebench.core.cme as cgd
import cmebench.core.gradcam as gc
import cmebench.core.metrics as mc
import cmebench.core.viz as viz
from cmebench.data.eval_set import load_eval_filenames
from cmebench.core.preprocess import norm_for_arch
from cmebench.core.degradation import DegradationConfig, load_and_preprocess_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42  # фіксований seed для генераторів випадкових чисел (відтворюваність експерименту)
PANELS_DIR = Path("outputs/exp_001_panels")  # per-image PNG: original | alpha applied | alpha mask

# Gradient/forward methods scored against the CME mask. CAM is CNN-only (added per-arch below).
_GRAD_METHODS = ["gradcam", "gradcam++", "guided", "guided_gradcam"]


def methods_for_arch(arch: str) -> List[str]:
    # ResNet: full set (CAM + Grad-CAM family + Guided)
    if arch == "resnet":
        return ["cgd", "cam"] + _GRAD_METHODS + ["random"]
    # ViT: no classic CAM, keep Grad-CAM family and Guided
    if arch == "vit":
        return ["cgd"] + _GRAD_METHODS + ["random"]
    # GREED is a CAM-NATIVE CNN (head = conv1x1 -> class maps -> GAP): classic CAM (its own class map)
    # + the honest Grad-CAM family at concentrate_out_d. Same method set as ResNet.
    if arch == "greed":
        return ["cgd", "cam"] + _GRAD_METHODS + ["random"]
    # VGG19 / MobileNetV3 / MLP-Mixer: Grad-CAM family, no classic CAM (heads are not
    # GAP->single-Linear). Mixer uses a token->2D reshape like ViT (precedent: LibraGrad,
    # Mehri et al. 2024); its Grad-CAM is a measurable-but-fragile baseline. Same set as ViT.
    if arch in ("vgg19", "mobilenet", "mixer"):
        return ["cgd"] + _GRAD_METHODS + ["random"]
    raise ValueError(f"Unknown arch: {arch!r}")


def evaluate_image(
    model, arch, image_tensor, alpha, cgd_mask, target_class, area_frac, n_steps, random_k, seed
):
    """Score every method for one image through the shared evaluator."""
    out = {}
    out["cgd"] = mc.evaluate_importance_map(
        model, image_tensor, alpha, target_class, area_frac,
        cgd_mask=cgd_mask, n_steps=n_steps, method_mask=cgd_mask,  # CME scored on its OWN mask
    )
    grad_methods = [m for m in methods_for_arch(arch) if m not in ("cgd", "random")]
    for method in grad_methods:
        imp, _ = gc.importance_map(method, model, image_tensor, target_class=target_class, arch=arch)
        out[method] = mc.evaluate_importance_map(
            model, image_tensor, imp, target_class, area_frac, cgd_mask=cgd_mask, n_steps=n_steps
        )
    # Random baseline averaged over random_k smooth maps of equal area
    gen = torch.Generator().manual_seed(seed)
    H, W = alpha.shape
    rnd = [
        mc.evaluate_importance_map(
            model, image_tensor, mc.random_smooth_map((H, W), gen), target_class, area_frac,
            cgd_mask=cgd_mask, n_steps=n_steps,
        )
        for _ in range(random_k)
    ]
    keys = ["deletion_auc",  # Означає AUC для кривої видалення (видалення пікселів за спаданням важливості)
            "insertion_auc",  # Означає AUC для кривої вставки (вставка пікселів за спаданням важливості)
            "flip",  # Означає кількість змін класу при видаленні пікселів
            "confidence_drop",  # Означає зниження впевненості моделі при видаленні пікселів
            "margin_delta",  # Означає зміну маржі класу при видаленні пікселів
            "iou_vs_cgd"]  # Означає IoU (Intersection over Union) порівняно з CME. Для CME це завжди 1.0, для інших методів IoU порівнюється з CME.
    out["random"] = {k: float(np.mean([r[k] for r in rnd])) for k in keys}
    out["random"]["remove_area_share"] = rnd[0]["remove_area_share"]
    return out


def run_experiment(
    arch: str,
    model: torch.nn.Module,
    cfg: DegradationConfig,
    loss_mode: str = cgd.CGD_LOSS_MODE,
    max_area: float = cgd.CGD_MAX_AREA,
    limit: Optional[int] = None,
    n_steps: int = 50,
    random_k: int = 5,
    out_csv: Optional[str | Path] = None,
    filenames: Optional[List[str]] = None,
    seed: int = SEED,
    panels_dir: Optional[str | Path] = PANELS_DIR,
):
    """Run the full live experiment; returns (rows, summary). Saves CSV if out_csv given.

    `panels_dir` (default outputs/exp_001_panels): per-image PNG [original | alpha applied | mask]
    saved under <panels_dir>/<arch>/. Pass None to skip. `seed`: base RNG seed (per-image = seed+i).
    """
    device = next(model.parameters()).device
    mean, std = norm_for_arch(arch)  # вхідний простір МОДЕЛІ (GREED=[0,1], інші=ImageNet)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if filenames is None:
        filenames = load_eval_filenames()
    if limit:
        filenames = filenames[:limit]
    print(f"[{arch}] live experiment | images={len(filenames)} | loss={loss_mode} | cfg={cfg} | seed={seed} | norm={mean}")

    rows: List[dict] = []
    for i, fn in enumerate(filenames, start=1):
        image_tensor, _ = load_and_preprocess_image(Path(fn), mean, std)
        alpha, mask, area, res = cgd.cgd_importance(model, image_tensor, device, cfg, loss_mode, max_area)
        target_class = int(res["target_class"])
        if panels_dir is not None:
            viz.save_cgd_mask_panel(
                fn, alpha, Path(panels_dir) / arch / f"{i:03d}_{Path(fn).stem}.png",
                label=f"{arch} | {Path(fn).name} | target={target_class}",
            )
        data = evaluate_image(model, arch, image_tensor, alpha, mask, target_class, area, n_steps, random_k, seed=seed + i)
        label = 0 if Path(fn).parent.stem == "Cat" else 1
        for method, m in data.items():
            rows.append({
                "FileName": fn, "Label": label, "TargetClass": target_class, "Method": method,
                "DeletionAUC": round(m["deletion_auc"], 5), "InsertionAUC": round(m["insertion_auc"], 5),
                "Flip": m["flip"], "ConfidenceDrop": round(m["confidence_drop"], 5),
                "MarginDelta": round(m["margin_delta"], 5), "RemoveAreaShare": round(m["remove_area_share"], 5),
                "IoUvsCGD": round(m.get("iou_vs_cgd", float("nan")), 5),
            })
        if i % 10 == 0 or i == len(filenames):
            print(f"[{arch}] {i}/{len(filenames)}")

    summary = summarize(rows, arch)
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[{arch}] saved {out_csv}")
    return rows, summary


def summarize(rows: List[dict], arch: str) -> List[dict]:
    methods = methods_for_arch(arch)
    print(f"\n=== {arch}: aggregated means ===")
    print(f"{'method':16s} {'Del<':>7s} {'Ins>':>7s} {'Flip':>6s} {'ConfDrop':>9s} {'IoUvsCGD':>9s}")
    summary = []
    for method in methods:
        mr = [r for r in rows if r["Method"] == method]
        if not mr:
            continue
        agg = {k: float(np.mean([float(r[k]) for r in mr]))
               for k in ["DeletionAUC", "InsertionAUC", "Flip", "ConfidenceDrop", "IoUvsCGD"]}
        agg["method"] = method
        summary.append(agg)
        print(f"{method:16s} {agg['DeletionAUC']:7.3f} {agg['InsertionAUC']:7.3f} "
              f"{agg['Flip']:6.2f} {agg['ConfidenceDrop']:9.4f} {agg['IoUvsCGD']:9.3f}")
    return summary
