# Ported from cat_vs_dog/exp_001_metrics_common.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""Single shared evaluator for the CME-vs-Grad-CAM comparison.

Every method (CME alpha, Grad-CAM, Grad-CAM++, Guided Grad-CAM, Random) is reduced
to one normalized per-pixel importance map [H, W] in [0, 1] and fed through the SAME
functions here. The evaluator does not know where a map came from -- this is the
fairness guarantee of doc/comparison_protocol.md (section 1).

Metrics (doc/comparison_protocol.md section 2):
  * Deletion / Insertion AUC  -- primary, area-independent (sweeps all fractions).
  * FlipRate / ConfidenceDrop / MarginDelta @ equal area -- secondary, matched to the
    existing CME numbers in degradation_summary.md.
  * IoU(map, CME)             -- agreement.
  * Specificity coefficient   -- metric_of_method / metric_of_random.

Conventions:
  * image_tensor: (1, C, H, W), ALREADY preprocessed (ImageNet-normalized, 224x224)
    exactly as build_preprocess() in vit_degradation.py -- same space as CME.
  * baseline B: per-image mean color, expand_as(image) -- identical to CME's `base`
    in vit_degradation.py:104.
  * target_class: int. For binary models (ViT / fine-tuned ResNet50): 0=Cat, 1=Dog.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def mean_color_baseline(image_tensor: torch.Tensor) -> torch.Tensor:
    """Per-image mean color expanded to image shape (same as CME's base)."""
    return image_tensor.mean(dim=(2, 3), keepdim=True).expand_as(image_tensor)


def binarize_top_area(importance: np.ndarray, area_frac: float) -> np.ndarray:
    """Boolean mask (H, W) selecting the top `area_frac` most-important pixels."""
    if area_frac <= 0.0:
        return np.zeros_like(importance, dtype=bool)
    if area_frac >= 1.0:
        return np.ones_like(importance, dtype=bool)
    flat = importance.reshape(-1)
    k = int(round(area_frac * flat.size))
    k = max(1, min(k, flat.size))
    # threshold = k-th largest value
    thresh = np.partition(flat, flat.size - k)[flat.size - k]
    return importance >= thresh


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _target_prob(model: nn.Module, batch: torch.Tensor, target_class: int) -> np.ndarray:
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[:, target_class]
    return probs.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Primary: Deletion / Insertion AUC  (area-independent)
# --------------------------------------------------------------------------- #
def deletion_insertion(
    model: nn.Module,
    image_tensor: torch.Tensor,
    importance: np.ndarray,
    target_class: int,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    batch_steps: bool = True,
) -> Dict[str, object]:
    """Compute Deletion and Insertion curves + AUC for one image.

    Deletion : start from the image, progressively replace the MOST important pixels
               with the baseline. Lower AUC is better (important pixels really hold
               the decision).
    Insertion: start from the baseline, progressively reveal the most important
               pixels. Higher AUC is better.

    The pixel ordering and the step schedule are identical for both, and identical
    across all methods -- only `importance` differs.
    """
    device = _model_device(model)
    image = image_tensor.to(device)
    if baseline is None:
        baseline = mean_color_baseline(image)
    baseline = baseline.to(device)

    _, _, H, W = image.shape
    n_pixels = H * W
    order = np.argsort(importance.reshape(-1))[::-1]  # most important first
    order_t = torch.as_tensor(order.copy(), device=device, dtype=torch.long)

    fractions = np.linspace(0.0, 1.0, n_steps + 1)

    def _compose(mode: str) -> torch.Tensor:
        frames = []
        for f in fractions:
            k = int(round(f * n_pixels))
            keep = torch.zeros(n_pixels, device=device) if mode == "insertion" \
                else torch.ones(n_pixels, device=device)
            if k > 0:
                if mode == "insertion":
                    keep[order_t[:k]] = 1.0   # reveal top-k from the image
                else:
                    keep[order_t[:k]] = 0.0   # delete top-k -> baseline
            keep2d = keep.view(1, 1, H, W)
            frames.append(image * keep2d + baseline * (1.0 - keep2d))
        return torch.cat(frames, dim=0)

    def _curve(mode: str) -> np.ndarray:
        batch = _compose(mode)
        if batch_steps:
            return _target_prob(model, batch, target_class)
        return np.array(
            [_target_prob(model, batch[i : i + 1], target_class)[0] for i in range(batch.shape[0])]
        )

    del_curve = _curve("deletion")
    ins_curve = _curve("insertion")

    return {
        "fractions": fractions,
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
        "deletion_auc": float(np.trapz(del_curve, fractions)),
        "insertion_auc": float(np.trapz(ins_curve, fractions)),
    }


# --------------------------------------------------------------------------- #
# Secondary: FlipRate / ConfidenceDrop / MarginDelta @ equal area
# --------------------------------------------------------------------------- #
def _binary_margin(probs: np.ndarray, target_class: int) -> float:
    """p[target] - max p[others]  (works for binary and multiclass)."""
    p = probs.copy()
    target_p = float(p[target_class])
    p[target_class] = -np.inf
    return target_p - float(p.max())


def removal_metrics(
    model: nn.Module,
    image_tensor: torch.Tensor,
    importance: np.ndarray,
    target_class: int,
    area_frac: float,
    baseline: Optional[torch.Tensor] = None,
    remove_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Remove the top `area_frac` important pixels (remove-only, like CME) and measure.

    Mirrors the CME notebook's remove-only analysis: set the binarized important
    pixels to the baseline and check whether the prediction flips.

    `remove_mask` (bool H,W): use this exact significant region instead of re-binarizing
    `importance`. CME passes its OWN `cgd_mask` here so the metric matches the actual CME mask
    (otherwise top-k of the raw alpha selects a slightly different region → Flip can disagree
    with the notebook's remove-only check).
    """
    device = _model_device(model)
    image = image_tensor.to(device)
    if baseline is None:
        baseline = mean_color_baseline(image)
    baseline = baseline.to(device)

    with torch.no_grad():
        start_probs = torch.softmax(model(image), dim=1)[0].cpu().numpy()
    start_label = int(start_probs.argmax())

    if remove_mask is None:
        remove_mask = binarize_top_area(importance, area_frac)  # True = remove
    remove_mask = np.asarray(remove_mask, dtype=bool)
    keep2d = torch.as_tensor(~remove_mask, dtype=image.dtype, device=device).view(1, 1, *remove_mask.shape)
    removed = image * keep2d + baseline * (1.0 - keep2d)

    with torch.no_grad():
        remove_probs = torch.softmax(model(removed), dim=1)[0].cpu().numpy()
    remove_label = int(remove_probs.argmax())

    return {
        "remove_area_share": float(remove_mask.mean()),
        "flip": float(remove_label != start_label),
        "confidence_drop": float(start_probs[target_class] - remove_probs[target_class]),
        "margin_delta": float(
            _binary_margin(remove_probs, target_class) - _binary_margin(start_probs, target_class)
        ),
        "start_label": start_label,
        "remove_label": remove_label,
    }


# --------------------------------------------------------------------------- #
# Random baseline & specificity
# --------------------------------------------------------------------------- #
def random_smooth_map(
    shape: tuple, generator: torch.Generator, low_res: int = 14
) -> np.ndarray:
    """A smooth random importance map (low-res noise upsampled to `shape`).

    Matches the 'smooth random masks of the same area' baseline from the CME
    notebook (cell 18): random field -> bicubic upsample -> normalize.
    """
    H, W = shape
    noise = torch.rand((1, 1, low_res, low_res), generator=generator)
    up = F.interpolate(noise, size=(H, W), mode="bicubic", align_corners=False)
    arr = up.squeeze().numpy()
    arr = arr - arr.min()
    return (arr / max(float(arr.max()), 1e-8)).astype(np.float32)


def random_flip_rate(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    area_frac: float,
    n_random: int = 100,
    seed: int = 20260518,
    baseline: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Flip rate / mean confidence drop over `n_random` random masks of equal area."""
    _, _, H, W = image_tensor.shape
    gen = torch.Generator().manual_seed(seed)
    flips: List[float] = []
    drops: List[float] = []
    for _ in range(n_random):
        rnd = random_smooth_map((H, W), gen)
        m = removal_metrics(model, image_tensor, rnd, target_class, area_frac, baseline)
        flips.append(m["flip"])
        drops.append(m["confidence_drop"])
    return {
        "random_flip_rate": float(np.mean(flips)),
        "random_confidence_drop": float(np.mean(drops)),
    }


def specificity(method_value: float, random_value: float) -> float:
    """method / random (e.g. flip-rate specificity coefficient). Inf-safe."""
    if random_value <= 1e-8:
        return float("inf") if method_value > 0 else 0.0
    return float(method_value / random_value)


# --------------------------------------------------------------------------- #
# Unified per-image evaluation
# --------------------------------------------------------------------------- #
def evaluate_importance_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    importance: np.ndarray,
    target_class: int,
    area_frac: float,
    cgd_mask: Optional[np.ndarray] = None,
    n_steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    method_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run the full metric suite for one (image, importance map) pair.

    `area_frac` should be the SAME for every method on a given image (use the CME
    mask's remove-area-share) so the secondary metrics are comparable.
    `cgd_mask` (bool H,W) enables IoU(map, CME); pass None to skip.
    `method_mask` (bool H,W): this method's OWN equal-area significant region. CME passes its
    `cgd_mask` so Flip/ConfDrop/IoU reflect the actual CME mask; others leave it None and the
    region is binarize_top_area(importance, area_frac). Deletion/Insertion AUC are unaffected
    (they rank the continuous map, area-independent).
    """
    if baseline is None:
        baseline = mean_color_baseline(image_tensor.to(_model_device(model)))

    if method_mask is None:
        method_mask = binarize_top_area(importance, area_frac)
    else:
        method_mask = np.asarray(method_mask, dtype=bool)

    di = deletion_insertion(model, image_tensor, importance, target_class, baseline, n_steps)
    rm = removal_metrics(model, image_tensor, importance, target_class, area_frac, baseline, remove_mask=method_mask)

    out: Dict[str, object] = {
        "deletion_auc": di["deletion_auc"],
        "insertion_auc": di["insertion_auc"],
        "flip": rm["flip"],
        "confidence_drop": rm["confidence_drop"],
        "margin_delta": rm["margin_delta"],
        "remove_area_share": rm["remove_area_share"],
        "_curves": {
            "fractions": di["fractions"],
            "deletion": di["deletion_curve"],
            "insertion": di["insertion_curve"],
        },
    }
    if cgd_mask is not None:
        out["iou_vs_cgd"] = iou(method_mask, cgd_mask)
    return out
