# Ported from cat_vs_dog/exp_001_preprocess.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""exp_001 — per-architecture input normalization.

CRITICAL: the model must receive input in the SAME space it was trained in (see the training
scripts' `build_transforms`). Most nets use ImageNet normalization; GREED was trained with
`ToTensor()` only, i.e. raw [0,1] with NO normalization. Feeding GREED ImageNet-normalized input
is out-of-distribution and silently corrupts every result (wrong predictions + degenerate metrics).

This registry is the single source of truth for which (mean, std) each architecture expects.
A net with no normalization is represented as mean=0, std=1 (Normalize becomes the identity → [0,1]).
"""

from __future__ import annotations

from typing import List, Tuple

from cmebench.core.degradation import IMAGENET_MEAN, IMAGENET_STD

# Identity normalization == raw [0,1] (ToTensor only).
NO_NORM_MEAN: List[float] = [0.0, 0.0, 0.0]
NO_NORM_STD: List[float] = [1.0, 1.0, 1.0]

# arch -> (mean, std). Source: each exp_001_train_<arch>.py::build_transforms (GREED also in
# acc-100-ensemble notebook for ViT). EMPIRICALLY VERIFIED 2026-06-28 (accuracy in each space on
# labelled eval images): greed→[0,1] decisive (ImageNet=0.58); resnet & mobilenet→ImageNet decisive
# ([0,1]≈chance for mobilenet); vit/mixer/vgg19 robust to the shift (≈equal acc) but their TRAINING
# space is ImageNet, which is what we use for valid CME/baseline/metrics.
ARCH_NORM = {
    "vit":       (IMAGENET_MEAN, IMAGENET_STD),
    "resnet":    (IMAGENET_MEAN, IMAGENET_STD),
    # NB: timm mixer_b16_224's NATIVE cfg is 0.5/0.5, but we FINE-TUNED the head on ImageNet-norm
    # input (frozen backbone), so ImageNet is the correct INFERENCE space here — do NOT switch to 0.5.
    "mixer":     (IMAGENET_MEAN, IMAGENET_STD),
    "vgg19":     (IMAGENET_MEAN, IMAGENET_STD),
    "mobilenet": (IMAGENET_MEAN, IMAGENET_STD),
    "greed":     (NO_NORM_MEAN, NO_NORM_STD),   # ToTensor only → raw [0,1], NO ImageNet norm
}


def norm_for_arch(arch: str) -> Tuple[List[float], List[float]]:
    """Return (mean, std) for the architecture's input space. Defaults to ImageNet if unknown."""
    return ARCH_NORM.get(arch, (IMAGENET_MEAN, IMAGENET_STD))
