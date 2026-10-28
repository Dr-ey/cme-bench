"""Backgrounds B for CME generation and for perturbation-based evaluation.

All backgrounds are built in PIXEL space [0,1] and only then mapped to the MODEL input space with the
architecture's (mean, std). This matters: "black" in an ImageNet-normalized space is (0-mean)/std,
not 0 (0 there is the ImageNet-mean grey), while for GREED (raw [0,1] input) black is 0.

kinds:
  mean  — per-image mean colour (canonical CME baseline of the article)
  black — 0 in pixel space
  noise — i.i.d. U[0,1] per pixel and channel; seeded per image so every method sees the same noise
  blur  — Gaussian blur of the image in pixel space (RISE / Meaningful Perturbation style)

For `noise`, generation and evaluation MUST use different seed offsets (see NOISE_OFFSET_*), otherwise
CME is optimized against the very noise realization it is later evaluated with.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import torch
import torch.nn.functional as F

KINDS = ("mean", "black", "noise", "blur")
NOISE_OFFSET_GEN = 0
NOISE_OFFSET_EVAL = 1_000_003
BLUR_SIGMA = 10.0  # px on 224x224, as in RISE / Meaningful Perturbation


def image_seed(image_id: str, offset: int = 0) -> int:
    """Stable per-image seed (independent of Python's hash randomization)."""
    h = hashlib.sha1(image_id.replace("\\", "/").encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + offset) % (2**31 - 1)


def _view(v: Sequence[float], x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(v, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)


def to_pixel(x_model: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    return x_model * _view(std, x_model) + _view(mean, x_model)


def to_model(x_pix: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    return (x_pix - _view(mean, x_pix)) / _view(std, x_pix)


def gaussian_blur(x: torch.Tensor, sigma: float = BLUR_SIGMA) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding; kernel radius = ceil(3 sigma)."""
    radius = int(torch.ceil(torch.tensor(3.0 * sigma)).item())
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(coords**2) / (2.0 * sigma**2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    y = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    y = F.conv2d(y, k.expand(c, 1, 1, -1), groups=c)
    y = F.pad(y, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def make_background_pixel(x_pix: torch.Tensor, kind: str, seed: int = 0) -> torch.Tensor:
    """Background in pixel space [0,1], same shape as `x_pix` (1,C,H,W)."""
    if kind == "mean":
        return x_pix.mean(dim=(2, 3), keepdim=True).expand_as(x_pix).contiguous()
    if kind == "black":
        return torch.zeros_like(x_pix)
    if kind == "noise":
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.rand(x_pix.shape, generator=g, dtype=torch.float32).to(x_pix.device, x_pix.dtype)
    if kind == "blur":
        return gaussian_blur(x_pix, BLUR_SIGMA).clamp(0.0, 1.0)
    raise ValueError(f"Unknown background kind {kind!r}; expected one of {KINDS}")


def make_background(
    x_model: torch.Tensor, kind: str, mean: Sequence[float], std: Sequence[float],
    image_id: str = "", stage: str = "gen",
) -> torch.Tensor:
    """Background in the MODEL input space for an already-normalized image tensor.

    `stage` selects the noise seed offset ('gen' for map generation, 'eval' for metrics).
    For kind='mean' this equals the article's `image.mean(dim=(2,3))` in model space, because the
    per-channel normalization is affine.
    """
    if kind == "mean":  # computed directly in model space -> bit-identical to the article code
        return x_model.mean(dim=(2, 3), keepdim=True).expand_as(x_model).contiguous()
    offset = NOISE_OFFSET_GEN if stage == "gen" else NOISE_OFFSET_EVAL
    x_pix = to_pixel(x_model, mean, std)
    bg_pix = make_background_pixel(x_pix, kind, seed=image_seed(image_id, offset))
    return to_model(bg_pix, mean, std)
