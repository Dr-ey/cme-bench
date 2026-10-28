"""RISE (Petsiuk et al., BMVC 2018) with a configurable background.

S_c = 1/(N p) * sum_i P_c(x * M_i + B * (1 - M_i)) * M_i, with M_i bilinearly upsampled random binary
s x s grids (keep-probability p), randomly shifted by up to one cell. The original uses B = 0 in the
model input space; here B is the generation background (mean / black / noise / blur).
The same mask set (seeded by cfg.seed) is used for every image, as in the original.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from cmebench.methods.base import Context, ExplainResult, normalize01, predicted_class, register


@lru_cache(maxsize=2)
def _masks(n: int, s: int, p: float, size: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    cell = int(np.ceil(size / s))
    up = (s + 1) * cell
    grid = (torch.rand((n, 1, s, s), generator=g) < p).float()
    big = F.interpolate(grid, size=(up, up), mode="bilinear", align_corners=False)
    shifts = torch.randint(0, cell, (n, 2), generator=g)
    out = torch.empty((n, 1, size, size))
    for i in range(n):
        dy, dx = int(shifts[i, 0]), int(shifts[i, 1])
        out[i] = big[i, :, dy:dy + size, dx:dx + size]
    return out


@register("rise")
def rise(model, x, background, ctx: Context) -> ExplainResult:
    prm = ctx.params
    n, s, p, bs = int(prm["n_masks"]), int(prm["grid"]), float(prm["p"]), int(prm["batch"])
    size = x.shape[-1]
    masks = _masks(n, s, p, size, int(ctx.cfg["seed"]))
    x = x.to(ctx.device)
    bg = background.to(ctx.device) if background is not None else torch.zeros_like(x)
    target = predicted_class(model, x)
    sal = torch.zeros((size, size), device=ctx.device)
    with torch.no_grad():
        for i in range(0, n, bs):
            m = masks[i:i + bs].to(ctx.device)
            probs = torch.softmax(model(x * m + bg * (1.0 - m)), dim=1)[:, target]
            sal += (probs.view(-1, 1, 1) * m[:, 0]).sum(dim=0)
    sal = (sal / (n * p)).cpu().numpy()
    return ExplainResult(normalize01(sal), target, {"NMasks": n})
