"""Meaningful Perturbation (Fong & Vedaldi, ICCV 2017), deletion game, configurable background.

A low-resolution mask m (mask_res x mask_res, 1 = keep) is upsampled bilinearly to the image size;
the perturbed input is x * m + B * (1 - m) (original: B = blurred image, sigma 10). We minimize
    P_c(perturbed) + l1 * mean|1 - m| + tv * TV_beta(m)
with Adam, clamping m to [0,1] after every step, and random jitter of the composed input (shift up to
`jitter` px) to avoid adversarial artifacts. Saliency = 1 - m (upsampled): what had to be deleted.
Coefficients follow the widely used PyTorch reproduction of the paper (mean-normalized norms).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cmebench.methods.base import Context, ExplainResult, normalize01, predicted_class, register


def _tv(m: torch.Tensor, beta: float) -> torch.Tensor:
    dy = (m[..., 1:, :] - m[..., :-1, :]).abs().pow(beta).mean()
    dx = (m[..., :, 1:] - m[..., :, :-1]).abs().pow(beta).mean()
    return dx + dy


@register("mp")
def meaningful_perturbation(model, x, background, ctx: Context) -> ExplainResult:
    prm = ctx.params
    x = x.to(ctx.device)
    bg = background.to(ctx.device) if background is not None else torch.zeros_like(x)
    size = x.shape[-2:]
    target = predicted_class(model, x)
    r = int(prm["mask_res"])
    m = torch.ones((1, 1, r, r), device=ctx.device, requires_grad=True)
    opt = torch.optim.Adam([m], lr=float(prm["lr"]))
    g = torch.Generator().manual_seed(int(ctx.cfg["seed"]))
    jit = int(prm.get("jitter", 0))
    for _ in range(int(prm["iterations"])):
        up = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
        inp = x * up + bg * (1.0 - up)
        if jit > 0:
            dy, dx = (int(v) for v in torch.randint(-jit, jit + 1, (2,), generator=g))
            inp = torch.roll(inp, shifts=(dy, dx), dims=(2, 3))
        prob = torch.softmax(model(inp), dim=1)[0, target]
        loss = prob + float(prm["l1"]) * (1.0 - m).abs().mean() + float(prm["tv"]) * _tv(m, float(prm["tv_beta"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            m.clamp_(0.0, 1.0)
    with torch.no_grad():
        up = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
        final_prob = float(torch.softmax(model(x * up + bg * (1.0 - up)), dim=1)[0, target])
    sal = (1.0 - up)[0, 0].detach().cpu().numpy()
    return ExplainResult(normalize01(sal), target, {"FinalProb": final_prob, "DeletedMass": float(sal.mean())})
