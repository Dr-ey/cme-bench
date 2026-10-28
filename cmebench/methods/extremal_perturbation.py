"""Extremal Perturbations (Fong, Patrick & Vedaldi, ICCV 2019) via the authors' TorchRay package.

TorchRay is CC BY-NC 4.0, so it is NOT vendored: install.sh/.cmd run `pip install torchray==1.0.0.2
--no-deps` (EP needs only torch/numpy/matplotlib/packaging, not pycocotools/pymongo).

Background: TorchRay builds a perturbation pyramid between the input and a "maximally perturbed"
image — Gaussian blur (σ_max 20, the original) or fade-to-zero. We keep the original blur pyramid
for gen_bg='blur' and, for mean/black/noise, use a fade pyramid towards that background
(y_l = B + (x - B) * l/(L-1)); for B = 0 this is TorchRay's own fade perturbation.
Preservation game, areas from config; saliency = sum of the area masks (authors' recommendation for
Deletion/Insertion), normalized.
"""
from __future__ import annotations

import threading

import torch

from cmebench.methods.base import Context, ExplainResult, normalize01, predicted_class, register

_LOCK = threading.Lock()


def _fade_class(base_cls, background: torch.Tensor):
    class FadeToBackground(base_cls):
        def __init__(self, input, num_levels=8, max_blur=20, type=None):  # noqa: A002 (TorchRay signature)
            self.type = "fade_to_background"
            self.num_levels = num_levels
            with torch.no_grad():
                lv = torch.linspace(0, 1, num_levels, device=input.device).view(-1, 1, 1, 1)
                self.pyramid = background + (input - background) * lv
    return FadeToBackground


@register("ep")
def extremal_perturbation(model, x, background, ctx: Context) -> ExplainResult:
    try:
        from torchray.attribution import extremal_perturbation as ep_mod
    except ImportError as e:  # pragma: no cover
        raise ImportError("TorchRay missing: pip install torchray==1.0.0.2 --no-deps") from e
    prm = ctx.params
    x = x.to(ctx.device)
    target = predicted_class(model, x)
    areas = [float(a) for a in prm["areas"]]
    torch.manual_seed(int(ctx.cfg["seed"]))
    with _LOCK:  # we temporarily swap the module's Perturbation class
        original = ep_mod.Perturbation
        try:
            if ctx.gen_bg != "blur":
                bg = background.to(ctx.device) if background is not None else torch.zeros_like(x)
                ep_mod.Perturbation = _fade_class(original, bg)
            masks, _ = ep_mod.extremal_perturbation(
                model, x, target, areas=areas, max_iter=int(prm["iterations"]),
                perturbation=ep_mod.BLUR_PERTURBATION, variant=ep_mod.PRESERVE_VARIANT,
                reward_func=ep_mod.simple_reward, resize=True, smooth=0,
            )
        finally:
            ep_mod.Perturbation = original
    sal = masks.sum(dim=0)[0].detach().cpu().numpy()
    return ExplainResult(normalize01(sal), target, {"Areas": ",".join(str(a) for a in areas)})

