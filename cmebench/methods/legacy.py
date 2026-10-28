"""Methods of the article (unchanged algorithms): CME, CAM, Grad-CAM, Grad-CAM++, Guided BP,
Guided Grad-CAM, Random. Thin adapters over cmebench.core.* (ported exp_001 code)."""
from __future__ import annotations

import numpy as np
import torch

from cmebench.core import cme as cme_core
from cmebench.core import cme_fast
from cmebench.core import gradcam as gc
from cmebench.core.degradation import DegradationConfig
from cmebench.core.metrics import random_smooth_map
from cmebench.methods.base import Context, ExplainResult, normalize01, predicted_class, register


def cme_config(cfg) -> DegradationConfig:
    c = cfg["cme"]
    return DegradationConfig(
        iterations=int(c["iterations"]), lr=float(c["lr"]), weight_class=float(c["weight_class"]),
        weight_blur=float(c["weight_blur"]), weight_sharpness=float(c["weight_sharpness"]),
        err_method=c["err_method"], init_alpha_logit=float(c["init_alpha_logit"]),
    )


def run_cme(model, x, background, ctx: Context, loss_mode=None, target_class=None, snapshot_steps=None):
    """Raw CME call shared by the 'cme' method and the fidelity/sanity stages."""
    c = ctx.cfg["cme"]
    cfg = cme_config(ctx.cfg)
    loss_mode = loss_mode or c["loss_mode"]
    if c.get("fast") and snapshot_steps is None:
        return cme_fast.cgd_importance_fast(model, x, ctx.device, cfg, loss_mode, float(c["max_area"]),
                                            base=background, target_class=target_class)
    return cme_core.cgd_importance(model, x, ctx.device, cfg, loss_mode, float(c["max_area"]),
                                   base=background, target_class=target_class, snapshot_steps=snapshot_steps)


def fidelity_fields(res: dict, alpha_2d: np.ndarray) -> dict:
    """Prediction preservation, residual logit error and retained area (reviewer: 'sufficient evidence')."""
    z_b = res["target_logits"].float()
    z_t = res["final_logits"].float()
    c = int(res["target_class"])
    k = z_b.shape[1]
    others = [j for j in range(k) if j != c]
    margin = lambda z: float(z[0, c] - z[0, others].max())  # noqa: E731
    p_b = torch.softmax(z_b, 1)[0, c].item()
    p_t = torch.softmax(z_t, 1)[0, c].item()
    a = np.sort(alpha_2d.reshape(-1))[::-1]
    cum = np.cumsum(a) / max(float(a.sum()), 1e-12)
    return {
        "PredPreserved": int(int(z_t.argmax(1)) == c),
        "LogitMSE": float(torch.mean((z_t - z_b) ** 2)),
        "DeltaMargin": margin(z_t) - margin(z_b),
        "DeltaProb": p_t - p_b,
        "RetainedMass": float(alpha_2d.mean()),
        "Area05": float((alpha_2d > 0.5).mean()),
        "Area90": float((np.searchsorted(cum, 0.9) + 1) / a.size),
        "Iterations": int(res.get("iterations_run", 0)),
    }


@register("cme")
def cme(model, x, background, ctx: Context) -> ExplainResult:
    alpha_2d, mask, area, res = run_cme(model, x, background, ctx)
    extra = fidelity_fields(res, alpha_2d)
    extra["BinaryArea"] = area
    extra["binary_mask"] = mask
    # The article uses the continuous alpha as the CME importance map (Deletion/Insertion ranking).
    return ExplainResult(alpha_2d.astype(np.float32), int(res["target_class"]), extra)


def _gc(method: str):
    def fn(model, x, background, ctx: Context) -> ExplainResult:
        target = predicted_class(model, x)
        sal, cls = gc.importance_map(method, model, x.to(ctx.device), target, arch=ctx.arch.replace("_old", ""))
        return ExplainResult(normalize01(sal), int(cls))
    return fn


for _name in ("cam", "gradcam", "gradcam++", "guided", "guided_gradcam"):
    register(_name)(_gc(_name))


@register("random")
def random_map(model, x, background, ctx: Context) -> ExplainResult:
    """One smooth random map per image (seeded by image id). The article's Random control averages
    `evaluation.random_k` such maps — the evaluator draws the extra ones itself."""
    from cmebench.backgrounds import image_seed

    g = torch.Generator().manual_seed(image_seed(ctx.image_id, 7))
    return ExplainResult(random_smooth_map(tuple(x.shape[-2:]), g), predicted_class(model, x))


@register("cme_adaptive")
def cme_adaptive(model, x, background, ctx: Context) -> ExplainResult:
    """CME with the configuration selected by cmebench/adaptive.py for this architecture (or the
    canonical one when the held-out validation did not justify a change)."""
    from cmebench.store import read_json, run_dir

    sel = read_json(run_dir(ctx.cfg) / "adaptive" / f"{ctx.arch}_selected.json")
    if sel is None:
        raise FileNotFoundError(f"run the 'adaptive' stage for {ctx.arch} first")
    s = sel["selected"]
    w_class = float(s["w_class"])
    cfg = dict(ctx.cfg, cme=dict(ctx.cfg["cme"], weight_class=w_class, weight_blur=round(1.0 - w_class, 6),
                                 weight_sharpness=float(s["w_sharp"]), err_method=s["err_method"], fast=False))
    c2 = Context(ctx.arch, ctx.dataset, ctx.image_id, ctx.mean, ctx.std, ctx.device, ctx.params, cfg, ctx.gen_bg)
    alpha_2d, mask, area, res = run_cme(model, x, background, c2, loss_mode=s["loss_mode"])
    extra = fidelity_fields(res, alpha_2d)
    extra.update({"BinaryArea": area, "binary_mask": mask, "Fallback": int(sel["fallback_to_canonical"])})
    return ExplainResult(alpha_2d.astype(np.float32), int(res["target_class"]), extra)
