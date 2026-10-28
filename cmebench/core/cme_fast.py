# Ported from cat_vs_dog/exp_001_cgd_fast.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""exp_001 — FAST variant of the CME optimizer. Canonical `exp_001_cgd.py` is UNTOUCHED.

Same math as `exp_001_cgd.optimize_alpha_ext` (same losses, same config, same return contract),
only the EXECUTION is optimized. Implemented speedups (all numerically equivalent unless noted):

  1. Frozen model weights — `requires_grad_(False)` for the whole net during optimization, so
     backward() computes gradients ONLY for alpha, not for every ViT/ResNet weight. Alpha
     gradients are identical (weight grads were computed but never used by the reference).
  2. Cached Sobel kernels — the reference rebuilds the 5x5 kernels on every iteration
     (`vit_degradation._build_sobel_kernels`); here they are built once per (C, device, dtype).
  3. Deferred loss sync — the reference calls `.item()` 4x per iteration (a GPU->CPU sync each
     step); here losses are kept as GPU tensors and flushed every `sync_every` steps in one sync.
     ⚠️ With early stopping this delays the stop check to the next flush boundary: the stop is
     detected at the same iteration, but alpha has advanced up to `sync_every-1` extra steps.
     `sync_every=1` reproduces the reference exactly.
  4. Optional AMP (`use_amp=True`) — model forward in bf16/fp16 autocast, losses/alpha in fp32.
     NOT bit-exact (fp rounding); OFF by default. Canonical maps should stay fp32.
  5. Optional `enable_fast_backends()` — cudnn.benchmark + TF32 matmul. May introduce tiny
     run-to-run differences; opt-in from the entry-point, never a side effect of import.

Config (CGD_CONFIG / CGD_LOSS_MODE / CGD_MAX_AREA), `binarize_alpha_2d`, early-stop logic and
`_class_loss` are IMPORTED from exp_001_cgd — the single source of truth. The diff between the
two optimizers is therefore exactly the speedups above.

Equivalence & speed check: `exp_001_bench_cgd.py` (reference vs fast on the same eval images).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cmebench.core.cme import (
    CGD_CONFIG,
    CGD_LOSS_MODE,
    CGD_MAX_AREA,
    EarlyStopConfig,
    LossMode,
    _class_loss,
    _EarlyStopper,
    binarize_alpha_2d,
)
from cmebench.core.degradation import DegradationConfig, _build_sobel_kernels

__all__ = [
    "CGD_CONFIG",
    "CGD_LOSS_MODE",
    "CGD_MAX_AREA",
    "EarlyStopConfig",
    "binarize_alpha_2d",
    "enable_fast_backends",
    "optimize_alpha_ext_fast",
    "cgd_importance_fast",
]

# --------------------------------------------------------------------------- #
# (2) Sobel kernel cache: identical tensors to _build_sobel_kernels, built once.
# --------------------------------------------------------------------------- #
_SOBEL_CACHE: Dict[Tuple[int, str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def _sobel_kernels_cached(channels: int, device: torch.device, dtype: torch.dtype):
    key = (channels, str(device), dtype)
    if key not in _SOBEL_CACHE:
        _SOBEL_CACHE[key] = _build_sobel_kernels(channels, device, dtype)
    return _SOBEL_CACHE[key]


def _smoothness_loss_fast(alpha: torch.Tensor, err_method: str) -> torch.Tensor:
    """Same math as vit_degradation._smoothness_loss, with cached kernels."""
    channels = alpha.shape[1]
    kx, ky = _sobel_kernels_cached(channels, alpha.device, alpha.dtype)
    gx = F.conv2d(alpha, kx, padding=1, groups=channels)
    gy = F.conv2d(alpha, ky, padding=1, groups=channels)
    if err_method == "abs":
        return gx.abs().mean() + gy.abs().mean()
    return gx.square().mean() + gy.square().mean()


def enable_fast_backends() -> None:
    """(5) Opt-in: cudnn autotune + TF32. May cause tiny (last-bit) numeric differences."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class _frozen_params:
    """(1) Temporarily set requires_grad=False on all model params; restore on exit."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.saved: List[Tuple[torch.nn.Parameter, bool]] = []

    def __enter__(self):
        self.saved = [(p, p.requires_grad) for p in self.model.parameters()]
        for p, _ in self.saved:
            p.requires_grad_(False)
        return self

    def __exit__(self, *exc):
        for p, flag in self.saved:
            p.requires_grad_(flag)
        return False


def optimize_alpha_ext_fast(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    config: DegradationConfig,
    loss_mode: LossMode = "logits",
    early_stop: Optional[EarlyStopConfig] = None,
    use_amp: bool = False,
    sync_every: int = 25,
    base: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
) -> Dict[str, object]:
    """Drop-in replacement for exp_001_cgd.optimize_alpha_ext (same args + same return dict).

    Extra args: `use_amp` (half-precision forward, not bit-exact) and `sync_every` (loss flush
    period; 1 == reference behavior, larger == fewer GPU syncs; only matters with early_stop).
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    amp_device = "cuda" if image_tensor.is_cuda else "cpu"
    amp_dtype = (
        torch.bfloat16
        if (amp_device == "cpu" or torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    with _frozen_params(model):
        with torch.no_grad():
            target_logits = model(image_tensor).detach()
            if target_class is None:
                target_class = int(target_logits.argmax(dim=1).item())

        # Фон B — середній колір зображення (канонічний baseline, як у reference); cme-bench: або заданий.
        if base is None:
            base = image_tensor.mean(dim=(2, 3), keepdim=True).expand_as(image_tensor)
        else:
            base = base.to(device).expand_as(image_tensor)

        alpha_logits = torch.nn.Parameter(
            torch.full_like(image_tensor, fill_value=float(config.init_alpha_logit), device=device)
        )
        optimizer = torch.optim.Adam([alpha_logits], lr=config.lr)

        history = {"step": [], "class_loss": [], "smooth_loss": [], "energy_loss": [], "total_loss": []}
        # (3) GPU-side loss buffer: flushed to `history` (one sync) every `sync_every` steps.
        buffer: List[torch.Tensor] = []  # each entry: stacked (class, smooth, energy, total)

        stopper = _EarlyStopper(early_stop) if early_stop is not None else None
        stop_reason: Optional[str] = None
        iterations_run = config.iterations
        steps_done = 0

        def flush() -> bool:
            """Move buffered losses to history; feed the stopper. True -> stop fired."""
            nonlocal stop_reason, iterations_run
            if not buffer:
                return False
            start = len(history["step"])
            values = torch.stack(buffer).cpu().tolist()  # single GPU->CPU sync
            buffer.clear()
            for offset, (c, s, e, t) in enumerate(values):
                step_idx = start + offset
                history["step"].append(step_idx)
                history["class_loss"].append(c)
                history["smooth_loss"].append(s)
                history["energy_loss"].append(e)
                history["total_loss"].append(t)
                if stopper is not None and stop_reason is None:
                    stop_reason = stopper.update(t)
                    if stop_reason is not None:
                        iterations_run = step_idx + 1
            return stop_reason is not None

        for step in range(config.iterations):
            alpha = torch.sigmoid(alpha_logits)
            mixed = base * (1.0 - alpha) + image_tensor * alpha

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                logits = model(mixed)
            logits = logits.float()  # losses always in fp32

            class_loss = _class_loss(logits, target_logits, loss_mode, target_class)
            smooth_loss = _smoothness_loss_fast(alpha, config.err_method)
            energy_loss = alpha.square().mean()

            reg = config.weight_sharpness * smooth_loss + (1.0 - config.weight_sharpness) * energy_loss
            total_loss = config.weight_class * class_loss + config.weight_blur * reg

            buffer.append(
                torch.stack([class_loss.detach(), smooth_loss.detach(), energy_loss.detach(), total_loss.detach()])
            )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps_done = step + 1

            if stopper is not None and steps_done % max(1, sync_every) == 0 and flush():
                break

        flush()
        if stop_reason is None:
            iterations_run = steps_done

        with torch.no_grad():
            alpha = torch.sigmoid(alpha_logits)
            mixed = base * (1.0 - alpha) + image_tensor * alpha
            final_logits = model(mixed)

    return {
        "alpha": alpha.detach().cpu(),
        "final_logits": final_logits.detach().float().cpu(),
        "target_logits": target_logits.detach().cpu(),
        "target_class": target_class,
        "history": history,
        "iterations_run": iterations_run,
        "stop_reason": stop_reason,
    }


def cgd_importance_fast(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    config: DegradationConfig,
    loss_mode: LossMode = CGD_LOSS_MODE,
    max_area: float = CGD_MAX_AREA,
    early_stop: Optional[EarlyStopConfig] = None,
    use_amp: bool = False,
    sync_every: int = 25,
    base: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
):
    """Fast counterpart of exp_001_cgd.cgd_importance (ext variant only).

    Returns (alpha_2d [H,W] float32, cgd_mask bool [H,W], area_frac, raw_result).
    """
    res = optimize_alpha_ext_fast(
        model, image_tensor, device, config, loss_mode,
        early_stop=early_stop, use_amp=use_amp, sync_every=sync_every,
        base=base, target_class=target_class,
    )
    alpha_2d = res["alpha"].mean(dim=1).squeeze(0).numpy().astype(np.float32)
    mask, _ = binarize_alpha_2d(alpha_2d, max_area)
    return alpha_2d, mask, float(mask.mean()), res
