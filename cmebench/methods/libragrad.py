"""LibraGrad (Mehri, Soleymani Baghshah & Pilehvar, CVPR 2025) + Input x Gradient on tokens.

LibraGrad leaves the forward pass unchanged and rebalances the backward pass of a Transformer
(rules as in the authors' self-contained reference implementation, NightMachinery/LibraGrad,
notebooks/llama3.ipynb):
  * gated activations (GELU = x * Phi(x)): the gate Phi(x) is detached;
  * LayerNorm: the normalizing denominator (std) is detached;
  * attention: queries and keys are detached (gradient flows through the values only);
  * element-wise products of two streams: gradient halved (not present in ViT-B/16 / Mixer-B/16 MLPs).
The base attribution is Input x Gradient of the target logit w.r.t. the token embeddings entering
the first block (sum over the embedding dimension; ViT's CLS token dropped), reshaped to 14x14 and
upsampled bilinearly. Applies to ViT (torchvision) and MLP-Mixer (timm).
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmebench.methods.base import Context, ExplainResult, normalize01, register


class LibraGELU(nn.Module):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == "tanh":
            gate = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
        else:
            gate = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return x * gate.detach()


class LibraLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm) -> None:
        super().__init__()
        self.ln = ln

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.ln.normalized_shape), 0))
        mu = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        y = (x - mu) / torch.sqrt(var + self.ln.eps).detach()
        if self.ln.elementwise_affine:
            y = y * self.ln.weight + self.ln.bias
        return y


@contextlib.contextmanager
def libra_mode(model: nn.Module):
    """Swap GELU / LayerNorm modules and patch scaled_dot_product_attention; restore on exit."""
    swapped = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.GELU):
                new = LibraGELU(child.approximate)
            elif isinstance(child, nn.LayerNorm):
                new = LibraLayerNorm(child)
            else:
                continue
            setattr(parent, name, new)
            swapped.append((parent, name, child))
    sdpa = F.scaled_dot_product_attention

    def sdpa_libra(query, key, value, *args, **kwargs):
        return sdpa(query.detach(), key.detach(), value, *args, **kwargs)

    F.scaled_dot_product_attention = sdpa_libra
    try:
        yield model
    finally:
        F.scaled_dot_product_attention = sdpa
        for parent, name, child in swapped:
            setattr(parent, name, child)


def _first_block(model: nn.Module, arch: str) -> nn.Module:
    if arch == "vit":
        return getattr(model, "vit", model).encoder.layers[0]
    if arch == "mixer":
        return model.blocks[0]
    raise ValueError(f"LibraGrad is implemented for vit / mixer, not {arch!r}")


@register("libragrad")
def libragrad(model, x, background, ctx: Context) -> ExplainResult:
    arch = ctx.arch.replace("_old", "")
    x = x.to(ctx.device)
    with torch.no_grad():
        ref_logits = model(x)
    target = int(ref_logits.argmax(dim=1).item())
    store = {}

    def pre_hook(mod, inputs):
        t = inputs[0]
        t.requires_grad_(True)
        t.retain_grad()
        store["tokens"] = t

    h = _first_block(model, arch).register_forward_pre_hook(pre_hook)
    try:
        with libra_mode(model), torch.enable_grad():
            logits = model(x.clone().requires_grad_(True))
            logits[0, target].backward()
    finally:
        h.remove()
    tok = store["tokens"]
    attr = (tok * tok.grad).sum(dim=-1)[0].detach()        # (n_tokens,)
    if arch == "vit":
        attr = attr[1:]                                      # drop CLS
    side = int(round(math.sqrt(attr.numel())))
    amap = attr.view(1, 1, side, side)
    amap = F.interpolate(amap, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    fwd_dev = float((logits.detach() - ref_logits).abs().max())  # forward must be unchanged (~0)
    return ExplainResult(normalize01(amap.cpu().numpy()), target, {"ForwardMaxAbsDiff": fwd_dev})
