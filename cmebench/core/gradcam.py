# Ported from cat_vs_dog/exp_001_gradcam_common.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""Shared Grad-CAM family implementations for the CME-vs-Grad-CAM comparison.

This module is hand-rolled (no `pytorch-grad-cam` dependency) so that:
  * the output is always a single normalized per-pixel importance map [H, W] in
    [0, 1] -- the "common importance interface" required by ``metrics_common`` and
    described in doc/comparison_protocol.md (section 1);
  * the math is transparent for the paper (doc/Grad-CAM.md, section 2);
  * we keep full control over the ViT adaptation (doc/Grad-CAM.md, section 1.1/3.3).

Supported methods (rows of the comparison table, doc/comparison_protocol.md section 4):
  * GradCAM            -- coarse, faithful (passes sanity checks).
  * GradCAMpp          -- Grad-CAM++ weights.
  * GuidedBackprop     -- pixel-level saliency (see caveat below).
  * GuidedGradCAM      -- GradCAM x GuidedBackprop.

CAVEAT (doc/Grad-CAM.md section 2.4): GuidedBackprop is defined for ReLU networks.
ResNet50 uses nn.ReLU modules -> it works. ViT uses GELU and has no nn.ReLU modules,
so guided backprop silently degrades to a vanilla input gradient there. This is
honest and intentional -- it is exactly the limitation we want to measure, not hide.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _normalize_map(x: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2D map to [0, 1]."""
    x = x - x.min()
    denom = max(float(x.max()), 1e-8)
    return (x / denom).astype(np.float32)


def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Turn ViT token sequence (B, 1+N, D) into a CNN-style map (B, D, H, W).

    Drops the leading [CLS] token and reshapes the N patch tokens into an HxW grid.
    For 224x224 input with patch 16 -> N = 14*14 = 196 (sequence length 197).
    If you change the input size, recompute height/width (e.g. 384 -> 24).
    """
    # tensor: (B, 1 + H*W, D)
    spatial = tensor[:, 1:, :]
    b, n, d = spatial.shape
    if n != height * width:
        raise ValueError(
            f"vit_reshape_transform: token count {n} != {height}*{width}={height * width}. "
            "Adjust height/width to match the input resolution."
        )
    result = spatial.reshape(b, height, width, d)
    return result.permute(0, 3, 1, 2).contiguous()


def mixer_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Turn an MLP-Mixer token sequence (B, N, D) into a CNN-style map (B, D, H, W).

    Mixer is attention-free but keeps patch tokens in a fixed grid with NO class token; for 224x224
    patch-16 input N = 14*14 = 196. CAM-family thus applies via the same token->2D reshape as ViT.
    Precedent: LibraGrad (Mehri et al., 2024, arXiv:2411.16760) evaluates Grad-CAM-family on MLP-Mixer.
    """
    b, n, d = tensor.shape
    if n != height * width:
        raise ValueError(
            f"mixer_reshape_transform: token count {n} != {height}*{width}={height * width}. "
            "Adjust height/width to match the input resolution."
        )
    result = tensor.reshape(b, height, width, d)
    return result.permute(0, 3, 1, 2).contiguous()


def get_resnet_target_layer(model: nn.Module) -> nn.Module:
    """Last bottleneck block of ResNet50 (works whether or not it is wrapped)."""
    base = getattr(model, "resnet", model)
    base = getattr(base, "backbone", base)
    return base.layer4[-1]


def get_vit_target_layer(model: nn.Module) -> nn.Module:
    """LayerNorm before self-attention in the last ViT encoder block.

    Handles the project's EnsembleModel wrapper (model.vit.encoder...) as well as a
    bare torchvision vit_b_16 (model.encoder...). See doc/Grad-CAM.md section 3.3 and
    the 'why ln_1' note in doc/Grad-CAM_4_ViT/Grad-CAM_4_ViT.md.
    """
    base = getattr(model, "vit", model)
    return base.encoder.layers[-1].ln_1


def get_greed_target_layer(model: nn.Module) -> nn.Module:
    """Honest Grad-CAM target for GREED: the last 32-channel conv before the class maps (7x7).

    GREED tail: concentrate_out_c(32) -> concentrate_out_d(32) -> concentrate_out_e(2 class maps)
    -> GAP. concentrate_out_d is the richest spatial feature layer for a standard Grad-CAM.
    """
    return model.concentrate_out_d


def get_vgg_target_layer(model: nn.Module) -> nn.Module:
    """Last conv in VGG's feature extractor (512ch, 7x7) — standard VGG Grad-CAM target. timm vgg19.

    (Classic CAM is N/A for VGG: the head is FC-based, not GAP->single-Linear — only Grad-CAM family.)
    """
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    return convs[-1]


def get_mobilenet_target_layer(model: nn.Module) -> nn.Module:
    """Last MobileNetV3 block's output (spatial features ~960x7x7, before global_pool). timm.

    (Classic CAM is N/A: after GAP the head is conv_head->act->classifier, not a single Linear —
    only Grad-CAM family.)
    """
    return model.blocks[-1]


def get_mixer_target_layer(model: nn.Module) -> nn.Module:
    """Final token LayerNorm of MLP-Mixer (output (B, 196, D)) — Grad-CAM target via reshape. timm.

    Used with mixer_reshape_transform (token->14x14). Classic CAM is N/A (no GAP->single-Linear).
    NB: gradient attributions on attention-free Mixer are known fragile (cf. LibraGrad) — same caveat
    as ViT; treat Mixer Grad-CAM as a measurable-but-fragile baseline, not ground truth.
    """
    return model.norm


# --------------------------------------------------------------------------- #
# CAM core (Grad-CAM and Grad-CAM++)
# --------------------------------------------------------------------------- #
class _HookedCAM:
    """Base class: captures activations and gradients of a single target layer."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model.eval()
        self.reshape_transform = reshape_transform
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out) -> None:
        self._activations = out

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "_HookedCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    # weights computed by subclasses ---------------------------------------- #
    def _weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        device = _model_device(self.model)
        # Require grad on the INPUT so the target layer's output requires grad too. Without
        # this, a frozen backbone (requires_grad=False everywhere) yields a non-grad activation
        # and the backward hook never fires -> gradients stay None.
        input_tensor = input_tensor.to(device).clone().detach().requires_grad_(True)

        # Inplace activations (GREED's nn.ReLU(inplace=True), MobileNet's Hardswish/ReLU(inplace=True),
        # …) overwrite the hooked layer's output in place, which corrupts the saved activation and
        # breaks full_backward_hook ("view is being modified inplace"). Disable inplace on ANY module
        # exposing `inplace=True` for this pass, then restore.
        inplace_mods = [mod for mod in self.model.modules() if getattr(mod, "inplace", False)]
        for r in inplace_mods:
            r.inplace = False
        try:
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[:, target_class].sum().backward()
        finally:
            for r in inplace_mods:
                r.inplace = True

        activations = self._activations
        gradients = self._gradients
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        activations = activations.detach()
        gradients = gradients.detach()

        weights = self._weights(activations, gradients)  # (B, C, 1, 1)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (B, 1, h, w)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam_map = cam.squeeze().cpu().numpy()
        return _normalize_map(cam_map), target_class


class GradCAM(_HookedCAM):
    """Classic Grad-CAM (Selvaraju et al., 2017): weights = GAP of gradients."""

    def _weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        return gradients.mean(dim=(2, 3), keepdim=True)


class GradCAMpp(_HookedCAM):
    """Grad-CAM++ (Chattopadhay et al., 2018), jacobgil-style approximation.

    Uses only first-order gradients to approximate the higher-order terms (valid
    under the common exp-score assumption); documented as an approximation.
    """

    def _weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        grad2 = gradients.pow(2)
        grad3 = grad2 * gradients
        sum_act = activations.sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2.0 * grad2 + sum_act * grad3
        alpha_denom = torch.where(
            alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom)
        )
        alpha = grad2 / (alpha_denom + 1e-8)
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        return weights


# --------------------------------------------------------------------------- #
# Classic CAM (Zhou et al., 2016) — CNN with GAP + single Linear only
# --------------------------------------------------------------------------- #
class CAM:
    """Classic Class Activation Mapping.

    CAM_c = ReLU( sum_k w[c,k] * A_k ), with A = last conv feature maps and w = fc weights.
    Valid ONLY for CNNs that end in global-average-pool -> one Linear (e.g. ResNet50). It is
    NOT applicable to ViT (no conv GAP + fc) — that architectural restriction is itself a result.
    No gradients needed (forward-only).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        base = getattr(model, "resnet", model)
        if not (hasattr(base, "layer4") and isinstance(getattr(base, "fc", None), nn.Linear)):
            raise ValueError("CAM requires a CNN with `layer4` + Linear `fc` (e.g. ResNet50).")
        self.fc = base.fc
        self.target_layer = base.layer4[-1]
        self.activations: Optional[torch.Tensor] = None
        self._handle = self.target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out) -> None:
        self.activations = out

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "CAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        device = _model_device(self.model)
        x = input_tensor.to(device)
        with torch.no_grad():
            logits = self.model(x)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            acts = self.activations  # (1, C, h, w)
            weights = self.fc.weight[target_class]  # (C,)
            cam = F.relu((weights.view(1, -1, 1, 1) * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
            cam_map = cam.squeeze().cpu().numpy()
        return _normalize_map(cam_map), target_class


def greed_cam(
    model: nn.Module, input_tensor: torch.Tensor, target_class: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Classic CAM for GREED = its BUILT-IN class map: ReLU(concentrate_out_e)[target_class].

    GREED's head is conv1x1(32->2) + GAP, so logit_c = GAP(ReLU(concentrate_out_e)_c): the class
    channel IS the class activation map (classic CAM in its purest form — no fc weights to fold).
    Forward-only. The map is 7x7, upsampled to the input size (coarse — that is CAM's known limit).
    """
    model = model.eval()
    device = _model_device(model)
    x = input_tensor.to(device)
    captured = {}
    handle = model.concentrate_out_e.register_forward_hook(
        lambda m, i, o: captured.__setitem__("a", o)
    )
    try:
        with torch.no_grad():
            logits = model(x)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            class_maps = F.relu(captured["a"])  # (1, 2, h, w) non-negative class maps
            cam = class_maps[:, target_class : target_class + 1]  # (1, 1, h, w)
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
            cam_map = cam.squeeze().cpu().numpy()
    finally:
        handle.remove()
    return _normalize_map(cam_map), target_class


# --------------------------------------------------------------------------- #
# Guided Backpropagation  (ReLU-only; degrades to vanilla grad on ViT)
# --------------------------------------------------------------------------- #
class _GuidedBackpropReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        positive = (inp > 0).type_as(inp)
        out = inp * positive
        ctx.save_for_backward(inp, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, _ = ctx.saved_tensors
        pos_in = (inp > 0).type_as(grad_output)
        pos_grad = (grad_output > 0).type_as(grad_output)
        return grad_output * pos_in * pos_grad


class _GuidedReLUModule(nn.Module):
    def forward(self, x):  # noqa: D401
        return _GuidedBackpropReLU.apply(x)


class GuidedBackprop:
    """Guided backprop saliency. Swaps nn.ReLU modules with a guided variant.

    On models without nn.ReLU modules (e.g. ViT with GELU) NO swap happens and this
    becomes a plain input-gradient saliency -- see module docstring / Grad-CAM.md 2.4.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        self._swapped: List[Tuple[nn.Module, str, nn.Module]] = []
        self._swap_relus(self.model)

    def _swap_relus(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                self._swapped.append((module, name, child))
                setattr(module, name, _GuidedReLUModule())
            else:
                self._swap_relus(child)

    def restore(self) -> None:
        for parent, name, original in self._swapped:
            setattr(parent, name, original)
        self._swapped = []

    def __enter__(self) -> "GuidedBackprop":
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    def saliency(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        device = _model_device(self.model)
        x = input_tensor.to(device).clone().detach().requires_grad_(True)

        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[:, target_class].sum().backward()

        grad = x.grad.detach()[0]  # (C, H, W)
        sal = grad.abs().amax(dim=0).cpu().numpy()  # (H, W)
        return _normalize_map(sal), target_class


def guided_gradcam_map(cam_map: np.ndarray, guided_map: np.ndarray) -> np.ndarray:
    """Pointwise product of an (already 224x224) Grad-CAM map and guided saliency."""
    return _normalize_map(cam_map * guided_map)


# --------------------------------------------------------------------------- #
# Convenience dispatcher
# --------------------------------------------------------------------------- #
def importance_map(
    method: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    arch: str = "resnet",
) -> Tuple[np.ndarray, int]:
    """Return a normalized [H, W] importance map for the requested method.

    method: 'cam' | 'gradcam' | 'gradcam++' | 'guided' | 'guided_gradcam'
    arch:   'resnet' | 'vit' | 'greed'  (selects target layer / reshape; 'guided' works for any arch).
            GREED is CAM-native: 'cam' reads its class map (concentrate_out_e), Grad-CAM hooks
            concentrate_out_d.
    """
    method = method.lower()
    # CAM is handled first: it finds its own layer internally and must reject ViT cleanly
    # (before any arch-specific target-layer resolution that assumes a ViT/ResNet structure).
    if method == "cam":
        # GREED's head IS a CAM (conv1x1->class maps->GAP): read its native class map.
        if arch == "greed":
            return greed_cam(model, input_tensor, target_class)
        if arch != "resnet":
            raise ValueError(
                "CAM (classic) is only for CNNs with GAP+fc (ResNet50) or GREED's native head; N/A for ViT."
            )
        with CAM(model) as cam:
            return cam(input_tensor, target_class)

    # Guided backprop does not require a target feature layer, so it can be run on
    # architectures beyond resnet/vit (e.g. MLP-Mixer). On non-ReLU nets it degrades
    # to vanilla input-gradient saliency (documented behavior).
    if method == "guided":
        with GuidedBackprop(model) as gb:
            return gb.saliency(input_tensor, target_class)

    if arch == "vit":
        target_layer = get_vit_target_layer(model)
        reshape = vit_reshape_transform
    elif arch == "resnet":
        target_layer = get_resnet_target_layer(model)
        reshape = None
    elif arch == "greed":
        target_layer = get_greed_target_layer(model)  # honest Grad-CAM at concentrate_out_d
        reshape = None
    elif arch == "vgg19":
        target_layer = get_vgg_target_layer(model)  # last conv in features (512x7x7)
        reshape = None
    elif arch == "mobilenet":
        target_layer = get_mobilenet_target_layer(model)  # last block (960x7x7)
        reshape = None
    elif arch == "mixer":
        target_layer = get_mixer_target_layer(model)  # final token LayerNorm (B,196,D)
        reshape = mixer_reshape_transform
    else:
        raise ValueError(f"Unknown arch: {arch!r}")

    if method in ("gradcam", "grad-cam"):
        with GradCAM(model, target_layer, reshape) as cam:
            return cam(input_tensor, target_class)
    if method in ("gradcam++", "grad-cam++", "gradcampp"):
        with GradCAMpp(model, target_layer, reshape) as cam:
            return cam(input_tensor, target_class)
    if method in ("guided_gradcam", "guided-grad-cam"):
        with GradCAM(model, target_layer, reshape) as cam:
            cam_map, cls = cam(input_tensor, target_class)
        with GuidedBackprop(model) as gb:
            guided, _ = gb.saliency(input_tensor, cls)
        return guided_gradcam_map(cam_map, guided), cls
    raise ValueError(f"Unknown method: {method!r}")
