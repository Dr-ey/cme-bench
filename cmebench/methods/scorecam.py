"""Score-CAM (Wang et al., CVPRW 2020) on the same target layers as the article's Grad-CAM.

Each activation channel A_k (reshaped to 2-D for ViT / Mixer exactly as for Grad-CAM) is upsampled
and min-max normalized to a mask H_k; the class score of x * H_k + B * (1 - H_k) gives the channel
weight; weights are softmax-normalized over channels (as in the pytorch-grad-cam reference) and
S = ReLU(sum_k w_k A_k), upsampled. The original uses B = 0 in model space; here B is configurable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cmebench.core import gradcam as gc
from cmebench.methods.base import Context, ExplainResult, normalize01, predicted_class, register

_LAYER = {
    "resnet": (gc.get_resnet_target_layer, None),
    "vit": (gc.get_vit_target_layer, gc.vit_reshape_transform),
    "greed": (gc.get_greed_target_layer, None),
    "vgg19": (gc.get_vgg_target_layer, None),
    "mobilenet": (gc.get_mobilenet_target_layer, None),
    "mixer": (gc.get_mixer_target_layer, gc.mixer_reshape_transform),
}


@register("scorecam")
def scorecam(model, x, background, ctx: Context) -> ExplainResult:
    x = x.to(ctx.device)
    bg = background.to(ctx.device) if background is not None else torch.zeros_like(x)
    get_layer, reshape = _LAYER[ctx.arch.replace("_old", "")]
    store = {}
    h = get_layer(model).register_forward_hook(lambda mod, i, o: store.__setitem__("a", o.detach()))
    try:
        with torch.no_grad():
            target = int(model(x).argmax(dim=1).item())
    finally:
        h.remove()
    a = store["a"]
    a = reshape(a) if reshape is not None else a                      # (1, K, h, w)
    up = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
    mn = up.flatten(1).min(dim=1).values.view(-1, 1, 1)
    mx = up.flatten(1).max(dim=1).values.view(-1, 1, 1)
    masks = ((up - mn) / (mx - mn).clamp_min(1e-8)).unsqueeze(1)      # (K, 1, H, W)
    scores = []
    bs = int(ctx.params.get("batch", 32))
    with torch.no_grad():
        for i in range(0, masks.shape[0], bs):
            m = masks[i:i + bs]
            scores.append(torch.softmax(model(x * m + bg * (1.0 - m)), dim=1)[:, target])
    w = torch.softmax(torch.cat(scores), dim=0).view(1, -1, 1, 1)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return ExplainResult(normalize01(cam.cpu().numpy()), target, {"Channels": int(masks.shape[0])})
