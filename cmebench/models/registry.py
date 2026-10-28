"""Model registry: (dataset, arch) -> frozen eval-mode model + its input normalization.

catsdogs   — the article's binary classifiers (Cat=0, Dog=1); weights from weights/ (GitHub Release).
imagenetv2 — the same backbones with their original ImageNet-1k heads (no training); tags pinned here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

from cmebench.core.preprocess import IMAGENET_MEAN, IMAGENET_STD, norm_for_arch
from cmebench.models.greed import load_greed_binary
from cmebench.models.heads import (
    load_mixer_binary,
    load_mobilenet_binary,
    load_resnet50_binary,
    load_vgg19_binary,
)
from cmebench.models.vit import EnsembleModel, load_vit_checkpoint

ARCHS: Dict[str, List[str]] = {
    "catsdogs": ["resnet", "vgg19", "mobilenet", "greed", "vit", "mixer"],
    "imagenetv2": ["resnet", "vgg19", "mobilenet", "vit", "mixer"],  # no ImageNet GREED exists
}

# Article weights (file names kept from the article project).
CATSDOGS_WEIGHTS: Dict[str, str] = {
    "vit": "vit_pretrained_model.pth",
    "resnet": "exp_001_resnet50_binary.pth",
    "mixer": "exp_001_mixer_binary.pth",
    "greed": "exp_001_greed_binary.pth",
    "vgg19": "exp_001_vgg19_binary.pth",
    "mobilenet": "exp_001_mobilenet_binary.pth",
    "mixer_old": "exp_001_mixer_binary_old.pth",  # trained without the neutral class (bias study)
}

# ImageNet-1k weights (pinned tags; the same backbones as the catsdogs models).
IMAGENET_TIMM: Dict[str, str] = {
    "vgg19": "vgg19.tv_in1k",
    "mobilenet": "mobilenetv3_large_100.ra_in1k",
    "mixer": "mixer_b16_224.goog_in21k_ft_in1k",
}
MIXER_IMAGENET_NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # timm pretrained_cfg of the Mixer tag


# Datasets evaluated with the article's Cats-vs-Dogs classifiers.
MODEL_SET = {"catsdogs": "catsdogs", "oxford": "catsdogs", "imagenetv2": "imagenetv2"}


def norm_for(dataset: str, arch: str) -> Tuple[List[float], List[float]]:
    """Input normalization the model was trained with (load-bearing, see core/preprocess.py)."""
    dataset = MODEL_SET.get(dataset, dataset)
    if dataset == "imagenetv2":
        return MIXER_IMAGENET_NORM if arch == "mixer" else (IMAGENET_MEAN, IMAGENET_STD)
    return norm_for_arch(arch.replace("_old", ""))


def num_classes(dataset: str) -> int:
    return 1000 if dataset == "imagenetv2" else 2


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_model(dataset: str, arch: str, device: torch.device, weights_dir: str | Path = "weights") -> nn.Module:
    """Return a frozen eval-mode model on `device`. Parameters have requires_grad=False; methods that
    need input gradients set requires_grad on the input, which works with frozen parameters."""
    weights_dir = Path(weights_dir)
    dataset = MODEL_SET.get(dataset, dataset)
    if dataset == "catsdogs":
        path = weights_dir / CATSDOGS_WEIGHTS[arch]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run scripts/20_get_models.py")
        base = arch.replace("_old", "")
        loaders = {
            "vit": lambda p: load_vit_checkpoint(p, device=device, num_classes=2)[0],
            "resnet": lambda p: load_resnet50_binary(p, device),
            "mixer": lambda p: load_mixer_binary(p, device),
            "greed": lambda p: load_greed_binary(p, device),
            "vgg19": lambda p: load_vgg19_binary(p, device),
            "mobilenet": lambda p: load_mobilenet_binary(p, device),
        }
        return _freeze(loaders[base](path).to(device))

    if dataset == "imagenetv2":
        if arch == "resnet":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif arch == "vit":
            # Same wrapper as the catsdogs ViT so every method resolves layers identically (model.vit...).
            model = EnsembleModel(num_classes=1000)
            model.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        elif arch in IMAGENET_TIMM:
            model = timm.create_model(IMAGENET_TIMM[arch], pretrained=True)
        else:
            raise ValueError(f"No ImageNet model for arch {arch!r}")
        return _freeze(model.to(device))

    raise ValueError(f"Unknown dataset {dataset!r}")
