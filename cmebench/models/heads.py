# Ported from cat_vs_dog/exp_001_train_{resnet50_binary,mixer,mobilenet,vgg19}.py ("model layer" blocks).
"""Binary (Cat=0, Dog=1) heads on frozen ImageNet backbones: builders + checkpoint loaders."""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

CLASS_NAMES = ["Cat", "Dog"]  # index 0 = Cat, 1 = Dog (do not reorder)
MIXER_MODEL_NAME = "mixer_b16_224"
MOBILENET_MODEL_NAME = "mobilenetv3_large_100"
VGG19_MODEL_NAME = "vgg19"


def build_resnet50_binary(pretrained: bool = True) -> nn.Module:
    """ResNet50 with frozen backbone and a fresh binary head (Cat=0, Dog=1)."""
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    for p in model.parameters():
        p.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # trainable head
    return model


def load_resnet50_binary(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    """Reload the trained binary ResNet50 for the comparison notebook."""
    device = torch.device(device)
    model = build_resnet50_binary(pretrained=False)
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model.eval().to(device)


def build_mixer_binary(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    model = timm.create_model(MIXER_MODEL_NAME, pretrained=pretrained, num_classes=len(CLASS_NAMES))

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if name.startswith("head"):
                p.requires_grad_(True)

    return model


def load_mixer_binary(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    device = torch.device(device)
    model = build_mixer_binary(pretrained=False, freeze_backbone=False)
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model.eval().to(device)


def build_mobilenet_binary(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    model = timm.create_model(MOBILENET_MODEL_NAME, pretrained=pretrained, num_classes=len(CLASS_NAMES))

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
        # Розморожуємо ЛИШЕ класифікаційну голову через timm-API get_classifier() (у timm
        # MobileNetV3 вона зветься 'classifier', тож name.startswith('head') тут НЕ спрацював би).
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)

    return model


def load_mobilenet_binary(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    device = torch.device(device)
    model = build_mobilenet_binary(pretrained=False, freeze_backbone=False)
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model.eval().to(device)


def build_vgg19_binary(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    model = timm.create_model(VGG19_MODEL_NAME, pretrained=pretrained, num_classes=len(CLASS_NAMES))

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
        # Розморожуємо ЛИШЕ класифікаційну голову через timm-API get_classifier() (робастно для
        # будь-якої архітектури; пряме name.startswith('head') ненадійне для не-Mixer моделей).
        for p in model.get_classifier().parameters():
            p.requires_grad_(True)

    return model


def load_vgg19_binary(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    device = torch.device(device)
    model = build_vgg19_binary(pretrained=False, freeze_backbone=False)
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model.eval().to(device)
