# Ported from cat_vs_dog/vit_model_loader.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torchvision import models


class EnsembleModel(nn.Module):
    """ViT-B/16 classifier with a binary head.

    This mirrors the model shape used in archive/notebooks_legacy/acc-100-ensemble-pytorch.ipynb.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.vit = models.vit_b_16(weights=None)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


def _sanitize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize common checkpoint key prefixes."""
    sanitized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            sanitized[key[len("module.") :]] = value
        else:
            sanitized[key] = value
    return sanitized


def load_vit_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    num_classes: int = 2,
) -> Tuple[nn.Module, torch.device]:
    """Load ViT model from a .pth checkpoint and move it to device.

    Returns (model, device).
    """
    resolved_device = torch.device(device)
    model = EnsembleModel(num_classes=num_classes)

    checkpoint = torch.load(str(checkpoint_path), map_location=resolved_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _sanitize_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[load_vit_checkpoint] Non-strict load report")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    model.to(resolved_device)
    model.eval()
    return model, resolved_device
