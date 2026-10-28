# Ported from cat_vs_dog/exp_001_train_greed.py ("model layer" block).
"""GREED: compact residual CNN + greedy class-map head (conv1x1 -> ReLU -> GAP). Input raw [0,1]."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class GreedNet(nn.Module):
    def __init__(self, low_filters: int = 32, high_filters: int = 64, res_blocks: int = 5) -> None:
        super().__init__()
        self.res_blocks = res_blocks
        # Input image 224x224x3 -> convert_to_inner -> 224x224x32 -> res_blocks -> 7x7x32 -> concentrate_out_* -> 7x7x2 -> GAP -> 2
        self.convert_to_inner = nn.Conv2d(3, low_filters, kernel_size=5, padding=2)
        self.init_bn = nn.BatchNorm2d(low_filters)

        self.convolution = nn.ModuleList(
            [nn.Conv2d(low_filters, high_filters, kernel_size=5, padding=2) for _ in range(res_blocks)]
        )
        self.concentrate = nn.ModuleList(
            [nn.Conv2d(high_filters, low_filters, kernel_size=1, padding=0) for _ in range(res_blocks)]
        )
        self.normalisation = nn.ModuleList([nn.BatchNorm2d(low_filters) for _ in range(res_blocks)])

        # TensorFlow used padding='same' with kernel=2. In PyTorch, use padding='same' to keep HxW.
        self.concentrate_out_c = nn.Conv2d(low_filters, 32, kernel_size=2, padding="same")
        self.concentrate_out_d = nn.Conv2d(32, 32, kernel_size=2, padding="same")
        self.concentrate_out_e = nn.Conv2d(32, 2, kernel_size=1, padding=0) # final output: 2 channels (Cat=0, Dog=1) - include attention map (7x7). Greed features from architecture

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.relu(self.convert_to_inner(x))
        net = self.init_bn(net)

        for i in range(self.res_blocks):
            start = self.pool(net)
            net = self.relu(self.convolution[i](net))
            net = self.relu(self.concentrate[i](net))
            net = self.pool(net)
            net = self.normalisation[i](net)
            net = net + start

        net = self.relu(self.concentrate_out_c(net))
        net = self.relu(self.concentrate_out_d(net))
        net = self.relu(self.concentrate_out_e(net))
        net = self.gap(net).flatten(1)
        return net


def load_greed_binary(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    device = torch.device(device)
    model = GreedNet()
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model.eval().to(device)
