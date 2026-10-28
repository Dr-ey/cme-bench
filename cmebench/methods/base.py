"""Uniform explainer interface + registry.

Every method returns one normalized saliency map S in [0,1]^(H,W) (higher = more important) and an
`extra` dict of scalars/arrays stored alongside the map. Evaluation never sees how S was produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch


@dataclass
class ExplainResult:
    saliency: np.ndarray                       # [H,W] float32 in [0,1]
    target_class: int
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class Context:
    arch: str
    dataset: str
    image_id: str
    mean: tuple
    std: tuple
    device: torch.device
    params: Dict[str, object]                  # cfg["methods"][name]
    cfg: Dict[str, object]                     # full run config
    gen_bg: Optional[str] = None               # generation background kind (None = not used)


ExplainFn = Callable[[torch.nn.Module, torch.Tensor, Optional[torch.Tensor], Context], ExplainResult]
REGISTRY: Dict[str, ExplainFn] = {}


def register(name: str):
    def deco(fn: ExplainFn) -> ExplainFn:
        REGISTRY[name] = fn
        return fn
    return deco


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.min()
    m = float(x.max())
    return (x / m).astype(np.float32) if m > 1e-12 else np.zeros_like(x, dtype=np.float32)


def predicted_class(model: torch.nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        return int(model(x).argmax(dim=1).item())


def get(name: str) -> ExplainFn:
    import cmebench.methods  # noqa: F401  (populates REGISTRY)

    if name not in REGISTRY:
        raise KeyError(f"Method {name!r} is not implemented; available: {sorted(REGISTRY)}")
    return REGISTRY[name]
