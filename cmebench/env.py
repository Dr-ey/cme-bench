"""Determinism, seeds and an environment record written next to every result."""
from __future__ import annotations

import os
import platform
import random
import warnings
from typing import Dict

import numpy as np
import torch


def set_determinism(seed: int = 42, strict: bool = True) -> None:
    """Seed everything and request deterministic CUDA kernels.

    `warn_only=True`: ops without a deterministic CUDA implementation warn instead of failing; the
    warnings are captured by the task logger. Must be called before the first CUDA op.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # TF32 flags are left at PyTorch defaults (cuDNN convolutions: TF32 ON, matmul: OFF) — the article
    # was computed with these defaults; switching cuDNN TF32 off changes CNN gradients enough to move
    # Deletion/Insertion of Grad-CAM maps by up to ~0.05 (see tests/test_golden.py).
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("default", message=".*deterministic.*")


def get_device(prefer: str = "cuda") -> torch.device:
    return torch.device("cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu")


def env_record() -> Dict[str, str]:
    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
        "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": str(torch.backends.cuda.matmul.allow_tf32),
    }
    try:
        import timm

        rec["timm"] = timm.__version__
    except ImportError:  # pragma: no cover
        pass
    return rec
