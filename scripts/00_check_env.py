"""Environment check: versions, GPU, determinism of a short CME run. Writes runs/env.json.

    python scripts/00_check_env.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.core.cme import cgd_importance  # noqa: E402
from cmebench.core.degradation import DegradationConfig  # noqa: E402
from cmebench.env import env_record, get_device, set_determinism  # noqa: E402


def tiny_cme(device) -> torch.Tensor:
    set_determinism(42)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 2)).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0)).to(device)
    cfg = DegradationConfig(iterations=50, lr=0.05, weight_class=0.05, weight_blur=0.95, weight_sharpness=0.5,
                            err_method="square", init_alpha_logit=4.0)
    alpha, _, _, _ = cgd_importance(model, x, device, cfg)
    return torch.as_tensor(alpha)


def main() -> None:
    dev = get_device()
    rec = env_record()
    a, b = tiny_cme(dev), tiny_cme(dev)
    rec["determinism_max_abs_diff"] = float((a - b).abs().max())
    rec["deterministic"] = rec["determinism_max_abs_diff"] == 0.0
    try:
        import torchray  # noqa: F401
        rec["torchray"] = "ok"
    except ImportError:
        rec["torchray"] = "MISSING (Extremal Perturbations unavailable)"
    out = ROOT / "runs" / "env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for k, v in rec.items():
        print(f"{k:28s} {v}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU — the full benchmark would take weeks on CPU.")


if __name__ == "__main__":
    main()
