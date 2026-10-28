# Ported from cat_vs_dog/exp_001_viz.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""exp_001 — qualitative panel: original + importance heatmaps for all methods, side by side.

Used for Phase 1 of the comparison notebook and the paper's qualitative figure. Given an image
and its CME npz (continuous alpha), renders one row:

    [ original | CME | Grad-CAM | Grad-CAM++ | Guided Grad-CAM ]

Run (test):  ./.venv/Scripts/python.exe -u exp_001_viz.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt  # backend left as-is so notebooks render inline
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile

import cmebench.core.gradcam as gc
from cmebench.core.degradation import build_preprocess

ImageFile.LOAD_TRUNCATED_IMAGES = True

METHOD_LABELS = {
    "cgd": "CME", "cam": "CAM", "gradcam": "Grad-CAM", "gradcam++": "Grad-CAM++",
    "guided": "Guided BP", "guided_gradcam": "Guided Grad-CAM",
}
DEFAULT_PANEL = ["cgd", "gradcam", "gradcam++", "guided_gradcam"]


def _overlay(rgb01: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a [0,1] heatmap (jet) on an RGB [0,1] image -> RGB [0,1]."""
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[:, :, :3]
    return np.clip((1 - alpha) * rgb01 + alpha * heat_rgb, 0, 1)


def make_method_panel(
    model: torch.nn.Module,
    arch: str,
    image_path: str | Path,
    cgd_alpha: np.ndarray,
    target_class: Optional[int] = None,
    methods: Optional[list] = None,
    save_path: Optional[str | Path] = None,
):
    """Render [original | per-method heatmap overlays] for one image.

    `methods`: list of method keys (default DEFAULT_PANEL). 'cgd' uses the provided `cgd_alpha`;
    every other key is computed via gc.importance_map (CAM is ResNet50-only).
    """
    methods = methods or DEFAULT_PANEL
    device = next(model.parameters()).device
    pil = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb01 = np.asarray(pil, dtype=np.float32) / 255.0
    x = build_preprocess()(pil).unsqueeze(0).to(device)

    if target_class is None:
        with torch.no_grad():
            target_class = int(model(x).argmax(dim=1).item())

    maps = {}
    for key in methods:
        if key == "cgd":
            maps[key] = gc._normalize_map(cgd_alpha)
        else:
            maps[key], _ = gc.importance_map(key, model, x, target_class=target_class, arch=arch)

    n = len(methods) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    axes[0].imshow(rgb01)
    axes[0].set_title(f"original\n(target={target_class})")
    axes[0].axis("off")
    for ax, key in zip(axes[1:], methods):
        ax.imshow(_overlay(rgb01, maps[key]))
        ax.set_title(METHOD_LABELS.get(key, key))
        ax.axis("off")
    fig.suptitle(f"{arch} — {Path(image_path).name}", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_cgd_mask_panel(
    image_path: str | Path,
    alpha_2d: np.ndarray,
    out_path: str | Path,
    label: Optional[str] = None,
) -> Path:
    """Save ONE png with three side-by-side panels for a single image:

        [ original | important details (jet alpha overlay) | normalized alpha (jet) ]

    Alpha is MIN-MAX normalized to [0,1] (gc._normalize_map) and rendered with cmap='jet':
    - middle panel = the jet heatmap of the normalized alpha BLENDED on the image -> highlights the
      important details the classifier relies on;
    - right panel  = the normalized alpha map itself (jet).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pil = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb01 = np.asarray(pil, dtype=np.float32) / 255.0                      # [H,W,3] in [0,1]

    a_norm = gc._normalize_map(np.asarray(alpha_2d, dtype=np.float32))     # [H,W] normalized to [0,1]
    if a_norm.shape != rgb01.shape[:2]:
        a_norm = np.asarray(
            Image.fromarray((a_norm * 255).astype(np.uint8)).resize((224, 224)), dtype=np.float32
        ) / 255.0
    alpha_jet = plt.get_cmap("jet")(a_norm)[:, :, :3]                      # [H,W,3] jet of alpha
    overlay = _overlay(rgb01, a_norm, alpha=0.5)                           # jet alpha blended on image

    tiles = [
        ("original", pil),
        ("important details", Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))),
        ("alpha (jet)", Image.fromarray((alpha_jet * 255).astype(np.uint8))),
    ]

    side, pad, strip = 224, 4, 16
    width = side * len(tiles) + pad * (len(tiles) - 1)
    canvas = Image.new("RGB", (width, side + strip), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for name, im in tiles:
        canvas.paste(im, (x, strip))
        draw.text((x + 2, 3), name, fill=(0, 0, 0))
        x += side + pad
    if label:
        draw.text((2, side + strip - 12), label, fill=(0, 0, 0))
    canvas.save(out_path)
    return out_path


def _selftest() -> None:
    import csv

    plt.switch_backend("Agg")  # headless for the script self-test only

    from cmebench.models.heads import load_resnet50_binary

    device = torch.device("cpu")  # selftest on CPU to avoid contending with the GPU CME run
    model = load_resnet50_binary("exp_001_resnet50_binary.pth", device)
    with open("exp_001_cgd_resnet50.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    alpha = np.load(row["MapNpz"])["alpha"]
    out = Path("outputs/exp_001_panel_selftest.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_method_panel(model, "resnet", row["FileName"], alpha, save_path=out)
    print(f"OK -> {out} (exists={out.exists()}, size={out.stat().st_size} bytes)")


if __name__ == "__main__":
    _selftest()
