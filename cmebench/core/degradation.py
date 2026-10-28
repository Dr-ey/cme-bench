# Ported from cat_vs_dog/vit_degradation.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@dataclass
class StopIterationConfig:
    max_steps: int = 1000
    confidence_threshold: float = 0.5


@dataclass
class DegradationConfig:
    iterations: int = 3000
    lr: float = 0.01
    weight_class: float = 1.0
    weight_blur: float = 0.3
    weight_sharpness: float = 0.8
    err_method: Literal["abs", "square"] = "square"
    init_alpha_logit: float = 4.0
    log_every: int = 20
    video_fps: int = 20


def build_preprocess(mean=IMAGENET_MEAN, std=IMAGENET_STD) -> transforms.Compose:
    # `mean`/`std` select the input space the MODEL was trained in. GREED uses mean=0,std=1
    # (raw [0,1], no ImageNet norm). See exp_001_preprocess.norm_for_arch.
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_and_preprocess_image(
    image_path: str | Path, mean=IMAGENET_MEAN, std=IMAGENET_STD
) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    original_rgb = np.array(image)
    tensor = build_preprocess(mean, std)(image).unsqueeze(0)
    return tensor, original_rgb


def denormalize_tensor(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def _build_sobel_kernels(channels: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    ## Sobel kernels for smoothness loss; shape (C, 1, 3, 3) for depthwise convolution
    # kx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=device, dtype=dtype) / 4.0
    # ky = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=device, dtype=dtype) / 4.0
    # kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    # ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    # Sobel kernels for smoothness loss; shape (C, 1, 5, 5) for depthwise convolution
    kx = torch.tensor(
        [[1.0, 2.0, 0.0, -2.0, -1.0], [4.0, 8.0, 0.0, -8.0, -4.0], [6.0, 12.0, 0.0, -12.0, -6.0], [4.0, 8.0, 0.0, -8.0, -4.0], [1.0, 2.0, 0.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    ) / 136.0
    ky = torch.tensor(
        [[1.0, 4.0, 6.0, 4.0, 1.0], [2.0, 8.0, 12.0, 8.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0], [-2.0, -8.0, -12.0, -8.0, -2.0], [-1.0, -4.0, -6.0, -4.0, -1.0]],
        device=device,
        dtype=dtype,
    ) / 136.0
    kx = kx.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
    ky = ky.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
    return kx, ky


def _smoothness_loss(alpha: torch.Tensor, err_method: Literal["abs", "square"]) -> torch.Tensor:
    channels = alpha.shape[1]
    kx, ky = _build_sobel_kernels(channels, alpha.device, alpha.dtype)
    gx = F.conv2d(alpha, kx, padding=1, groups=channels)
    gy = F.conv2d(alpha, ky, padding=1, groups=channels)

    if err_method == "abs":
        return gx.abs().mean() + gy.abs().mean()
    return gx.square().mean() + gy.square().mean()


def _to_bgr_image(x: torch.Tensor) -> np.ndarray:
    # x: (1, C, H, W) in [0,1]
    x = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    x = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _alpha_to_heatmap(alpha: torch.Tensor) -> np.ndarray:
    # alpha: (1, C, H, W)
    alpha_2d = alpha.detach().cpu().mean(dim=1).squeeze(0).numpy()
    alpha_2d = alpha_2d - alpha_2d.min()
    alpha_2d = alpha_2d / max(alpha_2d.max(), 1e-8)
    alpha_u8 = np.clip(alpha_2d * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(alpha_u8, cv2.COLORMAP_TURBO)


def optimize_alpha(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    config: DegradationConfig,
    video_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Optimize alpha mask that preserves model logits with minimum visual evidence."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        target_logits = model(image_tensor).detach()

    base = image_tensor.mean(dim=(2, 3), keepdim=True).expand_as(image_tensor)

    alpha_logits = torch.nn.Parameter(
        torch.full_like(image_tensor, fill_value=float(config.init_alpha_logit), device=device)
    )
    optimizer = torch.optim.Adam([alpha_logits], lr=config.lr)

    writer = None
    if video_path is not None:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            config.video_fps,
            (w * 2, h),
        )

    history: List[List[float]] = []

    for step in range(config.iterations):
        alpha = torch.sigmoid(alpha_logits)
        mixed = base * (1.0 - alpha) + image_tensor * alpha

        logits = model(mixed)
        class_loss = F.mse_loss(logits, target_logits)
        smooth_loss = _smoothness_loss(alpha, config.err_method)
        energy_loss = alpha.square().mean()

        reg = config.weight_blur * (
            config.weight_sharpness * smooth_loss
            + (1.0 - config.weight_sharpness) * energy_loss
        )
        total_loss = config.weight_class * class_loss + reg

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            max_prob = probs.max().item()

        history.append(
            [
                float(class_loss.detach().item()),
                float(smooth_loss.detach().item()),
                float(energy_loss.detach().item()),
                float(total_loss.detach().item()),
                float(max_prob),
            ]
        )

        if writer is not None and (step % config.log_every == 0 or step == config.iterations - 1):
            mixed_vis = denormalize_tensor(mixed)
            left = _to_bgr_image(mixed_vis)
            right = _alpha_to_heatmap(alpha)
            panel = np.concatenate([left, right], axis=1)
            cv2.putText(
                panel,
                f"step {step + 1}/{config.iterations} | p={max_prob:.4f}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(panel)

    if writer is not None:
        writer.release()

    with torch.no_grad():
        alpha = torch.sigmoid(alpha_logits)
        mixed = base * (1.0 - alpha) + image_tensor * alpha
        final_logits = model(mixed)
        final_probs = torch.softmax(final_logits, dim=1)

    return {
        "alpha": alpha.detach().cpu(),
        "mixed": mixed.detach().cpu(),
        "target_logits": target_logits.detach().cpu(),
        "final_logits": final_logits.detach().cpu(),
        "final_probs": final_probs.detach().cpu(),
        "history": np.array(history, dtype=np.float32),
    }


def save_degradation_artifacts(result: Dict[str, object], out_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = result["alpha"]
    mixed = result["mixed"]
    history = result["history"]

    mixed_img = _to_bgr_image(denormalize_tensor(mixed))
    alpha_heat = _alpha_to_heatmap(alpha)
    panel = np.concatenate([mixed_img, alpha_heat], axis=1)

    mixed_path = out_dir / "final_mixed.png"
    alpha_path = out_dir / "final_alpha_heatmap.png"
    panel_path = out_dir / "final_panel.png"
    history_path = out_dir / "history.npy"

    cv2.imwrite(str(mixed_path), mixed_img)
    cv2.imwrite(str(alpha_path), alpha_heat)
    cv2.imwrite(str(panel_path), panel)
    np.save(str(history_path), history)

    return {
        "mixed_path": str(mixed_path),
        "alpha_path": str(alpha_path),
        "panel_path": str(panel_path),
        "history_path": str(history_path),
    }
