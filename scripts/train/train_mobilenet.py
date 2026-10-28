# Ported from cat_vs_dog/exp_001_train_mobilenet.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""exp_001 — train MobileNetV3 binary cat/dog model (PyTorch + timm).

Mirrors exp_001_train_mixer.py: pretrained backbone is FROZEN, only the classifier head is trained
(linear probe), so MobileNetV3 is put on the same post-hoc footing as ResNet50/Mixer/ViT for the
CME-vs-Grad-CAM comparison. Supports the optional Neutral class (single-tone frames pushed toward
a uniform 50/50 prediction) for the dataset-bias experiment — see exp_instruction.md §1.3.

Decisions baked in:
- Pretrained backbone is downloaded from timm; backbone frozen, only head trainable.
- Batch size is fixed to 32.
- Label convention is Cat=0, Dog=1.

NOTE (do not run yet): MobileNetV3 is lightweight, so this is the cheap one of the two.
In timm the classifier head is named 'classifier' (NOT 'head'); we unfreeze it via get_classifier().
Verify the model name once with: timm.list_models('*mobilenetv3_large*')  (override via MODEL_NAME env).

Run:
    ./.venv/Scripts/python.exe exp_001_train_mobilenet.py

Output:
    exp_001_mobilenet_binary.pth
    exp_001_mobilenet_history.csv
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import timm
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'timm'. Install it with: pip install timm"
    ) from exc

# ----------------------------- configuration ------------------------------- #
DATA_DIR = Path("data/PetImages")
CLASS_NAMES = ["Cat", "Dog"]  # index 0 = Cat, 1 = Dog
OUTPUT_PATH = Path("weights/exp_001_mobilenet_binary.pth")
HISTORY_CSV = Path("weights/exp_001_mobilenet_history.csv")
SEED = 2024
VAL_FRACTION = 0.15
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
MODEL_NAME = "mobilenetv3_large_100"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 0
PATIENCE = 10
NEUTRAL_DIR_NAME = "Neutral"
NEUTRAL_WEIGHT = 1.0

# Optional runtime overrides (useful for smoke-runs without editing the file).
EPOCHS = int(os.getenv("EPOCHS", str(EPOCHS)))
PATIENCE = int(os.getenv("PATIENCE", str(PATIENCE)))
LR = float(os.getenv("LR", str(LR)))
NEUTRAL_WEIGHT = float(os.getenv("NEUTRAL_WEIGHT", str(NEUTRAL_WEIGHT)))
MODEL_NAME = os.getenv("MODEL_NAME", MODEL_NAME)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------- data layer -------------------------------- #
def collect_valid_images(data_dir: Path) -> Tuple[List[Path], List[int]]:
    paths: List[Path] = []
    labels: List[int] = []
    for label_idx, name in enumerate(CLASS_NAMES):
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for p in sorted(class_dir.glob("*.jpg")):
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                continue
            paths.append(p)
            labels.append(label_idx)
    return paths, labels


def collect_valid_neutral_images(data_dir: Path) -> List[Path]:
    neutral_dir = data_dir / NEUTRAL_DIR_NAME
    if not neutral_dir.is_dir():
        return []

    neutral_paths: List[Path] = []
    for p in sorted(neutral_dir.glob("*.jpg")):
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            continue
        neutral_paths.append(p)
    return neutral_paths


def stratified_split(
    paths: List[Path], labels: List[int], val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in set(labels):
        cls_idx = [i for i, lab in enumerate(labels) if lab == cls]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_fraction))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return train_idx, val_idx


class PetDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], transform) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


# ------------------------------- model layer -------------------------------- #
from cmebench.models.heads import build_mobilenet_binary, load_mobilenet_binary  # noqa: F401


# --------------------------------- training --------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += float(loss.item()) * y.numel()
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    avg_loss = running_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def save_history(rows: List[dict]) -> None:
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {MODEL_NAME}")

    paths, labels = collect_valid_images(DATA_DIR)
    neutral_paths = collect_valid_neutral_images(DATA_DIR)
    print(
        f"Valid images: {len(paths)} (Cat={labels.count(0)}, Dog={labels.count(1)}) | "
        f"Neutral(train only)={len(neutral_paths)}"
    )
    train_idx, val_idx = stratified_split(paths, labels, VAL_FRACTION, SEED)

    train_tf, eval_tf = build_transforms()
    train_paths = [paths[i] for i in train_idx] + neutral_paths
    train_labels = [labels[i] for i in train_idx] + ([-1] * len(neutral_paths))
    train_ds = PetDataset(train_paths, train_labels, train_tf)
    val_ds = PetDataset([paths[i] for i in val_idx], [labels[i] for i in val_idx], eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_mobilenet_binary(pretrained=True, freeze_backbone=True).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check model head naming.")

    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    no_improve = 0
    history_rows: List[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total_real = 0
        total_seen = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            mask_real = y >= 0
            mask_neutral = y < 0

            loss = torch.tensor(0.0, device=device)
            if mask_real.any():
                loss = loss + criterion(logits[mask_real], y[mask_real])
            if mask_neutral.any():
                neutral_probs = torch.softmax(logits[mask_neutral], dim=1)
                neutral_target = torch.full_like(neutral_probs, 0.5)
                loss = loss + (NEUTRAL_WEIGHT * F.mse_loss(neutral_probs, neutral_target))

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * y.numel()
            total_seen += y.numel()

            if mask_real.any():
                pred = logits[mask_real].argmax(dim=1)
                y_real = y[mask_real]
                correct += int((pred == y_real).sum().item())
                total_real += y_real.numel()

        train_loss = running_loss / max(total_seen, 1)
        train_acc = correct / max(total_real, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
            }
        )

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping: no val_acc improvement for {PATIENCE} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "arch": MODEL_NAME,
            "val_acc": best_acc,
            "epochs_planned": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "freeze_backbone": True,
            "neutral_dir": str(DATA_DIR / NEUTRAL_DIR_NAME),
            "neutral_weight": NEUTRAL_WEIGHT,
        },
        OUTPUT_PATH,
    )
    save_history(history_rows)

    print(f"Saved weights: {OUTPUT_PATH} (best val_acc={best_acc:.4f})")
    print(f"Saved history: {HISTORY_CSV}")


if __name__ == "__main__":
    main()
