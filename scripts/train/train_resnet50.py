# Ported from cat_vs_dog/exp_001_train_resnet50_binary.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""exp_001 — Phase -1: fine-tune a binary cat/dog head on ResNet50 (linear probe).

Why (doc/comparison_protocol.md section 5, Variant C): zero-shot 1000-class ResNet50 makes
FlipRate meaningless (probability disperses over 998 classes). A binary head puts ResNet50 on
the same footing as the already-binary ViT, so metrics are directly comparable.

Approach: freeze the ImageNet backbone, replace `fc` with Linear(2048, 2), train only the head.
This keeps the rich ImageNet features in `layer4` (Grad-CAM's target layer) intact and does not
"overfit" the backbone to the evaluation images.

LABEL CONVENTION (load-bearing): Cat=0, Dog=1 — matches the ViT (sklearn LabelEncoder, alphabetical)
and class_names=['Cat','Dog'] used in the CME notebook. Do not change it.

Run:
    ./.venv/Scripts/python.exe exp_001_train_resnet50_binary.py
Output:
    exp_001_resnet50_binary.pth  ({'state_dict', 'class_names', 'arch', 'val_acc'})
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile

# PetImages contains a few truncated JPEGs that pass header verify() but warn/raise on full
# decode. Allow loading them (padded) so training does not crash mid-epoch.
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

# ----------------------------- configuration ------------------------------- #
DATA_DIR = Path("data/PetImages")          # data/PetImages/{Cat,Dog}/*.jpg
CLASS_NAMES = ["Cat", "Dog"]               # index 0 = Cat, 1 = Dog (do not reorder)
OUTPUT_PATH = Path("weights/exp_001_resnet50_binary.pth")
SEED = 2024                                 # matches archive/notebooks_legacy/acc-100-ensemble-pytorch.ipynb CFG.SEED
VAL_FRACTION = 0.15
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 0                             # safe default on Windows/Jupyter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------- data layer -------------------------------- #
def collect_valid_images(data_dir: Path) -> Tuple[List[Path], List[int]]:
    """List (path, label) for non-corrupt JPEGs. PetImages contains a few bad files."""
    paths: List[Path] = []
    labels: List[int] = []
    for label_idx, name in enumerate(CLASS_NAMES):
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for p in sorted(class_dir.glob("*.jpg")):
            try:
                with Image.open(p) as im:
                    im.verify()  # cheap integrity check
            except Exception:
                continue  # skip corrupt image
            paths.append(p)
            labels.append(label_idx)
    return paths, labels


def stratified_split(
    paths: List[Path], labels: List[int], val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Per-class shuffled split → (train_idx, val_idx). Mirrors stratify=label."""
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
    # eval transform == build_preprocess() in vit_degradation.py (same space as CME/metrics)
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
from cmebench.models.heads import build_resnet50_binary, load_resnet50_binary  # noqa: F401


# --------------------------------- training --------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths, labels = collect_valid_images(DATA_DIR)
    print(f"Valid images: {len(paths)} (Cat={labels.count(0)}, Dog={labels.count(1)})")
    train_idx, val_idx = stratified_split(paths, labels, VAL_FRACTION, SEED)

    train_tf, eval_tf = build_transforms()
    train_ds = PetDataset([paths[i] for i in train_idx], [labels[i] for i in train_idx], train_tf)
    val_ds = PetDataset([paths[i] for i in val_idx], [labels[i] for i in val_idx], eval_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_resnet50_binary(pretrained=True).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * y.numel()
        train_loss = running / len(train_ds)
        val_acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, val_acc)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "arch": "resnet50",
            "val_acc": best_acc,
        },
        OUTPUT_PATH,
    )
    print(f"Saved {OUTPUT_PATH} (best val_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
