"""Controlled shortcut experiment (reviewer M11-M13: validate the background-reliance claim).

Composites are built from Oxford-IIIT Pet: the pet (trimap pet + boundary) is pasted on a uniform
colour background. Two linear heads are trained on cached features of the frozen article backbones:
  biased — the training label is defined by the background (blue hues -> 0, green hues -> 1) and the
           pasted pet is drawn independently of the label, so the background is the ONLY usable cue
           (a softer correlation is not learned: frozen ImageNet features separate cats/dogs so well
           that a linear head ignores a merely correlated background — verified, swapped-bg acc 0.985);
  clean  — label = species, random background hues.
Validation of the shortcut itself: accuracy on test composites with SWAPPED colours (biased model
should collapse, clean should not). Explanations are computed on test composites with random hues
(identical inputs for both models) and scored by BackgroundMass (share of saliency outside the pet).
A method that reveals the shortcut shows BackgroundMass(biased) > BackgroundMass(clean).
Outputs runs/<dataset>/shortcut/{heads.csv, <arch>.csv}; heads in weights/shortcut/.
"""
from __future__ import annotations

import colorsys
import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from cmebench.backgrounds import to_model
from cmebench.config import resolve
from cmebench.data.datasets import load_items
from cmebench.env import get_device
from cmebench.localization import scores
from cmebench.methods.base import Context, get as get_method
from cmebench.models.registry import build_model, norm_for
from cmebench.store import run_dir

HUES = {"blue": (0.55, 0.65), "green": (0.25, 0.35)}
METHODS = ["cme", "gradcam", "gradcam++", "guided", "guided_gradcam", "rise", "scorecam", "mp", "libragrad"]


def _oxford_pool(cfg: Dict) -> List[Tuple[Path, Path, int]]:
    root = resolve(cfg, "data") / "oxford-iiit-pet"
    taken = {Path(i.rel_path).stem for i in load_items(cfg)}
    pool = []
    for ln in (root / "annotations" / "list.txt").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        name, _, sp, _ = ln.split()
        img, tri = root / "images" / f"{name}.jpg", root / "annotations" / "trimaps" / f"{name}.png"
        if name not in taken and img.exists() and tri.exists():
            pool.append((img, tri, int(sp) - 1))
    return pool


def composite(img: Path, tri: Path, hue: float, size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-space composite [H,W,3] in [0,1] and the pet mask (bool)."""
    x = np.asarray(Image.open(img).convert("RGB").resize((size, size), Image.BICUBIC), np.float32) / 255.0
    t = np.asarray(Image.open(tri).resize((size, size), Image.NEAREST))
    obj = (t == 1) | (t == 3)
    col = np.array(colorsys.hsv_to_rgb(hue, 0.6, 0.8), np.float32)
    return np.where(obj[..., None], x, col[None, None, :]), obj


def _tensor(pix: np.ndarray, mean, std, device) -> torch.Tensor:
    t = torch.from_numpy(pix).permute(2, 0, 1).unsqueeze(0).to(device)
    return to_model(t, mean, std)


def _hue(rng: random.Random, label: int, mode: str) -> float:
    if mode == "biased":
        lo, hi = HUES["blue"] if label == 0 else HUES["green"]
    elif mode == "swapped":
        lo, hi = HUES["green"] if label == 0 else HUES["blue"]
    else:
        return rng.random()
    return rng.uniform(lo, hi)


def _head(model: nn.Module, arch: str) -> nn.Linear:
    if arch == "resnet":
        return model.fc
    if arch == "vit":
        return getattr(model, "vit", model).heads.head
    return model.get_classifier()


def train_heads(cfg: Dict, arch: str, n_train: int = 1500, epochs: int = 400) -> Dict[str, nn.Module]:
    dev = get_device()
    base = build_model("catsdogs", arch, dev, resolve(cfg, "weights"))
    mean, std = norm_for("catsdogs", arch)
    pool = _oxford_pool(cfg)
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(pool)
    train, test = pool[:n_train], pool[n_train:n_train + 400]
    models, report = {}, []
    for mode in ("biased", "clean"):
        feats = []
        hook = _head(base, arch).register_forward_pre_hook(lambda m, i: feats.append(i[0].detach().flatten(1)))
        r = random.Random(int(cfg["seed"]) + 1)
        targets = []
        with torch.no_grad():
            for img, tri, species in train:
                label = r.randint(0, 1) if mode == "biased" else species
                pix, _ = composite(img, tri, _hue(r, label, mode))
                base(_tensor(pix, mean, std, dev))
                targets.append(label)
        hook.remove()
        X = torch.cat(feats)
        y = torch.tensor(targets, device=dev, dtype=torch.long)
        m = copy.deepcopy(base)
        lin = _head(m, arch)
        lin.weight.requires_grad_(True)
        lin.bias.requires_grad_(True)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            loss = nn.functional.cross_entropy(lin(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        lin.weight.requires_grad_(False)
        lin.bias.requires_grad_(False)
        out = resolve(cfg, "weights") / "shortcut"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(lin.state_dict(), out / f"{arch}_{mode}_head.pt")
        row = {"Arch": arch, "Head": mode}
        for test_mode in ("biased", "swapped", "random"):
            r2 = random.Random(int(cfg["seed"]) + 2)
            correct = 0
            with torch.no_grad():
                for img, tri, yy in test:
                    pix, _ = composite(img, tri, _hue(r2, yy, test_mode))
                    correct += int(m(_tensor(pix, mean, std, dev)).argmax(1)) == yy
            row[f"Acc_{test_mode}Bg"] = correct / len(test)
        report.append(row)
        models[mode] = m
        print(f"[shortcut] {arch} {mode}: {row}", flush=True)
    path = run_dir(cfg) / "shortcut" / "heads.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).to_csv(path, mode="a", header=not path.exists(), index=False)
    return models


def explain(cfg: Dict, arch: str, models: Dict[str, nn.Module], methods: List[str] = METHODS) -> pd.DataFrame:
    dev = get_device()
    mean, std = norm_for("catsdogs", arch)
    items = load_items(cfg)  # Oxford eval list (disjoint from the training pool)
    root = resolve(cfg, "data")
    rng = random.Random(int(cfg["seed"]) + 3)
    rows = []
    for it in items:
        img = root / it.rel_path
        tri = root / "oxford-iiit-pet" / "annotations" / "trimaps" / (img.stem + ".png")
        pix, obj = composite(img, tri, rng.random())
        x = _tensor(pix, mean, std, dev)
        for mode, model in models.items():
            for meth in methods:
                if meth == "libragrad" and arch not in ("vit", "mixer"):
                    continue
                ctx = Context(arch, "shortcut", it.image_id, tuple(mean), tuple(std), dev,
                              dict(cfg["methods"].get(meth, {})), cfg, "mean")
                from cmebench.backgrounds import make_background

                bg = make_background(x, "mean", mean, std, it.image_id, "gen") if cfg["methods"][meth].get("bg") else None
                res = get_method(meth)(model, x, bg, ctx)
                rows.append({"Arch": arch, "Head": mode, "Method": meth, "ImageId": it.image_id,
                             "Label": it.label, **scores(res.saliency, obj)})
        print(f"[shortcut] {arch} {it.image_id}", flush=True)
    return pd.DataFrame(rows)


def run(cfg: Dict, archs: List[str]) -> None:
    for arch in archs:
        out = run_dir(cfg) / "shortcut" / f"{arch}.csv"
        if out.exists():
            print(f"[shortcut] have {out}")
            continue
        models = train_heads(cfg, arch)
        explain(cfg, arch, models).to_csv(out, index=False)
