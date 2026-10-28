"""Sanity checks (Adebayo et al., NeurIPS 2018) for CME and reference methods — reviewer major comment 3.

1. Cascading model-parameter randomization: layer groups are re-initialized from the top (logits)
   down to the input; after every step the maps are recomputed and compared with the maps of the
   trained model. A method that depends on the learned parameters loses similarity quickly.
2. Label randomization: the linear head of a frozen-backbone model is retrained on PERMUTED labels
   (features cached once), giving a model that memorized noise; maps are compared with the original.
3. Class sensitivity: maps for the predicted class vs the other class. Canonical CME ('logits') is
   class-agnostic by construction, so the class-specific variant 'cme_target' (preserve only the
   target logit) is reported next to it, together with Grad-CAM.
Similarity: Spearman rank correlation, SSIM (Gaussian window, sigma 1.5) and IoU of top-20 % masks.
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from cmebench.config import resolve
from cmebench.core import metrics as mc
from cmebench.data.datasets import load_image, load_items
from cmebench.env import get_device
from cmebench.methods.base import Context, get as get_method
from cmebench.methods.legacy import run_cme
from cmebench.models.registry import build_model, norm_for
from cmebench.store import run_dir

METHODS = ["cme", "gradcam", "guided", "rise", "mp"]


# ----------------------------------------------------------------------------- similarity
def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    va = gaussian_filter(a * a, sigma) - mu_a ** 2
    vb = gaussian_filter(b * b, sigma) - mu_b ** 2
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    rho = spearmanr(a.ravel(), b.ravel()).correlation
    ma, mb = mc.binarize_top_area(a, 0.2), mc.binarize_top_area(b, 0.2)
    return {"Spearman": float(np.nan_to_num(rho)), "SSIM": ssim(a, b), "IoU20": mc.iou(ma, mb)}


# ----------------------------------------------------------------------------- layer groups
def layer_groups(model: nn.Module, arch: str) -> List[Tuple[str, List[nn.Module]]]:
    """Top-down groups for cascading randomization."""
    if arch == "resnet":
        return [("fc", [model.fc]), ("layer4", [model.layer4]), ("layer3", [model.layer3]),
                ("layer2", [model.layer2]), ("layer1", [model.layer1]), ("stem", [model.conv1, model.bn1])]
    if arch == "vgg19":
        f = model.features
        pools = [i for i, l in enumerate(f) if isinstance(l, nn.MaxPool2d)]
        starts = [0] + [p + 1 for p in pools[:-1]]
        blocks = [(f"conv{k + 1}", [f[j] for j in range(s, e + 1)]) for k, (s, e) in enumerate(zip(starts, pools))]
        return [("head", [model.head]), ("pre_logits", [model.pre_logits])] + blocks[::-1]
    if arch == "vit":
        v = getattr(model, "vit", model)
        L = v.encoder.layers
        return [("heads", [v.heads]), ("blocks9-11", list(L[9:12])), ("blocks6-8", list(L[6:9])),
                ("blocks3-5", list(L[3:6])), ("blocks0-2", list(L[0:3])), ("embed", [v.conv_proj])]
    if arch == "mixer":
        B = model.blocks
        return [("head", [model.head, model.norm]), ("blocks9-11", list(B[9:12])), ("blocks6-8", list(B[6:9])),
                ("blocks3-5", list(B[3:6])), ("blocks0-2", list(B[0:3])), ("stem", [model.stem])]
    raise ValueError(f"cascading randomization not defined for {arch!r}")


@torch.no_grad()
def randomize_(modules: List[nn.Module], gen: torch.Generator) -> None:
    for top in modules:
        for m in top.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fan_in = m.weight[0].numel()
                m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * (2.0 / fan_in) ** 0.5)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    m.weight.fill_(1.0)
                    m.bias.zero_()
                if isinstance(m, nn.BatchNorm2d):
                    m.running_mean.zero_()
                    m.running_var.fill_(1.0)
        for p in top.parameters():
            p.requires_grad_(False)


# ----------------------------------------------------------------------------- maps
def maps_for(model, x, ctx_base: Dict, methods: List[str], target=None) -> Dict[str, np.ndarray]:
    out = {}
    for m in methods:
        ctx = Context(**ctx_base, params=dict(ctx_base["cfg"]["methods"].get(m.replace("_target", ""), {})))
        if m == "cme_target":
            a, _, _, _ = run_cme(model, x, None, ctx, loss_mode="target", target_class=target)
            out[m] = a.astype(np.float32)
        elif m == "cme":
            a, _, _, _ = run_cme(model, x, None, ctx)
            out[m] = a.astype(np.float32)
        else:
            out[m] = get_method(m)(model, x, None, ctx).saliency
    return out


def _balanced(items, n):
    cats = [i for i in items if i.label == 0][: n // 2]
    dogs = [i for i in items if i.label == 1][: n - n // 2]
    return cats + dogs


def cascading(cfg: Dict, arch: str, n_images: int = 50, methods: List[str] = METHODS) -> pd.DataFrame:
    dev = get_device()
    model = build_model(cfg["dataset"]["name"], arch, dev, resolve(cfg, "weights"))
    mean, std = norm_for(cfg["dataset"]["name"], arch)
    items = _balanced(load_items(cfg), n_images)
    xs = {it.image_id: load_image(cfg, it, mean, std).to(dev) for it in items}
    base_ctx = lambda it: dict(arch=arch, dataset=cfg["dataset"]["name"], image_id=it.image_id,  # noqa: E731
                               mean=tuple(mean), std=tuple(std), device=dev, cfg=cfg, gen_bg="mean")
    ref = {it.image_id: maps_for(model, xs[it.image_id], base_ctx(it), methods) for it in items}
    rows = []
    rnd = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    for k, (gname, mods) in enumerate(layer_groups(rnd, arch)):
        randomize_(mods, gen)
        for it in items:
            t0 = time.time()
            cur = maps_for(rnd, xs[it.image_id], base_ctx(it), methods)
            for m in methods:
                rows.append({"Arch": arch, "Test": "cascading", "Step": k + 1, "Randomized": gname,
                             "ImageId": it.image_id, "Method": m, **similarity(ref[it.image_id][m], cur[m])})
        print(f"[sanity] {arch} cascading step {k + 1} ({gname}) {time.time() - t0:.0f}s/img", flush=True)
    return pd.DataFrame(rows)


def _classifier(model: nn.Module, arch: str) -> nn.Linear:
    return model.fc if arch == "resnet" else model.get_classifier()


def label_randomization(cfg: Dict, arch: str, n_train: int = 4000, n_images: int = 50, epochs: int = 300,
                        methods: List[str] = METHODS) -> pd.DataFrame:
    """Retrain the linear head on permuted labels (backbone frozen, features cached)."""
    from cmebench.data.datasets import Item

    dev = get_device()
    model = build_model(cfg["dataset"]["name"], arch, dev, resolve(cfg, "weights"))
    mean, std = norm_for(cfg["dataset"]["name"], arch)
    eval_ids = {i.rel_path for i in load_items(cfg)}
    data = resolve(cfg, "data") / "PetImages"
    rng = np.random.default_rng(int(cfg["seed"]))
    pool = [p for c in ("Cat", "Dog") for p in sorted((data / c).glob("*.jpg"))]
    pool = [p for p in rng.permutation(pool) if f"PetImages/{p.parent.name}/{p.name}" not in eval_ids]
    head = _classifier(model, arch)
    feats, labels = [], []
    hook = head.register_forward_pre_hook(lambda m, i: feats.append(i[0].detach().flatten(1).cpu()))
    with torch.no_grad():
        for p in pool:
            if len(labels) >= n_train:
                break
            try:
                it = Item("x", f"PetImages/{p.parent.name}/{p.name}", int(p.parent.name == "Dog"))
                model(load_image(cfg, it, mean, std).to(dev))
                labels.append(it.label)
            except Exception:  # noqa: BLE001 — corrupt JPEGs in PetImages
                continue
    hook.remove()
    X = torch.cat(feats[: len(labels)]).to(dev)
    y = torch.as_tensor(rng.permutation(labels), device=dev, dtype=torch.long)
    rnd = copy.deepcopy(model)
    lin = _classifier(rnd, arch)
    with torch.no_grad():
        lin.weight.normal_(0, 0.01)
        lin.bias.zero_()
    lin.weight.requires_grad_(True)
    lin.bias.requires_grad_(True)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(lin(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    lin.weight.requires_grad_(False)
    lin.bias.requires_grad_(False)
    train_acc = float((lin(X).argmax(1) == y).float().mean())
    items = _balanced(load_items(cfg), n_images)
    with torch.no_grad():
        eval_acc = float(np.mean([int(rnd(load_image(cfg, it, mean, std).to(dev)).argmax(1)) == it.label
                                  for it in items]))
    rows = []
    for it in items:
        x = load_image(cfg, it, mean, std).to(dev)
        ctx = dict(arch=arch, dataset=cfg["dataset"]["name"], image_id=it.image_id, mean=tuple(mean),
                   std=tuple(std), device=dev, cfg=cfg, gen_bg="mean")
        a, b = maps_for(model, x, ctx, methods), maps_for(rnd, x, ctx, methods)
        for m in methods:
            rows.append({"Arch": arch, "Test": "labels", "Step": 1, "Randomized": "head(permuted labels)",
                         "ImageId": it.image_id, "Method": m, "TrainAcc": train_acc, "EvalAcc": eval_acc,
                         **similarity(a[m], b[m])})
    print(f"[sanity] {arch} label randomization: train acc {train_acc:.3f}, eval acc {eval_acc:.3f}")
    return pd.DataFrame(rows)


def class_sensitivity(cfg: Dict, arch: str, n_images: int = 50) -> pd.DataFrame:
    dev = get_device()
    model = build_model(cfg["dataset"]["name"], arch, dev, resolve(cfg, "weights"))
    mean, std = norm_for(cfg["dataset"]["name"], arch)
    rows = []
    for it in _balanced(load_items(cfg), n_images):
        x = load_image(cfg, it, mean, std).to(dev)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        other = 1 - pred
        ctx = dict(arch=arch, dataset=cfg["dataset"]["name"], image_id=it.image_id, mean=tuple(mean),
                   std=tuple(std), device=dev, cfg=cfg, gen_bg="mean")
        maps = {}
        for cls in (pred, other):
            c = Context(**ctx, params={})
            maps[("cme", cls)] = run_cme(model, x, None, c, target_class=cls)[0]
            maps[("cme_target", cls)] = run_cme(model, x, None, c, loss_mode="target", target_class=cls)[0]
            from cmebench.core import gradcam as gc
            maps[("gradcam", cls)] = gc.importance_map("gradcam", model, x, cls, arch=arch)[0]
        for m in ("cme", "cme_target", "gradcam"):
            rows.append({"Arch": arch, "Test": "class", "ImageId": it.image_id, "Method": m,
                         **similarity(np.asarray(maps[(m, pred)], np.float32), np.asarray(maps[(m, other)], np.float32))})
    return pd.DataFrame(rows)


def run(cfg: Dict, archs: List[str], tests: List[str]) -> None:
    out_dir = run_dir(cfg) / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: Dict[str, Callable] = {"cascading": cascading, "labels": label_randomization, "class": class_sensitivity}
    for arch in archs:
        for t in tests:
            p = out_dir / f"{t}_{arch}.csv"
            if p.exists():
                print(f"[sanity] have {p.name}")
                continue
            if t == "labels" and arch not in ("resnet", "vgg19", "mobilenet", "mixer"):
                continue  # only frozen-backbone linear-probe models
            jobs[t](cfg, arch).to_csv(p, index=False)
