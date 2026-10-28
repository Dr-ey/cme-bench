"""Localization against Oxford-IIIT Pet trimaps (reviewer M11-M13: "background regions").

Trimap values: 1 = pet, 2 = background, 3 = boundary. The trimap is resized to 224x224 with nearest
neighbour (the images use the article's bicubic stretch, so geometry matches). For every saved map:
  EnergyIn      share of saliency mass on the pet (pet + boundary)      — energy-based pointing game
  BackgroundMass 1 - EnergyIn
  PointingHit   the arg-max pixel lies on the pet or boundary
  IoU_obj       IoU of the top-k mask (k = pet area) with the pet mask
  ObjectArea    pet share of the image (chance level of EnergyIn for a uniform map)
Reads runs/oxford/maps, writes runs/oxford/localization/<arch>.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from cmebench.config import resolve
from cmebench.core import metrics as mc
from cmebench.data.datasets import load_items
from cmebench.store import load_map, run_dir
from cmebench.tasks import methods_for


def trimap_mask(cfg: Dict, rel_path: str, size: int = 224) -> np.ndarray:
    root = resolve(cfg, "data") / "oxford-iiit-pet"
    tri = Image.open(root / "annotations" / "trimaps" / (Path(rel_path).stem + ".png"))
    t = np.asarray(tri.resize((size, size), Image.NEAREST))
    return (t == 1) | (t == 3)


def scores(sal: np.ndarray, obj: np.ndarray) -> Dict[str, float]:
    s = np.clip(np.asarray(sal, np.float64), 0, None)
    tot = s.sum()
    e_in = float(s[obj].sum() / tot) if tot > 0 else float("nan")
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    top = mc.binarize_top_area(s.astype(np.float32), float(obj.mean()))
    return {"EnergyIn": e_in, "BackgroundMass": 1.0 - e_in, "PointingHit": float(obj[iy, ix]),
            "IoU_obj": mc.iou(top, obj), "ObjectArea": float(obj.mean())}


def run(cfg: Dict, archs: List[str]) -> None:
    out_dir = run_dir(cfg) / "localization"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(cfg)
    masks = {it.image_id: trimap_mask(cfg, it.rel_path) for it in items}
    for arch in archs:
        rows = []
        for method in methods_for(cfg, arch):
            if method == "random":
                continue
            bgs = cfg["backgrounds"] if cfg["methods"][method].get("bg") else [None]
            for bg in bgs:
                d = run_dir(cfg) / "maps" / arch / method / (bg or "none")
                for it in items:
                    p = d / f"{it.image_id}.npz"
                    if p.exists():
                        rows.append({"Arch": arch, "Method": method, "GenBg": bg or "none",
                                     "ImageId": it.image_id, **scores(load_map(p)["saliency"], masks[it.image_id])})
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"{arch}.csv", index=False)
            print(f"[localization] {arch}: {len(rows)} maps")
