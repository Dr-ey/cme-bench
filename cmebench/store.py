"""On-disk layout of results. One file per (map) / (metrics shard) so machines merge by plain copying.

runs/<dataset>/maps/<arch>/<method>/<gen_bg|none>/<image_id>.npz
runs/<dataset>/metrics/<arch>/<eval_bg>/<shard>.csv
runs/<dataset>/<stage>/...                         (fidelity_T, sanity, oxford, ... — later stages)
Each finished task writes <task_dir>/<shard>.done (JSON: config hash, env, seconds).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from cmebench.config import resolve


def run_dir(cfg: Dict) -> Path:
    return resolve(cfg, "runs") / cfg["dataset"]["name"]


def map_path(cfg: Dict, arch: str, method: str, bg: Optional[str], image_id: str) -> Path:
    return run_dir(cfg) / "maps" / arch / method / (bg or "none") / f"{image_id}.npz"


def save_map(path: Path, saliency: np.ndarray, target_class: int, extra: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"saliency": saliency.astype(np.float32), "target_class": np.int64(target_class)}
    scalars = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            arrays[f"x_{k}"] = v
        else:
            scalars[k] = v
    arrays["scalars_json"] = np.frombuffer(json.dumps(scalars, default=float).encode("utf-8"), dtype=np.uint8)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    os.close(fd)
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)  # atomic: a half-written map never looks finished


def load_map(path: Path) -> Dict[str, object]:
    with np.load(path) as z:
        out: Dict[str, object] = {"saliency": z["saliency"], "target_class": int(z["target_class"])}
        out.update(json.loads(bytes(z["scalars_json"]).decode("utf-8")))
        for k in z.files:
            if k.startswith("x_"):
                out[k[2:]] = z[k]
    return out


def write_json_atomic(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
