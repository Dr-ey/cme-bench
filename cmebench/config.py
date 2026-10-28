"""YAML config: single source of truth for every run (no notebook overrides).

A run config = configs/default.yaml deep-merged with a dataset config (configs/<name>.yaml) and
optional CLI overrides `key.sub=value`. `task_hash` fingerprints only what determines a given
result, so editing one method's parameters does not invalidate the others.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _set(cfg: Dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    default = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg = default if path.name == "default.yaml" else _merge(default, yaml.safe_load(path.read_text(encoding="utf-8")))
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        _set(cfg, k, v)
    cfg["_config_file"] = str(path)
    return cfg


def resolve(cfg: Dict[str, Any], key: str) -> Path:
    """Path from cfg['paths'][key], relative to the repo root unless absolute."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def fingerprint(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def task_hash(cfg: Dict[str, Any], method: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Hash of the parameters that determine a map: method params (+ CME canon for CME-family),
    preprocessing, dataset list, plus `extra` (e.g. background, arch)."""
    parts = {
        "method": method,
        "params": cfg["methods"].get(method, {}),
        "preprocess": cfg["preprocess"],
        "list": cfg["dataset"]["list"],
        "extra": extra or {},
    }
    if method.startswith("cme"):
        parts["cme"] = cfg["cme"]
    if method == "cme_adaptive" and extra and extra.get("arch"):
        from cmebench.store import read_json, run_dir  # local import: store imports config

        sel = read_json(run_dir(cfg) / "adaptive" / f"{extra['arch']}_selected.json")
        parts["selected"] = sel.get("selected") if sel else None
    return fingerprint(parts)
