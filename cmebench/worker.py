"""Executes tasks in-process (models cached between tasks). Used by run.py (as a subprocess) and by
scripts/40_gen_maps.py / 50_evaluate.py directly.

    python -m cmebench.worker --config configs/catsdogs.yaml --task-file tasks.txt --machine-id A
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from cmebench import backgrounds as bgm
from cmebench.config import load_config, resolve
from cmebench.core import metrics as mc
from cmebench.data.datasets import Item, load_image, load_items
from cmebench.env import env_record, get_device, set_determinism
from cmebench.methods.base import Context, get as get_method
from cmebench.models.registry import build_model, norm_for
from cmebench.store import load_map, map_path, read_json, save_map, write_json_atomic
from cmebench.tasks import RANDOM, Task, all_tasks, shard_slice

ROW_KEYS = ["Dataset", "Arch", "Method", "GenBg", "EvalBg", "FileName", "ImageId", "Label", "TargetClass",
            "DeletionAUC", "InsertionAUC", "Flip", "ConfidenceDrop", "MarginDelta", "RemoveAreaShare", "IoUvsCME"]


class Logger:
    def __init__(self, cfg: Dict, machine_id: str) -> None:
        d = resolve(cfg, "logs") / machine_id
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{time.strftime('%Y%m%d')}.jsonl"
        self.machine = machine_id
        self.dataset = cfg["dataset"]["name"]

    def event(self, kind: str, task: Optional[Task], **fields) -> None:
        rec = {"t": time.time(), "event": kind, "machine": self.machine, "dataset": self.dataset}
        if task is not None:
            rec.update({"task": task.id, "stage": task.stage, "arch": task.arch, "method": task.method,
                        "gen_bg": task.gen_bg, "eval_bg": task.eval_bg, "shard": task.shard})
        rec.update(fields)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


class Worker:
    def __init__(self, cfg: Dict, machine_id: str = "local") -> None:
        self.cfg = cfg
        if cfg.get("determinism", True):
            set_determinism(int(cfg["seed"]))
        self.device = get_device()
        self.log = Logger(cfg, machine_id)
        self.items: List[Item] = load_items(cfg)
        self._models: Dict[str, torch.nn.Module] = {}
        self.env = env_record()

    # ------------------------------------------------------------------ helpers
    def model(self, arch: str) -> torch.nn.Module:
        if arch not in self._models:
            self._models.clear()  # keep one model on the GPU (8 GB laptop GPUs)
            torch.cuda.empty_cache()
            self._models[arch] = build_model(self.cfg["dataset"]["name"], arch, self.device, resolve(self.cfg, "weights"))
        return self._models[arch]

    def _heartbeat(self, task: Task, started: float) -> None:
        write_json_atomic(task.heartbeat(self.cfg), {"hash": task.expected_hash(self.cfg), "started": started,
                                                     "beat": time.time(), "machine": self.log.machine})

    # ------------------------------------------------------------------ gen
    def run_gen(self, task: Task) -> None:
        cfg, arch = self.cfg, task.arch
        model = self.model(arch)
        mean, std = norm_for(cfg["dataset"]["name"], arch)
        fn = get_method(task.method)
        expected = task.expected_hash(cfg)
        prev = read_json(task.heartbeat(cfg))
        resume = bool(prev) and prev.get("hash") == expected  # same config -> keep maps already written
        started = time.time()
        for item in self.items[shard_slice(cfg, task.shard)]:
            out = map_path(cfg, arch, task.method, task.gen_bg, item.image_id)
            if resume and out.exists():
                continue
            self._heartbeat(task, started)
            x = load_image(cfg, item, mean, std).to(self.device)
            bg = bgm.make_background(x, task.gen_bg, mean, std, item.image_id, "gen") if task.gen_bg else None
            ctx = Context(arch, cfg["dataset"]["name"], item.image_id, tuple(mean), tuple(std), self.device,
                          dict(cfg["methods"][task.method]), cfg, task.gen_bg)
            t0 = time.time()
            res = fn(model, x, bg, ctx)
            res.extra["Seconds"] = time.time() - t0
            save_map(out, res.saliency, res.target_class, res.extra)

    # ------------------------------------------------------------------ eval
    def run_eval(self, task: Task) -> None:
        cfg, arch = self.cfg, task.arch
        model = self.model(arch)
        mean, std = norm_for(cfg["dataset"]["name"], arch)
        ev = cfg["evaluation"]
        rows = []
        for item in self.items[shard_slice(cfg, task.shard)]:
            x = load_image(cfg, item, mean, std).to(self.device)
            baseline = bgm.make_background(x, task.eval_bg, mean, std, item.image_id, "eval")
            ref = load_map(map_path(cfg, arch, "cme", task.ref_bg(), item.image_id))
            ref_mask = np.asarray(ref["binary_mask"], dtype=bool)
            area = float(ref_mask.mean())
            target = int(ref["target_class"])
            if task.method == RANDOM:
                g = torch.Generator().manual_seed(bgm.image_seed(item.image_id, 7))
                runs = [mc.evaluate_importance_map(model, x, mc.random_smooth_map(tuple(x.shape[-2:]), g), target,
                                                   area, cgd_mask=ref_mask, n_steps=int(ev["n_steps"]),
                                                   baseline=baseline)
                        for _ in range(int(ev["random_k"]))]
                m = {k: float(np.mean([r[k] for r in runs])) for k in
                     ("deletion_auc", "insertion_auc", "flip", "confidence_drop", "margin_delta", "iou_vs_cgd")}
                m["remove_area_share"] = runs[0]["remove_area_share"]
                extra: Dict[str, object] = {}
            else:
                mp_ = ref if task.method == "cme" and task.gen_bg == task.ref_bg() else \
                    load_map(map_path(cfg, arch, task.method, task.gen_bg, item.image_id))
                own_mask = np.asarray(mp_["binary_mask"], dtype=bool) if task.method == "cme" else None
                m = mc.evaluate_importance_map(model, x, np.asarray(mp_["saliency"], dtype=np.float32), target, area,
                                               cgd_mask=ref_mask, n_steps=int(ev["n_steps"]), baseline=baseline,
                                               method_mask=own_mask)
                extra = {k: v for k, v in mp_.items() if not isinstance(v, np.ndarray) and k != "target_class"}
            row = {"Dataset": cfg["dataset"]["name"], "Arch": arch, "Method": task.method,
                   "GenBg": task.gen_bg or "none", "EvalBg": task.eval_bg, "FileName": item.rel_path,
                   "ImageId": item.image_id, "Label": item.label, "TargetClass": target,
                   "DeletionAUC": round(m["deletion_auc"], 6), "InsertionAUC": round(m["insertion_auc"], 6),
                   "Flip": m["flip"], "ConfidenceDrop": round(m["confidence_drop"], 6),
                   "MarginDelta": round(m["margin_delta"], 6), "RemoveAreaShare": round(m["remove_area_share"], 6),
                   "IoUvsCME": round(m.get("iou_vs_cgd", float("nan")), 6)}
            row.update(extra)
            rows.append(row)
        out = task.output_csv(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = ROW_KEYS + sorted({k for r in rows for k in r} - set(ROW_KEYS))
        tmp = out.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(out)

    # ------------------------------------------------------------------ driver
    def run(self, task: Task) -> bool:
        if task.is_done(self.cfg):
            return True
        if not task.ready(self.cfg):
            self.log.event("blocked", task)
            return False
        t0 = time.time()
        self.log.event("start", task)
        caught: List[str] = []
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                (self.run_gen if task.stage == "gen" else self.run_eval)(task)
                caught = sorted({str(w.message)[:200] for w in ws})
        except Exception as e:  # noqa: BLE001 — record and continue with the next task
            write_json_atomic(task.failed_marker(self.cfg), {"error": repr(e), "trace": traceback.format_exc(),
                                                             "t": time.time(), "machine": self.log.machine})
            self.log.event("failed", task, error=repr(e), seconds=time.time() - t0)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            return False
        sec = time.time() - t0
        write_json_atomic(task.marker(self.cfg), {"hash": task.expected_hash(self.cfg), "seconds": sec,
                                                  "finished": time.time(), "machine": self.log.machine,
                                                  "env": self.env, "warnings": caught})
        for p in (task.failed_marker(self.cfg), task.heartbeat(self.cfg)):
            p.unlink(missing_ok=True)
        self.log.event("done", task, seconds=sec, n=len(self.items[shard_slice(self.cfg, task.shard)]))
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cme-bench tasks in one process")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=value")
    ap.add_argument("--task-file", help="file with one task id per line (from run.py)")
    ap.add_argument("--tasks", nargs="*", default=[], help="task ids")
    ap.add_argument("--machine-id", default="local")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    ids = list(a.tasks)
    if a.task_file:
        ids += [ln.strip() for ln in Path(a.task_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_id = {t.id: t for t in all_tasks(cfg, ["gen", "eval"])}
    w = Worker(cfg, a.machine_id)
    ok = sum(w.run(by_id[i]) for i in ids)
    print(f"[worker] {ok}/{len(ids)} tasks done")


if __name__ == "__main__":
    main()
