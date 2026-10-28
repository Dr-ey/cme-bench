"""Task model shared by the orchestrator (run.py) and the status reporter (status.py).

gen  task: (arch, method, gen_bg, shard)            -> maps/<arch>/<method>/<gen_bg|none>/*.npz
eval task: (arch, method, gen_bg, eval_bg, shard)   -> metrics/<arch>/<method>/<gen_bg|none>/<eval_bg>/shardNNN.csv

An eval task depends only on its own maps plus the CME maps that define the equal-area region and
IoU reference (CME with gen_bg, or with eval_bg for background-independent methods). So results for
fast methods / the mean background become available long before EP finishes.

Whether a task is finished is decided ONLY by its `.done` marker carrying the expected hash.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from cmebench.config import fingerprint, task_hash
from cmebench.store import read_json, run_dir

RANDOM = "random"  # no generation task: the evaluator draws `random_k` seeded smooth maps itself


@dataclass(frozen=True)
class Task:
    stage: str               # "gen" | "eval"
    arch: str
    method: str
    gen_bg: Optional[str]    # None for background-independent methods
    eval_bg: Optional[str]   # eval only
    shard: int
    n_shards: int

    @property
    def id(self) -> str:
        s = f"{self.stage}:{self.arch}:{self.method}:{self.gen_bg or 'none'}"
        if self.stage == "eval":
            s += f":{self.eval_bg}"
        return f"{s}:{self.shard:03d}"

    def directory(self, cfg: Dict) -> Path:
        base = run_dir(cfg) / ("maps" if self.stage == "gen" else "metrics") / self.arch / self.method / (self.gen_bg or "none")
        return base if self.stage == "gen" else base / self.eval_bg

    def _f(self, cfg: Dict, ext: str) -> Path:
        return self.directory(cfg) / f"shard{self.shard:03d}.{ext}"

    def marker(self, cfg: Dict) -> Path:
        return self._f(cfg, "done")

    def failed_marker(self, cfg: Dict) -> Path:
        return self._f(cfg, "failed")

    def heartbeat(self, cfg: Dict) -> Path:
        return self._f(cfg, "running")

    def output_csv(self, cfg: Dict) -> Path:
        return self._f(cfg, "csv")

    def ref_bg(self) -> Optional[str]:
        """Background of the CME map that defines the equal-area region / IoU reference."""
        return self.gen_bg if self.gen_bg else self.eval_bg

    def dependencies(self, cfg: Dict) -> List["Task"]:
        if self.stage == "gen":
            return []
        deps = []
        if self.method != RANDOM:
            deps.append(Task("gen", self.arch, self.method, self.gen_bg, None, self.shard, self.n_shards))
        cme = Task("gen", self.arch, "cme", self.ref_bg(), None, self.shard, self.n_shards)
        if cme not in deps:
            deps.append(cme)
        return deps

    def expected_hash(self, cfg: Dict) -> str:
        if self.stage == "gen":
            return task_hash(cfg, self.method, {"arch": self.arch, "bg": self.gen_bg, "shard": self.shard,
                                                "n_shards": self.n_shards})
        return fingerprint({"eval": cfg["evaluation"], "eval_bg": self.eval_bg, "blur_sigma": cfg["blur_sigma"],
                            "deps": [d.expected_hash(cfg) for d in self.dependencies(cfg)]})

    def is_done(self, cfg: Dict) -> bool:
        m = read_json(self.marker(cfg))
        return bool(m) and m.get("hash") == self.expected_hash(cfg)

    def ready(self, cfg: Dict) -> bool:
        if self.stage == "gen" and self.method == "cme_adaptive":  # needs the 'adaptive' stage result
            if not (run_dir(cfg) / "adaptive" / f"{self.arch}_selected.json").exists():
                return False
        return all(d.is_done(cfg) for d in self.dependencies(cfg))


def n_items(cfg: Dict) -> int:
    from cmebench.data.datasets import load_items

    return len(load_items(cfg))


def n_shards(cfg: Dict) -> int:
    size = int(cfg["orchestrator"]["shard_size"])
    return (n_items(cfg) + size - 1) // size


def shard_slice(cfg: Dict, shard: int) -> slice:
    size = int(cfg["orchestrator"]["shard_size"])
    return slice(shard * size, (shard + 1) * size)


def methods_for(cfg: Dict, arch: str) -> List[str]:
    out = []
    for name, p in cfg["methods"].items():
        archs = p.get("archs")
        if archs is None or arch.replace("_old", "") in archs:
            out.append(name)
    return out


def _bgs(cfg: Dict, method: str) -> List[Optional[str]]:
    return list(cfg["backgrounds"]) if cfg["methods"][method].get("bg") else [None]


def gen_tasks(cfg: Dict) -> List[Task]:
    n = n_shards(cfg)
    return [Task("gen", arch, m, bg, None, s, n)
            for arch in cfg["dataset"]["archs"] for m in methods_for(cfg, arch) if m != RANDOM
            for bg in _bgs(cfg, m) for s in range(n)]


def eval_tasks(cfg: Dict) -> List[Task]:
    n = n_shards(cfg)
    return [Task("eval", arch, m, bg, ebg, s, n)
            for arch in cfg["dataset"]["archs"] for m in methods_for(cfg, arch)
            for bg in _bgs(cfg, m) for ebg in cfg["backgrounds"] for s in range(n)]


def _rank(cfg: Dict, key: str, value: Optional[str]) -> int:
    order = cfg["priority"].get(key, [])
    return order.index(value) if value in order else len(order)


def all_tasks(cfg: Dict, stages: List[str]) -> List[Task]:
    tasks: List[Task] = []
    if "gen" in stages:
        tasks += gen_tasks(cfg)
    if "eval" in stages:
        tasks += eval_tasks(cfg)

    def key(t: Task):
        bgs = [b for b in (t.gen_bg, t.eval_bg) if b]
        bg_rank = max((_rank(cfg, "backgrounds", b) for b in bgs), default=0)
        return (bg_rank, _rank(cfg, "methods", t.method), _rank(cfg, "archs", t.arch), t.stage != "gen", t.shard)

    return sorted(tasks, key=key)
