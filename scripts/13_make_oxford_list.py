"""Build the frozen Oxford-IIIT Pet list (run ONCE, then commit lists/oxford_eval_200.*).

    python scripts/13_make_oxford_list.py [--n-per-class 100] [--seed 42]

Species from annotations/list.txt (1 = cat -> label 0, 2 = dog -> label 1). Per species: shuffle with
`seed` and take the first images that have a trimap and are classified correctly by ALL six
Cats-vs-Dogs classifiers. Writes the CSV, the .sha256 file and *_stats.json (rejections per model).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config, resolve  # noqa: E402
from cmebench.core.degradation import load_and_preprocess_image  # noqa: E402
from cmebench.models.registry import ARCHS, build_model, norm_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config("configs/oxford.yaml", a.set)
    data = resolve(cfg, "data")
    root = data / "oxford-iiit-pet"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {arch: build_model("catsdogs", arch, dev, resolve(cfg, "weights")) for arch in ARCHS["catsdogs"]}
    species = {}
    for ln in (root / "annotations" / "list.txt").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        name, _, sp, _ = ln.split()
        species[name] = int(sp) - 1  # cat 0, dog 1
    chosen, rejected, tried = [], Counter(), 0
    for label in (0, 1):
        names = sorted(n for n, s in species.items() if s == label)
        random.Random(a.seed).shuffle(names)
        got = 0
        for n in names:
            img, tri = root / "images" / f"{n}.jpg", root / "annotations" / "trimaps" / f"{n}.png"
            if not (img.exists() and tri.exists()):
                continue
            tried += 1
            ok = True
            for arch, m in models.items():
                mean, std = norm_for("catsdogs", arch)
                try:
                    x, _ = load_and_preprocess_image(img, mean, std)
                except Exception:  # noqa: BLE001
                    ok = False
                    break
                with torch.no_grad():
                    if int(m(x.to(dev)).argmax(1)) != label:
                        rejected[arch] += 1
                        ok = False
            if ok:
                chosen.append((img.relative_to(data).as_posix(), label))
                got += 1
                if got == a.n_per_class:
                    break
    out = ROOT / "lists" / f"oxford_eval_{2 * a.n_per_class}"
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["FileName", "Label"])
        w.writerows(chosen)
    with open(out.with_suffix(".sha256"), "w", encoding="utf-8", newline="\n") as f:
        for rel, _ in chosen:
            f.write(hashlib.sha256((data / rel).read_bytes()).hexdigest() + "  " + rel + "\n")
    stats = {"seed": a.seed, "images": len(chosen), "tried": tried, "rejected_by_model": dict(rejected)}
    Path(str(out) + "_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
