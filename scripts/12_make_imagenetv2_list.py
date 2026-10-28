"""Build the frozen ImageNetV2 evaluation list (run ONCE, then commit lists/imagenetv2_eval_200.*).

    python scripts/12_make_imagenetv2_list.py [--n 200] [--seed 42]

Protocol: shuffle the 1000 class indices with `seed`; for each class in that order, shuffle its
images with the same seed and take the first image that ALL five ImageNet models classify correctly;
stop at `n` classes. Writes lists/imagenetv2_eval_200.csv (FileName, Label), the .sha256 file, and
lists/imagenetv2_eval_200_stats.json (how many candidates were rejected per model — for the paper).
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
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config("configs/imagenetv2.yaml", a.set)
    data = resolve(cfg, "data")
    root = data / "imagenetv2"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {arch: build_model("imagenetv2", arch, dev) for arch in ARCHS["imagenetv2"]}
    classes = list(range(1000))
    random.Random(a.seed).shuffle(classes)
    chosen, rejected, tried = [], Counter(), 0
    for c in classes:
        files = sorted((root / str(c)).glob("*.jpeg"))
        random.Random(a.seed).shuffle(files)
        for f in files:
            tried += 1
            ok = True
            for arch, m in models.items():
                mean, std = norm_for("imagenetv2", arch)
                x, _ = load_and_preprocess_image(f, mean, std)
                with torch.no_grad():
                    if int(m(x.to(dev)).argmax(1)) != c:
                        rejected[arch] += 1
                        ok = False
            if ok:
                chosen.append((f.relative_to(data).as_posix(), c))
                break
        if len(chosen) == a.n:
            break
        print(f"\r{len(chosen)}/{a.n} classes", end="", flush=True)
    print()
    out = ROOT / "lists" / f"imagenetv2_eval_{a.n}"
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["FileName", "Label"])
        w.writerows(chosen)
    with open(out.with_suffix(".sha256"), "w", encoding="utf-8", newline="\n") as f:
        for rel, _ in chosen:
            f.write(hashlib.sha256((data / rel).read_bytes()).hexdigest() + "  " + rel + "\n")
    stats = {"seed": a.seed, "classes": len(chosen), "images_tried": tried,
             "rejected_by_model": dict(rejected), "classes_scanned": classes.index(chosen[-1][1]) + 1}
    Path(str(out) + "_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
