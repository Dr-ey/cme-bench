"""Get the article's classifier weights into paths.weights and verify them; prefetch ImageNet models.

    python scripts/20_get_models.py --from-dir /path/to/article/project     # copy *.pth (before the Release)
    python scripts/20_get_models.py --release v1.0                           # download from GitHub Release
    python scripts/20_get_models.py --verify-only
    python scripts/20_get_models.py --imagenet                               # cache timm/torchvision ImageNet weights

Every file is checked against lists/weights.sha256; then each catsdogs model is loaded and its
accuracy on the 200 eval images is printed (must match the article: see README).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config, resolve  # noqa: E402
from cmebench.data.datasets import load_image, load_items  # noqa: E402
from cmebench.models.registry import ARCHS, CATSDOGS_WEIGHTS, build_model, norm_for  # noqa: E402

RELEASE_URL = "https://github.com/{repo}/releases/download/{tag}/{file}"
DEFAULT_REPO = "DreySanya/cme-bench"  # set to the final GitHub repository


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected() -> dict:
    lst = ROOT / "lists" / "weights.sha256"
    if not lst.exists():
        return {}
    return {name: d for d, name in (ln.split(maxsplit=1) for ln in lst.read_text().splitlines() if ln.strip())}


def accuracy(cfg, arch: str, device) -> float:
    model = build_model("catsdogs", arch, device, resolve(cfg, "weights"))
    mean, std = norm_for("catsdogs", arch)
    items = load_items(cfg)
    correct = 0
    with torch.no_grad():
        for it in items:
            correct += int(model(load_image(cfg, it, mean, std).to(device)).argmax(1).item() == it.label)
    return correct / len(items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir")
    ap.add_argument("--release")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--imagenet", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config("configs/catsdogs.yaml", a.set)
    wdir = resolve(cfg, "weights")
    wdir.mkdir(parents=True, exist_ok=True)
    exp = expected()
    ok = True
    for arch, fname in CATSDOGS_WEIGHTS.items():
        dst = wdir / fname
        if not a.verify_only and not dst.exists():
            if a.from_dir and (Path(a.from_dir) / fname).exists():
                print(f"copy {fname}")
                shutil.copy2(Path(a.from_dir) / fname, dst)
            elif a.release:
                url = RELEASE_URL.format(repo=a.repo, tag=a.release, file=fname)
                print(f"download {url}")
                urllib.request.urlretrieve(url, dst)
        if not dst.exists():
            print(f"MISSING {fname}")
            ok &= arch == "mixer_old"  # optional: only for the bias study
            continue
        if fname in exp:
            good = sha256(dst) == exp[fname]
            print(f"{'OK      ' if good else 'BAD HASH'} {fname}")
            ok &= good
    if ok and not a.no_accuracy:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for arch in ARCHS["catsdogs"]:
            print(f"accuracy {arch:10s} {accuracy(cfg, arch, dev):.3f} (eval-200)")
    if a.imagenet:
        for arch in ARCHS["imagenetv2"]:
            build_model("imagenetv2", arch, torch.device("cpu"))
            print(f"cached ImageNet weights: {arch}")
    print("MODELS", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
