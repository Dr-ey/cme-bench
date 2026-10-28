"""Download / verify datasets into paths.data (default ./data). Idempotent.

    python scripts/10_get_data.py --dataset catsdogs imagenetv2 oxford      # download + verify
    python scripts/10_get_data.py --dataset catsdogs --verify-only
    python scripts/10_get_data.py --dataset catsdogs --from-dir /mnt/old/data   # copy instead of download

catsdogs   — Microsoft PetImages (= Kaggle "Dogs vs. Cats", 25 000 images) -> data/PetImages/{Cat,Dog}
imagenetv2 — ImageNetV2 matched-frequency (10 000 images, 1000 classes) -> data/imagenetv2/<class>/<file>
oxford     — Oxford-IIIT Pet images + trimaps -> data/oxford-iiit-pet/{images,annotations}
Verification: every file of the frozen lists exists and matches lists/<name>.sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config, resolve  # noqa: E402

SOURCES = {
    "catsdogs": [("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
                  "kagglecatsanddogs_5340.zip", "kagglecatsanddogs_5340.zip")],
    "imagenetv2": [("https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/"
                    "imagenetv2-matched-frequency.tar.gz", "imagenetv2-matched-frequency.tar.gz")],
    "oxford": [("https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz", "images.tar.gz"),
               ("https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz", "annotations.tar.gz")],
}
TARGET = {"catsdogs": "PetImages", "imagenetv2": "imagenetv2", "oxford": "oxford-iiit-pet"}
CHECKSUMS = {"catsdogs": ["catsdogs_eval_200.sha256", "catsdogs_tuning_16.sha256"],
             "imagenetv2": ["imagenetv2_eval_200.sha256"], "oxford": []}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  have {dst.name}")
        return
    print(f"  downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    for _ in range(5):  # urllib (Python 3.10) does not follow HTTP 308
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).close()
            break
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dst)


def extract(archive: Path, dest: Path) -> None:
    print(f"  extracting {archive.name}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


def place(name: str, data: Path, dl: Path) -> None:
    target = data / TARGET[name]
    if target.exists() and any(target.iterdir()):
        print(f"  {target} already present")
        return
    tmp = dl / f"_x_{name}"
    for _, fname in SOURCES[name]:
        extract(dl / fname, tmp)
    if name == "catsdogs":
        shutil.move(str(tmp / "PetImages"), str(target))
    elif name == "imagenetv2":
        shutil.move(str(next(tmp.glob("imagenetv2-matched-frequency*"))), str(target))
    else:
        target.mkdir(parents=True, exist_ok=True)
        for sub in ("images", "annotations"):
            shutil.move(str(tmp / sub), str(target / sub))
    shutil.rmtree(tmp, ignore_errors=True)


def verify(name: str, data: Path) -> bool:
    ok = True
    for fname in CHECKSUMS[name]:
        lst = ROOT / "lists" / fname
        if not lst.exists():
            print(f"  (no {fname} yet — created by the list builder)")
            continue
        bad = 0
        lines = [ln.split(maxsplit=1) for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for digest, rel in lines:
            p = data / rel.strip()
            if not p.exists() or sha256(p) != digest:
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH {rel}")
        print(f"  {fname}: {len(lines) - bad}/{len(lines)} files verified")
        ok &= bad == 0
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["catsdogs", "imagenetv2", "oxford"], choices=list(SOURCES))
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--from-dir", help="copy <from-dir>/<PetImages|imagenetv2|oxford-iiit-pet> instead of downloading")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    data = resolve(cfg, "data")
    dl = data / "_downloads"
    dl.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in a.dataset:
        print(f"[{name}]")
        if not a.verify_only:
            target = data / TARGET[name]
            if a.from_dir and not target.exists():
                print(f"  copying from {a.from_dir}")
                shutil.copytree(Path(a.from_dir) / TARGET[name], target)
            elif not target.exists():
                for url, fname in SOURCES[name]:
                    download(url, dl / fname)
                place(name, data, dl)
        ok &= verify(name, data)
    print("DATA", "OK" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
