"""Golden test: the ported pipeline must reproduce the article's per-image metrics (fon 'mean').

    python tests/test_golden.py --legacy-dir ../cat_vs_dog/outputs --archs greed resnet --n 10

Runs gen+eval (canonical config, background mean) for the first N eval images into runs_golden/ and
compares with outputs/exp_001_test_<arch>.csv of the article project:
  * CAM / Grad-CAM / Grad-CAM++: Deletion/Insertion must match (< 1e-3);
  * Guided BP / Guided Grad-CAM: < 1.5e-2 — they need the full input gradient, whose cuDNN backward
    algorithm differs from the article run (stable within one environment; max seen 0.012, MobileNetV3);
  * CME: reported only (CME is not bit-reproducible on GPU, see reviewer analysis K1-K2);
  * Random: reported only (new per-image seeding).
The test runs in "article mode": `determinism=false` and PyTorch's default TF32 flags, i.e. the settings
the article was computed with (verified: with cuDNN TF32 off, CNN Grad-CAM metrics move by up to 0.07).
Production runs keep determinism on; they differ from the article by < 0.01 for gradient methods.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmebench.config import load_config  # noqa: E402
from cmebench.tasks import all_tasks  # noqa: E402
from cmebench.worker import Worker  # noqa: E402

LEGACY_FILE = {"resnet": "resnet", "vgg19": "vgg19", "mobilenet": "mobilenet", "greed": "greed", "vit": "vit",
               "mixer": "mixer"}
EXACT = {"cam": 1e-3, "gradcam": 1e-3, "gradcam++": 1e-3, "guided": 1.5e-2, "guided_gradcam": 1.5e-2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-dir", required=True, help="article project outputs/ with exp_001_test_<arch>.csv")
    ap.add_argument("--archs", nargs="+", default=["greed", "resnet"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    overrides = ["determinism=false", f"dataset.limit={a.n}", f"orchestrator.shard_size={a.n}", "backgrounds=[mean]",
                 "paths.runs=runs_golden", f"dataset.archs=[{','.join(a.archs)}]"] + a.set
    cfg = load_config("configs/catsdogs.yaml", overrides)
    keep = set(EXACT) | {"cme", "random"}
    tasks = [t for t in all_tasks(cfg, ["gen", "eval"]) if t.method in keep]
    w = Worker(cfg, "golden")
    pending = tasks
    for _ in range(3):  # dependencies resolve within a few passes
        pending = [t for t in pending if not w.run(t)]
        if not pending:
            break

    new = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "runs_golden/catsdogs/metrics/**/*.csv"), recursive=True))
    ok = True
    for arch in a.archs:
        old = pd.read_csv(Path(a.legacy_dir) / f"exp_001_test_{LEGACY_FILE[arch]}.csv")
        old["FileName"] = old.FileName.str.replace("\\", "/", regex=False).str.replace("data/", "", n=1, regex=False)
        old["Method"] = old.Method.replace({"cgd": "cme"})
        m = new[new.Arch == arch].merge(old, on=["FileName", "Method"], suffixes=("", "_old"))
        m["dDel"] = (m.DeletionAUC - m.DeletionAUC_old).abs()
        m["dIns"] = (m.InsertionAUC - m.InsertionAUC_old).abs()
        s = m.groupby("Method")[["dDel", "dIns"]].max()
        print(f"\n== {arch}: max |new - article| over {a.n} images")
        print(s.round(5).to_string())
        worst = s.loc[s.index.isin(list(EXACT))].max(axis=1)
        bad = worst[[worst[m] > EXACT[m] for m in worst.index]]
        if len(bad):
            ok = False
            print(f"   FAIL (exact methods differ): {list(bad.index)}")
    print("\nGOLDEN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
