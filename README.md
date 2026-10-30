# cme-bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23808635.svg)](https://doi.org/10.5281/zenodo.23808635)

Reproducible benchmark for **Competitive Mask Erosion (CME)** — an input-space, architecture-agnostic
saliency method — against CAM, Grad-CAM, Grad-CAM++, Guided Backpropagation, Guided Grad-CAM, RISE,
Meaningful Perturbation, Extremal Perturbations, Score-CAM and LibraGrad, on six classifiers
(ResNet50, VGG19, MobileNetV3, GREED, ViT-B/16, MLP-Mixer-B/16), with four perturbation backgrounds
(mean colour, black, uniform noise, blur) and two tasks (Dogs vs. Cats; ImageNetV2 as a multiclass control).

The CME code, model definitions and evaluator are ported unchanged from the article project
(`exp_001_*`); see `tests/test_golden.py`, which checks that this repository reproduces the article's
per-image metrics.

## Quick start

```bash
./install.sh                    # Linux / Git-Bash      (Windows cmd: install.cmd)
source .venv/bin/activate       # Windows: .venv\Scripts\activate.bat

python scripts/10_get_data.py --dataset catsdogs imagenetv2      # download + SHA-256 verification
python scripts/20_get_models.py --release v1.0                   # weights from the GitHub Release
python scripts/20_get_models.py --imagenet --no-accuracy         # cache ImageNet weights

python scripts/run.py --config configs/smoke.yaml                # 5-minute smoke test
python scripts/run.py --config configs/catsdogs.yaml             # full benchmark (days)
python scripts/status.py --config configs/catsdogs.yaml --watch 300
```

Two machines: `run.py --machine-id A --part 0/2` and `--machine-id B --part 1/2`, then merge
`runs/` and `logs/` by copying (every task writes its own files). `status.py` shows the merged progress.

## Layout

| Path | Content |
|---|---|
| `configs/` | `default.yaml` (single source of truth: CME canon, backgrounds, method parameters, priorities), `catsdogs.yaml`, `imagenetv2.yaml`, `smoke.yaml` |
| `lists/` | frozen evaluation lists + SHA-256 of every listed image and of the weights |
| `cmebench/core/` | ported article code: CME (`cme.py`, `cme_fast.py`), Grad-CAM family (`gradcam.py`), evaluator (`metrics.py`), preprocessing |
| `cmebench/methods/` | uniform explainer interface; new methods: `rise.py`, `meaningful_perturbation.py`, `extremal_perturbation.py` (TorchRay), `scorecam.py`, `libragrad.py` |
| `cmebench/backgrounds.py` | mean / black / noise / blur, built in pixel space and normalized per architecture |
| `cmebench/tasks.py`, `worker.py` | task model (gen / eval), execution, `.done` markers with config hashes |
| `scripts/` | `00_check_env`, `10_get_data`, `12_make_imagenetv2_list`, `20_get_models`, `train/`, `run.py`, `status.py` |

Results: `runs/<dataset>/maps/<arch>/<method>/<gen_bg>/<image>.npz` and
`runs/<dataset>/metrics/<arch>/<method>/<gen_bg>/<eval_bg>/shardNNN.csv` (same columns as the article's
`exp_001_test_*.csv`, plus `Dataset, GenBg, EvalBg` and CME fidelity fields).

## Protocol notes

* Labels: Cat = 0, Dog = 1. GREED takes raw [0,1] input; every other model ImageNet normalization
  (ImageNet Mixer: 0.5/0.5).
* Preprocessing (as in the article code): bicubic resize to 224×224, no centre crop.
* CME canon: T = 1000, Adam lr 0.05, θ₀ = 4, w_class 0.05, w_blur (w_reg) 0.95, w_sharp 0.5, squared
  Sobel smoothness, logit preservation, equal-area cap 20 %. For K > 2 classes the logit MSE is scaled by K/2.
* Backgrounds are built in pixel space; noise uses different seeds for map generation and evaluation.
* Extremal Perturbations use the authors' TorchRay (CC BY-NC 4.0), installed with `--no-deps`, not vendored.

## Full run order (what the article needs)

One command runs everything below in order and can be restarted at any time:
`python scripts/run_all.py [--machine-id A --part 0/2]` (log: `logs/run_all_<id>.log`).

```bash
# 0. once per machine
./install.sh && source .venv/bin/activate
python scripts/10_get_data.py --dataset catsdogs imagenetv2 oxford
python scripts/20_get_models.py --release v1.0 --imagenet
python tests/test_golden.py --legacy-dir <article>/outputs --n 5      # optional, needs the article CSVs

# 1. lists (already committed — rebuild only to audit)
python scripts/12_make_imagenetv2_list.py
python scripts/13_make_oxford_list.py

# 2. main comparison: 12 methods x 4 backgrounds (generation x evaluation)
python scripts/run.py --config configs/catsdogs.yaml       [--machine-id A --part 0/2]
python scripts/run.py --config configs/imagenetv2.yaml
python scripts/run.py --config configs/oxford.yaml

# 3. stages for the reviewer
python scripts/run.py --config configs/catsdogs.yaml --stages extra --only stage=fidelity_T
python scripts/run.py --config configs/catsdogs.yaml --stages extra --only stage=sanity
python scripts/run.py --config configs/catsdogs.yaml --stages extra --only stage=adaptive
python scripts/run.py --config configs/catsdogs.yaml --only method=cme_adaptive      # Adaptive CME maps + metrics
python scripts/run.py --config configs/catsdogs.yaml --stages extra --only stage=complementarity
python scripts/run.py --config configs/oxford.yaml   --stages extra --only stage=oxford
python scripts/run.py --config configs/oxford.yaml   --stages extra --only stage=shortcut

# 4. tables and figures
python scripts/60_stats.py   --config configs/catsdogs.yaml   (and imagenetv2, oxford)
python scripts/61_table2.py  --config configs/catsdogs.yaml   # Table 2 of the article
python scripts/62_area_auc.py --config configs/catsdogs.yaml  # Deletion/Insertion AUC over 0-20 % of the area
python scripts/65_figures.py --config configs/catsdogs.yaml
python scripts/status.py     --config configs/catsdogs.yaml --failed
```

## Results, weights, citation

* **Results** behind the article revision (Cats-vs-Dogs, mean and blur backgrounds): [`results/`](results/README.md).
* **Weights** of the six classifiers: GitHub Release [v1.0](https://github.com/Dr-ey/cme-bench/releases/tag/v1.0), downloaded and SHA-256-checked by `scripts/20_get_models.py --release v1.0`.
* **Citation:** see [`CITATION.cff`](CITATION.cff); archived on Zenodo, DOI [10.5281/zenodo.23808635](https://doi.org/10.5281/zenodo.23808635) (all versions). Code: MIT licence ([`LICENSE`](LICENSE)); Extremal Perturbations rely on TorchRay (CC BY-NC 4.0), which is installed separately and not distributed here.

## Status of stages

| Stage | Reviewer item | State |
|---|---|---|
| 12 methods × 4 generation × 4 evaluation backgrounds, Cats-vs-Dogs / ImageNetV2 / Oxford | M2, M3–M5 | implemented, smoke-tested |
| Golden test vs the article (CAM/Grad-CAM/Grad-CAM++ ≤ 1e-5; Guided ≤ 0.012) — PASS on 6 archs | reproducibility | implemented |
| CME fidelity fields (preservation, logit MSE, retained area) in every CME map | major 4 | implemented |
| `fidelity_T` — T = 100…4000 snapshots | major 4 | implemented, smoke-tested |
| `sanity` — cascading / label randomization / class sensitivity | major 3 | implemented, smoke-tested |
| `adaptive` — 100-image pool, held-out validation, fallback to canonical, selection on 'blur' | M7–M8 | implemented, smoke-tested |
| `oxford` — energy pointing game / IoU with trimaps | M11–M13 | implemented, smoke-tested |
| `shortcut` — controlled background shortcut (biased vs clean heads) | M11–M13 | implemented, smoke-tested (biased head: 5 % acc on swapped backgrounds) |
| `complementarity` — CME-only / competitor-only / shared regions, combined map | M11–M13 | implemented, smoke-tested |
| Statistics: Holm, bootstrap CI, rank-biserial, ceiling shares, ranking stability | M9 | implemented |
| Figures | — | implemented |
