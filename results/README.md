# Results used in the article revision (state of 30.10.2026, release v1.0)

Everything here is produced by the scripts of this repository from `runs/` (which is not in git);
the files are copied so that the numbers of the article can be checked without rerunning anything.

## Coverage (Cats-vs-Dogs, eval-200, six architectures)

- **Main comparison:** canonical CME and 11 competing methods (CAM, Grad-CAM, Grad-CAM++, Guided BP, Guided Grad-CAM, RISE, Score-CAM, MP, EP, LibraGrad, Random).
  - Generation: CME and the perturbation methods with the mean-colour background. CME additionally with blur, which is the equal-area reference for the background-free methods evaluated on blur.
  - Evaluation: mean colour and Gaussian blur σ = 10.
- **`fidelity_T`:** CME snapshots at T = 100 … 4000, all six architectures.
- **Not in this release:**
  - black / noise backgrounds;
  - Adaptive CME maps;
  - sanity checks, complementarity;
  - Oxford-IIIT Pet and ImageNetV2.

  The code for them is included and runs with `scripts/run_all.py`.

## Files

| File | Content | Produced by |
|---|---|---|
| `catsdogs/metrics_per_image.csv` | one row per (architecture, method, generation bg, evaluation bg, image): Deletion/Insertion AUC, Flip, ConfidenceDrop, MarginDelta, IoU with CME, CME fidelity fields | `scripts/run.py` (eval tasks) |
| `catsdogs/tables/summary.csv` | means per (architecture, method, backgrounds) | `scripts/60_stats.py` |
| `catsdogs/tables/paired_cme.csv` | CME vs each method on Gap, diagonal reference (see `cmebench/stats.py`) | `scripts/60_stats.py` |
| `catsdogs/tables/table2_all.csv` | **Table 2:** CME (generated on mean) vs every competitor; one Holm family per architecture and evaluation background | `scripts/61_table2.py` |
| `catsdogs/tables/area_auc_mean_20.csv` | Deletion/Insertion AUC over 0–20% of the area, per image | `scripts/62_area_auc.py` |
| `catsdogs/tables/ranking_stability.csv` | Kendall τ of method rankings between backgrounds | `scripts/60_stats.py` |
| `catsdogs/tables/cme_fidelity.csv` | CME preservation, residual logit error, retained area | `scripts/60_stats.py` |
| `catsdogs/tables/ceiling.csv` | share of images in the Random-map ceiling region (Del and Ins > 0.9) | `scripts/60_stats.py` |
| `catsdogs/fidelity_T/*.csv` | CME fidelity versus the number of iterations T | `run.py --stages extra --only stage=fidelity_T` |

Gap = Deletion AUC − Insertion AUC (lower is better). In `table2_all.csv`, `dGap` = competitor − CME, so positive values favour CME.
