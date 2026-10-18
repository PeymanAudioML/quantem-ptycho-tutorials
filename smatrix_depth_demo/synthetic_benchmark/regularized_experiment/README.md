# Regularized experiment

Code and results for the regularization study: penalties on the single-defocus S-matrix fit, and depth penalties in
QuantEM for P3_XYZ. Plan: [`../reports/PLAN_SMATRIX_REGULARIZATION.md`](../reports/PLAN_SMATRIX_REGULARIZATION.md).
Everything for this study lives here; the shared benchmark code is not modified.

**Specimen: P3_XYZ** (3 Si grains stacked in depth: [001] on top, [110] middle, [111] bottom, tilted boundaries). This
is not the P3_XY specimen (columnar grains) used by the earlier runs in `../runs/poly_si_3grain/P3_XY/`. Full P3_XYZ
runs are written by the runner to `../runs/poly_si_3grain/P3_XYZ/`; their summaries and plots are copied here.

| folder | content |
|---|---|
| `results/step0_gt_axial_tv/`, `results/step0_gt_axial_tv_slices/` | P3_XYZ ground truth only (no reconstruction) |
| `results/tvz_sweep/` | P3_XYZ QuantEM depth TV `tv_weight_z` = 0/10/100/1000 at 20 slices, arm F; [README](results/tvz_sweep/README.md), from [`run_tvz_sweep.sh`](run_tvz_sweep.sh) |
| `results/tvz_35sl/` | same at 35 slices (w = 1000, 1750) vs 20 slices; [README](results/tvz_35sl/README.md), from [`run_tvz35.sh`](run_tvz35.sh) |
| `results/p3xyz_baseline/` | P3_XYZ unregularized baseline: QuantEM A, S-matrix fit, QuantEM E/F (dataset 1, all patterns); done 2026-10-17, [README](results/p3xyz_baseline/README.md), from [`run_p3xyz_baseline.sh`](run_p3xyz_baseline.sh) |

| step | code | results | status |
|---|---|---|---|
| 0: ground-truth check of axial TV on P3_XYZ | [`step0_gt_axial_tv.py`](step0_gt_axial_tv.py) | [`results/step0_gt_axial_tv/`](results/step0_gt_axial_tv/README.md) (+ `_slices` cross-check) | done 2026-10-17 |

Run scripts from `synthetic_benchmark/` with the repo environment: `../../.venv/bin/python regularized_experiment/<script>`.

## Step 0 in one paragraph

On the true potential at QuantEM's 35 slices, the slab-to-slab change separates grain boundaries from grain interiors
only weakly (AUC 0.68), and 91 % of an axial TV penalty would fall inside grains. Lateral low-pass makes it worse (the
grains are all Si). Thicker slices (10 Å) help a lot (AUC 0.81). A local orientation fingerprint separates boundaries
best (AUC 0.89–0.97). Next candidate: QuantEM's built-in `tv_weight_z` with 20 slices of 10 Å.
