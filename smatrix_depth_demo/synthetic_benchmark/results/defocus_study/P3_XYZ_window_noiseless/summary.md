# Defocus study: P3_XYZ_window_noiseless

Data: `/mnt/data/quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark/data/poly_si_3grain/poly_si_3grain_xyz/P3_XYZ/window_noiseless`; truth: `/mnt/data/quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark/data/poly_si_3grain/poly_si_3grain_xyz/P3_XYZ/ground_truth`
Fits: 80 epochs, warm-up 8, batch 16, lr_S 0.02, all patterns, trainable C10/C12a/C12b on every dataset except the reference (fixed: the gauge).

| dataset | nominal defocus (A) = -C10 |
|---|---|
| def0 (reference) | 166.7 |
| def1 | 100 |
| def2 | 33.3 |

Order of adding datasets: def0 -> def2 -> def1

## Fitted probe aberrations (C10 / C12a / C12b, A)

| subset | def0 | def1 | def2 | loss | meas./unknown | fit (h) |
|---|---|---|---|---|---|---|
| 1: d0 | -166.70 / 0.00 / 0.00 |  |  | 2.078e-05 | 0.25 | 1.19 |
| 2: d0+d2 | -166.70 / 0.00 / 0.00 |  | -42.95 / -2.91 / 4.84 | 6.297e-04 | 0.49 | 2.37 |
| 3: d0+d2+d1 | -166.70 / 0.00 / 0.00 | -87.85 / 6.07 / -4.09 | -43.36 / -2.94 / 4.85 | 9.323e-04 | 0.74 | 3.53 |
| truth | -166.70 / 0.00 / 0.00 | -88.00 / 6.00 / -4.00 | -43.30 / -3.00 / 5.00 | | | |

## Recovery vs truth

| subset | probe RMS phase error, worst dataset (rad) | slab corr (mean) | projection corr (mean) | selectivity | depth hits | mean depth error (A) | focus curve vs truth | projected corr | grain accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 1: d0 | 0.0000 | 0.838 | 0.728 | +0.110 | 6 / 10 | 8.0 | 0.15 | 0.824 | 0.890 |
| 2: d0+d2 | 0.0079 | 0.882 | 0.805 | +0.078 | 6 / 10 | 8.0 | 0.09 | 0.850 | 0.861 |
| 3: d0+d2+d1 | 0.0042 | 0.882 | 0.811 | +0.072 | 5 / 10 | 10.0 | 0.08 | 0.855 | 0.859 |

Metrics (truth used here only):

- **probe RMS phase error**: fitted vs true probe inside the aperture, piston and tilt removed; worst dataset of the subset. The reference dataset is fixed at its nominal value, so it is 0 if that is exact.
- **slab corr**: refocused phase at z = 10, 30, ... A vs the true potential of the +-10 A slab around z (mean over sections). **projection corr**: the same sections vs the whole projected potential. **selectivity** = slab - projection: > 0 means the sections carry depth information beyond the projection.
- **depth hits**: sections whose best-matching true slab is the right one. **mean depth error**: |z of best-matching slab - z| averaged over sections (20 A grid).
- **focus curve vs truth**: correlation of the S-matrix focus curve with the true lateral-variance depth profile.  **projected corr**: mid-depth unwrapped phase vs the true projected potential.
- **grain accuracy**: 3D grain map (regularized_experiment/smatrix_grains_3d.py) vs the true grain layout; chance = majority class.

## Files

- `metrics_vs_defocus.png`: metrics vs number of defocus datasets
- `sections_vs_truth.png`: refocused S-matrix sections per subset (and truth)
- `depth_confusion.png`: section vs true slab correlation matrices (diagonal = right depth)
- `summary.json`: all numbers; `configs/`: generated configs; `logs/`: fit logs
- fits: `runs/<config name>/<subset>/<run>/` (S.npy, refocused_phase.npy, summary.json)
