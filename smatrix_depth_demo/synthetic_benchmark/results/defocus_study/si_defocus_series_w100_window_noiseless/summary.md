# Defocus study: si_defocus_series_w100_window_noiseless

Data: `/mnt/data/quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark/data/iman_data/si_defocus_series_w100/window_noiseless`; truth: `/mnt/data/quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark/data/iman_data/si_defocus_series_w100/ground_truth`
Fits: 80 epochs, warm-up 8, batch 16, lr_S 0.02, all patterns, trainable C10/C12a/C12b on every dataset except the reference (fixed: the gauge).

| dataset | nominal defocus (A) = -C10 |
|---|---|
| def0 (reference) | 0 |
| def1 | 60 |

Order of adding datasets: def0 -> def1

## Fitted probe aberrations (C10 / C12a / C12b, A)

| subset | def0 | def1 | loss | meas./unknown | fit (h) |
|---|---|---|---|---|---|
| 1: d0 | 0.00 / 0.00 / 0.00 |  | 3.071e-04 | 0.72 | 0.11 |
| 1: d1 |  | -60.00 / 0.00 / 0.00 | 4.536e-04 | 0.72 | 0.11 |
| 2: d0+d1 | 0.00 / 0.00 / 0.00 | -59.19 / -0.02 / -0.04 | 9.018e-04 | 1.43 | 0.22 |
| truth | -0.00 / 0.00 / 0.00 | -60.00 / 0.00 / 0.00 | | | |

## Recovery vs truth

| subset | probe RMS phase error, worst dataset (rad) | slab corr (mean) | projection corr (mean) | selectivity | depth hits | mean depth error (A) | focus curve vs truth | projected corr |
|---|---|---|---|---|---|---|---|---|
| 1: d0 | 0.0000 | 0.914 | 0.927 | -0.013 | 2 / 8 | 52.5 | -0.06 | 0.931 |
| 1: d1 | 0.0000 | 0.911 | 0.925 | -0.013 | 1 / 8 | 55.0 | -0.05 | 0.929 |
| 2: d0+d1 | 0.0114 | 0.916 | 0.930 | -0.013 | 1 / 8 | 57.5 | -0.06 | 0.922 |

Metrics (truth used here only):

- **probe RMS phase error**: fitted vs true probe inside the aperture, piston and tilt removed; worst dataset of the subset. The reference dataset is fixed at its nominal value, so it is 0 if that is exact.
- **slab corr**: refocused phase at z = 10, 30, ... A vs the true potential of the +-10 A slab around z (mean over sections). **projection corr**: the same sections vs the whole projected potential. **selectivity** = slab - projection: > 0 means the sections carry depth information beyond the projection.
- **depth hits**: sections whose best-matching true slab is the right one. **mean depth error**: |z of best-matching slab - z| averaged over sections (20 A grid).
- **focus curve vs truth**: correlation of the S-matrix focus curve with the true lateral-variance depth profile.  **projected corr**: mid-depth unwrapped phase vs the true projected potential.

## Files

- `metrics_vs_defocus.png`: metrics vs number of defocus datasets
- `sections_vs_truth.png`: refocused S-matrix sections per subset (and truth)
- `depth_confusion.png`: section vs true slab correlation matrices (diagonal = right depth)
- `summary.json`: all numbers; `configs/`: generated configs; `logs/`: fit logs
- fits: `runs/<config name>/<subset>/<run>/` (S.npy, refocused_phase.npy, summary.json)
