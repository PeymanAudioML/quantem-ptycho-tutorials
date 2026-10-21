# Defocus study: si_defocus_series_w200_window_noiseless

Data: `/mnt/data/quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark/data/iman_data/si_defocus_series_w200/window_noiseless`; truth: `/mnt/data/quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark/data/iman_data/si_defocus_series_w200/ground_truth`
Fits: 80 epochs, warm-up 8, batch 8, lr_S 0.02, all patterns, trainable C10/C12a/C12b on every dataset except the reference (fixed: the gauge).

| dataset | nominal defocus (A) = -C10 |
|---|---|
| def0 (reference) | 0 |
| def1 | 50 |
| def2 | 80 |
| def3 | 150 |

Order of adding datasets: def0 -> def3 -> def2 -> def1

## Fitted probe aberrations (C10 / C12a / C12b, A)

| subset | def0 | def1 | def2 | def3 | loss | meas./unknown | fit (h) |
|---|---|---|---|---|---|---|---|
| 1: d0 | 0.00 / 0.00 / 0.00 |  |  |  | 2.210e-04 | 0.33 | 1.86 |
| 1: d2 |  |  | -80.00 / 0.00 / 0.00 |  | 3.184e-04 | 0.33 | 1.86 |
| 4: d0+d1+d2+d3 | 0.00 / 0.00 / 0.00 | -49.85 / -0.00 / -0.00 | -79.71 / -0.01 / 0.01 | -149.78 / -0.01 / -0.00 | 4.680e-04 | 1.32 | 7.29 |
| truth | -0.00 / 0.00 / 0.00 | -50.00 / 0.00 / 0.00 | -80.00 / 0.00 / 0.00 | -150.00 / 0.00 / 0.00 | | | |

## Recovery vs truth

| subset | probe RMS phase error, worst dataset (rad) | slab corr (mean) | projection corr (mean) | selectivity | depth hits | mean depth error (A) | focus curve vs truth | projected corr |
|---|---|---|---|---|---|---|---|---|
| 1: d0 | 0.0000 | 0.915 | 0.929 | -0.013 | 1 / 8 | 55.0 | -0.06 | 0.933 |
| 1: d2 | 0.0000 | 0.919 | 0.932 | -0.014 | 1 / 8 | 55.0 | -0.06 | 0.932 |
| 4: d0+d1+d2+d3 | 0.0042 | 0.919 | 0.933 | -0.013 | 1 / 8 | 60.0 | -0.06 | 0.925 |

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

## Projected potential and atomic columns, 4-dataset fits (added 2026-10-20)

Pixel-density scatter (log) with the least-squares line and the ideal slope-1 line, column-core pixels (<= 0.5 A)
highlighted, peak height per atomic column (25 columns) and the column-averaged radial profile, vs truth.

| fit | file | corr | column peak / truth (0.1 A blurred truth) |
|---|---|---|---|
| QuantEM k4Q32 (32 planes, probe from S-matrix) | `k4_quantem_Q32_projected.png`, `k4_quantem_Q32_columns.png` | 0.983 | 1.25 +- 0.03 |
| S-matrix k4 (mid-depth unwrapped phase, z = 80 A) | `k4_smatrix_projected_columns.png` | 0.925 | 0.31 +- 0.01 |

- The S-matrix estimate is the phase of the refocused wave, not a reconstruction: its column peaks saturate at about
  2.5 rad (truth about 8.5) and its columns are wider, but positions are right.
- The QuantEM plots use the unblurred truth sampled at the QuantEM pixels (`plot_quantem.py`); there the true peaks
  vary 11-16 rad with the sub-pixel position of each column and QuantEM / truth = 0.76 +- 0.05. Si column cores are
  sharper than the 0.1 A pixel, so the peak ratio depends on the truth blur; the table above uses the 0.1 A blur
  for both. Correlation is much less sensitive (0.974 unblurred vs 0.983 blurred).
- Made with `plot_quantem.py` and `plot_smatrix_columns.py`; metrics in `k4_quantem_Q32_metrics.json` and
  `k4_smatrix_projected_metrics.json`.

## Depth sections: S-matrix and QuantEM vs truth (`sections_all_vs_truth.png`)

Rows 1-3: refocused S-matrix phase per subset (probe fixed); rows 4-5: def2 alone with S-matrix and probe (C10, C12)
updated together, started at C10 = -80 (true) and -100; row 6: true +-10 A slabs (S-matrix field). Rows 7-9: QuantEM potential
summed over the slices in the same 20 A slab (k4 runs: F 32 planes probe trained, F 32 planes probe fixed, E 40 planes
probe trained); row 10: true slabs in the QuantEM field. Mean r over the 8 sections: S-matrix 0.915-0.919; QuantEM F
0.938 (trained) / 0.939 (fixed), E 0.916. Each panel has its own grey scale (S-matrix phase vs QuantEM potential).
The true slabs are nearly identical at every depth here (Si [001]), so r measures lateral fidelity, not depth
separation. Redraw: `plot_study_sections.py --name <study> --quantem <subset tag>/<run> ...`.

## Single dataset (def2), S-matrix and probe updated together

`runs/.../k1_d2/smatrix_window_noiseless_s0_jointC80_single_d0_joint` and `..._jointC100_single_d0_joint`
(run_benchmark.py smatrix --single-dataset --single-probe-mode joint, 80 epochs, warm-up 8; C10, C12a, C12b trained).

| run | C10 start (A) | C10 fitted (A) | probe RMS phase error (rad) | slab corr | depth hits | loss |
|---|---|---|---|---|---|---|
| def2, probe fixed (study) | -80 | -80.00 | 0.000 | 0.919 | 1 / 8 | 3.18e-4 |
| def2, S + probe | -80 | -80.06 | 0.004 | 0.919 | 1 / 8 | 3.23e-4 |
| def2, S + probe | -100 | -100.27 | 0.286 | 0.920 | 1 / 8 | 3.23e-4 |

True C10 = -80 A. Started 20 A off, the probe stays 20 A off, yet the loss and the slab correlation are the same:
with one dataset a probe defocus error is absorbed by the S-matrix (a shift of the S-matrix along z), so it cannot
be detected or corrected. Si [001] slabs are the same at every depth, so a 20 A depth shift is invisible in the
slab metrics; a specimen with depth structure would show it. A second defocus fixes the gauge (4-dataset fit:
C10 within 0.3 A of truth).
