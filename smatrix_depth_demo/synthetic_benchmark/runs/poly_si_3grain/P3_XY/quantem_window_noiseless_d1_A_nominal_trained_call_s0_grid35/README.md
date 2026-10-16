# QuantEM arm A, 35 uniform slices, flat start, nominal probe

`quantem_window_noiseless_d1_A_nominal_trained_call_s0_grid35`

**What:** baseline for the three-focus S-matrix comparison (arms A/C/D/E `*_grid35`). 35 uniform slabs (the count of
the S-informed grid, so the arms are matched), flat start, nominal probe. No S-matrix information enters this arm.

**Results:** loss 271 → 75.0 (25 min). Projected corr 0.896, slope 0.81. Slab / projection corr 0.597 / 0.677.
Probe RMS 0.094 rad.

Same configuration as Q0 (`*_A_nominal_trained_call_s0_sfQ`), which reproduces it exactly.
Report: `reports/QUANTEM_P3_XY.md`.

## Files

| file | content |
|---|---|
| `summary.json` | arm, probe mode, slice boundaries, learning rates, loss history, object-init and probe-transfer info (also in `results/quantem/poly_si_3grain/P3_XY/<run>/`) |
| `quantem.npz` | `object_potential` (slices × 120 × 120, rad), `initial_potential`, `slice_boundaries`, `probe`, `initial_probe`, `loss` |
| `plots/` | `plot_quantem.py` output vs ground truth: `slices.png`, `projected.png`, `depth.png`, `probe.png`, `metrics.json` |

Only this README is tracked in git (`runs/` is gitignored).

QuantEM settings shared by every arm here: dataset 1 only (nominal C10 = −100 Å, true −88 Å), 200 iterations, Adam
object lr 5e-3, batch 64, positivity on the object, object-only warm-up of 40 iterations, then object + probe
(probe lr 1e-4).
