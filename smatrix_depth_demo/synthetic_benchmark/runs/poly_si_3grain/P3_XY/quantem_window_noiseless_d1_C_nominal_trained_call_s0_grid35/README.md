# QuantEM arm C, S-informed slices (three-focus S), flat start

`quantem_window_noiseless_d1_C_nominal_trained_call_s0_grid35`

**What:** 35 slabs placed by `quantem_export.slice_grid` from the depth candidates of the **three-focus** S-matrix
(`smatrix_window_noiseless_s0_full80`), flat start, nominal probe.

**Results:** loss 275.6 → 76.2 (56 min). Projected corr 0.896, slope 0.80. Slab / projection corr 0.600 / 0.677.
Probe RMS 0.079 rad. Same as arm A: the slice placement adds nothing on columnar grains.

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
