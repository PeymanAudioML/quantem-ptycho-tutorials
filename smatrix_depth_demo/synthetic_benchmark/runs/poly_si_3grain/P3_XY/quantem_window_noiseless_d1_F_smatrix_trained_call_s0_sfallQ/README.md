# Handoff from the all-pattern single-focus S-matrix: arm F (35 uniform slices)

`quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ`

**What:** as Q3, but the object and probe come from `smatrix_window_noiseless_s0_sfall_single_d1_joint` (all patterns,
no held-out split). 35 uniform slabs, scale 6.0 picked on all patterns. Made by another session.

**Results:** loss 94.2 → 10.9 (25 min). Projected corr 0.963, slope 0.94. Slab / projection corr 0.701 / 0.790. Probe RMS
0.065 rad. Data amplitude loss 0.0037. Close to Q3: the extra 10 % of patterns in the S-matrix does not change the
handoff.

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
