# Handoff Q3: single-focus S-matrix object AND probe

`quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfQ`

**What:** Q3 = arm F × `smatrix_trained`. Both the initial object and the initial probe come from
`smatrix_window_noiseless_s0_sf80_single_d1_joint`. Scale 8.0 was chosen on the S-matrix training positions. Probe
transfer overlap 0.99996.

**Results:** loss 89.8 → 9.82 (25 min). Projected corr 0.967, slope 0.95. Slab / projection corr 0.678 / 0.763. Probe RMS
0.086 rad. Data amplitude loss 0.0033 on all patterns and 0.0032 on the 281 S-matrix held-out positions, so the
gain is not tied to the positions the S-matrix was fitted on. The same as Q1: the probe
handover adds nothing, because the joint probe never left nominal.

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
