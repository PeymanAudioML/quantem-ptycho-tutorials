# QuantEM arm D, S-informed slices + seeded candidate spots (three-focus S)

`quantem_window_noiseless_d1_D_nominal_trained_call_s0_grid35`

**What:** arm C plus an initial potential with the 12 three-focus S-matrix depth candidates seeded as spots
(`quantem_export.initial_potential`), nominal probe.

**Results:** loss 267 → 68.2 (26 min). Projected corr 0.904, slope 0.82. Slab / projection corr 0.613 / 0.692.
Probe RMS 0.061 rad. A small gain over A and C.

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
