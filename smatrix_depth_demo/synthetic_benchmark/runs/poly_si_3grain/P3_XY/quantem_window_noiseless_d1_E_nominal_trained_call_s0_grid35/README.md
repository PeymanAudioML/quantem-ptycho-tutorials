# QuantEM arm E, S-informed slices + S-matrix object start (three-focus S)

`quantem_window_noiseless_d1_E_nominal_trained_call_s0_grid35`

**What:** initial object = the **three-focus** S-matrix's own projected estimate (phase of the S refocused to T/2,
unwrapped, spread over the slabs by the focus curve), scaled by the factor with the lowest QuantEM forward loss
(grid 0.5–6; picked 2.5). 35 S-informed slabs, nominal probe.

**Results:** loss 135.4 → 9.3 (56 min). Projected corr 0.971, slope 0.95. Slab / projection corr 0.671 / 0.758.
Probe RMS 0.038 rad. This was the best arm, but it inherits three-focus information, so it is not an equal-data
comparison.

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
