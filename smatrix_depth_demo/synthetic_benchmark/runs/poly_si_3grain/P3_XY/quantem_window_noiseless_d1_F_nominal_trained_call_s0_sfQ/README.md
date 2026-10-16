# Handoff Q1: single-focus S-matrix object, nominal probe

`quantem_window_noiseless_d1_F_nominal_trained_call_s0_sfQ`

**What:** Q1 = arm F × `nominal_trained`. Arm F is arm E's object conversion on the **uniform** 35-slab grid. The
initial object comes from the single-focus joint S-matrix `smatrix_window_noiseless_s0_sf80_single_d1_joint`, and the
probe is nominal. The object scale was picked by QuantEM forward loss on the S-matrix **training positions only**
(2528 patterns). The grid extends past 6 while its top is best; it picked 8.0.

**Results:** loss 89.9 → 9.85 (25 min). Projected corr 0.966, slope 0.94. Slab / projection corr 0.678 / 0.766. Probe RMS
0.086 rad. The object start is what helps: corr 0.896 → 0.966. The
slices are still not depth-selective (slab corr < projection corr).

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
