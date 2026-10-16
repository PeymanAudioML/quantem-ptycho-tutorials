# Handoff Q2: flat object, single-focus joint S-matrix probe

`quantem_window_noiseless_d1_A_smatrix_trained_call_s0_sfQ`

**What:** Q2 = arm A × `smatrix_trained`. Flat start, but the initial probe is the probe of the single-focus joint
S-matrix run `smatrix_window_noiseless_s0_sf80_single_d1_joint`. 35 uniform slabs.

**Transfer check:** QuantEM's initial probe vs the S-matrix probe at the beams: overlap 0.99996, defocus sign correct,
intensity ratio 1.000.

**Results:** loss 271 → 75.05 (25 min). Projected corr 0.896, slope 0.81. Probe RMS 0.094 rad. The same as Q0, because the
transferred probe equals the nominal one (C10 −100.04 vs −100), so handing over the probe changes nothing.

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
