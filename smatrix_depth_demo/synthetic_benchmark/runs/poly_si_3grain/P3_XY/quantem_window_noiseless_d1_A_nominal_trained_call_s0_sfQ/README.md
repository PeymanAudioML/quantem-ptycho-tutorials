# Handoff Q0: flat object, nominal probe (35 uniform slices)

`quantem_window_noiseless_d1_A_nominal_trained_call_s0_sfQ`

**What:** control of the single-focus handoff (Q0 = arm A × `nominal_trained`). 35 uniform slabs, flat start,
nominal probe. Run together with Q1–Q3 (`run_single_focus_joint.sh`, tag `sfQ`, `--single-focus-handoff`). It uses
nothing from the S-matrix run named in `summary.json`.

**Results:** loss 271 → 74.95 (25 min), identical to `*_A_nominal_trained_call_s0_grid35`. Projected corr 0.896, slope
0.81. Slab / projection corr 0.597 / 0.677. Probe RMS 0.094 rad. Data amplitude loss 0.0258
(`results/single_focus_joint/P3_XY/quantem_data_loss.json`).

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
