# Handoff from the all-pattern single-focus S-matrix: arm E (S-informed slices)

`quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ`

**What:** object, probe **and** slice grid from `smatrix_window_noiseless_s0_sfall_single_d1_joint` (joint, all 2809
patterns). Its 50 wide candidates saturate the slab cap, which gives **40** uniform 5 Å slabs (not 35), so this is not
matched to Q0–Q3. Scale picked on all patterns (the S run had no held-out split): 6.0 (8.0 was worse). Made by another
session (`run_single_focus_all.sh`).

**Results:** loss 93.5 → 11.3 (31 min). Projected corr 0.962, slope 0.94. Slab / projection corr 0.659 / 0.767. Probe RMS
0.071 rad. Data amplitude loss 0.0038.

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
