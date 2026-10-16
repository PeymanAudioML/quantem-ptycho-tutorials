# QuantEM arm A, 25 uniform slices, flat start, nominal probe (standalone)

`quantem_window_noiseless_d1_A_nominal_trained_call_s0_pure`

**What:** plain single-dataset QuantEM with no S-matrix input of any kind. 25 uniform 8 Å slabs, flat object
(1e-3), nominal probe refined after the warm-up (log `quantem_P3_XY_pure.log`). Arm A uses no S-matrix input; the
`smatrix_run` field in `summary.json` is a default and is not read.

**Results:** loss 300.6 → 80.4 (15 min). Projected potential vs truth: corr 0.894, slope 0.80. Mean slab corr 0.689
vs projection corr 0.724, so no depth selectivity. Final probe RMS 0.083 rad.

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
