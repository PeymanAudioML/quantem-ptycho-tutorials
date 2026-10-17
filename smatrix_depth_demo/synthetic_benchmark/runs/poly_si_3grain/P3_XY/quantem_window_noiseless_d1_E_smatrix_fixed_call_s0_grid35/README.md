# QuantEM arm E with the three-focus S-matrix object AND probe (probe fixed)

`quantem_window_noiseless_d1_E_smatrix_fixed_call_s0_grid35`

**What:** as `E_smatrix_trained_*_grid35`, but the three-focus S-matrix probe (0.0042 rad from the truth) is never
updated. Object start scale 2.5.

**Results:** loss 8.27 (12.6 at iteration 40; 25 min). Projected corr 0.9721, slope 0.950, RMSE 0.401 rad: the best
projection of any P3_XY run, by a small margin. Mean slab corr 0.705. Depth profile corr −0.06. Probe stays at
0.0042 rad. Not converged: loss still changes 5.9 % over the last 10 % of iterations.

## Files

| file | content |
|---|---|
| `summary.json` | arm, probe mode, slice boundaries, learning rates, loss history, object-init and probe-transfer info (also in `results/quantem/poly_si_3grain/P3_XY/<run>/`) |
| `quantem.npz` | `object_potential` (slices × 120 × 120, rad), `initial_potential`, `slice_boundaries`, `probe`, `initial_probe`, `loss` |
| `plots/` | `plot_quantem.py` output vs ground truth: `slices.png`, `projected.png`, `depth.png`, `probe.png`, `metrics.json` |

Only this README is tracked in git (`runs/` is gitignored).

Command (both probe modes in one call):
`$PY run_benchmark.py quantem --config configs/poly_si_3grain.yaml --specimen P3_XY --smatrix-run smatrix_window_noiseless_s0_full80 --arms E --probes smatrix_trained,smatrix_fixed --iters 200 --device cuda --tag grid35`

QuantEM settings: dataset 1 only (nominal C10 = −100 Å, true −88 Å), 35 S-informed slabs, 200 iterations, Adam object
lr 5e-3, batch 64, positivity on the object, object-only warm-up of 40 iterations.
