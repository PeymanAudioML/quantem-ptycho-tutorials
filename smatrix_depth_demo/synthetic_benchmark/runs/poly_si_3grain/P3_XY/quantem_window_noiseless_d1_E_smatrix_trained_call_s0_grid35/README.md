# QuantEM arm E with the three-focus S-matrix object AND probe (probe trained)

`quantem_window_noiseless_d1_E_smatrix_trained_call_s0_grid35`

**What:** as `E_nominal_trained_*_grid35`, but the probe starts from the dataset-1 probe of the three-focus fit
`smatrix_window_noiseless_s0_full80` (C10 −87.91 Å; 0.0042 rad from the truth after transfer) and is trained from
iteration 41 (probe lr 1e-4). Object start scale 2.5 (initial loss 173 vs 195 with the nominal probe).

**Results:** loss 96.4 → 8.18 (12.6 at iteration 40; 25 min), vs 9.32 for the nominal-probe E. Projected corr 0.9716,
slope 0.949, RMSE 0.403 rad (unchanged from nominal-probe E). Mean slab corr 0.711 (vs 0.671). Depth profile corr −0.06.
Probe **degrades** during training: 0.0042 → 0.063 rad (nominal-probe E ends at 0.038). Not converged: loss still changes
6.6 % over the last 10 % of iterations.

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
