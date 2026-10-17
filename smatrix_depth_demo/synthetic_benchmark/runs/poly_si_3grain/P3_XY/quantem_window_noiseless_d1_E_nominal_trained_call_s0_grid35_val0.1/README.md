# QuantEM arm E, nominal probe (0.282 rad from the truth), trained from iteration 41, 10 % held-out patterns

`quantem_window_noiseless_d1_E_nominal_trained_call_s0_grid35_val0.1`

**What:** repeat of `quantem_window_noiseless_d1_E_nominal_trained_call_s0_grid35` with QuantEM's held-out split (`--val-ratio 0.1`, grid: every 10th pattern, 281 of 2809
held out, trained on 2528). The held-out patterns are never trained on and are also excluded when the object start
scale is chosen (scale 2.5, picked on 2528 patterns). Report section 6.

**Results:** training loss 141.4 → 9.299; held-out loss 10.849 at the end, best 10.305 at
iteration 156 (23 min). Projected corr 0.9709, slope
0.946. Mean slab corr 0.682. Probe RMS 0.282 →
0.0390 rad. Comparison: [`../compare_heldout/`](../compare_heldout/COMPARE.md).

## Files

| file | content |
|---|---|
| `summary.json` | settings incl. `val_ratio`, `n_val`, `n_train`, `val_loss_history`, `final_val_loss`, `best_val_iter` (also in `results/quantem/poly_si_3grain/P3_XY/<run>/`) |
| `quantem.npz` | `object_potential`, `initial_potential`, `slice_boundaries`, `probe`, `initial_probe`, `loss`, `val_loss`, `val_indices` |
| `plots/` | `plot_quantem.py` output vs ground truth: `slices.png`, `projected.png`, `depth.png`, `probe.png`, `metrics.json` |

Only this README is tracked in git (`runs/` is gitignored).

Command (all three held-out runs in one call):
`$PY run_benchmark.py quantem --config configs/poly_si_3grain.yaml --specimen P3_XY --smatrix-run smatrix_window_noiseless_s0_full80 --arms E --probes nominal_trained,smatrix_trained,smatrix_fixed --val-ratio 0.1 --iters 200 --device cuda --tag grid35`
