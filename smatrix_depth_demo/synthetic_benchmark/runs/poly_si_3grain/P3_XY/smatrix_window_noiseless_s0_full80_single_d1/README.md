# S-matrix, dataset 1 only, probe fixed (first equal-data reference)

`smatrix_window_noiseless_s0_full80_single_d1`

**What:** S-matrix fitted on dataset 1 alone (2809 patterns), the same data QuantEM sees. The nominal probe is fixed
(`--single-dataset`, before the joint mode existed). The fit is underdetermined (0.246 measurements per unknown vs
the 0.7 limit) and ran only with `--allow-underdetermined`. The failed check is saved in `summary.json`.

**Settings:** as `full80`, but one dataset and a fixed probe: 80 epochs, 8 warm-up epochs, lr_S 0.02, batch 16, **no
held-out data**, best epoch by **running** epoch loss (epoch 77). 72 min. Log: `smatrix_P3_XY_full80_single_d1.log`.

**Results:**
- Probe: nominal (C10 −100, no astigmatism), RMS phase error 0.282 rad vs the truth.
- Training loss 2.1e-5, 50× lower than the three-dataset fit, yet every object measure is worse (overfitting).
- Projected estimate: corr 0.934, slope 0.14.
- Depth: invents two layers at 56 and 174 Å. All 50 candidates fall into these two clusters, and none is reliable.

Truth for dataset 1 (evaluation only): C10 −88.0, C12a 6.0, C12b −4.0 Å. P3_XY is three columnar grains running through the full 200 Å thickness, so it has no depth structure: any localised depth peak is false.

Superseded for the fixed-vs-joint comparison by `smatrix_window_noiseless_s0_sf80_single_d1`, which has a held-out
split and fresh-loss checkpoint selection. Report: `reports/QUANTEM_P3_XY.md` section 5.

## Files

| file | content |
|---|---|
| `summary.json` | settings, geometry check, losses, probe coefficients, depth candidates (also copied to `results/smatrix/poly_si_3grain/P3_XY/<run>/`) |
| `arrays.npz` | depth grid, focus curve, loss history, probe at the 1313 beams, beam vectors, held-out mask |
| `S.npy` | recovered S-matrix, complex64 (1313 beams × S grid); ~400 MB |
| `refocused_phase.npy`, `refocused_amplitude.npy` | S refocused from −40 to 240 Å in 2 Å steps, cropped to the scan field (phase normalised by the slice mean) |
| `best.pt` | S and probe corrections at the selected epoch |
| `checkpoint.pt` | S, probe, both Adam states and RNG state for `--resume` |
| `monitor/epoch_NNN.npz` | snapshots for `plot_fit_monitor.py` (loss, coefficients, focus curve, phase sections, mid-depth wave) |

Only this README is tracked in git (`runs/` is gitignored); the small outputs are in `results/smatrix/`.
