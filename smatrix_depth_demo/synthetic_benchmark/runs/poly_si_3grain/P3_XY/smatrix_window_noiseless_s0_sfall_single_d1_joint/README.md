# Single-focus S-matrix, joint probe, ALL patterns (no held-out split)

`smatrix_window_noiseless_s0_sfall_single_d1_joint`

**What:** the joint single-focus fit repeated on all 2809 patterns of dataset 1 (`--heldout 0`), so that the S-matrix
used for the QuantEM handoff sees the same patterns as QuantEM. Made by another session
(`scripts/run_single_focus_all.sh`, log `single_focus_all.log`, commit f4739d5). It is the source for `*_sfallQ`.

**Settings:** identical to `..._sf80_single_d1_joint` except **no held-out data**, so there is no validation loss. Best
epoch by fresh full-training-set loss (epoch 76). 78 min.

**Results (ground truth used only for evaluation):**
- Probe: C10 −100.01, C12a 0.12, C12b 0.01 Å, still at nominal. Overlap 0.961, RMS 0.281 rad.
- Training loss 2.0e-5.
- Projected corr 0.934, slope 0.141. With 10 % more data, these match the earlier fixed-probe run `full80_single_d1`.
- Depth: layers at 56 and 174 Å, both false. Selectivity −0.009. 50 candidates, none reliable.

Truth for dataset 1 (evaluation only): C10 −88.0, C12a 6.0, C12b −4.0 Å. P3_XY is three columnar grains running through the full 200 Å thickness, so it has no depth structure: any localised depth peak is false.

Slice plot: `results/single_focus_joint/P3_XY/smatrix_slices_smatrix_window_noiseless_s0_sfall_single_d1_joint.png`.

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
