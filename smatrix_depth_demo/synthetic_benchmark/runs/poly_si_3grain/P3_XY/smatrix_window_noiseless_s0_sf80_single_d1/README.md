# Single-focus S-matrix, probe FIXED (seed 0, held-out split)

`smatrix_window_noiseless_s0_sf80_single_d1`

**What:** control arm of the single-focus fixed-vs-joint comparison. S-matrix on dataset 1 only, nominal probe held
fixed (`--single-probe-mode fixed`, the default).

**Command:** `scripts/run_single_focus_joint.sh` (2nd step):
`run_benchmark.py smatrix --config configs/poly_si_3grain.yaml --specimen P3_XY --single-dataset --dataset 1
--allow-underdetermined --epochs 80 --warmup 8 --batch 16 --lr-S 0.02 --mode simultaneous --heldout 0.1
--split-seed 0 --select-every 1 --checkpoint-every 10 --monitor-every 10 --seeds 0 --tag sf80`

**Settings:** vacuum S start, 80 epochs, cosine lr decay. **10 % of scan positions held out** (281 of 2809, split seed
0), never used for fitting or selection. Best epoch chosen by a **fresh full-training-set loss** evaluated after every
epoch (epoch 76 selected). 69 min.

**Results (ground truth used only for evaluation):**
- Training loss 6.7e-6, held-out loss 0.107: 16,000× higher. The fit memorises the training positions.
- Probe: nominal, overlap 0.961, RMS phase error 0.282 rad.
- Projected estimate (T/2 refocus, unwrapped): corr 0.913, slope 0.115.
- Depth: focus-curve layers at 54 and 178 Å, both false. Sections show no depth selectivity (slab − projection corr
  −0.009), and the focus curve vs the true profile has corr 0.04. 50 candidates: 20 shallow, 30 deep, none reliable.

Truth for dataset 1 (evaluation only): C10 −88.0, C12a 6.0, C12b −4.0 Å. P3_XY is three columnar grains running through the full 200 Å thickness, so it has no depth structure: any localised depth peak is false.

The joint counterpart is `..._sf80_single_d1_joint`. Seeds 1–2 were never run (the queue stopped after the QuantEM step).

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
