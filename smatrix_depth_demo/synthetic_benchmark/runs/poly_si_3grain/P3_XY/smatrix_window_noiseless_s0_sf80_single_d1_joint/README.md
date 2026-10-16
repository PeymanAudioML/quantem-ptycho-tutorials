# Single-focus S-matrix, probe updated JOINTLY (experimental, seed 0, held-out split)

`smatrix_window_noiseless_s0_sf80_single_d1_joint`

**What:** the experimental single-focus joint mode. S-matrix and probe aberrations C10, C12a, C12b are updated together
from dataset 1 alone (`--single-probe-mode joint`, `JointModel(ref=None)`: no gauge dataset; the aperture amplitude
stays fixed). The probe phase is **not identifiable** from one dataset: any per-beam probe phase can be moved into S
without changing a single pattern (`tests/test_single_focus_joint.py` verifies this to <1e-5). This run is also the
S-matrix source for the QuantEM handoff arms `*_sfQ`.

**Command:** `scripts/run_single_focus_joint.sh` (1st step): same as the fixed run plus
`--single-probe-mode joint --trainable C10,C12a,C12b --lr-probe 0.005`.

**Settings:** identical to `..._sf80_single_d1` (data, held-out split, vacuum S start, 80 epochs, lr_S schedule, fresh
full-training-set selection). Probe: lr 0.005 (Adam, rad at the aperture edge), the same cosine decay, updates from
epoch 9 after the 8-epoch warm-up. 71 min.

**Results (ground truth used only for evaluation):**
- Probe update check passed: gradients finite and nonzero in all 72 post-warm-up epochs, and the coefficients changed.
  But they moved at most 0.22 Å and ended at C10 −100.04, C12a 0.07, C12b 0.05 Å, i.e. at the start, not the truth.
  The probe gradient fell ~130× as S absorbed the mismatch.
- Probe overlap 0.961, RMS phase error 0.282 rad: the same as the fixed probe.
- Training loss 6.7e-6, held-out 0.107: the same as fixed.
- Projected corr 0.913, slope 0.115. Layers 54 / 178 Å. Selectivity −0.009. All the same as fixed.
- **Conclusion for seed 0: joint single-focus recovery did not improve the probe, the projected object or the depth.**
  The result is indistinguishable from the fixed probe.

Truth for dataset 1 (evaluation only): C10 −88.0, C12a 6.0, C12b −4.0 Å. P3_XY is three columnar grains running through the full 200 Å thickness, so it has no depth structure: any localised depth peak is false.

Evaluation: `evaluate_single_focus.py` → `results/single_focus_joint/P3_XY/`. Seeds 1–2 and the alternative probe starts
(C10 −130 / −70, astigmatic) were not run.

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
