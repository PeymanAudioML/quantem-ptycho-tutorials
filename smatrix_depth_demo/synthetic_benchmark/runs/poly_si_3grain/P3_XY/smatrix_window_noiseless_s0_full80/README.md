# S-matrix + probe, all three defocus datasets (reference fit)

`smatrix_window_noiseless_s0_full80`

**What:** joint S-matrix + probe fit on all three defocus datasets of P3_XY (8427 patterns, window detector,
noiseless). Dataset 0 is the gauge (its nominal probe is fixed); datasets 1 and 2 learn C10, C12a, C12b. This is the
**three-focus** reconstruction. It is the S-matrix input for QuantEM arms C, D and E (`*_grid35`). It must **not** be
used by the single-focus handoff arms.

**Settings:** 80 epochs, 8 warm-up epochs, simultaneous updates, lr_S 0.02, batch 16, no held-out data, best epoch by
running epoch loss. Ran in several resumed parts (`smatrix_P3_XY_full80*.log`), 189 min fit time.
Geometry check: 0.739 measurements per unknown (passes).

**Results:**
- Probe (dataset 1): C10 −87.91, C12a 6.07, C12b −4.09 Å, i.e. within 0.1 Å of the truth; RMS phase error 0.0035 rad.
  Dataset 2: −44.04 / −2.92 / 4.84 Å (truth −43.3 / −3 / 5).
- Final training loss 1.1e-3.
- Projected estimate vs truth: corr 0.944, slope 0.36.
- Depth: 12 candidates, all near 100 Å with ±61–85 Å uncertainty, none reliable; one focus-curve layer at 108 Å.
  This is the honest answer for columnar grains.

Report: `reports/QUANTEM_P3_XY.md`, sections 2.1 and 5.

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
