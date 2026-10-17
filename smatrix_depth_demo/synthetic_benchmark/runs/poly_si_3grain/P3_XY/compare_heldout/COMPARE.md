# QuantEM: probe handoff with 10 % held-out patterns (arm E), P3_XY

Dataset 1, 200 iterations, probe nominal_trained, smatrix_trained, smatrix_fixed. Depth selectivity = mean slab corr (vs own true slab) − mean corr vs the whole projection; > 0 means the slices carry depth information.

| run | planes | final loss | projected corr | slope | NRMSE | mean slab corr | mean proj corr | depth selectivity | depth profile corr | probe overlap | probe RMS (rad) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E-nom: nominal probe, trained | 35 | 9.299 | 0.971 | 0.95 | 0.24 | 0.682 | 0.770 | -0.088 | 0.09 | 0.9992 | 0.039 |
| E-St: S-matrix probe, trained | 35 | 8.536 | 0.972 | 0.95 | 0.24 | 0.718 | 0.804 | -0.086 | -0.05 | 0.9980 | 0.063 |
| E-Sf: S-matrix probe, fixed | 35 | 8.506 | 0.972 | 0.95 | 0.24 | 0.709 | 0.797 | -0.088 | -0.01 | 1.0000 | 0.004 |

**Caveat (added by hand):** the 281 held-out patterns are held out from QuantEM only. The S-matrix fit
`smatrix_window_noiseless_s0_full80`, which supplies the object start of all three runs and the probe of E-St/E-Sf,
had no held-out split and trained on all 2809 dataset-1 patterns. The three runs share this leak, so they compare
fairly with each other, but the held-out loss is not a clean generalisation test of the S-matrix start. Report 6.1.
