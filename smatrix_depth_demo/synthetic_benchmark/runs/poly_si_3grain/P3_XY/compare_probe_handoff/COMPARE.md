# QuantEM: probe handoff from the three-focus S-matrix (arm E), P3_XY

Dataset 1, 200 iterations, probe nominal_trained, smatrix_trained, smatrix_fixed. Depth selectivity = mean slab corr (vs own true slab) − mean corr vs the whole projection; > 0 means the slices carry depth information.

| run | planes | final loss | projected corr | slope | NRMSE | mean slab corr | mean proj corr | depth selectivity | depth profile corr | probe overlap | probe RMS (rad) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E-nom: nominal probe, trained | 35 | 9.316 | 0.971 | 0.95 | 0.24 | 0.671 | 0.758 | -0.087 | 0.10 | 0.9992 | 0.038 |
| E-St: S-matrix probe, trained | 35 | 8.18 | 0.972 | 0.95 | 0.24 | 0.711 | 0.799 | -0.088 | -0.06 | 0.9980 | 0.063 |
| E-Sf: S-matrix probe, fixed | 35 | 8.266 | 0.972 | 0.95 | 0.23 | 0.705 | 0.793 | -0.088 | -0.06 | 1.0000 | 0.004 |
