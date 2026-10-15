# QuantEM: uniform vs S-matrix-informed slices, P3_XY

Dataset 1, 200 iterations, probe nominal_trained. Depth selectivity = mean slab corr (vs own true slab) − mean corr vs the whole projection; > 0 means the slices carry depth information.

| run | planes | final loss | projected corr | slope | NRMSE | mean slab corr | mean proj corr | depth selectivity | depth profile corr | probe overlap | probe RMS (rad) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: uniform slices, flat start (25 planes) | 25 | 80.41 | 0.894 | 0.80 | 0.45 | 0.689 | 0.724 | -0.035 | -0.06 | 0.9964 | 0.083 |
| A: uniform slices, flat start (35 planes) | 35 | 74.95 | 0.896 | 0.81 | 0.44 | 0.597 | 0.677 | -0.080 | 0.02 | 0.9955 | 0.094 |
| C: S-informed slices, flat start (35 planes) | 35 | 76.23 | 0.896 | 0.80 | 0.44 | 0.600 | 0.677 | -0.077 | -0.01 | 0.9967 | 0.079 |
| D: S-informed slices + start (35 planes) | 35 | 68.18 | 0.904 | 0.82 | 0.43 | 0.613 | 0.692 | -0.079 | 0.02 | 0.9980 | 0.061 |
| E: S-informed slices, S-matrix object start (35 planes) | 35 | 9.316 | 0.971 | 0.95 | 0.24 | 0.671 | 0.758 | -0.087 | 0.10 | 0.9992 | 0.038 |
