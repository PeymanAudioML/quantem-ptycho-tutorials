# Single-focus S-matrix (dataset 1): fixed vs joint probe, P3_XY

Ground truth used for evaluation only. Joint = experimental (probe phase not identifiable from one dataset).

| run | best epoch | train loss | held-out loss | held-out / train | C10 | C12a | C12b | probe overlap | probe RMS (rad) | proj corr | slope | best-z corr (z) | slab / proj corr | selectivity | focus vs truth | layers (A) | candidates (reliable) | z median [IQR] | median unc. | multi-peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joint s0 | 76 | 6.700e-06 | 1.068e-01 | 15945.1 | -100.04 | 0.07 | 0.05 | 0.9609 | 0.282 | 0.913 | 0.115 | 0.915 (106) | 0.882 / 0.890 | -0.009 | 0.04 | 54, 178 | 50 (0) | 180 [61, 184] | 79 | 0.76 |

True probe (dataset 1): C10 -88.0, C12a 6.0, C12b -4.0 A; nominal C10 -100.0.

| group | n | loss_train | loss_heldout | C10 | C12a | C12b | probe_overlap | probe_rms_rad | proj_corr | proj_slope | selectivity | focus_vs_truth | n_layers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joint | 1 | 6.7e-06 ± 0 | 0.1068 ± 0 | -100 ± 0 | 0.06827 ± 0 | 0.04563 ± 0 | 0.9609 ± 0 | 0.2822 ± 0 | 0.9134 ± 0 | 0.1149 ± 0 | -0.008593 ± 0 | 0.04015 ± 0 | 2 ± 0 |
