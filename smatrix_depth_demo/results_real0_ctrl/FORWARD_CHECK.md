# Forward-model check (QuantEM forward path, no optimisation)

L_data = Σ(√I_pred − √I_meas)² / Σ I_meas over all scan positions and detector pixels (measured = the noise-free patterns as QuantEM sees them).  Bright field = |q| ≤ α/λ.  QuantEM loss = its own `error_estimate` (l2_amplitude), comparable with the training loss.

Truth probe resampled to the QuantEM grid: overlap 0.9974, RMS phase 0.049 rad vs the original.
Measured dark-field share of the intensity: 0.027.

| configuration | slabs | L_data | bright field | dark field | QuantEM loss |
|---|---|---|---|---|---|
| rr_2s_fixed/baseline: recon object + recon probe | 2 | 0.0107 | 0.0015 | 0.3396 | 14.663 |
| rr_2s_fixed/baseline: recon object + truth probe | 2 | 0.0094 | 0.0017 | 0.2830 | 12.867 |
| rr_2s_train/baseline: recon object + recon probe | 2 | 0.0107 | 0.0015 | 0.3394 | 14.649 |
| rr_2s_train/baseline: recon object + truth probe | 2 | 0.0094 | 0.0017 | 0.2829 | 12.866 |
| rr_8s_fixed/baseline: recon object + recon probe | 8 | 0.0106 | 0.0015 | 0.3369 | 14.537 |
| rr_8s_fixed/baseline: recon object + truth probe | 8 | 0.0094 | 0.0017 | 0.2815 | 12.784 |
| truth object, 8 x 0.75 A slabs + truth probe (model floor) | 8 | 0.0087 | 0.0014 | 0.2692 | 11.862 |
| truth object, 2 x 3 A slabs + truth probe | 2 | 0.0087 | 0.0014 | 0.2700 | 11.890 |
| truth object, 1 slab (projected) + truth probe | 1 | 0.0088 | 0.0014 | 0.2719 | 11.962 |
| truth object, 8 slabs + nominal probe (QuantEM start) | 8 | 0.0099 | 0.0011 | 0.3235 | 13.536 |
| vacuum (zero potential) + truth probe (reference) | 2 | 0.0120 | 0.0020 | 0.3668 | 16.015 |

Final training loss (last iteration, includes batch averaging): rr_2s_fixed/baseline 14.714, rr_2s_train/baseline 14.700, rr_8s_fixed/baseline 14.613
