# Forward-model check (QuantEM forward path, no optimisation)

L_data = Σ(√I_pred − √I_meas)² / Σ I_meas over all scan positions and detector pixels (measured = the noise-free patterns as QuantEM sees them).  Bright field = |q| ≤ α/λ.  QuantEM loss = its own `error_estimate` (l2_amplitude), comparable with the training loss.

Truth probe resampled to the QuantEM grid: overlap 0.9974, RMS phase 0.049 rad vs the original.
Measured dark-field share of the intensity: 0.027.

| configuration | slabs | L_data | bright field | dark field | QuantEM loss |
|---|---|---|---|---|---|
| results_real0_2slice/baseline: recon object + recon probe | 2 | 0.0107 | 0.0015 | 0.3395 | 14.649 |
| results_real0_2slice/baseline: recon object + truth probe | 2 | 0.0094 | 0.0017 | 0.2830 | 12.868 |
| truth object, 8 x 0.75 A slabs + truth probe (model floor) | 8 | 0.0087 | 0.0014 | 0.2692 | 11.862 |
| truth object, 2 x 3 A slabs + truth probe | 2 | 0.0087 | 0.0014 | 0.2700 | 11.890 |
| truth object, 1 slab (projected) + truth probe | 1 | 0.0088 | 0.0014 | 0.2719 | 11.962 |
| truth object, 8 slabs + nominal probe (QuantEM start) | 8 | 0.0099 | 0.0011 | 0.3235 | 13.536 |
| vacuum (zero potential) + truth probe (reference) | 2 | 0.0120 | 0.0020 | 0.3668 | 16.015 |

Final training loss (last iteration, includes batch averaging): results_real0_2slice/baseline 14.699
