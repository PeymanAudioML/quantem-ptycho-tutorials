# Forward-model consistency: P3_XY

**Au signal** (vacuum vs measured): L = 1.76e-01. A usable forward model must be well below it.
L_data = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas over all scan positions and pixels (0 = identical). Measured = generated noiseless patterns. Ground truth only; nothing fitted except F2's single global amplitude scale.

## F1: independent torch multislice vs abTEM (window variant)

| propagator | L_data | bright field | dark field | max abs diff / max |
|---|---|---|---|---|
| antialias_2/3 | 2.40e-03 | 2.02e-03 | 1.28e-02 | 1.35e-01 |
| no_antialias | 4.17e-03 | 2.77e-03 | 4.28e-02 | 1.99e-01 |
| vacuum (Au signal reference) | 1.76e-01 | 1.58e-01 | 6.47e-01 | 8.45e-01 |

## F2: S-matrix forward model with a numerically computed reference S (multislice of 1313 plane waves (antialias_2/3))

| S reference plane | measured variant | amplitude scale | L_data | bright field | dark field |
|---|---|---|---|---|---|
| exit | window | 1.0630 | 2.44e-03 | 2.02e-03 | 1.41e-02 |
| exit | pixelated | 1.0598 | 8.72e-03 | 6.50e-03 | 6.37e-02 |
| entrance | window | 1.0630 | 2.45e-03 | 2.03e-03 | 1.43e-02 |
| entrance | pixelated | 1.0598 | 8.72e-03 | 6.51e-03 | 6.36e-02 |

## F3: QuantEM forward path (truth potential sampled on QuantEM's grid, true probe)

QuantEM grid: 0.200 A pixels, object [120, 120], first scan pixel [8.0, 8.0].

| variant | dataset | configuration | L_data | bright field | dark field | QuantEM loss |
|---|---|---|---|---|---|---|
| window | def0 | truth potential (100 x 2 A) + true probe | 7.99e-03 | 5.78e-03 | 6.96e-02 | 22.4627 |
| window | def0 | vacuum + true probe | 1.81e-01 | 1.65e-01 | 6.32e-01 | 508.4285 |
| window | def1 | truth potential (100 x 2 A) + true probe | 5.70e-03 | 3.75e-03 | 6.03e-02 | 16.0301 |
| window | def1 | truth potential (100 x 2 A) + nominal probe | 1.78e-02 | 1.51e-02 | 9.33e-02 | 49.9697 |
| window | def1 | truth projected (1 slab) + true probe | 3.82e-01 | 1.96e-01 | 5.59e+00 | 1072.8020 |
| window | def1 | vacuum + true probe | 1.62e-01 | 1.45e-01 | 6.52e-01 | 455.2267 |
| window | def2 | truth potential (100 x 2 A) + true probe | 6.96e-03 | 4.82e-03 | 6.65e-02 | 19.5583 |
| window | def2 | vacuum + true probe | 1.78e-01 | 1.60e-01 | 6.55e-01 | 498.3725 |
| pixelated | def1 | truth potential (100 x 2 A) + true probe | 6.77e-03 | 4.11e-03 | 6.91e-02 | 19.0359 |
| pixelated | def1 | vacuum + true probe | 1.59e-01 | 1.38e-01 | 6.27e-01 | 445.0312 |
