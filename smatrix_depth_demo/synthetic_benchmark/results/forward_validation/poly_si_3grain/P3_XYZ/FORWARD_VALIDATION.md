# Forward-model consistency: P3_XYZ

**Au signal** (vacuum vs measured): L = 2.10e-01. A usable forward model must be well below it.
L_data = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas over all scan positions and pixels (0 = identical). Measured = generated noiseless patterns. Ground truth only; nothing fitted except F2's single global amplitude scale.

## F1: independent torch multislice vs abTEM (window variant)

| propagator | L_data | bright field | dark field | max abs diff / max |
|---|---|---|---|---|
| antialias_2/3 | 2.09e-03 | 1.19e-03 | 1.39e-02 | 9.96e-02 |
| no_antialias | 4.96e-03 | 1.63e-03 | 4.87e-02 | 1.27e-01 |
| vacuum (Au signal reference) | 2.10e-01 | 1.66e-01 | 7.98e-01 | 8.60e-01 |

## F2: S-matrix forward model with a numerically computed reference S (multislice of 1313 plane waves (antialias_2/3))

| S reference plane | measured variant | amplitude scale | L_data | bright field | dark field |
|---|---|---|---|---|---|
| exit | window | 1.0632 | 2.20e-03 | 1.25e-03 | 1.46e-02 |
| exit | pixelated | 1.0602 | 8.08e-03 | 5.65e-03 | 3.81e-02 |
| entrance | window | 1.0632 | 2.21e-03 | 1.25e-03 | 1.47e-02 |
| entrance | pixelated | 1.0602 | 8.08e-03 | 5.65e-03 | 3.81e-02 |

## F3: QuantEM forward path (truth potential sampled on QuantEM's grid, true probe)

QuantEM grid: 0.200 A pixels, object [120, 120], first scan pixel [8.0, 8.0].

| variant | dataset | configuration | L_data | bright field | dark field | QuantEM loss |
|---|---|---|---|---|---|---|
| window | def0 | truth potential (100 x 2 A) + true probe | 7.76e-03 | 3.85e-03 | 5.89e-02 | 21.8033 |
| window | def0 | vacuum + true probe | 2.03e-01 | 1.59e-01 | 7.85e-01 | 571.2570 |
| window | def1 | truth potential (100 x 2 A) + true probe | 6.37e-03 | 2.59e-03 | 5.56e-02 | 17.8979 |
| window | def1 | truth potential (100 x 2 A) + nominal probe | 2.42e-02 | 1.91e-02 | 9.06e-02 | 67.9678 |
| window | def1 | truth projected (1 slab) + true probe | 3.72e-01 | 2.24e-01 | 2.31e+00 | 1046.2953 |
| window | def1 | vacuum + true probe | 2.08e-01 | 1.62e-01 | 8.06e-01 | 582.9297 |
| window | def2 | truth potential (100 x 2 A) + true probe | 7.87e-03 | 3.83e-03 | 6.05e-02 | 22.1177 |
| window | def2 | vacuum + true probe | 2.16e-01 | 1.71e-01 | 8.09e-01 | 607.4678 |
| pixelated | def1 | truth potential (100 x 2 A) + true probe | 7.29e-03 | 3.03e-03 | 5.84e-02 | 20.4990 |
| pixelated | def1 | vacuum + true probe | 2.04e-01 | 1.56e-01 | 7.81e-01 | 572.7424 |
