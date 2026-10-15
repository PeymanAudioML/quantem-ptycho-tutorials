# Forward-model consistency: A_z100

**Au signal** (vacuum vs measured): L = 3.24e-04. A usable forward model must be well below it.
L_data = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas over all scan positions and pixels (0 = identical). Measured = generated noiseless patterns. Ground truth only; nothing fitted except F2's single global amplitude scale.

## F1: independent torch multislice vs abTEM (window variant)

| propagator | L_data | bright field | dark field | max abs diff / max |
|---|---|---|---|---|
| antialias_2/3 | 1.23e-05 | 5.43e-07 | 1.76e-03 | 2.48e-02 |
| no_antialias | 1.14e-04 | 5.40e-07 | 1.69e-02 | 2.67e-02 |
| vacuum (Au signal reference) | 3.24e-04 | 1.50e-04 | 2.62e-02 | 3.62e-01 |

## F2: S-matrix forward model with a numerically computed reference S (multislice of 1313 plane waves (antialias_2/3))

| S reference plane | measured variant | amplitude scale | L_data | bright field | dark field |
|---|---|---|---|---|---|
| exit | window | 1.0657 | 1.26e-04 | 5.23e-05 | 1.11e-02 |
| exit | pixelated | 1.0630 | 5.40e-03 | 2.57e-03 | 2.40e-01 |
| entrance | window | 1.0657 | 1.26e-04 | 5.23e-05 | 1.11e-02 |
| entrance | pixelated | 1.0630 | 5.40e-03 | 2.57e-03 | 2.40e-01 |

## F3: QuantEM forward path (truth potential sampled on QuantEM's grid, true probe)

QuantEM grid: 0.200 A pixels, object [120, 120], first scan pixel [8.0, 8.0].

| variant | dataset | configuration | L_data | bright field | dark field | QuantEM loss |
|---|---|---|---|---|---|---|
| window | def0 | truth potential (100 x 2 A) + true probe | 1.68e-04 | 3.93e-05 | 1.99e-02 | 0.4886 |
| window | def0 | vacuum + true probe | 3.87e-04 | 2.11e-04 | 2.74e-02 | 1.0966 |
| window | def1 | truth potential (100 x 2 A) + true probe | 9.84e-05 | 4.94e-06 | 1.40e-02 | 0.2892 |
| window | def1 | truth potential (100 x 2 A) + nominal probe | 1.23e-04 | 2.12e-05 | 1.53e-02 | 0.3584 |
| window | def1 | truth projected (1 slab) + true probe | 5.33e-04 | 2.90e-04 | 3.67e-02 | 1.5100 |
| window | def1 | vacuum + true probe | 2.49e-04 | 9.03e-05 | 2.39e-02 | 0.7045 |
| window | def2 | truth potential (100 x 2 A) + true probe | 1.46e-04 | 3.07e-05 | 1.68e-02 | 0.4163 |
| window | def2 | vacuum + true probe | 3.46e-04 | 1.61e-04 | 2.70e-02 | 0.9660 |
| pixelated | def1 | truth potential (100 x 2 A) + true probe | 2.96e-03 | 3.25e-04 | 1.78e-01 | 8.3257 |
| pixelated | def1 | vacuum + true probe | 3.12e-03 | 4.09e-04 | 1.84e-01 | 8.7824 |
