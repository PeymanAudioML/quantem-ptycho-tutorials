# Forward-model consistency: A_z100

L_data = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas over all scan positions and pixels (0 = identical). Measured = generated noiseless patterns. Ground truth only; nothing fitted except F2's single global amplitude scale.

## F1: independent torch multislice vs abTEM (window variant)

| propagator | L_data | bright field | dark field | max abs diff / max |
|---|---|---|---|---|
| antialias_2/3 | 3.45e-05 | 2.23e-06 | 2.05e-03 | 2.97e-02 |
| no_antialias | 2.83e-04 | 2.69e-06 | 1.78e-02 | 2.74e-02 |

## F2: S-matrix forward model with a numerically computed reference S (multislice of 293 plane waves (antialias_2/3))

| S reference plane | measured variant | amplitude scale | L_data | bright field | dark field |
|---|---|---|---|---|---|
| exit | window | 0.9809 | 1.66e-02 | 3.32e-03 | 8.45e-01 |
| exit | pixelated | 0.9801 | 3.93e-02 | 2.10e-03 | 9.35e-01 |
| entrance | window | 0.9809 | 1.66e-02 | 3.34e-03 | 8.45e-01 |
| entrance | pixelated | 0.9801 | 3.93e-02 | 2.10e-03 | 9.35e-01 |

## F3: QuantEM forward path (truth potential sampled on QuantEM's grid, true probe)

QuantEM grid: 0.200 A pixels, object [64, 64], first scan pixel [8.0, 8.0].

| variant | dataset | configuration | L_data | bright field | dark field | QuantEM loss |
|---|---|---|---|---|---|---|
| window | def0 | truth potential (100 x 2 A) + true probe | 6.02e-03 | 4.43e-03 | 1.26e-01 | 3.7615 |
| window | def0 | truth potential (100 x 2 A) + nominal probe | 6.02e-03 | 4.43e-03 | 1.26e-01 | 3.7615 |
| window | def0 | truth projected (1 slab) + true probe | 7.84e-03 | 5.96e-03 | 1.50e-01 | 4.9030 |
| window | def0 | vacuum + true probe | 6.40e-03 | 5.18e-03 | 9.85e-02 | 3.9989 |
| window | def1 | truth potential (100 x 2 A) + true probe | 1.15e-03 | 5.63e-04 | 3.88e-02 | 0.7201 |
| window | def1 | truth potential (100 x 2 A) + nominal probe | 8.49e-04 | 2.72e-04 | 3.77e-02 | 0.5306 |
| window | def1 | truth projected (1 slab) + true probe | 2.79e-03 | 1.81e-03 | 6.52e-02 | 1.7436 |
| window | def1 | vacuum + true probe | 1.45e-03 | 9.27e-04 | 3.48e-02 | 0.9057 |
| window | def2 | truth potential (100 x 2 A) + true probe | 4.86e-03 | 3.63e-03 | 6.92e-02 | 3.0368 |
| window | def2 | truth potential (100 x 2 A) + nominal probe | 5.16e-03 | 3.83e-03 | 7.48e-02 | 3.2264 |
| window | def2 | truth projected (1 slab) + true probe | 6.62e-03 | 5.09e-03 | 8.67e-02 | 4.1388 |
| window | def2 | vacuum + true probe | 5.41e-03 | 4.21e-03 | 6.78e-02 | 3.3780 |
| pixelated | def0 | truth potential (100 x 2 A) + true probe | 1.67e-02 | 2.33e-03 | 2.78e-01 | 10.4612 |
| pixelated | def0 | truth potential (100 x 2 A) + nominal probe | 1.67e-02 | 2.33e-03 | 2.78e-01 | 10.4612 |
| pixelated | def0 | truth projected (1 slab) + true probe | 1.85e-02 | 3.82e-03 | 2.85e-01 | 11.5701 |
| pixelated | def0 | vacuum + true probe | 1.71e-02 | 3.05e-03 | 2.72e-01 | 10.7033 |
| pixelated | def1 | truth potential (100 x 2 A) + true probe | 1.90e-02 | 1.74e-03 | 3.32e-01 | 11.8493 |
| pixelated | def1 | truth potential (100 x 2 A) + nominal probe | 1.87e-02 | 1.75e-03 | 3.26e-01 | 11.6764 |
| pixelated | def1 | truth projected (1 slab) + true probe | 2.06e-02 | 2.97e-03 | 3.41e-01 | 12.9013 |
| pixelated | def1 | vacuum + true probe | 1.93e-02 | 2.11e-03 | 3.31e-01 | 12.0526 |
| pixelated | def2 | truth potential (100 x 2 A) + true probe | 2.01e-02 | 2.12e-03 | 3.47e-01 | 12.5876 |
| pixelated | def2 | truth potential (100 x 2 A) + nominal probe | 2.05e-02 | 2.30e-03 | 3.51e-01 | 12.8089 |
| pixelated | def2 | truth projected (1 slab) + true probe | 2.19e-02 | 3.53e-03 | 3.56e-01 | 13.6951 |
| pixelated | def2 | vacuum + true probe | 2.06e-02 | 2.66e-03 | 3.46e-01 | 12.8736 |
