# Geometry check: synthetic_benchmark/configs/benchmark_200A_K96.yaml

300 kV (lambda 0.01969 A), 20 mrad; dr 0.2 A, window K = 96 (19.2 A), grid Y = 288 (57.6 A); scan 53 x 53 at 0.40 A (20.8 A field); 3 defoci, reference dataset 0.

Patterns 8427; S-matrix beams 1313; S pixels covered 40000; measurements 77,663,232 vs unknown reals 105,040,000 -> **0.739 measurements per unknown**. Depth resolution lambda/alpha^2 = 49.2 A; abTEM band limit 32.8 mrad.

| check | value | limit | ok | note |
|---|---|---|---|---|
| beams in window grid | 1.016 | 2.5 | yes | aperture radius vs Nyquist [1/A] |
| abTEM band limit | 32.81 | 26 | yes | mrad; use smaller dr |
| window size | 19.2 | 7.652 | yes | window [A] vs probe cone 2*alpha*167 A + 1.0 A |
| scan inside S grid | 1 | 1 | yes | all windows inside Y x Y |
| measurements/unknowns | 0.7394 | 0.7 | yes | 3 datasets x 2809 positions, 1313 beams |
| scan step = whole S-grid pixels | 2 | 2 | yes | 0.400 A = 2 px |
| pixelated detector = window detector | 0 | 0 | yes | window dk = 1/(K dr) = 0.05208 1/A; pixelated: 3x3 bins of 1/(Y dr) = 0.01736 1/A |
| true aberrations inside window | 19.2 | 7.652 | yes | with the TRUE defoci |

Memory (MB): patterns_per_variant 310.7, s_matrix_complex64 871.2, s_matrix_with_adam_state 2613.7, forward_window_gather_per_16_patterns 1548.9, potential 33.2

Overall: PASS
