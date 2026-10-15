# Geometry check: synthetic_benchmark/configs/benchmark_200A.yaml

300 kV (lambda 0.01969 A), 20 mrad; dr 0.2 A, window K = 48 (9.6 A), grid Y = 96 (19.2 A); scan 25 x 25 at 0.40 A (9.6 A field); 3 defoci, reference dataset 0.

Patterns 1875; S-matrix beams 293; S pixels covered 9216; measurements 4,320,000 vs unknown reals 5,400,576 -> **0.800 measurements per unknown**. Depth resolution lambda/alpha^2 = 49.2 A; abTEM band limit 32.8 mrad.

| check | value | limit | ok | note |
|---|---|---|---|---|
| beams in window grid | 1.016 | 2.5 | yes | aperture radius vs Nyquist [1/A] |
| abTEM band limit | 32.81 | 26 | yes | mrad; use smaller dr |
| window size | 9.6 | 7.652 | yes | window [A] vs probe cone 2*alpha*167 A + 1.0 A |
| scan inside S grid | 1 | 1 | yes | all windows inside Y x Y |
| measurements/unknowns | 0.7999 | 0.7 | yes | 3 datasets x 625 positions, 293 beams |
| scan step = whole S-grid pixels | 2 | 2 | yes | 0.400 A = 2 px |
| pixelated detector = window detector | 0 | 0 | yes | window dk = 1/(K dr) = 0.10417 1/A; pixelated: 2x2 bins of 1/(Y dr) = 0.05208 1/A |
| true aberrations inside window | 9.6 | 7.652 | yes | with the TRUE defoci |

Memory (MB): patterns_per_variant 17.3, s_matrix_complex64 21.6, s_matrix_with_adam_state 64.8, forward_window_gather_per_16_patterns 86.4, potential 3.7

Overall: PASS
