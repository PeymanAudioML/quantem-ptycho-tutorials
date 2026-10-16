# Runs: poly-Si P3_XY (`configs/poly_si_3grain.yaml`)

Reconstruction outputs for specimen P3_XY: three columnar Si grains through a 200 Å slab, window detector,
noiseless, three defocus datasets (nominal C10 −166.7 / −100 / −33.3 Å). Each directory has a README with its purpose,
the command and settings, the results against the ground truth and its files. The large arrays stay local
(`runs/` is gitignored). Small copies are in `results/smatrix/` and `results/quantem/`, and the reports are in `reports/`.

Naming: `smatrix_<detector>_<noise>_s<seed>_<tag>[_single_d<dataset>][_joint]` and
`quantem_<detector>_<noise>_d<dataset>_<arm>_<probe mode>_c<candidates>_s<seed>_<tag>`.

| run | what |
|---|---|
| [`smatrix_window_noiseless_s0_full80`](smatrix_window_noiseless_s0_full80/README.md) | S-matrix + probe, all three defocus datasets (reference fit) |
| [`smatrix_window_noiseless_s0_full80_single_d1`](smatrix_window_noiseless_s0_full80_single_d1/README.md) | S-matrix, dataset 1 only, probe fixed (first equal-data reference) |
| [`smatrix_window_noiseless_s0_sf80_single_d1`](smatrix_window_noiseless_s0_sf80_single_d1/README.md) | Single-focus S-matrix, probe FIXED (seed 0, held-out split) |
| [`smatrix_window_noiseless_s0_sf80_single_d1_joint`](smatrix_window_noiseless_s0_sf80_single_d1_joint/README.md) | Single-focus S-matrix, probe updated JOINTLY (experimental, seed 0, held-out split) |
| [`smatrix_window_noiseless_s0_sfall_single_d1_joint`](smatrix_window_noiseless_s0_sfall_single_d1_joint/README.md) | Single-focus S-matrix, joint probe, ALL patterns (no held-out split) |
| [`quantem_window_noiseless_d1_A_nominal_trained_call_s0_pure`](quantem_window_noiseless_d1_A_nominal_trained_call_s0_pure/README.md) | QuantEM arm A, 25 uniform slices, flat start, nominal probe (standalone) |
| [`quantem_window_noiseless_d1_A_nominal_trained_call_s0_grid35`](quantem_window_noiseless_d1_A_nominal_trained_call_s0_grid35/README.md) | QuantEM arm A, 35 uniform slices, flat start, nominal probe |
| [`quantem_window_noiseless_d1_C_nominal_trained_call_s0_grid35`](quantem_window_noiseless_d1_C_nominal_trained_call_s0_grid35/README.md) | QuantEM arm C, S-informed slices (three-focus S), flat start |
| [`quantem_window_noiseless_d1_D_nominal_trained_call_s0_grid35`](quantem_window_noiseless_d1_D_nominal_trained_call_s0_grid35/README.md) | QuantEM arm D, S-informed slices + seeded candidate spots (three-focus S) |
| [`quantem_window_noiseless_d1_E_nominal_trained_call_s0_grid35`](quantem_window_noiseless_d1_E_nominal_trained_call_s0_grid35/README.md) | QuantEM arm E, S-informed slices + S-matrix object start (three-focus S) |
| [`quantem_window_noiseless_d1_A_nominal_trained_call_s0_sfQ`](quantem_window_noiseless_d1_A_nominal_trained_call_s0_sfQ/README.md) | Handoff Q0: flat object, nominal probe (35 uniform slices) |
| [`quantem_window_noiseless_d1_A_smatrix_trained_call_s0_sfQ`](quantem_window_noiseless_d1_A_smatrix_trained_call_s0_sfQ/README.md) | Handoff Q2: flat object, single-focus joint S-matrix probe |
| [`quantem_window_noiseless_d1_F_nominal_trained_call_s0_sfQ`](quantem_window_noiseless_d1_F_nominal_trained_call_s0_sfQ/README.md) | Handoff Q1: single-focus S-matrix object, nominal probe |
| [`quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfQ`](quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfQ/README.md) | Handoff Q3: single-focus S-matrix object AND probe |
| [`quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ`](quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ/README.md) | Handoff from the all-pattern single-focus S-matrix: arm E (S-informed slices) |
| [`quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ`](quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ/README.md) | Handoff from the all-pattern single-focus S-matrix: arm F (35 uniform slices) |

## Groups

- **Three-focus S-matrix and the QuantEM arms that use it:** `smatrix_..._full80`; QuantEM `*_grid35` (A, C, D, E) and
  `*_pure` (A, 25 slices). Report: `reports/QUANTEM_P3_XY.md`.
- **Single-focus S-matrix (dataset 1 only):** `full80_single_d1` (fixed probe, first reference); `sf80_single_d1`
  (fixed) vs `sf80_single_d1_joint` (experimental joint probe), matched with a 10 % held-out split; `sfall_single_d1_joint`
  (joint, all patterns). Evaluation: `results/single_focus_joint/P3_XY/`.
- **Single-focus handoff to QuantEM (35 uniform slices):** Q0 `A_nominal_trained_*_sfQ`, Q1 `F_nominal_trained_*_sfQ`,
  Q2 `A_smatrix_trained_*_sfQ`, Q3 `F_smatrix_trained_*_sfQ` from `sf80_single_d1_joint`. Also `E/F_smatrix_trained_*_sfallQ`
  from `sfall_single_d1_joint`.

## Status of the single-focus experiment (seed 0 only)

Only seed 0 of the fixed and joint fits exists. Seeds 1–2 and the alternative starting probes (C10 −130 / −70,
astigmatic) were never run, because the queue stopped after the QuantEM step. With one seed:
- **Probe accuracy:** no improvement. The joint probe stayed at nominal (C10 −100.04 vs truth −88), RMS 0.282 rad, the
  same as fixed.
- **Projected object:** no improvement (corr 0.913, slope 0.115 in both).
- **Depth:** no improvement. Both show the same false layers at ~54 / 178 Å.
- **QuantEM:** the S-matrix *object* start helps (projected corr 0.896 → 0.966, loss 75 → 9.8). The S-matrix *probe*
  adds nothing, because it equals the nominal probe.
