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
| [`quantem_window_noiseless_d1_E_smatrix_trained_call_s0_grid35`](quantem_window_noiseless_d1_E_smatrix_trained_call_s0_grid35/README.md) | QuantEM arm E, three-focus S-matrix object AND probe, probe trained |
| [`quantem_window_noiseless_d1_E_smatrix_fixed_call_s0_grid35`](quantem_window_noiseless_d1_E_smatrix_fixed_call_s0_grid35/README.md) | QuantEM arm E, three-focus S-matrix object AND probe, probe fixed |
| [`quantem_window_noiseless_d1_E_nominal_trained_call_s0_grid35_val0.1`](quantem_window_noiseless_d1_E_nominal_trained_call_s0_grid35_val0.1/README.md) | QuantEM arm E, nominal probe trained, 10 % held-out patterns |
| [`quantem_window_noiseless_d1_E_smatrix_trained_call_s0_grid35_val0.1`](quantem_window_noiseless_d1_E_smatrix_trained_call_s0_grid35_val0.1/README.md) | QuantEM arm E, three-focus S-matrix probe trained, 10 % held-out patterns |
| [`quantem_window_noiseless_d1_E_smatrix_fixed_call_s0_grid35_val0.1`](quantem_window_noiseless_d1_E_smatrix_fixed_call_s0_grid35_val0.1/README.md) | QuantEM arm E, three-focus S-matrix probe fixed, 10 % held-out patterns |
| [`quantem_window_noiseless_d1_A_nominal_trained_call_s0_sfQ`](quantem_window_noiseless_d1_A_nominal_trained_call_s0_sfQ/README.md) | Handoff Q0: flat object, nominal probe (35 uniform slices) |
| [`quantem_window_noiseless_d1_A_smatrix_trained_call_s0_sfQ`](quantem_window_noiseless_d1_A_smatrix_trained_call_s0_sfQ/README.md) | Handoff Q2: flat object, single-focus joint S-matrix probe |
| [`quantem_window_noiseless_d1_F_nominal_trained_call_s0_sfQ`](quantem_window_noiseless_d1_F_nominal_trained_call_s0_sfQ/README.md) | Handoff Q1: single-focus S-matrix object, nominal probe |
| [`quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfQ`](quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfQ/README.md) | Handoff Q3: single-focus S-matrix object AND probe |
| [`quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ`](quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ/README.md) | Handoff from the all-pattern single-focus S-matrix: arm E (S-informed slices) |
| [`quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ`](quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ/README.md) | Handoff from the all-pattern single-focus S-matrix: arm F (35 uniform slices) |

Comparison folder (not a run): [`compare_probe_handoff/`](compare_probe_handoff/COMPARE.md): arm E with the nominal
probe vs the three-focus S-matrix probe (trained, fixed); `COMPARE.md`, `compare_metrics.png`, `compare_slices.png`.
Report section 6.
[`compare_heldout/`](compare_heldout/COMPARE.md): the same three arm-E probe set-ups with 10 % held-out patterns
(`*_val0.1`); adds `train_vs_heldout_loss.png`.

## Defocus datasets and starting points

| dataset | nominal C10 | true C10 / C12a / C12b (Å) | patterns | used by |
|---|---|---|---|---|
| 0 | −166.7 Å | −166.7 / 0 / 0 (reference, exact) | 2809 | 3-focus S-matrix fit only |
| **1** | **−100.0 Å** | **−88.0 / 6.0 / −4.0** | 2809 | 3-focus S-matrix fit, all 1-focus S-matrix fits, **every QuantEM run** |
| 2 | −33.3 Å | −43.3 / −3.0 / 5.0 | 2809 | 3-focus S-matrix fit only |

Source: `configs/poly_si_3grain.yaml` (`defocus.nominal_C10`, `perturbations` = true − nominal).

**The three defocus datasets are used only by the S-matrix fit `full80`.** Every QuantEM run (`_d1_` in the name;
`dataset = 1` in its `summary.json`) reconstructs from the **dataset 1 patterns alone** (2809; 2528 trained + 281 held
out in `_val0.1`). In `run_benchmark.py quantem()` only `def{--dataset}` is loaded, and every `forward_check.make_pt`
call uses it. From the S-matrix run, QuantEM reads only saved results (`summary.json`, `arrays.npz`): dataset 1's probe
coefficients and beams, the depth candidates and (arm E) the projected S-matrix object. So "3-focus" or "1-focus" below
means which S-matrix fit the starting point came from, not what QuantEM trains on. QuantEM probe starts: `nominal` =
C10 −100 Å, no astigmatism (12 Å off); `smatrix` from `full80` = C10 −87.91, C12a 6.07, C12b −4.09 Å (within 0.1 Å);
`smatrix` from the 1-focus joint fits ≈ nominal (C10 −100.0). Arm A `grid35` takes only its slice count (35) from the
three-focus S-informed grid, to match arms C/D/E; nothing else from the S-matrix fit reaches it.

Name keys: `single_d1` = one defocus (otherwise all three); `_joint` = probe fitted
with S (otherwise fixed). For QuantEM, `nominal`/`smatrix` = where the probe starts and `trained`/`fixed` = whether
QuantEM updates it; arms E and F start from the S-matrix object.

### S-matrix fits

| run | defocus datasets | probe |
|---|---|---|
| `smatrix_..._s0_full80` | 3: datasets 0, 1, 2 (8427 patterns) | joint: dataset 0 is the fixed reference; datasets 1 and 2 learn C10, C12a, C12b. Dataset 1 → −87.91 / 6.07 / −4.09, dataset 2 → −44.04 / −2.92 / 4.84 Å |
| `smatrix_..._full80_single_d1` | 1: dataset 1 | fixed (nominal) |
| `smatrix_..._sf80_single_d1` | 1: dataset 1 | fixed (nominal), 10 % held out |
| `smatrix_..._sf80_single_d1_joint` | 1: dataset 1 | joint, 10 % held out; probe stayed at nominal (C10 −100.04 vs truth −88) |
| `smatrix_..._sfall_single_d1_joint` | 1: dataset 1 | joint, all patterns; probe stayed at nominal (C10 −100.01) |

### QuantEM runs

All QuantEM runs train on dataset 1 (C10 −100 Å nominal, −88 Å true).

| run | S-matrix source | object start | probe start → in QuantEM |
|---|---|---|---|
| `A_nominal_trained_*_pure` | none | flat | nominal → trained |
| `A_nominal_trained_*_grid35` | none (slice count 35 matched to the 3-focus grid) | flat | nominal → trained |
| `C_nominal_trained_*_grid35` | 3-focus | flat (S only sets the slices) | nominal → trained |
| `D_nominal_trained_*_grid35` | 3-focus | seeded candidate spots | nominal → trained |
| `E_nominal_trained_*_grid35` (+`_val0.1`) | 3-focus | S-matrix object | nominal → trained |
| `E_smatrix_trained_*_grid35` (+`_val0.1`) | 3-focus | S-matrix object | S-matrix probe → trained |
| `E_smatrix_fixed_*_grid35` (+`_val0.1`) | 3-focus | S-matrix object | S-matrix probe → fixed |
| `A_nominal_trained_*_sfQ` (Q0) | none | flat | nominal → trained |
| `F_nominal_trained_*_sfQ` (Q1) | 1-focus (`sf80_single_d1_joint`) | S-matrix object | nominal → trained |
| `A_smatrix_trained_*_sfQ` (Q2) | 1-focus (`sf80_single_d1_joint`) | flat | S-matrix probe → trained |
| `F_smatrix_trained_*_sfQ` (Q3) | 1-focus (`sf80_single_d1_joint`) | S-matrix object | S-matrix probe → trained |
| `E/F_smatrix_trained_*_sfallQ` | 1-focus (`sfall_single_d1_joint`) | S-matrix object | S-matrix probe → trained |

Started from both the S-matrix object and the S-matrix probe: `E_smatrix_trained`, `E_smatrix_fixed` (3-focus),
`F_smatrix_trained_*_sfQ` and `E/F_smatrix_trained_*_sfallQ` (1-focus). The probe is held fixed in QuantEM only in
`E_smatrix_fixed_*` (and its `_val0.1` copy). With one defocus the S-matrix probe equals the nominal probe, so in the
1-focus runs only the object start helps.

## Groups

- **Three-focus S-matrix and the QuantEM arms that use it:** `smatrix_..._full80`; QuantEM `*_grid35` (A, C, D, E with the nominal probe; E with the S-matrix probe, trained and fixed) and
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
