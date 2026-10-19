# P3_XYZ with three defocus datasets in the S-matrix fit

Script: [`../../run_p3xyz_3defocus.sh`](../../run_p3xyz_3defocus.sh). Log: `../../p3xyz_3defocus.log`. Run on 2026-10-19,
02:39–07:08 UTC on a T4 (S-matrix fit 3.5 h, QuantEM 55 min).

- **S-matrix** `smatrix_window_noiseless_s0_full80`:
  - all three defocus datasets, all patterns (no held-out data), 80 epochs, same settings as P3_XY full80
  - dataset 0 is the gauge; C10/C12 of datasets 1 and 2 are learned
- **QuantEM** is still trained on **dataset 1 only**, with the same settings as the 1-defocus runs:
  - F, 20 slices, TV w = 0 and w = 1000
  - E, 35 slices (S-informed grid)
  - the only change is the S-matrix the runs start from

## 1. Probe recovery (S-matrix fit)

| | C10 (Å) | C12a / C12b (Å) |
|---|---|---|
| dataset 1: 3-defocus fit / truth | −87.79 / −88.0 | 6.06, −4.13 / 6, −4 |
| dataset 2: 3-defocus fit / truth | −43.31 / −43.3 | −2.95, 4.85 / −3, 5 |
| dataset 1: 1-defocus fit (probe fixed) | −100 (nominal) | 0, 0 |

- **Three datasets:** the probe is recovered to about 0.2 Å.
- **One dataset:** the probe is fixed at the nominal value, and the 12 Å defocus error moves into S.

## 2. QuantEM depth bands (40 Å), 1-defocus vs 3-defocus start

| run | data loss | projected corr | band diag | diag per band |
|---|---|---|---|---|
| A, 35 slices, no S (reference) | 77.08 | 0.853 | 0.795 | 0.80 / 0.71 / 0.74 / 0.81 / 0.91 |
| F20 w0, 1-def | 15.87 | 0.968 | 0.942 | 0.96 / 0.94 / 0.94 / 0.93 / 0.95 |
| **F20 w0, 3-def** | **9.11** | **0.974** | **0.961** | 0.96 / 0.95 / 0.96 / 0.96 / 0.97 |
| F20 w1000, 1-def | 17.05 | 0.958 | 0.943 | 0.95 / 0.94 / 0.94 / 0.93 / 0.95 |
| F20 w1000, 3-def | 10.28 | 0.960 | 0.956 | 0.96 / 0.95 / 0.95 / 0.95 / 0.97 |
| E35, 1-def | 16.34 | 0.950 | 0.918 | 0.94 / 0.90 / 0.90 / 0.90 / 0.95 |
| E35, 3-def | 10.45 | 0.960 | 0.943 | 0.94 / 0.93 / 0.94 / 0.94 / 0.96 |

- **Every arm improves** with the 3-defocus start:
  - data loss falls by about 40 %
  - the band diagonal rises by 0.013–0.025
  - the deep bands (80–200 Å) gain the most
- **Below the true object's loss:** the 3-defocus runs reach about 9–10. The true object at 20 slices scores 24.3 (see
  `../forward_slice_convergence/`), so they still absorb forward-model error.
- **QuantEM's probe drifts away from the start:**
  - it starts at RMS 0.006 rad from the 3-defocus S-matrix
  - it ends at 0.04 rad (E35), 0.09 rad (F w0) and 0.10 rad (F w1000)
  - with this good a starting probe, freezing it (`--probe-lr 0`) may be worth testing

## 3. 3D grain maps

Voxel accuracy, total and per 40 Å band (method as in [`../smatrix_grains_3d/`](../smatrix_grains_3d/README.md)).

| input | 1-defocus start | 3-defocus start |
|---|---|---|
| S-matrix | 0.88 (1.00/.87/.85/.80/.88) | 0.86 (0.97/.76/.90/.85/.81) |
| QuantEM F20 w1000 | 0.95 (1.00/.95/.95/.86/.97) | **0.96** (1.00/.94/.94/.94/.99) |
| QuantEM E35 | — | 0.95 (0.98/.96/.96/.92/.90) |
| method on the true potential | 0.97 | 0.97 |

- **QuantEM F w1000 from the 3-defocus S** is almost at the method's limit (0.96 vs 0.97). The weak 120–160 Å band
  rises from 0.86 to 0.94.
- **The S-matrix's own grain map does not improve** (0.86 vs 0.88):
  - the [001]/[110] boundary now comes out about 10–15 Å too shallow (x–z section), where before it came out too deep
  - the [110]/[111] boundary is spread wider
  - so the correct probe removes the depth offset, but the S-matrix's depth resolution of tens of Å stays the same

## 4. S-matrix depth features (`depth_features_1_vs_3_defocus.png`)

- **1-defocus S:** features split into two groups, at about 50 Å and about 180 Å, which are the false layers seen
  before.
- **3-defocus S:**
  - one broad focus peak at 114 Å
  - all features picked at 95–130 Å, with response widths of about 120 Å
  - no false layers, but no per-atom depth resolution either
  - neither fit's focus curve follows the true depth profile (corr −0.13 / 0.08)
- **The grain orientation map (section 3) is the better depth read-out** of the S-matrix than per-feature focus.

## Files

| file / folder | content |
|---|---|
| `smatrix_window_noiseless_s0_full80/` | S-matrix summary, arrays, fit-monitor plots |
| `depth_features_1_vs_3_defocus.png` | feature depth picks, 1 vs 3 defocus |
| `smatrix_slices_smatrix_window_noiseless_s0_full80.png` | S-matrix refocused slices vs truth and QuantEM E35 |
| `quantem_*_full80Q20_tvz0`, `_tvz1000`, `_full80Q35` | QuantEM summaries and plots (slices, depth, probe) |
| `depth_bands.{md,json,png}` | section 2 |
| `grains_3d_F20_w1000/`, `grains_3d_E35/` | section 3 (3D maps, sections, overlay `grains_overlay_2d.png`) |

## Caveats

- Noiseless, one seed, one specimen.
- QuantEM still sees only dataset 1. Training it on all three datasets was not tested.
