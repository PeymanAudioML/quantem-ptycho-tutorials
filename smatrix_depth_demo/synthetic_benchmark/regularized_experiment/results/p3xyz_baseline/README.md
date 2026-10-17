# P3_XYZ baseline (no regularization)

Specimen P3_XYZ: 3 Si grains stacked in depth ([001] top, [110] middle, [111] bottom, tilted boundaries).
Dataset 1 only (one defocus, nominal C10 −100 Å, true −88 Å), all 2809 patterns, no held-out split, no regularization
(S-matrix: amplitude loss only; QuantEM: data loss + positivity). Chain: [`../../run_p3xyz_baseline.sh`](../../run_p3xyz_baseline.sh),
log `../../p3xyz_baseline.log`, run 2026-10-17 20:37–23:14 UTC on a T4. Full runs in `../../../runs/poly_si_3grain/P3_XYZ/`.

| folder | what |
|---|---|
| `quantem_..._A_nominal_trained_call_s0_xyz_single` | QuantEM arm A: flat start, 35 uniform slices, nominal probe (trained after 40 iters), no S-matrix |
| `smatrix_window_noiseless_s0_sfall_single_d1` | S-matrix fit, probe fixed, 80 epochs, train loss 1.9e-5, 0.25 measurements/unknown; `smatrix_slices_*.png` = refocused S per slab vs truth |
| `quantem_..._E_smatrix_trained_call_s0_sfallQ` | arm E: S-matrix object start (scale 4.0) + slice grid from 31 S depth candidates (came out ~uniform 5–7 Å) |
| `quantem_..._F_smatrix_trained_call_s0_sfallQ` | arm F: S-matrix object start (scale 4.0), 35 uniform slices |

## Results

| | final loss | projected corr (slope) | slab corr: own slab / whole projection | probe RMS error (rad) |
|---|---|---|---|---|
| A (no S) | 77.1 | 0.853 (0.78) | 0.57 / 0.45 | 0.185 |
| S refocused | — | — | 0.72 / 0.69 | fixed at nominal (0.28) |
| E | 16.3 | 0.950 (0.93) | 0.63 / 0.51 | 0.091 |
| F | **16.1** | **0.950** (0.93) | **0.66** / 0.52 | **0.087** |

Own-slab / projection corr by depth band:

| depth | true grain | S refocused | A | E | F |
|---|---|---|---|---|---|
| 0–51 Å | [001] | **0.84 / 0.56** | 0.61 / 0.36 | 0.67 / 0.39 | 0.68 / 0.40 |
| 51–103 Å | [001]→[110] | 0.68 / 0.72 | 0.53 / 0.47 | 0.65 / 0.58 | 0.69 / 0.60 |
| 103–154 Å | [110]→[111] | 0.74 / 0.78 | 0.61 / 0.52 | 0.68 / 0.57 | 0.70 / 0.60 |
| 154–200 Å | [111] | 0.60 / 0.70 | 0.54 / 0.45 | 0.59 / 0.57 | 0.57 / 0.48 |

(S-refocused values read from the plot titles; its slices are smoother than QuantEM's, which raises both of its numbers,
so compare its own-vs-projection gap, not its absolute values with QuantEM's.)

## Findings

1. The S-matrix start helps a lot again: loss 77 → 16, projected corr 0.85 → 0.95, probe error halved. E ≈ F (the S
   depth candidates covered all depths, so E's grid is close to uniform).
2. All reconstructions have some depth selectivity (own-slab corr above projection corr), unlike P3_XY where none could.
3. The refocused S-matrix places the top [001] grain clearly (0.84 own vs 0.56 projection) but not the deeper grains
   (own ≤ projection below 51 Å). QuantEM after hand-off keeps a gap of about 0.1–0.3 at every depth but sharpens no band strongly.
4. Loss stalls at ~16 (P3_XY hand-off reached 9–11), and the slice next to the exit surface (189–194 Å) is a ring artefact in every arm (corr 0.15–0.17).
5. This is the baseline for the regularized runs: S-matrix R2/R1 and QuantEM `tv_weight_z` at 20 slices.
