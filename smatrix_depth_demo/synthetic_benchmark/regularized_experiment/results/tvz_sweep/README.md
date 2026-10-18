# C1′: QuantEM depth TV (`tv_weight_z`) on P3_XYZ, 20 slices of 10 Å

Arm F (start from the S-matrix object of `smatrix_window_noiseless_s0_sfall_single_d1`, scale 4.0), 20 uniform slices,
dataset 1 only (one defocus), all 2809 patterns, 200 iterations (probe trained after 40), positivity, plus QuantEM's
built-in depth TV: w · mean |V[k+1] − V[k]|. Code: [`../../run_tvz_sweep.sh`](../../run_tvz_sweep.sh) →
[`../../run_quantem_tv.py`](../../run_quantem_tv.py) (shared runner unchanged; TV and data loss recorded separately),
[`../../eval_depth_bands.py`](../../eval_depth_bands.py). Run 2026-10-17/18, ~12 min per run on a T4.

## Results

| run | slices | w | data loss | projected corr | slice vs own slab / vs projection | depth profile corr | mean \|ΔV\| between slices | 40 Å band selectivity (perfect = 0.659) |
|---|---|---|---|---|---|---|---|---|
| F baseline | 35 | 0 | 16.06 | 0.950 | 0.66 / 0.52 * | 0.36 * | 0.023 * | 0.592 |
| F | 20 | 0 | **15.87** | **0.968** | 0.816 / 0.586 | 0.473 | 0.031 | 0.606 |
| F | 20 | 10 | 15.85 | 0.968 | 0.827 / 0.594 | 0.471 | 0.028 | 0.607 |
| F | 20 | 100 | 16.01 | 0.965 | 0.859 / 0.613 | 0.475 | 0.018 | 0.607 |
| F | 20 | 1000 | 17.05 | 0.958 | **0.894** / 0.624 | **0.589** | 0.0085 | **0.611** |

\* 5.7 Å slices: not directly comparable with the 10 Å rows. The 40 Å band columns compare all runs on the same footing
([`depth_bands.md`](depth_bands.md), [`depth_bands.png`](depth_bands.png)); all runs place all 5 bands correctly.

## Findings

1. **10 Å slices are free:** the 20-slice control fits the data slightly better than 35 slices (15.87 vs 16.06) and
   matches the projection better (0.968 vs 0.950).
2. **TV improves slice-level depth placement, increasingly with w:** own-slab corr 0.82 → 0.89, own-minus-projection gap
   0.23 → 0.27, depth profile corr 0.47 → 0.59 (w 0 → 1000). The top slice (0–10 Å) goes from 0.59 to 0.89, and the
   ring artefact next to the exit surface (180–190 Å) disappears (0.72 → 0.94). Cost: data loss +7 % at w = 1000;
   w = 100 costs +1 % for a smaller gain; w = 10 does nothing.
3. **At 40 Å scale there is little left to gain:** the baseline is already at 0.59–0.61 of a possible 0.66 and every
   run places all five bands; TV moves it 0.606 → 0.611.
4. w = 1000 is the largest tested and still improving; a weight between 1000 and ~3000 may be better. Not tested.
