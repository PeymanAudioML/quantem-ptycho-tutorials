# Depth TV at 35 slices vs 20 slices (P3_XYZ, arm F)

Same as [`../tvz_sweep/`](../tvz_sweep/README.md) (S-matrix object start, probe from nominal, dataset 1 only, all 2809
patterns, 200 iterations) but 35 slices of 5.71 Å, `tv_weight_z` = 1000 (same as the best 20-slice run) and 1750
(1000 × 10/5.71, compensating the thinner slices). Code: [`../../run_tvz35.sh`](../../run_tvz35.sh). Run 2026-10-18, ~26 min per run.

| run | data loss | projected corr | slice vs own slab / vs projection (gap) | depth profile corr | top slice / exit-side slice corr | 40 Å band selectivity (perfect 0.659) |
|---|---|---|---|---|---|---|
| 35 slices, w 0 (baseline) | 16.06 | 0.950 | 0.661 / 0.523 (0.14) | 0.36 | 0.31 / 0.15 | 0.592 |
| 35 slices, w 1000 | 18.15 | 0.907 | 0.747 / 0.581 (0.17) | 0.42 | 0.72 / 0.54 | 0.573 |
| 35 slices, w 1750 | 18.58 | 0.897 | 0.751 / 0.581 (0.17) | 0.46 | 0.80 / 0.54 | 0.571 |
| 20 slices, w 0 | **15.87** | **0.968** | 0.816 / 0.586 (0.23) | 0.47 | 0.59 / 0.72 | 0.606 |
| 20 slices, w 1000 | 17.05 | 0.958 | **0.894** / 0.624 (**0.27**) | **0.59** | **0.89 / 0.94** | **0.611** |

(Slice-level columns compare 5.71 Å with 10 Å slices; the 40 Å band column, [`depth_bands.md`](depth_bands.md) /
[`depth_bands.png`](depth_bands.png), compares all runs on the same footing.)

## Findings

1. **At 35 slices TV helps slice by slice but hurts overall:** own-slab corr 0.66 → 0.75, top slice 0.31 → 0.80, exit-side
   ring 0.15 → 0.54; but the projection gets worse (0.950 → 0.90), the 40 Å bands get worse (selectivity 0.592 → 0.571),
   and the data loss rises 13–16 %.
2. **20 slices with w = 1000 stays best on every depth measure,** at a smaller data cost (+7 %). This matches step 0:
   at 5.71 Å, real atoms change a lot between slices inside a grain, so TV pushes against real structure.
3. Raising w from 1000 to 1750 at 35 slices changes little.
