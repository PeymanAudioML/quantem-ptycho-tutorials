# 3D grain map from the S-matrix recovery (P3_XYZ)

Code: [`../../smatrix_grains_3d.py`](../../smatrix_grains_3d.py) (2026-10-19, CPU, under 1 min). Input: the
single-defocus S-matrix run `smatrix_window_noiseless_s0_sfall_single_d1` (dataset 1, all patterns, fixed nominal probe).
The best QuantEM run (`quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ20_tvz1000`, see
[`../tvz_sweep/`](../tvz_sweep/README.md)) goes through the same pipeline. For it, the 10 Å slice containing each depth
is cropped to the same field, on the same 0.2 Å pixels.

## Method

1. **Volume:** the S-matrix refocused to depths 0–200 Å (refocused phase, every 4 Å), over the scanned field (0.2 Å
   pixels, about 21 Å wide).
2. **Local lattice fingerprint at every voxel:**
   - |FFT| of a 6.4 Å Hann-windowed tile around the voxel, on a 0.8 Å lateral grid.
   - Lowest frequencies removed, then unit-normalised.
   - The three grains have different orientations, so their fingerprints differ.
3. **Unsupervised clustering:** PCA (10 components) and k-means with k = 3, over all voxels. The truth does not enter.
4. **Scoring only:** each cluster is given the name and colour of the true grain it overlaps best (best one-to-one
   assignment). Accuracy is then measured against the true grain map (the specimen's construction rule).
5. **Reference:** the same steps on the true potential (10 Å running slabs at the same depths). This shows what the
   method can do with perfect depth information.

## Results

| input | voxel accuracy | per 40 Å band (0–40, …, 160–200) |
|---|---|---|
| S-matrix recovery | 0.88 | 1.00 / 0.87 / 0.85 / 0.80 / 0.88 |
| best QuantEM: arm F (S-matrix start), 20 slices, depth TV w = 1000 | **0.95** | 1.00 / 0.95 / 0.95 / 0.86 / 0.97 |
| true potential (reference) | 0.97 | 0.99 / 0.98 / 0.98 / 0.94 / 0.99 |
| majority class (always guess the commonest grain) | 0.38 | |

- **Tile size:** 4.8 / 6.4 / 9.6 Å tiles give S-matrix accuracy 0.86 / 0.88 / 0.88.
- **Order and shape:** the S-matrix map shows the right order ([001] top, [110] middle, [111] bottom) and the tilted
  boundaries.
- **Boundary depths:** off by about 10–20 Å. The [001]/[110] boundary comes out deeper than the truth. Part of this may
  be the probe defocus error: the probe was fixed at the nominal C10 of −100 Å, while the truth is −88 Å, so depths are
  shifted by about 12 Å. Not verified.
- **Why this works when the correlation metrics did not:** slice-by-slice correlation with the true slab was poor below
  the top band (`../p3xyz_baseline/`). Lattice orientation survives the S-matrix's depth blur far better than the exact
  atom contrast does.

## Files

| file | content |
|---|---|
| `grains_3d.png` | 3D voxel maps, front quarter cut away: truth / S-matrix / best QuantEM / method on the true potential |
| `grains_sections.png` | x–z and y–z sections with boundaries (black), and grain share vs depth |
| `grains_3d.json`, `grains_3d.npz` | accuracies, settings, label volumes |

## Caveats

- **k = 3 was given.** In practice the number of grains would have to be chosen from the data.
- **Small field:** only about 21 Å wide, with 6.4 Å tiles. Lateral boundary positions are coarse, about one tile.
- **Ideal data:** one noiseless specimen, one seed.

## P3_XY (columnar grains), for comparison: [`P3_XY/`](P3_XY/)

The same script with `--specimen P3_XY --run smatrix_window_noiseless_s0_full80` (the 3-defocus S-matrix fit) and
`--quantem-run quantem_window_noiseless_d1_E_smatrix_trained_call_s0_grid35` (best P3_XY QuantEM run: arm E, 35 slices).

| input | voxel accuracy | per 40 Å band |
|---|---|---|
| S-matrix recovery (3 defocus) | 0.92 | 0.92 / 0.92 / 0.92 / 0.92 / 0.93 |
| QuantEM arm E, 35 slices | 0.94 | 0.93 / 0.94 / 0.93 / 0.94 / 0.93 |
| method on the true potential | 0.93 | 0.91 / 0.94 / 0.94 / 0.94 / 0.94 |

- **Columns recovered:** all three inputs show the grains as vertical columns through the full 200 Å.
- **Accuracy limit:** all three are at the method's limit (about 0.93). In P3_XY the boundaries are vertical, so every
  depth has the same share of tiles that straddle a boundary.
- **QuantEM errors:** QuantEM's map has small patches labelled with the wrong grain at some depths (for example green
  inside the [110] grain at x 4–8 Å). These are depth artefacts in its slices. The S-matrix map is cleaner but places
  one boundary about 3 Å off laterally.
