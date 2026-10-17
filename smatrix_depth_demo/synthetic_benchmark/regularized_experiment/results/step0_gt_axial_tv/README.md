# Step 0: is axial TV defensible on the true P3_XYZ potential?

Ground truth only, no reconstruction. Code: [`../../step0_gt_axial_tv.py`](../../step0_gt_axial_tv.py) (run
2026-10-17, CPU, ~1 min). Full table: [`summary.md`](summary.md); all numbers: [`metrics.json`](metrics.json).

## Question

Axial TV penalises the slab-to-slab change |V(z+1) − V(z)| at each lateral position. It helps only if that change is
large at grain boundaries and small inside grains. Otherwise it mostly penalises real crystal structure. We measure this
on the true potential, cut into slabs like a QuantEM reconstruction, for three representations of each slab:

| representation | what it is |
|---|---|
| `raw` | slab potential (phase σV per Å), no filtering: what QuantEM's built-in `tv_weight_z` acts on |
| `lp<σ>` | the same, Gaussian low-passed laterally (σ = 0.5–3 Å): the "coarse-scale contrast" idea |
| `spectrum` | local orientation fingerprint: \|FFT\| of 6.4 Å Hann-windowed tiles (stride 1.6 Å), unit-normalised; step = L2 distance between neighbouring slabs |

Each neighbouring slab pair at each pixel is classed as **within grain** (both slabs entirely in one grain) or
**boundary** (the two slabs' majority grains differ). Scores:

- **AUC:** probability that a boundary step exceeds a within-grain step (0.5 = no separation, 1 = perfect).
- **Mean ratio:** mean boundary step / mean within-grain step.
- **Boundary share of TV:** fraction of the total TV sum that comes from boundary pairs. The rest penalises structure
  inside grains.

## Method notes

- **Slabs are built from the atoms** (`atoms.extxyz`): every atom with its centre in the slab, rendered as a 0.35 Å
  Gaussian, scaled to the abTEM projection (rendered vs abTEM projection: corr 0.927). This is abTEM's atom-to-slice
  rule at any slab thickness.
- **Why not re-bin the 2 Å abTEM slices:** they alternate about 2× in mean potential (18.5 vs 36 V·Å) because atomic
  planes fall unevenly into 2 Å bins. Re-binning them into 5.71 Å slabs mixes 2 and 3 slices and adds steps that are
  not in the specimen. That version is kept in [`../step0_gt_axial_tv_slices/`](../step0_gt_axial_tv_slices/) as a
  cross-check: for slab thicknesses that are multiples of 2 Å (4, 8, 10 Å) both versions agree (AUC within 0.04).
- **Grain labels:** the construction rule of the specimen (nearest seed in the metric (1, 1, 0.05) with lateral
  periodic images) on a 1 Å depth grid. It matches the grain id of all 31,967 atoms (agreement 1.0000).
- **Region:** whole 57.6 Å cell (periodic). Scores for the scanned field alone (18.4–39.2 Å) are close (e.g. raw at 35
  slabs: AUC 0.71 vs 0.68).

## Results

| slabs | slab | representation | AUC | mean within | mean boundary | ratio | boundary share of TV (share of pairs) |
|---|---|---|---|---|---|---|---|
| 35 (QuantEM) | 5.71 Å | raw | 0.68 | 0.0044 | 0.0084 | 1.9 | 0.09 (0.05) |
| 35 | 5.71 Å | lp2 | 0.63 | 0.0009 | 0.0011 | 1.3 | 0.06 (0.05) |
| 35 | 5.71 Å | spectrum | **0.89** | 0.28 | 0.54 | 1.9 | 0.11 (0.06) |
| 20 | 10 Å | raw | 0.81 | 0.0019 | 0.0082 | **4.4** | **0.29** (0.09) |
| 20 | 10 Å | lp2 | 0.51 | 0.0008 | 0.0008 | 1.0 | 0.08 (0.09) |
| 20 | 10 Å | spectrum | **0.97** | 0.19 | 0.56 | 3.0 | 0.28 (0.11) |

Raw steps in rad/Å; spectrum steps are fingerprint distances (0–2).

Mean within-grain raw step by grain at 35 slabs: [001] 0.0010, [110] 0.0056, [111] 0.0069, vs boundary 0.0084.

## Findings

1. **Axial TV on the raw potential at 35 slices is weakly justified.** Boundary steps are larger on average (1.9×), but
   the distributions overlap heavily (AUC 0.68), and 91 % of the TV sum comes from pairs inside grains. A
   `tv_weight_z` penalty would mostly act on real crystal structure. The concern raised in review is confirmed.
2. **Within-grain change depends on slab thickness vs each grain's repeat along the beam**
   ([`within_grain_by_orientation.png`](within_grain_by_orientation.png)). The repeats are about 5.43 Å for [001], 3.84 Å
   for [110] and 9.41 Å for [111]. A grain looks almost constant in z when the slab thickness is close to a multiple of
   its repeat: [001] at 5.71 Å, [110] at 4 and 8 Å, [111] at 10 Å. No thickness suits all three orientations.
3. **Lateral low-pass makes it worse, not better.** All three grains are silicon with the same density, so coarse
   contrast doesn't tell them apart (AUC 0.51 at 10 Å). What remains is plane counting: how many atomic planes fall in
   each slab. That shows up as horizontal bands across whole grains in [`xz_section_35slabs.png`](xz_section_35slabs.png).
   The "coarse-scale V" version of axial TV (C2 in the plan) is dropped.
4. **Thicker slices make raw TV much more defensible.** At 10 Å (20 slices): AUC 0.81, boundary steps 4.4× larger on
   average, boundary share of TV 29 %. 10 Å is still about 5× finer than the depth resolution (λ/α² ≈ 49 Å), so little
   depth information is lost. Multislice accuracy at 10 Å slices still needs checking.
5. **An orientation fingerprint separates boundaries best.** AUC 0.89 at 5.71 Å and 0.97 at 10 Å, with no knowledge of
   the grain orientations. It is a concrete form of the "structural descriptor G" from the review. Even so, within-grain
   pairs still carry most of the TV sum at 35 slices (89 %), and using it needs a custom differentiable loss in QuantEM.

## Implications for the plan

| option | verdict |
|---|---|
| C1: QuantEM `tv_weight_z` on raw V, 35 slices | Not recommended: 91 % of the penalty acts inside grains |
| C1′: `tv_weight_z` on raw V, 20 slices (10 Å) | **Worth testing first**: built in (one line), AUC 0.81; check multislice accuracy at 10 Å |
| C2: TV on low-passed V | Dropped: low-pass erases the grain contrast |
| C3: TV on the local-spectrum fingerprint | Most promising representation (AUC 0.89–0.97); needs a custom QuantEM loss |

**Caveats:**
- This is the true potential. Reconstructed slices are blurred in depth (~49 Å) and noisier, so separation will be
  lower in practice.
- The rendered atoms approximate abTEM's potential (corr 0.93).
- Tile-based fingerprints are coarse laterally (6.4 Å tiles). The scanned-field AUC is not computed for them, because
  the field holds too few tiles.

## Files

| file | content |
|---|---|
| `separability_vs_slab.png` | AUC, mean ratio and boundary share of TV vs slab thickness, every representation |
| `step_histograms_35slabs.png` | within-grain vs boundary step distributions at 35 slabs (raw, lp2, spectrum) |
| `xz_section_35slabs.png` | x–z section at the scan-field centre: grain map, true slab potential, raw / low-passed / fingerprint steps, true boundaries in cyan |
| `within_grain_by_orientation.png` | mean within-grain step per grain orientation vs slab thickness (raw, lp2) |
| `slab_grids/slabs_{35,20}_{field,cell}.png` | every true slab (rows 1, 3, ...) and its axial-TV step to the next slab (rows 2, 4, ...), laid out like the P3_XY `plots/slices.png`; scanned field or whole cell; code [`../../step0_plot_slabs.py`](../../step0_plot_slabs.py) |
| `metrics.json`, `summary.md` | all numbers, including the scanned-field subset |

Reproduce (from `synthetic_benchmark/`):

```
../../.venv/bin/python regularized_experiment/step0_gt_axial_tv.py                   # this folder
../../.venv/bin/python regularized_experiment/step0_gt_axial_tv.py --source slices   # cross-check folder
../../.venv/bin/python regularized_experiment/step0_plot_slabs.py && ../../.venv/bin/python regularized_experiment/step0_plot_slabs.py --region cell   # slab_grids/
```
