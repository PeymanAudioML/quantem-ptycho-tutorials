# M6 summary: first GPU run (05a9de4) and what changes

## Reading of the first run (200 iterations, 1 run per arm)

- **c1_z50 "0.0 everywhere" is not a scoring bug**: QuantEM did put the Au at ~50 A in every arm, but the 13 uniform
  planes are 7.7 A apart and one plane centre is exactly 50.0 A; the peak-based score can only return plane positions.
- **c1_z50 measures the noise**: detection found nothing, so B was identical to A and D to C, yet losses differ by up
  to 0.135 and depth centroids span 49.8-57.2 A. QuantEM's batch shuffling was unseeded: run-to-run noise ~ +-0.07 in
  loss and ~ +-4 A in depth, as large as most differences between arms.
- **vac_pair**: the S-matrix candidates were 0 / 72 A for true 20 / 80 A (surface attraction of the shallow feature,
  as in the contrast scan). The reconstructed profiles do show two lobes (~0-25 and ~60-90 A), but the peak-based
  score picked noisy spikes (e.g. 71 / 87 A); scoring was the main failure there.
- **Indications (not yet significant)**: on the four single-feature specimens B (uniform slices + S potential) is
  never worse than A in depth (c025_z20 -11.2 -> 0.5 A, vac_z70 -8.2 -> -2.3 A, two equal) and has the lower loss in
  all four (by 0.04-0.06). In c025_z20 the candidate was 22 A off (42 vs 20 A) and QuantEM still ended 0.5 A off with
  it: the "guide, not truth" behaviour. C (S slices, flat potential) is often worse than A (-10.2, -12.8 A).

## Changes in m6_quantem.py (step 2 only; the step-1 .npz files stay valid)

1. QuantEM's rng is seeded per run (`Ptychography.from_models(rng=seed)`): identical setups are now identical on
   CPU (checked; GPU may keep tiny floating-point differences).
2. `--repeats N` seeds per arm; the report gives mean +- std and the B / C / D - A differences paired by seed.
3. Depth = centroid of the contiguous half-maximum region around each feature's main peak, inside windows split at
   the midpoints between true depths (no plane quantisation, no spike picking); FWHM reported.

Re-run: `../.venv/bin/python m6_quantem.py --device gpu --iters 200 --repeats 5`.

## Conclusions from the 5-seed run (74d96a3; 200 iterations, Adam 5e-3, probe fixed at the S-matrix estimate)

Paired by seed against arm A (uniform slices, flat potential); negative = better. Seed spread is now small (loss std
0.001-0.02, depth std 0.1-4.8 A), so differences several std wide are real.

| specimen | candidates (truth) | B - A: S potential | C - A: S slices | D - A: both |
|---|---|---|---|---|
| vac_z30 | 31 (30) | -0.7 +- 1.0 A, loss -0.049 | +2.0 +- 3.3 A, +0.000 | -0.2 +- 1.3 A, -0.048 |
| vac_z70 | 72 (70) | **-1.9 +- 0.6 A**, loss -0.059 | +0.9 +- 1.6 A, +0.003 | **-1.8 +- 0.5 A**, -0.056 |
| c025_z50 | 51 (50) | +0.2 +- 1.2 A, loss -0.044 | +0.9 +- 0.8 A, +0.001 | +1.3 +- 2.1 A, -0.037 |
| c025_z20 | 42 (20) | +1.0 +- 0.7 A, loss -0.037 | +1.5 +- 0.2 A, +0.005 | +2.2 +- 0.9 A, -0.028 |
| vac_pair | 0, 72 (20, 80) | **+4.8 +- 1.2 A**, loss -0.045 | **+6.2 +- 1.6 A**, +0.001 | **+8.3 +- 0.9 A**, -0.040 |
| c1_z50 | none | identical (no candidates) | identical | identical |

1. **Diffraction fit: supported.** The S-informed potential (B, D) lowers the final loss on every specimen with
   candidates, by 0.03-0.06, i.e. 3-30x the seed spread.
2. **Depth: not supported as a general improvement.** QuantEM alone (A) is already within 4 A on the single features
   (and 0.3-4 A on vac_pair). B improves depth only on vac_z70 (-1.9 A). With a wrong candidate the prior pulls
   QuantEM: vac_pair's shallow candidate at 0 A (true 20 A) costs 4.8-8.3 A; c025_z20's 22 A-off candidate costs
   ~1 A (QuantEM mostly recovers, the "guide, not truth" behaviour, but not for free).
3. **S-informed slices (C) never help**: no loss gain, depth equal or worse. Slice placement alone carries no useful
   information here (consistent with the plan's note that finer spacing does not create axial information).
4. **Dense carbon (c1_z50)**: the S-matrix stage finds no candidates, so nothing is passed; the pipeline degrades to
   QuantEM alone rather than injecting false priors.
5. **Not evaluated**: uncertainty coverage (needs M4). The fixed candidate widths (8 / 30 A) are not calibrated, and
   vac_pair shows what is missing: a surface-attracted candidate was passed as reliable.

Consequences: keep the S-informed potential (B) as the default hand-off; drop S-informed slice placement as a default
(keep as an option for M7); M4 must flag or widen near-surface candidates before they are passed to QuantEM.
