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
