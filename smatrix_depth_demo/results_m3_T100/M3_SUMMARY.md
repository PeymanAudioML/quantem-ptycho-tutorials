# M3 interim summary, 100 A slab (CPU run)

3-atom Au cluster in amorphous carbon (0.1 atoms/A^3, whole slab), abTEM data, 0.2 A pixels, 0.4 A scan step,
joint S + probe fit, differential pipeline templates (environment A), held-out cluster in environment B.

1. **Probe bias solved on a dense sample**: probe estimate within 0.9 A in every run (M1 sparse sample: one defocus
   5.5-7.3 A off).
2. **Templates transfer to real atoms**: with the test column's own background removed (oracle), depth within 1.3 A
   at 10 / 50 / 90 A, residual 0.0006-0.003.
3. **Carbon in the same column limits accuracy**: raw single-feature template fit gives 0 / 50.7 / 99.7 A (errors
   <= 10 A, toward the surfaces), residual ~30x the oracle. Tried and rejected (all worse): PCA background components
   (2-8, errors up to 88 A; they absorb the feature), subtracting the average background of environment A or of the
   test's own clear area (errors up to 84 A), non-negative depth-profile deconvolution (peaks at 0 / 67.5 / 7.5 A).
   Reason: carbon atoms give the same kind of depth response as Au, at every depth in the column.
4. **Feature counting fails on raw data**: residual floor 0.137, so no pair (40 / 60 / 80 A) is selected as two.

Consequence: placement to ~10 A remains usable as a QuantEM starting point (M5/M6 test); counting and separation need
a minimum feature-to-background contrast, to be mapped next (cluster size x support density).
