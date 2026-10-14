# M1 summary (interpretation of M1_RESULTS.md)

Run: `calibrate_m1.py all --device cuda` on a T4, 22 reconstructions, 237 s each (about 14x faster than 4 CPU cores).

**M1 passes.** Templates built through the reconstruction pipeline fix what exact-S templates could not:

| criterion | target | result |
|---|---|---|
| residual floor | well below exact-S templates | 0.0007-0.0017 vs 0.026-0.046 (about 30x lower); floor 0.0033 |
| single-feature depth, held-out cluster | within ~3 A | within 1 A at 30 / 150 / 270 A, both noise realisations (exact-S templates: up to -6 A) |
| negative control | single clusters stay single | 3 / 3 |
| pairs 80 / 120 A | found as two | 109 / 190 (true 110 / 190); 88 / 214 (true 90 / 210) |
| reference plane | no drift | 0 A in all tests |

- **40 A pair is not resolved**: one feature at 151 A; its one-feature residual (0.0023) is below the floor, so the
  data do not distinguish it from a single feature. The two-feature limit is between 40 and 80 A; M2 maps it.
- **Systematic probe bias on this sparse sample**: dataset 2's C10 is 5.5-7.3 A off and dataset 1's C12a ~+2 A in all
  22 runs. Depths are unaffected (M0/M1), but a sparse specimen gives a biased probe; check on denser samples in M3.
- Limits: one cluster type and geometry, templates from one noise realisation, two noise realisations per test depth.

Next: M2 (`calibrate_m2.py all --device cuda`, 30 reconstructions, ~2 h on the T4).
