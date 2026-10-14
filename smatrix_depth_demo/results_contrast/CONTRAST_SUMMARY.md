# Contrast scan summary (100 A slab, CPU)

Au cluster size (3 / 6 / 12 atoms) x carbon density (0 / 0.025 / 0.05 / 0.1 atoms/A^3); abTEM data, joint fit,
differential templates every 25 A (exit-surface wrap-around fixed), amplitude-phase limit pi/4. Tests at 15 / 50 / 85 A.
Full table: CONTRAST_RESULTS.md.

Raw depth error (max over 15 / 50 / 85 A):

| Au atoms | rho 0 | 0.025 | 0.05 | 0.1 |
|---|---|---|---|---|
| 3 | 1 | 7 | 15 | 15 |
| 6 | 3 | 6 | 14 | 15 |
| 12 | 3 | 16 | 28 | 5 |

1. **Templates are not the limit**: background-subtracted (oracle) errors <= 3 A everywhere except one 5 A case.
2. **Main failure is surface attraction**: near-surface features (15, 85 A) are fitted at the surfaces (0, 100 A);
   the mid-slab feature is within 6 A in 11 / 12 cases.
3. **No single contrast threshold**: success is not monotonic in cluster size or density, and the response-contrast
   metric is not monotonic either, so it is not used as a gate.
4. **Residual flag**: every case with single-feature residual > 0.03 failed (> 10 A); low residual is not a
   guarantee (12 Au at 0.025: residual 0.009, error 16 A). Report residuals with every depth.
5. **Feature counting fails in carbon** at 40 / 60 A separations (pair/single residual ratio < 1) for all sizes and
   densities; it works only in vacuum (and 3 Au at 0.025, d = 60 A).
6. **Probe improves with matrix density**: max error 3.0-3.5 A in vacuum, 0.8-0.9 A at 0.1 atoms/A^3.

Decision: skip M3 at 300 A for now (3x more carbon per column would worsen the same limits). Next is M5/M6: do
candidates good to ~6-15 A, with a residual reliability flag, improve QuantEM's reconstruction?
