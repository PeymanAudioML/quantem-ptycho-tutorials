# M2 summary (interpretation of M2_RESULTS.md, T4 run)

Pairs of identical 3-atom phase clusters in one column, 300 A slab, 20 mrad, joint fit, M1 pipeline templates.

| separation | 20 A | 40 A | 60 A | 80 A | 120 A |
|---|---|---|---|---|---|
| found as two, within tolerance (3 centres) | 0 / 3 | 0 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| depth errors | - | - | <= 3 A | <= 2 A | <= 4 A |

- **Two-feature limit between 40 and 60 A** on reconstructed data; the noise-free exact-S ideal resolves 40 A, but
  with an 11 A bias. Reconstruction costs ~10-20 A of separation.
- **False splits on single clusters: 0 / 10** (floor from noise realisation 0, tested on 1 and 2).
- **Artefact**: for unresolved pairs centred at 100 A (d = 20, 40) the selector added a spurious feature at 300 A
  (the exit surface), with residuals only just above the floor (0.0048 / 0.0054 vs 0.0046). Likely the mirror /
  edge ambiguity found in M3; M1/M2 analysis now uses the amplitude-phase limit (`--max-phase`, default pi/4).
  To confirm, re-run `python calibrate_m2.py analyze` (minutes, uses the saved S-matrices).
