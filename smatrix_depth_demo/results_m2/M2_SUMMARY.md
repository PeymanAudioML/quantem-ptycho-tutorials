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

## Closure (amplitude check, ddab0d3)

- Spurious edge features removed by `min_amp_ratio`: d = 20 / 40 A pairs at c = 100 A now return one feature at
  100 / 101 A (correct for an unresolved pair).
- Two-feature amplitude ratio (weaker / stronger): real pairs (60-120 A) 0.86-1.0; unresolved pairs 0.05-0.14,
  except d = 40 A at c = 200 A: 0.31 (0.44 / 1.43). Default threshold raised from 0.3 to 0.5, the middle of the gap
  (0.31 vs 0.86); no M2 result changes.
- **M2 closed**: two-feature limit between 40 and 60 A; 0 / 10 false splits; resolved pairs within 4 A.
