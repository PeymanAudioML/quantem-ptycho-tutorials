# One defocus instead of three (stand-in)

Same specimen and true probe errors as HANDOVER.md (true C10 -44 A vs metadata -50 A, C12a/b 8/-5 A). One seed, CPU,
40 S-matrix epochs, 60 QuantEM iterations, pixelated probe lr 1e-4. Runs:

    python run_realization.py --data def_083 def_050 def_017                  # 3 defoci (REPORT.md)
    python run_realization.py --data def_050 --force                          # 1 defocus, same scan (REPORT_one_defocus_forced.md)
    python run_realization.py --data dense/def_050                            # 1 defocus, scan step 0.2 A instead of 0.4 A
                                                                              # (make_standin with STEP_PX=1; REPORT_one_defocus_dense.md)

| | 3 defoci | 1 defocus, sparse, --force | 1 defocus, 2x finer scan |
|---|---|---|---|
| measurements / unknowns (need >= 0.7) | 0.79 | 0.26 (refused without --force) | 1.0 |
| S-matrix probe C10 / C12 (true -44 / 9.4 A) | -44.2 / 7.9 | -50.0 / 0 (metadata, not refined) | -50.0 / 0 |
| S-matrix probe vs truth, RMS phase | 0.040 rad | 0.276 rad | 0.276 rad |
| S-matrix feature depths (true 30, 70 A) | 33.4, 74.1 | 30.5, 73.7 + false 43.4, 90.9 | 34.3, 83.8 + false 46.3 |
| QuantEM sinit probe at the end | 0.073 rad | 0.213 rad | 0.152 rad |
| QuantEM sinit depth-profile peaks | 24.5, 68.7 | 28.0, 68.6 | 34.1, 52.1, 65.6, 78.5, 91.9 |
| per-slab object correlation baseline -> sinit | 0.853 -> 0.902 | 0.840 -> 0.909 | 0.761 -> 0.833 |

Reading:
* With one dataset that dataset is the gauge reference, so the S-matrix stage keeps the metadata probe; probe errors are
  left to QuantEM, which corrects them only partly (0.15-0.21 rad vs 0.073 rad with three defoci).
* Depth sectioning comes from the convergence angle, so slice locations still come out, with more false features; the
  S-informed start still beats QuantEM on its own in both single-defocus runs.
* The finer scan passed the data check but did not improve on the forced sparse run here; not explained, single seed.
* For one defocus: --force (the S-matrix result is only a starting point) and --probe-model parametric (lets QuantEM fix
  defocus and astigmatism itself; 0.276 -> 0.083 rad in HANDOVER.md).
