# M5: QuantEM hand-off dry run

Setup: abTEM data, 3 Au in vacuum at z = 50 A, 100 A slab, one dataset (C10 = -38 A), 0.2 A pixels, K = 32,
17 x 17 scan at 0.4 A; QuantEM 0.1.9 at the uv.lock commit 0f60f90 (pip), CPU.

Data hand-off: patterns fftshifted to put the bright-field disc at the centre; Dataset4dstem sampling
(0.4 A, 0.4 A, lam/(K dr) = 3.08 mrad, 3.08 mrad); PtychographyDatasetRaster.preprocess(force_com_rotation=0).

Findings (tentative: 1 slice, 20 iterations, positivity off):
- Object grid: sampling 0.2 A (= dr), roi 32 x 32 (= K), obj_shape_full = scanned field (32 px) + 2 x padding.
- Coordinates: QuantEM object pixel = S-grid pixel - K/2 + padding (Au at S-grid (29, 26) -> object (21, 18) with
  padding 8). quantem_export.initial_potential must use origin = (K/2 - padding) * dr.
- Defocus: QuantEM probe_params["defocus"] = -C10 (positive = focus inside the specimen). Decided by the multislice
  test below; the single-slice loss comparison (which favoured +C10) cannot resolve depth and was wrong.
- Default object constraint positivity=True clamps a zero-initialised potential at 0 in this version (object stayed
  exactly 0 for 30 iterations); constraints are a plain dict here, e.g. {"object": {"positivity": False}}.
- Version: the tutorial notebooks import PtychoObjConstraintParams, which the pinned commit 0f60f90 does not have;
  the GPU machine's quantem version must be checked before the M5/M6 scripts are finalised.

## Multislice dry run (m5_dry_run.py)

Same data; 40 iterations, Adam lr 5e-3 on the object, probe fixed, positivity on, 64 patterns per batch.
Arms: uniform = 20 planes 5 A apart, potential 1e-3; export = quantem_export slice grid (dz 2.5-8 A, 17 planes) and
initial potential (0.3 rad per Au, depth support N(50 A, 8 A)) on QuantEM's 48 x 48 grid (origin 1.6 A).

| QuantEM defocus | arm | loss start -> end | Au-column depth (peak / centroid) | time |
|---|---|---|---|---|
| +C10 = -38 (wrong sign) | uniform | 2.856 -> 2.118 | 2.5 / 4.2 A | 11 s |
| +C10 = -38 (wrong sign) | export | 2.680 -> 2.167 | 3.9 / 10.3 A | 8 s |
| -C10 = +38 | uniform | 2.688 -> 1.977 | 47.5 / 49.2 A | 12 s |
| -C10 = +38 | export | 2.299 -> 1.919 | 45.1 / 49.1 A | 9 s |

True depth 50 A. Conclusions: QuantEM defocus = -C10; its z = 0 is the entrance surface (same as ours); the export is
accepted on QuantEM's grid with the right lateral offset. The export arm starts and ends at a lower loss with fewer
planes (one specimen, exact candidates: a hint for M6, not a result). M5 dry run passes.
