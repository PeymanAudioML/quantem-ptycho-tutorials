# M5 notes: QuantEM hand-off conventions (work in progress)

Setup: abTEM data, 3 Au in vacuum at z = 50 A, 100 A slab, one dataset (C10 = -38 A), 0.2 A pixels, K = 32,
17 x 17 scan at 0.4 A; QuantEM 0.1.9 at the uv.lock commit 0f60f90 (pip), CPU.

Data hand-off: patterns fftshifted to put the bright-field disc at the centre; Dataset4dstem sampling
(0.4 A, 0.4 A, lam/(K dr) = 3.08 mrad, 3.08 mrad); PtychographyDatasetRaster.preprocess(force_com_rotation=0).

Findings (tentative: 1 slice, 20 iterations, positivity off):
- Object grid: sampling 0.2 A (= dr), roi 32 x 32 (= K), obj_shape_full = scanned field (32 px) + 2 x padding.
- Coordinates: QuantEM object pixel = S-grid pixel - K/2 + padding (Au at S-grid (29, 26) -> object (21, 18) with
  padding 8). quantem_export.initial_potential must use origin = (K/2 - padding) * dr.
- Defocus: QuantEM probe_params["defocus"] = our C10 (negative = focus inside): defocus -38 reaches loss 2.19 with
  the strongest positive peak on an Au atom; +38 reaches 2.36.
- Default object constraint positivity=True clamps a zero-initialised potential at 0 in this version (object stayed
  exactly 0 for 30 iterations); constraints are a plain dict here, e.g. {"object": {"positivity": False}}.
- Version: the tutorial notebooks import PtychoObjConstraintParams, which the pinned commit 0f60f90 does not have;
  the GPU machine's quantem version must be checked before the M5/M6 scripts are finalised.
