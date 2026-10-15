# Realization reconstruction: S-matrix + probe (joint) vs QuantEM multislice

Data: 1 defocus datasets (def_050), 300 kV, 20.0 mrad, thickness 100 A, scan step 0.4 A.

## Probe

| dataset | nominal C10 | S-matrix C10 | S-matrix C12a, C12b | metadata C10 | S-matrix probe vs truth (overlap, RMS phase err rad) |
|---|---|---|---|---|---|
| 0 (ref) | -50.0 | -50.0 | 0.0, 0.0 | -50.0 | 0.9627, 0.276 |

## What QuantEM was started from

| QuantEM input | baseline | sinit |
|---|---|---|
| probe | nominal: C10 -50.0 A, C12 0.0 A, phi12 0 deg | smatrix: C10 -50.0 A, C12 0.0 A, phi12 0 deg |
| probe hand-over check (initial QuantEM probe vs S-matrix probe overlap) | 0.9926 | 0.9926 |
| slice planes | uniform, 22 slabs | smatrix, 22 slabs |
| initial object | flat | candidates |
| geometry | 0.200 A px, z = 0 entrance, defocus = -C10 | 0.200 A px, z = 0 entrance, defocus = -C10 |
| optimiser | object lr 0.005 x 60 it; pixelated probe lr 0.0001 from it 13 | object lr 0.005 x 60 it; pixelated probe lr 0.0001 from it 13 |
| probe changed by QuantEM (1 - overlap initial/final) | 7.99e-03 | 7.45e-03 |

QuantEM results (dataset 0):

* baseline: final loss 0.62776; probe vs truth: start overlap 0.9578 / RMS phase 0.276 rad -> end 0.9728 / 0.207 rad
* sinit: final loss 0.53992; probe vs truth: start overlap 0.9578 / RMS phase 0.276 rad -> end 0.9718 / 0.213 rad

## Slice locations (depth, A, z = 0 entrance)

* S-matrix focus-curve layers (phase_var): [82.0]
* S-matrix lateral features: 4, depths [30.5, 43.4, 73.7, 90.9] (support sigma 14.8 A)
* QuantEM baseline depth-profile peaks: [25.0, 56.8, 70.5, 84.1]
* QuantEM sinit depth-profile peaks: [28.0, 68.6]

Slices used to initialise QuantEM:

* uniform: 22 slabs of 4.55 A
* S-informed: 22 slabs, boundaries [0.0, 7.5, 14.3, 20.2, 25.5, 30.4, 35.0, 39.5, 44.0, 48.6, 53.2, 57.9, 62.4, 66.6, 70.6, 74.4, 77.9, 81.4, 84.8, 88.3, 91.9, 95.7, 100.0]
* truth potential depth-profile peaks: [31.0, 71.0]
* atoms in the scanned field: 6 (Au), z range 30.0-70.0 A

## Object

Correlation with the true projected phase over the scanned field:

| image | correlation |
|---|---|
| S-matrix phase z=50 | 0.884 |
| S-matrix phase z=82 | 0.782 |
| QuantEM baseline projected | 0.942 |
| QuantEM sinit projected | 0.961 |

Per-slab correlation of QuantEM slices with the true potential in the same slab (mean over slabs that contain potential):

* baseline: 0.840 over 2 slabs
* sinit: 0.909 over 2 slabs

## How the two differ

* S-matrix + probe (joint): one linear operator S for the whole specimen, fitted to all defocus datasets at once together with each dataset's probe aberrations. No slices are assumed; depth comes afterwards from refocusing S, so it is a measurement with a depth resolution of about lambda/alpha^2 (49 A here), not a reconstructed potential.
* QuantEM multislice: a potential per slice on a fixed slice grid, fitted to one dataset with a probe model. Depth is where the optimiser puts the potential; with uniform slices nothing tells it where features are, the S-informed start (sinit) gives it the S-matrix depths, slice spacing and probe defocus.

Files: smatrix/ (probe_real_space.npy, refocused_phase.npy, focus_curve.npy, smatrix.json, sections.png), quantem_<arm>/ (object_potential.npy, initial_potential.npy, slice_boundaries.npy, probe_real_space.npy, initial_probe_real_space.npy, loss.npy, arm.json, slices.png), probes.png, depth_profiles.png, objects.png, summary.json.
