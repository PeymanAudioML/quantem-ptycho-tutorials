# Realization reconstruction: S-matrix + probe (joint) vs QuantEM multislice

Data: 3 defocus datasets (def_083, def_050, def_017), 300 kV, 20.0 mrad, thickness 100 A, scan step 0.4 A.

## Probe

| dataset | nominal C10 | S-matrix C10 | S-matrix C12a, C12b | truth C10 | S-matrix probe vs truth (overlap, RMS phase err rad) |
|---|---|---|---|---|---|
| 0 (ref) | -83.0 | -83.0 | 0.0, 0.0 | -83.0 (metadata) | 1.0000, 0.000 |
| 1 | -50.0 | -44.2 | 6.8, -4.1 | -50.0 (metadata) | 0.9992, 0.040 |
| 2 | -17.0 | -22.3 | -3.3, 4.5 | -17.0 (metadata) | 0.9990, 0.044 |

## What QuantEM was started from

| QuantEM input | baseline | sprobe | sinit |
|---|---|---|---|
| probe | nominal: C10 -50.0 A, C12 0.0 A, phi12 0 deg | smatrix: C10 -44.2 A, C12 7.9 A, phi12 -75 deg | smatrix: C10 -44.2 A, C12 7.9 A, phi12 -75 deg |
| probe hand-over check (initial QuantEM probe vs S-matrix probe overlap) | 0.9665 | 0.9926 | 0.9926 |
| slice planes | uniform, 18 slabs | uniform, 18 slabs | smatrix, 18 slabs |
| initial object | flat | flat | candidates |
| geometry | 0.200 A px, z = 0 entrance, defocus = -C10 | 0.200 A px, z = 0 entrance, defocus = -C10 | 0.200 A px, z = 0 entrance, defocus = -C10 |
| optimiser | object lr 0.005 x 60 it; pixelated probe lr 0.0001 from it 13 | object lr 0.005 x 60 it; pixelated probe lr 0.0001 from it 13 | object lr 0.005 x 60 it; pixelated probe lr 0.0001 from it 13 |
| probe changed by QuantEM (1 - overlap initial/final) | 7.97e-03 | 5.33e-03 | 5.03e-03 |

QuantEM results (dataset 1):

* baseline: final loss 0.63483; probe vs truth: start overlap 0.9578 / RMS phase 0.276 rad -> end 0.9728 / 0.207 rad
* sprobe: final loss 0.56991; probe vs truth: start overlap 0.9918 / RMS phase 0.040 rad -> end 0.9891 / 0.076 rad
* sinit: final loss 0.49797; probe vs truth: start overlap 0.9918 / RMS phase 0.040 rad -> end 0.9893 / 0.073 rad

## Slice locations (depth, A, z = 0 entrance)

* S-matrix focus-curve layers (phase_var): [58.0]
* S-matrix lateral features: 2, depths [33.4, 74.1] (support sigma 14.8 A)
* QuantEM baseline depth-profile peaks: [25.0, 58.3]
* QuantEM sprobe depth-profile peaks: [19.4, 58.3, 69.4]
* QuantEM sinit depth-profile peaks: [24.5, 68.7]

Slices used to initialise QuantEM:

* uniform: 18 slabs of 5.56 A
* S-informed: 18 slabs, boundaries [0.0, 7.8, 15.0, 21.5, 27.4, 32.9, 38.1, 43.1, 47.9, 52.7, 57.3, 61.8, 66.4, 71.0, 75.8, 80.9, 86.5, 92.9, 100.0]
* truth potential depth-profile peaks: [31.0, 71.0]
* atoms in the scanned field: 6 (Au), z range 30.0-70.0 A

## Object

Correlation with the true projected phase over the scanned field:

| image | correlation |
|---|---|
| S-matrix phase z=50 | 0.949 |
| S-matrix phase z=58 | 0.947 |
| QuantEM baseline projected | 0.939 |
| QuantEM sprobe projected | 0.939 |
| QuantEM sinit projected | 0.951 |

Per-slab correlation of QuantEM slices with the true potential in the same slab (mean over slabs that contain potential):

* baseline: 0.853 over 2 slabs
* sprobe: 0.842 over 2 slabs
* sinit: 0.902 over 2 slabs

## How the two differ

* S-matrix + probe (joint): one linear operator S for the whole specimen, fitted to all defocus datasets at once together with each dataset's probe aberrations. No slices are assumed; depth comes afterwards from refocusing S, so it is a measurement with a depth resolution of about lambda/alpha^2 (49 A here), not a reconstructed potential.
* QuantEM multislice: a potential per slice on a fixed slice grid, fitted to one dataset with a probe model. Depth is where the optimiser puts the potential; with uniform slices nothing tells it where features are, the S-informed start (sinit) gives it the S-matrix depths, slice spacing and probe defocus.

Files: smatrix/ (probe_real_space.npy, refocused_phase.npy, focus_curve.npy, smatrix.json, sections.png), quantem_<arm>/ (object_potential.npy, initial_potential.npy, slice_boundaries.npy, probe_real_space.npy, initial_probe_real_space.npy, loss.npy, arm.json, slices.png), probes.png, depth_profiles.png, objects.png, summary.json.
