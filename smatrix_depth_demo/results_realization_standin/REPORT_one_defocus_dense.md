# Realization reconstruction: S-matrix + probe (joint) vs QuantEM multislice

Data: 1 defocus datasets (def_050), 300 kV, 20.0 mrad, thickness 100 A, scan step 0.2 A.

## Probe

| dataset | nominal C10 | S-matrix C10 | S-matrix C12a, C12b | metadata C10 | S-matrix probe vs truth (overlap, RMS phase err rad) |
|---|---|---|---|---|---|
| 0 (ref) | -50.0 | -50.0 | 0.0, 0.0 | -50.0 | 0.9627, 0.276 |

## What QuantEM was started from

| QuantEM input | baseline | sinit |
|---|---|---|
| probe | nominal: C10 -50.0 A, C12 0.0 A, phi12 0 deg | smatrix: C10 -50.0 A, C12 0.0 A, phi12 0 deg |
| probe hand-over check (initial QuantEM probe vs S-matrix probe overlap) | 0.9926 | 0.9926 |
| slice planes | uniform, 20 slabs | smatrix, 20 slabs |
| initial object | flat | candidates |
| geometry | 0.200 A px, z = 0 entrance, defocus = -C10 | 0.200 A px, z = 0 entrance, defocus = -C10 |
| optimiser | object lr 0.005 x 60 it; pixelated probe lr 0.0001 from it 13 | object lr 0.005 x 60 it; pixelated probe lr 0.0001 from it 13 |
| probe changed by QuantEM (1 - overlap initial/final) | 1.97e-02 | 1.80e-02 |

QuantEM results (dataset 0):

* baseline: final loss 3.01101; probe vs truth: start overlap 0.9578 / RMS phase 0.276 rad -> end 0.9807 / 0.160 rad
* sinit: final loss 2.59148; probe vs truth: start overlap 0.9578 / RMS phase 0.276 rad -> end 0.9816 / 0.152 rad

## Slice locations (depth, A, z = 0 entrance)

* S-matrix focus-curve layers (phase_var): [74.0]
* S-matrix lateral features: 3, depths [34.3, 46.3, 83.8] (support sigma 14.8 A)
* QuantEM baseline depth-profile peaks: [47.5, 77.5]
* QuantEM sinit depth-profile peaks: [34.1, 52.1, 65.6, 78.5, 91.9]

Slices used to initialise QuantEM:

* uniform: 20 slabs of 5.00 A
* S-informed: 20 slabs, boundaries [0.0, 7.8, 15.0, 21.3, 26.8, 31.8, 36.5, 40.9, 45.4, 49.8, 54.3, 58.9, 63.4, 67.8, 72.1, 76.4, 80.6, 84.9, 89.4, 94.4, 100.0]
* truth potential depth-profile peaks: [31.0, 71.0]
* atoms in the scanned field: 6 (Au), z range 30.0-70.0 A

## Object

Correlation with the true projected phase over the scanned field:

| image | correlation |
|---|---|
| S-matrix phase z=50 | 0.899 |
| S-matrix phase z=74 | 0.831 |
| QuantEM baseline projected | 0.867 |
| QuantEM sinit projected | 0.919 |

Per-slab correlation of QuantEM slices with the true potential in the same slab (mean over slabs that contain potential):

* baseline: 0.761 over 2 slabs
* sinit: 0.833 over 2 slabs

## How the two differ

* S-matrix + probe (joint): one linear operator S for the whole specimen, fitted to all defocus datasets at once together with each dataset's probe aberrations. No slices are assumed; depth comes afterwards from refocusing S, so it is a measurement with a depth resolution of about lambda/alpha^2 (49 A here), not a reconstructed potential.
* QuantEM multislice: a potential per slice on a fixed slice grid, fitted to one dataset with a probe model. Depth is where the optimiser puts the potential; with uniform slices nothing tells it where features are, the S-informed start (sinit) gives it the S-matrix depths, slice spacing and probe defocus.

Files: smatrix/ (probe_real_space.npy, refocused_phase.npy, focus_curve.npy, smatrix.json, sections.png), quantem_<arm>/ (object_potential.npy, initial_potential.npy, slice_boundaries.npy, probe_real_space.npy, initial_probe_real_space.npy, loss.npy, arm.json, slices.png), probes.png, depth_profiles.png, objects.png, summary.json.
