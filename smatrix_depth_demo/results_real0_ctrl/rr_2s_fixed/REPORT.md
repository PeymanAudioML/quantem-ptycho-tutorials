# Realization reconstruction: S-matrix + probe (joint) vs QuantEM multislice

Data: 1 defocus datasets (4dstem_quantem_noisefree), 80 kV, 20.0 mrad, thickness 6 A, scan step 4 A.

## Probe

| dataset | nominal C10 | S-matrix C10 | S-matrix C12a, C12b | metadata C10 | S-matrix probe vs truth (overlap, RMS phase err rad) |
|---|---|---|---|---|---|
| 0 (ref) | -500.0 | -500.0 | 0.0, 0.0 | -500.0 | - |

## What QuantEM was started from

| QuantEM input | baseline |
|---|---|
| probe | nominal: C10 -500.0 A, C12 0.0 A, phi12 0 deg |
| probe hand-over check (initial QuantEM probe vs S-matrix probe overlap) | 0.9974 |
| slice planes | uniform, 2 slabs |
| initial object | flat |
| geometry | 0.200 A px, z = 0 entrance, defocus = -C10 |
| optimiser | object lr 0.005 x 200 it; parametric probe lr 0.0 from it 41 |
| probe changed by QuantEM (1 - overlap initial/final) | 0.00e+00 |
| learned aberrations (parametric) | C10 -500.0 A, C12 0.0 A, phi12 0 deg |

QuantEM results (dataset 0):

* baseline: final loss 14.71415

## Slice locations (depth, A, z = 0 entrance)

* S-matrix focus-curve layers (phase_var): [0.0]
* S-matrix lateral features: 200, depths [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.6, 0.8, 2.3, 2.4, 3.0, 3.0, 4.1, 4.3, 4.5, 4.6, 4.6, 5.2, 5.3, 5.4, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 5.7] (support sigma 31.3 A)
* QuantEM baseline depth-profile peaks: []

Slices used to initialise QuantEM:

* uniform: 2 slabs of 3.00 A
* S-informed: 2 slabs, boundaries [0.0, 3.0, 6.0]

## Object


## How the two differ

* S-matrix + probe (joint): one linear operator S for the whole specimen, fitted to all defocus datasets at once together with each dataset's probe aberrations. No slices are assumed; depth comes afterwards from refocusing S, so it is a measurement with a depth resolution of about lambda/alpha^2 (104 A here), not a reconstructed potential.
* QuantEM multislice: a potential per slice on a fixed slice grid, fitted to one dataset with a probe model. Depth is where the optimiser puts the potential; with uniform slices nothing tells it where features are, the S-informed start (sinit) gives it the S-matrix depths, slice spacing and probe defocus.

Files: smatrix/ (probe_real_space.npy, refocused_phase.npy, focus_curve.npy, smatrix.json, sections.png), quantem_<arm>/ (object_potential.npy, initial_potential.npy, slice_boundaries.npy, probe_real_space.npy, initial_probe_real_space.npy, loss.npy, arm.json, slices.png), probes.png, depth_profiles.png, objects.png, summary.json.
