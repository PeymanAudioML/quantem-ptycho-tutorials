# Realization reconstruction: S-matrix + probe (joint) vs QuantEM multislice

Data: 3 defocus datasets (def_083, def_050, def_017), 300 kV, 20.0 mrad, thickness 100 A, scan step 0.4 A.
(Stand-in from make_standin.py: two 3-atom Au-like clusters at 30 and 70 A in a 100 A slab, 300 kV, 20 mrad, 3 defoci; CPU run, 40 epochs / 40 QuantEM iterations, --detect-k 4.)

## Probe

| dataset | nominal C10 | S-matrix C10 | S-matrix C12a, C12b | truth C10 | S-matrix probe vs truth (overlap, RMS phase err rad) |
|---|---|---|---|---|---|
| 0 (ref) | -83.0 | -83.0 | 0.0, 0.0 | -83.0 (metadata) | 1.0000, 0.000 |
| 1 | -50.0 | -50.9 | -0.2, 0.2 | -50.0 (metadata) | 0.9998, 0.019 |
| 2 | -17.0 | -19.2 | -0.7, 0.2 | -17.0 (metadata) | 0.9990, 0.046 |

QuantEM (dataset 1, started from defocus 50.9 A, probe refined):

* uniform: final loss 0.58159; probe vs truth overlap 0.9784, RMS phase error 0.164 rad
* sinit: final loss 0.50546; probe vs truth overlap 0.9802, RMS phase error 0.154 rad

## Slice locations (depth, A, z = 0 entrance)

* S-matrix focus-curve layers (phase_var): [58.0]
* S-matrix lateral features: 2, depths [33.6, 74.0] (support sigma 14.8 A)
* QuantEM uniform depth-profile peaks: [25.0, 58.3]
* QuantEM sinit depth-profile peaks: [30.2, 73.4]

Slices used to initialise QuantEM:

* uniform: 18 slabs of 5.56 A
* S-informed: 18 slabs, boundaries [0.0, 7.8, 15.1, 21.6, 27.5, 32.9, 38.1, 43.1, 48.0, 52.7, 57.3, 61.8, 66.4, 71.0, 75.8, 80.9, 86.5, 92.9, 100.0]
* truth potential depth-profile peaks: [31.0, 71.0]
* atoms in the scanned field: 6 (Au), z range 30.0-70.0 A

## Object

Correlation with the true projected phase over the scanned field:

| image | correlation |
|---|---|
| S-matrix phase z=50 | 0.951 |
| S-matrix phase z=58 | 0.949 |
| QuantEM uniform projected | 0.950 |
| QuantEM sinit projected | 0.961 |

Per-slab correlation of QuantEM slices with the true potential in the same slab (mean over slabs that contain potential):

* uniform: 0.883 over 2 slabs
* sinit: 0.921 over 2 slabs

## How the two differ

* S-matrix + probe (joint): one linear operator S for the whole specimen, fitted to all defocus datasets at once together with each dataset's probe aberrations. No slices are assumed; depth comes afterwards from refocusing S, so it is a measurement with a depth resolution of about lambda/alpha^2 (49 A here), not a reconstructed potential.
* QuantEM multislice: a potential per slice on a fixed slice grid, fitted to one dataset with a probe model. Depth is where the optimiser puts the potential; with uniform slices nothing tells it where features are, the S-informed start (sinit) gives it the S-matrix depths, slice spacing and probe defocus.

Files: smatrix/ (probe_real_space.npy, refocused_phase.npy, focus_curve.npy, smatrix.json, sections.png), quantem_uniform/ and quantem_sinit/ (object_potential.npy, initial_potential.npy, slice_boundaries.npy, probe_real_space.npy, loss.npy, slices.png), probes.png, depth_profiles.png, objects.png, summary.json.
