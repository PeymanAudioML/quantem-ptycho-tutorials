# S-matrix recovery + depth sectioning (CPU demo)

Reproduces the two key algorithms of Brown et al., *A single-projection three-dimensional
reconstruction algorithm for STEM data* (arXiv:2011.07652) on simulated data:

1. **S-matrix recovery** – amplitude-flow gradient descent for the scattering matrix
   `S_b(r)` from 4D-STEM data recorded at several probe defocus values
   (authors' `GradDS/AmpflowS.py`).
2. **Depth sectioning** – shift each beam to the origin, apply a Fresnel propagator plus
   paraxial shift, sum the beams coherently (authors' `GradDS/Optical_sectioning.py`).

The authors' scripts need CUDA/CuPy, `py_multislice`, removed PyTorch APIs and raw data that
is not in the archive, so `smatrix_depth_demo.py` is a NumPy re-implementation of the same
algorithms, with a small built-in multislice simulator for the test data.

## Test sample / data
300 kV, 20 mrad. Three phase layers in a 300 Å thick sample: ring of atoms (z = 30 Å),
5×5 square lattice (z = 150 Å), triangle outline (z = 270 Å). Probe defoci −200, −100, 0 Å
(probe focused at z = 200, 100, 0 Å). 25×25 scan (0.8 Å step) × 64×64 detector × 3 defoci,
Poisson noise at 2×10⁵ e⁻/pattern. Reconstruction: 529 beams, 25 iterations
(`MU=60 NITER=25 python smatrix_depth_demo.py`, ≈15 min on 4 CPU cores), then
`python make_figures.py`.

## Results (`results/`)
| file | content |
|---|---|
| `fig1_sample.png` | ground-truth layers + geometry |
| `fig2_4dstem_data.png` | example diffraction patterns and PACBED per defocus |
| `fig3_loss.png` | amplitude-flow convergence (0.050 → 0.0017) |
| `fig6_smatrix_beams.png` | recovered S-matrix entries for three input beams |
| `fig4_depth_sections.png`, `depth_sections.gif` | phase sections vs depth |
| `fig5_depth_localisation.png` | correlation with each true layer vs depth |

Depth localisation (centre of the in-focus plateau): ring 30 Å → 30 Å, square 150 Å → 150 Å,
triangle 270 Å → 275 Å; depth of field about 50–80 Å (≈ 2λ/α²).

## Notes / caveats
* Depth axis: diffraction intensities are insensitive to free-space propagation of the exit
  wave, so S is only defined up to its output plane. The input plane is fixed by the known
  illumination (entrance surface), so sections are labelled by depth from the entrance surface
  (forward propagation by z). This is my convention; the authors' `t` axis in `Test.py` may be
  defined differently.
* Simulated (noisy, noise-free-model-consistent) data, not experimental data; no distortion
  correction or alignment (`Aligner.py`, `Distortion_correct.py`) was needed or used.
* Large intermediate files (`Smatrix_complex64.npy`, `layers_and_data.npz`) are git-ignored.

## Comparison with tcBF (`compare_tcbf.py`)
Same simulated data, same defocus values, same scoring (correlation of the section with each true
layer). tcBF here = bright-field disk only (429 pixels inside 0.9 of the aperture edge); every BF
pixel image is shifted by -λ·k·(z − z_f) (geometric sign, not fitted) and summed, for each trial
depth z. A separate CTF-correction step (as in "aberration-corrected" BF variants) is **not**
implemented. Scan sampling (0.8 Å) limits tcBF resolution; the S-matrix also uses the
dark-field region of the diffraction pattern.

| method | ring (30 Å) found / r | square (150 Å) | triangle (270 Å) |
|---|---|---|---|
| tcBF, df=0 only (1/3 of data) | 50 Å / +0.78 | 140 Å / +0.84 | 230 Å / +0.44 |
| tcBF, df=−200 only (1/3 of data) | 55 Å / −0.82 | 175 Å / −0.91 | 275 Å / +0.76 |
| tcBF, all 3 defocus (same data) | 25 Å / −0.84 | 165 Å / +0.77 | 265 Å / +0.78 |
| S-matrix + refocusing (all 3) | 30 Å / +0.87 | 150 Å / +0.85 | 275 Å / +0.83 |

Figures: `fig7_compare_images.png`, `fig8_compare_depth_curves.png`; numbers in
`comparison_summary.json`. Negative r = contrast-inverted image (CTF effect).

## Harder geometries: where S-matrix beats tcBF (`run_config.py`, `sweep_compare.py`)
6 overlapping layers (ring, square lattice, triangle, plus, X, diamond; all centred on the same
axis; 50 Å spacing, z = 25…275 Å), phase per atom 1.5 and 2.5 rad (baseline: 3 layers, 1.0 rad).
Same data for both methods (all 3 defocus, 2×10⁵ e⁻/pattern, 25 S-matrix iterations, single
noise draw). "Resolved" = depth error ≤ 25 Å and |r| > 0.5 with the true layer.

| geometry | method | layers resolved | mean abs. depth error | mean abs. r | contrast-inverted layers |
|---|---|---|---|---|---|
| 3 layers, 1.0 rad | tcBF (3 df) | 3/3 | 8.5 Å | 0.80 | 1 |
| | S-matrix | 3/3 | 1.7 Å | 0.85 | 0 |
| 6 layers, 1.5 rad | tcBF (3 df) | 3/6 | 24.2 Å | 0.55 | 3 |
| | S-matrix | 6/6 | 10.9 Å | 0.70 | 0 |
| 6 layers, 2.5 rad | tcBF (3 df) | 2/6 | 49.2 Å | 0.47 | 4 |
| | S-matrix | 6/6 | 10.9 Å | 0.67 | 0 |

Figures and numbers: `results_sweep/` (`sweep_summary.png`, `sweep_images_*.png`,
`sweep_curves_*.png`, `sweep_summary.json`); per-config data in `results_six_*/`.
Caveats: the S-matrix sections of overlapping middle layers show clear cross-talk (r ≈ 0.5–0.6);
the outer layers are found at the edge of the 0–300 Å scan range; the strong-phase run is less
converged (loss 0.0094 vs 0.0042); one noise draw, no repeats.

## Polycrystal test (`run_poly.py`, `analyze_poly.py`, `results_poly/`)
CaAl₂Si₂O₈ (the authors' 24-atom structure file), 8 randomly rotated 3-D Voronoi grains, 150 Å thick
(17,807 atoms, 60 slices of 2.5 Å; projected phase mean 2.3 rad, max 10 rad, i.e. strongly dynamical).
Probe defoci −150/−75/0 Å, 30 S-matrix iterations (final loss 0.0146, still slowly decreasing).
Reference = sum of the true slice phases within ±20 Å of each section depth; scored by the correlation
of each section with the true slab at every depth (`fig_p3_depth_matrices.png`).

| method (full-band truth) | mean r at correct depth | mean depth error | within 10 Å | selectivity | inverted (of 16) |
|---|---|---|---|---|---|
| tcBF, df=0 only | +0.58 | 12.5 Å | 69% | 0.44 | 1 |
| tcBF, 3 defocus | +0.09 | 17.5 Å | 50% | 0.36 | 8 |
| S-matrix + refocusing | +0.75 | 9.4 Å | 81% | 0.61 | 0 |

(selectivity = |r| at the right depth minus mean |r| at depths ≥ 60 Å away). Against the same truth
blurred to the 1.6 Å scan sampling, every score drops (S-matrix 0.34, tcBF df=0 0.19) and the ranking is
unchanged. Caveats: static atoms and Gaussian potentials (strength ∝ Z^0.7), zone-axis grains only;
all methods are only weakly depth-selective (a section correlates with slabs ±40 Å away because the depth
of field is comparable to the slab); the advantage here is moderate, not the large gap seen in the
overlapping-layer test; one random structure and one noise draw; S-matrix not fully converged.

## tcBF → ACBF with quantem's own implementation (`run_quantem_acbf.py`, `make_quantem_figs.py`, `results_quantem/`)
quantem (`DirectPtychography`) defines tcBF = `parallax` kernel and ACBF = `single-sideband` kernel; its
workflow fits the aberrations (here defocus C10) with the parallax kernel, then reconstructs with the
ACBF kernel. I ran it on the same simulated 4D data (needs torch + quantem; CPU). Depth sections = ACBF /
parallax reconstructions at C10 = z − z_f (z_f = −df). The sign of C10 and the detector orientation were
calibrated once on the 3-layer data (`results_quantem_calibration.json`: +1/no transpose gave peaks at
40/140/270 Å for layers at 30/150/270 Å; the other three options scored far worse).
"ACBF 3 df" = average of the ACBF sections from the three defocus datasets.

| geometry | method | resolved | mean depth err | mean r |
|---|---|---|---|---|
| 3 layers | tcBF mine (3 df) | 3/3 | 8.5 Å | 0.80 |
| | tcBF quantem (df=0) | 1/3 | 30.0 Å | 0.75 |
| | ACBF quantem (df=0) | 3/3 | 6.7 Å | 0.82 |
| | **ACBF quantem (3 df)** | 3/3 | 1.8 Å | 0.86 |
| | S-matrix | 3/3 | 1.7 Å | 0.85 |
| 6 layers, 1.5 rad | tcBF mine (3 df) | 3/6 | 24.2 Å | 0.55 |
| | ACBF quantem (df=0) | 5/6 | 6.7 Å | 0.57 |
| | **ACBF quantem (3 df)** | 6/6 | 10.9 Å | 0.68 |
| | S-matrix | 6/6 | 10.9 Å | 0.70 |
| polycrystal | tcBF mine (3 df) | – | 17.5 Å | 0.09 |
| | ACBF quantem (df=0) | – | 11.9 Å | 0.62 |
| | **ACBF quantem (3 df)** | – | 9.4 Å | 0.69 |
| | S-matrix | – | 9.4 Å | 0.74 |

**Important correction to the earlier comparisons:** my hand-written tcBF (shift-and-sum, no CTF correction)
was a weak baseline. With quantem's aberration-corrected BF (ACBF) and the same three defocus datasets, the
bright-field approach matches the S-matrix to within about 0.05 in correlation and has the same depth error
in all three geometries. The S-matrix images are somewhat cleaner (less ringing) and its correlation is
slightly higher, but in this simulated, weak-to-moderate-phase setting that is a small difference.
The parallax kernel gives a blank image when the section depth equals the probe focus (C10 = 0).
tcBF defocus fits (`summary.json`, `fits`) land on one dominant depth per dataset, e.g. 30 Å for df=−200
in the 3-layer case, but vary between 30 and 260 Å for the 6-layer sample.

## Paper-faithful tcBF / acBF vs S-matrix (`paper_acbf.py`, `run_paper_compare.py`, `results_paper/`)
Implements Ma, Lee, Shi, Muller, Zeltmann, "Parallax Depth Sectioning and 3D Reconstruction in 4D-STEM" in NumPy
from its equations (defocus-only): D = I − I0; tcBF = Σ_Θ D̃ e^{−i2πq·Θ(Δf−z)} (Eq. 4); acBF additionally multiplies each
pixel/frequency by e^{−i arg PCTFres_z} (Eq. 5–7), evaluated at depth z. tcBF and acBF are alternatives
(acBF = tcBF + per-pixel phase correction), each from **one** 4D dataset; each of my 3 defocus datasets is processed on its
own. Conventions fixed once on the 3-layer data: detector handedness Θ = −λk, and a global sign flip of the acBF output
(uniform at all depths and structures; Rose's phase-sign convention vs my e^{+iφ}). "3 df summed" is my extension (sum of the
three single-dataset acBF stacks), the only variant with the same total data as the S-matrix.

| structure | tcBF paper, single df (mean err / mean\|r\|) | acBF paper, single df | acBF paper, 3 df summed* | S-matrix (3 df) |
|---|---|---|---|---|
| 3 layers | 29.5 Å / 0.80 | 11.7 Å / 0.85 | 3.5 Å / 0.88 | 1.7 Å / 0.85 |
| 6 layers, 1.5 rad | 34.2 Å / 0.59 | 15.6 Å / 0.66 | 11.7 Å / 0.71 | 10.9 Å / 0.70 |
| 6 layers, 2.5 rad | 37.0 Å / 0.53 | 20.9 Å / 0.57 | 13.4 Å / 0.64 | 10.9 Å / 0.67 |
| polycrystal (r vs slab, 16 depths) | 13.1/14.4/11.9 Å, r = −0.48/−0.04/+0.58 | 14.4–16.9 Å, r = +0.56…+0.65 | 9.4 Å / +0.69 | 9.4 Å / +0.74 |

Layers resolved (|err| ≤ 25 Å and |r| > 0.5): 3-layer acBF 3 df 3/3, S-matrix 3/3; 6 layers 1.5 rad 6/6 vs 6/6;
6 layers 2.5 rad acBF 3 df 5/6 vs S-matrix 6/6. Single-dataset acBF depends strongly on the defocus (e.g. 3 layers:
1.8 Å at df=−200, 21.7 Å at df=−100, 11.6 Å at df=0), consistent with the paper's advice to defocus beyond about twice the
depth of field. Figures: `fig_pp1_summary.png`, `fig_pp_images_*.png`; numbers: `summary.json`.
Caveats: simulated data, one noise draw, weak/moderate-phase regime; the paper itself warns the linear model fails for thick
or heavy samples.

## Check against the authors' own code (`check_authors_code.py`, `debug_first_chunk.py`, `results_authors/`)
The authors' **unmodified** `GradDS/AmpflowS.py` and `Optical_sectioning.py` were run on my simulated 3-layer data
(CPU only): `cupy` replaced by a NumPy stand-in (`authors_shim/cupy`), removed NumPy aliases restored, and the legacy
`torch.fft(x, ndim, normalized)` re-implemented. Their `padding=1.5` option gives the same 192×192 (38.4 Å) S-matrix grid and
529 beams as mine. **Caveat:** the Dropbox download hosts are blocked in my sandbox, so I re-typed the authors' two files
from the text returned by the Dropbox connector; they are NOT byte-identical to `SHA256_MANIFEST.json` (49 and 33 bytes
shorter, probably whitespace, unverified). Please re-run `check_authors_code.py` with the original files to confirm.

| test | result |
|---|---|
| Refocusing: authors' `depth_section_reconstruction` vs my `depth_sections`, same S-matrix | phase images correlate 1.0000 at every depth with `t = −z` (and ≈0.2 with `t = +z`, which also confirms my depth convention) |
| Recovery, first chunk, illumination vector | identical (ratio exactly K, phase difference 1.5e-8 rad) |
| Recovery, first chunk, forward prediction | identical (max difference 1.9e-8) |
| Residual on pixels where the model predicts a non-zero amplitude (13% in iteration 1) | identical (2e-8) |
| Residual on the other pixels (model amplitude numerically 0, ~49% of residual energy) | **differs**: the phase of a numerical zero is rounding noise, so the two codes pick different arbitrary phases (≈200 pixels per pattern). Intrinsic to amplitude flow from a vacuum start |
| Back-projection update given the same residual (step size mapped by μ_authors = MU·nscan/(nbeams·overlap) = 0.369 for MU = 60) | identical (7e-5 relative, float32) |
| After 2 iterations at equivalent step | S-matrices differ (relative difference 1.04, per-beam update correlation 0.76), due to the arbitrary phases above |
| **Final outcome**: authors' code with its own defaults (μ = 1.0, 10 iterations, 1885 s on CPU) vs mine (MU = 60, 25 iterations) | depth sections correlate 0.99–1.00 at every depth; layers found at 29.9/150.0/275.1 Å (authors) vs 30.0/150.0/275.1 Å (mine); depth error 1.7 Å for both; mean \|r\| 0.86 vs 0.85 |
| My forward-model amplitude loss on all 1875 patterns | vacuum 0.0572, authors' final S 0.00127, my final S 0.00153 (the authors' larger step converges faster per iteration) |

Conclusion: my S-matrix recovery and refocusing implement the authors' algorithm; they agree step by step except for the
arbitrary phase on numerically zero pixels, and they reach equivalent depth results on this dataset. My version runs about
6× faster per iteration on CPU (≈32 s vs ≈190 s), my step-size parameter MU is not the authors' μ, and the equivalence was
shown for one dataset (3 layers, one noise draw), not for experimental data, scan distortion, specimen tilt or diffraction shift.
