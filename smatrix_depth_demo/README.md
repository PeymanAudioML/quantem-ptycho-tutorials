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
