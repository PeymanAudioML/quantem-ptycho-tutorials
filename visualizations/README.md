# Step-by-step figures: white-noise object phase retrieval

`white_noise_steps.py` re-runs `simulation_notebooks/white-noise-object.ipynb` and
`white_noise_object_phase_retrieval.ipynb` end to end and saves one figure per step to
`white_noise_steps/`. It runs on CPU in about 5 minutes (most of it is the 200-trial Optuna search).

```bash
source .venv/bin/activate
python visualizations/white_noise_steps.py
```

| Figure | What it shows |
|---|---|
| `step1_object.png` | The white-noise phase object: flat \|FFT\|, random Fourier phases |
| `step2_probe.png` | Aperture, aberration phase χ(k) (defocus + astigmatism), the resulting probe in real space |
| `step3_forward_model.png` | One scan position: object × probe → exit wave → \|FFT\|² diffraction, and the weak signal I − ⟨I⟩ |
| `step3b_scan_overlap.png` | Overlapping probe footprints on the scan grid |
| `step4_rotation_dataset.png` | The −13° detector rotation, mean pattern, virtual bright-field image |
| `step5_reconstructions.png` | Five direct-ptychography kernels (SSB, OBF, MF, parallax, iCOM) vs ground truth |
| `step6_transfer_functions.png` | \|FFT\| of each reconstruction, i.e. its transfer function, plus radial profiles |
| `step7_wrong_parameters.png` | SSB with defocus, astigmatism or rotation ignored |
| `step8_parallax_shifts.png` | Image shift per bright-field detector pixel: the aberrations in vector form |
| `step9_hyperparameter_fit.png` | Optuna vs cross-correlation vs least-squares recovery of C10, C12, φ12 and rotation |
| `probe_and_pattern_sizes.png` | Size and shape of the probe (real space) and diffraction pattern (reciprocal space), annotated — from `probe_and_pattern_sizes.py` |
| `transfer_function_ellipse.png` | \|FFT(recon)\| with the notebook's rotation (−13 read as rad) vs the correct one, plus predicted astigmatic zero-rings — from `transfer_function_ellipse.py` |

Note: with the quantEM commit pinned in `uv.lock`, `DirectPtychography` expects `rotation_angle`
in **radians**. The tutorial notebook passes `rotation_angle=-13` (meant as degrees), which quantEM
reads as −13 rad ≈ −24.8°. This script passes `np.deg2rad(-13)`. With that fix the SSB
reconstruction's correlation with ground truth goes from 0.31 to 0.65.
