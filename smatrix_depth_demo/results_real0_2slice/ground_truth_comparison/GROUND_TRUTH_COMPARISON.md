# Ground-truth comparison (saved results only; nothing was re-reconstructed)

Run: `results_real0_2slice`.  Data: `data/realization_0/4dstem_quantem_noisefree` (37x37 scan, 4 A step, 80 kV, 20 mrad, focus 500 A, noise-free).  Specimen: Si, **6 A thick**, one defocus.

## Missing outputs

* quantem_sprobe/ (arm not run or not saved in results_real0_2slice)
* quantem_sinit/ (arm not run or not saved in results_real0_2slice)
* No 40-slice QuantEM result exists (that run was stopped before saving).
* sprobe and sinit were never run for this dataset, so only the QuantEM **baseline** (2 uniform slabs, nominal probe, parametric probe updated) can be evaluated.

Truth orientation check: mean projected potential at the atoms is 692 with array axes (x, y) and 83.8 transposed -> axes (x, y) used.

## 1. Projected object

Physical potential: the QuantEM object is a phase per slice in rad (obj_type potential); its sum over slices is compared with the true projected potential times the interaction constant (sigma = 0.001009 rad/(V A), assuming potential.zarr holds V*A per slice).  The S-matrix refocused phase is **not** a physical potential (relative phase of a refocused wave): only the correlation is meaningful, not the scale or the error in rad.

Correlation with the true projected potential (full scanned field; blur sigma = Gaussian width applied to the truth to mimic the reconstruction's resolution):

| image | kind | σ=0 Å | σ=0.5 Å | σ=1 Å | σ=1.5 Å | σ=2 Å |
|---|---|---|---|---|---|---|
| QuantEM baseline | physical | 0.508 | 0.465 | 0.301 | 0.150 | 0.093 |
| S-matrix refocused phase z=0 | contrast | 0.253 | 0.344 | 0.291 | 0.163 | 0.093 |
| S-matrix refocused phase z=3 | contrast | 0.255 | 0.347 | 0.293 | 0.164 | 0.093 |
| S-matrix refocused phase z=6 | contrast | 0.257 | 0.350 | 0.295 | 0.164 | 0.093 |
| S-matrix refocused phase z=7 (best |corr|) | contrast | 0.257 | 0.350 | 0.295 | 0.165 | 0.093 |

Same, interior only (10 Å border removed):

| image | σ=0 Å | σ=0.5 Å | σ=1 Å | σ=1.5 Å | σ=2 Å |
|---|---|---|---|---|---|
| QuantEM baseline | 0.538 | 0.486 | 0.316 | 0.160 | 0.103 |
| S-matrix refocused phase z=0 | 0.274 | 0.369 | 0.306 | 0.167 | 0.096 |
| S-matrix refocused phase z=3 | 0.277 | 0.372 | 0.308 | 0.168 | 0.096 |
| S-matrix refocused phase z=6 | 0.280 | 0.375 | 0.311 | 0.169 | 0.096 |
| S-matrix refocused phase z=7 (best |corr|) | 0.281 | 0.377 | 0.312 | 0.169 | 0.096 |

QuantEM physical-scale agreement (full field; truth blurred by the σ with the highest correlation):

| image | best σ (Å) | corr | slope (recon = s·truth + b) | offset (rad) | RMSE raw (rad) | RMSE after fit (rad) | NRMSE after fit |
|---|---|---|---|---|---|---|---|
| QuantEM baseline | 0 | 0.508 | 0.50 | -0.009 | 0.099 | 0.074 | 0.86 |

Figure: `1_projected_object.png` (top: images; middle: raw difference; bottom: difference after the fitted scale and offset).

## 2. Depth

Truth: 8 slabs of 0.75 Å in the scanned field; mean projected phase per slab [0.009999999776482582, 0.009999999776482582, 0.010999999940395355, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582] rad (total 0.082 rad).
Truth atoms in the scanned field: 6070, z = 3.00 ± 1.73 Å (range 0.00-6.00).

QuantEM baseline: 2 slabs.  Mean phase per slab (recon vs truth, rad) and in-slab correlation:

| slab (Å) | recon mean | truth mean | in-slab corr |
|---|---|---|---|
| 0.0–3.0 | 0.016 | 0.041 | 0.175 |
| 3.0–6.0 | 0.016 | 0.041 | 0.170 |

Two 3 Å slabs of a 6 Å sheet with a 20 mrad probe: the split between slabs is **not** constrained by the data (depth resolution ≈ λ/α² ≈ 104 Å), so these per-slab numbers are not depth measurements.

S-matrix focus curve: maximum at z = -1.2 Å over a scanned depth range of -1–7 Å; the region above half maximum spans -1–3 Å.  The expected depth resolution is λ/α² = 104 Å, >> the 6 Å specimen.  The 200 detected lateral features all carry depth ≈ 0.17 Å and the detected layer at [0.04999999999999982]: **these are not credible z-coordinates** (they sit at the edge of the search range and the sample is thinner than the resolution).

Figure: `2_depth.png`.

## 3. Probe

| probe | overlap with truth | RMS phase error (rad) |
|---|---|---|
| S-matrix (dataset 0 = gauge reference: kept at nominal) | 0.9994 | 0.000 |
| QuantEM baseline: start | 0.9986 | 0.000 |
| QuantEM baseline: final | 0.9963 | 0.069 |

QuantEM baseline learned aberrations: {'C10': -491.96044921875, 'C12': -0.012946367263793945, 'phi12': 5.631860256195068} (truth metadata: C10 = -500 Å, C12 = 0).

Overlap = |<est, truth>| / (|est||truth|) on the aperture beams, after removing piston and tilt from the phase and re-centring the intensity centroid (same definition as run_realization.py).

Figure: `3_probe.png`.

## 4. Atomic structure

Scanned field: 6070 atoms; 4242 projected columns in the interior (10 Å border excluded; atoms within 0.4 Å laterally merged).  Atoms are plotted as given (x = array axis 1 of the data/potential, y = axis 2); lateral localisation = distance from each projected column to the nearest local maximum of the image.

Two directions.  **Maxima → columns** (how far each image maximum is from the nearest atomic column; the meaningful one when maxima are blobs of several atoms) and **columns → maxima** (how far each column is from a maximum; limited by density since there are far more columns than maxima).

| image | columns | maxima | maxima→column median (Å) | random points→column (Å) | maxima within 0.5 Å of a column | columns→maximum median (Å) | random points (Å) | ≤1.0 Å |
|---|---|---|---|---|---|---|---|---|
| truth projected (blur σ=0.5 Å, benchmark) | 4242 | 450 | 0.22 | 0.81 | 0.92 | 2.85 | 2.86 | 0.22 |
| QuantEM baseline | 4242 | 376 | 0.19 | 0.84 | 0.97 | 3.46 | 3.13 | 0.16 |
| S-matrix refocused phase z=0 | 4242 | 128 | 0.49 | 0.81 | 0.52 | 5.69 | 5.36 | 0.05 |
| S-matrix refocused phase z=3 | 4242 | 129 | 0.49 | 0.80 | 0.52 | 5.65 | 5.35 | 0.05 |
| S-matrix refocused phase z=6 | 4242 | 137 | 0.46 | 0.86 | 0.55 | 5.42 | 5.22 | 0.05 |
| S-matrix refocused phase z=7 (best |corr|) | 4242 | 138 | 0.46 | 0.87 | 0.56 | 5.40 | 5.18 | 0.05 |

The column density (about 0.28 columns/Å², mean spacing ~1.9 Å) is much higher than the number of distinguishable image maxima, so this nearest-maximum distance is limited by density, not only by the reconstruction: compare with the truth benchmark and the random-points column.  Image maxima are blobs of several atoms (see the zoom figure); atoms inside a blob (e.g. dumbbells) are not separated.

Fourier ring correlation, QuantEM baseline vs true projected potential (unblurred truth): 0.54 at 0.1 Å⁻¹, 0.56 at 0.2 Å⁻¹, 0.87 at 0.3 Å⁻¹, 0.67 at 0.4 Å⁻¹, 0.70 at 0.5 Å⁻¹, 0.40 at 0.7 Å⁻¹, 0.35 at 1 Å⁻¹, 0.19 at 1.5 Å⁻¹, 0.08 at 2 Å⁻¹.  First crossing below 0.5 above 0.3 Å⁻¹: 0.66 Å⁻¹ (period 1.5 Å).
Si lattice spacings: (111) 3.14 Å, (220) 1.92 Å, (311) 1.64 Å.

Figures: `4_atoms_overlay.png`, `4_atoms_overlay_zoom.png` (24 Å crop).

**z-coordinates (reported separately):** the true atoms span z = 0.00–6.00 Å (mean 3.00 Å). No reconstruction here provides a credible z estimate: the QuantEM objects have two 3 Å slabs (the split between slabs is unconstrained, see section 2) and the S-matrix depths are at the edge of the search range (see section 2). Any improvement above is a better **projected** structure, not axial localisation.

## Loss curves

S-matrix final loss 7.12e-13 (one dataset, geometry check failed by ~25×, so a low loss does not mean S is determined).  QuantEM baseline: 18.4 → 14.7 over 60 it.

Figure: `5_loss_curves.png`.
