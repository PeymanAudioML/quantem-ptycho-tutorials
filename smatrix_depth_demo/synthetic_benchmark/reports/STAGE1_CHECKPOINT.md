# Stage 1 checkpoint: one noiseless specimen, before any reconstruction

Specimen `A_z100`: three Au atoms at z = 100 Å in a 200 Å vacuum slab, configuration `configs/benchmark_200A.yaml`
(version 1). No reconstruction has been run. Per the execution plan (section 17), the benchmark stops here: the
forward-model check found a model mismatch larger than the specimen's own signal.

Reproduce:

    cd smatrix_depth_demo/synthetic_benchmark
    python check_geometry.py                      # results/geometry/benchmark_200A.{json,md}
    python generate_data.py --specimens A_z100    # data/au_single/A_z100 (gitignored, ~26 s on 4 CPU cores)
    python validate_forward.py --specimen A_z100  # results/forward_validation/A_z100/
    python diagnostics/window_periodic_probe.py   # causes of the mismatch (numbers below)
    python diagnostics/quantem_registration.py
    python diagnostics/window_vs_aperture_edge.py

Software: Python 3.13.16, abTEM 1.0.10, ASE 3.29.0, PyTorch 2.14.1, NumPy 2.5.3, SciPy 1.18.1, zarr 3.4.1, QuantEM commit
0f60f9089269 (recorded per specimen in `ground_truth/truth.json`).

## 1. Microscope geometry (strict check, no override)

| quantity | value |
|---|---|
| energy, wavelength | 300 kV, 0.01969 Å |
| convergence semiangle | 20 mrad; depth resolution λ/α² = 49.2 Å |
| pixel, window, grid | 0.2 Å; K = 48 (9.6 Å); Y = 96 (19.2 Å, abTEM periodic cell) |
| scan | 25 × 25 at 0.4 Å = exactly 2 px; 9.6 Å field; every window inside the grid |
| defoci (nominal C10) | −166.7 / −100.0 / −33.3 Å; reference (gauge) = dataset 0 |
| true − nominal (C10, C12a, C12b) | dataset 1: +12, +6, −4 Å; dataset 2: −10, −3, +5 Å |
| patterns | 1875 (3 × 625), 48 × 48 px, reciprocal pixel 0.1042 Å⁻¹ |
| S-matrix beams | 293 |
| measurements per unknown | **0.800** (4.32 M measurements, 5.40 M unknown reals) |
| abTEM band limit | 32.8 mrad (> 1.3 α) |
| window vs probe cone | 9.6 Å ≥ 7.65 Å needed (nominal and true defoci) |
| memory | patterns 17 MB per variant; S + Adam 65 MB; forward gather 86 MB per 16 patterns |

All checks pass (`results/geometry/benchmark_200A.md`).

## 2. Ground truth generated and verified

* **Atoms:** 3 Au atoms (`calibrate_m3.gold_cluster`, seed 11, ≥ 2.5 Å apart) at z = 100 Å. Lateral centroid (9.84, 9.13) Å; field centre 9.6 Å. Atoms are kept inside [0.001, 199.999] Å.
* **Potential:** abTEM `Potential(gpts=(96, 96), slice_thickness=2.0)` with defaults (Lobato parametrisation, infinite projection): 100 slices of 2 Å, 0.2 Å sampling.
  * Units, checked directly: the transmission equals exp(iσV) to 0.0, so V is the projected potential per slice in V·Å, with σ = 6.526×10⁻⁴ rad/(V·Å).
  * Orientation and slicing: every Au column peaks in slice 50 (z = 100 Å) at the expected pixel (offset 0 px).
* **Probes:** true and nominal per dataset, real space (corner-centred, unit norm) and Fourier space, in `ground_truth/probe_d*.zarr`.
* **Datasets:** two detector variants × {noiseless, Poisson at 10⁵ e⁻}, in the `run_realization.py` layout. The metadata carries only the nominal defocus; the true aberrations are in `ground_truth/` only.
  * **window:** |FFT| of the K × K exit-wave window. Captures **97.8 %** of the exit intensity.
  * **pixelated:** abTEM `PixelatedDetector` (verified to equal |FFT|² × Y² within 8×10⁻⁷), integrated over 2 × 2 pixels. Zero frequency at pixel 23.75. Captures 99.95 %.

## 3. Forward-model consistency (truth only, nothing optimised)

L_data = Σ(√I_pred − √I_meas)² / Σ I_meas. **Scale of the specimen's signal:** vacuum vs the data gives L = 0.8–1.2×10⁻³. A forward model has to be well below that to be usable.

| model | L_data | verdict |
|---|---|---|
| F1 independent torch multislice, abTEM's 2/3 anti-aliasing | 3.5×10⁻⁵ | simulation verified |
| F1 without anti-aliasing | 2.8×10⁻⁴ | the anti-aliasing matters in the dark field |
| F2 S-matrix model, exact reference S, true probes | **1.7×10⁻²** (dark field 85 %) | **~15× the Au signal** |
| F3 QuantEM, true potential + true probe, def0 / 1 / 2 | 6.0 / 1.2 / 4.9 ×10⁻³ | barely below vacuum (6.4 / 1.5 / 5.4×10⁻³) |

### Diagnosis

* **Registration is not the cause.** QuantEM's loss is lowest at zero lateral shift of the truth object, and worse at ±1 px and at ±24 px.
* **The cause is the window model.**
  * Both the S-matrix model (beams on the 1/(K·dr) grid) and QuantEM (probe on a 48-pixel window) represent the probe as periodic with period K·dr = 9.6 Å.
  * The simulated probe has the hard aperture's slowly decaying tails: 2.8 % of its intensity lies beyond the window radius.
  * Restricting the true probe to the window's reciprocal grid reproduces the S-matrix misfit (L = 1.35–2.0×10⁻² vs F2's 1.7×10⁻²).
  * The residual sits on the bright-field disc edge. This is very likely the same thin ring that dominated the `realization_0` QuantEM forward check.

### Model error / Au signal

L of the periodic-window model divided by L of vacuum, against the true data. Measured on a 5 × 5 scan around the cluster in a 38.4 Å cell.

| aperture edge | K = 48 (9.6 Å) | K = 96 (19.2 Å) | intensity beyond K/2 (K = 48 / 96) |
|---|---|---|---|
| hard (as generated) | 5.0 | 2.45 | 2.8 % / 1.1 % |
| linear taper 1 mrad | 3.4 | 0.24 | 1.6 % / 0.17 % |
| linear taper 2 mrad | 1.3 | **0.03** | 0.9 % / 0.02 % |
| linear taper 4 mrad | **0.30** | 0.01 | 0.4 % / 0.01 % |

## 4. Decision needed before stage 1 continues

Reconstructing the current data would mostly measure this model error, not the methods. Options, all keeping the 0.4 Å step:

| option | geometry | model error / signal | cost | changes |
|---|---|---|---|---|
| **A (recommended)** | K = 96, Y = 192, scan 49 × 49 (19.2 Å), 2 mrad aperture taper | 0.03 | 7203 patterns, 1201 beams, ratio 0.750; S + Adam 1.1 GB, gather 1.4 GB per 16 patterns: GPU | taper in the simulated probe and in both models' probes |
| B | K = 48 as requested, 4 mrad taper | 0.30 | as now | same as A; residual misfit still 30 % of the Au signal |
| C | as requested (hard aperture, K = 48) | 5.0 | as now | none; report every result against this floor |

* **Taper:** a soft aperture edge is a physical probe choice made before generating data, not a change to the ground truth after the fact. It matches abTEM's and QuantEM's soft-aperture defaults; the `realization_0` generator also used `soft_aperture: true`.
* **Code changes A and B need:**
  * per-beam aperture amplitudes in `simple_joint.JointModel`, small and backwards-compatible;
  * QuantEM initialised from the same probe array.
* **Option A size:** at K = 96 the window cone check has ample margin. 47 × 47 positions would give ratio 0.719.
