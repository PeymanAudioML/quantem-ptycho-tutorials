# Synthetic 4D-STEM benchmark: S-matrix + probe recovery vs QuantEM multislice

Physically simulated (abTEM multislice through atomic potentials) three-defocus 4D-STEM data with complete ground
truth, and scripts that reconstruct with the joint S-matrix + probe fit (`simple_joint.py`) and QuantEM, then evaluate
probe, potential, slice locations and feature depths against the truth.

Ground truth is read only by `validate_forward.py`, `evaluate_reconstruction.py` and the explicitly labelled oracle
probe mode (`true_fixed`) of `run_benchmark.py quantem`. Feature detection and depth estimation never see it.

## Configurations

| file | geometry | status |
|---|---|---|
| `configs/benchmark_200A.yaml` (v1) | K = 48, Y = 96, 25 × 25 scan, hard aperture | kept on record. Its forward models miss the data by 5× the Au signal (`reports/STAGE1_CHECKPOINT.md`), so it is not used for reconstructions |
| `configs/benchmark_200A_K96.yaml` (v2, default) | K = 96, Y = 288, 53 × 53 scan at 0.4 Å, 2 mrad soft aperture edge, ratio 0.739 | primary benchmark |

Both use 300 kV, 20 mrad, a 200 Å thickness and 0.2 Å pixels. The three nominal defoci are C10 = −166.7 / −100 / −33.3 Å; the true aberrations are perturbed on datasets 1 and 2, and dataset 0 is the gauge reference. Specimens: A (one Au cluster), B (two in one column), C (three, stacked or separated), D (Au in amorphous carbon, plus carbon-only controls).

## Pipeline

    cd smatrix_depth_demo/synthetic_benchmark
    python check_geometry.py                                   # strict; no override
    python generate_data.py --specimens A_z100                 # data/<config>/<group>/<specimen>/ (gitignored)
    python validate_forward.py --specimen A_z100               # forward-model consistency (truth only)
    python run_benchmark.py smatrix --specimen A_z100 --device cuda
    python run_benchmark.py quantem --specimen A_z100 --device cuda                     # arms A-D
    python run_benchmark.py quantem --specimen A_z100 --arms A,D \
        --probes true_fixed,nominal_fixed,smatrix_fixed,nominal_trained,smatrix_trained --device cuda
    python evaluate_reconstruction.py smatrix --specimen A_z100
    python evaluate_reconstruction.py quantem --specimen A_z100

### Single-dataset recovery

To recover the object and probe using only the middle-focus dataset, without loading any S-matrix results or
ground-truth probes, run from this directory:

    ../../.venv/bin/python run_benchmark.py quantem --config configs/poly_si_3grain.yaml \
        --specimen P3_XY --single-dataset --dataset 1 --num-slices 25 --tag single25 --device cuda

This selects arm A (uniform slices, flat initial object) and a nominal probe that is refined after the object-only
warm-up. The 25 slices are explicitly chosen here, not estimated from depth peaks. Without `--num-slices`, the
count is `ceil(thickness / dz_max)` (12 with the default settings). Single-dataset runs have separate output names
ending in `_single`; their summaries record `smatrix_run: null`. `--probes nominal_fixed` keeps the probe fixed.

`smatrix --single-dataset --dataset 1` also selects only this dataset, fixes its nominal probe, and records the
original dataset index. Its output name ends in `_single_d1`. The existing strict geometry check still applies:
the current polycrystal geometry has approximately 0.246 intensity measurements per real S-matrix unknown with
one dataset, below the 0.7 threshold, so this S-matrix fit is rejected. A single dataset cannot independently
identify the probe and an unconstrained S-matrix in this model; fixing a nominal probe also makes refocused depth
dependent on that probe's calibration. The standalone QuantEM path does not require this S-matrix fit.
`--allow-underdetermined` (opt-in) fits it anyway, logging and saving the failed check. On P3_XY that fit kept the
lateral image but invented two depth layers (reports/QUANTEM_P3_XY.md, section 5).

### Layout of one specimen

    data/<config>/<group>/<specimen>/
        ground_truth/
            atoms.extxyz        per-atom feature id (-1 = carbon)
            potential.zarr      V·A per slice; phase = sigma·V
            probe_d<i>.zarr     true probe, real and Fourier space
            truth.json          features, nominal / true aberrations, reference, slicing, checks, versions, seeds, hashes
        <window|pixelated>_<noiseless|poisson>/def<i>/
            4dstem_quantem/     Dataset4dstem directory for run_realization.py / QuantEM; nominal metadata only
            probe.zarr, potential.zarr, atoms.extxyz   links to ground_truth/

Each run directory can also be passed straight to the realization pipeline:

    python ../run_realization.py --data <specimen>/window_noiseless/def0 .../def1 .../def2

## Conventions

* **Axes:** array axis 0 = ase x, axis 1 = ase y, and z is the depth below the entrance surface.
* **Defocus:** C10 < 0 means the focus lies |C10| below the entrance. QuantEM's defocus is −C10.
* **Intensities:** fraction of the incident electrons per detector pixel. Poisson counts have expectation fraction × 10⁵.
* **Detector variants:**
  * `window`: |FFT|² of the K × K exit-wave window, the S-matrix forward model;
  * `pixelated`: abTEM `PixelatedDetector` on the full exit wave, integrated over (Y/K)² pixels, with the zero frequency at a sub-pixel position.

See the docstring of `bench.py` for details.
