# S-matrix + probe joint recovery and refocusing: files, input data, how to run

This covers only the S-matrix stage: fit the S-matrix and the probe aberrations jointly from multi-defocus 4D-STEM data,
then refocus the S-matrix to depth sections. QuantEM, plotting and evaluation are not needed for it.

**Quick start:** to measure how much each extra defocus dataset improves the recovery, do sections 0 and 7.
- Section 0 gets the code.
- Section 7 is one command: you point it at your data folder and it does the rest.

## 0. Get the code and set up Python

The code is on GitHub (public): **https://github.com/PeymanAudioML/quantem-ptycho-tutorials**, branch
**`claude/lucid-euler-6ccwsp`**. It is not on `main` yet.

### Download

```bash
git clone https://github.com/PeymanAudioML/quantem-ptycho-tutorials.git
cd quantem-ptycho-tutorials
git checkout claude/lucid-euler-6ccwsp
```

### Update later (pull the newest code)

```bash
cd quantem-ptycho-tutorials
git checkout claude/lucid-euler-6ccwsp
git pull origin claude/lucid-euler-6ccwsp
```

### Python environment

This needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and Python 3.11.

```bash
cd quantem-ptycho-tutorials
bash setup_env.sh          # creates .venv: quantem, torch (CUDA 12.8 wheels on Linux), numpy, scikit-image, matplotlib
uv pip install --python .venv/bin/python abtem==1.0.10 ase==3.29.0 pyyaml zarr scipy
```

- **Where the commands run:** every command below runs from `quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark/`.
- **Which Python:** they use `../../.venv/bin/python`. Any Python with these packages works.
- **GPU:** an NVIDIA GPU is strongly recommended. On the CPU, pass `--device cpu`; it is much slower.

## 1. Code files (8)

Paths relative to `smatrix_depth_demo/`.

| file | role | used from it |
|---|---|---|
| `synthetic_benchmark/run_benchmark.py` | **entry point** (`smatrix` stage) | `main()`, `smatrix()`, `load_variant()`, `fit_subset()`, `make_monitor()` |
| `synthetic_benchmark/bench.py` | config, paths | `make_setup()`, `specimen_dir()`, `versions()` |
| `synthetic_benchmark/configs/poly_si_3grain.yaml` | settings (example config) | microscope, sampling, defocus, specimen → data folder |
| `simple_joint.py` | **the method** | `Setup`, `JointModel`, `prepare_data`, `fit`, `refocus`, `detect_features`, `feature_response`, `check_geometry` |
| `run_realization.py` | reads the data | `load_dataset`, `centre_patterns`, `parabolic`, `local_peaks` |
| `quantem_export.py` | imported by `run_realization.py` | nothing (import only) |
| `abtem_data.py`, `calibrate_m3.py` | imported by `bench.py` | nothing (import only) |

Python packages: numpy, torch, zarr (v3), scipy, pyyaml. A GPU is optional but recommended: the 3-dataset P3_XYZ fit
takes 3.5 h on a T4.

### Call path

```
run_benchmark.py main()                         stage "smatrix"
└─ smatrix(cfg, a)                              run_benchmark.py:85
   ├─ load_variant()                            reads <data dir>/<variant>_<noise>/def0, def1, ...
   │    └─ run_realization.load_dataset / centre_patterns
   ├─ sj.check_geometry()                       enough measurements per unknown?
   ├─ sj.prepare_data()                         intensities → amplitudes, positions, dataset index
   ├─ sj.JointModel()                           unknowns: S (vacuum start) + aberration corrections per dataset
   ├─ fit_subset() → sj.fit()                   joint S + probe optimisation (Adam, amplitude loss)
   ├─ sj.refocus(S, setup, depths)              depth sections −0.2T … 1.2T
   ├─ sj.detect_features / feature_response     feature depth picks, focus curve
   └─ save outputs (section 5)
```

### What the model does

- **Forward model** (`JointModel.forward`):
  - cut a K×K window of S at each scan position
  - weight each beam by the probe of that pattern's dataset, ψ_d(k) = A(k)·exp(−iχ_d(k))
  - sum the beams, FFT
- **Loss:** Σ(|Z| − √I)².
- **Probe unknowns:** C10, C12a and C12b per dataset (`--trainable`).
  - The reference dataset (config `defocus.reference`) stays at its nominal values and fixes the gauge.
  - The first `--warmup` epochs fit S only. After that, S and the probe are updated together.
- **Refocusing** (`simple_joint.refocus`):
  - each beam is demodulated, Fresnel-propagated by z (measured from the entrance surface), and all beams are summed
  - the result is cropped to the scanned field

## 2. Input data: what is needed for multiple defocus

**One dataset per defocus**, same specimen, same scan, same microscope.

| requirement | why |
|---|---|
| **≥ 2 datasets at different defocus** (3 used here: nominal C10 = −166.7, −100, −33.3 Å for a 200 Å specimen, i.e. focus near bottom / middle / top) | with one dataset the probe phase and S are not separable (gauge ambiguity: a defocus error becomes a depth shift). Different defoci make the probe identifiable. |
| **same scan positions in every dataset** | positions are read from the last dataset and used for all |
| **scan positions on the S-grid**: multiples of `dr` (0.2 Å here) | checked in `load_variant`; raises "scan positions are not on the S-grid" |
| **detector = K×K pixels with dk = 1/(K·dr)** (96 px, 0.05208 Å⁻¹ here) | the window model needs it; checked in `load_variant` |
| **enough data**: measurements / unknowns ≥ 0.7 | `check_geometry` stops the run otherwise (`--allow-underdetermined` overrides) |
| nominal defocus of each dataset in its metadata | starting probe; the fit refines C10/C12 of the non-reference datasets |
| one dataset designated as reference, with exactly known aberrations | config `defocus.reference` (dataset 0 here) |

### Per-dataset file format

Each dataset is a QuantEM `Dataset4dstem` saved as zarr (v3).

```
def<d>/4dstem_quantem/
  _array/       float32 (Ny, Nx, K, K)   diffraction intensities, e.g. (53, 53, 96, 96)
  _sampling/    [step_y, step_x, dk_y, dk_x] = [0.4, 0.4, 0.05208, 0.05208]   (Å, Å, 1/Å, 1/Å)
  _metadata/    attributes and small arrays (any sub-group; looked up by name):
     energy_eV               300000.0
     semiangle_cutoff_mrad   20.0
     defocus_A               nominal defocus = −C10, e.g. 100.0 for C10 = −100 Å
     aperture_taper_mrad     2.0 (soft aperture edge; optional)
     padded_scan_positions_A (Ny, Nx, 2) scan positions in Å (or physical_scan_positions_A)
     zero_frequency_pixel    (2,) pixel of k = 0 (optional; default K/2)
```

- **Intensities:** the code normalises them itself, so any linear scale works.
- **Data variant:** the `window` variant matches the S-matrix forward model exactly: |FFT|² of the K×K exit-wave window.
  - `pixelated` and `poisson` variants also load.
  - The run name records the variant and noise, e.g. `smatrix_window_noiseless_...`.
- **Required:** energy, semiangle and defocus are required. Missing ones raise an error.

## 3. Where to put the input data

The data folder is built from the config and command line:

```
synthetic_benchmark/data/<config name>/<specimen group>/<specimen>/<variant>_<noise>/def<d>/4dstem_quantem
```

| piece | where it is set | P3_XYZ example |
|---|---|---|
| `<config name>` | config `name:` | `poly_si_3grain` |
| `<specimen group>` | config `specimens: <specimen>: group:` | `poly_si_3grain_xyz` |
| `<specimen>` | `--specimen` | `P3_XYZ` |
| `<variant>_<noise>` | `--variant` (default `window`), `--noise` (default `noiseless`) | `window_noiseless` |
| `def<d>` | d = 0 … (number of entries in config `defocus.nominal_C10`) − 1 | `def0`, `def1`, `def2` |

So the P3_XYZ run reads:

```
synthetic_benchmark/data/poly_si_3grain/poly_si_3grain_xyz/P3_XYZ/window_noiseless/def0/4dstem_quantem
                                                                                 /def1/4dstem_quantem
                                                                                 /def2/4dstem_quantem
```

- **`data/` is in `.gitignore`**, so it is not on GitHub. Regenerate it with `generate_data.py` (abTEM) or copy it over.

### To use your own data

1. **Copy the config** (e.g. `configs/my_sample.yaml`) and set:
   - `name:`
   - `microscope:`: `energy_ev`, `alpha_mrad`, `aperture_taper_mrad`
   - `sampling:`
     - `dr_A`: S-matrix pixel; must divide the scan step
     - `K`: detector size; must satisfy dk = 1/(K·dr)
   - `specimen: thickness_A`: sets the refocus range
   - `defocus:`
     - `nominal_C10`: one entry per dataset; this sets how many datasets are read
     - `reference`: the dataset whose aberrations are trusted
   - `specimens: <MyName>: group: <folder>`
2. **Put each defocus dataset** in `data/<name>/<folder>/<MyName>/window_noiseless/def<d>/4dstem_quantem`, in the format
   of section 2. You can use any `<variant>_<noise>` folder name from the allowed choices and pass the matching flags.

## 4. How to run

From `smatrix_depth_demo/synthetic_benchmark/`:

```bash
# joint S + probe from all defocus datasets, then refocus (the P3_XYZ "full80" run)
../../.venv/bin/python run_benchmark.py smatrix --config configs/poly_si_3grain.yaml --specimen P3_XYZ \
    --epochs 80 --heldout 0.0 --tag full80 --device cuda
```

### Useful options

| option | default | meaning |
|---|---|---|
| `--epochs`, `--warmup` | | epochs in total, and S-only epochs at the start (8 used) |
| `--batch`, `--lr-S`, `--lr-probe` | 16, 0.02, 0.005 | minibatch size, learning rates for S and the probe |
| `--trainable` | `C10,C12a,C12b` | aberrations fitted on the non-reference datasets |
| `--heldout` | | fraction of scan positions kept out of the fit (0.0 = use all) |
| `--dz` | | refocus step in Å (2 used) |
| `--monitor-every N` | | save a snapshot every N epochs (plot with `plot_fit_monitor.py`) |
| `--single-dataset --dataset 1` | | use one dataset only; the probe is fixed at nominal |
| `--single-probe-mode joint` | | (with `--single-dataset`) update the probe too. Experimental: not identifiable. |

## 5. Outputs

`runs/<config name>/<specimen>/smatrix_<variant>_<noise>_s<seed>_<tag>/`:

| file | content |
|---|---|
| `S.npy` | fitted S-matrix, (beams, Y, Y) complex. P3_XYZ: 1313 × 200 × 200 |
| `refocused_phase.npy` | phase of the refocused sections, (depths, field, field), cropped to the scanned field |
| `refocused_amplitude.npy` | amplitude of the same sections |
| `arrays.npz` | depth grid, focus curve, loss history, probe on the beams |
| `summary.json` | settings, loss, **fitted aberrations per dataset** (`coefficients`, order in `names`), feature depths |

`summary.json` and `arrays.npz` are also copied to `results/smatrix/<config name>/<specimen>/<run>/`.

Refocusing again later, e.g. on a finer depth grid (run from `synthetic_benchmark/`; tested, reproduces
`refocused_phase.npy` exactly):

```python
import dataclasses, sys
import numpy as np
sys.path.insert(0, "..")
import bench, simple_joint as sj

cfg = bench.load_config("configs/poly_si_3grain.yaml")
S = np.load("runs/poly_si_3grain/P3_XYZ/smatrix_window_noiseless_s0_full80/S.npy")
setup = dataclasses.replace(bench.make_setup(cfg), Y=S.shape[-1])  # S grid = scan extent + window, not config Y
vol = sj.refocus(S, setup, np.arange(0, 201, 1.0), device="cuda")    # (201, Y, Y) complex
c0, c1 = setup.K // 2, setup.Y - setup.K // 2 + 1                    # crop to the scanned field
phase = np.angle(vol / vol.mean(axis=(1, 2), keepdims=True))[:, c0:c1, c0:c1]
```

## 6. Result for P3_XYZ (3 defocus, full80)

| dataset | fitted C10 / C12a / C12b (Å) | true C10 / C12a / C12b (Å) |
|---|---|---|
| 1 | −87.79 / 6.06 / −4.13 | −88.0 / 6 / −4 |
| 2 | −43.31 / −2.95 / 4.85 | −43.3 / −3 / 5 |

- **Reference:** dataset 0 stays at −166.7 Å.
- **One dataset instead:** the probe stays at the nominal −100 Å, and that 12 Å error moves into S as a depth shift.
- **Full comparison:** `regularized_experiment/results/p3xyz_3defocus/README.md`.

## 7. Defocus study: 1 vs 2 vs 3 vs … defocus datasets, compared with the truth

Script: [`defocus_study.py`](defocus_study.py). **You only give the folder that holds your defocus datasets.** It does
the rest:
- reads the microscope settings from the data
- writes the configs
- links the data into place
- runs one S-matrix + probe fit per number of datasets
- refocuses each fit
- compares each fit with the ground truth
- writes one report

### Your data folder

```
/path/to/my_sample/                 <- give this to --data
    def0/4dstem_quantem/            dataset at defocus 0 (format: section 2), or def0/ directly holding _array, ...
    def1/4dstem_quantem/
    def2/4dstem_quantem/
    def3/...                        any number of datasets
/path/to/my_sample/../ground_truth/ optional; used for the comparison only (found automatically, or --truth DIR)
```

- **Settings:** all datasets must share energy, aperture, detector and scan; only the defocus differs. The script
  checks this.
- **Metadata:** energy, semiangle, aperture taper, defocus, dk and scan positions are read from each dataset's metadata
  (section 2). Use `--energy`, `--alpha`, `--taper` or `--thickness` to override or add missing values.
- **Ground truth folder** (as written by `generate_data.py`):
  - `truth.json`: thickness, potential slicing, `aberrations.true`/`nominal` per dataset
  - `potential.zarr`: true potential per 2 Å slice
  - `probe_d<d>.zarr`: true probe per dataset
  - `atoms.extxyz`
  - `polycrystal.json`: optional; adds the 3D grain map
- **Without truth:** the study still runs. It reports the fitted aberrations and losses only.

### Run

```bash
cd quantem-ptycho-tutorials/smatrix_depth_demo/synthetic_benchmark
../../.venv/bin/python defocus_study.py --data /path/to/my_sample --dry-run    # check the plan (no fitting)
nohup ../../.venv/bin/python defocus_study.py --data /path/to/my_sample > study.out 2>&1 &
```

For the P3_XYZ benchmark data of this repo. The data is not on GitHub, so first simulate it with abTEM (a GPU is used
if available):

```bash
../../.venv/bin/python generate_data.py --config configs/poly_si_3grain.yaml --specimens P3_XYZ
../../.venv/bin/python defocus_study.py --data data/poly_si_3grain/poly_si_3grain_xyz/P3_XYZ/window_noiseless
```

### Which subsets are fitted

- **Default:** nested subsets, starting from the reference dataset (`--reference`, default 0). Each step adds the
  dataset whose focus is farthest from those already used.
- **P3_XYZ example:** foci 166.7 / 100 / 33.3 Å give `[def0]`, `[def0, def2]`, `[def0, def2, def1]`.
- **The probe:**
  - The reference dataset's probe is never fitted. It is fixed at its nominal value, which sets the gauge.
  - With one dataset, the probe is therefore exactly the reference's nominal probe.
  - With 2 or more datasets, C10/C12a/C12b of the others are fitted together with S.
- **Your own subsets:** `--subsets "1;0,1;0,1,2"`. This example starts from def1 alone: one defocus whose probe is not
  known exactly, which is the realistic single-focus case. `--max-k 2` stops after two datasets.
- **Restarting:** fits that already exist are skipped, and stopped fits resume from their checkpoint. Re-running the
  same command continues the study. Use `--refit` to start over, or `--report-only` to rebuild the report.

### Run time

- **Scaling:** time grows with the number of patterns. On a T4 GPU at 80 epochs, one 53×53 dataset takes about
  1.2 h, 2 datasets about 2.3 h, 3 about 3.5 h.
- **Total:** the default 1 + 2 + 3 study takes about 7 h.
- **Faster check:** `--epochs 20`.

### Output: `results/defocus_study/<name>/`

`<name>` defaults to `<parent folder>_<data folder>`, or set it with `--name`.

| file | content |
|---|---|
| `summary.md` | tables: fitted vs true aberrations per subset; recovery vs truth metrics |
| `metrics_vs_defocus.png` | each metric vs number of defocus datasets |
| `sections_vs_truth.png` | refocused S-matrix phase sections per subset, with the true ±10 Å slabs |
| `depth_confusion.png` | correlation of every S-matrix section with every true slab (a sharp diagonal means right depth) |
| `grains/<subset>/` | 3D grain maps (only with `polycrystal.json`) |
| `summary.json`, `configs/`, `logs/` | all numbers, the generated configs, the fit logs |

The fits themselves (`S.npy`, `refocused_phase.npy`, …) are in `runs/defocus_study_<name>/<subset>/`. Linked data is in
`data/defocus_study_<name>/`; it holds links only, with no copies.

### Metrics (truth used only here, never in the fits)

| metric | meaning | better |
|---|---|---|
| probe RMS phase error | fitted vs true probe phase inside the aperture (piston and tilt removed), worst dataset | lower |
| fitted vs true C10 / C12 | per dataset, in `summary.md` | closer |
| slab corr | refocused phase at z = 10, 30, … Å vs the true potential of the ±10 Å slab at z | higher |
| selectivity | slab corr − correlation with the whole projection: depth information beyond the projection | higher |
| depth hits, mean depth error | for each section, is the best-matching true slab the right one; how far off (Å) | more, lower |
| focus curve vs truth | S-matrix focus curve vs the true depth profile | higher |
| projected corr | mid-depth phase vs the true projected potential | higher |
| grain accuracy | 3D grain map vs the true grains (polycrystals only) | higher |
