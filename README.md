# quantEM ptychography tutorials

Tutorial notebooks for 4D-STEM phase retrieval (direct and iterative ptychography) with
[quantEM](https://github.com/electronmicroscopy/quantem).

The notebooks come from
[electronmicroscopy/quantem-tutorials](https://github.com/electronmicroscopy/quantem-tutorials/tree/main/tutorials/diffractive_imaging)
(MIT license, see `LICENSE`), packaged here with a standalone `uv` environment.

## Setup

```bash
bash setup_env.sh                 # creates .venv and the "Python (quantem-tutorials)" Jupyter kernel
source .venv/bin/activate
```

quantEM is installed from the `main` branch of its GitHub repo; the exact commit is pinned in `uv.lock`.

## Data

Datasets (`*.zip`) are not tracked. Most notebooks load from `../../data/`:

- `white-noise-object_defocus+stig.zip`: generate with `simulation_notebooks/white-noise-object.ipynb` (numpy only, seconds).
- `ducky_*`, `STO_*`, `apoF_*`: generate with the other `simulation_notebooks/` (require abTEM).
- `gold_ptycho.zip` (used by `ptycho_experimental_workflow.ipynb`): see the link in that notebook.
- MOSS-6 ([Zenodo 13958144](https://zenodo.org/records/13958144)) and the multislice dataset: see links in `ptycho_iter_04_MOSS6.ipynb` / `ptycho_iter_05_multislice.ipynb`.

## Suggested order

1. Direct methods: `white_noise_object_phase_retrieval`, `ducky_dataset_phase_retrieval`, `crystalline_sample_phase_retrieval`, `hyperparameter_optimization`
2. Iterative: `ptycho_iter_01_lite` → `02_full` → `03_constraints` → `06_hyperparameters`
3. Real data and scaling: `ptycho_experimental_workflow`, `ptycho_iter_04_MOSS6`, `ptycho_iter_05_multislice`, `ptycho_iter_08_multi_gpu`, `hpc/`
