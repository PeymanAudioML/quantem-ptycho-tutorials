# realization_0 controlled QuantEM runs: findings

Data: `data/realization_0/4dstem_quantem_noisefree` (Si, 6 Å thick, one defocus 500 Å, 80 kV, 20 mrad, 37×37 scan, 4 Å step).
All runs reuse the saved S-matrix stage (`--reuse-smatrix --force --dr 0.4`), QuantEM baseline arm (uniform slices, flat
object, nominal probe), 200 iterations. Commands: `run_ctrl.sh`. Per-run ground-truth reports: `<run>/ground_truth_comparison/`.

| run | slices | probe | projected corr | probe end (overlap / RMS rad) | maxima→column median, ≤0.5 Å | final loss |
|---|---|---|---|---|---|---|
| rr_2s_fixed | 2 × 3 Å | fixed (`--probe-lr 0`) | 0.511 | 0.9986 / 0.000 | 0.20 Å, 98% | 14.71 |
| rr_2s_train | 2 × 3 Å | updated (parametric, lr 0.2) | 0.510 | 0.9963 / 0.068 | 0.19 Å, 97% | 14.70 |
| rr_8s_fixed | 8 × 0.75 Å | fixed | 0.479 | 0.9986 / 0.000 | 0.21 Å, 97% | 14.61 |
| (earlier, 60 it) results_real0_2slice | 2 × 3 Å | updated | 0.508 | 0.9963 / 0.069 | 0.19 Å, 97% | 14.70 |

* 60 → 200 iterations changes nothing (plateau).
* Updating the probe degrades it and does not improve the object; fixed probe is preferred here.
* 8 slices: phase piles into the first and last slab (0.008 / 0.009 rad vs ~0.002–0.003 between; truth ~0.010 in every
  slab), in-slab correlations ≈ 0: no depth recovered.

## Forward check (`FORWARD_CHECK.md`, `forward_check.py`)

QuantEM's own forward path, no optimisation. QuantEM loss: vacuum 16.0, reconstructions 14.5–14.7, reconstruction object +
truth probe 12.8–12.9, **truth object + truth probe 11.9** (8, 2 or 1 truth slabs: 11.86 / 11.89 / 11.96).

About 80 % of the truth floor sits in a thin ring just outside the bright-field disc (0.48–0.70 Å⁻¹, 2.45 % of the
intensity), where the truth fits no better than vacuum: the measured disc edge is broader than the model's (detector
4×4 block-sum of a 0.1 Å-sampled simulation vs QuantEM's point-sampled 40 Å window). Beyond the edge the truth object
explains the dark field to ~7–10 % (vacuum ~80–90 %). Registration checked: ±0.2 Å shifts and axis swap all fit worse.
Candidate fixes (not yet tried): QuantEM `vacuum_probe_intensity`, or `detector_mask` excluding the edge ring.

## Why the projected correlation is ~0.51 (rr_2s_fixed, Hann-windowed bands)

| k (Å⁻¹) | truth power | corr | recon/truth amplitude |
|---|---|---|---|
| < 0.10 | 0.9 % | 0.35 | 2.22 |
| 0.10–0.25 | 3.5 % | 0.62 | 1.07 |
| 0.25–0.50 | 39 % | 0.79 | 0.81 |
| 0.50–1.0 | 40 % | 0.68 | 0.99 |
| 1.0–2.5 | 15.5 % | 0.24 | 1.12 |

The Si lattice (0.25–1 Å⁻¹) is recovered well with about the right amplitude (0.97 overall). The correlation is pulled
down by the sub-Å atom cores (> 1 Å⁻¹) and spurious long-range phase (< 0.1 Å⁻¹). The fitted slope of ~0.5 in the
reports is regression dilution (slope ≈ corr × amplitude ratio), **not** a factor-2 scale error.

Units: `potential.zarr` slices are projected potential in V·Å (abTEM transmission exp(i σ V), σ = 1.0087e-3 rad/(V·Å) at
80 kV), so φ_true = σ ΣV_k as used.
