# Joint S-matrix and probe-aberration reconstruction (PyTorch)

Technical report for the extension of `smatrix_depth_demo` with joint S-matrix / electron-probe optimisation
following Pelz *et al.*, *Phase-retrieval algorithm for the reconstruction of the scattering matrix from 4D-STEM data*,
PRR **3**, 023159 (2021) (Secs. II.B–F, Algorithm 1, Apps. A and D), with depth localisation evaluated in the style of
Terzoudis-Lumsden *et al.* (axial response, BF vs BF+DF).
Everything here is simulated data, CPU only (4 cores, no GPU).

The existing NumPy implementation (`smatrix_depth_demo.py`) and all earlier experiments are unchanged; the only edit
to it is an optional `S_init=None` argument of `reconstruct()` used by the regression tests (default behaviour
identical).

## 1. Files

| file | content |
|---|---|
| `probe_aberrations.py` | Eq. 23 aberration basis (C10, C12, C21, C23, C30; optional C32, C34), polar ↔ Cartesian conversion, `AberrationProbe` (`torch.nn.Module`, coefficients are an `nn.Parameter`) |
| `joint_smatrix_probe.py` | geometry, forward model (Eq. 9), losses (Eqs. 11/12), S update (Stage A), probe update (Stage B), Algorithm-1 schedule (`JointReconstructor`), generalised multislice simulator with arbitrary aberrations, ground-truth S, metrics |
| `run_joint_probe.py` | experiments (`exp`) and the full-size NumPy regression/timing benchmark (`bench`) |
| `make_joint_figs.py` | summary tables (`results_joint/summary_table.md/.json`) and figures (`results_joint/figures/`) |
| `tests/test_joint_probe.py` | 12 tests: Eq. 23, NumPy regression, gradient checks, smoke tests |
| `run_all_joint.sh`, `run_joint_followup.sh` | the exact experiment queue used for this report |
| `results_joint/` | per-run JSON (history, coefficients, metrics), depth sections, S-beam samples, figures, tables |

### Install / run

```bash
pip install torch numpy matplotlib pytest          # CPU wheels are enough; CUDA is used automatically if present
cd smatrix_depth_demo
python -m pytest -q tests                          # 12 tests, ~4 min on 1 CPU thread
python run_joint_probe.py bench                    # full-size PyTorch vs NumPy regression (results_joint/bench_regression.json)
python run_joint_probe.py exp --sample reduced_s2 --config B --variants fixed,joint,oracle --seeds 0,1,2,3,4
./run_all_joint.sh                                 # everything in this report (~6 h on 4 CPU cores)
python make_joint_figs.py                          # tables + figures
```

Options: `--sched '{"loss": "poisson"}'` (Eq. 11 instead of Eq. 12), `'{"probe_optimizer": "sgd"}'` (plain gradient
step as in Algorithm 1), `'{"det_mask": "bf"}'` (bright-field-only data term), `'{"reg_lambda": 1e-3}'` (quadratic
prior towards the nominal probe), `--trainable C10,C12a,C12b,C21a,C21b,C23a,C23b,C30`, `--dose`, `--threads`.
Device: `jp.pick_device("auto")` → CUDA, then MPS, then CPU (MPS untested: complex64 FFT support there depends on the
PyTorch version).

## 2. Equation mapping

| Pelz et al. | implementation | verified by |
|---|---|---|
| Eq. 9 forward model `A_{k,d} = F[Σ_b Ψ_{d,b} e^{-2πi h_b·ρ} [C_{k,d} S]_b]` | `AberrationProbe.illumination` (Ψ_{d,b}·phase ramp) + `gather_windows` (cropping operator C, K×K window) + `forward_chunk` (batched Σ_b, ortho FFT) | `test_torch_fixed_probe_matches_numpy`, `bench` |
| Eq. 11 Poisson likelihood | `chunk_loss(kind="poisson")` = Σ (N y − N I ln N y), N = dose·normalisation | probe-gradient test (Poisson variant) |
| Eq. 12 amplitude loss | `chunk_loss(kind="amplitude")` = Σ (\|Z\| − √I)² | gradient tests |
| Eqs. 13–14 joint objective over (S, Ψ) | `JointReconstructor` (both blocks minimise the same data term; optional `reg_lambda·‖θ‖²`) | smoke tests |
| Eqs. 15, 22, A3 (∂L/∂S*) | `s_update_sweep`: S ← S − η·conj(P)/\|P\|²·C*ᵀ F⁻¹(Z − √I Z/\|Z\|) | `test_s_gradients_autograd_vs_fd_and_adjoint` (autograd = finite differences = 2× the hand-written adjoint) |
| Eq. 20 Poisson residual | `s_update_sweep` with `loss="poisson"`: Z(1 − I/y) (y clamped at 1e-6·max) | — |
| Eqs. 17, 19, 21 (∂L/∂Ψ, chain rule to the aberration coefficients) | autograd through `AberrationProbe.beam_coefficients` with S detached (`probe_loss_backward`) | `test_probe_gradients_autograd_vs_finite_difference` (amplitude + Poisson, float64, all 8 coefficients) |
| Eq. 23 aberration function | `basis_function`, `AberrationProbe.chi` | `test_eq23_cartesian_equals_polar`, `test_c10_is_existing_defocus_phase`, `test_scale_gives_one_radian_at_aperture_edge` |
| Algorithm 1 (block coordinate descent) | `JointReconstructor.run` | see §3 |
| Sec. II.F gauge | reference dataset 0 fixed (`mask[ref] = 0`); vacuum start; S entrance-referenced; output-plane search in the S metric | config G experiment, §6 |

PyTorch convention: for a real loss of a complex tensor `S.grad = ∂L/∂Re S + i ∂L/∂Im S = 2 ∂L/∂S*`; the test checks
exactly this factor against the Eq. 22 adjoint.

### Eq. 23 verification

`χ(α, φ) = (2π/λ) Σ C_{n,m} α^{n+1} cos[m(φ − φ_{n,m})]/(n+1)`, `Ψ = A e^{−iχ}`, α = λ|k|, φ = atan2(k_y, k_x), all C in Å.

* The literal polar formula (`chi_eq23_polar`) and the Cartesian form used for optimisation
  (`C cos[m(φ−φ₀)] = C_a cos mφ + C_b sin mφ`) agree to 1e-12 for random coefficients of all orders (test).
* With C10 = Δf the probe phase is `exp(−iπλΔf k²)`, i.e. exactly `probe_in(df)` of the existing code (test), so
  the existing defocus convention (negative Δf = focus inside the sample) is preserved.
* The generalised simulator with C10-only coefficients reproduces the existing simulator **bitwise**
  (`test_simulator_reproduces_existing`, `np.array_equal`).
* Normalisation: the trainable parameter θ_j = ΔC_j / s_j with s_j = λ(n+1)/(2π α_max^{n+1}), so θ_j = 1 is a
  maximum phase change of 1 rad at the aperture edge (300 kV, 20 mrad: C10, C12: 15.7 Å; C21, C23: 1175 Å;
  C30: 7.8·10⁴ Å = 7.8 μm) — this puts very different orders on one scale for Adam and gradient clipping.

## 3. Algorithm 1 — what is implemented and how it differs

Pelz Algorithm 1: initialise S (vacuum) and Ψ (aperture + nominal aberrations); repeat {gradient step on S with Ψ
fixed; gradient step on Ψ with S fixed}.

Implemented schedule (`Schedule`, defaults used for the experiments in brackets):

1. **Warm-up:** S-only sweeps with the nominal probe [5].
2. **Alternating cycles** [8 cycles × (3 S sweeps + 2 probe steps)].
3. **Final refinement:** S sweeps with step × 0.3 [3] and probe steps with lr × 0.3 [1].

| item | Pelz Algorithm 1 | here | why |
|---|---|---|---|
| S step | full-batch gradient step, preconditioned | sequential minibatch sweep (chunks of 20 patterns, forward pass with the S at chunk start, then window-by-window update), preconditioner conj(P)/\|P\|², η = μ/(B·overlap) | identical to the existing NumPy / Brown et al. update, so the fixed-probe path is a strict regression of the existing code |
| probe step | gradient step on Ψ (or its aberrations) | Adam on the normalised Cartesian coefficients θ, gradient-norm clipping at 1, minibatch (random 34 % of patterns per step) | robustness to the very different curvature of each coefficient; `probe_optimizer="sgd"` gives the plain step |
| probe parameterisation | Ψ pixel-wise with optional aberration fit | aberration coefficients only (Eq. 23), amplitude = analytic top-hat | the task asks for an Eq. 23 parameterisation; Pelz initialise the amplitude from the mean intensity |
| gauge | Sec. II.F | dataset 0 fixed; only relative aberrations learned | common aberrations are unobservable (config G) |
| ADMM / regularisation (App. D) | ADMM for S with TV-type priors | **not implemented**; optional quadratic prior on θ only | out of scope for this pass |
| precision | — | complex64 for S and data, float64 for θ and the aberration phase | — |

Compute logging: every run records wall time per S sweep and probe step, the total runtime and the peak memory
(`torch.cuda.max_memory_allocated` on GPU, process max RSS on CPU). Seeds: noise realisation = `seed`, probe
minibatches from `torch.Generator().manual_seed(seed·100003 + it)`. No finite differences are used anywhere in the
optimisation (only in the tests).

## 4. Validation

`python -m pytest -q tests` → **12 passed** (241 s, 1 CPU thread; `results_joint/pytest_result.txt`).

| test | checks |
|---|---|
| `test_eq23_cartesian_equals_polar` | Cartesian Eq. 23 = literal polar Eq. 23 |
| `test_c10_is_existing_defocus_phase` | C10 term = existing `probe_in(df)` phase |
| `test_beams_match_numpy` | beam list identical to the NumPy code |
| `test_scale_gives_one_radian_at_aperture_edge` | coefficient normalisation |
| `test_simulator_reproduces_existing` | generalised simulator = existing simulator (bitwise) |
| `test_depth_sections_match_existing` | `depth_sections_general` = existing `depth_sections` |
| `test_torch_fixed_probe_matches_numpy` | PyTorch S update = NumPy `reconstruct` (rel. diff < 1e-4 after 3 iterations) |
| `test_probe_gradients_autograd_vs_finite_difference[amplitude/poisson]` | probe gradients vs central differences (float64), reference-dataset gradient exactly 0 |
| `test_s_gradients_autograd_vs_fd_and_adjoint` | S gradient vs finite differences (real and imaginary) and vs the Eq. 22 adjoint |
| `test_probe_only_recovers_relative_error_with_true_S` | with the true S and model-consistent data, probe-only optimisation recovers relative ΔC10/C12 to < 0.3 Å |
| `test_joint_smoke_runs_and_reduces_loss` | joint run reduces the loss |

Full-size regression on the existing 3-layer data (`run_joint_probe.py bench`, `results_joint/bench_regression.json`,
2 iterations, MU = 60):

| start | NumPy loss | PyTorch loss | rel. ‖S_torch − S_numpy‖ / ‖S_numpy − S₀‖ | time / iteration |
|---|---|---|---|---|
| perturbed S₀ | 0.0528756, 0.0371145 | 0.0528756, 0.0371145 | 2.1·10⁻⁵ | NumPy ≈ 12 s, PyTorch ≈ 8.7 s |
| vacuum S₀ | 0.05036, 0.03190 | 0.05034, 0.03186 | 0.98 (see below) | |

From a vacuum start most beams are exactly zero in many pixels, so Z/|Z| is undefined at |Z| ≈ 0 and float32 round-off
in the two FFT libraries picks different phases there; the losses agree to 4 digits but S differs in those
unconstrained phases. From a generic (perturbed) start the two implementations agree to float32 precision.

## 5. Experiments

All data are simulated with the generalised multislice simulator (`simulate_4dstem`, true aberrations per dataset),
300 kV, 20 mrad, three datasets at nominal defoci −200/−100/0 Å (dataset 0 = −200 Å is the reference), 64×64 detector,
Poisson noise at 2·10⁵ e⁻/pattern unless stated, 529 beams, vacuum S start. The schedule is identical for every run:
5 warm-up S sweeps, 8 cycles × (3 S + 2 probe), 3 final S + 1 probe (32 S sweeps), μ = 60, probe lr 0.3 (Adam),
clip 1, 34 % of patterns per probe step, trainable C10, C12a, C12b on datasets 1 and 2.

| variant | probe used for S | meaning |
|---|---|---|
| **fixed** | nominal (acquisition) aberrations, not refined | conventional reconstruction with a miscalibrated probe ("before refinement") |
| **joint** | nominal start, refined by Stage B | this work ("after refinement") |
| **oracle** | true aberrations, not refined | upper bound |

| config | true probe |
|---|---|
| A | = nominal (no error) |
| B | dataset 1: ΔC10 = +25 Å, \|C12\| = 15 Å at 30°; dataset 2: ΔC10 = −20 Å, \|C12\| = 12 Å at −50° (relative errors only; dataset 0 exact) |
| G | ΔC10 = +25 Å on **all** datasets including the reference (common error = gauge demo) |

| sample | grid / scan | measurement / unknown ratio O ¹ |
|---|---|---|
| `reduced_s4` | 3 random-atom layers (z = 30/150/270 Å, 300 Å slab), 128² grid, 13×13 scan, 0.8 Å step | 0.16 |
| `reduced_s2` | same sample, 26×26 scan, 0.4 Å step | 0.60 |
| `reduced_s1` | same sample, 52×52 scan, 0.2 Å step | 2.4 |
| `layers3` | existing 3-layer sample (ring/square/triangle), 192² grid, 25×25 scan, 0.8 Å step | 0.28 |
| `six_phi1p5` | existing 6 overlapping layers (25…275 Å, 1.5 rad), same geometry | 0.28 |
| `poly` | existing CaAl₂Si₂O₈ polycrystal (8 grains, 150 Å), same geometry | 0.28 |

¹ O = (3 datasets × positions × 64² detector pixels) / (2 × 529 beams × pixels of S touched by the windows). Pelz
et al. (Sec. III.A, Eq. 24) report that S-matrix retrieval needs O ≳ 4; none of the geometries reaches it, the
existing 0.8 Å-step data are an order of magnitude below.

Statistics: `reduced_s2/B`, `reduced_s4/B` have 5 noise realisations, `reduced_s2/A` 3, `B_bf` 2, everything else 1
(full-size runs take 6–8 min each, `reduced_s1` 25–35 min). Std over noise seeds is tiny at 2·10⁵ e⁻/pattern
(the reconstructions are limited by sampling and model, not by shot noise); single-seed numbers must therefore be
read with the systematic effects below in mind, not with a noise error bar. Complete table:
`results_joint/summary_table.md`; figures: `results_joint/figures/`.

### 5.1 Probe recovery (Experiment B)

| sample | O | fraction of injected relative aberration recovered ² | residual ΔC10 d1 / d2 (Å, injected +25 / −20) | probe-wavefunction error d1,d2: fixed → joint |
|---|---|---|---|---|
| reduced_s4 (n=5) | 0.16 | 0.15 ± 0.01 | −18.9 ± 0.1 / 18.3 ± 0.2 | 0.524 → 0.460 ± 0.006 |
| layers3 | 0.28 | 0.26 | −16.6 / 15.9 | 0.524 → 0.403 |
| six_phi1p5 | 0.28 | 0.26 | −17.2 / 15.4 | 0.524 → 0.400 |
| poly | 0.28 | 0.54 | −10.6 / 5.5 | 0.524 → 0.274 |
| reduced_s2 (n=5) | 0.60 | 0.83 ± 0.00 | −4.2 ± 0.1 / 3.1 ± 0.3 | 0.524 → 0.091 ± 0.003 |
| reduced_s1 | 2.4 | 0.85 | −4.1 / 3.1 | 0.524 → 0.078 |

² 1 − ‖C_joint − C_true‖ / ‖C_nominal − C_true‖ over (C10, C12a, C12b) of datasets 1, 2. The probe error is
‖Ψ_est e^{iφ} − Ψ_true‖ / ‖Ψ_true‖ over the 529 beam coefficients after optimal global phase φ.

Figures: `aberrations_truth_vs_recovered.png` (injected vs recovered, all B groups), `loss_coeffs_<group>.png`
(loss and coefficient trajectories), `probe_<group>.png` (aberration phase on the aperture: injected, recovered,
residual, and the real-space probe profile).

Findings:

* **The joint optimisation recovers relative aberrations only where the scan oversamples S enough.** At the existing
  0.8 Å scan step (O ≈ 0.2–0.3) it recovers 15–26 % of the injected error on the layered samples; the loss still
  decreases (fixed 0.00170 → joint 0.00155 on `layers3`) because S absorbs most of the probe error. Halving the step on
  the same sample (O = 0.6) raises recovery to 83 %, and the residual ΔC10 of ≈ 4 Å is then the same at O = 2.4.
  This is the oversampling dependence predicted by Pelz et al.; it is the dominant effect in this study.
* The polycrystal recovers more (54 %) at the same O as `layers3` — plausibly because its strong, structured
  diffraction (crystalline grains) constrains the probe better than sparse phase objects; with one seed this is an
  observation, not an established trend.
* The recovered probes in the O ≥ 0.6 cases are not exact: ≈ 4 Å ΔC10 and ≈ 1–3 Å C12 remain after 32 iterations, the
  trajectories (`loss_coeffs_reduced_s2_B.png`) are still drifting slowly at the end, and the joint loss stays above the
  oracle loss (0.00491 vs 0.00465). Longer schedules were not run.
* **Loss decrease is not evidence of a correct probe.** Config A (true probe = nominal) on `layers3`: the joint
  optimisation *moves away* from the true probe (ΔC10 +1.6 / +2.4 Å, probe error 0 → 0.061) while lowering the loss
  marginally (0.00129 → 0.00128). The cause is model mismatch: the S-matrix model uses discrete, window-periodic beams,
  the simulator a continuous hard aperture whose Airy tails are cut by the 64-pixel window; the optimiser trades probe
  error for model error. On the reduced sample the same bias is 0.2–0.4 Å. This bias (and the 0.8 Å-step result) is
  why the probe-recovery numbers above are quoted against ground truth and not inferred from the loss.

### 5.2 Gauge (config G)

A +25 Å defocus error on **all three** datasets (including the reference) is not recovered at all: joint C10 errors
−25.4 / −25.2 Å vs −25 / −25 Å for the fixed probe (fraction recovered −0.01), and the joint loss equals the fixed loss
(0.00640 vs 0.00640). The data are explained equally well by S with a shifted output plane: the S metric's best
output-plane match moves to z_out = −30 Å, and the depth sections shift by the common defocus (layers found at
39.8 / 180.0 / 285.1 Å instead of 24.9 / 150.1 / 275.2 Å for the oracle; mean depth error 18.3 Å vs 3.5 Å).
This is the Sec. II.F ambiguity: **a defocus (or any aberration) common to all datasets is not identifiable from the
intensities and maps directly into an absolute depth offset.** Absolute depth therefore depends on the calibration
of the reference dataset (or another physical constraint such as a known surface), not on the joint optimisation.

### 5.3 Depth localisation (Terzoudis-Lumsden-style axial response)

Depth sections every 10 Å (forward propagation of the entrance-referenced S by z, `depth_sections_general`); for each
true layer, the correlation of every section with that layer (axial response, `axial_<group>.png`), the found depth
(centroid of the plateau within 0.03 of the maximum), the axial FWHM of the response and the cross-talk (mean
\|r\| of the other layers at the true depth, relative to the layer itself).

| sample (config B) | depth error fixed / joint / oracle (Å) | axial FWHM fixed / joint / oracle (Å) | cross-talk fixed / joint / oracle |
|---|---|---|---|
| reduced_s4 (n=5) | 5.1 / 5.1 / 3.5 | 107 / 103 / 103 | 0.10 / 0.10 / 0.11 |
| reduced_s2 (n=5) | 5.1 / 3.4 / 3.4 | 107 / 107 / 104 | 0.10 / 0.11 / 0.11 |
| reduced_s1 | 5.1 / 3.5 / 3.5 | 107 / 107 / 103 | 0.10 / 0.11 / 0.11 |
| layers3 | 6.7 / 5.1 / 1.7 | 107 / 110 / 103 | 0.09 / 0.09 / 0.09 |
| six_phi1p5 | 9.2 / 9.2 / 10.9 | 125 / 125 / 120 | 0.42 / 0.41 / 0.38 |
| poly (slab metric ³) | 10.0 / 10.0 / 9.4 | – | selectivity 0.54 / 0.58 / 0.61 |

³ polycrystal: the true structure summed over ±20 Å around each of 16 depths (0–150 Å); depth error = mean distance
between a section and the slab it correlates best with (10 Å grid, so 10 Å = typically one grid step); selectivity =
mean \|r\| with the slab at the same depth minus mean \|r\| with slabs ≥ 60 Å away.

What the numbers support:

* **No improvement in depth resolution.** The axial FWHM (≈ 100–110 Å, set by λ/α² and the 3-defocus data) and the
  cross-talk are unchanged by probe refinement, and even the oracle probe does not change them. The joint
  optimisation does **not** improve depth resolution in these experiments.
* **Depth placement:** the relative probe errors of config B shift individual layer positions by one plateau step
  (e.g. `reduced_s2`, z = 30 Å layer found at 19.9 Å with the miscalibrated probe, 24.8 Å with the joint or oracle
  probe; other layers unchanged). Where the probe is recovered (O ≥ 0.6) the joint result matches the oracle; at the
  0.8 Å step it does not (`reduced_s4`: 5.1 vs 3.5 Å). These differences are ≤ 5 Å on a 10 Å section grid, i.e.
  they are real (reproduced across all 5 noise seeds, std < 0.01 Å) but small.
* On the 6-layer sample, the oracle probe is not better than the miscalibrated one (10.9 vs 9.2 Å, one seed): with
  overlapping 50-Å-spaced layers and a ≈ 120 Å axial response, depth placement is limited by the depth resolution
  itself, not by the probe.
* The polycrystal shows a small, consistent ordering fixed < joint < oracle in diagonal correlation (0.70 / 0.72 / 0.74)
  and selectivity (0.54 / 0.58 / 0.61) — one seed, so indicative only.
* The largest depth effect seen anywhere is the **common** (unidentifiable) defocus error of config G (§5.2), which
  joint optimisation cannot correct.

### 5.4 Bright-field only vs bright-field + dark-field

`reduced_s2/B_bf`: data term and S residual restricted to the bright-field disk (`det_mask="bf"`), n = 2.

| | BF + DF (n=5) | BF only (n=2) |
|---|---|---|
| joint: fraction recovered | 0.83 ± 0.00 | 0.84 ± 0.01 |
| joint: depth error (Å) | 3.4 | 3.4 |
| oracle: depth error / axial FWHM (Å) | 3.4 / 104 | 3.5 / 103 |
| oracle: S NRMSE (scanned region) | 0.258 | 0.228 |

For this weak-phase, sparse sample, discarding the dark field changes neither probe recovery nor the axial response;
the S NRMSE over the scanned region is even somewhat lower with BF only (0.228 vs 0.258; the cause was not
investigated). Terzoudis-Lumsden et al. argue
that dark-field scattering carries additional depth information for strong scatterers; this sample does not test that
regime, so **no conclusion about the value of DF for depth sectioning should be drawn from this experiment**.

<!-- DOSE -->

### 5.6 Experiment A (known probe) and runtime / memory

| | runtime (s) | peak memory (MB) ⁴ |
|---|---|---|
| reduced_s2 fixed / joint (n=5) | 318 ± 81 / 363 ± 21 | 1463 / 1537 |
| reduced_s1 fixed / joint | 1472 / 2030 | 1360 / 1691 |
| full size (192², 625 positions) fixed / joint | 340–353 / 412–463 | 1220–1242 / 2377–2406 |

Probe refinement adds ≈ 15–40 % runtime (17 probe steps with backward passes over 34 % of the data each).
Config A on `reduced_s2` (n=3): joint and fixed give identical loss (0.00531), depth error (3.5 Å) and S error (0.259);
the joint coefficients stay within 0.2–0.4 Å of the truth. On `layers3` A see §5.1 (model-mismatch drift).

⁴ CPU: process maximum RSS (`ru_maxrss`), which is monotone over the process lifetime; runs of one invocation are
executed in the order fixed → joint → oracle, so for the joint and oracle runs the number is an upper bound that
includes the earlier runs. The full-size joint runs need roughly 2.4 GB (autograd buffers of the probe minibatches);
fixed-probe runs ≈ 1.2 GB. On CUDA `torch.cuda.max_memory_allocated` is recorded instead.

## 6. Gauge ambiguities

1. **Common probe phase ↔ S (Pelz Sec. II.F):** Ψ_{d,b} S_b only enters as a product, so any phase pattern shared by
   all datasets is absorbed by S. Fixed by keeping dataset 0 at its nominal aberrations; only *relative* aberrations
   are learned and reported. Demonstrated in config G.
2. **Output plane of S:** intensities are invariant to free-space propagation of the exit wave, so S is defined up to
   its output plane. The reconstructions start from vacuum (S = identity at the entrance plane), sections are made by
   forward propagation from the entrance surface, and the S error metric searches z_out ∈ [−60, 60] Å.
3. **Per-beam constant phases of S** (unobservable global phase per beam when only intensities are measured): removed
   before computing S errors and in the S-beam figures.
4. Common defocus between the reference and the true probe → absolute depth offset (config G). Not removable without
   additional calibration (e.g. a known surface or an independently calibrated reference).

## 7. Deviations from the paper and limitations

* S update = sequential minibatch amplitude flow (Brown et al. / existing NumPy code) with the |Ψ|⁻² preconditioner,
  not a single full-batch step; probe update = Adam on normalised Cartesian aberration coefficients with clipping,
  not a plain gradient step on Ψ (SGD available).
* Probe amplitude is the analytic top-hat with the existing normalisation; Pelz et al. also refine/initialise the
  amplitude from the data. Partial coherence, source size and detector MTF are not modelled.
* ADMM and the regularisers of App. D are not implemented (only an optional quadratic prior on the probe corrections,
  unused in the reported runs).
* Only C10 and C12 were refined in the reported experiments. C21, C23 and C30 are implemented and gradient-checked,
  but were not trained: higher orders couple weakly to the 20-mrad, 3-defocus data, and a common C30/C21 is not
  identifiable at all (gauge). **No claim is made that arbitrary or common aberrations can be identified.**
* The full-size samples (`layers3`, `six_phi1p5`, `poly`) are run at the existing 0.8 Å scan step (O ≈ 0.28);
  Nyquist-sampled full-size data (0.2 Å step, 16× more patterns, ≈ 2 h per run) were not attempted. Probe-recovery
  numbers for those samples are therefore limited by sampling, not by the method.
* Model mismatch (discrete window-periodic beams vs a continuous aperture) biases the recovered aberrations by a few Å;
  model-consistent data (`simulate_from_smatrix`) remove it (probe-only test < 0.3 Å).
* Poisson loss (Eq. 11) and the Eq. 20 residual are implemented and gradient-checked, but all reported runs use the
  amplitude loss (Eq. 12); the Poisson S residual uses a clamped denominator.
* MPS is supported by device selection only and untested; no GPU was available, all timings are 4-core CPU.
* Single seed for the full-size and polycrystal runs (cost); multi-seed statistics only on the reduced sample.
* No result of Pelz et al. or Terzoudis-Lumsden et al. is claimed to be numerically reproduced: the experiments use
  my own simulated samples, and the comparison is methodological (the same algorithm and the same type of axial
  analysis), not a reproduction of their figures. The analytic pseudo-CTF of Terzoudis-Lumsden (their Figs. 2/7) was
  not computed.

## 8. Summary

* The joint S / probe reconstruction (Pelz Algorithm 1 with an Eq. 23 aberration probe) is implemented in PyTorch,
  regression-tested against the existing NumPy reconstruction (agreement to float32 precision from a generic start)
  and gradient-checked (autograd = finite differences = Eq. 22 adjoint).
* Relative defocus/astigmatism errors between datasets are recovered to ≈ 85 % (residual ≈ 4 Å ΔC10) when the scan
  step is ≤ 0.4 Å, and only to 15–26 % at the existing 0.8 Å step, where S absorbs the probe error.
* A common aberration of all datasets is not identifiable and appears as an absolute depth offset.
* Probe refinement corrects layer *placement* by up to one 5 Å plateau step where the probe is recovered; it does not
  change the axial resolution (FWHM ≈ 105 Å) or the cross-talk. No improvement in depth resolution is claimed.
