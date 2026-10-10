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

<!-- RESULTS -->
