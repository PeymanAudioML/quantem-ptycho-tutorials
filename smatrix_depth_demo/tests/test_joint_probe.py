"""Tests for the joint S-matrix / probe reconstruction.  Run:  cd smatrix_depth_demo && python -m pytest -q tests

Small geometries keep everything on CPU in well under a few minutes.  The NumPy reference module
(smatrix_depth_demo) reads its geometry from module globals, which are patched to the same small geometry.
"""
import math, os, sys
import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import smatrix_depth_demo as m                      # noqa: E402
import probe_aberrations as pa                      # noqa: E402
import joint_smatrix_probe as jp                    # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))


@pytest.fixture
def small(monkeypatch):
    """Small geometry applied consistently to the NumPy reference module and to the new code."""
    g = jp.Geometry(K=16, M=2, dr=0.25, nscan=5, step=2)
    for k, v in dict(DR=g.dr, K=g.K, M=g.M, Y=g.Y, X=g.Y, LW=g.LW, STEP=g.step, NSCAN=g.nscan,
                     DEFOCI=[-40.0, -20.0, 0.0], LAYER_Z=[5.0, 25.0, 45.0], THICK=50.0, DOSE=2e5, MU=4.0).items():
        monkeypatch.setattr(m, k, v)
    return g


def random_layers(g, seed=0, n_atoms=4):
    rng = np.random.default_rng(seed)
    c = g.Y * g.dr / 2
    out = []
    for _ in range(3):
        pos = c + rng.uniform(-1.5, 1.5, (n_atoms, 2))
        out.append(m.atoms_to_phase([tuple(p) for p in pos], [0.8] * n_atoms))
    return out


# ------------------------------------------------------------------------- probe model (Eq. 23)
def test_eq23_cartesian_equals_polar():
    rng = np.random.default_rng(1)
    lam = 1 / jp.wavev(3e5)
    ky, kx = rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200)
    terms = {(1, 0): (37.0, 0.0), (1, 2): (12.0, 0.4), (2, 1): (300.0, -1.1), (2, 3): (150.0, 2.0), (3, 0): (2e4, 0.0)}
    cart = pa.polar_to_cartesian(terms)
    chi_c = sum(v * pa.basis_function(n, ky, kx, lam) for n, v in cart.items())
    np.testing.assert_allclose(chi_c, pa.chi_eq23_polar(ky, kx, lam, terms), rtol=1e-12, atol=1e-12)
    back = pa.cartesian_to_polar(cart)
    for key, (C, phi) in terms.items():
        assert abs(back[key][0] - C) < 1e-9
        mm = key[1]
        if mm:
            assert abs(math.remainder(mm * (back[key][1] - phi), 2 * math.pi)) < 1e-9


def test_c10_is_existing_defocus_phase(small):
    g = small
    for df in (-40.0, 0.0, 17.0):
        A = pa.probe_on_grid([df], ["C10"], g.Y, g.Y, g.dr, g.lam, g.alpha)
        qy = np.fft.fftfreq(g.Y, g.dr)[:, None]; qx = np.fft.fftfreq(g.Y, g.dr)[None, :]
        ref = m.aperture() * np.exp(-1j * np.pi * m.LAM * (qy ** 2 + qx ** 2) * df)
        np.testing.assert_allclose(A, ref, rtol=0, atol=1e-10)


def test_beams_match_numpy(small):
    by, bx = small.beams()
    my, mx = m.beam_list()
    assert np.array_equal(by, my) and np.array_equal(bx, mx)


def test_scale_gives_one_radian_at_aperture_edge(small):
    g = small
    by, bx = g.beams()
    pr = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, np.zeros((3, 8)))
    phi = np.linspace(0, 2 * np.pi, 721)
    k = g.alpha / g.lam
    for j, nm in enumerate(pr.names):
        b = pa.basis_function(nm, k * np.sin(phi), k * np.cos(phi), g.lam)
        assert abs(np.abs(b).max() * float(pr.scale[j]) - 1.0) < 1e-3


# ------------------------------------------------------------------------- simulator and depth sections
def test_simulator_reproduces_existing(small):
    g = small
    layers = random_layers(g)
    dps_ref, c_ref = m.simulate(layers, np.random.default_rng(7))
    C = pa.coefficient_matrix([{"C10": df} for df in m.DEFOCI])
    dps, c = jp.simulate_4dstem(jp.ops_layers(layers, m.LAYER_Z, m.THICK, g), C, g, dose=m.DOSE, rng=np.random.default_rng(7))
    assert np.array_equal(c, c_ref)
    assert np.array_equal(dps, dps_ref)


def test_depth_sections_match_existing(small):
    g = small
    rng = np.random.default_rng(3)
    nb = len(g.beams()[0])
    S = (rng.normal(size=(nb, g.Y, g.Y)) + 1j * rng.normal(size=(nb, g.Y, g.Y))).astype(np.complex64)
    depths = np.array([0.0, 10.0, 35.0])
    ref = m.depth_sections(S, m.beam_list(), depths)
    out = jp.depth_sections_general(S, g, depths)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-5


# ------------------------------------------------------------------------- regression vs NumPy reconstruction
def test_torch_fixed_probe_matches_numpy(small):
    g = small
    layers = random_layers(g)
    dps, coords = m.simulate(layers, np.random.default_rng(11))
    S0 = jp.vacuum_smatrix(g).numpy()
    rng = np.random.default_rng(5)                 # perturbed start: removes the arbitrary phase of numerically-zero |Z|
    S0 = (S0 + 0.05 * (rng.normal(size=S0.shape) + 1j * rng.normal(size=S0.shape))).astype(np.complex64)
    m.NITER = 3
    logs = []
    S_np, _, loss_np = m.reconstruct(dps, coords, logs.append, S_init=S0)
    meas = jp.Measurements.from_dps(dps, coords, m.DOSE)
    by, bx = g.beams()
    nominal = pa.coefficient_matrix([{"C10": df} for df in m.DEFOCI])
    probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, norm=1 / math.sqrt(len(by) * g.K ** 2))
    sch = jp.Schedule(mode="s_only", s_method="analytic", warmup_s=3, cycles=0, final_s=0, mu=m.MU)
    rec = jp.JointReconstructor(meas, g, probe, sch, S_init=S0, log=lambda *_: None)
    S_t = rec.run().numpy()
    rel = np.linalg.norm(S_t - S_np) / np.linalg.norm(S_np - S0)
    assert rel < 1e-4, rel
    np.testing.assert_allclose(rec.history["s_loss"], loss_np, rtol=1e-4)


# ------------------------------------------------------------------------- gradient checks (float64)
def _tiny_problem(g, seed=0):
    layers = random_layers(g, seed)
    true = pa.coefficient_matrix([{"C10": -40.0}, {"C10": -15.0, "C12a": 6.0, "C21b": 200.0},
                                  {"C10": 4.0, "C23a": 300.0, "C30": 1e4, "C12b": -5.0}])
    dps, coords = jp.simulate_4dstem(jp.ops_layers(layers, [5.0, 25.0, 45.0], 50.0, g), true, g)
    meas = jp.Measurements.from_dps(dps, coords, 1e5)
    by, bx = g.beams()
    rng = np.random.default_rng(seed)
    S = jp.vacuum_smatrix(g, dtype=torch.complex128).numpy()
    S = S + 0.3 * (rng.normal(size=S.shape) + 1j * rng.normal(size=S.shape))
    nominal = pa.coefficient_matrix([{"C10": -40.0}, {"C10": -20.0}, {"C10": 0.0}])
    probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, norm=1 / math.sqrt(len(by) * g.K ** 2))
    with torch.no_grad():
        probe.theta.add_(torch.as_tensor(rng.normal(scale=0.3, size=probe.theta.shape)))
    return meas, probe, torch.as_tensor(S)


def _loss(S, probe, meas, g, kind="amplitude"):
    il = probe.illumination(meas.d_idx, meas.pos.double() * g.dr).to(S.dtype)
    Z = jp.forward_chunk(jp.gather_windows(S, meas.pos, g.K), il)
    return jp.chunk_loss(Z, meas.amp.double(), kind, counts=1e3)


@pytest.mark.parametrize("kind", ["amplitude", "poisson"])
def test_probe_gradients_autograd_vs_finite_difference(small, kind):
    g = small
    meas, probe, S = _tiny_problem(g)
    L = _loss(S, probe, meas, g, kind)
    L.backward()
    grad = probe.theta.grad.clone()
    h = 1e-6
    for d in (1, 2):
        for j, nm in enumerate(probe.names):
            with torch.no_grad():
                probe.theta[d, j] += h; lp = float(_loss(S, probe, meas, g, kind))
                probe.theta[d, j] -= 2 * h; lm = float(_loss(S, probe, meas, g, kind))
                probe.theta[d, j] += h
            fd = (lp - lm) / (2 * h)
            assert abs(fd - float(grad[d, j])) <= 1e-5 * max(1.0, abs(fd)), (nm, d, fd, float(grad[d, j]))
    assert torch.all(grad[0] == 0)                   # reference dataset is not trainable


def test_s_gradients_autograd_vs_fd_and_adjoint(small):
    g = small
    meas, probe, S = _tiny_problem(g, seed=2)
    Sv = S.clone().requires_grad_(True)
    with torch.no_grad():
        il = probe.illumination(meas.d_idx, meas.pos.double() * g.dr).to(S.dtype)
    Z = jp.forward_chunk(jp.gather_windows(Sv, meas.pos, g.K), il)
    L = jp.chunk_loss(Z, meas.amp.double())
    L.backward()
    G = Sv.grad.clone()
    # finite differences of real and imaginary parts at random entries
    rng = np.random.default_rng(0)
    cr = g.crop()
    for _ in range(12):
        b = int(rng.integers(S.shape[0])); y = int(rng.integers(cr[0].start, cr[0].stop)); x = int(rng.integers(cr[1].start, cr[1].stop))
        for part, unit in (("re", 1.0), ("im", 1j)):
            h = 1e-6
            Sp = S.clone(); Sp[b, y, x] += unit * h
            Sm_ = S.clone(); Sm_[b, y, x] -= unit * h
            fd = (float(_loss_S(Sp, il, meas, g)) - float(_loss_S(Sm_, il, meas, g))) / (2 * h)
            ag = float(G[b, y, x].real if part == "re" else G[b, y, x].imag)
            assert abs(fd - ag) <= 1e-5 * max(1.0, abs(fd)), (part, fd, ag)
    # analytic adjoint (Pelz Eq. 22 / A3): dL/dS* = sum_j C_j^T [conj(P_j) F^-1(Z - a Z/|Z|)]; torch grad = 2 dL/dS*
    with torch.no_grad():
        Zd = Z.detach(); a = meas.amp.double()
        R = torch.fft.ifft2(Zd - a * Zd / Zd.abs(), norm="ortho")
        adj = torch.zeros_like(S); h = g.K // 2
        for j, (y0, x0) in enumerate(meas.pos.tolist()):
            adj[:, y0 - h:y0 + h, x0 - h:x0 + h] += torch.conj(il[j])[:, None, None] * R[j][None]
    assert torch.allclose(G, 2 * adj, rtol=1e-9, atol=1e-12)


def _loss_S(S, il, meas, g):
    Z = jp.forward_chunk(jp.gather_windows(S, meas.pos, g.K), il)
    return jp.chunk_loss(Z, meas.amp.double())


# ------------------------------------------------------------------------- residuals / S update for every data term
def _bf_mask(g):
    q = np.fft.fftfreq(g.K, g.dr)
    return torch.as_tensor(((q[:, None] ** 2 + q[None, :] ** 2) <= (g.alpha / g.lam) ** 2).astype(np.float64))


@pytest.mark.parametrize("kind,mask", [("amplitude", False), ("amplitude", True), ("poisson", False), ("poisson", True)])
def test_residual_is_wirtinger_gradient_of_loss(small, kind, mask):
    """s_residual = dD/dZ* (amplitude) and dD/dZ* / (2N) (Poisson), with and without a bright-field detector mask.
    torch's complex gradient is 2 dD/dZ*."""
    g = small
    rng = np.random.default_rng(3)
    Z = torch.as_tensor(rng.normal(size=(6, g.K, g.K)) + 1j * rng.normal(size=(6, g.K, g.K))).requires_grad_(True)
    a = torch.as_tensor(np.abs(rng.normal(size=(6, g.K, g.K))))
    a[0, :3] = 0.0                                   # zero-count pixels
    N = 37.0
    m_ = _bf_mask(g) if mask else None
    jp.chunk_loss(Z, a, kind, counts=N, det_mask=m_).backward()
    R = jp.s_residual(Z.detach(), a, kind, det_mask=m_)
    scale = 2.0 if kind == "amplitude" else 4.0 * N
    assert torch.allclose(R, Z.grad / scale, rtol=1e-10, atol=1e-12)


def test_poisson_residual_low_counts_and_near_zero_intensity(small):
    """Near-zero predicted intensity: the clamp keeps the residual finite and equal to the clamped formula; zero
    counts give R = Z/2 (pure intensity decrease); away from the floor the residual is the exact gradient."""
    g = small
    Z = torch.full((1, g.K, g.K), 1.0 + 0.0j, dtype=torch.complex128)
    Z[0, 0, 0] = 1e-9                                 # y = 1e-18 << floor = 1e-6 * max y
    Z[0, 0, 1] = 1e-2                                 # y = 1e-4 > floor: exact
    a = torch.full((1, g.K, g.K), 0.5, dtype=torch.float64)
    a[0, 1, :] = 0.0                                  # zero counts
    R = jp.s_residual(Z, a, "poisson", poisson_floor=1e-6)
    assert torch.isfinite(R).all()
    assert torch.allclose(R[0, 0, 0], 0.5 * Z[0, 0, 0] * (1 - 0.25 / 1e-6))
    assert torch.allclose(R[0, 0, 1], 0.5 * Z[0, 0, 1] * (1 - 0.25 / 1e-4))
    assert torch.allclose(R[0, 1, 5], 0.5 * Z[0, 1, 5])
    # near convergence the Poisson step equals the amplitude step (same MU for both losses)
    Zc = torch.full((1, g.K, g.K), 0.5 * (1 + 1e-4) + 0j, dtype=torch.complex128)
    ac = torch.full((1, g.K, g.K), 0.5, dtype=torch.float64)
    Ra, Rp = jp.s_residual(Zc, ac, "amplitude"), jp.s_residual(Zc, ac, "poisson")
    assert torch.allclose(Rp, Ra, rtol=2e-4)


@pytest.mark.parametrize("kind,det", [("amplitude", "full"), ("poisson", "full"), ("amplitude", "bf"), ("poisson", "bf")])
def test_s_update_is_preconditioned_gradient_step(small, kind, det):
    """One S sweep with a single chunk == S - eta/|Psi|^2 * dD/dS* (autograd), for both losses and the BF mask.
    This checks the hand-written Poisson/masked S update end to end (residual + adjoint + preconditioner).
    The Poisson intensity floor is disabled here (poisson_floor=0) so the update must equal the exact gradient; the
    floor itself is tested in test_poisson_floor_only_changes_clamped_pixels."""
    g = small
    meas, probe, S = _tiny_problem(g, seed=4)
    S32 = S.to(torch.complex64)
    sch = jp.Schedule(mode="s_only", s_method="analytic", loss=kind, det_mask=det, chunk=meas.J, warmup_s=1, cycles=0, final_s=0,
                      poisson_floor=0.0)
    rec = jp.JointReconstructor(meas, g, probe, sch, S_init=S32, log=lambda *_: None)
    rec.s_update_sweep(rec.eta)
    dS = (rec.S - S32).to(torch.complex128)
    Sv = S32.to(torch.complex128).requires_grad_(True)
    with torch.no_grad():
        il = probe.illumination(meas.d_idx, meas.pos.double() * g.dr).to(Sv.dtype)
    N = 50.0
    Z = jp.forward_chunk(jp.gather_windows(Sv, meas.pos, g.K), il)
    jp.chunk_loss(Z, meas.amp.double(), kind, counts=N, det_mask=rec.det_mask.double() if det == "bf" else None).backward()
    scale = 2.0 if kind == "amplitude" else 4.0 * N
    expected = -rec.eta / probe.norm ** 2 * Sv.grad / scale
    rel = float((dS - expected).norm() / expected.norm())
    assert rel < 2e-4, rel


def test_poisson_floor_only_changes_clamped_pixels(small):
    """With the default floor the Poisson residual differs from the exact gradient only where y < floor, and there
    it is bounded by |Z| I / floor / 2 instead of diverging like I / y."""
    g = small
    meas, probe, S = _tiny_problem(g, seed=4)
    with torch.no_grad():
        il = probe.illumination(meas.d_idx, meas.pos.double() * g.dr).to(S.dtype)
        Z = jp.forward_chunk(jp.gather_windows(S, meas.pos, g.K), il)
    a = meas.amp.double()
    y = Z.abs() ** 2
    clamped = y < 1e-6 * y.amax(dim=(-2, -1), keepdim=True)
    R_exact = jp.s_residual(Z, a, "poisson", poisson_floor=0.0)
    R_floor = jp.s_residual(Z, a, "poisson", poisson_floor=1e-6)
    assert clamped.any()                                         # this problem has at least one near-dark pixel
    assert torch.equal(R_exact[~clamped], R_floor[~clamped])
    assert (R_floor[clamped].abs() < R_exact[clamped].abs()).all()


def test_final_loss_is_loss_of_returned_state():
    """summary()['final_loss'] is the loss of the returned S and the final probe (after the last probe update)."""
    g, nominal, true, S_true, meas, by, bx, norm = _smoke_setup()
    probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, trainable=["C10"], norm=norm)
    sch = jp.Schedule(mode="joint", warmup_s=2, cycles=2, s_per_cycle=1, p_per_cycle=1, final_s=1, final_p=1,
                      mu=20.0, probe_lr=0.05)
    rec = jp.JointReconstructor(meas, g, probe, sch, log=lambda *_: None)
    S = rec.run()
    with torch.no_grad():
        il = probe.illumination(meas.d_idx, meas.pos.double() * g.dr).to(S.dtype)
        Z = jp.forward_chunk(jp.gather_windows(S, meas.pos, g.K), il)
        L = float(((Z.abs() - meas.amp) ** 2).sum()) / float((meas.amp.double() ** 2).sum())
    assert abs(rec.summary()["final_loss"] - L) <= 1e-5 * L
    assert rec.history["p_iter"][-1] == rec.history["s_iter"][-1]     # a probe step came after the last S sweep


def test_axial_fwhm_main_peak_interpolated():
    z = np.arange(0.0, 101.0, 10.0)
    tri = np.maximum(0.0, 1 - np.abs(z - 50) / 30)              # triangle, FWHM 30 exactly
    w, open_ = jp.axial_fwhm(z, tri)
    assert abs(w - 30.0) < 1e-9 and not open_
    two = tri + 0.9 * np.maximum(0.0, 1 - np.abs(z - 90) / 10)    # second, disconnected peak must not widen it
    assert abs(jp.axial_fwhm(z, two)[0] - 30.0) < 1e-9
    w_edge, open_edge = jp.axial_fwhm(z, np.maximum(0.0, 1 - z / 40))   # peak at the range start: lower bound
    assert open_edge and abs(w_edge - 20.0) < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_run_matches_cpu():
    """Same joint run on CUDA and CPU (all tensors on one device; results exported with .cpu())."""
    g, nominal, true, S_true, meas, by, bx, norm = _smoke_setup()
    res = {}
    for dev in ("cpu", "cuda"):
        probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, trainable=["C10", "C12a", "C12b"], norm=norm)
        sch = jp.Schedule(mode="joint", warmup_s=2, cycles=2, s_per_cycle=2, p_per_cycle=1, final_s=1, final_p=1,
                          mu=20.0, probe_lr=0.05, probe_fraction=1.0)
        rec = jp.JointReconstructor(meas, g, probe, sch, device=dev, log=lambda *_: None)
        S = rec.run()
        assert S.device.type == dev and probe.theta.device.type == dev
        sec = jp.depth_sections_general(S, g, [0.0, 20.0], device=dev)
        res[dev] = (S.cpu().numpy(), probe.coefficients().detach().cpu().numpy(), sec)
    assert np.linalg.norm(res["cuda"][0] - res["cpu"][0]) / np.linalg.norm(res["cpu"][0]) < 1e-3
    assert np.abs(res["cuda"][1] - res["cpu"][1]).max() < 0.05
    assert np.abs(res["cuda"][2] - res["cpu"][2]).max() / np.abs(res["cpu"][2]).max() < 1e-3


# ------------------------------------------------------------------------- smoke tests
def _smoke_setup(seed=4):
    g = jp.Geometry(K=32, M=2, dr=0.25, nscan=7, step=2)
    rng = np.random.default_rng(seed)
    c = g.Y * g.dr / 2
    yy = (np.arange(g.Y) * g.dr)[:, None]; xx = (np.arange(g.Y) * g.dr)[None, :]
    layers = []
    for _ in range(3):
        ph = np.zeros((g.Y, g.Y))
        for p in c + rng.uniform(-2.5, 2.5, (5, 2)):
            ph += 0.8 * np.exp(-((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / (2 * 0.45 ** 2))
        layers.append(ph)
    lz, thick = [10.0, 40.0, 70.0], 80.0
    nominal = pa.coefficient_matrix([{"C10": -60.0}, {"C10": -30.0}, {"C10": 0.0}])
    true = nominal.copy(); true[1, 0] += 12.0; true[2, 1] += 8.0; true[2, 0] -= 9.0     # relative C10 / C12a errors
    S_true = jp.true_smatrix(jp.ops_layers(layers, lz, thick, g), g, thick, reference="exit").astype(np.complex64)
    by, bx = g.beams()
    norm = 1 / math.sqrt(len(by) * g.K ** 2)
    true_probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, true, norm=norm)
    dps, coords = jp.simulate_from_smatrix(S_true, true_probe, g)          # model-consistent data
    meas = jp.Measurements.from_dps(dps, coords, 1e5)
    return g, nominal, true, S_true, meas, by, bx, norm


def test_probe_only_recovers_relative_error_with_true_S():
    """Model-consistent data + exact S fixed: probe-only optimisation must recover the relative aberration errors."""
    g, nominal, true, S_true, meas, by, bx, norm = _smoke_setup()
    probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, trainable=["C10", "C12a", "C12b"], norm=norm)
    sch = jp.Schedule(mode="probe_only", cycles=80, p_per_cycle=1, probe_lr=0.1, grad_clip=0)
    jp.JointReconstructor(meas, g, probe, sch, S_init=S_true, log=lambda *_: None).run()
    err = probe.coefficients().detach().numpy() - true
    assert np.abs(err[:, :3]).max() < 0.3, err[:, :3]


def test_joint_smoke_runs_and_reduces_loss():
    """Joint S + probe from vacuum on model-consistent data: runs, loss decreases, relative defocus error shrinks."""
    g, nominal, true, S_true, meas, by, bx, norm = _smoke_setup()
    probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, trainable=["C10", "C12a", "C12b"], norm=norm)
    sch = jp.Schedule(mode="joint", warmup_s=5, cycles=8, s_per_cycle=3, p_per_cycle=1, final_s=2, final_p=1,
                      mu=20.0, probe_lr=0.05)
    rec = jp.JointReconstructor(meas, g, probe, sch, log=lambda *_: None)
    rec.run()
    h = rec.history
    assert h["s_loss"][-1] < 0.5 * h["s_loss"][0]
    e0 = np.abs(nominal - true)[1:, :3].sum()
    e1 = np.abs(probe.coefficients().detach().numpy() - true)[1:, :3].sum()
    assert e1 < e0, (e0, e1)


@pytest.mark.parametrize("opt", ["adam", "sgd"])
def test_autograd_s_update_reduces_loss(opt):
    """Autograd + torch.optim S update (default s_method) drives the data loss down on the smoke problem."""
    g, nominal, true, S_true, meas, by, bx, norm = _smoke_setup()
    probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, trainable=[], norm=norm)
    sch = jp.Schedule(mode="s_only", s_optimizer=opt, s_lr=0.01 if opt == "adam" else 0.0, warmup_s=6, cycles=0, final_s=0, mu=20.0)
    rec = jp.JointReconstructor(meas, g, probe, sch, log=lambda *_: None)
    S = rec.run()
    assert not S.requires_grad
    l = rec.history["s_loss"]
    assert l[-1] < 0.7 * l[0], l
