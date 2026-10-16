"""Single-focus joint S + probe mode: gauge handling, probe gradients, and the probe/S ambiguity.
Run: python -m pytest -q tests/test_single_focus_joint.py"""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simple_joint as sj                              # noqa: E402

ST = sj.Setup(dr=0.3, K=32, Y=64)


def model(ref, nominal=(-100.0, 0.0, 0.0), seed=0):
    y = np.arange(ST.K // 2, ST.Y - ST.K // 2 + 1, 4)
    pos = np.stack(np.meshgrid(y, y, indexing="ij"), -1).reshape(-1, 2)
    nom = np.zeros((1, len(sj.NAMES)))
    nom[0, :3] = nominal
    m = sj.JointModel(ST, pos, np.zeros(len(pos), int), nom, ref=ref, device="cpu")
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        m.S.mul_(torch.as_tensor(np.exp(0.3j * rng.normal(size=m.S.shape)).astype(np.complex64)))
    return m


def test_reference_mask():
    assert float(model(ref=0).mask.abs().sum()) == 0.0                 # sole dataset frozen (fixed mode)
    m = model(ref=None)
    trained = [sj.NAMES.index(n) for n in ("C10", "C12a", "C12b")]
    assert m.mask[0, trained].tolist() == [1.0, 1.0, 1.0]
    assert float(m.mask.sum()) == 3.0                                  # nothing else (aperture amplitude is a buffer)


def test_probe_gradient_finite_nonzero_and_coefficients_move():
    m = model(ref=None)
    with torch.no_grad():
        target = model(ref=None, nominal=(-90.0, 4.0, -3.0))(torch.arange(len(m.pos))).abs()
    hist = sj.fit(m, target, epochs=3, warmup=1, batch=8, lr_S=0.0, lr_probe=0.02, log=lambda s: None)
    tr = m.fit_trace
    assert tr["probe_grad_norm_mean"][0] is None                       # warm-up: no probe step
    assert all(tr["probe_grad_finite"][1:]) and all(g > 0 for g in tr["probe_grad_norm_mean"][1:])
    c = m.coefficients().detach().numpy()[0, :3]
    assert np.abs(c - [-100.0, 0.0, 0.0]).max() > 1e-3
    assert len(hist) == 3


def test_fresh_selection_loss_is_used():
    m = model(ref=None)
    with torch.no_grad():
        target = m(torch.arange(len(m.pos))).abs() * 1.01
    calls = []
    sj.fit(m, target, epochs=4, warmup=1, batch=8, lr_S=0.01, select_fn=lambda mm: calls.append(1) or 1.0 / len(calls),
           select_every=2, log=lambda s: None)
    assert m.fit_trace["select_loss"] == [None, 1.0, None, 0.5] and m.best_epoch == 4


def test_probe_phase_compensated_by_smatrix_leaves_patterns_unchanged():
    """Change the probe aberrations by dchi and multiply every beam S_b by exp(+i dchi(k_b)): same diffraction."""
    m = model(ref=None)
    idx = torch.arange(len(m.pos))
    with torch.no_grad():
        Z0 = m(idx)
        psi0 = m.probe()[0]
        m.dC[0, :3] = torch.tensor([1.7, -0.9, 0.6], dtype=torch.float64)   # ~27 A defocus + astigmatism
        dphi = torch.angle(psi0) - torch.angle(m.probe()[0])            # exp(i dphi) * psi_new = psi_old
        Z_probe_only = m(idx)
        m.S.mul_(torch.exp(1j * dphi).to(m.S.dtype)[:, None, None])
        Z1 = m(idx)
    rel = lambda a, b: float((a.abs() - b.abs()).abs().max() / b.abs().max())
    assert rel(Z_probe_only, Z0) > 1e-2                                # the probe change alone is visible
    assert rel(Z1, Z0) < 1e-5                                          # compensated: invisible (float32 precision)
    assert float((Z1 - Z0).abs().max() / Z0.abs().max()) < 1e-5        # even the complex far field matches


def test_cli_joint_mode_is_opt_in(monkeypatch):
    import pytest
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synthetic_benchmark"))
    import run_benchmark as rb
    cfg = dict(name="t", specimen=dict(thickness_A=200.0), defocus=dict(nominal_C10=[-1.0, -2.0, -3.0], reference=0),
               _path="t", _sha256="t")
    monkeypatch.setattr(rb.bench, "load_config", lambda p: cfg)
    calls = []
    monkeypatch.setattr(rb, "smatrix", lambda c, a: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "smatrix", "--single-dataset"])
    rb.main()
    assert calls[-1].single_probe_mode == "fixed" and calls[-1].probe_init == []
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "smatrix", "--single-dataset", "--single-probe-mode", "joint",
                                      "--probe-init", "C10=-120"])
    rb.main()
    assert calls[-1].single_probe_mode == "joint" and calls[-1].probe_init == [("C10", -120.0)]
    for bad in (["smatrix", "--single-probe-mode", "joint"], ["smatrix", "--single-dataset", "--probe-init", "C10=1"]):
        monkeypatch.setattr(sys, "argv", ["run_benchmark.py", *bad])
        with pytest.raises(SystemExit):
            rb.main()
