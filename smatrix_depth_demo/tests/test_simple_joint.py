"""Tests for simple_joint's forward model, refocusing convention and depth tools.  Run: python -m pytest -q tests"""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simple_joint as sj                              # noqa: E402
import joint_smatrix_probe as jp                       # noqa: E402
import probe_aberrations as pa                         # noqa: E402

ST = sj.Setup(dr=0.3, K=32, Y=64)
THICK = 200.0
DEPTHS = np.arange(-40.0, THICK + 40.0 + 1e-6, 5.0)


def blobs(seed, n=3, spread=2.0):
    rng = np.random.default_rng(seed)
    r = np.arange(ST.Y) * ST.dr
    c = ST.Y * ST.dr / 2
    img = np.zeros((ST.Y, ST.Y))
    for y, x in c + rng.uniform(-spread, spread, (n, 2)):
        img += np.exp(-((r[:, None] - y) ** 2 + (r[None, :] - x) ** 2) / (2 * 0.45 ** 2))
    return img


def test_forward_matches_joint_smatrix_probe():
    g = jp.Geometry(K=ST.K, M=2, dr=ST.dr, nscan=5, step=3)
    pos = np.tile(g.scan_coords().reshape(-1, 2), (2, 1)); didx = np.repeat([0, 1], 25)
    nominal = np.zeros((2, 8)); nominal[:, 0] = [-50.0, 0.0]; nominal[1, 1] = 5.0
    m = sj.JointModel(ST, pos, didx, nominal)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        m.S.add_(torch.as_tensor((0.2 * rng.normal(size=m.S.shape)).astype(np.complex64)))
    by, bx = g.beams()
    probe = pa.AberrationProbe(by / g.LW, bx / g.LW, g.lam, g.alpha, nominal, norm=1 / math.sqrt(len(by) * g.K ** 2))
    with torch.no_grad():
        Z1 = m(torch.arange(len(pos)))
        il = probe.illumination(torch.as_tensor(didx), torch.as_tensor(pos).double() * g.dr).to(torch.complex64)
        Z2 = jp.forward_chunk(jp.gather_windows(m.S.detach(), torch.as_tensor(pos), g.K), il)
    assert float((Z1 - Z2).abs().max() / Z2.abs().max()) < 1e-5


def test_exact_smatrix_matches_reference_implementation():
    g = jp.Geometry(K=ST.K, M=2, dr=ST.dr)
    L, lz = [blobs(1), blobs(2)], [40.0, 150.0]
    for ref in ("entrance", "exit"):
        a = sj.exact_smatrix(ST, L, lz, THICK, reference=ref).numpy()
        b = jp.true_smatrix(jp.ops_layers(L, lz, THICK, g), g, THICK, reference=ref)
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-5


def test_depth_convention_template_fit_on_exact_s():
    """z = 0 entrance surface: a single feature at an off-grid depth is found by template fitting, with templates from
    a different cluster, from an entrance-referenced exact S."""
    z0_grid = np.arange(0.0, THICK + 1e-6, 20.0)
    timg = blobs(11)
    h = np.array([sj.feature_response(sj.refocus(sj.exact_smatrix(ST, [timg], [z], THICK), ST, DEPTHS), timg / timg.max())
                  for z in z0_grid])
    img = blobs(22)
    for z0 in (35.0, 125.0):
        r = sj.feature_response(sj.refocus(sj.exact_smatrix(ST, [img], [z0], THICK), ST, DEPTHS), img / img.max())
        assert abs(sj.fit_depth(r, h, z0_grid)[0] - z0) <= 3.0


def test_reference_plane_recovers_known_shift():
    S = sj.exact_smatrix(ST, [blobs(3)], [100.0], THICK)
    q = np.fft.fftfreq(ST.Y, ST.dr)
    shifted = np.fft.ifft2(np.fft.fft2(S.numpy()) * np.exp(-1j * np.pi * ST.lam * (-30.0) * (q[:, None] ** 2 + q[None, :] ** 2)))
    dz, _, err = sj.reference_plane(shifted, S.numpy(), ST, np.arange(-60.0, 61.0, 5.0))
    assert dz == 30.0 and err.min() < 1e-4
