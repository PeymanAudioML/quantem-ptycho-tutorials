"""Tests for run_realization helpers: detector centring / binning, probe comparison, loading the realization layout."""
import os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simple_joint as sj                                # noqa: E402
import run_realization as rr                             # noqa: E402

SETUP = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.2, K=32, Y=64)


def probe(N, dr, c10):
    q = np.fft.fftfreq(N, dr)
    qy, qx = np.meshgrid(q, q, indexing="ij")
    B = sj.aberration_basis(torch.as_tensor(qy.ravel()), torch.as_tensor(qx.ravel()), SETUP.lam).numpy()
    c = np.zeros(len(sj.NAMES))
    c[0] = c10
    chi = (c @ B).reshape(N, N)
    return np.fft.ifft2((np.hypot(qy, qx) <= SETUP.alpha / SETUP.lam) * np.exp(-1j * chi))


@pytest.mark.parametrize("b", [1, 2, 4])
def test_centre_and_bin_put_zero_frequency_at_bin_centre(b):
    N = 64
    zero = np.array([31.6, 32.3])                         # sub-pixel, off-centre zero frequency
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    spot = np.exp(-((yy - zero[0]) ** 2 + (xx - zero[1]) ** 2) / (2 * 1.5 ** 2))
    d = np.tile(spot, (2, 2, 1, 1)).astype(np.float32)
    x, c = rr.bin_patterns(rr.centre_patterns(d, zero, b), b)
    p = x[0, 0]
    cy = (p.sum(1) * np.arange(p.shape[0])).sum() / p.sum()
    cx = (p.sum(0) * np.arange(p.shape[1])).sum() / p.sum()
    assert abs(cy - c) < 0.1 and abs(cx - c) < 0.1


def test_probe_match_is_shift_invariant_and_sensitive():
    ky, kx = SETUP.beams()
    ref = rr.probe_at_beams(probe(64, 0.2, -83.0), 0.2, ky, kx)
    for p, dr in ((np.roll(probe(64, 0.2, -83.0), (7, -3), (0, 1)), 0.2), (np.fft.fftshift(probe(128, 0.1, -83.0)), 0.1)):
        ov, rms = rr.probe_match(rr.probe_at_beams(p, dr, ky, kx), ref, ky, kx)
        assert ov > 0.9999 and rms < 1e-3
    ov, rms = rr.probe_match(rr.probe_at_beams(probe(64, 0.2, -90.0), 0.2, ky, kx), ref, ky, kx)
    assert ov < 0.995 and rms > 0.05


def test_local_peaks():
    z = np.linspace(0, 100, 101)
    c = np.exp(-(z - 30) ** 2 / 20) + 0.8 * np.exp(-(z - 70) ** 2 / 20) + 0.01 * np.sin(z)
    assert np.allclose(rr.local_peaks(z, c, 5, frac=0.5, min_sep=10), [30, 70], atol=1.0)


def test_load_standin(tmp_path):
    import make_standin
    make_standin.main(str(tmp_path), dose=None)
    a = type("A", (), dict(energy=None, alpha=None, defocus=None))()
    d = rr.load_dataset(str(tmp_path / "def_050"), a, 0)
    assert d["data"].shape == (17, 17, 32, 32)
    assert d["energy"] == 300e3 and d["alpha_mrad"] == 20.0 and d["defocus"] == 50.0
    assert np.allclose(d["zero"], 16) and np.allclose(d["step"], 0.4) and np.allclose(d["cell"][-1], 100.0)
    assert np.allclose(d["positions_A"][0, 0], [3.2, 3.2])
    tr = rr.load_truth([str(tmp_path / "def_050")], lambda m: None)
    assert tr["potential"][0].shape == (50, 64, 64) and tr["atoms"][1].shape == (6, 3)
