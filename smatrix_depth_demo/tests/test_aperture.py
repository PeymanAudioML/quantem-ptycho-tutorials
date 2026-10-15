"""Soft aperture edge (Setup.taper): hard default unchanged, taper amplitudes, beams and probes consistent."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simple_joint as sj                                # noqa: E402
import abtem_data as ad                                  # noqa: E402


def test_hard_default_unchanged():
    s = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.2, K=48, Y=96)
    n = np.fft.fftfreq(s.K, 1.0 / s.K)
    ny, nx = np.meshgrid(n, n, indexing="ij")
    k = 1.0 / (s.K * s.dr)
    old = np.hypot(ny, nx) * k <= s.alpha / s.lam                 # the pre-taper definition
    assert len(s.beams()[0]) == int(old.sum())
    m = sj.JointModel(s, np.array([[24, 24]]), np.array([0]), np.zeros((1, len(sj.NAMES))), trainable=())
    assert torch.allclose(m.amp, torch.ones_like(m.amp))


def test_taper_amplitude_and_beams():
    hard = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.2, K=96, Y=192)
    soft = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.2, K=96, Y=192, taper=2e-3)
    a = np.array([0.0, 18.9e-3, 19e-3, 20e-3, 21e-3, 21.1e-3]) / soft.lam
    assert np.allclose(soft.aperture(a), [1, 1, 1, 0.5, 0, 0])
    kb = np.hypot(*soft.beams()) * soft.lam
    assert len(kb) > len(hard.beams()[0]) and kb.max() < 21e-3
    m = sj.JointModel(soft, np.array([[48, 48]]), np.array([0]), np.zeros((1, len(sj.NAMES))), trainable=())
    assert np.allclose(m.amp.numpy(), soft.aperture(np.hypot(*soft.beams())))
    assert float(m.amp.min()) > 0


def test_entrance_probe_matches_model_probe():
    """The simulated probe's Fourier coefficients at the beams equal JointModel.probe() up to one scale factor."""
    s = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.2, K=48, Y=96, taper=2e-3)
    c = np.zeros(len(sj.NAMES))
    c[:3] = [-90.0, 6.0, -4.0]
    p = ad.entrance_probe(s, c)
    F = np.fft.fft2(p)
    ky, kx = s.beams()
    f = s.Y // s.K
    iy, ix = np.round(ky * s.Y * s.dr).astype(int) % s.Y, np.round(kx * s.Y * s.dr).astype(int) % s.Y
    m = sj.JointModel(s, np.array([[24, 24]]), np.array([0]), c[None], trainable=())
    psi = m.probe().detach().numpy()[0]
    ratio = F[iy, ix] / psi
    assert np.allclose(ratio, ratio[0], rtol=1e-4)
    assert f == 2
