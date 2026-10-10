"""Differentiable electron-probe model with an aberration basis (Pelz et al., PRR 3, 023159 (2021), Eq. 23).

Eq. 23 (Pelz et al.):   chi_d(alpha, phi) = (2 pi / lambda) * sum_{n,m} C_{n,m} alpha^{n+1} cos[m (phi - phi_{n,m})] / (n + 1)
                        Psi_d(q) = A(q) exp(-i chi_d(q)),   A = circular top-hat aperture.

Conventions used here (identical to smatrix_depth_demo.py):
  * alpha = lambda |k| (small-angle), phi = atan2(k_y, k_x), k in 1/Angstrom, all C in Angstrom.
  * With C10 = df, the C10 term gives exp(-i pi lambda df k^2), i.e. exactly the probe phase used by the
    existing simulator/reconstruction (probe defocus df; negative df = focus inside the sample).
  * For m > 0 the polar pair (C_{n,m}, phi_{n,m}) is stored in Cartesian form
        C_{n,m} cos[m(phi - phi_{n,m})] = Ca cos(m phi) + Cb sin(m phi),  Ca = C cos(m phi_nm), Cb = C sin(m phi_nm),
    which is mathematically identical to Eq. 23 but avoids the angle singularity at C = 0 during optimisation.
  * Trainable parameters are dimensionless: theta_j = Delta C_j / s_j with s_j chosen so that theta_j = 1 produces a
    maximum phase of 1 rad on the aperture (|basis_j| <= 1/s_j).  This balances coefficients of very different units.

Gauge (Pelz et al. Sec. II.F): Psi_{d,b} and S_b enter only as a product, so a phase pattern common to all datasets can
be moved into S without changing any intensity.  Only aberrations that DIFFER between datasets are identifiable from
the data.  The model therefore keeps one reference dataset fixed at its nominal (calibrated) aberrations and learns
relative corrections for the others.
"""
import math
import numpy as np
import torch
import torch.nn as nn

# name -> (n, m, component)   component: None (m = 0), "a" (cos m phi) or "b" (sin m phi)
ABERRATION_BASIS = {
    "C10": (1, 0, None),                       # defocus
    "C12a": (1, 2, "a"), "C12b": (1, 2, "b"),  # two-fold astigmatism
    "C21a": (2, 1, "a"), "C21b": (2, 1, "b"),  # axial coma
    "C23a": (2, 3, "a"), "C23b": (2, 3, "b"),  # three-fold astigmatism
    "C30": (3, 0, None),                       # spherical aberration
    "C32a": (3, 2, "a"), "C32b": (3, 2, "b"),  # star aberration (optional)
    "C34a": (3, 4, "a"), "C34b": (3, 4, "b"),  # four-fold astigmatism (optional)
}
DEFAULT_NAMES = ["C10", "C12a", "C12b", "C21a", "C21b", "C23a", "C23b", "C30"]


def basis_function(name, ky, kx, lam, xp=np):
    """chi contribution of a unit (1 Angstrom) coefficient `name` at wave vectors (ky, kx) [1/A]."""
    n, mm, comp = ABERRATION_BASIS[name]
    alpha = lam * xp.sqrt(ky ** 2 + kx ** 2)
    phi = xp.arctan2(ky, kx)
    radial = (2 * math.pi / lam) * alpha ** (n + 1) / (n + 1)
    if comp is None:
        return radial
    return radial * (xp.cos(mm * phi) if comp == "a" else xp.sin(mm * phi))


def chi_eq23_polar(ky, kx, lam, terms):
    """Literal Eq. 23 with polar coefficients: terms = {(n, m): (C_nm [A], phi_nm [rad])}. NumPy, for verification."""
    alpha = lam * np.sqrt(ky ** 2 + kx ** 2)
    phi = np.arctan2(ky, kx)
    chi = np.zeros_like(alpha, dtype=float)
    for (n, mm), (C, phi0) in terms.items():
        chi += C * alpha ** (n + 1) * np.cos(mm * (phi - phi0)) / (n + 1)
    return 2 * math.pi / lam * chi


def polar_to_cartesian(terms):
    """{(n,m): (C, phi)} -> {name: value} in this module's Cartesian convention."""
    out = {}
    for (n, mm), (C, phi0) in terms.items():
        if mm == 0:
            out[f"C{n}{mm}"] = C
        else:
            out[f"C{n}{mm}a"] = C * math.cos(mm * phi0)
            out[f"C{n}{mm}b"] = C * math.sin(mm * phi0)
    return out


def cartesian_to_polar(coeffs):
    """{name: value} -> {(n,m): (C, phi)} (phi in rad, defined modulo 2 pi / m)."""
    out = {}
    for name, v in coeffs.items():
        n, mm, comp = ABERRATION_BASIS[name]
        if comp is None:
            out[(n, mm)] = (float(v), 0.0)
    pairs = {}
    for name, v in coeffs.items():
        n, mm, comp = ABERRATION_BASIS[name]
        if comp is not None:
            pairs.setdefault((n, mm), [0.0, 0.0])["ab".index(comp)] = float(v)
    for (n, mm), (a, b) in pairs.items():
        out[(n, mm)] = (math.hypot(a, b), math.atan2(b, a) / mm)
    return out


class AberrationProbe(nn.Module):
    """Per-dataset probe coefficients Psi_{d,b} = norm * A_b * exp(-i chi_d(k_b)) for the S-matrix beams.

    nominal:   (D, P) array of nominal coefficients [A] for the P names (acquisition parameters / initial guess)
    trainable: names that may be refined; the reference dataset `ref_index` is never refined (gauge fixing).
    """

    def __init__(self, beam_ky, beam_kx, lam, alpha_max, nominal, names=None, trainable=None, ref_index=0,
                 norm=1.0, device="cpu", param_dtype=torch.float64):
        super().__init__()
        self.names = list(names or DEFAULT_NAMES)
        self.trainable_names = list(trainable if trainable is not None else self.names)
        self.lam, self.ref_index = float(lam), int(ref_index)
        ky = torch.as_tensor(np.asarray(beam_ky, float), dtype=param_dtype, device=device)
        kx = torch.as_tensor(np.asarray(beam_kx, float), dtype=param_dtype, device=device)
        self.register_buffer("ky", ky)
        self.register_buffer("kx", kx)
        basis = torch.stack([basis_function(nm, ky, kx, self.lam, xp=torch) for nm in self.names])   # (P, B)
        self.register_buffer("basis", basis)
        # scale: theta = 1 -> max |phase| = 1 rad on the aperture edge (alpha_max), per basis function
        scale = []
        for nm in self.names:
            n, mm, comp = ABERRATION_BASIS[nm]
            scale.append(lam * (n + 1) / (2 * math.pi * alpha_max ** (n + 1)))
        self.register_buffer("scale", torch.tensor(scale, dtype=param_dtype, device=device))
        nominal = torch.as_tensor(np.asarray(nominal, float), dtype=param_dtype, device=device)
        self.register_buffer("nominal", nominal)                                      # (D, P) [A]
        D, P = nominal.shape
        mask = torch.zeros(D, P, dtype=param_dtype, device=device)
        for j, nm in enumerate(self.names):
            if nm in self.trainable_names:
                mask[:, j] = 1.0
        mask[self.ref_index] = 0.0                                                     # reference dataset fixed
        self.register_buffer("mask", mask)
        self.theta = nn.Parameter(torch.zeros(D, P, dtype=param_dtype, device=device))  # dimensionless corrections
        self.norm = float(norm)

    @property
    def n_datasets(self):
        return self.nominal.shape[0]

    def coefficients(self):
        """(D, P) aberration coefficients in Angstrom (nominal + masked correction)."""
        return self.nominal + self.mask * self.theta * self.scale

    def chi(self):
        """(D, B) aberration phase at the beam wave vectors (Eq. 23)."""
        return self.coefficients() @ self.basis

    def beam_coefficients(self):
        """(D, B) complex Psi_{d,b} = norm * exp(-i chi_d(k_b)) (A_b = 1 for all beams inside the aperture)."""
        return self.norm * torch.exp(-1j * self.chi())

    def illumination(self, d_idx, pos_A):
        """Illumination vector P_{j,b} = Psi_{d(j),b} exp(-2 pi i k_b . r_j) for measurements j (Eq. 9 phase factors).

        d_idx: (J,) long dataset index, pos_A: (J, 2) probe positions in Angstrom (y, x).  Returns (J, B) complex.
        """
        psi = self.beam_coefficients()[d_idx]
        ramp = -2 * math.pi * (pos_A[:, :1] * self.ky[None] + pos_A[:, 1:] * self.kx[None])
        return psi * torch.exp(1j * ramp)

    def regularizer(self):
        """Quadratic penalty on the normalised corrections (optional prior towards the nominal probe)."""
        return (self.mask * self.theta).pow(2).sum()

    def as_dicts(self):
        C = self.coefficients().detach().cpu().numpy()
        return [{nm: float(C[d, j]) for j, nm in enumerate(self.names)} for d in range(C.shape[0])]


def coefficient_matrix(dicts, names=None):
    """List (per dataset) of {name: value} -> (D, P) array (missing names = 0)."""
    names = list(names or DEFAULT_NAMES)
    return np.array([[float(d.get(nm, 0.0)) for nm in names] for d in dicts])


def probe_on_grid(coeffs, names, ny, nx, dr, lam, alpha_max, xp=np):
    """Fourier-space probe A(q) exp(-i chi(q)) on an ny x nx grid with pixel dr (unnormalised), for simulation/plots."""
    qy = xp.fft.fftfreq(ny, dr)[:, None] + 0 * xp.fft.fftfreq(nx, dr)[None, :]
    qx = 0 * xp.fft.fftfreq(ny, dr)[:, None] + xp.fft.fftfreq(nx, dr)[None, :]
    chi = 0.0
    for nm, c in zip(names, coeffs):
        if c != 0.0:
            chi = chi + c * basis_function(nm, qy, qx, lam, xp=xp)
    A = (qy ** 2 + qx ** 2) <= (alpha_max / lam) ** 2
    return A * xp.exp(-1j * chi)
