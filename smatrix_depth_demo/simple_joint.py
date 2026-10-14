"""Joint S-matrix + probe reconstruction from 4D-STEM diffraction patterns, and refocusing (PyTorch).

One file, three steps:

    setup = Setup(energy_ev=300e3, alpha=20e-3, dr=0.2, K=64, Y=128)       # microscope + sampling
    model = JointModel(setup, positions, dataset_index, nominal)          # unknowns: S and probe aberrations
    fit(model, amplitudes)                                                 # joint gradient descent (autograd + Adam)
    volume = refocus(model.S, setup, depths)                               # complex exit wave at each depth

Forward model (Pelz et al., PRR 3, 023159 (2021), Eq. 9):

    Z_j(q) = FFT_K[ sum_b  Psi_d(k_b) exp(-2 pi i k_b . r_j)  S_b(window around r_j) ]
    predicted amplitude = |Z_j|,   measured amplitude a_j = sqrt(I_j / max_j sum I_j)

    S_b(r)      : the S-matrix, one complex image per beam k_b inside the aperture           (trainable)
    Psi_d(k)    : probe of dataset d = norm * exp(-i chi_d(k)), chi from aberration coefficients (trainable)
    r_j, d_j    : scan position (S-grid pixels) and dataset (e.g. defocus) of pattern j

Loss: amplitude loss  sum_j sum_q (|Z_j(q)| - a_j(q))^2   (Eq. 12), normalised by sum a^2.

Gauge: S and Psi only appear as a product, so an aberration COMMON to all datasets cannot be found from the data.
Dataset `ref` keeps its nominal aberrations; the others learn corrections relative to it.  Absolute depth (refocus)
therefore depends on the reference dataset's calibration.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

# ------------------------------------------------------------------------------------------------ setup / probe
ABERRATIONS = {            # name: (n, m, "a" = cos(m phi) | "b" = sin(m phi) | None)   chi term ~ alpha^(n+1)
    "C10": (1, 0, None),                        # defocus [A]; negative = focus inside the sample
    "C12a": (1, 2, "a"), "C12b": (1, 2, "b"),   # two-fold astigmatism
    "C21a": (2, 1, "a"), "C21b": (2, 1, "b"),   # axial coma
    "C23a": (2, 3, "a"), "C23b": (2, 3, "b"),   # three-fold astigmatism
    "C30": (3, 0, None),                        # spherical aberration
}
NAMES = list(ABERRATIONS)


@dataclass
class Setup:
    energy_ev: float = 300e3
    alpha: float = 20e-3     # probe convergence semi-angle [rad]
    dr: float = 0.2          # real-space pixel [A]
    K: int = 64              # detector / window size [px]; the pattern is the FFT of a K x K window
    Y: int = 128             # S-matrix grid size [px] (square, multiple of K)

    @property
    def lam(self):
        """Relativistic electron wavelength [A]."""
        hc, m0c2 = 1.23984193e4, 5.109989461e5
        return hc / math.sqrt(self.energy_ev * (self.energy_ev + 2 * m0c2))

    def beams(self):
        """Wave vectors (ky, kx) [1/A] of the S-matrix beams: window reciprocal grid points inside the aperture."""
        n = np.fft.fftfreq(self.K, 1.0 / self.K)
        ny, nx = np.meshgrid(n, n, indexing="ij")
        k = 1.0 / (self.K * self.dr)
        inside = np.hypot(ny, nx) * k <= self.alpha / self.lam
        return ny[inside] * k, nx[inside] * k


def aberration_basis(ky, kx, lam, names=NAMES):
    """(P, B) phase [rad] of a 1 A coefficient for each aberration (Pelz Eq. 23, Cartesian form)."""
    alpha = lam * torch.sqrt(ky ** 2 + kx ** 2)
    phi = torch.atan2(ky, kx)
    rows = []
    for nm in names:
        n, m, comp = ABERRATIONS[nm]
        radial = 2 * math.pi / lam * alpha ** (n + 1) / (n + 1)
        rows.append(radial if comp is None else radial * (torch.cos(m * phi) if comp == "a" else torch.sin(m * phi)))
    return torch.stack(rows)


def aberration_scale(setup, names=NAMES):
    """Coefficient [A] giving 1 rad at the aperture edge, per aberration: the unit of the trainable corrections."""
    return torch.tensor([setup.lam * (ABERRATIONS[nm][0] + 1) / (2 * math.pi * setup.alpha ** (ABERRATIONS[nm][0] + 1))
                         for nm in names], dtype=torch.float64)


# ------------------------------------------------------------------------------------------------ model
class JointModel(torch.nn.Module):
    """Unknowns: S (B, Y, Y) complex and aberration corrections (D, P).  Known: geometry, scan, nominal aberrations.

    positions:     (J, 2) int, scan positions in S-grid pixels (centre of the K x K window, must be >= K/2 from edges)
    dataset_index: (J,) int, which dataset (probe) recorded pattern j
    nominal:       (D, P) nominal aberration coefficients [A] in the order of `names` (e.g. the set defoci)
    trainable:     aberration names refined on the non-reference datasets
    """

    def __init__(self, setup, positions, dataset_index, nominal, trainable=("C10", "C12a", "C12b"), ref=0,
                 names=NAMES, device="cpu"):
        super().__init__()
        self.setup, self.names, self.ref = setup, list(names), ref
        unknown = set(trainable) - set(self.names)
        if unknown:
            raise ValueError(f"unknown aberrations {unknown}")
        dev = torch.device(device)
        ky, kx = (torch.as_tensor(v, dtype=torch.float64, device=dev) for v in setup.beams())
        self.register_buffer("ky", ky)
        self.register_buffer("kx", kx)
        self.register_buffer("basis", aberration_basis(ky, kx, setup.lam, self.names))              # (P, B)
        self.register_buffer("scale", aberration_scale(setup, self.names).to(dev))                 # (P,)
        nominal = torch.as_tensor(np.asarray(nominal, float), dtype=torch.float64, device=dev)
        self.register_buffer("nominal", nominal)                                                    # (D, P)
        mask = torch.zeros_like(nominal)
        for j, nm in enumerate(self.names):
            if nm in trainable:
                mask[:, j] = 1.0
        mask[ref] = 0.0                                                     # gauge: reference dataset fixed
        self.register_buffer("mask", mask)
        self.register_buffer("pos", torch.as_tensor(np.asarray(positions), dtype=torch.long, device=dev))
        self.register_buffer("didx", torch.as_tensor(np.asarray(dataset_index), dtype=torch.long, device=dev))
        B, K = len(ky), setup.K
        self.norm = 1.0 / math.sqrt(B * K * K)                             # unit total intensity per pattern
        self.dC = torch.nn.Parameter(torch.zeros_like(nominal))           # corrections, 1 = 1 rad at aperture edge
        self.S = torch.nn.Parameter(vacuum_smatrix(setup, device=dev))    # (B, Y, Y) complex64
        self.register_buffer("win", torch.arange(-(K // 2), K - K // 2, device=dev))

    def coefficients(self):
        """(D, P) aberration coefficients [A]."""
        return self.nominal + self.mask * self.dC * self.scale

    def probe(self):
        """(D, B) complex probe value at each beam, Psi_d(k_b) = norm * exp(-i chi_d(k_b))."""
        return self.norm * torch.exp(-1j * (self.coefficients() @ self.basis))

    def forward(self, idx, fixed="none"):
        """Predicted complex far-field Z (n, K, K) for patterns idx.
        fixed = "probe" (gradient only to S), "S" (gradient only to the probe) or "none"."""
        S = self.S.detach() if fixed == "S" else self.S
        psi = self.probe().detach() if fixed == "probe" else self.probe()
        pos = self.pos[idx]
        r = pos.to(torch.float64) * self.setup.dr
        illum = psi[self.didx[idx]] * torch.exp(-2j * math.pi * (r[:, :1] * self.ky + r[:, 1:] * self.kx))
        rows = (pos[:, 0, None] + self.win)[:, :, None]                    # (n, K, 1)
        cols = (pos[:, 1, None] + self.win)[:, None, :]                    # (n, 1, K)
        W = S[:, rows, cols]                                               # (B, n, K, K): window of every beam
        exit_wave = torch.einsum("nb,bnyx->nyx", illum.to(S.dtype), W)
        return torch.fft.fft2(exit_wave, norm="ortho")


def vacuum_smatrix(setup, device="cpu"):
    """S_b(r) = exp(2 pi i k_b . r): the S-matrix of empty space (starting guess)."""
    ky, kx = setup.beams()
    r = np.arange(setup.Y) * setup.dr
    S = np.exp(2j * np.pi * ky[:, None, None] * r[None, :, None]) * np.exp(2j * np.pi * kx[:, None, None] * r[None, None, :])
    return torch.as_tensor(S.astype(np.complex64), device=device)


def amplitude_loss(Z, a):
    """Eq. 12: sum (|Z| - a)^2."""
    return ((Z.abs() - a) ** 2).sum()


# ------------------------------------------------------------------------------------------------ data
def prepare_data(dps, positions, device="cpu"):
    """dps (D, Ny, Nx, K, K) intensities + positions (Ny, Nx, 2) [S-grid px] ->
    amplitudes (J, K, K), positions (J, 2), dataset_index (J,), with J = D * Ny * Nx."""
    D, Ny, Nx, K, _ = dps.shape
    I = np.asarray(dps, np.float32).reshape(D * Ny * Nx, K, K)
    amp = np.sqrt(np.clip(I, 0, None) / I.sum(axis=(1, 2)).max())
    pos = np.tile(np.asarray(positions).reshape(-1, 2), (D, 1))
    didx = np.repeat(np.arange(D), Ny * Nx)
    return torch.as_tensor(amp, device=device), pos, didx


# ------------------------------------------------------------------------------------------------ optimisation
def fit(model, amp, epochs=40, warmup=8, batch=16, lr_S=0.02, lr_probe=None, mode="simultaneous", probe_steps=5,
        seed=0, log=print):
    """Joint S + probe optimisation by gradient descent (autograd + Adam).  Returns the loss history (per epoch).

    The first `warmup` epochs fit S only (nominal probe).  Afterwards, per `mode`:
      "simultaneous": every minibatch updates S and the probe together from the same gradient (default; no extra
                      cost, many small probe steps).  lr_probe default 0.005.
      "alternating":  Pelz Algorithm 1 - an S pass with the probe fixed, then `probe_steps` full-batch probe steps
                      with S fixed.  lr_probe default 0.05.
    Learning rates are in Adam units: S per complex entry, probe in rad at the aperture edge.  Both follow a cosine
    decay to 5 % over the epochs.
    """
    if mode not in ("simultaneous", "alternating"):
        raise ValueError(mode)
    if lr_probe is None:
        lr_probe = 0.005 if mode == "simultaneous" else 0.05
    gen = torch.Generator().manual_seed(seed)
    J = amp.shape[0]
    a2 = float((amp.double() ** 2).sum())
    opt_S = torch.optim.Adam([model.S], lr=lr_S)
    opt_P = torch.optim.Adam([model.dC], lr=lr_probe)
    train_probe = bool(model.mask.any())
    history = []
    for ep in range(epochs):
        f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * ep / max(1, epochs - 1)))   # cosine decay
        for g in opt_S.param_groups:
            g["lr"] = lr_S * f
        for g in opt_P.param_groups:
            g["lr"] = lr_probe * f
        joint_now = train_probe and ep >= warmup

        # pass over all patterns in random minibatches
        total = 0.0
        for idx in torch.randperm(J, generator=gen).to(amp.device).split(batch):
            both = joint_now and mode == "simultaneous"
            opt_S.zero_grad()
            opt_P.zero_grad()
            loss = amplitude_loss(model(idx, fixed="none" if both else "probe"), amp[idx])
            loss.backward()
            opt_S.step()
            if both:
                opt_P.step()
            total += float(loss.detach())
        msg = f"epoch {ep + 1:3d}  loss {total / a2:.6f}"

        # alternating mode: full-batch probe steps with S fixed
        if joint_now and mode == "alternating":
            for _ in range(probe_steps):
                opt_P.zero_grad()
                for idx in torch.arange(J, device=amp.device).split(batch):
                    (amplitude_loss(model(idx, fixed="S"), amp[idx]) / a2).backward()
                opt_P.step()
        if joint_now:
            dC = (model.coefficients() - model.nominal).detach().cpu().numpy()
            msg += "  dC [A] " + np.array2string(dC[:, model.mask.any(0).cpu().numpy()], precision=2,
                                                 suppress_small=True).replace("\n", "")
        history.append(total / a2)
        log(msg)
    return history


# ------------------------------------------------------------------------------------------------ refocusing
def refocus(S, setup, depths, device=None):
    """Complex exit-wave sections of the S-matrix at the given depths [A]: (Z, Y, Y).

    Each beam is demodulated (its carrier exp(2 pi i k_b r) removed), propagated by z with the Fresnel propagator of
    its tilted plane wave, exp(-i pi lam z (|q + k_b|^2 - |k_b|^2)), and all beams are summed coherently.  z is measured
    from the entrance surface of the specimen (depth into the sample).
    """
    S = torch.as_tensor(S.detach() if torch.is_tensor(S) else S, device=device)
    dev = S.device
    ky, kx = (torch.as_tensor(v, dtype=torch.float64, device=dev) for v in setup.beams())
    r = torch.arange(setup.Y, dtype=torch.float64, device=dev) * setup.dr
    demod = (torch.exp(-2j * math.pi * ky[:, None, None] * r[None, :, None])
             * torch.exp(-2j * math.pi * kx[:, None, None] * r[None, None, :])).to(S.dtype)
    Sq = torch.fft.fft2(S * demod)                                       # (B, Y, Y) spectrum of each demodulated beam
    q = torch.as_tensor(np.fft.fftfreq(setup.Y, setup.dr), dtype=torch.float64, device=dev)
    out = torch.empty((len(depths), setup.Y, setup.Y), dtype=S.dtype, device=dev)
    for i, z in enumerate(depths):
        quad = torch.exp(-1j * math.pi * setup.lam * z * (q[:, None] ** 2 + q[None, :] ** 2))
        py = torch.exp(-2j * math.pi * setup.lam * z * q[None, :] * ky[:, None])      # (B, Y) cross term q . k_b
        px = torch.exp(-2j * math.pi * setup.lam * z * q[None, :] * kx[:, None])
        acc = torch.einsum("byx,by,bx->yx", Sq, py.to(S.dtype), px.to(S.dtype))
        out[i] = torch.fft.ifft2(acc * quad.to(S.dtype))
    return out.cpu().numpy()


# ------------------------------------------------------------------------------------------------ depth estimation
FOCUS_METRICS = {
    "intensity_var": lambda w: (np.abs(w) ** 2).std(axis=(1, 2)) / (np.abs(w) ** 2).mean(axis=(1, 2)),   # intensity contrast
    "phase_var": lambda w: np.angle(w).reshape(len(w), -1).var(axis=1),                                   # phase contrast
    "gradient": lambda w: (np.abs(np.diff(np.abs(w), axis=1)).mean(axis=(1, 2))
                           + np.abs(np.diff(np.abs(w), axis=2)).mean(axis=(1, 2))),                       # amplitude sharpness
}


def estimate_depths(S, setup, depths, metric="intensity_var", n_peaks=5, min_sep=None, roi=None):
    """Candidate depths [A] where the refocused S-matrix is sharpest.  Returns (peak_depths, scores, curve).

    Scans `refocus` over `depths`, scores each slice with `metric` (a FOCUS_METRICS key or a callable
    (Z, Y, Y) complex -> (Z,)), and returns the `n_peaks` highest local maxima at least `min_sep` apart (default 5 steps),
    best first.  `roi` = (y0, y1, x0, x1) restricts the score to part of the field (e.g. to avoid edge artefacts).
    This is a heuristic: linear refocusing, one reconstruction, no uncertainty.  Absolute z is tied to the reference
    dataset's calibration.  Compare peaks across several fits (different starts) to judge how stable they are.
    """
    depths = np.asarray(depths, float)
    vol = refocus(S, setup, depths)
    if roi is not None:
        y0, y1, x0, x1 = roi
        vol = vol[:, y0:y1, x0:x1]
    fn = FOCUS_METRICS[metric] if isinstance(metric, str) else metric
    curve = np.asarray(fn(vol), float)
    gap = int(min_sep / abs(depths[1] - depths[0])) if min_sep else 5
    is_max = np.r_[False, (curve[1:-1] > curve[:-2]) & (curve[1:-1] >= curve[2:]), False]
    peaks = []
    for i in np.flatnonzero(is_max)[np.argsort(-curve[is_max])]:
        if all(abs(i - p) >= gap for p in peaks):
            peaks.append(i)
        if len(peaks) == n_peaks:
            break
    return depths[peaks], curve[peaks], curve


# ------------------------------------------------------------------------------------------------ depth convention + fitting
# Depth convention: z = 0 is the specimen entrance surface, z = T the exit surface; C10 < 0 puts the probe focus at
# depth |C10| inside the specimen.  `refocus` assumes an ENTRANCE-referenced S (S_b = exit wave of plane wave b,
# propagated back by T).  For a recovered S this is a gauge, not a property of the data: propagating every beam by the
# same dz changes only the phase of the predicted far field.  It holds approximately because `fit` starts from the
# vacuum S-matrix; measure it with `reference_plane` when a ground truth is available.

def exact_smatrix(setup, layers, layer_z, thickness, reference="entrance", device=None):
    """Ground-truth S (B, Y, Y) of a layered phase specimen: multislice of every beam's plane wave (propagate to each
    layer, multiply by exp(i phase), propagate to the exit), then propagated back by `thickness` if reference ==
    "entrance".  Same physics as demo_simple_joint.simulate."""
    S = vacuum_smatrix(setup, device=device).to(torch.complex128)
    dev = S.device
    q = torch.as_tensor(np.fft.fftfreq(setup.Y, setup.dr), dtype=torch.float64, device=dev)
    q2 = q[:, None] ** 2 + q[None, :] ** 2
    prop = lambda dz, f: torch.fft.ifft2(torch.fft.fft2(f) * torch.exp(-1j * math.pi * setup.lam * dz * q2))
    zs = [0.0] + list(layer_z)
    for i, L in enumerate(layers):
        S = prop(zs[i + 1] - zs[i], S) * torch.exp(1j * torch.as_tensor(np.asarray(L), device=dev))
    S = prop(thickness - zs[-1], S)
    if reference == "entrance":
        S = prop(-thickness, S)
    return S.to(torch.complex64)


def feature_response(volume, weight):
    """Complex depth response (Nz,) of a feature: sum_xy w(x,y) [R(x,y,z) / <R(z)> - 1] / sum w.

    volume: refocused sections (Nz, Y, X) from `refocus`; weight: (Y, X) lateral footprint of the feature (e.g. Gaussian
    blobs at its atoms).  Dividing by the slice mean removes the overall scale and the global phase of each section."""
    vol = np.asarray(volume)
    norm = vol / vol.mean(axis=(1, 2), keepdims=True)
    w = np.asarray(weight, float)
    return ((norm - 1) * w).sum(axis=(1, 2)) / w.sum()


def fit_depth(r, templates, z0_grid, step=1.0):
    """Depth of a feature by complex template fitting.

    r:          (Nz,) measured complex response (`feature_response`) on the same depth grid as the templates
    templates:  (N0, Nz) complex responses of a known feature at depths z0_grid (N0,), e.g. from `exact_smatrix`
    For each candidate z0 (templates interpolated linearly between grid depths, `step` A apart) solve
        min_{a, b complex} || r - a h(z0) - b ||^2
    and return (best z0, z0 candidates, residual curve normalised by ||r - mean r||^2)."""
    z0_grid = np.asarray(z0_grid, float)
    cand = np.arange(z0_grid[0], z0_grid[-1] + 1e-9, step)
    r = np.asarray(r, complex)
    rc = r - r.mean()
    res = np.empty(len(cand))
    for i, z in enumerate(cand):
        j = min(np.searchsorted(z0_grid, z, side="right") - 1, len(z0_grid) - 2)
        t = (z - z0_grid[j]) / (z0_grid[j + 1] - z0_grid[j])
        h = (1 - t) * templates[j] + t * templates[j + 1]
        hc = h - h.mean()                                  # the constant b absorbs the means
        a = np.vdot(hc, rc) / max(np.vdot(hc, hc).real, 1e-30)
        res[i] = np.linalg.norm(rc - a * hc) ** 2
    res /= max(np.linalg.norm(rc) ** 2, 1e-30)
    return float(cand[np.argmin(res)]), cand, res


def reference_plane(S_rec, S_true, setup, dz_grid, crop=None):
    """Propagation distance dz that best maps S_rec onto S_true (per-beam phase aligned), i.e. where the recovered S's
    reference plane sits relative to S_true's.  Returns (best dz, dz_grid, normalised error curve)."""
    Sr = torch.as_tensor(np.asarray(S_rec)).to(torch.complex128)
    St = np.asarray(S_true)
    q = np.fft.fftfreq(setup.Y, setup.dr)
    q2 = torch.as_tensor(q[:, None] ** 2 + q[None, :] ** 2)
    Srq = torch.fft.fft2(Sr)
    cr = crop or (slice(None), slice(None))
    b = St[(slice(None),) + cr]
    err = []
    for dz in dz_grid:
        a = torch.fft.ifft2(Srq * torch.exp(-1j * math.pi * setup.lam * dz * q2)).numpy()[(slice(None),) + cr]
        c = np.einsum("bij,bij->b", np.conj(a), b)
        ph = np.where(np.abs(c) > 0, c / np.maximum(np.abs(c), 1e-30), 1.0)
        err.append(float(np.linalg.norm(a * ph[:, None, None] - b) / np.linalg.norm(b)))
    err = np.array(err)
    return float(np.asarray(dz_grid)[np.argmin(err)]), np.asarray(dz_grid), err


def _template_at(templates, z0_grid, z):
    j = min(np.searchsorted(z0_grid, z, side="right") - 1, len(z0_grid) - 2)
    t = (z - z0_grid[j]) / (z0_grid[j + 1] - z0_grid[j])
    return (1 - t) * templates[j] + t * templates[j + 1]


def fit_depths_joint(R, templates, z0_grid, n_features, min_sep=20.0, refine=10.0, step=2.0):
    """Joint depth fit of `n_features` features whose responses overlap.

    R: (m, Nz) complex responses (one row per lateral footprint; m = 1 for a single column).  Model:
        R_i(z) = sum_k A_ik h(z; z_k) + b_i        (A, b complex, solved by least squares for every depth set)
    Depth sets are searched on the template grid (features >= min_sep apart), then refined +-refine A in `step` A.
    Returns (depths sorted, relative residual = ||R - model||^2 / ||R - mean||^2)."""
    import itertools
    z0_grid = np.asarray(z0_grid, float)
    R = np.atleast_2d(np.asarray(R, complex))
    Rc = (R - R.mean(axis=1, keepdims=True)).T                      # (Nz, m)
    norm = max(np.linalg.norm(Rc) ** 2, 1e-30)

    def cost(zs):
        H = np.array([_template_at(templates, z0_grid, z) for z in zs]).T
        H = H - H.mean(axis=0)
        A = np.linalg.lstsq(H, Rc, rcond=None)[0]
        return np.linalg.norm(Rc - H @ A) ** 2 / norm

    ok = lambda zs: all(b - a >= min_sep for a, b in zip(zs, zs[1:]))
    best = min((cost(c), c) for c in itertools.combinations(z0_grid, n_features) if ok(c))[1]
    fine = [np.arange(max(z0_grid[0], z - refine), min(z0_grid[-1], z + refine) + 1e-9, step) for z in best]
    res, best = min((cost(c), c) for c in itertools.product(*fine) if ok(c))
    return [float(z) for z in best], float(res)


def select_n_features(R, templates, z0_grid, floor, max_features=3, min_gain=0.5, **kw):
    """Fit 1 .. max_features features; add one more only while the current model's residual is above `floor` AND the
    extra feature cuts the residual by at least `min_gain` (fraction).  Returns (selected depths, {n: (depths, res)}).

    floor is the residual a SINGLE feature leaves with these templates (template mismatch + noise).  It must be
    calibrated on single-feature negative controls processed exactly like the data: ~0.002 for exact-S responses vs
    exact-S templates, but ~0.04 for recovered-S responses vs exact-S templates (M0), where a lower floor would split
    one feature into two.  The residual-gain test alone is not safe: fitting template mismatch always lowers it."""
    fits = {}
    for n in range(1, max_features + 1):
        fits[n] = fit_depths_joint(R, templates, z0_grid, n, **kw)
    n_sel = 1
    for n in range(2, max_features + 1):
        if fits[n_sel][1] > floor and fits[n][1] <= (1 - min_gain) * fits[n_sel][1]:
            n_sel = n
        else:
            break
    return fits[n_sel][0], fits
