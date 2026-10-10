"""Joint S-matrix / probe-aberration reconstruction in PyTorch (Pelz et al., PRR 3, 023159 (2021), Sec. II.B-F, Alg. 1).

Mapping to the paper (see README "Joint S-matrix / probe optimisation" for the full table):
  * forward operator A_{k,d}(S, Psi_d)         Eq. 9  -> `forward_chunk`  (crop C_{k,d}, phase factors, coherent sum, F_r)
  * intensities y = |A|^2                       Eq. 10
  * losses D_Amp, D_Pois                        Eq. 11-12 -> `chunk_loss`
  * joint objective, block coordinate descent   Eq. 13-14, Alg. 1 -> `JointReconstructor.run`
  * S gradient (adjoint, Eq. 15 / A3)           Eq. 22 (amplitude), Eq. 20 (Poisson) -> `s_update_sweep`
        The existing (Brown et al.-based) update S_win -= eta * conj(P_j)/|P_j|^2 * F^-1[Z - a Z/|Z|] is Eq. 22 with a
        constant diagonal preconditioner 1/|Psi|^2 (|Psi| is constant on the aperture) and is reproduced exactly.
  * probe gradient                               Eq. 17/19/21 via PyTorch autograd through Eq. 23 -> `probe_update`
  * probe model (aberration polynomials)         Eq. 23 -> probe_aberrations.AberrationProbe
  * ambiguities / initialisation                 Sec. II.F: vacuum S init (S_b = exp(2 pi i h_b.r)), reference dataset fixed

Conventions are those of smatrix_depth_demo.py: window K x K pixels of DR Angstrom, S grid Y = X = M*K, beams on the
window reciprocal grid (k = n / (K*DR)), illumination norm 1/sqrt(B K^2), orthonormal FFTs, amplitudes normalised by the
largest total count, probe positions = window centres in global pixels, S referenced to the entrance surface.
"""
import math, os, time, resource, json, threading
from dataclasses import dataclass, field, asdict
import numpy as np
import torch

from probe_aberrations import AberrationProbe, probe_on_grid, DEFAULT_NAMES


def wavev(E):
    hc, m0c2 = 1.23984193e4, 5.109989461e5
    return math.sqrt(E * (E + 2 * m0c2)) / hc


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS lacks several complex kernels (complex FFT/bmm support varies by version); only used when requested explicitly
    return torch.device("cpu")


def peak_memory_mb(device):
    """Process-lifetime peak (CUDA: max allocated since the last reset; CPU: max RSS, monotone over the process)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: kB -> MB (process peak)


def _current_rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except (OSError, ValueError):
        return float("nan")


class PeakMemory:
    """Per-run peak memory.  CUDA: reset_peak_memory_stats + max_memory_allocated.  CPU: the process max RSS never
    resets, so the current RSS is sampled in a background thread (Linux /proc; NaN elsewhere) during the block."""

    def __init__(self, device, interval=0.05):
        self.device, self.interval, self.peak = torch.device(device), interval, float("nan")

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device); torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self._stop = threading.Event(); self.peak = _current_rss_mb()
            def poll():
                while not self._stop.wait(self.interval):
                    self.peak = max(self.peak, _current_rss_mb())
            self._t = threading.Thread(target=poll, daemon=True); self._t.start()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device); self.peak = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        else:
            self._stop.set(); self._t.join(); self.peak = max(self.peak, _current_rss_mb())
        return False


# ----------------------------------------------------------------------------------------------- geometry
@dataclass
class Geometry:
    ev: float = 3.0e5
    alpha: float = 20e-3
    dr: float = 0.2
    K: int = 64
    M: int = 3
    nscan: int = 25
    step: int = 4

    @property
    def lam(self):
        return 1.0 / wavev(self.ev)

    @property
    def LW(self):
        return self.K * self.dr

    @property
    def Y(self):
        return self.K * self.M

    X = Y

    def beams(self):
        """Window reciprocal-grid indices (ny, nx) inside the aperture, same order as smatrix_depth_demo.beam_list."""
        n = np.fft.fftfreq(self.K, 1.0 / self.K).astype(int)
        ny, nx = np.meshgrid(n, n, indexing="ij")
        m = np.sqrt(ny ** 2 + nx ** 2) / self.LW <= self.alpha / self.lam
        return ny[m], nx[m]

    def scan_coords(self):
        c0 = self.Y // 2 - (self.nscan // 2) * self.step
        a = c0 + self.step * np.arange(self.nscan)
        return np.stack(np.meshgrid(a, a, indexing="ij"), -1)                 # (nscan, nscan, 2) global px

    def crop(self):
        c0 = self.Y // 2 - (self.nscan // 2) * self.step
        return np.s_[c0:c0 + self.nscan * self.step, c0:c0 + self.nscan * self.step]


STANDARD = Geometry()


# ----------------------------------------------------------------------------------------------- data container
@dataclass
class Measurements:
    amp: torch.Tensor          # (J, K, K) float32, sqrt(I / max_total), FFT order
    d_idx: torch.Tensor        # (J,) long, dataset (defocus) index
    pos: torch.Tensor          # (J, 2) long, window centre in global pixels
    max_total: float           # largest total intensity (normalisation)
    dose: float                # electrons per pattern (for Poisson loss in counts)

    @staticmethod
    def from_dps(dps, coords, dose, device="cpu"):
        ndf, ny, nx = dps.shape[:3]
        pats = dps.reshape(ndf * ny * nx, *dps.shape[-2:]).astype(np.float32)
        tot = pats.sum(axis=(1, 2))
        amp = np.sqrt(pats / tot.max()).astype(np.float32)
        d_idx = np.repeat(np.arange(ndf), ny * nx)
        pos = np.tile(coords.reshape(-1, 2), (ndf, 1))
        return Measurements(torch.as_tensor(amp, device=device), torch.as_tensor(d_idx, device=device),
                            torch.as_tensor(pos, device=device), float(tot.max()), float(dose))

    @property
    def J(self):
        return self.amp.shape[0]

    def to(self, device):
        return Measurements(self.amp.to(device), self.d_idx.to(device), self.pos.to(device), self.max_total, self.dose)


# ----------------------------------------------------------------------------------------------- S matrix basics
def vacuum_smatrix(geom, dtype=torch.complex64, device="cpu"):
    """S_b(r) = exp(2 pi i k_b . r) (Pelz Sec. II.F; identical to the NumPy initialisation)."""
    by, bx = geom.beams()
    kby, kbx = by / geom.LW, bx / geom.LW
    iy = np.arange(geom.Y)[None, :] * geom.dr
    S = (np.exp(2j * np.pi * kby[:, None] * iy)[:, :, None] * np.exp(2j * np.pi * kbx[:, None] * iy)[:, None, :])
    return torch.as_tensor(S.astype(np.complex64 if dtype == torch.complex64 else np.complex128), device=device)


def coverage_eta(meas, geom, nb, mu):
    """Step size eta = mu / (B * median window overlap), exactly as the NumPy reconstruction."""
    cover = np.zeros((geom.Y, geom.Y))
    K = geom.K
    for r in meas.pos.cpu().numpy():
        cover[r[0] - K // 2: r[0] + K // 2, r[1] - K // 2: r[1] + K // 2] += 1
    ncov = np.median(cover[cover > 0.5 * cover.max()])
    return mu / (nb * ncov), float(ncov)


def gather_windows(S, pos, K):
    """C_{k,d} S for a chunk of positions: (J, B, K, K) copy of the cropped windows (Eq. 7/9)."""
    h = K // 2
    return torch.stack([S[:, int(y) - h:int(y) + h, int(x) - h:int(x) + h] for y, x in pos.tolist()])


def forward_chunk(W, illum):
    """Eq. 9 for a chunk: Z_j = F[ sum_b P_{j,b} [C_j S]_b ] (orthonormal FFT).  W: (J,B,K,K), illum: (J,B)."""
    J, B, K, _ = W.shape
    z = torch.bmm(illum[:, None, :].to(W.dtype), W.reshape(J, B, K * K)).reshape(J, K, K)
    return torch.fft.fft2(z, norm="ortho")


def chunk_loss(Z, a, kind="amplitude", counts=1.0, eps=1e-12, det_mask=None):
    """Eq. 12 (amplitude: sum (|Z| - a)^2) or Eq. 11 (Poisson NLL in electron counts: sum y - I ln y)."""
    if kind == "amplitude":
        r = (Z.abs() - a) ** 2
    elif kind == "poisson":
        y = Z.real ** 2 + Z.imag ** 2
        r = counts * y - counts * a ** 2 * torch.log(counts * y + eps)
    else:
        raise ValueError(kind)
    if det_mask is not None:
        r = r * det_mask
    return r.sum()


def s_residual(Z, a, kind="amplitude", zero_threshold=1e-12, poisson_floor=1e-6, det_mask=None):
    """Exit-wave residual R used by the S update (S_win -= eta conj(P)/|P|^2 F^-1[R]).

    amplitude (Eq. 22):  R = Z - a Z/|Z|            = dD_Amp/dZ*            (Z/|Z| := 0 where |Z| <= zero_threshold)
    poisson   (Eq. 20):  R = Z (1 - I/y) / 2        = dD_Pois/dZ* / (2 N)   (N = counts; y clamped from below at
                         poisson_floor * max_k y of the same pattern).  The factor 1/2 makes the step equal to the
                         amplitude step near convergence (Z(1 - a^2/|Z|^2) ~ 2 (Z - a Z/|Z|) for |Z| ~ a), so MU means
                         the same for both losses.
    det_mask (K, K) zeroes pixels that are not fitted (e.g. dark field for a bright-field-only data term)."""
    mag = Z.abs()
    if kind == "amplitude":
        Zn = torch.where(mag > zero_threshold, Z / torch.clamp(mag, min=zero_threshold), torch.zeros_like(Z))
        R = Z - a * Zn
    elif kind == "poisson":
        y = mag ** 2
        floor = poisson_floor * y.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-30)
        R = 0.5 * Z * (1 - a ** 2 / torch.maximum(y, floor).clamp(min=1e-30))
    else:
        raise ValueError(kind)
    return R if det_mask is None else R * det_mask


# ----------------------------------------------------------------------------------------------- generalized simulator
def fresnel_np(geom, dz):
    qy = np.fft.fftfreq(geom.Y, geom.dr)[:, None]
    qx = np.fft.fftfreq(geom.Y, geom.dr)[None, :]
    return np.exp(-1j * np.pi * geom.lam * dz * (qy ** 2 + qx ** 2)).astype(np.complex64)


def ops_layers(layers, layer_z, thick, geom):
    """Operation list reproducing smatrix_depth_demo.simulate: propagate to each layer, transmit, propagate to exit."""
    zs = [0.0] + list(layer_z) + [thick]
    ops = []
    for i, L in enumerate(layers):
        ops += [("prop", zs[i + 1] - zs[i]), ("trans", np.exp(1j * L).astype(np.complex64))]
    ops.append(("prop", zs[-1] - zs[-2]))
    return ops


def ops_slices(slices, dz):
    """Operation list reproducing run_poly.simulate_poly: transmit each slice, then propagate dz."""
    ops = []
    for sl in slices:
        ops += [("trans", np.exp(1j * sl).astype(np.complex64)), ("prop", dz)]
    return ops


def _apply_ops(psi, ops, props):
    for kind, v in ops:
        if kind == "prop":
            psi = np.fft.ifft2(np.fft.fft2(psi) * props[v])
        else:
            psi = psi * v
    return psi


def simulate_4dstem(ops, probe_coeffs, geom, dose=0.0, rng=None, names=None):
    """4D-STEM data for arbitrary per-dataset probe aberrations (probe_coeffs: (D, P) Angstrom, Eq. 23 basis).

    With only C10 = df non-zero this reproduces smatrix_depth_demo.simulate exactly (same arithmetic and RNG order).
    dose = 0 returns noiseless intensities.  Returns dps (D, nscan, nscan, K, K) float32 and coords.
    """
    names = list(names or DEFAULT_NAMES)
    props = {v: fresnel_np(geom, v) for k, v in ops if k == "prop"}
    coords = geom.scan_coords()
    K = geom.K
    D = len(probe_coeffs)
    dps = np.zeros((D, geom.nscan, geom.nscan, K, K), np.float32)
    for d in range(D):
        A = probe_on_grid(list(probe_coeffs[d]), names, geom.Y, geom.Y, geom.dr, geom.lam, geom.alpha)
        p = np.fft.ifft2(A)
        p0 = (p / np.sqrt(np.sum(np.abs(p) ** 2))).astype(np.complex64)
        for iy in range(geom.nscan):
            for ix in range(geom.nscan):
                ry, rx = coords[iy, ix]
                psi = _apply_ops(np.roll(p0, (ry, rx), axis=(0, 1)), ops, props)
                win = psi[ry - K // 2: ry + K // 2, rx - K // 2: rx + K // 2]
                I = np.abs(np.fft.fft2(win, norm="ortho")) ** 2
                if dose > 0:
                    I = rng.poisson(I * dose).astype(np.float32) / dose
                dps[d, iy, ix] = I
    return dps, coords


def simulate_from_smatrix(S, probe, geom, chunk=64):
    """Model-consistent ("inverse-crime") intensities |A(S, Psi)|^2 from the S-matrix forward model itself (Eq. 9-10).
    Used only for optimiser identifiability tests; the physical experiments use simulate_4dstem."""
    coords = geom.scan_coords()
    D = probe.n_datasets
    pos = torch.as_tensor(np.tile(coords.reshape(-1, 2), (D, 1)))
    d_idx = torch.as_tensor(np.repeat(np.arange(D), geom.nscan ** 2))
    St = torch.as_tensor(S)
    out = []
    with torch.no_grad():
        il = probe.illumination(d_idx, pos.double() * geom.dr).to(St.dtype)
        for c0 in range(0, len(pos), chunk):
            Z = forward_chunk(gather_windows(St, pos[c0:c0 + chunk], geom.K), il[c0:c0 + chunk])
            out.append((Z.real ** 2 + Z.imag ** 2).float())
    I = torch.cat(out).cpu().numpy().reshape(D, geom.nscan, geom.nscan, geom.K, geom.K)
    return I, coords


def add_poisson(dps_clean, dose, seed):
    rng = np.random.default_rng(seed)
    return (rng.poisson(dps_clean * dose).astype(np.float32) / dose).astype(np.float32)


def true_smatrix(ops, geom, thick, reference="entrance"):
    """Ground-truth S: multislice of each beam's plane wave.  reference="exit" returns the exit waves; "entrance"
    back-propagates them by the thickness (the recovered S is entrance-referenced, see README depth convention)."""
    by, bx = geom.beams()
    props = {v: fresnel_np(geom, v) for k, v in ops if k == "prop"}
    back = fresnel_np(geom, -thick)
    S0 = vacuum_smatrix(geom).numpy()
    out = np.empty_like(S0)
    for b in range(len(by)):
        psi = _apply_ops(S0[b], ops, props)
        out[b] = psi if reference == "exit" else np.fft.ifft2(np.fft.fft2(psi) * back)
    return out


# ----------------------------------------------------------------------------------------------- depth sectioning
def depth_sections_general(S, geom, depths, device=None):
    """Port of smatrix_depth_demo.depth_sections for any geometry (shift beams to origin, Fresnel + paraxial shift,
    coherent sum).  Forward propagation by z from the entrance-referenced S (sign -1).  Returns complex (Z, Y, X)."""
    St = torch.as_tensor(S, device=device)
    dev = St.device
    by, bx = geom.beams()
    M, Y, dr, lam = geom.M, geom.Y, geom.dr, geom.lam
    Sq = torch.fft.fft2(St)
    for b in range(len(by)):
        Sq[b] = torch.roll(Sq[b], (-M * int(by[b]), -M * int(bx[b])), dims=(0, 1))
    q = torch.as_tensor(np.fft.fftfreq(Y, dr), dtype=torch.float64, device=dev)
    kby = torch.as_tensor(by / geom.LW, dtype=torch.float64, device=dev)
    kbx = torch.as_tensor(bx / geom.LW, dtype=torch.float64, device=dev)
    out = torch.zeros((len(depths), Y, Y), dtype=Sq.dtype, device=dev)
    for i, z in enumerate(depths):
        py = torch.exp(-1j * z * 2 * np.pi * lam * q[None, :] * kby[:, None]).to(Sq.dtype)
        px = torch.exp(-1j * z * 2 * np.pi * lam * q[None, :] * kbx[:, None]).to(Sq.dtype)
        acc = torch.einsum("byx,by,bx->yx", Sq, py, px)
        quad = torch.exp(-1j * z * np.pi * lam * (q[:, None] ** 2 + q[None, :] ** 2)).to(Sq.dtype)
        out[i] = torch.fft.ifft2(acc * quad)
    return out.cpu().numpy()


# ----------------------------------------------------------------------------------------------- optimiser
@dataclass
class Schedule:
    mode: str = "joint"          # "joint", "s_only" (fixed probe), "probe_only" (S fixed)
    warmup_s: int = 5            # S-only iterations before probe refinement starts
    cycles: int = 6              # alternating cycles
    s_per_cycle: int = 3         # S sweeps per cycle
    p_per_cycle: int = 1         # probe updates per cycle
    final_s: int = 3             # final refinement S sweeps (reduced step)
    final_p: int = 1             # final refinement probe updates (reduced lr)
    final_factor: float = 0.3    # step / learning-rate reduction in the final refinement
    mu: float = 60.0             # S step (same parameterisation as the NumPy MU; analytic S update, and SGD default lr)
    s_method: str = "autograd"   # S update: "autograd" (torch autograd + optimizer) or "analytic" (hand-written Eq. 22)
    s_optimizer: str = "adam"    # autograd S update: "adam" or "sgd"
    s_lr: float = 0.01           # autograd S learning rate (Adam; set <= 0 with SGD for auto lr); SGD uses eta / (2 mean|P|^2), the analytic step size, if s_lr <= 0
    probe_lr: float = 0.05       # probe learning rate in normalised units (1 = 1 rad at aperture edge)
    probe_optimizer: str = "adam"   # "adam" or "sgd" (sgd = plain gradient step as in Alg. 1)
    grad_clip: float = 1.0       # max norm of the normalised probe gradient (0 = off)
    reg_lambda: float = 0.0      # weight of the quadratic prior on the probe corrections
    loss: str = "amplitude"      # data term for both blocks: "amplitude" (Eq. 12) or "poisson" (Eq. 11)
    chunk: int = 20              # measurements per minibatch (also the stale-S chunk of the S update)
    probe_fraction: float = 1.0  # fraction of measurements used per probe gradient (random subset if < 1)
    zero_threshold: float = 1e-12    # |Z| below this -> Z/|Z| set to 0 (as the NumPy code)
    poisson_floor: float = 1e-6      # Poisson residual: y clamped at this fraction of the pattern's max predicted y
    det_mask: str = "full"       # "full" detector or "bf" (bright-field disk only, loss and residual masked)
    seed: int = 0

    def n_s_iters(self):
        if self.mode == "probe_only":
            return 0
        return self.warmup_s + self.cycles * self.s_per_cycle + self.final_s


class JointReconstructor:
    def __init__(self, meas, geom, probe, schedule, S_init=None, device="cpu", log=print):
        self.device, self.log = pick_device(device) if isinstance(device, str) else torch.device(device), log
        self.geom, self.sch = geom, schedule
        self.meas = meas.to(self.device)                       # one device for data, probe and S
        self.probe = probe.to(self.device)
        self.S = (vacuum_smatrix(geom, device=self.device) if S_init is None
                  else torch.as_tensor(S_init, dtype=torch.complex64, device=self.device).clone())
        self.S_opt = None                                       # lazily built optimizer for the autograd S update
        meas = self.meas
        self.nb = self.S.shape[0]
        self.eta, self.ncov = coverage_eta(meas, geom, self.nb, schedule.mu)
        self.pos_A = meas.pos.to(torch.float64) * geom.dr
        self.pos_list = meas.pos.cpu().tolist()                 # host copy for the window loop (no device syncs)
        self.sum_a2 = float((meas.amp.double() ** 2).sum())
        self.det_mask = None
        if schedule.det_mask == "bf":
            q = np.fft.fftfreq(geom.K, geom.dr)
            m = (q[:, None] ** 2 + q[None, :] ** 2) <= (geom.alpha / geom.lam) ** 2
            self.det_mask = torch.as_tensor(m.astype(np.float32), device=self.device)
        self.history = dict(s_iter=[], s_loss=[], s_time=[], p_iter=[], p_loss=[], p_time=[], p_grad=[], coeffs=[],
                            eval=[])
        self.coeff_snap(0)

    # -- helpers
    def illumination(self, idx, grad=False):
        with torch.set_grad_enabled(grad):
            return self.probe.illumination(self.meas.d_idx[idx], self.pos_A[idx]).to(self.S.dtype)

    def coeff_snap(self, it):
        self.history["coeffs"].append((it, self.probe.coefficients().detach().cpu().numpy().tolist()))

    # -- Stage A: S update (normalised amplitude flow == preconditioned Eq. 22; Poisson residual = Eq. 20)
    @torch.no_grad()
    def s_update_sweep(self, eta):
        S, K, h = self.S, self.geom.K, self.geom.K // 2
        J, tot = self.meas.J, 0.0
        illum_all = self.illumination(slice(None))
        thr = self.sch.zero_threshold
        for c0 in range(0, J, self.sch.chunk):
            idx = slice(c0, min(c0 + self.sch.chunk, J))
            il, pos, a = illum_all[idx], self.meas.pos[idx], self.meas.amp[idx]
            Z = forward_chunk(gather_windows(S, pos, K), il)
            tot += float(((Z.abs() - a) ** 2).sum())        # stale running estimate (S before this chunk's update)
            resid = s_residual(Z, a, self.sch.loss, thr, self.sch.poisson_floor, self.det_mask)
            R = torch.fft.ifft2(resid, norm="ortho").to(S.dtype)
            for j, (y0, x0) in enumerate(self.pos_list[idx]):
                coef = (eta * torch.conj(il[j]) / torch.abs(il[j]) ** 2).to(S.dtype)
                S[:, y0 - h:y0 + h, x0 - h:x0 + h] -= coef[:, None, None] * R[j][None]
        return tot / self.sum_a2

    # -- Stage A (autograd): same minibatch structure as above, but the gradient comes from torch autograd on the
    #    Eq. 9 forward model + data loss and the step is taken by a torch.optim optimizer on S itself.
    def make_s_opt(self):
        sch = self.sch
        self.S.requires_grad_(True)
        if sch.s_optimizer == "adam":
            return torch.optim.Adam([self.S], lr=sch.s_lr)
        if sch.s_lr > 0:
            lr = sch.s_lr
        else:                                                   # torch grad = 2 dL/dS*; Eq. 22 also divides by |P|^2
            lr = 0.5 * self.eta / float((self.illumination(slice(None)).abs() ** 2).mean())
        return torch.optim.SGD([self.S], lr=lr)

    def s_update_sweep_autograd(self, scale=1.0):
        if self.S_opt is None:
            self.S_opt = self.make_s_opt()
        opt, K, J = self.S_opt, self.geom.K, self.meas.J
        base = [g.setdefault("base_lr", g["lr"]) for g in opt.param_groups]
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        counts = self.meas.dose * self.meas.max_total
        with torch.no_grad():
            illum_all = self.illumination(slice(None))
        tot = 0.0
        for c0 in range(0, J, self.sch.chunk):
            idx = slice(c0, min(c0 + self.sch.chunk, J))
            opt.zero_grad(set_to_none=True)
            Z = forward_chunk(gather_windows(self.S, self.meas.pos[idx], K), illum_all[idx])
            a = self.meas.amp[idx]
            loss = chunk_loss(Z, a, self.sch.loss, counts, det_mask=self.det_mask)
            loss.backward()
            tot += float(((Z.detach().abs() - a) ** 2).sum())   # stale running estimate, as in the analytic sweep
            opt.step()
        return tot / self.sum_a2

    # -- Stage B: probe update (S detached; autograd through Eq. 23)
    def probe_loss_backward(self, subset=None):
        K, J = self.geom.K, self.meas.J
        order = torch.arange(J, device=self.device) if subset is None else subset.to(self.device)
        counts = self.meas.dose * self.meas.max_total
        norm = self.sum_a2 * (counts if self.sch.loss == "poisson" else 1.0)
        total = 0.0
        for c0 in range(0, len(order), self.sch.chunk):
            idx = order[c0:c0 + self.sch.chunk]
            with torch.no_grad():
                W = gather_windows(self.S, self.meas.pos[idx], K)          # constant: no gradient w.r.t. S
            il = self.illumination(idx, grad=True)
            Z = forward_chunk(W, il)
            l = chunk_loss(Z, self.meas.amp[idx], self.sch.loss, counts, det_mask=self.det_mask) / norm
            l.backward()
            total += float(l.detach())
            del W, Z, l
        if self.sch.reg_lambda > 0:
            r = self.sch.reg_lambda * self.probe.regularizer()
            r.backward()
            total += float(r)
        return total

    def probe_step(self, opt, it):
        t0 = time.time()
        opt.zero_grad()
        subset = None
        if self.sch.probe_fraction < 1.0:
            g = torch.Generator().manual_seed(self.sch.seed * 100003 + it)
            n = max(1, int(self.sch.probe_fraction * self.meas.J))
            subset = torch.randperm(self.meas.J, generator=g)[:n]
        loss = self.probe_loss_backward(subset)
        gnorm = float(self.probe.theta.grad.norm())
        if self.sch.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([self.probe.theta], self.sch.grad_clip)
        opt.step()
        h = self.history
        h["p_iter"].append(it); h["p_loss"].append(loss); h["p_time"].append(time.time() - t0); h["p_grad"].append(gnorm)
        self.coeff_snap(it)
        return loss

    @torch.no_grad()
    def evaluate_loss(self):
        """Loss of the CURRENT (S, probe) over the whole dataset, after all updates (the per-sweep s_loss is a running
        estimate taken before each chunk's update).  Returns relative amplitude loss on the full detector and on the
        fitted pixels (= full unless det_mask), plus the Poisson NLL per pattern when loss == "poisson"."""
        K, J = self.geom.K, self.meas.J
        illum = self.illumination(slice(None))
        full = fitted = pois = 0.0
        counts = self.meas.dose * self.meas.max_total
        a2_fit = float(((self.meas.amp.double() ** 2) * (1 if self.det_mask is None else self.det_mask)).sum())
        for c0 in range(0, J, self.sch.chunk):
            idx = slice(c0, min(c0 + self.sch.chunk, J))
            Z = forward_chunk(gather_windows(self.S, self.meas.pos[idx], K), illum[idx])
            a = self.meas.amp[idx]
            r = (Z.abs() - a) ** 2
            full += float(r.sum())
            fitted += float(r.sum() if self.det_mask is None else (r * self.det_mask).sum())
            if self.sch.loss == "poisson":
                pois += float(chunk_loss(Z, a, "poisson", counts, det_mask=self.det_mask))
        out = dict(rel_amplitude_loss=full / self.sum_a2, rel_amplitude_loss_fitted=fitted / a2_fit)
        if self.sch.loss == "poisson":
            out["poisson_nll_per_pattern"] = pois / J
        return out

    def make_opt(self, lr):
        if self.sch.probe_optimizer == "adam":
            return torch.optim.Adam([self.probe.theta], lr=lr)
        return torch.optim.SGD([self.probe.theta], lr=lr)

    def _s(self, it, eta):
        t0 = time.time()
        if self.sch.s_method == "autograd":
            loss = self.s_update_sweep_autograd(eta / self.eta)  # eta carries only the final-refinement factor here
        else:
            loss = self.s_update_sweep(eta)
        h = self.history
        h["s_iter"].append(it); h["s_loss"].append(loss); h["s_time"].append(time.time() - t0)
        self.log(f"  S iter {it:3d}  rel. amplitude loss {loss:.5f}  ({h['s_time'][-1]:.1f}s)")

    def run(self):
        """Algorithm 1 (block coordinate descent) with warm-up, alternating cycles and final refinement."""
        with PeakMemory(self.device) as pm:
            self._run()
        self.peak_mb = pm.peak
        return self.S.detach()

    def _run(self):
        sch, t_start = self.sch, time.time()
        torch.manual_seed(sch.seed)
        opt = self.make_opt(sch.probe_lr)
        it = 0
        if sch.mode == "probe_only":
            for c in range(sch.cycles * sch.p_per_cycle):
                l = self.probe_step(opt, c + 1)
                self.log(f"  probe step {c + 1}: loss {l:.5f}")
        else:
            for _ in range(sch.warmup_s):
                it += 1; self._s(it, self.eta)
            for c in range(sch.cycles):
                for _ in range(sch.s_per_cycle):
                    it += 1; self._s(it, self.eta)
                if sch.mode == "joint":
                    for _ in range(sch.p_per_cycle):
                        l = self.probe_step(opt, it)
                        self.log(f"  probe update after S iter {it}: loss {l:.5f}  dC [A] = "
                                 + np.array2string(self.probe.coefficients().detach().cpu().numpy()
                                                   - self.probe.nominal.cpu().numpy(), precision=1, suppress_small=True).replace("\n", ""))
            for g in opt.param_groups:
                g["lr"] = sch.probe_lr * sch.final_factor
            for _ in range(sch.final_s):
                it += 1; self._s(it, self.eta * sch.final_factor)
            if sch.mode == "joint":
                for _ in range(sch.final_p):
                    self.probe_step(opt, it)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.runtime = time.time() - t_start
        self.final_eval = self.evaluate_loss()                  # loss of the returned (S, probe), after the last update
        self.history["eval"].append((it, self.final_eval))

    def summary(self):
        return dict(runtime_s=self.runtime, peak_memory_mb=self.peak_mb, device=str(self.device), eta=self.eta, ncov=self.ncov,
                    final_loss=self.final_eval["rel_amplitude_loss"], final_eval=self.final_eval,
                    last_sweep_loss=self.history["s_loss"][-1] if self.history["s_loss"] else None,
                    schedule=asdict(self.sch), coefficients=self.probe.as_dicts())


# ----------------------------------------------------------------------------------------------- metrics
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def probe_error(psi_est, psi_true):
    """Relative probe error after optimal global-phase alignment, per dataset: (D,)."""
    out = []
    for e, t in zip(psi_est, psi_true):
        c = np.vdot(e, t)
        ph = c / abs(c) if abs(c) > 0 else 1.0
        out.append(float(np.linalg.norm(e * ph - t) / np.linalg.norm(t)))
    return np.array(out)


def smatrix_error(S_rec, S_true_entr, geom, z_out=None):
    """NRMSE of S over the scanned region after gauge alignment: per-beam phase and (optionally searched) output-plane
    propagation distance z_out (the output plane of S is not fixed by intensities, Sec. II.F / README)."""
    cr = geom.crop()
    zs = [0.0] if z_out is None else z_out
    best = None
    Sq = np.fft.fft2(S_true_entr)
    q = np.fft.fftfreq(geom.Y, geom.dr)
    q2 = q[:, None] ** 2 + q[None, :] ** 2
    for z in zs:
        St = np.fft.ifft2(Sq * np.exp(-1j * np.pi * geom.lam * z * q2)) if z != 0 else S_true_entr
        a, b = S_rec[(slice(None),) + cr], St[(slice(None),) + cr]
        c = np.einsum("bij,bij->b", np.conj(a), b)
        ph = np.where(np.abs(c) > 0, c / np.maximum(np.abs(c), 1e-30), 1.0)
        e = float(np.linalg.norm(a * ph[:, None, None] - b) / np.linalg.norm(b))
        if best is None or e < best[0]:
            best = (e, float(z))
    return dict(nrmse=best[0], z_out=best[1])


def axial_fwhm(depths, a):
    """Full width at half maximum of the contiguous peak containing the maximum of a (baseline = min of a), with
    linear interpolation of both half-maximum crossings.  Returns (width, open); open = True when the peak runs into
    the end of the depth range on either side, in which case the width is only a lower bound."""
    depths, a = np.asarray(depths, float), np.asarray(a, float)
    ib = int(np.argmax(a)); half = a.min() + 0.5 * (a[ib] - a.min())
    lo = ib
    while lo > 0 and a[lo - 1] >= half:
        lo -= 1
    hi = ib
    while hi < len(a) - 1 and a[hi + 1] >= half:
        hi += 1
    def cross(i_in, i_out):                     # interpolate between an inside (>= half) and an outside sample
        return depths[i_in] + (depths[i_out] - depths[i_in]) * (a[i_in] - half) / (a[i_in] - a[i_out])
    z_lo = cross(lo, lo - 1) if lo > 0 else depths[0]
    z_hi = cross(hi, hi + 1) if hi < len(a) - 1 else depths[-1]
    return float(z_hi - z_lo), bool(lo == 0 or hi == len(a) - 1)


def layer_metrics(phase_stack, layers_c, layer_z, depths):
    """Depth localisation for layered samples: correlation with each true layer vs depth (Terzoudis-Lumsden style
    axial response), found depth (centre of the contiguous |r| plateau), error, axial FWHM (contiguous main peak,
    interpolated; fwhm_open flags a lower bound), cross-talk."""
    C = np.array([[corr(phase_stack[i], L) for L in layers_c] for i in range(len(depths))])
    rows = []
    for j, z in enumerate(layer_z):
        a = np.abs(C[:, j]); ib = int(np.argmax(a)); thr = a[ib] - 0.03
        lo = hi = ib
        while lo > 0 and a[lo - 1] >= thr: lo -= 1
        while hi < len(a) - 1 and a[hi + 1] >= thr: hi += 1
        w = a[lo:hi + 1]; zc = float(np.sum(depths[lo:hi + 1] * w) / np.sum(w))
        fwhm, fwhm_open = axial_fwhm(depths, a)
        iz = int(np.argmin(np.abs(depths - z)))
        others = [abs(C[iz, k]) for k in range(len(layer_z)) if k != j]
        xt = float(np.mean(others) / max(abs(C[iz, j]), 1e-9)) if others else float("nan")
        rows.append(dict(true_z=float(z), found_z=zc, err=zc - float(z), r=float(C[ib, j]), r_at_true=float(C[iz, j]),
                         fwhm=fwhm, fwhm_open=fwhm_open, crosstalk=xt))
    err = np.array([r["err"] for r in rows])
    return dict(layers=rows, mean_abs_depth_err=float(np.abs(err).mean()), max_abs_depth_err=float(np.abs(err).max()),
                mean_r=float(np.mean([r["r"] for r in rows])), mean_fwhm=float(np.mean([r["fwhm"] for r in rows])),
                mean_crosstalk=float(np.mean([r["crosstalk"] for r in rows])),
                n_resolved=int(sum(abs(r["err"]) <= 25 and abs(r["r"]) > 0.5 for r in rows)), curves=C.tolist())


def slab_metrics(phase_stack, slabs_c, depths):
    """Polycrystal: correlation of each section with the true slab (+-20 A) at every depth."""
    n = len(depths)
    Mx = np.array([[corr(phase_stack[i], slabs_c[j]) for j in range(n)] for i in range(n)])
    A = np.abs(Mx); diag = np.diag(Mx)
    loc = np.array([abs(depths[np.argmax(A[i])] - depths[i]) for i in range(n)])
    sel = [A[i, i] - np.mean([A[i, j] for j in range(n) if abs(depths[j] - depths[i]) >= 60]) for i in range(n)]
    return dict(mean_diag_r=float(diag.mean()), mean_depth_err=float(loc.mean()), frac_within_10A=float((loc <= 10).mean()),
                selectivity=float(np.mean(sel)), n_inverted=int((diag < 0).sum()), matrix=Mx.tolist())
