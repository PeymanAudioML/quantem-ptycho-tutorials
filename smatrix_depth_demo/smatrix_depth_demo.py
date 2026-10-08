"""
S-matrix recovery + depth sectioning from multi-defocus 4D-STEM (CPU / NumPy).

Re-implementation of the two algorithms in Brown et al., arXiv:2011.07652
(ancillary code: GradDS/AmpflowS.py and GradDS/Optical_sectioning.py):

  1. Amplitude-flow gradient descent for the scattering matrix S_b(r) from
     4D-STEM data recorded at several probe defocus values.
  2. Depth sectioning: shift each beam to the origin, apply a Fresnel
     propagator + paraxial shift, and coherently sum the beams.

The original code needs CUDA/CuPy and py_multislice, so here the 4D data are
generated with a small built-in multislice simulation of a 3-layer test sample.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- parameters
EV = 3.0e5                 # accelerating voltage (eV)
ALPHA = 20e-3              # convergence semi-angle (rad)
DR = 0.2                   # real-space pixel (A)
K = 64                     # diffraction pattern size (pixels) -> window
M = 3                      # S grid = M * K
Y = X = K * M              # global (S-matrix) grid, 38.4 A
LW = K * DR                # window size (A)
STEP = 4                   # scan step in pixels (0.8 A)
NSCAN = 25                 # NSCAN x NSCAN scan
DEFOCI = [-200.0, -100.0, 0.0]   # A ; focus at z = -df below entrance surface
THICK = 300.0              # A
LAYER_Z = [30.0, 150.0, 270.0]
DOSE = float(os.environ.get("DOSE", 2e5))   # electrons per diffraction pattern
NITER = int(os.environ.get("NITER", 30))
MU = float(os.environ.get("MU", 1.5))
SEED = 0


def wavev(E):
    hc, m0c2 = 1.23984193e4, 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc


KW = wavev(EV)             # 1/lambda
LAM = 1.0 / KW             # lambda (A)


# -------------------------------------------------------------- test sample
def atoms_to_phase(pos, amp, sigma=0.45):
    """Sum of gaussian atoms on the global periodic grid (radians)."""
    y = (np.arange(Y) * DR)[:, None]
    x = (np.arange(X) * DR)[None, :]
    ph = np.zeros((Y, X))
    L = Y * DR
    for (py, px), a in zip(pos, amp):
        dy = (y - py + L / 2) % L - L / 2
        dx = (x - px + L / 2) % L - L / 2
        ph += a * np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
    return ph


def build_layers():
    c = Y * DR / 2
    # layer 1 (z=30): ring of 14 atoms
    ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    ring = [(c + 6.0 * np.sin(a), c + 6.0 * np.cos(a)) for a in ang] + [(c, c)]
    # layer 2 (z=150): 5x5 square lattice
    sq = [(c + 2.6 * i, c + 2.6 * j) for i in range(-2, 3) for j in range(-2, 3)]
    # layer 3 (z=270): triangle outline
    tri = []
    v = np.array([[-6.5, -6.0], [6.5, -6.0], [0.0, 6.5]])
    for a in range(3):
        for t in np.linspace(0, 1, 6, endpoint=False):
            p = v[a] * (1 - t) + v[(a + 1) % 3] * t
            tri.append((c + p[0], c + p[1]))
    return [atoms_to_phase(ring, [1.0] * len(ring)),
            atoms_to_phase(sq, [1.0] * len(sq)),
            atoms_to_phase(tri, [1.0] * len(tri))]


# -------------------------------------------------------------- simulation
def fresnel(dz):
    qy = np.fft.fftfreq(Y, DR)[:, None]
    qx = np.fft.fftfreq(X, DR)[None, :]
    return np.exp(-1j * np.pi * LAM * dz * (qy ** 2 + qx ** 2)).astype(np.complex64)


def aperture():
    qy = np.fft.fftfreq(Y, DR)[:, None]
    qx = np.fft.fftfreq(X, DR)[None, :]
    return (qy ** 2 + qx ** 2) <= (ALPHA * KW) ** 2


def probe_in(df):
    qy = np.fft.fftfreq(Y, DR)[:, None]
    qx = np.fft.fftfreq(X, DR)[None, :]
    k2 = qy ** 2 + qx ** 2
    A = aperture() * np.exp(-1j * np.pi * LAM * k2 * df)
    p = np.fft.ifft2(A)
    return (p / np.sqrt(np.sum(np.abs(p) ** 2))).astype(np.complex64)


def simulate(layers, rng):
    zs = [0.0] + LAYER_Z + [THICK]
    props = [fresnel(zs[i + 1] - zs[i]) for i in range(len(zs) - 1)]
    trans = [np.exp(1j * l).astype(np.complex64) for l in layers]
    ndf = len(DEFOCI)
    dps = np.zeros((ndf, NSCAN, NSCAN, K, K), np.float32)
    c0 = Y // 2 - (NSCAN // 2) * STEP
    coords = np.zeros((NSCAN, NSCAN, 2), int)
    for idf, df in enumerate(DEFOCI):
        p0 = probe_in(df)
        for iy in range(NSCAN):
            for ix in range(NSCAN):
                ry, rx = c0 + iy * STEP, c0 + ix * STEP
                coords[iy, ix] = (ry, rx)
                psi = np.roll(p0, (ry, rx), axis=(0, 1))
                for t, P in zip(trans, props):
                    psi = np.fft.ifft2(np.fft.fft2(psi) * P)   # propagate
                    psi = psi * t
                # final propagation (270 A -> 300 A) to the exit plane
                psi = np.fft.ifft2(np.fft.fft2(psi) * props[-1])
                win = psi[ry - K // 2: ry + K // 2, rx - K // 2: rx + K // 2]
                I = np.abs(np.fft.fft2(win, norm="ortho")) ** 2
                if DOSE > 0:
                    I = rng.poisson(I * DOSE).astype(np.float32) / DOSE
                dps[idf, iy, ix] = I
    return dps, coords


# --------------------------------------------------- amplitude-flow recovery
def beam_list():
    n = np.fft.fftfreq(K, 1.0 / K).astype(int)           # window k indices
    ny, nx = np.meshgrid(n, n, indexing="ij")
    kb = np.sqrt(ny ** 2 + nx ** 2) / LW
    m = kb <= ALPHA * KW
    return ny[m], nx[m]


def reconstruct(dps, coords, log):
    ndf = dps.shape[0]
    by, bx = beam_list()
    nb = len(by)
    kby, kbx = by / LW, bx / LW
    k2 = kby ** 2 + kbx ** 2
    log(f"beams in aperture: {nb}   S-matrix: {nb} x {Y} x {X}")

    # identity (vacuum) initialisation: S_b(r) = exp(2 pi i k_b . r)
    iy = np.arange(Y)[None, :] * DR
    Sm = (np.exp(2j * np.pi * kby[:, None] * iy)[:, :, None]
          * np.exp(2j * np.pi * kbx[:, None] * iy)[:, None, :]).astype(np.complex64)

    pats, cc, ids = [], [], []
    for idf in range(ndf):
        for a in range(dps.shape[1]):
            for b in range(dps.shape[2]):
                pats.append(dps[idf, a, b]); cc.append(coords[a, b]); ids.append(idf)
    pats = np.asarray(pats); cc = np.asarray(cc); ids = np.asarray(ids)
    nscan = len(pats)

    tot = pats.sum(axis=(1, 2))
    amp = np.sqrt(pats / tot.max()).astype(np.float32)     # amplitude data

    norm = 1.0 / np.sqrt(nb * K * K)
    illum = np.zeros((nscan, nb), np.complex64)
    for i in range(nscan):
        chi = -np.pi * LAM * k2 * DEFOCI[ids[i]]
        sh = -2 * np.pi * (kby * cc[i, 0] + kbx * cc[i, 1]) * DR
        illum[i] = norm * np.exp(1j * (chi + sh))

    cover = np.zeros((Y, X))
    for r in cc:
        cover[r[0] - K // 2: r[0] + K // 2, r[1] - K // 2: r[1] + K // 2] += 1
    ncov = np.median(cover[cover > 0.5 * cover.max()])
    eta = MU / (nb * ncov)
    log(f"patterns: {nscan}, median window overlap: {ncov:.0f}, step eta={eta:.2e}")

    nchunk = 20
    loss = []
    for it in range(NITER):
        t0 = time.time()
        tot_err = 0.0
        for c0 in range(0, nscan, nchunk):
            idx = range(c0, min(c0 + nchunk, nscan))
            z = np.empty((len(idx), K, K), np.complex64)
            for j, i in enumerate(idx):
                y0, x0 = cc[i, 0] - K // 2, cc[i, 1] - K // 2
                win = Sm[:, y0:y0 + K, x0:x0 + K]
                z[j] = np.tensordot(illum[i], win, axes=(0, 0))
            Z = np.fft.fft2(z, axes=(-2, -1), norm="ortho")
            a = amp[list(idx)]
            mag = np.abs(Z)
            tot_err += np.sum((mag - a) ** 2)
            Zn = np.where(mag > 1e-12, Z / np.maximum(mag, 1e-12), 0)
            R = np.fft.ifft2(Z - a * Zn, axes=(-2, -1), norm="ortho").astype(np.complex64)
            for j, i in enumerate(idx):
                y0, x0 = cc[i, 0] - K // 2, cc[i, 1] - K // 2
                coef = (eta * np.conj(illum[i]) / np.abs(illum[i]) ** 2).astype(np.complex64)
                Sm[:, y0:y0 + K, x0:x0 + K] -= coef[:, None, None] * R[j][None]
        loss.append(tot_err / np.sum(amp ** 2))
        log(f"iter {it + 1:3d}/{NITER}  rel. amplitude loss {loss[-1]:.5f}  ({time.time() - t0:.1f}s)")
    return Sm, (by, bx), np.asarray(loss)


# --------------------------------------------------------- depth sectioning
def depth_sections(Sm, beams, depths):
    """Refocus the recovered S-matrix to trial depths (cf. depth_section_reconstruction).

    Each beam is shifted to the origin in q-space, then given a Fresnel
    propagator plus the paraxial beam shift, and all beams are summed coherently.

    Depth convention: the illumination (known probe phases) fixes the *input*
    plane of S at the entrance surface, while free-space propagation of the
    output plane is invisible in diffraction intensities.  The recovered S is
    therefore referenced to the entrance surface, and a layer at depth z comes
    into focus after propagating forward by z (sign -1 below).
    """
    by, bx = beams
    nb = len(by)
    Sq = np.fft.fft2(Sm, axes=(-2, -1))
    for b in range(nb):                        # shift each beam to the origin
        Sq[b] = np.roll(Sq[b], (-M * by[b], -M * bx[b]), axis=(0, 1))
    qy = np.fft.fftfreq(Y, DR)
    qx = np.fft.fftfreq(X, DR)
    kby, kbx = by / LW, bx / LW
    out = np.zeros((len(depths), Y, X), np.complex64)
    for i, z in enumerate(depths):
        py = np.exp(-1j * z * 2 * np.pi * LAM * qy[None, :] * kby[:, None]).astype(np.complex64)
        px = np.exp(-1j * z * 2 * np.pi * LAM * qx[None, :] * kbx[:, None]).astype(np.complex64)
        acc = np.einsum("byx,by,bx->yx", Sq, py, px, optimize=True)
        quad = np.exp(-1j * z * np.pi * LAM * (qy[:, None] ** 2 + qx[None, :] ** 2))
        out[i] = np.fft.ifft2(acc * quad)
    return out


def main():
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    rng = np.random.default_rng(SEED)
    log(f"E={EV/1e3:.0f} kV  lambda={LAM*100:.3f} pm  alpha={ALPHA*1e3:.0f} mrad  dose/DP={DOSE:g}")
    layers = build_layers()
    t0 = time.time()
    dps, coords = simulate(layers, rng)
    log(f"simulated 4D-STEM {dps.shape} in {time.time() - t0:.1f}s")
    np.savez_compressed(os.path.join(OUT, "layers_and_data.npz"),
                        layers=np.array(layers), dps=dps, coords=coords)
    Sm, beams, loss = reconstruct(dps, coords, log)
    np.save(os.path.join(OUT, "Smatrix_complex64.npy"), Sm)      # large, git-ignored
    depths = np.arange(0.0, THICK + 1, 10.0)
    EW = depth_sections(Sm, beams, depths)
    np.savez_compressed(os.path.join(OUT, "sections.npz"), depths=depths, loss=loss,
                        ew=EW, beams_y=beams[0], beams_x=beams[1])
    with open(os.path.join(OUT, "run_log.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
