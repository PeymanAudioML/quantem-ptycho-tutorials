"""Polycrystalline test sample: CaAl2Si2O8 (24-atom cell, authors' structure file 1000048.p1),
8 randomly rotated 3-D Voronoi grains, 150 A thick.  Runs the S-matrix pipeline -> results_poly/.
Approximations: static atoms (no thermal motion), Gaussian projected potentials with
strength ~ Z^0.7 (not Kirkland/Lobato), grains rotated about the beam axis only (all zone-axis)."""
import os, json, time
import numpy as np
import smatrix_depth_demo as m

THICK = 150.0
DZ = 2.5
NSL = int(round(THICK / DZ))
SIGMA = 0.35          # A, gaussian width of an atom
KAPPA = 0.045         # peak phase = KAPPA * Z^0.7 (rad)
NGRAIN = 8
SEED = 3

# --- structure (VASP-like file from the authors' archive): a=b=7.685, c=5.0636 A
A, C = 7.685, 5.0636
SP = {"Ca": 20, "Al": 13, "Si": 14, "O": 8}
ATOMS = [  # (element, fx, fy, fz)
 ("Ca", .338899991, .161099994, .510399996), ("Ca", .661100009, .838899960, .510399996),
 ("Ca", .161099994, .661100009, .489600004), ("Ca", .838899960, .338899991, .489600004),
 ("Al", 0, 0, 0), ("Al", .5, .5, 0),
 ("Si", .143399997, .356599988, .954000037), ("Si", .856599957, .643400012, .954000037),
 ("Si", .356599988, .856599957, .046000004), ("Si", .643400012, .143399997, .046000004),
 ("O", .5, 0, .176499997), ("O", 0, .5, .823500003),
 ("O", .142700005, .357300011, .283499985), ("O", .857299980, .642699927, .283499985),
 ("O", .357300011, .857299980, .716500038), ("O", .642699927, .142700005, .716500038),
 ("O", .087599997, .167799990, .807799954), ("O", .912400019, .832199979, .807799954),
 ("O", .167799990, .912400019, .192200010), ("O", .832199979, .087599997, .192200010),
 ("O", .412400019, .667799990, .192200010), ("O", .587600012, .332199979, .192200010),
 ("O", .332199979, .412400019, .807799954), ("O", .667799990, .587600012, .807799954)]


def build_atoms(rng):
    L = m.Y * m.DR
    seeds = np.column_stack([rng.uniform(0, L, NGRAIN), rng.uniform(0, L, NGRAIN), rng.uniform(0, THICK, NGRAIN)])
    theta = rng.uniform(0, np.pi / 2, NGRAIN)
    shift = rng.uniform(0, 1, (NGRAIN, 3))
    out = []
    nij = int(np.ceil(1.6 * L / A)) + 1
    ii, jj = np.meshgrid(np.arange(-nij, nij + 1), np.arange(-nij, nij + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    ncell = int(np.ceil(THICK / C)) + 2
    for g in range(NGRAIN):
        ct, st = np.cos(theta[g]), np.sin(theta[g])
        for el, fx, fy, fz in ATOMS:
            for n in range(-1, ncell):
                x0 = A * (ii + fx + shift[g, 0]); y0 = A * (jj + fy + shift[g, 1])
                z = (n + fz + shift[g, 2]) * C
                if not (0 <= z < THICK):
                    continue
                x = seeds[g, 0] + ct * (x0) - st * (y0)
                y = seeds[g, 1] + st * (x0) + ct * (y0)
                inbox = (x >= 0) & (x < L) & (y >= 0) & (y < L)     # no wrapping: avoids duplicate atoms
                x, y = x[inbox], y[inbox]
                if len(x) == 0:
                    continue
                # keep atoms that are nearest to this grain's seed (periodic in x,y)
                dx = (x[:, None] - seeds[None, :, 0] + L / 2) % L - L / 2
                dy = (y[:, None] - seeds[None, :, 1] + L / 2) % L - L / 2
                d2 = dx ** 2 + dy ** 2 + (z - seeds[None, :, 2]) ** 2
                keep = np.argmin(d2, axis=1) == g
                for xx, yy in zip(x[keep], y[keep]):
                    out.append((xx, yy, z, SP[el]))
    return np.array(out), seeds, theta


def slice_phases(atoms):
    ph = np.zeros((NSL, m.Y, m.X), np.float32)
    r = int(np.ceil(4 * SIGMA / m.DR))
    off = np.arange(-r, r + 1)
    for x, y, z, Z in atoms:
        k = min(int(z // DZ), NSL - 1)
        cy, cx = int(round(y / m.DR)), int(round(x / m.DR))
        yy = (cy + off) % m.Y; xx = (cx + off) % m.X
        dy = (cy + off) * m.DR - y; dx = (cx + off) * m.DR - x
        g = KAPPA * Z ** 0.7 * np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2 * SIGMA ** 2))
        ph[k][np.ix_(yy, xx)] += g.astype(np.float32)
    return ph


def simulate_poly(slices, rng):
    P = m.fresnel(DZ)
    trans = np.exp(1j * slices).astype(np.complex64)
    ndf = len(m.DEFOCI)
    dps = np.zeros((ndf, m.NSCAN, m.NSCAN, m.K, m.K), np.float32)
    c0 = m.Y // 2 - (m.NSCAN // 2) * m.STEP
    coords = np.zeros((m.NSCAN, m.NSCAN, 2), int)
    for idf, df in enumerate(m.DEFOCI):
        p0 = m.probe_in(df)
        for iy in range(m.NSCAN):
            for ix in range(m.NSCAN):
                ry, rx = c0 + iy * m.STEP, c0 + ix * m.STEP
                coords[iy, ix] = (ry, rx)
                psi = np.roll(p0, (ry, rx), axis=(0, 1))
                for k in range(NSL):
                    psi = psi * trans[k]
                    psi = np.fft.ifft2(np.fft.fft2(psi) * P)
                win = psi[ry - m.K // 2: ry + m.K // 2, rx - m.K // 2: rx + m.K // 2]
                I = np.abs(np.fft.fft2(win, norm="ortho")) ** 2
                if m.DOSE > 0:
                    I = rng.poisson(I * m.DOSE).astype(np.float32) / m.DOSE
                dps[idf, iy, ix] = I
        print("defocus", df, "simulated", flush=True)
    return dps, coords


if __name__ == "__main__":
    m.OUT = os.path.join(m.HERE, "results_poly"); os.makedirs(m.OUT, exist_ok=True)
    m.THICK = THICK
    m.DEFOCI = [-150.0, -75.0, 0.0]
    rng = np.random.default_rng(SEED)
    atoms, seeds, theta = build_atoms(rng)
    print("atoms:", len(atoms), " grains:", NGRAIN, " rotation (deg):", np.round(np.degrees(theta), 1))
    slices = slice_phases(atoms)
    print("slice phase: max %.2f rad, mean per slice %.4f" % (slices.max(), slices.mean()))
    json.dump(dict(name="poly", thickness=THICK, dz=DZ, nslices=NSL, defoci=m.DEFOCI, seeds=seeds.tolist(),
                   theta_deg=np.degrees(theta).tolist(), natoms=int(len(atoms)), mu=m.MU, niter=m.NITER,
                   dose=m.DOSE), open(os.path.join(m.OUT, "config.json"), "w"))
    m.build_layers = lambda: slices
    m.simulate = lambda layers, rng_: simulate_poly(layers, rng)
    m.main()
