"""abTEM version of the polycrystal 4D-STEM simulation (same atoms, cell, grains and 2.5 A slicing as the original run_poly.py).

Needs abtem + ase (NOT in the project env; install in a separate venv:  uv venv abvenv && uv pip install --python abvenv/bin/python abtem).
Atoms: CaAl2Si2O8-labelled 24-atom cell (actually Ca4Al2Si4O14), 8 Voronoi grains rotated about the beam, 150 A, seed 3.
Step 1 (project env):  python make_abtem_probes.py B     -> results_poly/abtem_probes_B.npz  (true entrance probes incl. aberration errors)
Step 2 (abTEM env):    python make_poly_abtem.py B [step_px=4]
Potential: Lobato parametrization, 'infinite' projection, static atoms (no thermal motion).  Entrance waves = the probes from step 1
(300 kV, 20 mrad, Eq. 23 aberrations), shifted by whole pixels to each scan position, propagated by abTEM's multislice.  Patterns =
squared FFT of the K x K window of the exit wave centred on each probe position (same cropping as the NumPy simulation), noiseless,
sum|psi_in|^2 = 1.  NB: abTEM band-limits the multislice at 2/3 of Nyquist (~33 mrad here).
Outputs: results_poly/abtem_dps_clean_{cfg}_s{step}.npz (dps[df, iy, ix, ky, kx], coords),
         results_poly/abtem_truth_slices.npz (projected potential per slice, [slice, y, x]),
         results_joint/cache/clean_<sample>_{cfg}.npz  (cache read by run_joint_probe.py --sample poly_abtem / poly_abtem{D}, step 4 only)."""
import os, sys, time, json
import numpy as np
import abtem
from abtem.core.axes import OrdinalAxis
from ase import Atoms
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import smatrix_depth_demo as m
import geom_spec as gs

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
    L = 192 * 0.2                                         # 38.4 A box (kept as the original float expression)
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




config = sys.argv[1] if len(sys.argv) > 1 else "B"
setup = sys.argv[3] if len(sys.argv) > 3 else "a20"; sp = gs.spec(setup); sfx = gs.suffix(setup)
step = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "0" else sp["step"]
nscan = sp["nscan"] if step == sp["step"] else int(round(sp["nscan"] * sp["step"] / step))
probes = np.load(os.path.join(HERE, "results_poly", f"abtem_probes_{config}{sfx}.npz"))["probes"]      # (D, y, x)
atoms, _, _ = build_atoms(np.random.default_rng(SEED))
L = sp["Y"] * sp["dr"]
at = Atoms(numbers=atoms[:, 3].astype(int), positions=atoms[:, :3], cell=[L, L, THICK], pbc=[True, True, False])
pot = abtem.Potential(at, gpts=sp["Y"], slice_thickness=DZ, projection="infinite", parametrization="lobato")
assert pot.gpts == (sp["Y"], sp["Y"]) and pot.num_slices == NSL
pot_arr = pot.build().compute()
np.savez_compressed(os.path.join(HERE, "results_poly", f"abtem_truth_slices{sfx}.npz"),
                    slices=np.asarray(pot_arr.array).transpose(0, 2, 1).astype(np.float32))      # (slice, x, y) -> (slice, y, x)

c0 = sp["Y"] // 2 - (nscan // 2) * step                    # same scan grid as jp.Geometry(nscan, step)
a = c0 + step * np.arange(nscan)
coords = np.stack(np.meshgrid(a, a, indexing="ij"), -1)  # (iy, ix, 2) pixels (row=y, col=x)
K = sp["K"]
dps = np.zeros((len(probes), nscan, nscan, K, K), np.float32)
for d, p0 in enumerate(probes):
    t0 = time.time()
    arr = np.stack([np.roll(p0, tuple(coords[iy, ix]), axis=(0, 1)).T for iy in range(nscan) for ix in range(nscan)])   # (N, x, y)
    w = abtem.Waves(arr, energy=m.EV, extent=L, ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(len(arr))))])
    ew = np.asarray(w.multislice(pot_arr).compute().array)                                  # (N, x, y)
    for n in range(len(arr)):
        iy, ix = divmod(n, nscan); ry, rx = coords[iy, ix]
        win = ew[n, rx - K // 2: rx + K // 2, ry - K // 2: ry + K // 2].T                   # -> (y, x)
        dps[d, iy, ix] = np.abs(np.fft.fft2(win, norm="ortho")) ** 2
    print(f"dataset {d}: {nscan}x{nscan} patterns in {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(HERE, "results_poly", f"abtem_dps_clean_{config}{sfx}_s{step}.npz"), dps=dps, coords=coords)
if step == sp["step"]:
    os.makedirs(os.path.join(HERE, "results_joint", "cache"), exist_ok=True)
    sname = ("poly_abtem" if len(probes) == 3 else f"poly_abtem{len(probes)}") + sfx          # sample name used by run_joint_probe.py
    np.savez(os.path.join(HERE, "results_joint", "cache", f"clean_{sname}_{config}.npz"), dps=dps, coords=coords)
print("saved", dps.shape, " mean total intensity per window %.4f" % dps.sum((-1, -2)).mean())
