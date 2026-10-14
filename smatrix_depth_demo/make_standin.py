"""Small simulated stand-in in the realization layout (one folder per defocus), to test run_realization.py end to end.

Each folder mirrors realization_0: 4dstem_quantem/ (Dataset4dstem zarr: _array (Ny, Nx, Q, Q) fftshifted patterns,
_sampling [step, step, dk, dk] in A / 1/A, _metadata/simulation/... with energy, semiangle, defocus, zero-frequency
pixel, padded scan positions, computational cell), probe.zarr (entrance probe, real space, corner-centred),
potential.zarr (projected potential per slice, eV/e * A) and atoms.extxyz.

Specimen: 100 A slab at 300 kV, 20 mrad; two 3-atom phase clusters (1 rad per atom) at depths 30 and 70 A at different
lateral positions; probe focus depths 83, 50, 17 A (defocus = -C10, abTEM / QuantEM sign).  The patterns come from the
built-in multislice (demo_simple_joint.simulate), so the probe and depths are known exactly.

    python make_standin.py out_dir            # -> out_dir/def_083, def_050, def_017
"""
import json, os, sys
import numpy as np
import zarr

import simple_joint as sj
import demo_simple_joint as demo

T = 100.0
SETUP = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.2, K=32, Y=64)
FOCI = [83.0, 50.0, 17.0]
CLUSTERS = [((4.8, 4.8), 30.0), ((8.0, 8.0), 70.0)]       # (lateral centre (A, cell frame), depth A)
STEP_PX = 2
SLICE = 2.0


def sigma_interaction(energy_ev):
    """Interaction constant [rad / (V A)]."""
    lam = sj.Setup(energy_ev=energy_ev).lam
    m0c2 = 5.109989461e5
    return 2 * np.pi / (lam * energy_ev) * (m0c2 + energy_ev) / (2 * m0c2 + energy_ev)


def specimen(rng):
    r = np.arange(SETUP.Y) * SETUP.dr
    layers, atoms = [], []
    for (cy, cx), z in CLUSTERS:
        ph = np.zeros((SETUP.Y, SETUP.Y))
        for y0, x0 in np.array([cy, cx]) + rng.uniform(-0.8, 0.8, (3, 2)):
            ph += np.exp(-((r[:, None] - y0) ** 2 + (r[None, :] - x0) ** 2) / (2 * 0.45 ** 2))
            atoms.append((y0, x0, z))
        layers.append(ph)
    return np.array(layers), [z for _, z in CLUSTERS], atoms


def write_array(path, arr):
    zarr.create_array(store=path, data=np.ascontiguousarray(arr), overwrite=True)


def write_group(path, attrs):
    g = zarr.open_group(path, mode="a")
    g.attrs.update(attrs)


def main(out, dose=1e5, seed=0):
    rng = np.random.default_rng(seed)
    layers, layer_z, atoms = specimen(rng)
    g = np.arange(SETUP.K // 2, SETUP.Y - SETUP.K // 2 + 1, STEP_PX)
    positions = np.stack(np.meshgrid(g, g, indexing="ij"), -1)
    coeffs = np.zeros((len(FOCI), len(sj.NAMES)))
    coeffs[:, 0] = [-f for f in FOCI]
    dps = demo.simulate(SETUP, layers, layer_z, T, coeffs, positions, dose=dose, seed=seed)
    sig = sigma_interaction(SETUP.energy_ev)
    nz = int(round(T / SLICE))
    pot = np.zeros((nz, SETUP.Y, SETUP.Y), np.float32)
    for L, z in zip(layers, layer_z):
        pot[min(int(z // SLICE), nz - 1)] += (L / sig).astype(np.float32)
    q = np.fft.fftfreq(SETUP.Y, SETUP.dr)
    qy, qx = np.meshgrid(q, q, indexing="ij")
    import torch
    basis = sj.aberration_basis(torch.as_tensor(qy.ravel()), torch.as_tensor(qx.ravel()),
                                SETUP.lam).numpy().reshape(len(sj.NAMES), SETUP.Y, SETUP.Y)
    aperture = np.hypot(qy, qx) <= SETUP.alpha / SETUP.lam
    axes_xy = lambda i: {f"axis_{i}": {"label": "x", "units": "A", "sampling": SETUP.dr, "offset": 0.0},
                         f"axis_{i + 1}": {"label": "y", "units": "A", "sampling": SETUP.dr, "offset": 0.0}}
    for d, f in enumerate(FOCI):
        root = os.path.join(out, f"def_{f:03.0f}")
        ds = os.path.join(root, "4dstem_quantem")
        write_group(ds, {"_name": "stand-in (built-in multislice)", "_signal_units": "intensity"})
        write_array(os.path.join(ds, "_array"), np.fft.fftshift(dps[d], axes=(-2, -1)).astype(np.float32))
        step = STEP_PX * SETUP.dr
        dk = 1.0 / (SETUP.K * SETUP.dr)
        write_array(os.path.join(ds, "_sampling"), np.array([step, step, dk, dk]))
        sim = os.path.join(ds, "_metadata", "simulation")
        write_group(os.path.join(ds, "_metadata"), {})
        write_group(sim, {})
        write_group(os.path.join(sim, "probe"), {"energy_eV": SETUP.energy_ev, "semiangle_cutoff_mrad": 1e3 * SETUP.alpha,
                                                 "defocus_A": f})
        write_group(os.path.join(sim, "probe", "aberrations"), {"C12": 0.0, "phi12": 0.0})
        write_group(os.path.join(sim, "geometry"), {"defocus_reference": "computational-cell entrance, z=0"})
        write_array(os.path.join(sim, "geometry", "computational_cell_A"), np.array([SETUP.Y * SETUP.dr] * 2 + [T]))
        write_group(os.path.join(sim, "detector"), {})
        write_array(os.path.join(sim, "detector", "zero_frequency_pixel"), np.array([SETUP.K / 2] * 2))
        write_array(os.path.join(sim, "padded_scan_positions_A"), positions * SETUP.dr)
        chi = np.tensordot(coeffs[d], basis, axes=1)
        p0 = np.fft.ifft2(aperture * np.exp(-1j * chi))
        p0 /= np.sqrt((np.abs(p0) ** 2).sum())
        write_group(os.path.join(root, "probe.zarr"), {"metadata0": {"energy": SETUP.energy_ev, "semiangle_cutoff": 1e3 * SETUP.alpha,
                                                                     "real_space_centering": "corner", "axes": axes_xy(0)}})
        write_array(os.path.join(root, "probe.zarr", "array0"), p0.astype(np.complex64))
        write_group(os.path.join(root, "potential.zarr"), {"metadata0": {
            "units": "eV / e", "axes": axes_xy(1),
            "kwargs": {"slice_thickness": {"_type": "tuple", "_value": [SLICE] * nz}}}})
        write_array(os.path.join(root, "potential.zarr", "array0"), pot)
        with open(os.path.join(root, "atoms.extxyz"), "w") as fh:
            fh.write(f"{len(atoms)}\nLattice=\"{SETUP.Y * SETUP.dr} 0 0 0 {SETUP.Y * SETUP.dr} 0 0 0 {T}\" "
                     "Properties=species:S:1:pos:R:3\n")
            for y, x, z in atoms:
                fh.write(f"Au {y:.4f} {x:.4f} {z:.4f}\n")
        print(f"wrote {root}: {dps[d].shape[:2]} scan, {SETUP.K}x{SETUP.K} patterns, focus depth {f} A")
    json.dump(dict(cluster_depths=layer_z, foci=FOCI, thickness=T), open(os.path.join(out, "truth.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "standin")
