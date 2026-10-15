"""4D-STEM diffraction patterns simulated with abTEM (real atomic potentials), in the layout simple_joint.py expects.

    pip install abtem          # (pulls in ase)

    atoms  = layered_atoms(setup, layer_z=[30, 150, 270], thickness=300)       # or any ase.Atoms, cell = [L, L, thickness]
    dps    = simulate_abtem(setup, atoms, coeffs, positions)                    # (D, Ny, Nx, K, K)

The probe is built here (aperture + aberrations in simple_joint's convention: C10 < 0 = focus inside the sample) and
passed to abTEM as the entrance wave at z = 0, so abTEM's own aberration conventions never enter.  abTEM then does the
multislice through the atomic potential; each pattern is |FFT|^2 of the K x K window of the exit wave around the scan
position (the same cropping as the S-matrix model).

Note: abTEM band-limits the multislice to 2/3 of Nyquist, i.e. scattering angles below lam / (3 dr).  With a 20 mrad
probe use dr <= 0.2 A (cut-off ~33 mrad), otherwise the dark field is lost.
"""
import math

import numpy as np

import simple_joint as sj


def band_limit_mrad(setup):
    """Largest scattering angle abTEM keeps [mrad]."""
    return 1e3 * setup.lam / (3 * setup.dr)


def layered_atoms(setup, layer_z, thickness, atoms_per_layer=6, element="Au", spread=3.5, min_dist=1.8, seed=1):
    """ase.Atoms with `atoms_per_layer` atoms of `element` at each depth in layer_z [A], placed at random within
    +-spread A of the field centre (at least min_dist apart).  Returns (atoms, xy) with xy[i] = (y, x) positions [A]
    of layer i (for the ground-truth images)."""
    from ase import Atoms
    rng = np.random.default_rng(seed)
    L = setup.Y * setup.dr
    c = L / 2
    pos, nums, xy = [], [], []
    for z in layer_z:
        pts = []
        while len(pts) < atoms_per_layer:
            p = c + rng.uniform(-spread, spread, 2)
            if all(np.hypot(*(p - q)) >= min_dist for q in pts):
                pts.append(p)
        xy.append(np.array(pts))
        for y, x in pts:
            pos.append((x, y, z))                     # ase: (x, y, z)
            nums.append(element)
    atoms = Atoms(symbols=nums, positions=pos, cell=[L, L, thickness], pbc=[True, True, False])
    return atoms, xy


def layer_images(setup, xy, sigma=0.45):
    """Ground-truth image per layer: Gaussian blobs at the atom positions, (n_layers, Y, Y) with rows = y."""
    r = np.arange(setup.Y) * setup.dr
    out = []
    for pts in xy:
        img = np.zeros((setup.Y, setup.Y))
        for y0, x0 in pts:
            img += np.exp(-((r[:, None] - y0) ** 2 + (r[None, :] - x0) ** 2) / (2 * sigma ** 2))
        out.append(img)
    return np.array(out)


def entrance_probe(setup, coeffs):
    """Probe at the entrance surface on the Y x Y grid (rows = y), unit norm: IFFT[aperture * exp(-i chi)]."""
    import torch
    q = np.fft.fftfreq(setup.Y, setup.dr)
    qy, qx = np.meshgrid(q, q, indexing="ij")
    basis = sj.aberration_basis(torch.as_tensor(qy.ravel()), torch.as_tensor(qx.ravel()), setup.lam).numpy()
    chi = (np.asarray(coeffs, float) @ basis).reshape(setup.Y, setup.Y)
    aperture = setup.aperture(np.hypot(qy, qx))          # hard (taper 0) or soft-edged, as in JointModel
    p = np.fft.ifft2(aperture * np.exp(-1j * chi))
    return (p / np.sqrt((np.abs(p) ** 2).sum())).astype(np.complex64)


def simulate_abtem(setup, atoms, coeffs, positions, slice_thickness=2.0, dose=None, seed=0, batch=64, log=print):
    """Patterns (D, Ny, Nx, K, K) for probes with aberrations coeffs (D, P) [A] at scan positions (Ny, Nx, 2) [px].
    dose: electrons per pattern for Poisson noise (None = noiseless)."""
    import abtem
    from abtem.core.axes import OrdinalAxis

    if band_limit_mrad(setup) < 1.3e3 * setup.alpha:
        log(f"warning: abTEM band limit {band_limit_mrad(setup):.0f} mrad leaves little dark field for a "
            f"{1e3 * setup.alpha:.0f} mrad probe; use a smaller dr")
    Y, K = setup.Y, setup.K
    L = Y * setup.dr
    potential = abtem.Potential(atoms, gpts=Y, slice_thickness=slice_thickness).build().compute(progress_bar=False)
    pos = np.asarray(positions).reshape(-1, 2)
    dps = np.zeros((len(coeffs), len(pos), K, K), np.float32)
    h = K // 2
    for d, c in enumerate(np.asarray(coeffs, float)):
        p0 = entrance_probe(setup, c)
        for b0 in range(0, len(pos), batch):
            chunk = pos[b0:b0 + batch]
            # abTEM arrays are (x, y): transpose our (y, x) probe after shifting it to each scan position
            arr = np.stack([np.roll(p0, (int(ry), int(rx)), axis=(0, 1)).T for ry, rx in chunk])
            waves = abtem.Waves(arr, energy=setup.energy_ev, extent=L,
                                ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(len(arr))))])
            exit_waves = np.asarray(waves.multislice(potential).compute(progress_bar=False).array)         # (n, x, y)
            for n, (ry, rx) in enumerate(chunk):
                win = exit_waves[n, rx - h:rx + h, ry - h:ry + h].T                          # back to (y, x)
                dps[d, b0 + n] = np.abs(np.fft.fft2(win, norm="ortho")) ** 2
        log(f"abTEM: dataset {d + 1}/{len(coeffs)} done")
    dps = dps.reshape(len(coeffs), *np.asarray(positions).shape[:2], K, K)
    if dose:
        dps = np.random.default_rng(seed).poisson(dps * dose).astype(np.float32) / dose
    return dps
