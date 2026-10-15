"""Shared code of the synthetic 4D-STEM benchmark: configuration, specimens, abTEM potential and multislice, file layout.

Conventions (all saved files)
  * Array axis 0 = ase x, axis 1 = ase y; z = depth below the entrance surface, 0..T.  The abTEM potential, the probes,
    the scan positions and the diffraction patterns all use this order, so no transposes appear anywhere.
    (simple_joint calls axis 0 "y"; only the label differs.)
  * Aberrations: simple_joint coefficients, C10 < 0 = focus |C10| below the entrance; C12a/C12b azimuth
    phi = atan2(k_axis0, k_axis1).  QuantEM defocus = -C10 (run_realization.quantem_aberrations converts the rest).
  * Potential: abTEM projected potential per slice in V*A (its metadata says "eV / e"); transmission exp(i sigma V)
    exactly (verified in build_potential).  Slice k covers [k dz, (k + 1) dz); an atom at depth z sits in slice
    floor(z / dz) (abTEM "infinite" projection, Lobato parametrisation, abTEM defaults).
  * Intensities: fraction of the incident electrons per detector pixel (unit-norm entrance probe).  Noisy data are
    Poisson counts with expectation intensity x dose.
  * Detector variants: "window" = |FFT_ortho|^2 of the K x K exit-wave window around the scan position (the S-matrix
    forward model); "pixelated" = abTEM PixelatedDetector(max_angle=None) on the full Y x Y exit wave (normalised by
    Y^2), integrated over (Y/K) x (Y/K) native pixels per detector pixel.  Same reciprocal pixel 1 / (K dr).
Ground truth is written to ground_truth/ only; the datasets carry the nominal (not the true) aberrations.
"""
import datetime, hashlib, json, math, os, platform, subprocess, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import simple_joint as sj                    # noqa: E402
import abtem_data as ad                      # noqa: E402
import calibrate_m3 as m3                    # noqa: E402

DEFAULT_CONFIG = os.path.join(HERE, "configs", "benchmark_200A.yaml")


# ------------------------------------------------------------------------------------------------ configuration
def load_config(path=DEFAULT_CONFIG):
    raw = open(path, "rb").read()
    cfg = yaml.safe_load(raw)
    cfg["_path"] = os.path.relpath(path, PARENT)
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def make_setup(cfg):
    m, s = cfg["microscope"], cfg["sampling"]
    return sj.Setup(energy_ev=float(m["energy_ev"]), alpha=1e-3 * float(m["alpha_mrad"]), dr=float(s["dr_A"]),
                    K=int(s["K"]), Y=int(s["Y"]))


def scan_positions(cfg):
    """(Ny, Nx, 2) integer scan positions [S-grid px] (window centres), axis-0 coordinate first."""
    s = cfg["sampling"]
    ny, nx = s["scan_shape"]
    g0 = s["scan_start_px"] + np.arange(ny) * s["scan_step_px"]
    g1 = s["scan_start_px"] + np.arange(nx) * s["scan_step_px"]
    return np.stack(np.meshgrid(g0, g1, indexing="ij"), -1).astype(int)


def coefficients(cfg):
    """nominal, true (D, P) aberration coefficients [A] in simple_joint order, and the reference dataset index."""
    d = cfg["defocus"]
    D = len(d["nominal_C10"])
    nominal = np.zeros((D, len(sj.NAMES)))
    nominal[:, sj.NAMES.index("C10")] = d["nominal_C10"]
    true = nominal.copy()
    idx = [sj.NAMES.index(n) for n in ("C10", "C12a", "C12b")]
    for i, p in enumerate(d["perturbations"]):
        true[i, idx] += p
    ref = int(d["reference"])
    if not np.allclose(true[ref], nominal[ref]):
        raise ValueError("the reference dataset must have exact nominal aberrations (S-matrix gauge)")
    return nominal, true, ref


def versions():
    import abtem, ase, torch, scipy, zarr
    out = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
               abtem=abtem.__version__, ase=ase.__version__, zarr=zarr.__version__, yaml=yaml.__version__)
    try:
        import quantem, glob
        f = glob.glob(os.path.join(os.path.dirname(quantem.__file__), "..", "quantem*.dist-info", "direct_url.json"))
        out["quantem"] = json.load(open(f[0]))["vcs_info"]["commit_id"] if f else getattr(quantem, "__version__", "?")
    except Exception as e:                                   # quantem is not needed to generate data
        out["quantem"] = f"not importable ({type(e).__name__})"
    try:
        out["repo_commit"] = subprocess.check_output(["git", "-C", PARENT, "rev-parse", "HEAD"], text=True).strip()
        out["repo_dirty"] = bool(subprocess.check_output(["git", "-C", PARENT, "status", "--porcelain"], text=True).strip())
    except Exception:
        out["repo_commit"] = "unknown"
    return out


# ------------------------------------------------------------------------------------------------ specimens
def build_specimen(cfg, setup, name):
    """ase.Atoms (axis-0 coordinate = ase x) and the ground-truth feature list.  Ground truth is used only here
    and in evaluation."""
    from ase import Atoms
    spec = cfg["specimens"][name]
    T = float(cfg["specimen"]["thickness_A"])
    eps = float(cfg["specimen"]["z_eps_A"])
    L = setup.Y * setup.dr
    cl = cfg["cluster"]
    feats, au = [], []
    for i, f in enumerate(spec["features"]):
        xy = m3.gold_cluster(setup, int(f.get("seed", cl["seed"])), n=int(f.get("n_atoms", cl["n_atoms"])),
                             spread=float(cl["spread_A"])) + np.asarray(f.get("offset_A", [0.0, 0.0]), float)
        z = float(f["z"])
        if not (eps <= z <= T - eps):
            raise ValueError(f"{name}: feature {i} at z = {z} outside [{eps}, {T - eps}]")
        pos = np.array([(a0, a1, z) for a0, a1 in xy])
        feats.append(dict(id=i, element="Au", n_atoms=len(pos), z=z, centroid_xy=xy.mean(0).tolist(),
                          positions=pos.tolist()))
        au.append(pos)
    au = np.vstack(au) if au else np.zeros((0, 3))
    C = np.zeros((0, 3))
    if spec.get("carbon"):
        c = spec["carbon"]
        C = m3.carbon(setup, T, int(c["seed"]), density=float(c["density"]))
        if len(au):
            C = C[np.linalg.norm(C[:, None, :] - au[None, :, :], axis=-1).min(axis=1) > 2.0]
        C[:, 2] = np.clip(C[:, 2], eps, T - eps)
    pos = np.vstack([C, au]) if len(C) else au
    if (pos[:, :2] < 0).any() or (pos[:, :2] >= L).any() or (pos[:, 2] < eps).any() or (pos[:, 2] > T - eps).any():
        raise ValueError(f"{name}: atoms outside the cell")
    atoms = Atoms(symbols=["C"] * len(C) + ["Au"] * len(au), positions=pos, cell=[L, L, T], pbc=[True, True, False])
    feature_id = np.r_[np.full(len(C), -1), np.repeat(np.arange(len(feats)), [f["n_atoms"] for f in feats])]
    atoms.new_array("feature", feature_id.astype(int))
    return atoms, feats


def build_potential(cfg, setup, atoms):
    """abTEM potential (n, Y, Y) in V*A per slice, slice boundaries, and verification of units / orientation / slicing."""
    import abtem
    from abtem.core.energy import energy2sigma
    T = float(cfg["specimen"]["thickness_A"])
    dz = float(cfg["specimen"]["potential_slice_A"])
    pot = abtem.Potential(atoms, gpts=(setup.Y, setup.Y), slice_thickness=dz)
    pa = pot.build().compute(progress_bar=False)
    V = np.asarray(pa.array, np.float32)
    st = np.asarray(pot.slice_thickness, float)
    bounds = np.r_[0.0, np.cumsum(st)]
    sigma = float(energy2sigma(setup.energy_ev))
    t = np.asarray(pa.transmission_function(setup.energy_ev).compute(progress_bar=False).array)
    checks = dict(
        n_slices=int(len(V)), slice_thickness=float(st[0]), thickness_sum=float(bounds[-1]),
        sampling=list(map(float, pot.sampling)), shape=list(V.shape),
        transmission_equals_exp_i_sigma_V=float(np.abs(t - np.exp(1j * sigma * V)).max()),
        sigma_rad_per_V_A=sigma)
    # orientation and slicing: every Au atom's column must peak in slice floor(z / dz) near (x, y)
    au = np.flatnonzero(np.asarray(atoms.get_chemical_symbols()) == "Au")
    rows = []
    for i in au:
        x, y, z = atoms.positions[i]
        k = min(int(z // dz), len(V) - 1)
        a0, a1 = int(round(x / setup.dr)) % setup.Y, int(round(y / setup.dr)) % setup.Y
        w = V[k, max(a0 - 3, 0):a0 + 4, max(a1 - 3, 0):a1 + 4]
        rows.append(dict(atom=int(i), z=float(z), expected_slice=k,
                         peak_slice=int(np.argmax(V[:, a0, a1])), peak_offset_px=np.subtract(
                             np.unravel_index(np.argmax(w), w.shape), (min(3, a0), min(3, a1))).tolist()))
    checks["au_columns"] = rows
    checks["orientation_and_slicing_ok"] = bool(all(r["peak_slice"] == r["expected_slice"] and
                                                    max(map(abs, r["peak_offset_px"])) <= 1 for r in rows))
    checks["thickness_ok"] = bool(abs(bounds[-1] - T) < 1e-6)
    if not (checks["orientation_and_slicing_ok"] and checks["thickness_ok"] and
            checks["transmission_equals_exp_i_sigma_V"] < 1e-4):
        raise RuntimeError(f"potential verification failed: {json.dumps(checks)[:800]}")
    return V, bounds, sigma, pot, checks


# ------------------------------------------------------------------------------------------------ multislice
def simulate(setup, potential, coeffs, positions, batch=64, log=print):
    """Noiseless patterns for both detector variants, intensity fraction per pixel, centred (zero frequency at the
    returned pixel).  window: (D, Ny, Nx, K, K), zero at K/2; pixelated: (D, Ny, Nx, K, K), zero at zero_pix."""
    import abtem
    from abtem.core.axes import OrdinalAxis
    Y, K = setup.Y, setup.K
    b = Y // K
    if b * K != Y:
        raise ValueError("pixelated variant needs Y to be a multiple of K")
    L = Y * setup.dr
    pos = np.asarray(positions).reshape(-1, 2)
    D = len(coeffs)
    win = np.zeros((D, len(pos), K, K), np.float32)
    pix = np.zeros((D, len(pos), K, K), np.float32)
    h = K // 2
    det = abtem.PixelatedDetector(max_angle=None)
    check = None
    for d, c in enumerate(np.asarray(coeffs, float)):
        p0 = ad.entrance_probe(setup, c)                            # (Y, Y), axis 0 first, unit norm, corner-centred
        for b0 in range(0, len(pos), batch):
            chunk = pos[b0:b0 + batch]
            arr = np.stack([np.roll(p0, (int(r0), int(r1)), axis=(0, 1)) for r0, r1 in chunk])
            waves = abtem.Waves(arr, energy=setup.energy_ev, extent=L,
                                ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(len(arr))))])
            ex = waves.multislice(potential).compute(progress_bar=False)
            exa = np.asarray(ex.array)
            for n, (r0, r1) in enumerate(chunk):
                w = exa[n, r0 - h:r0 + h, r1 - h:r1 + h]
                win[d, b0 + n] = np.fft.fftshift(np.abs(np.fft.fft2(w, norm="ortho")) ** 2)
            full = np.asarray(det.detect(ex).compute(progress_bar=False).array, np.float64) / Y ** 2   # fftshifted
            if check is None:                                        # abTEM detector == fftshift |FFT_ortho|^2
                ref = np.fft.fftshift(np.abs(np.fft.fft2(exa[0], norm="ortho")) ** 2)
                check = float(np.abs(full[0] - ref).max() / ref.max())
            pix[d, b0:b0 + len(chunk)] = full.reshape(len(chunk), K, b, K, b).sum(axis=(2, 4))
        log(f"  abTEM multislice: dataset {d + 1}/{D} done")
    shape = np.asarray(positions).shape[:2]
    zero_pix = (Y / 2 - (b - 1) / 2) / b                             # native zero (Y/2) in binned pixel units
    return dict(window=win.reshape(D, *shape, K, K), pixelated=pix.reshape(D, *shape, K, K),
                zero_pixel=dict(window=K / 2, pixelated=zero_pix), pixelated_vs_fft_check=check, bin=b)


def poisson(intensity, dose, seed):
    return np.random.default_rng(seed).poisson(np.clip(intensity, 0, None) * dose).astype(np.float32)


# ------------------------------------------------------------------------------------------------ files
def write_array(path, arr):
    import zarr
    zarr.create_array(store=path, data=np.ascontiguousarray(arr), overwrite=True)


def write_group(path, attrs):
    import zarr
    g = zarr.open_group(path, mode="a")
    g.attrs.update(attrs)


def write_dataset(path, data, step, dk, zero, positions_A, cell, energy, alpha_mrad, nominal_focus, bench_meta):
    """One Dataset4dstem directory in the realization layout read by run_realization.load_dataset."""
    write_group(path, {"_name": bench_meta["label"], "_signal_units": bench_meta["signal_units"]})
    write_array(os.path.join(path, "_array"), data.astype(np.float32))
    write_array(os.path.join(path, "_sampling"), np.array([step, step, dk, dk], float))
    sim = os.path.join(path, "_metadata", "simulation")
    write_group(os.path.join(path, "_metadata"), {})
    write_group(sim, {})
    write_group(os.path.join(sim, "probe"), {"energy_eV": energy, "semiangle_cutoff_mrad": alpha_mrad,
                                             "defocus_A": float(nominal_focus), "defocus_meaning": "nominal, = -C10"})
    write_group(os.path.join(sim, "detector"), {"variant": bench_meta["variant"], "integration": bench_meta["integration"]})
    write_array(os.path.join(sim, "detector", "zero_frequency_pixel"), np.array([zero, zero], float))
    write_array(os.path.join(sim, "padded_scan_positions_A"), positions_A)
    write_group(os.path.join(sim, "geometry"), {"defocus_reference": "specimen entrance, z = 0",
                                                "axis_order": "axis 0 = ase x, axis 1 = ase y"})
    write_array(os.path.join(sim, "geometry", "computational_cell_A"), np.asarray(cell, float))
    write_group(os.path.join(path, "_metadata", "benchmark"), bench_meta)


def write_probe(path, probe_real, setup, true_c, nominal_c):
    axes = {"axis_0": {"label": "x", "units": "A", "sampling": setup.dr, "offset": 0.0},
            "axis_1": {"label": "y", "units": "A", "sampling": setup.dr, "offset": 0.0}}
    write_group(path, {"metadata0": {
        "energy": setup.energy_ev, "semiangle_cutoff": 1e3 * setup.alpha, "wavelength_A": setup.lam,
        "real_space_centering": "corner", "normalisation": "sum |psi|^2 = 1", "axes": axes,
        "aberration_names": sj.NAMES, "true_coefficients_A": list(map(float, true_c)),
        "nominal_coefficients_A": list(map(float, nominal_c)),
        "convention": "simple_joint: chi from aberration_basis, C10 < 0 = focus inside, phi = atan2(k_axis0, k_axis1)",
        "array1": "Fourier-space probe, FFT_ortho(array0), zero frequency at [0, 0]"}})
    write_array(os.path.join(path, "array0"), probe_real.astype(np.complex64))
    write_array(os.path.join(path, "array1"), np.fft.fft2(probe_real, norm="ortho").astype(np.complex64))


def write_potential(path, V, bounds, setup, sigma):
    st = np.diff(bounds)
    write_group(path, {"metadata0": {
        "label": "projected potential per slice", "units": "V A (abTEM label 'eV / e'); phase = sigma V",
        "sigma_rad_per_V_A": sigma, "slice_boundaries_A": list(map(float, bounds)),
        "axes": {"axis_0": {"label": "z", "units": "A", "values": {"_type": "tuple", "_value": list(map(float, bounds[1:]))}},
                 "axis_1": {"label": "x", "units": "A", "sampling": setup.dr, "offset": 0.0},
                 "axis_2": {"label": "y", "units": "A", "sampling": setup.dr, "offset": 0.0}},
        "kwargs": {"slice_thickness": {"_type": "tuple", "_value": list(map(float, st))}}}})
    write_array(os.path.join(path, "array0"), V.astype(np.float32))


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
