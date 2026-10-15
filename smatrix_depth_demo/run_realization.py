"""Diffraction patterns in -> probe, object and slice locations out, by two methods, saved side by side.

    1. S-matrix stage (simple_joint): joint S-matrix + probe fit on ALL defocus datasets, refocus through the depth,
       slice locations (focus curve peaks + per-feature depths), probe (aberrations per dataset).
    2. QuantEM multislice on ONE dataset; arms (--arms) with the same number of planes, iterations and optimiser.
       QuantEM updates BOTH object and probe in every arm: object only for --probe-warmup iterations, then both.
         baseline  uniform slices, flat potential, nominal probe (metadata defocus)       (QuantEM on its own)
         sprobe    uniform slices, flat potential, full S-matrix probe                     (probe hand-over only)
         sinit     S-informed slices, S-informed potential (--init-object), full S-matrix probe
       Full S-matrix probe = every fitted aberration (C10, C12 astigmatism, ...) converted to QuantEM's polar
       convention (quantem_aberrations; the azimuth axes differ), as a pixelated probe or, with
       --probe-model parametric, as QuantEM aberration coefficients that QuantEM keeps refining.
    3. Images (.png) and arrays (.npy) of every probe and object, the slice grids used to start QuantEM, and, when the
       ground truth sits next to the data (potential.zarr, probe.zarr, atoms.extxyz), a comparison of all of them.

Input: one folder per defocus in the realization layout (a folder holding 4dstem_quantem/, or the Dataset4dstem zarr
itself): _array (Ny, Nx, Q, Q) fftshifted patterns, _sampling [step, step, dk, dk] (A, 1/A), metadata with
energy_eV, semiangle_cutoff_mrad, defocus_A (= -C10, focus depth below the entrance), zero_frequency_pixel and
padded_scan_positions_A.  Anything missing can be given on the command line.  All datasets must share one scan.

    python run_realization.py --data r_def033 r_def100 r_def167 --out results_real --device cuda
    python run_realization.py --data ... --thickness 200 --defocus 33,100,167 --dr 0.2 --scan-roi 0,30,0,30
    python make_standin.py /tmp/standin && python run_realization.py --data /tmp/standin/def_* --out /tmp/rr

Runs in QuantEM's environment (numpy, scipy, torch, zarr, matplotlib, quantem); --no-quantem skips step 2.
The S-matrix stage refuses data with too few measurements per unknown (simple_joint.check_geometry); --force overrides.

Conventions: depth z = 0 is the entrance surface (cell z = 0 of the simulation); QuantEM defocus = -C10; QuantEM object
pixel j <-> S-grid coordinate (K/2) dr_S + (j - p0) dr_Q with p0 = QuantEM's first scan position (M5).  Absolute depth
of the S-matrix stage is tied to the reference dataset's defocus (gauge, --ref).
"""
import argparse, json, math, os, time, warnings
import numpy as np
import torch

import simple_joint as sj
import quantem_export as qe

warnings.filterwarnings("ignore")
PAD = 8
BASE = 1e-3


# ------------------------------------------------------------------------------------------------ loading
def find_dataset(path):
    for c in (path, os.path.join(path, "4dstem_quantem")):
        if os.path.isdir(os.path.join(c, "_array")):
            return c
    raise FileNotFoundError(f"no Dataset4dstem (_array) in {path} or {path}/4dstem_quantem")


def read_metadata(ds):
    """All group attributes and small arrays under _metadata, keyed by relative path."""
    import zarr
    attrs, arrays = {}, {}
    base = os.path.join(ds, "_metadata")
    for dp, _, fn in os.walk(base):
        if "zarr.json" not in fn:
            continue
        j = json.load(open(os.path.join(dp, "zarr.json")))
        rel = os.path.relpath(dp, base)
        if j.get("node_type") == "group":
            attrs[rel] = j.get("attributes", {})
        elif int(np.prod(j.get("shape", [0]))) <= 100_000:
            key = os.path.dirname(rel) if os.path.basename(rel) == "values" else rel
            arrays[key] = zarr.open_array(dp, mode="r")[...]
    return attrs, arrays


def lookup(attrs, arrays, key, prefer="probe"):
    hits = [(p, a[key]) for p, a in attrs.items() if a.get(key) is not None]
    hits += [(p, v) for p, v in arrays.items() if os.path.basename(p) == key]
    hits.sort(key=lambda h: (prefer not in h[0], "configuration" in h[0], len(h[0])))
    return hits[0][1] if hits else None


def load_dataset(path, a, d):
    import zarr
    ds = find_dataset(path)
    attrs, arrays = read_metadata(ds)
    data = zarr.open_array(os.path.join(ds, "_array"), mode="r")[...].astype(np.float32)
    samp = np.asarray(zarr.open_array(os.path.join(ds, "_sampling"), mode="r")[...], float)
    energy = a.energy or lookup(attrs, arrays, "energy_eV") or lookup(attrs, arrays, "energy")
    alpha = a.alpha or lookup(attrs, arrays, "semiangle_cutoff_mrad") or lookup(attrs, arrays, "semiangle_cutoff")
    defocus = a.defocus[d] if a.defocus else lookup(attrs, arrays, "defocus_A")
    zero = lookup(attrs, arrays, "zero_frequency_pixel")
    pos = lookup(attrs, arrays, "padded_scan_positions_A")
    if pos is None:
        pos = lookup(attrs, arrays, "physical_scan_positions_A")
    cell = lookup(attrs, arrays, "computational_cell_A")
    c12 = lookup(attrs, arrays, "C12")
    phi12 = lookup(attrs, arrays, "phi12")
    taper = getattr(a, "taper", None)
    taper = taper if taper is not None else lookup(attrs, arrays, "aperture_taper_mrad")
    missing = [n for n, v in (("energy", energy), ("semiangle", alpha), ("defocus", defocus)) if v is None]
    if missing:
        raise ValueError(f"{path}: {missing} not in the metadata; pass --energy / --alpha / --defocus")
    Q = data.shape[-1]
    return dict(path=path, data=data, step=samp[:2], dk=samp[2:], energy=float(energy), alpha_mrad=float(alpha),
                defocus=float(defocus), zero=np.asarray(zero if zero is not None else [Q / 2, Q / 2], float),
                positions_A=None if pos is None else np.asarray(pos, float), cell=None if cell is None else np.asarray(cell, float),
                C12=None if c12 is None else float(c12), phi12=None if phi12 is None else float(phi12),
                taper_mrad=float(taper or 0.0))


def centre_patterns(data, zero, b=1):
    """Shift the zero frequency to native pixel N/2 + (b-1)/2 (bilinear), so that b x b binning puts it at the centre
    of bin N/(2b)."""
    from scipy.ndimage import shift
    N = data.shape[-1]
    s = (N / 2 + (b - 1) / 2) - np.asarray(zero, float)
    if np.abs(s).max() < 1e-3:
        return data
    return shift(data, (0, 0, s[0], s[1]), order=1, mode="constant").astype(np.float32)


def bin_patterns(data, b):
    """b x b binning with bins aligned so the zero frequency (native N/2 + (b-1)/2) is the centre of bin floor(N/2b)."""
    if b == 1:
        return data, data.shape[-1] // 2
    N = data.shape[-1]
    s0 = (N // 2) % b
    M = (N - s0) // b
    d = data[..., s0:s0 + M * b, s0:s0 + M * b]
    d = d.reshape(*d.shape[:-2], M, b, M, b).sum(axis=(-3, -1))
    return d, (N // 2 - s0) // b


def choose_K(M, dk_b, step, dr_target):
    """Even window size K <= M (binned detector) with dr = 1 / (K dk_b) near dr_target and step / dr near an integer."""
    K0 = 1.0 / (dr_target * dk_b)
    best = None
    for K in range(4, M + 1, 2):
        dr = 1.0 / (K * dk_b)
        err = abs(step / dr - round(step / dr)) / max(round(step / dr), 1)
        score = (err > 0.02, abs(K - K0) / K0 + 5 * err)
        if best is None or score < best[0]:
            best = (score, K)
    return best[1]


# ------------------------------------------------------------------------------------------------ S-matrix stage
def smatrix_stage(dsets, a, out, log):
    d0 = dsets[0]
    lam = sj.Setup(energy_ev=d0["energy"]).lam
    b = a.bin
    pats = []
    for d in dsets:
        x = d["data"]
        if a.scan_roi:
            i0, i1, j0, j1 = a.scan_roi
            x = x[i0:i1, j0:j1]
        x = centre_patterns(x, d["zero"], b)
        tot = x.sum(axis=(-2, -1)).mean()
        x, c = bin_patterns(x, b)
        pats.append((x, c, tot))
    M = pats[0][0].shape[-1]
    dk_b = b * float(d0["dk"][0])
    step = float(d0["step"][0])
    K = a.K or choose_K(M, dk_b, step, a.dr)
    dr = 1.0 / (K * dk_b)
    step_px = step / dr
    if abs(step_px - round(step_px)) > 0.02 * max(round(step_px), 1):
        log(f"warning: scan step {step:.3f} A = {step_px:.3f} S-grid px, rounded to {round(step_px)} "
            f"(position error up to {abs(step_px - round(step_px)) * dr * pats[0][0].shape[0]:.2f} A)")
    step_px = max(int(round(step_px)), 1)
    dps, captured = [], []
    for x, c, tot in pats:
        lo = c - K // 2
        if lo < 0 or lo + K > x.shape[-1]:
            raise ValueError(f"K = {K} does not fit the binned detector ({x.shape[-1]} px, centre {c})")
        w = x[..., lo:lo + K, lo:lo + K]
        captured.append(float(w.sum(axis=(-2, -1)).mean() / max(tot, 1e-30)))
        dps.append(np.fft.ifftshift(w, axes=(-2, -1)))
    dps = np.array(dps, np.float32)
    Ny, Nx = dps.shape[1:3]
    Y = (max(Ny, Nx) - 1) * step_px + K
    setup = sj.Setup(energy_ev=d0["energy"], alpha=1e-3 * d0["alpha_mrad"], dr=dr, K=K, Y=Y,
                     taper=1e-3 * d0.get("taper_mrad", 0.0))
    gy, gx = K // 2 + np.arange(Ny) * step_px, K // 2 + np.arange(Nx) * step_px
    positions = np.stack(np.meshgrid(gy, gx, indexing="ij"), -1)
    nominal = np.zeros((len(dsets), len(sj.NAMES)))
    nominal[:, 0] = [-d["defocus"] for d in dsets]
    T = a.thickness
    nb = len(setup.beams()[0])
    log(f"S-matrix: {len(dsets)} datasets x {Ny}x{Nx} positions, bin {b}, K {K}, dr {dr:.4f} A, window {K * dr:.2f} A, "
        f"Y {Y}, {nb} beams, step {step_px} px, lambda/alpha^2 {lam / setup.alpha ** 2:.1f} A, "
        f"captured intensity {np.round(captured, 3).tolist()}")
    checks = sj.check_geometry(setup, positions, nominal, T, log=log)
    if any(not c["ok"] for c in checks) and not a.force:
        raise SystemExit("geometry check failed (see above); fix the data / --bin / --dr / --scan-roi, or --force")
    amp, pos, didx = sj.prepare_data(dps, positions, device=a.device)
    model = sj.JointModel(setup, pos, didx, nominal, trainable=tuple(a.trainable.split(",")), ref=a.ref, device=a.device)
    t0 = time.time()
    hist = sj.fit(model, amp, epochs=a.epochs, warmup=a.warmup, batch=a.batch, lr_S=a.lr_S, mode=a.mode,
                  log=lambda m: log("  " + m) if a.verbose else None)
    coeff = model.coefficients().detach().cpu().numpy()
    log(f"S-matrix fit: {a.epochs} epochs, {time.time() - t0:.0f} s, final loss {hist[-1]:.5f}; "
        f"C10 per dataset {np.round(coeff[:, 0], 1).tolist()} A (nominal {np.round(nominal[:, 0], 1).tolist()})")

    dz = a.dz or max(0.25, min(2.0, T / 50))
    zg = np.arange(-0.2 * T, 1.2 * T + dz / 2, dz)
    vol = sj.refocus(model.S, setup, zg, device=a.device)                # (Nz, Y, Y) complex
    c0, c1 = K // 2, Y - K // 2 + 1
    norm = vol / vol.mean(axis=(1, 2), keepdims=True)
    phase = np.angle(norm)
    curve = sj.FOCUS_METRICS[a.focus_metric](vol[:, c0:c1, c0:c1])
    inside = (zg >= 0) & (zg <= T)
    layers = local_peaks(zg[inside], curve[inside], a.max_layers, frac=0.25, min_sep=0.5 * lam / setup.alpha ** 2)

    det, strength, thr = sj.detect_features(vol, setup, k=a.detect_k)
    det = det[:a.max_features]
    sigma_z = a.depth_sigma if a.depth_sigma else 0.3 * lam / setup.alpha ** 2
    templates = None
    if a.templates:
        t = np.load(a.templates)
        templates = (t["templates"], t["z0_grid"])
        if not np.allclose(t["depth_grid"], zg):
            raise ValueError("--templates were built on a different depth grid (use the same --dz and thickness)")
    feats = []
    for y, x in det:
        w = sj.footprint(setup, [(y, x)])
        r = sj.feature_response(vol, w)
        if templates is not None:
            z = sj.fit_depth(r, *templates)[0]
        else:
            prof = np.abs(np.angle(1 + r))
            prof[~inside] = 0
            i = int(np.argmax(prof))
            z = zg[i] + (parabolic(prof, i) * dz if 0 < i < len(zg) - 1 else 0.0)
        iz = int(np.argmin(np.abs(zg - z)))
        py, px = int(round(y / dr)), int(round(x / dr))
        feats.append(dict(y=float(y), x=float(x), z=float(np.clip(z, 0, T)),
                          phase=float(np.clip(phase[iz, py, px], 0.05, 1.0))))
    log(f"S-matrix depth: focus-curve layers {np.round(layers, 1).tolist()} A; {len(feats)} lateral features "
        f"(detection threshold {thr:.3f}), depths {np.round(sorted(f['z'] for f in feats), 1).tolist()} A, "
        f"sigma {sigma_z:.1f} A")

    o = os.path.join(out, "smatrix")
    os.makedirs(o, exist_ok=True)
    ky, kx = setup.beams()
    probes = np.zeros((len(dsets), K, K), complex)
    psi = model.probe().detach().cpu().numpy()
    iy, ix = np.round(ky * K * dr).astype(int) % K, np.round(kx * K * dr).astype(int) % K
    for d in range(len(dsets)):
        F = np.zeros((K, K), complex)
        F[iy, ix] = psi[d]
        probes[d] = np.fft.fftshift(np.fft.ifft2(F))
    np.save(os.path.join(o, "probe_real_space.npy"), probes.astype(np.complex64))
    np.save(os.path.join(o, "depth_grid.npy"), zg)
    np.save(os.path.join(o, "refocused_phase.npy"), phase[:, c0:c1, c0:c1].astype(np.float32))
    np.save(os.path.join(o, "focus_curve.npy"), np.stack([zg, curve]))
    np.save(os.path.join(o, "loss.npy"), np.array(hist))
    if a.save_S:
        np.save(os.path.join(o, "S.npy"), model.S.detach().cpu().numpy())
    info = dict(K=K, dr=dr, Y=Y, bin=b, step_px=step_px, beams=nb, epochs=a.epochs, final_loss=hist[-1],
                names=sj.NAMES, coefficients=coeff.tolist(), nominal=nominal.tolist(), ref=a.ref,
                layers=list(map(float, layers)), features=feats, depth_sigma=sigma_z, captured=captured,
                geometry=checks, crop=[c0, c1])
    json.dump(info, open(os.path.join(o, "smatrix.json"), "w"), indent=1, default=float)
    np.savez(os.path.join(o, "stage.npz"), phase=phase.astype(np.float32), curve=curve, psi=psi, probes=probes,
             energy=setup.energy_ev, alpha=setup.alpha, taper=setup.taper)  # for --reuse-smatrix
    return dict(setup=setup, coeff=coeff, nominal=nominal, zg=zg, phase=phase, curve=curve, layers=layers, feats=feats,
                sigma_z=sigma_z, probes=probes, crop=(c0, c1), psi=psi, ky=ky, kx=kx, hist=hist, info=info)


def load_smatrix(out):
    """S-matrix stage results saved by a previous run (--reuse-smatrix): rerun QuantEM without refitting."""
    o = os.path.join(out, "smatrix")
    info = json.load(open(os.path.join(o, "smatrix.json")))
    z = np.load(os.path.join(o, "stage.npz"))
    setup = sj.Setup(energy_ev=float(z["energy"]), alpha=float(z["alpha"]), dr=info["dr"], K=info["K"], Y=info["Y"],
                     taper=float(z["taper"]) if "taper" in z.files else 0.0)
    ky, kx = setup.beams()
    return dict(setup=setup, coeff=np.array(info["coefficients"]), nominal=np.array(info["nominal"]),
                zg=np.load(os.path.join(o, "depth_grid.npy")), phase=z["phase"], curve=z["curve"], layers=info["layers"],
                feats=info["features"], sigma_z=info["depth_sigma"], probes=z["probes"], crop=tuple(info["crop"]),
                psi=z["psi"], ky=ky, kx=kx, hist=list(np.load(os.path.join(o, "loss.npy"))), info=info)


def parabolic(y, i):
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    return 0.0 if d >= 0 else float(np.clip(0.5 * (y[i - 1] - y[i + 1]) / d, -0.5, 0.5))


def local_peaks(z, c, n, frac=0.25, min_sep=0.0):
    """Up to n local maxima of c(z) above frac of its range, at least min_sep apart, sorted by depth."""
    c = np.asarray(c, float)
    if len(c) < 3:
        return []
    lo, hi = c.min(), c.max()
    is_max = np.r_[c[0] > c[1], (c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:]), c[-1] > c[-2]]
    kept = []
    for i in np.flatnonzero(is_max)[np.argsort(-c[is_max])]:
        if c[i] < lo + frac * (hi - lo):
            break
        if all(abs(z[i] - z[j]) >= min_sep for j in kept):
            kept.append(i)
        if len(kept) == n:
            break
    return sorted(float(z[i]) for i in kept)


# ------------------------------------------------------------------------------------------------ QuantEM stage
ARMS = {                         # name: (slices, object, probe)
    "baseline": ("uniform", "flat", "nominal"),        # QuantEM on its own
    "sprobe": ("uniform", "flat", "smatrix"),          # only the S-matrix probe (isolates the probe hand-over)
    "sinit": ("smatrix", None, "smatrix"),             # everything from the S-matrix stage (object: --init-object)
}


def quantem_aberrations(coeff):
    """S-matrix aberrations (simple_joint, Cartesian, azimuth from the 2nd array axis) -> QuantEM polar coefficients
    (C_nm, phi_nm, azimuth from the 1st array axis).  phi_Q = pi/2 - phi_S, so a cos(m phi_S) + b sin(m phi_S) =
    A cos(m phi_Q) + B sin(m phi_Q) with A = a cos(m pi/2) + b sin(m pi/2), B = a sin(m pi/2) - b cos(m pi/2).
    Checked against quantem.complex_probe.fourier_space_probe (tests/test_run_realization.py)."""
    out = {}
    for nm, (n, m, comp) in sj.ABERRATIONS.items():
        v = float(coeff[sj.NAMES.index(nm)])
        if comp is None:
            out[f"C{n}{m}"] = v
        elif comp == "a":
            b = float(coeff[sj.NAMES.index(nm[:-1] + "b")])
            A = v * math.cos(m * math.pi / 2) + b * math.sin(m * math.pi / 2)
            B = v * math.sin(m * math.pi / 2) - b * math.cos(m * math.pi / 2)
            out[f"C{n}{m}"] = math.hypot(A, B)
            out[f"phi{n}{m}"] = math.atan2(B, A) / m
    return out


def smatrix_aberrations(q):
    """Inverse of quantem_aberrations for the coefficients present in q -> {simple_joint name: value}."""
    out = {}
    for nm, (n, m, comp) in sj.ABERRATIONS.items():
        C = float(q.get(f"C{n}{m}", 0.0))
        if comp is None:
            if f"C{n}{m}" in q:
                out[nm] = C
        elif comp == "a" and f"C{n}{m}" in q:
            phi = float(q.get(f"phi{n}{m}", 0.0))
            A, B = C * math.cos(m * phi), C * math.sin(m * phi)
            c, s = math.cos(m * math.pi / 2), math.sin(m * math.pi / 2)
            out[nm], out[nm[:-1] + "b"] = A * c + B * s, A * s - B * c
    return out


def volume_potential(sm, grid, shape, dr_q, origin, power):
    """Initial potential from the refocused S-matrix volume: projected phase map P (refocused phase at mid-depth,
    clipped >= 0, zero outside the scanned field) split over the slabs by each pixel's depth weight |phase(z)|^power.
    Projected phase is conserved per pixel."""
    from scipy.ndimage import map_coordinates
    setup, zg, ph = sm["setup"], sm["zg"], sm["phase"]
    c0, c1 = sm["crop"]
    T = grid.thickness
    inside = (zg >= 0) & (zg <= T)
    P = np.zeros(ph.shape[1:])
    P[c0:c1, c0:c1] = np.clip(ph[int(np.argmin(np.abs(zg - T / 2))), c0:c1, c0:c1], 0, None)
    W = np.abs(ph[inside]) ** power
    zi = zg[inside]
    b = grid.boundaries
    F = np.stack([W[(zi >= b[k]) & ((zi < b[k + 1]) | (k == grid.num_slices - 1))].sum(0) for k in range(grid.num_slices)])
    F /= np.maximum(F.sum(0, keepdims=True), 1e-30)
    rows = (origin[0] + np.arange(shape[0]) * dr_q) / setup.dr
    cols = (origin[1] + np.arange(shape[1]) * dr_q) / setup.dr
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([map_coordinates(F[k] * P, [yy, xx], order=1, cval=0.0) for k in range(grid.num_slices)])


def quantem_stage(d, sm, a, out, log):
    from quantem.core.datastructures import Dataset4dstem
    from quantem.core.ml import OptimizerParams
    from quantem.diffractive_imaging import (PtychographyDatasetRaster, ProbePixelated, ProbeParametric,
                                             DetectorPixelated, Ptychography, ObjectPixelated)
    setup, T = sm["setup"], a.thickness
    lam = setup.lam
    x = d["data"]
    if a.scan_roi:
        i0, i1, j0, j1 = a.scan_roi
        x = x[i0:i1, j0:j1]
    x = centre_patterns(x, d["zero"], a.quantem_bin)
    x, _ = bin_patterns(x, a.quantem_bin)
    dk = a.quantem_bin * float(d["dk"][0])
    s = float(d["step"][0])
    dset = Dataset4dstem.from_array(x, name="realization", sampling=(s, s, 1e3 * lam * dk, 1e3 * lam * dk),
                                    units=("A", "A", "mrad", "mrad"))
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", probe_energy=d["energy"], force_com_rotation=0,
                     force_com_transpose=False, plot_rotation=False, plot_com=False)
    qd = a.quantem_dataset
    names = sj.NAMES
    trained = [nm for nm in a.trainable.split(",")]
    s_coeff = np.zeros(len(names))
    for nm in set(trained) | {"C10"}:
        s_coeff[names.index(nm)] = sm["coeff"][qd, names.index(nm)]
    probe_sets = {"smatrix": quantem_aberrations(s_coeff),
                  "nominal": quantem_aberrations(np.r_[-d["defocus"], np.zeros(len(names) - 1)])}
    keep = {"C10"} | {f"{p}{nm[1:3]}" for nm in trained if nm != "C10" for p in ("C", "phi")}
    probe_sets = {k: {q: v for q, v in p.items() if q in keep} for k, p in probe_sets.items()}

    def ptycho(obj, seed, which):
        params = {"energy": d["energy"], "semiangle_cutoff": d["alpha_mrad"], **probe_sets[which]}
        if a.probe_model == "parametric":
            probe = ProbeParametric.from_params(probe_params=params, learn_aberrations=True)
        else:
            probe = ProbePixelated.from_params(num_probes=1, probe_params=params)
        pt = Ptychography.from_models(dset=pdset, obj_model=obj, probe_model=probe, detector_model=DetectorPixelated(),
                                      device=a.qdevice, rng=seed)
        pt.preprocess(obj_padding_px=(PAD, PAD), batch_size=a.quantem_batch)
        return pt

    feats = sm["feats"]
    cands = [qe.DepthCandidate(y=f["y"], x=f["x"], modes=[(f["z"], sm["sigma_z"], 1.0)], phase=f["phase"]) for f in feats]
    layer_cands = [qe.DepthCandidate(y=0.0, x=0.0, modes=[(z, sm["sigma_z"], 1.0)]) for z in sm["layers"]]
    dz_max = a.dz_max or T / 12
    dz_min = a.dz_min or T / 40
    g_s = qe.slice_grid(cands + layer_cands, T, dz_max=dz_max, dz_min=dz_min, extra_slices=a.extra_slices)
    n = g_s.num_slices
    g_u = qe.SliceGrid(boundaries=np.linspace(0.0, T, n + 1), thickness=T)
    if n < 2:
        raise ValueError("fewer than 2 slices; lower --dz-max")
    probe_pt = ptycho(ObjectPixelated.from_uniform(num_slices=n, slice_thicknesses=float(g_u.slice_thicknesses[0]),
                                                   obj_type="potential"), 0, "nominal")
    shape = tuple(int(v) for v in probe_pt.obj_shape_full[1:])
    dr_q = float(np.asarray(probe_pt.sampling)[0])
    p0 = probe_pt.dset.scan_positions_px.detach().cpu().numpy().min(axis=0)
    origin = (setup.K / 2 * setup.dr - p0[0] * dr_q, setup.K / 2 * setup.dr - p0[1] * dr_q)
    ky, kx = sm["ky"], sm["kx"]
    s_probe = probe_at_beams(sm["probes"][qd], setup.dr, ky, kx)
    warm = a.iters // 5 if a.probe_warmup is None else a.probe_warmup
    plr = a.probe_lr if a.probe_lr is not None else (0.2 if a.probe_model == "parametric" else 1e-4)
    log(f"QuantEM: dataset {qd} (focus {d['defocus']:.1f} A nominal), object {shape} px at {dr_q:.4f} A, {n} planes, "
        f"{a.probe_model} probe; optimiser: object Adam lr {a.qlr} for {a.iters} iterations, probe Adam lr {plr} from "
        f"iteration {warm + 1}")
    log(f"  probe hand-over (QuantEM convention): S-matrix {fmt(probe_sets['smatrix'])}; nominal {fmt(probe_sets['nominal'])}")
    log(f"  S-informed slab thicknesses {np.round(g_s.slab_thicknesses, 1).tolist()} A")
    if plr <= 0:
        log("  warning: probe learning rate 0 -> QuantEM will NOT update the probe")
    res = {}
    for arm in a.arms.split(","):
        slices, objinit, which = ARMS[arm]
        objinit = objinit or a.init_object
        grid = g_s if slices == "smatrix" else g_u
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        phi = np.full((grid.num_slices,) + shape, BASE)
        if objinit == "candidates" and cands:
            phi += qe.initial_potential(cands, grid, shape, dr_q, origin=origin)
        elif objinit == "volume":
            phi += volume_potential(sm, grid, shape, dr_q, origin, a.volume_power)
        pt = ptycho(qe.to_quantem(phi, grid), a.seed, which)
        p_init = np.asarray(pt.probe)[0].copy()
        handover = probe_match(probe_at_beams(p_init, dr_q, ky, kx), s_probe, ky, kx)
        t0 = time.time()
        cons = {"object": {"positivity": True}}
        if warm > 0:
            pt.reconstruct(num_iters=min(warm, a.iters), reset=True, batch_size=a.quantem_batch, device=a.qdevice,
                           optimizer_params={"object": OptimizerParams.Adam(lr=a.qlr)}, constraints=cons)
        if a.iters > warm:
            opt = {"object": OptimizerParams.Adam(lr=a.qlr)}
            if plr > 0:
                opt["probe"] = OptimizerParams.Adam(lr=plr)
            pt.reconstruct(num_iters=a.iters - warm, reset=warm <= 0, batch_size=a.quantem_batch, device=a.qdevice,
                           optimizer_params=opt, constraints=cons)
        obj = pt.obj_model.obj.detach().cpu().numpy()
        losses = np.asarray(pt.iter_losses, float)
        probe = np.asarray(pt.probe)[0]
        change = 1.0 - probe_match(probe_at_beams(probe, dr_q, ky, kx), probe_at_beams(p_init, dr_q, ky, kx), ky, kx)[0]
        learned = None
        if a.probe_model == "parametric":
            learned = {k: float(v.detach().cpu()) for k, v in pt.probe_model.aberration_coefs.items()}
        o = os.path.join(out, f"quantem_{arm}")
        os.makedirs(o, exist_ok=True)
        np.save(os.path.join(o, "object_potential.npy"), obj.astype(np.float32))
        np.save(os.path.join(o, "initial_potential.npy"), phi.astype(np.float32))
        np.save(os.path.join(o, "slice_boundaries.npy"), grid.boundaries)
        np.save(os.path.join(o, "probe_real_space.npy"), probe.astype(np.complex64))
        np.save(os.path.join(o, "initial_probe_real_space.npy"), p_init.astype(np.complex64))
        np.save(os.path.join(o, "loss.npy"), losses)
        json.dump(dict(slices=slices, object=objinit, probe=which, probe_model=a.probe_model,
                       probe_params=probe_sets[which], learned_aberrations=learned,
                       learned_smatrix_convention=smatrix_aberrations(learned) if learned else None,
                       handover_overlap=handover[0], probe_change=change, iters=a.iters, probe_warmup=warm,
                       object_lr=a.qlr, probe_lr=plr), open(os.path.join(o, "arm.json"), "w"), indent=1)
        prof = depth_profile(obj, grid, field_px(p0, shape, d, a, dr_q))
        res[arm] = dict(obj=obj, grid=grid, probe=probe, p_init=p_init, losses=losses, profile=prof,
                        time=time.time() - t0, spec=(slices, objinit, which), handover=handover, change=change,
                        learned=learned, peaks=local_peaks(grid.planes + grid.slab_thicknesses / 2, prof, a.max_layers, frac=0.5))
        log(f"  {arm:8s} [{slices} slices, {objinit} object, {which} probe] loss {losses[-1]:.5f}, "
            f"{time.time() - t0:.0f} s; initial probe vs S-matrix probe overlap {handover[0]:.4f}; probe changed by "
            f"{change:.2e} (1 - overlap); profile peaks {np.round(res[arm]['peaks'], 1).tolist()} A"
            + (f"; learned {fmt(learned)}" if learned else ""))
    return dict(arms=res, dr=dr_q, p0=p0, shape=shape, origin=origin, g_s=g_s, g_u=g_u, dataset=qd,
                probe_sets=probe_sets, warm=warm, plr=plr)


def fmt(q):
    return ", ".join(f"{k} {v:.1f} A" if k.startswith("C") else f"{k} {math.degrees(v):.0f} deg" for k, v in q.items())


def field_px(p0, shape, d, a, dr_q):
    """Scanned field in QuantEM pixels (rows, cols)."""
    ny, nx = d["data"].shape[:2] if not a.scan_roi else (a.scan_roi[1] - a.scan_roi[0], a.scan_roi[3] - a.scan_roi[2])
    s = float(d["step"][0])
    return (int(p0[0]), int(min(shape[0], p0[0] + (ny - 1) * s / dr_q + 1)),
            int(p0[1]), int(min(shape[1], p0[1] + (nx - 1) * s / dr_q + 1)))


def depth_profile(obj, grid, f):
    """Lateral std of the potential inside the scanned field, per slab, per A of slab thickness."""
    y0, y1, x0, x1 = f
    return obj[:, y0:y1, x0:x1].std(axis=(1, 2)) / grid.slab_thicknesses


# ------------------------------------------------------------------------------------------------ ground truth
def load_truth(paths, log):
    import zarr
    tr = {}
    for p in paths:
        f = os.path.join(p, "potential.zarr")
        if "potential" not in tr and os.path.isdir(f):
            m = json.load(open(os.path.join(f, "zarr.json")))["attributes"]["metadata0"]
            ax = m.get("axes", {})
            samp = [float(ax[k]["sampling"]) for k in sorted(ax) if "sampling" in ax[k]]
            st = m.get("kwargs", {}).get("slice_thickness", {})
            st = st.get("_value", st) if isinstance(st, dict) else st
            pot = zarr.open_array(os.path.join(f, "array0"), mode="r")[...].astype(np.float32)
            tr["potential"] = (pot, samp[0], np.asarray(st, float))
        g = os.path.join(p, "probe.zarr")
        if os.path.isdir(g):
            m = json.load(open(os.path.join(g, "zarr.json")))["attributes"]["metadata0"]
            ax = m.get("axes", {})
            samp = [float(ax[k]["sampling"]) for k in sorted(ax) if "sampling" in ax[k]]
            tr.setdefault("probes", []).append((zarr.open_array(os.path.join(g, "array0"), mode="r")[...], samp[0]))
        else:
            tr.setdefault("probes", []).append(None)
        h = os.path.join(p, "atoms.extxyz")
        if "atoms" not in tr and os.path.exists(h):
            with open(h) as fh:
                n = int(fh.readline())
                fh.readline()
                rows = [fh.readline().split() for _ in range(n)]
            tr["atoms"] = ([r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows]))
    log(f"truth: potential {'yes' if 'potential' in tr else 'no'}, probes "
        f"{sum(p is not None for p in tr.get('probes', []))}, atoms {'yes' if 'atoms' in tr else 'no'}")
    return tr


def sigma_interaction(energy_ev):
    lam = sj.Setup(energy_ev=energy_ev).lam
    m0c2 = 5.109989461e5
    return 2 * np.pi / (lam * energy_ev) * (m0c2 + energy_ev) / (2 * m0c2 + energy_ev)


def sample_truth(pot, dr_t, rows_A, cols_A, z_lo, z_hi, slice_t, smooth_to):
    """Projected truth phase-potential over slices with centre in [z_lo, z_hi), sampled at cell coordinates."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    zc = np.cumsum(slice_t) - slice_t / 2
    sel = (zc >= z_lo) & (zc < z_hi)
    img = pot[sel].sum(axis=0) if sel.any() else np.zeros(pot.shape[1:])
    if smooth_to > dr_t:
        img = gaussian_filter(img, 0.5 * smooth_to / dr_t, mode="wrap")
    yy, xx = np.meshgrid(np.asarray(rows_A) / dr_t, np.asarray(cols_A) / dr_t, indexing="ij")
    return map_coordinates(img, [yy, xx], order=1, mode="grid-wrap")


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-30))


def probe_at_beams(p, dr, ky, kx):
    """Fourier coefficients of a real-space probe at the beams, after moving its (circular) intensity centroid to r = 0."""
    p = np.asarray(p)
    N = p.shape
    I = np.abs(p) ** 2
    c = [N[i] / (2 * np.pi) * np.angle((I.sum(axis=1 - i) * np.exp(2j * np.pi * np.arange(N[i]) / N[i])).sum())
         for i in (0, 1)]
    F = np.fft.fft2(p)
    iy, ix = np.round(ky * N[0] * dr).astype(int) % N[0], np.round(kx * N[1] * dr).astype(int) % N[1]
    return F[iy, ix] * np.exp(2j * np.pi * (iy * c[0] / N[0] + ix * c[1] / N[1]))


def probe_match(est, ref, ky, kx):
    """Overlap |<est, ref>| / (|est| |ref|) and RMS phase error [rad] inside the aperture after removing piston and tilt."""
    w = np.abs(ref) ** 2
    dphi = np.angle(ref * np.conj(est))
    dphi = np.angle(np.exp(1j * (dphi - np.angle((w * np.exp(1j * dphi)).sum()))))
    A = np.stack([np.ones_like(ky), ky, kx], 1) * np.sqrt(w)[:, None]
    c = np.linalg.lstsq(A, dphi * np.sqrt(w), rcond=None)[0]
    resid = dphi - (c[0] + c[1] * ky + c[2] * kx)
    rms = float(np.sqrt((w * resid ** 2).sum() / w.sum()))
    est2 = est * np.exp(1j * (c[0] + c[1] * ky + c[2] * kx))
    ov = float(np.abs(np.vdot(est2, ref)) / np.sqrt((np.abs(est2) ** 2).sum() * w.sum()))
    return ov, rms


# ------------------------------------------------------------------------------------------------ report
def report(dsets, sm, qm, tr, a, out, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup, T, K, dr = sm["setup"], a.thickness, sm["setup"].K, sm["setup"].dr
    c0, c1 = sm["crop"]
    d0 = dsets[0]
    pos0 = None
    if d0["positions_A"] is not None:
        P = d0["positions_A"]
        if a.scan_roi:
            P = P[a.scan_roi[0]:a.scan_roi[1], a.scan_roi[2]:a.scan_roi[3]]
        pos0 = P[0, 0]
    s_to_cell = lambda ys: pos0 + (np.asarray(ys) - K / 2 * dr) if pos0 is not None else None
    L = ["# Realization reconstruction: S-matrix + probe (joint) vs QuantEM multislice", "",
         f"Data: {len(dsets)} defocus datasets ({', '.join(os.path.basename(os.path.normpath(d['path'])) for d in dsets)}), "
         f"{d0['energy'] / 1e3:.0f} kV, {d0['alpha_mrad']:.1f} mrad, thickness {T:g} A, scan step {d0['step'][0]:g} A.", ""]
    summary = dict(smatrix=sm["info"])

    # probes ---------------------------------------------------------------------------------------------------
    L += ["## Probe", "", "| dataset | nominal C10 | S-matrix C10 | S-matrix C12a, C12b | metadata C10 | "
          "S-matrix probe vs truth (overlap, RMS phase err rad) |", "|---|---|---|---|---|---|"]
    ky, kx = sm["ky"], sm["kx"]
    i10, i12a, i12b = sj.NAMES.index("C10"), sj.NAMES.index("C12a"), sj.NAMES.index("C12b")
    probe_rows = []
    for i, d in enumerate(dsets):
        t = tr.get("probes", [None] * len(dsets))[i] if tr else None
        m = "-"
        if t is not None:
            ov, rms = probe_match(probe_at_beams(sm["probes"][i], dr, ky, kx), probe_at_beams(t[0], t[1], ky, kx), ky, kx)
            m = f"{ov:.4f}, {rms:.3f}"
            probe_rows.append(dict(dataset=i, smatrix_overlap=ov, smatrix_rms=rms))
        L.append(f"| {i}{' (ref)' if i == a.ref else ''} | {sm['nominal'][i, i10]:.1f} | {sm['coeff'][i, i10]:.1f} | "
                 f"{sm['coeff'][i, i12a]:.1f}, {sm['coeff'][i, i12b]:.1f} | {-d['defocus']:.1f} | {m} |")
    if qm:
        qd = qm["dataset"]
        t = tr.get("probes", [None] * len(dsets))[qd] if tr else None
        L += ["", "## What QuantEM was started from", "",
              "| QuantEM input | " + " | ".join(qm["arms"]) + " |", "|---|" + "---|" * len(qm["arms"])]
        cells = lambda f: " | ".join(f(r) for r in qm["arms"].values())
        L += [f"| probe | " + cells(lambda r: f"{r['spec'][2]}: {fmt(qm['probe_sets'][r['spec'][2]])}") + " |",
              f"| probe hand-over check (initial QuantEM probe vs S-matrix probe overlap) | "
              + cells(lambda r: f"{r['handover'][0]:.4f}") + " |",
              f"| slice planes | " + cells(lambda r: f"{r['spec'][0]}, {r['grid'].num_slices} slabs") + " |",
              f"| initial object | " + cells(lambda r: r["spec"][1]) + " |",
              f"| geometry | " + cells(lambda r: f"{qm['dr']:.3f} A px, z = 0 entrance, defocus = -C10") + " |",
              f"| optimiser | " + cells(lambda r: f"object lr {a.qlr} x {a.iters} it; {a.probe_model} probe lr "
                                                   f"{qm['plr']} from it {qm['warm'] + 1}") + " |",
              f"| probe changed by QuantEM (1 - overlap initial/final) | " + cells(lambda r: f"{r['change']:.2e}") + " |"]
        if any(r["learned"] for r in qm["arms"].values()):
            L.append("| learned aberrations (parametric) | " + cells(lambda r: fmt(r["learned"]) if r["learned"] else "-") + " |")
        L += ["", f"QuantEM results (dataset {qd}):", ""]
        for arm, r in qm["arms"].items():
            line = f"* {arm}: final loss {r['losses'][-1]:.5f}"
            if t is not None:
                ref = probe_at_beams(t[0], t[1], ky, kx)
                ov0, rms0 = probe_match(probe_at_beams(r["p_init"], qm["dr"], ky, kx), ref, ky, kx)
                ov, rms = probe_match(probe_at_beams(r["probe"], qm["dr"], ky, kx), ref, ky, kx)
                line += (f"; probe vs truth: start overlap {ov0:.4f} / RMS phase {rms0:.3f} rad -> "
                         f"end {ov:.4f} / {rms:.3f} rad")
                probe_rows.append(dict(dataset=qd, arm=arm, start_overlap=ov0, start_rms=rms0, overlap=ov, rms=rms))
            L.append(line)
    summary["probe"] = probe_rows

    fig, ax = plt.subplots(2, 2 + (len(qm["arms"]) if qm else 0) + (1 if tr.get("probes") else 0),
                           figsize=(16, 7), squeeze=False)
    qd = qm["dataset"] if qm else a.ref
    items = []
    t = tr.get("probes", [None] * len(dsets))[qd] if tr else None
    if t is not None:
        items.append(("truth", centre_crop(np.fft.fftshift(t[0]), t[1], K * dr), K * dr))
    items.append((f"S-matrix d{qd}", sm["probes"][qd], K * dr))
    items.append((f"S-matrix d{a.ref} (ref)" if qd != a.ref else "S-matrix (other)",
                  sm["probes"][a.ref if qd != a.ref else (qd + 1) % len(dsets)], K * dr))
    if qm:
        for arm, r in qm["arms"].items():
            p = r["probe"]
            items.append((f"QuantEM {arm}", centre_crop(np.roll(p, np.subtract(p.shape, np.unravel_index(
                np.argmax(np.abs(p)), p.shape)) - np.array(p.shape) // 2, (0, 1)), qm["dr"], K * dr), K * dr))
    for j, (name, p, ext) in enumerate(items[:ax.shape[1]]):
        e = [-ext / 2, ext / 2, ext / 2, -ext / 2]
        ax[0, j].imshow(np.abs(p), extent=e, cmap="magma")
        ax[0, j].set_title(f"{name} |probe|")
        ax[1, j].imshow(np.angle(p) * (np.abs(p) > 0.05 * np.abs(p).max()), extent=e, cmap="twilight")
        ax[1, j].set_title("phase (rad), axes in A")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "probes.png"), dpi=110)
    plt.close(fig)

    # slice locations ------------------------------------------------------------------------------------------
    L += ["", "## Slice locations (depth, A, z = 0 entrance)", "",
          f"* S-matrix focus-curve layers ({a.focus_metric}): {np.round(sm['layers'], 1).tolist()}",
          f"* S-matrix lateral features: {len(sm['feats'])}, depths "
          f"{np.round(sorted(f['z'] for f in sm['feats']), 1).tolist()} (support sigma {sm['sigma_z']:.1f} A)"]
    if qm:
        L += [f"* QuantEM {arm} depth-profile peaks: {np.round(r['peaks'], 1).tolist()}" for arm, r in qm["arms"].items()]
        L += ["", "Slices used to initialise QuantEM:", "",
              f"* uniform: {qm['g_u'].num_slices} slabs of {qm['g_u'].slab_thicknesses[0]:.2f} A",
              f"* S-informed: {qm['g_s'].num_slices} slabs, boundaries {np.round(qm['g_s'].boundaries, 1).tolist()}"]
    truth_peaks = None
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    zg = sm["zg"]
    nrm = lambda v: (np.asarray(v) - np.min(v)) / max(np.ptp(v), 1e-30)
    ax.plot(zg, nrm(sm["curve"]), "k-", label=f"S-matrix focus curve ({a.focus_metric})")
    for f in sm["feats"]:
        ax.axvline(f["z"], color="k", ls=":", lw=0.8)
    if qm:
        for (arm, r), c in zip(qm["arms"].items(), ("tab:blue", "tab:orange", "tab:red", "tab:purple")):
            g = r["grid"]
            ax.step(np.r_[g.planes, T], np.r_[nrm(r["profile"]), nrm(r["profile"])[-1]], where="post", color=c,
                    label=f"QuantEM {arm} (lateral std / A)")
        for b in qm["g_s"].boundaries:
            ax.axvline(b, ymin=0, ymax=0.04, color="tab:red", lw=1)
    if "potential" in tr and pos0 is not None:
        pot, dr_t, st = tr["potential"]
        f0 = np.round((pos0 - 0) / dr_t).astype(int)
        span = int((len(range(c0, c1)) - 1) * dr / dr_t)
        reg = pot[:, f0[0]:f0[0] + span + 1, f0[1]:f0[1] + span + 1]
        tp = reg.std(axis=(1, 2)) / st
        zc = np.cumsum(st) - st / 2
        ax.plot(zc, nrm(tp), color="tab:green", label="truth potential (lateral std / A)")
        truth_peaks = local_peaks(zc, tp, a.max_layers, frac=0.5)
        L.append(f"* truth potential depth-profile peaks: {np.round(truth_peaks, 1).tolist()}")
    if "atoms" in tr and pos0 is not None:
        sym, xyz = tr["atoms"]
        lo = pos0
        hi = pos0 + (c1 - c0 - 1) * dr
        m = (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
        if m.any():
            ax.hist(xyz[m, 2], bins=np.linspace(0, T, 51), weights=np.full(m.sum(), 1.0 / max(m.sum(), 1)) * 3,
                    color="tab:green", alpha=0.25, label="atoms in scanned field (z histogram)")
            elems = sorted(set(np.asarray(sym)[m]))
            L.append(f"* atoms in the scanned field: {int(m.sum())} ({', '.join(elems)}), z range "
                     f"{xyz[m, 2].min():.1f}-{xyz[m, 2].max():.1f} A")
    ax.axvspan(-0.2 * T, 0, color="0.9")
    ax.axvspan(T, 1.2 * T, color="0.9")
    ax.set_xlabel("depth z (A), 0 = entrance")
    ax.set_ylabel("normalised")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "depth_profiles.png"), dpi=110)
    plt.close(fig)
    summary["depth"] = dict(smatrix_layers=sm["layers"], smatrix_features=sm["feats"], truth_peaks=truth_peaks,
                            quantem={k: r["peaks"] for k, r in qm["arms"].items()} if qm else None)

    # object images --------------------------------------------------------------------------------------------
    L += ["", "## Object", ""]
    zg = sm["zg"]
    show = sorted(set([float(z) for z in sm["layers"]] + [T / 2]))
    panels = [(f"S-matrix phase z={z:.0f}", sm["phase"][int(np.argmin(np.abs(zg - z))), c0:c1, c0:c1], dr) for z in show]
    if qm:
        for arm, r in qm["arms"].items():
            y0, y1, x0, x1 = field_px(qm["p0"], qm["shape"], dsets[qm["dataset"]], a, qm["dr"])
            panels.append((f"QuantEM {arm} projected", r["obj"][:, y0:y1, x0:x1].sum(0), qm["dr"]))
    rows = []
    if "potential" in tr and pos0 is not None:
        pot, dr_t, st = tr["potential"]
        sig = sigma_interaction(d0["energy"])
        for name, img, s in panels:
            ny, nx = img.shape
            tz = sample_truth(pot, dr_t, pos0[0] + np.arange(ny) * s, pos0[1] + np.arange(nx) * s, 0, T, st, s) * sig
            rows.append((name, corr(img, tz)))
        tz = sample_truth(pot, dr_t, pos0[0] + np.arange(c1 - c0) * dr, pos0[1] + np.arange(c1 - c0) * dr, 0, T, st, dr) * sig
        panels.insert(0, ("truth projected phase (rad)", tz, dr))
        L += ["Correlation with the true projected phase over the scanned field:", "",
              "| image | correlation |", "|---|---|"] + [f"| {n} | {c:.3f} |" for n, c in rows]
        if qm:
            L += ["", "Per-slab correlation of QuantEM slices with the true potential in the same slab (mean over slabs "
                  "that contain potential):", ""]
            for arm, r in qm["arms"].items():
                g = r["grid"]
                y0, y1, x0, x1 = field_px(qm["p0"], qm["shape"], dsets[qm["dataset"]], a, qm["dr"])
                cs = []
                for k in range(g.num_slices):
                    img = r["obj"][k, y0:y1, x0:x1]
                    tz = sample_truth(pot, dr_t, pos0[0] + np.arange(img.shape[0]) * qm["dr"],
                                      pos0[1] + np.arange(img.shape[1]) * qm["dr"], g.boundaries[k], g.boundaries[k + 1], st, qm["dr"])
                    if tz.std() > 1e-6 * max(abs(tz).max(), 1e-30):
                        cs.append(corr(img, tz))
                L.append(f"* {arm}: {np.mean(cs):.3f} over {len(cs)} slabs" if cs else f"* {arm}: no slab with potential")
                rows.append((f"QuantEM {arm} per-slab mean", float(np.mean(cs)) if cs else float("nan")))
    summary["object_correlation"] = dict(rows)
    fig, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4), squeeze=False)
    for j, (name, img, s) in enumerate(panels):
        ax[0, j].imshow(img, cmap="gray", extent=[0, img.shape[1] * s, img.shape[0] * s, 0])
        ax[0, j].set_title(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "objects.png"), dpi=110)
    plt.close(fig)

    # slice montages -------------------------------------------------------------------------------------------
    zs = np.linspace(0, T, 9)
    montage(plt, [sm["phase"][int(np.argmin(np.abs(zg - z))), c0:c1, c0:c1] for z in zs],
            [f"z={z:.0f}" for z in zs], dr, os.path.join(out, "smatrix", "sections.png"), "S-matrix refocused phase")
    if qm:
        for arm, r in qm["arms"].items():
            g = r["grid"]
            y0, y1, x0, x1 = field_px(qm["p0"], qm["shape"], dsets[qm["dataset"]], a, qm["dr"])
            montage(plt, [r["obj"][k, y0:y1, x0:x1] for k in range(g.num_slices)],
                    [f"{g.boundaries[k]:.0f}-{g.boundaries[k + 1]:.0f}" for k in range(g.num_slices)], qm["dr"],
                    os.path.join(out, f"quantem_{arm}", "slices.png"), f"QuantEM {arm} potential per slab (A)")
    L += ["", "## How the two differ", "",
          "* S-matrix + probe (joint): one linear operator S for the whole specimen, fitted to all defocus datasets "
          "at once together with each dataset's probe aberrations. No slices are assumed; depth comes afterwards from "
          "refocusing S, so it is a measurement with a depth resolution of about lambda/alpha^2 "
          f"({setup.lam / setup.alpha ** 2:.0f} A here), not a reconstructed potential.",
          "* QuantEM multislice: a potential per slice on a fixed slice grid, fitted to one dataset with a probe model. "
          "Depth is where the optimiser puts the potential; with uniform slices nothing tells it where features are, "
          "the S-informed start (sinit) gives it the S-matrix depths, slice spacing and probe defocus.",
          "", "Files: smatrix/ (probe_real_space.npy, refocused_phase.npy, focus_curve.npy, smatrix.json, sections.png), "
          "quantem_<arm>/ (object_potential.npy, initial_potential.npy, slice_boundaries.npy, probe_real_space.npy, "
          "initial_probe_real_space.npy, loss.npy, arm.json, slices.png), probes.png, depth_profiles.png, objects.png, summary.json."]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L) + "\n")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    log("\n".join(L))


def centre_crop(p, s, ext):
    """Crop a centred image with pixel size s to about ext x ext A (or keep it if smaller)."""
    n = int(round(ext / s))
    c = np.array(p.shape) // 2
    if n >= min(p.shape):
        return p
    return p[c[0] - n // 2:c[0] - n // 2 + n, c[1] - n // 2:c[1] - n // 2 + n]


def montage(plt, imgs, titles, s, path, sup):
    n = len(imgs)
    c = min(n, 6)
    r = int(math.ceil(n / c))
    fig, ax = plt.subplots(r, c, figsize=(2.4 * c, 2.5 * r), squeeze=False)
    vmin, vmax = np.percentile(np.array([i.ravel() for i in imgs]), [1, 99.5])
    for j in range(r * c):
        a = ax[j // c, j % c]
        a.axis("off")
        if j < n:
            a.imshow(imgs[j], cmap="gray", vmin=vmin, vmax=vmax)
            a.set_title(titles[j], fontsize=8)
    fig.suptitle(sup)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="one folder per defocus (realization layout)")
    ap.add_argument("--out", default="results_realization")
    ap.add_argument("--thickness", type=float, default=None, help="specimen thickness [A] (default: cell z from metadata)")
    ap.add_argument("--energy", type=float, default=None, help="[eV], overrides metadata")
    ap.add_argument("--alpha", type=float, default=None, help="semiangle [mrad], overrides metadata")
    ap.add_argument("--taper", type=float, default=None,
                    help="soft aperture edge [mrad] (simple_joint.Setup.taper), overrides metadata (default 0 = hard)")
    ap.add_argument("--defocus", type=lambda s: [float(v) for v in s.split(",")], default=None,
                    help="focus depth per dataset [A] (= -C10), overrides metadata")
    ap.add_argument("--scan-roi", type=lambda s: [int(v) for v in s.split(",")], default=None, help="i0,i1,j0,j1")
    ap.add_argument("--bin", type=int, default=1, help="detector binning for the S-matrix stage")
    ap.add_argument("--dr", type=float, default=0.2, help="target S-grid pixel [A] (sets the detector crop K)")
    ap.add_argument("--K", type=int, default=None, help="detector crop after binning (overrides --dr)")
    ap.add_argument("--ref", type=int, default=0, help="gauge dataset: its defocus is kept at the nominal value")
    ap.add_argument("--trainable", default="C10,C12a,C12b")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-S", dest="lr_S", type=float, default=0.02)
    ap.add_argument("--mode", default="simultaneous", choices=["simultaneous", "alternating"])
    ap.add_argument("--dz", type=float, default=None, help="refocus depth step [A]")
    ap.add_argument("--focus-metric", default="phase_var", choices=list(sj.FOCUS_METRICS))
    ap.add_argument("--max-layers", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--detect-k", type=float, default=4.0,
                    help="feature detection threshold (MADs; stand-in: 6 missed one of two clusters, 4 found both)")
    ap.add_argument("--depth-sigma", type=float, default=None, help="depth support width [A] (default 0.3 lambda/alpha^2)")
    ap.add_argument("--templates", default=None, help="npz with templates, z0_grid, depth_grid for template depth fits")
    ap.add_argument("--save-S", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if the geometry check fails")
    ap.add_argument("--no-quantem", action="store_true")
    ap.add_argument("--quantem-dataset", type=int, default=None, help="default: focus nearest mid-specimen")
    ap.add_argument("--quantem-bin", type=int, default=1)
    ap.add_argument("--arms", default="baseline,sinit", help=f"comma list of {list(ARMS)}")
    ap.add_argument("--init-object", default="candidates", choices=["candidates", "volume", "flat"],
                    help="sinit object: Gaussians at the S-matrix features, or the refocused S-matrix volume")
    ap.add_argument("--volume-power", type=float, default=4.0, help="depth weight |phase(z)|^p for --init-object volume")
    ap.add_argument("--probe-model", default="pixelated", choices=["pixelated", "parametric"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--qlr", type=float, default=5e-3, help="QuantEM object learning rate")
    ap.add_argument("--probe-lr", type=float, default=None,
                    help="QuantEM probe learning rate (default 1e-4 pixelated, 0.2 A parametric; 0 = fixed probe). "
                         "Stand-in: 1e-3 degraded a good S-matrix probe from 0.04 to 0.21 rad RMS, 1e-4 kept 0.07")
    ap.add_argument("--probe-warmup", type=int, default=None, help="object-only iterations first (default iters/5)")
    ap.add_argument("--reuse-smatrix", action="store_true", help="load the S-matrix stage from --out instead of refitting")
    ap.add_argument("--quantem-batch", type=int, default=64)
    ap.add_argument("--dz-max", type=float, default=None, help="thickest S-informed slab [A] (default T/12)")
    ap.add_argument("--dz-min", type=float, default=None, help="thinnest S-informed slab [A] (default T/40)")
    ap.add_argument("--extra-slices", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="print the S-matrix loss every epoch")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.qdevice = "gpu" if a.device.startswith("cuda") else "cpu"
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(m):
        if m is None:
            return
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    dsets = [load_dataset(p, a, i) for i, p in enumerate(a.data)]
    shapes = {d["data"].shape for d in dsets}
    if len(shapes) > 1:
        raise ValueError(f"datasets differ in shape: {shapes}")
    if a.thickness is None:
        if dsets[0]["cell"] is None:
            raise ValueError("thickness not in the metadata; pass --thickness")
        a.thickness = float(dsets[0]["cell"][-1])
    for i, d in enumerate(dsets):
        log(f"dataset {i}: {d['path']}: scan {d['data'].shape[:2]}, detector {d['data'].shape[2:]}, step {d['step'][0]:g} A, "
            f"dk {d['dk'][0]:g} 1/A, {d['energy'] / 1e3:g} kV, {d['alpha_mrad']:g} mrad, focus depth {d['defocus']:g} A")
    if a.quantem_dataset is None:
        a.quantem_dataset = int(np.argmin([abs(d["defocus"] - a.thickness / 2) for d in dsets]))
    if a.reuse_smatrix:
        sm = load_smatrix(a.out)
        log(f"S-matrix stage loaded from {a.out}/smatrix (C10 {np.round(sm['coeff'][:, 0], 1).tolist()} A)")
    else:
        sm = smatrix_stage(dsets, a, a.out, log)
    qm = None
    if not a.no_quantem:
        qm = quantem_stage(dsets[a.quantem_dataset], sm, a, a.out, log)
    tr = load_truth(a.data, log)
    report(dsets, sm, qm, tr, a, a.out, log)


if __name__ == "__main__":
    main()
