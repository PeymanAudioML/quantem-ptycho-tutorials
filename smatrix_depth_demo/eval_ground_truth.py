"""Ground-truth evaluation of SAVED S-matrix and QuantEM results (no reconstruction is run).

    python eval_ground_truth.py --run results_real0_2slice --data data/realization_0/4dstem_quantem_noisefree \
        --truth data/realization_0 --out results_real0_2slice/ground_truth_comparison

Reads <run>/smatrix/ and <run>/quantem_<arm>/ written by run_realization.py, and the truth (potential.zarr, probe.zarr,
atoms.extxyz) from --truth.  Reports what is missing instead of failing.  Four comparisons:
  1 projected object   (QuantEM potential = physical phase [rad]; S-matrix refocused phase = contrast only)
  2 depth              (truth per-slab phase and atoms' z vs QuantEM slabs and the S-matrix focus curve)
  3 probe              (normalised overlap and RMS phase error vs probe.zarr)
  4 atomic structure   (atoms overlaid, lateral localisation of image maxima; z reported separately)
"""
import argparse, json, os
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

import run_realization as rr
import simple_joint as sj

ARMS = ("baseline", "sprobe", "sinit")
SMOOTH_A = (0.0, 0.5, 1.0, 1.5, 2.0)          # truth blur (Gaussian sigma, A) when matching a finite-resolution image
MARGINS_A = (0.0, 10.0)                        # exclude this border of the scanned field (probe radius ~ 10 A)


def fit_stats(img, ref):
    """Correlation, least-squares img = s*ref + b, RMSE raw and after the fit, NRMSE = fitted RMSE / std(img)."""
    c = rr.corr(img, ref)
    A = np.stack([ref.ravel(), np.ones(ref.size)], 1)
    (s, b), *_ = np.linalg.lstsq(A, img.ravel(), rcond=None)
    return dict(corr=c, slope=float(s), offset=float(b), rmse_raw=float(np.sqrt(np.mean((img - ref) ** 2))),
                rmse_fit=float(np.sqrt(np.mean((img - (s * ref + b)) ** 2))),
                nrmse_fit=float(np.sqrt(np.mean((img - (s * ref + b)) ** 2)) / max(img.std(), 1e-30)))


def interior(img, s, margin):
    m = int(round(margin / s))
    return img if m == 0 else img[m:-m, m:-m]


def truth_image(tr, rows, cols, z0, z1, sigma_A, sigma_int, swap=False):
    pot, dr_t, st = tr["potential"]
    if swap:
        pot = pot.transpose(0, 2, 1)
    return rr.sample_truth(pot, dr_t, rows, cols, z0, z1, st, 2 * sigma_A) * sigma_int


def frc(a, b, s, nbin=40):
    """Fourier ring correlation of two same-size images (pixel s A), Hann-windowed.  Returns k [1/A], FRC(k)."""
    n = min(a.shape)
    a, b = a[:n, :n] - a[:n, :n].mean(), b[:n, :n] - b[:n, :n].mean()
    w = np.outer(np.hanning(n), np.hanning(n))
    Fa, Fb = np.fft.fftshift(np.fft.fft2(a * w)), np.fft.fftshift(np.fft.fft2(b * w))
    ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n, s)), np.fft.fftshift(np.fft.fftfreq(n, s)), indexing="ij")
    k = np.hypot(ky, kx)
    edges = np.linspace(0, 0.5 / s, nbin + 1)
    idx = np.digitize(k.ravel(), edges) - 1
    num = np.bincount(idx, (Fa * np.conj(Fb)).real.ravel(), nbin + 1)[:nbin]
    den = np.sqrt(np.bincount(idx, (np.abs(Fa) ** 2).ravel(), nbin + 1)[:nbin] * np.bincount(idx, (np.abs(Fb) ** 2).ravel(), nbin + 1)[:nbin])
    return 0.5 * (edges[1:] + edges[:-1]), num / np.maximum(den, 1e-30)


def detect_peaks(img, s, min_sep_A=1.0, rel=0.5):
    """Local maxima of img (pixel size s) at least min_sep_A apart and above rel * (max - median) over the median."""
    n = max(3, int(round(min_sep_A / s)) | 1)
    mx = maximum_filter(img, size=n, mode="nearest")
    thr = np.median(img) + rel * (img.max() - np.median(img))
    iy, ix = np.nonzero((img == mx) & (img > thr))
    return np.stack([iy, ix], 1) * s


def localisation(img, s, origin, cols_xy, max_match=1.5):
    """Lateral error [A] from each projected atomic column to the nearest image maximum, and the unmatched maxima."""
    pk = detect_peaks(img, s) + np.asarray(origin)[None]
    if len(pk) == 0 or len(cols_xy) == 0:
        return dict(n_columns=len(cols_xy), n_peaks=len(pk))
    d = np.linalg.norm(cols_xy[:, None, :] - pk[None], axis=2)
    e = d.min(axis=1)
    back = d.min(axis=0)
    rng = np.random.default_rng(0)
    rnd = pk.min(axis=0) + rng.random((len(pk), 2)) * (pk.max(axis=0) - pk.min(axis=0))
    e_rand = np.linalg.norm(cols_xy[:, None, :] - rnd[None], axis=2).min(axis=1)
    return dict(n_columns=int(len(cols_xy)), n_peaks=int(len(pk)), median_err=float(np.median(e)),
                median_err_random_points=float(np.median(e_rand)),
                peak_to_column_median=float(np.median(back)),
                peak_to_column_random_points=float(np.median(np.linalg.norm(cols_xy[None] - rnd[:, None], axis=2).min(axis=1))),
                peak_to_column_frac_le_0p5=float(np.mean(back <= 0.5)),
                frac_le_0p5=float(np.mean(e <= 0.5)), frac_le_1p0=float(np.mean(e <= 1.0)),
                frac_le_1p5=float(np.mean(e <= 1.5)), false_peaks_gt_1p5=int(np.sum(back > max_match)))


def columns(xyz, tol=0.4):
    """Merge atoms that project onto the same lateral position (within tol A) into columns."""
    taken = np.zeros(len(xyz), bool)
    cols = []
    for i in range(len(xyz)):
        if taken[i]:
            continue
        m = (np.linalg.norm(xyz[:, :2] - xyz[i, :2], axis=1) < tol) & ~taken
        taken |= m
        cols.append(xyz[m, :2].mean(axis=0))
    return np.array(cols)


def build_quantem_geometry(d0, sm, a):
    """QuantEM pixel size and the pixel of the first scan position (as in run_realization.quantem_stage, no fitting)."""
    from quantem.core.datastructures import Dataset4dstem
    from quantem.diffractive_imaging import (PtychographyDatasetRaster, ProbePixelated, DetectorPixelated, Ptychography,
                                             ObjectPixelated)
    x = rr.centre_patterns(d0["data"], d0["zero"], 1)
    x, _ = rr.bin_patterns(x, 1)
    lam = sm["setup"].lam
    s, dk = float(d0["step"][0]), float(d0["dk"][0])
    dset = Dataset4dstem.from_array(x, name="g", sampling=(s, s, 1e3 * lam * dk, 1e3 * lam * dk),
                                    units=("A", "A", "mrad", "mrad"))
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", probe_energy=d0["energy"], force_com_rotation=0,
                     force_com_transpose=False, plot_rotation=False, plot_com=False)
    probe = ProbePixelated.from_params(num_probes=1, probe_params={"energy": d0["energy"], "defocus": d0["defocus"],
                                                                    "semiangle_cutoff": d0["alpha_mrad"]})
    pt = Ptychography.from_models(dset=pdset, obj_model=ObjectPixelated.from_uniform(num_slices=2, slice_thicknesses=3.0,
                                                                                    obj_type="potential"),
                                  probe_model=probe, detector_model=DetectorPixelated(), device="cpu", rng=0)
    pt.preprocess(obj_padding_px=(rr.PAD, rr.PAD), batch_size=64)
    return float(np.asarray(pt.sampling)[0]), pt.dset.scan_positions_px.detach().cpu().numpy().min(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "ground_truth_comparison")
    os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines, M, missing = [], {}, []
    log = lambda m: lines.append(m)

    # ---- inputs -----------------------------------------------------------------------------------------------
    args = SimpleNamespace(energy=None, alpha=None, defocus=None, scan_roi=None)
    d0 = rr.load_dataset(a.data, args, 0)
    T = float(d0["cell"][-1])
    sm = rr.load_smatrix(a.run)
    setup, dr_s = sm["setup"], sm["setup"].dr
    c0, c1 = sm["crop"]
    tr = rr.load_truth([a.truth], lambda *_: None)
    for k in ("potential", "atoms"):
        if k not in tr:
            missing.append(f"truth {k} (not found in {a.truth})")
    if not tr.get("probes") or tr["probes"][0] is None:
        missing.append("truth probe")
    sig = rr.sigma_interaction(d0["energy"])
    pos0 = d0["positions_A"][0, 0]
    ny_scan, nx_scan = d0["data"].shape[:2]
    step = float(d0["step"][0])
    arms = {}
    for arm in ARMS:
        o = os.path.join(a.run, f"quantem_{arm}")
        if os.path.exists(os.path.join(o, "object_potential.npy")):
            arms[arm] = dict(
                obj=np.load(os.path.join(o, "object_potential.npy")), bnd=np.load(os.path.join(o, "slice_boundaries.npy")),
                probe=np.load(os.path.join(o, "probe_real_space.npy")),
                p_init=np.load(os.path.join(o, "initial_probe_real_space.npy")), loss=np.load(os.path.join(o, "loss.npy")),
                info=json.load(open(os.path.join(o, "arm.json"))))
        else:
            missing.append(f"quantem_{arm}/ (arm not run or not saved in {a.run})")
    dr_q, p0 = build_quantem_geometry(d0, sm, a) if arms else (None, None)
    log("# Ground-truth comparison (saved results only; nothing was re-reconstructed)\n")
    log(f"Run: `{a.run}`.  Data: `{a.data}` ({ny_scan}x{nx_scan} scan, {step:g} A step, {d0['energy'] / 1e3:g} kV, "
        f"{d0['alpha_mrad']:g} mrad, focus {d0['defocus']:g} A, noise-free).  Specimen: Si, **{T:g} A thick**, one defocus.\n")
    log("## Missing outputs\n")
    log("\n".join(f"* {m}" for m in missing) or "* none")
    log("* No 40-slice QuantEM result exists (that run was stopped before saving).")
    log("* sprobe and sinit were never run for this dataset, so only the QuantEM **baseline** (2 uniform slabs, nominal "
        "probe, parametric probe updated) can be evaluated.\n")

    # orientation sanity: potential array axes (x, y) vs atoms ------------------------------------------------------
    swap = False
    if "potential" in tr and "atoms" in tr:
        pot, dr_t, st = tr["potential"]
        xyz = tr["atoms"][1]
        pr = pot.sum(0)
        ix, iy = np.round(xyz[:, 0] / dr_t).astype(int) % pr.shape[0], np.round(xyz[:, 1] / dr_t).astype(int) % pr.shape[1]
        v_xy, v_yx = pr[ix, iy].mean(), pr[iy % pr.shape[0], ix % pr.shape[1]].mean()
        swap = bool(v_yx > v_xy)
        M["orientation"] = dict(mean_pot_at_atoms_axis1_is_x=float(v_xy), mean_pot_at_atoms_axis1_is_y=float(v_yx),
                                transposed_truth_used=swap)
        log(f"Truth orientation check: mean projected potential at the atoms is {v_xy:.3g} with array axes (x, y) and "
            f"{v_yx:.3g} transposed -> {'transposed' if swap else 'axes (x, y)'} used.\n")

    # ---- 1. projected object --------------------------------------------------------------------------------------
    log("## 1. Projected object\n")
    log("Physical potential: the QuantEM object is a phase per slice in rad (obj_type potential); its sum over slices is "
        "compared with the true projected potential times the interaction constant "
        f"(sigma = {sig:.4g} rad/(V A), assuming potential.zarr holds V*A per slice).  The S-matrix refocused phase is "
        "**not** a physical potential (relative phase of a refocused wave): only the correlation is meaningful, not the "
        "scale or the error in rad.\n")
    imgs = {}                                        # name -> (image, pixel A, kind)
    if arms:
        y0, y1, x0, x1 = rr.field_px(p0, arms[next(iter(arms))]["obj"].shape[1:], d0, args, dr_q)
        for arm, r in arms.items():
            imgs[f"QuantEM {arm}"] = (r["obj"][:, y0:y1, x0:x1].sum(0), dr_q, "physical")
    zg = sm["zg"]
    ph = sm["phase"][:, c0:c1, c0:c1]
    best_z = None
    if "potential" in tr:
        rows = pos0[0] + np.arange(c1 - c0) * dr_s
        ref0 = truth_image(tr, rows, rows, 0, T, 0.5, sig, swap)
        cz = [abs(rr.corr(ph[i], ref0)) for i in range(len(zg))]
        best_z = int(np.argmax(cz))
        for zq in (0.0, T / 2, T):
            i = int(np.argmin(np.abs(zg - zq)))
            imgs[f"S-matrix refocused phase z={zg[i]:.0f}"] = (ph[i], dr_s, "contrast")
        imgs[f"S-matrix refocused phase z={zg[best_z]:.0f} (best |corr|)"] = (ph[best_z], dr_s, "contrast")
    tab = {}
    if "potential" in tr and imgs:
        for name, (img, s, kind) in imgs.items():
            rws = pos0[0] + np.arange(img.shape[0]) * s
            cls = pos0[1] + np.arange(img.shape[1]) * s
            tab[name] = dict(kind=kind)
            for mg in MARGINS_A:
                for sg in SMOOTH_A:
                    t = truth_image(tr, rws, cls, 0, T, max(sg, 0.0), sig, swap)
                    tab[name][f"margin{mg:g}_sigma{sg:g}"] = fit_stats(interior(img, s, mg), interior(t, s, mg))
            t_sw = truth_image(tr, rws, cls, 0, T, 1.0, sig, not swap)
            tab[name]["corr_if_truth_axes_swapped_sigma1"] = rr.corr(img, t_sw)
        M["projected"] = tab
        log("Correlation with the true projected potential (full scanned field; blur sigma = Gaussian width applied to "
            "the truth to mimic the reconstruction's resolution):\n")
        log("| image | kind | " + " | ".join(f"σ={s:g} Å" for s in SMOOTH_A) + " |")
        log("|---|---|" + "---|" * len(SMOOTH_A))
        for name, v in tab.items():
            log(f"| {name} | {v['kind']} | " + " | ".join(f"{v[f'margin0_sigma{s:g}']['corr']:.3f}" for s in SMOOTH_A) + " |")
        log("\nSame, interior only (10 Å border removed):\n")
        log("| image | " + " | ".join(f"σ={s:g} Å" for s in SMOOTH_A) + " |")
        log("|---|" + "---|" * len(SMOOTH_A))
        for name, v in tab.items():
            log(f"| {name} | " + " | ".join(f"{v[f'margin10_sigma{s:g}']['corr']:.3f}" for s in SMOOTH_A) + " |")
        log("\nQuantEM physical-scale agreement (full field; truth blurred by the σ with the highest correlation):\n")
        log("| image | best σ (Å) | corr | slope (recon = s·truth + b) | offset (rad) | RMSE raw (rad) | RMSE after fit (rad) | "
            "NRMSE after fit |")
        log("|---|---|---|---|---|---|---|---|")
        for name, v in tab.items():
            if v["kind"] != "physical":
                continue
            bs = max(SMOOTH_A, key=lambda s: v[f"margin0_sigma{s:g}"]["corr"])
            f = v[f"margin0_sigma{bs:g}"]
            log(f"| {name} | {bs:g} | {f['corr']:.3f} | {f['slope']:.2f} | {f['offset']:.3f} | {f['rmse_raw']:.3f} | "
                f"{f['rmse_fit']:.3f} | {f['nrmse_fit']:.2f} |")
            v["best_sigma"] = bs
        # figure
        names = list(imgs)
        fig, ax = plt.subplots(3, 1 + len(names), figsize=(3.3 * (1 + len(names)), 9.6), squeeze=False)
        s0 = dr_q or dr_s
        rws = pos0[0] + np.arange(int((ny_scan - 1) * step / s0) + 1) * s0
        ext = lambda s, n: [0, n * s, n * s, 0]
        tref = truth_image(tr, rws, rws, 0, T, 1.0, sig, swap)
        ax[0, 0].imshow(tref, cmap="gray", vmin=np.percentile(tref, 1), vmax=np.percentile(tref, 99), extent=ext(s0, tref.shape[0]))
        ax[0, 0].set_title("truth projected phase (rad)\nblur σ=1 Å", fontsize=8)
        ax[1, 0].axis("off"), ax[2, 0].axis("off")
        for j, name in enumerate(names, 1):
            img, s, kind = imgs[name]
            lo, hi = np.percentile(img, [1, 99])
            ax[0, j].imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=ext(s, img.shape[0]))
            ax[0, j].set_title(name + ("\n(phase, rad)" if kind == "physical" else "\n(contrast, not a potential)"),
                               fontsize=7)
            if kind == "physical":
                t = truth_image(tr, pos0[0] + np.arange(img.shape[0]) * s, pos0[1] + np.arange(img.shape[1]) * s, 0, T,
                                tab[name]["best_sigma"], sig, swap)
                f = tab[name][f"margin0_sigma{tab[name]['best_sigma']:g}"]
                dm = img - (f["slope"] * t + f["offset"])
                v = np.percentile(np.abs(dm), 99)
                ax[1, j].imshow(img - t, cmap="RdBu_r", vmin=-np.percentile(np.abs(img - t), 99), vmax=np.percentile(np.abs(img - t), 99),
                                extent=ext(s, img.shape[0]))
                ax[1, j].set_title("recon − truth (rad, raw)", fontsize=7)
                im = ax[2, j].imshow(dm, cmap="RdBu_r", vmin=-v, vmax=v, extent=ext(s, img.shape[0]))
                ax[2, j].set_title("recon − (s·truth+b) (rad)", fontsize=7)
            else:
                ax[1, j].axis("off"), ax[2, j].axis("off")
                ax[1, j].text(0.5, 0.5, "no difference map:\nnot in physical units", ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "1_projected_object.png"), dpi=110)
        plt.close(fig)
        log("\nFigure: `1_projected_object.png` (top: images; middle: raw difference; bottom: difference after the "
            "fitted scale and offset).")

    # ---- 2. depth ---------------------------------------------------------------------------------------------------
    log("\n## 2. Depth\n")
    D = {}
    if "potential" in tr:
        pot, dr_t, st = tr["potential"]
        zc = np.cumsum(st) - st / 2
        f0 = np.round(np.asarray(pos0) / dr_t).astype(int)
        span = int((ny_scan - 1) * step / dr_t)
        reg = (pot.transpose(0, 2, 1) if swap else pot)[:, f0[0]:f0[0] + span + 1, f0[1]:f0[1] + span + 1]
        tmean = reg.mean(axis=(1, 2)) * sig                          # mean phase per truth slab (same units as the projection)
        tstd = reg.std(axis=(1, 2)) / st
        D["truth_slab_mean_phase"] = dict(z_centres=zc.tolist(), mean_phase=tmean.tolist())
        log(f"Truth: {len(st)} slabs of {st[0]:g} Å in the scanned field; mean projected phase per slab "
            f"{np.round(tmean, 3).tolist()} rad (total {tmean.sum():.3f} rad).")
        if "atoms" in tr:
            xyz = tr["atoms"][1]
            m = (xyz[:, 0] >= pos0[0]) & (xyz[:, 0] <= pos0[0] + (ny_scan - 1) * step) & \
                (xyz[:, 1] >= pos0[1]) & (xyz[:, 1] <= pos0[1] + (nx_scan - 1) * step)
            D["atoms_in_field"] = dict(n=int(m.sum()), z_mean=float(xyz[m, 2].mean()), z_std=float(xyz[m, 2].std()),
                                       z_min=float(xyz[m, 2].min()), z_max=float(xyz[m, 2].max()))
            log(f"Truth atoms in the scanned field: {int(m.sum())}, z = {xyz[m, 2].mean():.2f} ± {xyz[m, 2].std():.2f} Å "
                f"(range {xyz[m, 2].min():.2f}-{xyz[m, 2].max():.2f}).")
        for arm, r in arms.items():
            g = r["bnd"]
            sl = r["obj"][:, y0:y1, x0:x1]
            rows_q = pos0[0] + np.arange(sl.shape[1]) * dr_q
            cols_q = pos0[1] + np.arange(sl.shape[2]) * dr_q
            per = []
            for k in range(len(g) - 1):
                tk = truth_image(tr, rows_q, cols_q, g[k], g[k + 1], 1.0, sig, swap)
                per.append(dict(z=[float(g[k]), float(g[k + 1])], recon_mean_phase=float(sl[k].mean()),
                                truth_mean_phase=float(tk.mean()), corr=rr.corr(sl[k], tk)))
            D[f"quantem_{arm}"] = per
            log(f"\nQuantEM {arm}: {len(g) - 1} slabs.  Mean phase per slab (recon vs truth, rad) and in-slab correlation:\n")
            log("| slab (Å) | recon mean | truth mean | in-slab corr |")
            log("|---|---|---|---|")
            for p in per:
                log(f"| {p['z'][0]:.1f}–{p['z'][1]:.1f} | {p['recon_mean_phase']:.3f} | {p['truth_mean_phase']:.3f} | "
                    f"{p['corr']:.3f} |")
            log("\nTwo 3 Å slabs of a 6 Å sheet with a 20 mrad probe: the split between slabs is **not** constrained by the "
                "data (depth resolution ≈ λ/α² ≈ %.0f Å), so these per-slab numbers are not depth measurements." %
                (setup.lam / setup.alpha ** 2))
    curve = np.asarray(sm["curve"], float)
    nrm = (curve - curve.min()) / max(np.ptp(curve), 1e-30)
    inside = (zg >= 0) & (zg <= T)
    pk = float(zg[int(np.argmax(curve))])
    half = zg[nrm >= 0.5]
    D["smatrix_focus_curve"] = dict(peak_z=pk, fwhm_region=[float(half.min()), float(half.max())] if len(half) else None,
                                    resolution_lambda_over_alpha2=float(setup.lam / setup.alpha ** 2),
                                    detected_layers=sm["layers"], feature_depths=sorted(f["z"] for f in sm["feats"])[:5],
                                    n_features=len(sm["feats"]))
    log(f"\nS-matrix focus curve: maximum at z = {pk:.1f} Å over a scanned depth range of {zg.min():.0f}–{zg.max():.0f} Å; "
        f"the region above half maximum spans {half.min():.0f}–{half.max():.0f} Å.  The expected depth resolution is "
        f"λ/α² = {setup.lam / setup.alpha ** 2:.0f} Å, >> the {T:g} Å specimen.  The {len(sm['feats'])} detected lateral "
        f"features all carry depth ≈ {np.median([f['z'] for f in sm['feats']]):.2f} Å and the detected layer at "
        f"{sm['layers']}: **these are not credible z-coordinates** (they sit at the edge of the search range and the sample "
        "is thinner than the resolution).")
    M["depth"] = D
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(zg, nrm, "k-", label="S-matrix focus curve (normalised)")
    ax[0].axvspan(0, T, color="tab:green", alpha=0.15, label=f"specimen 0–{T:g} Å")
    ax[0].set_xlabel("depth z (Å)"), ax[0].legend(fontsize=8), ax[0].set_title("S-matrix refocusing vs depth")
    if "potential" in tr:
        ax[1].step(np.r_[0, np.cumsum(st)], np.r_[tmean, tmean[-1]], where="post", color="tab:green", label="truth (per slab)")
        for arm, r in arms.items():
            g = r["bnd"]
            v = [p["recon_mean_phase"] for p in D[f"quantem_{arm}"]]
            ax[1].step(g, np.r_[v, v[-1]], where="post", label=f"QuantEM {arm}")
        ax[1].set_xlabel("depth z (Å)"), ax[1].set_ylabel("mean phase per slab (rad)"), ax[1].legend(fontsize=8)
        ax[1].set_title("phase per slab in the scanned field")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2_depth.png"), dpi=110)
    plt.close(fig)
    log("\nFigure: `2_depth.png`.")

    # ---- 3. probe -----------------------------------------------------------------------------------------------------
    log("\n## 3. Probe\n")
    P = {}
    ky, kx = sm["ky"], sm["kx"]
    tp = tr.get("probes", [None])[0]
    if tp is not None:
        ref = rr.probe_at_beams(tp[0], tp[1], ky, kx)
        rows_p = [("S-matrix (dataset 0 = gauge reference: kept at nominal)", sm["probes"][0], dr_s)]
        for arm, r in arms.items():
            rows_p += [(f"QuantEM {arm}: start", r["p_init"], dr_q), (f"QuantEM {arm}: final", r["probe"], dr_q)]
        log("| probe | overlap with truth | RMS phase error (rad) |")
        log("|---|---|---|")
        for name, p, s in rows_p:
            ov, rms = rr.probe_match(rr.probe_at_beams(p, s, ky, kx), ref, ky, kx)
            P[name] = dict(overlap=ov, rms_phase=rms)
            log(f"| {name} | {ov:.4f} | {rms:.3f} |")
        for arm, r in arms.items():
            P[f"learned_aberrations_{arm}"] = r["info"].get("learned_aberrations")
            log(f"\nQuantEM {arm} learned aberrations: {r['info'].get('learned_aberrations')} "
                f"(truth metadata: C10 = {-d0['defocus']:g} Å, C12 = 0).")
        log("\nOverlap = |<est, truth>| / (|est||truth|) on the aperture beams, after removing piston and tilt from the "
            "phase and re-centring the intensity centroid (same definition as run_realization.py).")
        panels = [("truth", rr.centre_crop(np.fft.fftshift(tp[0]), tp[1], setup.K * dr_s), setup.K * dr_s),
                  ("S-matrix", sm["probes"][0], setup.K * dr_s)]
        for arm, r in arms.items():
            p = r["probe"]
            panels.append((f"QuantEM {arm} final", rr.centre_crop(np.roll(p, np.subtract(p.shape, np.unravel_index(
                np.argmax(np.abs(p)), p.shape)) - np.array(p.shape) // 2, (0, 1)), dr_q, setup.K * dr_s), setup.K * dr_s))
        fig, ax = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 6.4), squeeze=False)
        for j, (n, p, e) in enumerate(panels):
            ee = [-e / 2, e / 2, e / 2, -e / 2]
            ax[0, j].imshow(np.abs(p), extent=ee, cmap="magma"), ax[0, j].set_title(f"{n} |probe|", fontsize=8)
            ax[1, j].imshow(np.angle(p) * (np.abs(p) > 0.05 * np.abs(p).max()), extent=ee, cmap="twilight")
            ax[1, j].set_title("phase (rad); axes in Å", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "3_probe.png"), dpi=110)
        plt.close(fig)
        log("\nFigure: `3_probe.png`.")
    else:
        log("truth probe missing.")
    M["probe"] = P

    # ---- 4. atoms -------------------------------------------------------------------------------------------------------
    log("\n## 4. Atomic structure\n")
    A = {}
    if "atoms" in tr:
        xyz = tr["atoms"][1]
        m = (xyz[:, 0] >= pos0[0]) & (xyz[:, 0] <= pos0[0] + (ny_scan - 1) * step) & \
            (xyz[:, 1] >= pos0[1]) & (xyz[:, 1] <= pos0[1] + (nx_scan - 1) * step)
        sel = xyz[m]
        inner = (sel[:, 0] >= pos0[0] + 10) & (sel[:, 0] <= pos0[0] + (ny_scan - 1) * step - 10) & \
                (sel[:, 1] >= pos0[1] + 10) & (sel[:, 1] <= pos0[1] + (nx_scan - 1) * step - 10)
        cols = columns(sel[inner])
        A["n_atoms_in_field"], A["n_columns_interior"] = int(len(sel)), int(len(cols))
        log(f"Scanned field: {len(sel)} atoms; {len(cols)} projected columns in the interior (10 Å border excluded; atoms "
            "within 0.4 Å laterally merged).  Atoms are plotted as given (x = array axis 1 of the data/potential, y = axis 2); "
            "lateral localisation = distance from each projected column to the nearest local maximum of the image.\n")
        log("Two directions.  **Maxima → columns** (how far each image maximum is from the nearest atomic column; the "
            "meaningful one when maxima are blobs of several atoms) and **columns → maxima** (how far each column is from "
            "a maximum; limited by density since there are far more columns than maxima).\n")
        log("| image | columns | maxima | maxima→column median (Å) | random points→column (Å) | maxima within 0.5 Å of a column | "
            "columns→maximum median (Å) | random points (Å) | ≤1.0 Å |")
        log("|---|---|---|---|---|---|---|---|---|")
        items = dict(imgs)
        if "potential" in tr:
            items = {"truth projected (blur σ=0.5 Å, benchmark)": (
                truth_image(tr, pos0[0] + np.arange(int((ny_scan - 1) * step / 0.2) + 1) * 0.2,
                            pos0[1] + np.arange(int((nx_scan - 1) * step / 0.2) + 1) * 0.2, 0, T, 0.5, sig, swap), 0.2,
                "benchmark"), **items}
        for name, (img, s, kind) in items.items():
            mg = int(round(10.0 / s))
            core = img[mg:-mg, mg:-mg]
            res = localisation(gaussian_filter(core, 0.5 / s), s, pos0 + 10.0, cols)
            A[name] = res
            if "median_err" in res:
                log(f"| {name} | {res['n_columns']} | {res['n_peaks']} | {res['peak_to_column_median']:.2f} | "
                    f"{res['peak_to_column_random_points']:.2f} | {res['peak_to_column_frac_le_0p5']:.2f} | "
                    f"{res['median_err']:.2f} | {res['median_err_random_points']:.2f} | {res['frac_le_1p0']:.2f} |")
        sh = list(items.items())
        fig, ax = plt.subplots(1, len(sh), figsize=(4.0 * len(sh), 4.2), squeeze=False)
        for j, (name, (img, s, kind)) in enumerate(sh):
            ax[0, j].imshow(img, cmap="gray", extent=[pos0[1], pos0[1] + img.shape[1] * s, pos0[0] + img.shape[0] * s, pos0[0]])
            ax[0, j].plot(sel[:, 1], sel[:, 0], "r.", ms=2.5)
            ax[0, j].set_xlim(pos0[1], pos0[1] + img.shape[1] * s), ax[0, j].set_ylim(pos0[0] + img.shape[0] * s, pos0[0])
            ax[0, j].set_title(name, fontsize=7)
        fig.suptitle("atoms (red) at their given (x → rows, y → columns); image x/y in Å", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "4_atoms_overlay.png"), dpi=110)
        plt.close(fig)
        # zoom
        zc0 = pos0 + np.array([60.0, 60.0])
        fig, ax = plt.subplots(1, len(sh), figsize=(4.0 * len(sh), 4.2), squeeze=False)
        for j, (name, (img, s, kind)) in enumerate(sh):
            r0, c0_ = ((zc0 - pos0) / s).astype(int)
            w = int(24 / s)
            crop = img[r0:r0 + w, c0_:c0_ + w]
            ax[0, j].imshow(crop, cmap="gray", extent=[zc0[1], zc0[1] + 24, zc0[0] + 24, zc0[0]])
            mm = (sel[:, 0] >= zc0[0]) & (sel[:, 0] <= zc0[0] + 24) & (sel[:, 1] >= zc0[1]) & (sel[:, 1] <= zc0[1] + 24)
            ax[0, j].plot(sel[mm, 1], sel[mm, 0], "r.", ms=7)
            ax[0, j].set_title(name, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "4_atoms_overlay_zoom.png"), dpi=110)
        plt.close(fig)
        log("\nThe column density (about %.2f columns/Å², mean spacing ~%.1f Å) is much higher than the number of "
            "distinguishable image maxima, so this nearest-maximum distance is limited by density, not only by the "
            "reconstruction: compare with the truth benchmark and the random-points column.  Image maxima are blobs of several "
            "atoms (see the zoom figure); atoms inside a blob (e.g. dumbbells) are not separated." % (len(cols) / ((ny_scan - 1) * step - 20) ** 2, 1 / np.sqrt(len(cols) / ((ny_scan - 1) * step - 20) ** 2)))
        if "potential" in tr and "QuantEM baseline" in imgs:
            img, s, _ = imgs["QuantEM baseline"]
            t = truth_image(tr, pos0[0] + np.arange(img.shape[0]) * s, pos0[1] + np.arange(img.shape[1]) * s, 0, T, 0.0, sig, swap)
            k, f = frc(img, t, s)
            hi = np.flatnonzero((f < 0.5) & (k > 0.3))
            A["frc_quantem_baseline"] = dict(k=k.tolist(), frc=f.tolist(),
                                              first_k_above_0p3_with_frc_lt_0p5=float(k[hi[0]]) if len(hi) else None)
            qs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)
            log("\nFourier ring correlation, QuantEM baseline vs true projected potential (unblurred truth): " +
                ", ".join(f"{f[np.argmin(abs(k - q))]:.2f} at {q:g} Å⁻¹" for q in qs) +
                (f".  First crossing below 0.5 above 0.3 Å⁻¹: {k[hi[0]]:.2f} Å⁻¹ (period {1 / k[hi[0]]:.1f} Å)." if len(hi) else "."))
            log("Si lattice spacings: (111) 3.14 Å, (220) 1.92 Å, (311) 1.64 Å.")
        log("\nFigures: `4_atoms_overlay.png`, `4_atoms_overlay_zoom.png` (24 Å crop).")
        log("\n**z-coordinates (reported separately):** the true atoms span z = %.2f–%.2f Å (mean %.2f Å). No reconstruction "
            "here provides a credible z estimate: the QuantEM objects have two 3 Å slabs (the split between slabs is "
            "unconstrained, see section 2) and the S-matrix depths are at the edge of the search range (see section 2). "
            "Any improvement above is a better **projected** structure, not axial localisation." %
            (sel[:, 2].min(), sel[:, 2].max(), sel[:, 2].mean()))
    M["atoms"] = A

    # ---- loss curves ------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1 + len(arms), figsize=(5 * (1 + len(arms)), 3.8), squeeze=False)
    ax[0, 0].semilogy(np.arange(1, len(sm["hist"]) + 1), sm["hist"])
    ax[0, 0].set_title("S-matrix fit loss (per epoch)"), ax[0, 0].set_xlabel("epoch")
    for j, (arm, r) in enumerate(arms.items(), 1):
        ax[0, j].plot(np.arange(1, len(r["loss"]) + 1), r["loss"])
        w = r["info"].get("probe_warmup")
        if w:
            ax[0, j].axvline(w + 0.5, color="r", ls=":", label=f"probe update starts (it {w + 1})")
            ax[0, j].legend(fontsize=7)
        ax[0, j].set_title(f"QuantEM {arm} loss"), ax[0, j].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "5_loss_curves.png"), dpi=110)
    plt.close(fig)
    M["loss"] = dict(smatrix_final=float(sm["hist"][-1]), **{f"quantem_{k}_final": float(r["loss"][-1]) for k, r in arms.items()})
    log("\n## Loss curves\n")
    log(f"S-matrix final loss {sm['hist'][-1]:.3g} (one dataset, geometry check failed by ~25×, so a low loss does not mean "
        "S is determined).  " + "  ".join(f"QuantEM {k}: {r['loss'][0]:.3g} → {r['loss'][-1]:.3g} over {len(r['loss'])} it."
                                           for k, r in arms.items()))
    log("\nFigure: `5_loss_curves.png`.")
    open(os.path.join(out, "GROUND_TRUTH_COMPARISON.md"), "w").write("\n".join(lines) + "\n")
    json.dump(M, open(os.path.join(out, "metrics.json"), "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
