"""Plots of one QuantEM run (run_benchmark.py quantem) against the ground truth: object slices, projected potential,
depth profile, probe and loss.

    python plot_quantem.py --config configs/poly_si_3grain.yaml --specimen P3_XY \
        --run quantem_window_noiseless_d1_A_nominal_trained_call_s0_pure

Reads runs/<config>/<specimen>/<run>/quantem.npz + summary.json; writes into <run>/plots/
  slices.png      every reconstructed slice (top) vs the true potential integrated over the same slab (bottom), scanned field
  projected.png   sum of all slices vs the true projected potential, difference, scatter, FRC
  depth.png       lateral std per A of each slab (reconstruction vs truth), per-slab correlation, loss curve
  probe.png       initial / final / true probe, real-space amplitude and aperture phase (piston and tilt removed)
  metrics.json    the numbers shown in the plots
Truth registration as evaluate_reconstruction.eval_quantem: truth integrated over each slab, sampled at the
reconstruction pixels (origin_A, first_scan_pixel), blurred to the QuantEM pixel.  Units: both in rad (sigma * V).
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
import eval_ground_truth as eg
import run_realization as rr
from evaluate_reconstruction import true_probes_at_beams


def aperture_phase(Fb, ref, ky, kx):
    """Phase of probe coefficients at the beams relative to ref's support, piston and tilt removed (as probe_match)."""
    w = np.abs(ref) ** 2
    ph = np.angle(Fb)
    ph = np.angle(np.exp(1j * (ph - np.angle((w * np.exp(1j * ph)).sum()))))
    A = np.stack([np.ones_like(ky), ky, kx], 1) * np.sqrt(w)[:, None]
    c = np.linalg.lstsq(A, ph * np.sqrt(w), rcond=None)[0]
    return ph - (c[0] + c[1] * ky + c[2] * kx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    setup = bench.make_setup(cfg)
    sd = bench.specimen_dir(cfg, a.specimen)
    gt = os.path.join(sd, "ground_truth")
    truth = json.load(open(os.path.join(gt, "truth.json")))
    T, sig = truth["thickness_A"], truth["potential"]["sigma_rad_per_V_A"]
    run = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.run)
    s = json.load(open(os.path.join(run, "summary.json")))
    q = np.load(os.path.join(run, "quantem.npz"))
    out = os.path.join(run, "plots")
    os.makedirs(out, exist_ok=True)
    obj, bnd, loss = q["object_potential"], q["slice_boundaries"], q["loss"]
    dr_q, o = s["dr_quantem"], s["origin_A"]
    j0 = np.array(s["first_scan_pixel"]).astype(int)
    j1 = j0 + np.round((np.array(truth["scan"]["shape"]) - 1) * truth["scan"]["step_A"] / dr_q).astype(int) + 1
    rows_A = o[0] + np.arange(j0[0], j1[0]) * dr_q
    cols_A = o[1] + np.arange(j0[1], j1[1]) * dr_q
    crop = obj[:, j0[0]:j1[0], j0[1]:j1[1]]
    tr = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)
    pot, dr_t, st = tr["potential"]
    t_slabs = np.array([rr.sample_truth(pot, dr_t, rows_A, cols_A, bnd[k] - 1e-6, bnd[k + 1] - 1e-6, st, dr_q) * sig
                        for k in range(len(bnd) - 1)])
    t_proj, r_proj = t_slabs.sum(0), crop.sum(0)
    dz = np.diff(bnd)
    zc = 0.5 * (bnd[1:] + bnd[:-1])
    slab_corr = np.array([rr.corr(crop[k], t_slabs[k]) if t_slabs[k].std() > 1e-9 else np.nan for k in range(len(dz))])
    proj_corr_per_slab = np.array([rr.corr(crop[k], t_proj) for k in range(len(dz))])
    proj = eg.fit_stats(r_proj, t_proj)
    k_frc, frc = eg.frc(r_proj, t_proj, dr_q)
    title = f"{a.specimen}: {a.run}  ({s['planes']} planes, {s['iters']} iters, probe {s['probe_mode']})"
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})

    # slices
    n = len(dz)
    ncol = min(n, 9)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(2.1 * ncol, 4.4 * nrow), squeeze=False)
    for k in range(nrow * ncol):
        r, c = divmod(k, ncol)
        for a_ in (ax[2 * r, c], ax[2 * r + 1, c]):
            a_.axis("off")
        if k >= n:
            continue
        ax[2 * r, c].imshow(crop[k], cmap="gray")
        ax[2 * r, c].set_title(f"QuantEM {bnd[k]:.0f}–{bnd[k + 1]:.0f} Å", fontsize=7)
        ax[2 * r + 1, c].imshow(t_slabs[k], cmap="gray")
        ax[2 * r + 1, c].set_title(f"truth; corr {slab_corr[k]:.2f} (proj {proj_corr_per_slab[k]:.2f})", fontsize=7)
    fig.suptitle(title + "\nreconstructed slices (rows 1, 3, ...) vs true σV over the same slab (rows 2, 4, ...); "
                 "corr = vs own slab, proj = vs whole projection", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slices.png"), dpi=90)
    plt.close(fig)

    # projected
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    sc = proj["slope"] * t_proj + proj["offset"]
    vmin, vmax = np.percentile(r_proj, [0.5, 99.5])
    for a_, img, t in ((ax[0], r_proj, "QuantEM: sum of slices (rad)"),
                       (ax[1], sc, f"truth σ·ΣV × slope {proj['slope']:.2f} + offset")):
        im = a_.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
        a_.set_title(t), a_.axis("off"), fig.colorbar(im, ax=a_, shrink=0.8)
    d = r_proj - sc
    dm = np.percentile(np.abs(d), 99)
    im = ax[2].imshow(d, cmap="RdBu_r", vmin=-dm, vmax=dm)
    ax[2].set_title(f"difference: RMSE {proj['rmse_fit']:.3f} rad, NRMSE {proj['nrmse_fit']:.2f}"), ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    ax[3].plot(t_proj.ravel(), r_proj.ravel(), ",", color="k", alpha=0.3)
    xx = np.array([t_proj.min(), t_proj.max()])
    ax[3].plot(xx, proj["slope"] * xx + proj["offset"], "r-", label=f"fit, slope {proj['slope']:.2f}")
    ax[3].set_xlabel("truth (rad)"), ax[3].set_ylabel("QuantEM (rad)"), ax[3].legend()
    ax[3].set_title(f"pixel scatter, corr {proj['corr']:.3f}")
    ax[4].plot(k_frc, frc, "k.-"), ax[4].axhline(0.5, color="r", ls=":")
    ax[4].set_xlabel("spatial frequency (1/Å)"), ax[4].set_title("FRC (QuantEM projection vs truth)"), ax[4].set_ylim(-0.1, 1.05)
    fig.suptitle(title + " — projected potential", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "projected.png"), dpi=90)
    plt.close(fig)

    # depth + loss
    rprof, tprof = crop.reshape(n, -1).std(1) / dz, t_slabs.reshape(n, -1).std(1) / dz
    nrm = lambda v: v / max(v.max(), 1e-30)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ax[0].step(bnd, np.r_[nrm(rprof), nrm(rprof)[-1]], where="post", color="k", label="QuantEM")
    ax[0].step(bnd, np.r_[nrm(tprof), nrm(tprof)[-1]], where="post", color="g", label="truth")
    ax[0].set_xlabel("depth z (Å)"), ax[0].set_ylabel("lateral std per Å (normalised)")
    ax[0].set_title(f"depth profile; corr {rr.corr(rprof, tprof):.2f}"), ax[0].legend()
    ax[1].plot(zc, slab_corr, "ko-", label="vs own slab")
    ax[1].plot(zc, proj_corr_per_slab, "s--", color="0.5", label="vs whole projection (no-depth control)")
    ax[1].set_xlabel("slab centre z (Å)"), ax[1].set_ylabel("correlation"), ax[1].legend(fontsize=8)
    ax[1].set_title(f"per-slab correlation; mean {np.nanmean(slab_corr):.2f} vs {np.mean(proj_corr_per_slab):.2f}")
    ax[2].semilogy(np.arange(1, len(loss) + 1), loss, "k-")
    ax[2].axvline(s["probe_warmup"], color="r", ls=":", label=f"probe training starts ({s['probe_warmup']})")
    ax[2].set_xlabel("iteration"), ax[2].set_title("QuantEM loss"), ax[2].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "depth.png"), dpi=90)
    plt.close(fig)

    # probe
    ky, kx = setup.beams()
    tp = true_probes_at_beams(gt, ky, kx, setup.dr)[s["dataset"]]
    pi, pf = q["initial_probe"], q["probe"]
    Fi, Ff = rr.probe_at_beams(pi, dr_q, ky, kx), rr.probe_at_beams(pf, dr_q, ky, kx)
    m_i, m_f = rr.probe_match(Fi, tp, ky, kx), rr.probe_match(Ff, tp, ky, kx)
    import zarr
    pt_real = zarr.open_array(os.path.join(gt, f"probe_d{s['dataset']}.zarr", "array0"), mode="r")[...]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for j, (p, t) in enumerate(((pi, f"initial ({s.get('probe_init', 'nominal')}): overlap {m_i[0]:.4f}, RMS {m_i[1]:.3f} rad"),
                                (pf, f"final (QuantEM): overlap {m_f[0]:.4f}, RMS {m_f[1]:.3f} rad"),
                                (pt_real, "truth"))):
        a_ = np.abs(p)
        c = np.array(a_.shape) // 2
        a_ = np.roll(a_, tuple(c - np.array(np.unravel_index(a_.argmax(), a_.shape))), (0, 1))   # peak to centre
        h = min(c.min(), int(round(4.0 / (dr_q if j < 2 else setup.dr))))
        ax[0, j].imshow(a_[c[0] - h:c[0] + h, c[1] - h:c[1] + h], cmap="viridis")
        ax[0, j].set_title(f"|probe|, ±4 Å — {t}", fontsize=8), ax[0, j].axis("off")
    for j, (F, t) in enumerate(((Fi, "initial"), (Ff, "final"), (tp, "truth"))):
        ph = aperture_phase(F, tp, ky, kx)
        scat = ax[1, j].scatter(kx, ky, c=ph, s=6, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[1, j].set_aspect("equal"), ax[1, j].set_title(f"aperture phase (rad), {t}; piston + tilt removed", fontsize=8)
        ax[1, j].set_xlabel("k (1/Å)")
    fig.colorbar(scat, ax=ax[1, :], shrink=0.8)
    fig.suptitle(title + f" — probe, dataset {s['dataset']}", fontsize=10)
    fig.savefig(os.path.join(out, "probe.png"), dpi=90)
    plt.close(fig)

    metrics = dict(run=a.run, planes=int(n), slice_boundaries_A=bnd.tolist(), final_loss=float(loss[-1]),
                   projected={k: v for k, v in proj.items()}, depth_profile_corr=rr.corr(rprof, tprof),
                   slab_corr=slab_corr.tolist(), slab_corr_vs_projection=proj_corr_per_slab.tolist(),
                   mean_slab_corr=float(np.nanmean(slab_corr)), mean_proj_corr=float(np.mean(proj_corr_per_slab)),
                   probe=dict(initial_overlap=m_i[0], initial_rms_rad=m_i[1], final_overlap=m_f[0], final_rms_rad=m_f[1]))
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("slab_corr", "slab_corr_vs_projection",
                                                                     "slice_boundaries_A")}, indent=1))


if __name__ == "__main__":
    main()
