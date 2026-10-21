"""Projected potential and atomic columns of a finished S-matrix fit (run_benchmark.py smatrix) against the ground
truth, in the same form as plot_quantem.py (projected.png + columns.png) so the two can be compared directly.

    python plot_smatrix_columns.py --config results/defocus_study/<study>/configs/k4_d0-d1-d2-d3.yaml \
        --specimen k4_d0-d1-d2-d3 --run runs/<config name>/<specimen>/smatrix_window_noiseless_s0_study

S-matrix estimate (as plot_fit_monitor / defocus_study): unwrapped phase of the refocused wave at mid-depth,
from <run>/refocused_phase.npy.  It approximates sigma * projected potential for a thin object only; for a thick
specimen it is not a reconstruction.  Truth: sigma * sum of all true potential slices, 0.1 A Gaussian blur.
Writes <run>/plots/smatrix_projected.png and <run>/plots/smatrix_metrics.json.
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
import eval_ground_truth as eg
from plot_fit_monitor import Truth, potential_estimate
from plot_quantem import background_offset, column_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--run", required=True, help="run directory")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    tr = Truth(cfg, a.specimen)
    s = json.load(open(os.path.join(a.run, "summary.json")))
    arr = np.load(os.path.join(a.run, "arrays.npz"))
    c0, c1 = s["crop"]
    rows, cols_ = tr.region(s["s_grid_offset_px"], c0, c1 - c0)
    zg = arr["depth_grid"]
    imid = int(np.argmin(np.abs(zg - tr.T / 2)))
    ph = np.load(os.path.join(a.run, "refocused_phase.npy"), mmap_mode="r")
    est = potential_estimate(np.exp(1j * np.asarray(ph[imid], float)))
    tru = np.asarray(tr.slab(-1, tr.T + 1, rows, cols_), float)
    dr = tr.setup.dr
    proj = eg.fit_stats(est, tru)
    b0 = background_offset(est, tru)
    cols = column_analysis(tru, est, dr)
    k_frc, frc = eg.frc(est, tru, dr)

    out = os.path.join(a.run, "plots")
    os.makedirs(out, exist_ok=True)
    title = f"{a.specimen}: {os.path.basename(os.path.normpath(a.run))} (S-matrix, refocused at z = {zg[imid]:.0f} Å)"
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ax = ax.ravel()
    sc = proj["slope"] * tru + proj["offset"]
    vmin, vmax = np.percentile(est, [0.5, 99.5])
    for a_, img, t in ((ax[0], est, "S-matrix estimate: unwrapped mid-depth phase (rad)"),
                       (ax[1], sc, f"truth σ·ΣV × slope {proj['slope']:.2f} + offset")):
        im = a_.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
        a_.set_title(t), a_.axis("off"), fig.colorbar(im, ax=a_, shrink=0.8)
    d = est - sc
    dm = np.percentile(np.abs(d), 99)
    im = ax[2].imshow(d, cmap="RdBu_r", vmin=-dm, vmax=dm)
    ax[2].set_title(f"difference: RMSE {proj['rmse_fit']:.3f} rad, NRMSE {proj['nrmse_fit']:.2f}"), ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    hb = ax[3].hexbin(tru.ravel(), est.ravel(), gridsize=80, bins="log", cmap="Greys", mincnt=1)
    fig.colorbar(hb, ax=ax[3], shrink=0.8, label="pixels per bin (log)")
    if cols["n"]:
        ax[3].plot(tru[cols["core"]], est[cols["core"]], ".", color="tab:orange", ms=1.5, alpha=0.5,
                   label=f"column cores (≤0.5 Å, {cols['n']} columns)")
    xx = np.array([tru.min(), tru.max()])
    ax[3].plot(xx, proj["slope"] * xx + proj["offset"], "r-", label=f"least-squares fit, slope {proj['slope']:.2f}")
    ax[3].plot(xx, xx + b0, "b--", label="ideal: slope 1 (offset from background)")
    ax[3].set_xlabel("truth (rad)"), ax[3].set_ylabel("S-matrix (rad)"), ax[3].legend(fontsize=7, loc="upper left")
    ax[3].set_title(f"pixel density, corr {proj['corr']:.3f}")

    cm = dict(n_columns=int(cols["n"]), background_offset=b0)
    if cols["n"]:
        tp_, rp_ = cols["t_peak"], cols["r_peak"] - b0
        ratio = rp_ / tp_
        cm.update(peak_ratio_mean=float(ratio.mean()), peak_ratio_std=float(ratio.std()),
                  peak_ratio_min=float(ratio.min()), peak_ratio_max=float(ratio.max()), nn_distance_A=cols["nn_A"])
        lim = [0, 1.05 * max(tp_.max(), rp_.max())]
        ax[4].plot(tp_, rp_, "o", color="tab:orange", mec="k", ms=5)
        ax[4].plot(lim, lim, "b--", label="ideal (y = x)")
        ax[4].set_xlim(lim), ax[4].set_ylim(lim), ax[4].set_aspect("equal"), ax[4].legend(fontsize=8)
        ax[4].set_xlabel("true column peak (rad)"), ax[4].set_ylabel("S-matrix column peak − background offset (rad)")
        ax[4].set_title(f"peak per column: S-matrix/truth = {ratio.mean():.2f} ± {ratio.std():.2f}")
        ax[5].plot(cols["prof_r"], cols["prof_t"], "g-", lw=2, label="truth")
        ax[5].plot(cols["prof_r"], cols["prof_q"] - b0, "k.-", label="S-matrix − background offset")
        ax[5].set_xlabel("distance from column centre (Å)"), ax[5].set_ylabel("phase (rad)"), ax[5].legend(fontsize=8)
        ax[5].set_title(f"column-averaged radial profile (to ½ neighbour distance, {cols['nn_A']:.2f} Å)")
        ax[6].axhline(0, color="0.6", lw=0.8)
        ax[6].plot(cols["prof_r"], cols["prof_q"] - b0 - cols["prof_t"], "r.-")
        ax[6].set_xlabel("distance from column centre (Å)"), ax[6].set_ylabel("S-matrix − truth (rad)")
        ax[6].set_title("profile difference\n< 0: missing at the core, > 0: spread to the shoulder", fontsize=9)
    ax[7].plot(k_frc, frc, "k.-"), ax[7].axhline(0.5, color="r", ls=":")
    ax[7].set_xlabel("spatial frequency (1/Å)"), ax[7].set_title("FRC (S-matrix estimate vs truth)")
    ax[7].set_ylim(-0.1, 1.05)
    fig.suptitle(title + " — projected potential and atomic columns", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "smatrix_projected.png"), dpi=90)
    plt.close(fig)

    metrics = dict(run=a.run, depth_A=float(zg[imid]), projected=proj, columns=cm)
    json.dump(metrics, open(os.path.join(out, "smatrix_metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
