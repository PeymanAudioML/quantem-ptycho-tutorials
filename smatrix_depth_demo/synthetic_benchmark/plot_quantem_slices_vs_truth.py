"""Every QuantEM slice next to the true potential of the same depth range, and the per-slice correlation vs depth.

Figures (in --out):
  <run>_slices_vs_truth.png   QuantEM slice (top) above the true sigma*V of the same slab (bottom), all slices, zoomed
                              to --zoom A; title: depth, number of atomic layers in the slab, corr vs own slab / projection
  <run>_corr_vs_depth.png     corr vs own slab and vs projection per slice, with the number of atomic layers per slab

Run from synthetic_benchmark/, e.g.
  ../../.venv/bin/python plot_quantem_slices_vs_truth.py --config <cfg> --specimen k4_d0-d1-d2-d3 --run <quantem run> --out <dir>
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bench
import run_realization as rr
from plot_atoms_slab import read_extxyz, layer_period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--zoom", type=float, default=16.0, help="side of the shown field [A], centred (0 = whole scan)")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = bench.load_config(a.config)
    sd = bench.specimen_dir(cfg, a.specimen)
    gt = os.path.join(sd, "ground_truth")
    truth = json.load(open(os.path.join(gt, "truth.json")))
    sig = truth["potential"]["sigma_rad_per_V_A"]
    xyz, _ = read_extxyz(os.path.join(gt, "atoms.extxyz"))
    period, step = layer_period(xyz[:, 2])
    zlayers = np.unique(np.round(xyz[:, 2], 2))
    tr = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)
    pot, dr_t, st = tr["potential"]

    rdir = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.run)
    s = json.load(open(os.path.join(rdir, "summary.json")))
    q = np.load(os.path.join(rdir, "quantem.npz"))
    dr, o = s["dr_quantem"], s["origin_A"]
    j0 = np.array(s["first_scan_pixel"]).astype(int)
    j1 = j0 + np.round((np.array(truth["scan"]["shape"]) - 1) * truth["scan"]["step_A"] / dr).astype(int) + 1
    obj, bnd = q["object_potential"][:, j0[0]:j1[0], j0[1]:j1[1]].astype(float), q["slice_boundaries"]
    rows_A, cols_A = o[0] + np.arange(j0[0], j1[0]) * dr, o[1] + np.arange(j0[1], j1[1]) * dr
    T = truth["thickness_A"]
    proj = rr.sample_truth(pot, dr_t, rows_A, cols_A, -1e-6, T - 1e-6, st, dr) * sig
    n = len(bnd) - 1
    slabs = [rr.sample_truth(pot, dr_t, rows_A, cols_A, bnd[k] - 1e-6, bnd[k + 1] - 1e-6, st, dr) * sig for k in range(n)]
    c_own = np.array([rr.corr(obj[k], slabs[k]) for k in range(n)])
    c_proj = np.array([rr.corr(obj[k], proj) for k in range(n)])
    nlay = np.array([int(((zlayers >= bnd[k] - 1e-6) & (zlayers < bnd[k + 1] - 1e-6)).sum()) for k in range(n)])

    # zoom window (centred)
    h, w = obj.shape[1:]
    m = h if a.zoom <= 0 else min(h, int(round(a.zoom / dr)))
    r0, c0 = (h - m) // 2, (w - m) // 2
    crop = (slice(r0, r0 + m), slice(c0, c0 + m))
    ext = [cols_A[c0], cols_A[c0 + m - 1], rows_A[r0 + m - 1], rows_A[r0]]

    nc = a.cols
    nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(2 * nr, nc, figsize=(1.9 * nc, 2.05 * 2 * nr), squeeze=False)
    for k in range(nr * nc):
        i, j = 2 * (k // nc), k % nc
        if k >= n:
            ax[i, j].axis("off"), ax[i + 1, j].axis("off")
            continue
        for axx, img in ((ax[i, j], obj[k]), (ax[i + 1, j], slabs[k])):
            v = img[crop]
            axx.imshow(v, cmap="gray", vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.5), extent=ext)
            axx.set_xticks([]), axx.set_yticks([])
        ax[i, j].set_title(f"QuantEM {bnd[k]:g}–{bnd[k + 1]:g} Å\nr {c_own[k]:.2f} (proj {c_proj[k]:.2f})", fontsize=7)
        ax[i + 1, j].set_title(f"truth, {nlay[k]} atomic layers", fontsize=7,
                               color="k" if nlay[k] == 4 else "tab:red")
    for i in range(nr):
        ax[2 * i, 0].set_ylabel("QuantEM", fontsize=8)
        ax[2 * i + 1, 0].set_ylabel("ground truth σV", fontsize=8)
    label = a.run.replace("quantem_window_noiseless_", "")
    fig.suptitle(f"{a.specimen} / {label}: each QuantEM slice (upper) vs the true potential of the same {bnd[1] - bnd[0]:g} Å "
                 f"slab (lower), centre {m * dr:.0f} Å field\nr = corr with the true slab (proj = with the whole projection), "
                 f"whole scan field; Si layers every {step:.2f} Å (red = slab misses a layer); own grey scale per panel",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    p1 = os.path.join(a.out, f"{label}_slices_vs_truth.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    zc = 0.5 * (bnd[1:] + bnd[:-1])
    fig, ax1 = plt.subplots(figsize=(9, 3.6))
    ax1.plot(zc, c_own, "o-", label="vs own true slab")
    ax1.plot(zc, c_proj, "s--", label="vs whole projection", alpha=0.7)
    for k in np.where(nlay < 4)[0]:
        ax1.axvspan(bnd[k], bnd[k + 1], color="tab:red", alpha=0.12, lw=0)
    ax1.set_xlabel("depth z (Å)")
    ax1.set_ylabel("correlation")
    ax1.set_title(f"{label}: per-slice correlation (red bands: slab holds only {nlay.min()} of 4 atomic layers)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(a.out, f"{label}_corr_vs_depth.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    out = dict(run=a.run, slices=bnd.tolist(), corr_own=c_own.round(3).tolist(), corr_proj=c_proj.round(3).tolist(),
               atomic_layers=nlay.tolist(), mean_corr_own=float(c_own.mean()), mean_corr_proj=float(c_proj.mean()),
               mean_own_4_layers=float(c_own[nlay == 4].mean()) if (nlay == 4).any() else None,
               mean_own_fewer_layers=float(c_own[nlay < 4].mean()) if (nlay < 4).any() else None)
    json.dump(out, open(os.path.join(a.out, f"{label}_slices_vs_truth.json"), "w"), indent=1)
    print(p1, p2, json.dumps({k: v for k, v in out.items() if k.startswith("mean")}))


if __name__ == "__main__":
    main()
