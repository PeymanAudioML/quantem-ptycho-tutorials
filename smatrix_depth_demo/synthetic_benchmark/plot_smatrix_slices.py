"""Refocused S-matrix phase at every QuantEM slab centre (rows 1, 3, ...) vs the true sigma*V over the same slab (rows 2, 4, ...),
laid out like plot_quantem.py's slices.png so the two can be compared slice by slice.

    python plot_smatrix_slices.py --config configs/poly_si_3grain.yaml --specimen P3_XY \
        --run smatrix_window_noiseless_s0_sf80_single_d1_joint \
        --quantem-run quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfQ

Reads runs/<config>/<specimen>/<run>/{refocused_phase.npy, arrays.npz, summary.json} and the slab boundaries of the QuantEM run;
writes results/single_focus_joint/<specimen>/smatrix_slices_<run>.png.  corr = vs own slab, proj = vs whole projection
(as plot_quantem.py).  The truth is used for display and correlation only.
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
from plot_fit_monitor import Truth, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--run", required=True, help="S-matrix run folder name")
    ap.add_argument("--quantem-run", required=True, help="QuantEM run whose slab boundaries are used")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    tr = Truth(cfg, a.specimen)
    rd = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen)
    s = json.load(open(os.path.join(rd, a.run, "summary.json")))
    zg = np.load(os.path.join(rd, a.run, "arrays.npz"))["depth_grid"]
    ph = np.load(os.path.join(rd, a.run, "refocused_phase.npy"), mmap_mode="r")
    bnd = np.load(os.path.join(rd, a.quantem_run, "quantem.npz"))["slice_boundaries"]
    c0, c1 = s["crop"]
    rows, cols = tr.region(s["s_grid_offset_px"], c0, c1 - c0)
    proj_t = tr.slab(-1, tr.T + 1, rows, cols)
    n = len(bnd) - 1
    ncol = min(n, 9)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(2.1 * ncol, 4.4 * nrow), squeeze=False)
    for k in range(nrow * ncol):
        r, c = divmod(k, ncol)
        for a_ in (ax[2 * r, c], ax[2 * r + 1, c]):
            a_.axis("off")
        if k >= n:
            continue
        zc = 0.5 * (bnd[k] + bnd[k + 1])
        i = int(np.argmin(np.abs(zg - zc)))
        est = np.asarray(ph[i], float)
        tru = tr.slab(bnd[k] - 1e-6, bnd[k + 1] - 1e-6, rows, cols)
        sc, st = corr(est, tru), corr(est, proj_t)
        ax[2 * r, c].imshow(est, cmap="gray")
        ax[2 * r, c].set_title(f"S refocused z={zg[i]:.0f} Å ({bnd[k]:.0f}–{bnd[k + 1]:.0f})", fontsize=7)
        ax[2 * r + 1, c].imshow(tru, cmap="gray")
        ax[2 * r + 1, c].set_title(f"truth; corr {sc:.2f} (proj {st:.2f})", fontsize=7)
    fig.suptitle(f"{a.specimen}: {a.run}  ({s.get('single_probe_mode') or ('%d defocus datasets, fitted' % len(s.get('coefficients', [])))} probe)\nrefocused S-matrix phase at each slab centre (rows 1, 3, ...) "
                 "vs true σV over the same slab (rows 2, 4, ...); corr = vs own slab, proj = vs whole projection", fontsize=9)
    fig.tight_layout()
    out = os.path.join(bench.HERE, "results", "single_focus_joint", a.specimen)
    os.makedirs(out, exist_ok=True)
    f = os.path.join(out, f"smatrix_slices_{a.run}.png")
    fig.savefig(f, dpi=90)
    print(f)


if __name__ == "__main__":
    main()
