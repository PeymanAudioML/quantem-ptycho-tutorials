"""Grain labels from smatrix_grains_3d.py drawn on the 2D refocused S-matrix slices (same orientation as
plot_smatrix_slices.py: rows down, columns across, scanned field). Checks that the 3D map matches what the 2D slices show.

Run from synthetic_benchmark/ after smatrix_grains_3d.py:
  ../../.venv/bin/python regularized_experiment/plot_grains_overlay.py --specimen P3_XY --run <S run> --grains <dir>
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
os.chdir(BENCH)
sys.path.insert(0, BENCH)
import bench  # noqa: E402

COL = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ZONE = ["[001]", "[110]", "[111]"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poly_si_3grain.yaml")
    ap.add_argument("--specimen", default="P3_XY")
    ap.add_argument("--run", required=True)
    ap.add_argument("--grains", required=True, help="output folder of smatrix_grains_3d.py for this run")
    ap.add_argument("--depths", default="2,50,100,150,198")
    ap.add_argument("--quantem-run", default="", help="QuantEM run whose labels are in grains_3d.npz (optional)")
    ap.add_argument("--quantem-label", default="QuantEM")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    rd = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.run)
    zg = np.load(os.path.join(rd, "arrays.npz"))["depth_grid"]
    ph = np.load(os.path.join(rd, "refocused_phase.npy"), mmap_mode="r")
    g = np.load(os.path.join(a.grains, "grains_3d.npz"))
    z, x = g["z"], g["x"]
    pred = g["S-matrix"]
    truth = g["truth"]
    dr = bench.make_setup(cfg).dr
    n = ph.shape[1]
    ext = [0, n * dr, n * dr, 0]
    ds = [float(v) for v in a.depths.split(",")]
    rows_def = [("S", ph, zg, pred, "S-matrix grains")]
    if a.quantem_run:
        from plot_fit_monitor import Truth
        qd = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.quantem_run)
        q, qs = np.load(os.path.join(qd, "quantem.npz")), json.load(open(os.path.join(qd, "summary.json")))
        s = json.load(open(os.path.join(rd, "summary.json")))
        tr = Truth(cfg, a.specimen)
        c0, c1 = s["crop"]
        rr_, cc_ = tr.region(s["s_grid_offset_px"], c0, c1 - c0)
        jr = np.round((rr_ * dr - qs["origin_A"][0]) / qs["dr_quantem"]).astype(int)
        jc = np.round((cc_ * dr - qs["origin_A"][1]) / qs["dr_quantem"]).astype(int)
        qvol = q["object_potential"][:, jr][:, :, jc]
        key = [k for k in g.files if k not in ("z", "x", "truth", "S-matrix", "truth_potential")][0]
        rows_def.append(("Q", qvol, q["slice_boundaries"], g[key], f"{a.quantem_label} grains"))
    nr = 2 * len(rows_def) + 1
    fig, ax = plt.subplots(nr, len(ds), figsize=(3.3 * len(ds), 3.4 * nr), squeeze=False)
    cmap = ListedColormap(COL)
    half = (x[1] - x[0]) / 2
    lab_ext = [x[0] - half, x[-1] + half, x[-1] + half, x[0] - half]
    for c, zz in enumerate(ds):
        k = int(np.argmin(np.abs(z - zz)))
        r = 0
        for kind, vol, zax, L, nm in rows_def:
            if kind == "S":
                i = int(np.argmin(np.abs(zax - zz)))
                ttl = f"S refocused z = {zax[i]:.0f} Å"
            else:
                i = int(np.clip(np.searchsorted(zax, zz, side="right") - 1, 0, len(zax) - 2))
                ttl = f"{a.quantem_label} slice {zax[i]:.0f}–{zax[i + 1]:.0f} Å"
            img = np.asarray(vol[i], float)
            ax[r, c].imshow(img, cmap="gray", extent=ext)
            ax[r, c].set_title(ttl, fontsize=8)
            ax[r + 1, c].imshow(img, cmap="gray", extent=ext)
            ax[r + 1, c].imshow(L[k], cmap=cmap, vmin=-0.5, vmax=2.5, alpha=0.35, extent=lab_ext, interpolation="nearest")
            ax[r + 1, c].contour(x, x, L[k].astype(float), levels=[0.5, 1.5], colors="w", linewidths=1.2)
            ax[r + 1, c].set_title(f"{nm} at z = {z[k]:.0f} Å (match {(L[k] == truth[k]).mean():.2f})", fontsize=8)
            r += 2
        ax[r, c].imshow(np.asarray(ph[int(np.argmin(np.abs(zg - zz)))], float), cmap="gray", extent=ext)
        ax[r, c].imshow(truth[k], cmap=cmap, vmin=-0.5, vmax=2.5, alpha=0.35, extent=lab_ext, interpolation="nearest")
        ax[r, c].contour(x, x, truth[k].astype(float), levels=[0.5, 1.5], colors="w", linewidths=1.2)
        ax[r, c].set_title(f"true grains at z = {z[k]:.0f} Å (on S image)", fontsize=8)
    for a_ in ax.ravel():
        a_.set_xlim(0, n * dr), a_.set_ylim(n * dr, 0), a_.set_xticks([]), a_.set_yticks([])
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[i], alpha=0.6) for i in range(3)]
    fig.legend(h, ZONE, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{a.specimen}: {a.run}\npairs of rows: reconstructed slice, then the grains "
                 "smatrix_grains_3d.py found on it (match = vs truth); last row: true grains", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = os.path.join(a.grains, "grains_overlay_2d.png")
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
