"""Step 0 slab grids: the true P3_XYZ potential cut into QuantEM-like slabs, laid out like runs/.../plots/slices.png.

Rows 1, 3, ...: true slab potential (σV per Å, from atoms), grain boundaries in that slab outlined in cyan.
Rows 2, 4, ...: axial-TV step |V(k+1) − V(k)| to the next slab, same outlines; title gives the mean step inside grains
and across boundaries for that pair. Ground truth only: no reconstruction exists for P3_XYZ yet.

Run from synthetic_benchmark/:  ../../.venv/bin/python regularized_experiment/step0_plot_slabs.py [--n-slabs 35,20]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0_gt_axial_tv as s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slabs", default="35,20")
    ap.add_argument("--region", default="field", choices=["field", "cell"], help="scanned field or whole cell")
    ap.add_argument("--atom-sigma", type=float, default=0.35)
    ap.add_argument("--out", default=os.path.join(s0.HERE, "results", "step0_gt_axial_tv", "slab_grids"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    truth, poly, phase, bnd, dr = s0.load()
    T, n = truth["thickness_A"], phase.shape[1]
    scan = truth["scan"]
    f0 = scan["positions_px_first"][0] * dr
    f1 = f0 + (scan["shape"][0] - 1) * scan["step_A"]
    sl = slice(int(round(f0 / dr)), int(round(f1 / dr)) + 1) if a.region == "field" else slice(0, n)
    ext_A = (f0, f1) if a.region == "field" else (0, n * dr)
    P, _ = s0.read_atoms()
    scale = phase.sum(0).sum() / s0.render(P, n, dr, a.atom_sigma, 1.0).sum()
    z_fine = np.arange(0.5, T, 1.0)
    lab_fine = s0.voxel_labels(poly, z_fine, n, dr)
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})

    for ns in [int(v) for v in a.n_slabs.split(",")]:
        edges = np.linspace(0, T, ns + 1)
        V = s0.slab_values("atoms", edges, P, phase, bnd, n, dr, a.atom_sigma, scale)[:, sl, sl]
        labs, pur = s0.slab_labels(lab_fine, z_fine, edges)
        labs, pur = labs[:, sl, sl], pur[:, sl, sl]
        within, boundary = s0.classes(labs, pur)
        step = np.abs(np.diff(V, axis=0))
        vmax_v, vmax_s = np.percentile(V, 99.8), np.percentile(step, 99.8)
        ncol = min(ns, 9)
        nrow = int(np.ceil(ns / ncol))
        fig, ax = plt.subplots(2 * nrow, ncol, figsize=(2.1 * ncol, 4.4 * nrow), squeeze=False)
        ext = (ext_A[0], ext_A[1], ext_A[1], ext_A[0])
        for k in range(nrow * ncol):
            r, c = divmod(k, ncol)
            for a_ in (ax[2 * r, c], ax[2 * r + 1, c]):
                a_.axis("off")
            if k >= ns:
                continue
            frac = [(labs[k] == g).mean() for g in range(3)]
            dom = s0.ZONES[int(np.argmax(frac))]
            mixed = (pur[k] < 1).mean()
            at = ax[2 * r, c]
            at.imshow(V[k], cmap="gray", vmin=0, vmax=vmax_v, extent=ext)
            at.contour(np.linspace(*ext_A, V.shape[1]), np.linspace(*ext_A, V.shape[1]), labs[k].astype(float),
                       levels=[0.5, 1.5], colors="cyan", linewidths=0.6)
            at.set_title(f"truth {edges[k]:.0f}–{edges[k + 1]:.0f} Å; {dom} {max(frac):.0%}", fontsize=7)
            ab = ax[2 * r + 1, c]
            if k == ns - 1:
                ab.set_title("(last slab: no step)", fontsize=7)
                continue
            ab.imshow(step[k], cmap="magma", vmin=0, vmax=vmax_s, extent=ext)
            # mark pixels where the two slabs' majority grains differ (boundary pairs)
            ab.contour(np.linspace(*ext_A, V.shape[1]), np.linspace(*ext_A, V.shape[1]),
                       boundary[k].astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
            mw = step[k][within[k]].mean() if within[k].any() else np.nan
            mb = step[k][boundary[k]].mean() if boundary[k].any() else np.nan
            ab.set_title(f"|ΔV| to next; in {mw*1e3:.1f}, bnd " + ("—" if np.isnan(mb) else f"{mb*1e3:.1f}"),
                         fontsize=7)
        tag = "scanned field" if a.region == "field" else "whole cell"
        fig.suptitle(f"P3_XYZ ground truth, {ns} slabs of {T / ns:.2f} Å ({tag}, {ext_A[0]:.1f}–{ext_A[1]:.1f} Å)\n"
                     "rows 1, 3, ...: true σV per slab (grain boundaries cyan); rows 2, 4, ...: axial-TV step |V(k+1)−V(k)| "
                     "(cyan = majority grain changes); in / bnd = mean step inside grains / across boundaries, 10⁻³ rad/Å",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (4.4 * nrow)))
        f = os.path.join(a.out, f"slabs_{ns}_{a.region}.png")
        fig.savefig(f, dpi=90)
        plt.close(fig)
        print("wrote", f)


if __name__ == "__main__":
    main()
