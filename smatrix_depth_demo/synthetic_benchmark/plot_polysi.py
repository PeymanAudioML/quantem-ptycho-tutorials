"""Figures 1-4 of the polycrystalline-Si benchmark from the generated ground truth (no reconstruction).

    python plot_polysi.py --config configs/poly_si_3grain.yaml --specimens P3_XY,P3_XYZ

1 3-D atomic structure coloured by grain; 2 XY / XZ / YZ views with boundary atoms; 3 ground-truth potential slices;
4 diffraction patterns at the three defoci (both detector variants).  Written to results/geometry/<config>/<specimen>/.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench

COL = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"])


def read_atoms(gt):
    from ase.io import read
    a = read(os.path.join(gt, "atoms.extxyz"))
    return a.positions, a.arrays["grain"], a.arrays["dist_gb"], a.arrays["self_boundary"].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poly_si_3grain.yaml")
    ap.add_argument("--specimens", default="P3_XY,P3_XYZ")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    setup = bench.make_setup(cfg)
    L, T = setup.Y * setup.dr, float(cfg["specimen"]["thickness_A"])
    pos = bench.scan_positions(cfg) * setup.dr
    lo, hi = pos[0, 0], pos[-1, -1]
    W = setup.K * setup.dr
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 9})
    for name in a.specimens.split(","):
        sd = bench.specimen_dir(cfg, name)
        gt = os.path.join(sd, "ground_truth")
        out = os.path.join(bench.HERE, "results", "geometry", cfg["name"], name)
        os.makedirs(out, exist_ok=True)
        P, G, DG, SB = read_atoms(gt)
        info = json.load(open(os.path.join(gt, "polycrystal.json")))
        ng = len(info["grains"])
        lab = [f"grain {g['id']}: [{''.join(map(str, g['zone']))}] +{g['inplane_deg']:g}°, {g['n_atoms']} atoms"
               for g in info["grains"]]

        # 1. 3-D structure (subsampled)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
        rng = np.random.default_rng(0)
        sub = rng.choice(len(P), min(len(P), 9000), replace=False)
        for g in range(ng):
            m = sub[G[sub] == g]
            ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=1.2, c=COL[g], label=lab[g], depthshade=False)
        ax.set_xlabel("x (Å)"), ax.set_ylabel("y (Å)"), ax.set_zlabel("depth z (Å)")
        ax.set_zlim(T, 0)
        ax.set_box_aspect((L, L, T * 0.6))
        ax.legend(loc="upper left", fontsize=8, markerscale=6)
        ax.set_title(f"{name}: {len(P)} Si atoms, {ng} grains (unrelaxed Voronoi; 9000 atoms shown)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "1_structure_3d.png"), dpi=130)
        plt.close(fig)

        # 2. XY / XZ / YZ views, boundary atoms (dist_gb < 1.5 A) in black, scanned field and illuminated region
        fig, ax = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw=dict(width_ratios=[1, 1, 1]))
        gb = (DG < 1.5) & ~SB
        for g in range(ng):
            m = (G == g) & ~gb
            ax[0].scatter(P[m, 1], P[m, 0], s=0.3, c=COL[g], label=lab[g])
        ax[0].scatter(P[gb, 1], P[gb, 0], s=0.6, c="k", label="within 1.5 Å of a boundary")
        ax[0].scatter(P[SB & (DG < 1.5), 1], P[SB & (DG < 1.5), 0], s=0.6, c="0.6", label="periodic self-boundary")
        for (l0, h0), ls, txt in (((lo, hi), "-", "scan centres"), ((lo - W / 2, hi + W / 2), "--", "illuminated (+window)")):
            ax[0].plot([l0[1], h0[1], h0[1], l0[1], l0[1]], [l0[0], l0[0], h0[0], h0[0], l0[0]], "k", ls=ls, lw=1)
        ax[0].set_xlim(0, L), ax[0].set_ylim(L, 0), ax[0].set_aspect("equal")
        ax[0].set_xlabel("y (Å)"), ax[0].set_ylabel("x (Å)"), ax[0].set_title("XY projection (all depths)")
        ax[0].legend(fontsize=7, markerscale=10, loc="lower left")
        for j, (k, kn) in enumerate(((0, "x"), (1, "y")), 1):
            band = np.abs(P[:, 1 - k] - L / 2) < 3.0                      # 6 Å slab through the cell centre
            for g in range(ng):
                m = band & (G == g) & ~gb
                ax[j].scatter(P[m, k], P[m, 2], s=0.5, c=COL[g])
            ax[j].scatter(P[band & gb, k], P[band & gb, 2], s=0.8, c="k")
            for e in (lo[k], hi[k]):
                ax[j].axvline(e, color="k", lw=0.8)
            ax[j].set_xlim(0, L), ax[j].set_ylim(T, 0)
            ax[j].set_xlabel(f"{kn} (Å)"), ax[j].set_ylabel("depth z (Å)")
            ax[j].set_title(f"{kn.upper()}Z section: 6 Å slab through the cell centre (vertical lines: scan field)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "2_grain_boundaries_xy_xz_yz.png"), dpi=130)
        plt.close(fig)

        # 3. potential slices
        import zarr
        V = zarr.open_array(os.path.join(gt, "potential.zarr", "array0"), mode="r")[...]
        zs = np.linspace(0, len(V) - 1, 8).astype(int)
        fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
        vmax = np.percentile(V, 99.9)
        for j, k in enumerate(zs):
            a_ = ax.flat[j]
            a_.imshow(V[k], cmap="magma", vmin=0, vmax=vmax, extent=[0, L, L, 0])
            a_.plot([lo[1], hi[1], hi[1], lo[1], lo[1]], [lo[0], lo[0], hi[0], hi[0], lo[0]], "c", lw=0.8)
            gz = np.bincount(G[np.abs(P[:, 2] - (k + 0.5) * 2.0) < 1.0], minlength=ng)
            a_.set_title(f"slice {k}: z {2*k:.0f}–{2*k+2:.0f} Å; atoms per grain {gz.tolist()}", fontsize=8)
            a_.set_xlabel("y (Å)"), a_.set_ylabel("x (Å)")
        fig.suptitle(f"{name}: ground-truth potential (V·Å per 2 Å slice, abTEM); cyan = scan field", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "3_potential_slices.png"), dpi=120)
        plt.close(fig)

        # 4. diffraction patterns (centre scan position) at the three defoci, both detector variants
        cy, cx = np.array(pos.shape[:2]) // 2
        fig, ax = plt.subplots(2, 3, figsize=(13, 8.6))
        nominal, true, ref = bench.coefficients(cfg)
        ext = 1e3 * setup.lam / (2 * setup.dr)
        for i, variant in enumerate(("window", "pixelated")):
            for d in range(3):
                I = zarr.open_array(os.path.join(sd, f"{variant}_noiseless", f"def{d}", "4dstem_quantem", "_array"),
                                    mode="r")[cy, cx]
                im = ax[i, d].imshow(np.log10(np.clip(I, 1e-9, None)), cmap="inferno", vmin=-8,
                                     extent=[-ext, ext, ext, -ext])
                ax[i, d].set_title(f"{variant}, def{d}: true focus {-true[d, 0]:.1f} Å", fontsize=9)
                ax[i, d].set_xlabel("mrad")
        fig.colorbar(im, ax=ax, shrink=0.6, label="log10 intensity fraction per pixel")
        fig.suptitle(f"{name}: diffraction at the central scan position (noiseless)", fontsize=11)
        fig.savefig(os.path.join(out, "4_diffraction_defoci.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"{name}: figures in {os.path.relpath(out, bench.HERE)}")


if __name__ == "__main__":
    main()
