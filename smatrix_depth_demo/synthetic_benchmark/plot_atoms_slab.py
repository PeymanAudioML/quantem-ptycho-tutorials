"""Atom positions (ground_truth/atoms.extxyz) next to the true and QuantEM potential of one depth slab.

Figures (in --out):
  atoms_3d.png              atoms over the scan field, z 0..--zmax-3d, coloured by atomic layer (z mod a), the slab boxed
  atoms_slab_<z0>-<z1>.png  per QuantEM run: true potential of the slab, QuantEM slices inside the slab (summed), QuantEM
                            projection; atoms of the slab overlaid (colour = layer), all other columns as open circles

Slab [z0, z1] in A (depth below the entrance surface, same z as the potential).  QuantEM slices are summed when they
lie inside the slab; a slice that only partly overlaps it is reported and left out.  Registration as plot_quantem.py
(summary.json origin_A / first_scan_pixel; image axis 0 = atom x, axis 1 = atom y).

Run from synthetic_benchmark/, e.g.
  ../../.venv/bin/python plot_atoms_slab.py --config <cfg> --specimen k4_d0-d1-d2-d3 --z0 0 --z1 5 \
      --runs quantem_..._F_smatrix_fixed_call_s0_k4Q32,quantem_..._E_smatrix_trained_call_s0_k4QE --out <dir>
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bench
import run_realization as rr

LAYER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def read_extxyz(path):
    with open(path) as f:
        f.readline()
        head = f.readline()
    lat = [float(v) for v in head.split('Lattice="')[1].split('"')[0].split()]
    xyz = np.loadtxt(path, skiprows=2, usecols=(1, 2, 3))
    return xyz, np.array(lat).reshape(3, 3)


def layer_period(z):
    """Lattice period along z and the layer spacing, from the distinct atomic z values."""
    zs = np.unique(np.round(z, 3))
    dz = np.diff(zs)
    step = float(np.median(dz))                     # layer spacing (a/4 for Si [001])
    return 4 * step, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--runs", required=True, type=lambda s: s.split(","))
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--z1", type=float, default=5.0)
    ap.add_argument("--zmax-3d", type=float, default=22.0, help="depth shown in the 3D plot [A]")
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
    layer = np.round((xyz[:, 2] % period) / step).astype(int) % 4
    tr = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)
    pot, dr_t, st = tr["potential"]
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    in_slab = (xyz[:, 2] >= a.z0) & (xyz[:, 2] < a.z1)
    tag = f"{a.z0:g}-{a.z1:g}"

    # field of view from the first run (all runs share the scan)
    s0 = json.load(open(os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.runs[0], "summary.json")))
    dr_q, o = s0["dr_quantem"], s0["origin_A"]
    j0 = np.array(s0["first_scan_pixel"]).astype(int)
    j1 = j0 + np.round((np.array(truth["scan"]["shape"]) - 1) * truth["scan"]["step_A"] / dr_q).astype(int) + 1
    rows_A = o[0] + np.arange(j0[0], j1[0]) * dr_q
    cols_A = o[1] + np.arange(j0[1], j1[1]) * dr_q
    ext = [cols_A[0] - dr_q / 2, cols_A[-1] + dr_q / 2, rows_A[-1] + dr_q / 2, rows_A[0] - dr_q / 2]
    pad = 0.3
    fov = ((xyz[:, 0] >= rows_A[0] - pad) & (xyz[:, 0] <= rows_A[-1] + pad) &
           (xyz[:, 1] >= cols_A[0] - pad) & (xyz[:, 1] <= cols_A[-1] + pad))
    zlay = sorted({round(float(z), 2) for z in xyz[in_slab & fov, 2]})
    print(f"layer spacing {step:.3f} A, period {period:.3f} A; slab {tag} A holds {len(zlay)} layers at z = {zlay}; "
          f"{int((in_slab & fov).sum())} atoms in the field of view")

    # ---- 3D
    m3 = fov & (xyz[:, 2] <= a.zmax_3d)
    fig = plt.figure(figsize=(8, 7.5))
    ax = fig.add_subplot(projection="3d")
    for L in range(4):
        sel = m3 & (layer == L)
        ins = sel & in_slab
        out = sel & ~in_slab
        ax.scatter(xyz[out, 1], xyz[out, 0], xyz[out, 2], s=10, c=LAYER_COLORS[L], alpha=0.25, depthshade=False)
        ax.scatter(xyz[ins, 1], xyz[ins, 0], xyz[ins, 2], s=40, c=LAYER_COLORS[L], edgecolors="k", linewidths=0.4,
                   depthshade=False, label=f"layer {L} (z mod {period:.2f} = {L * step:.2f} Å)")
    x0, x1, y0, y1 = cols_A[0], cols_A[-1], rows_A[0], rows_A[-1]
    for z in (a.z0, a.z1):                                         # slab box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5, "k-", lw=0.8)
    for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        ax.plot([xx, xx], [yy, yy], [a.z0, a.z1], "k-", lw=0.8)
    ax.set_xlabel("y (Å)")
    ax.set_ylabel("x (Å)")
    ax.set_zlabel("z = depth (Å)")
    ax.set_zlim(a.zmax_3d, 0)
    ax.set_box_aspect((x1 - x0, y1 - y0, a.zmax_3d * 0.8))
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"Si atoms over the scan field, z 0–{a.zmax_3d:g} Å; slab {tag} Å boxed (solid = inside)\n"
                 f"layer spacing {step:.2f} Å, each column repeats every {period:.2f} Å", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "atoms_3d.png"), dpi=110)
    plt.close(fig)

    # ---- 2D per run
    t_slab = rr.sample_truth(pot, dr_t, rows_A, cols_A, a.z0 - 1e-6, a.z1 - 1e-6, st, dr_q) * sig
    t_proj = rr.sample_truth(pot, dr_t, rows_A, cols_A, -1e-6, truth["thickness_A"] - 1e-6, st, dr_q) * sig
    fig, axs = plt.subplots(len(a.runs), 3, figsize=(13, 4.4 * len(a.runs)), squeeze=False)
    summary = {"slab_A": [a.z0, a.z1], "layers_z_A": zlay, "runs": {}}

    def overlay(ax_, depth_mask, title):
        others = fov & ~depth_mask
        cols = np.unique(np.round(xyz[others][:, :2], 2), axis=0)
        ax_.scatter(cols[:, 1], cols[:, 0], s=40, facecolors="none", edgecolors="w", linewidths=0.6, alpha=0.6)
        for L in range(4):
            sel = fov & depth_mask & (layer == L)
            ax_.scatter(xyz[sel, 1], xyz[sel, 0], s=18, c=LAYER_COLORS[L], edgecolors="k", linewidths=0.3,
                        label=f"layer {L}")
        ax_.set_xlim(ext[0], ext[1])
        ax_.set_ylim(ext[2], ext[3])
        ax_.set_title(title, fontsize=8)
        ax_.set_xlabel("y (Å)")
        ax_.set_ylabel("x (Å)")

    for i, run in enumerate(a.runs):
        rdir = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, run)
        q = np.load(os.path.join(rdir, "quantem.npz"))
        obj, bnd = q["object_potential"][:, j0[0]:j1[0], j0[1]:j1[1]], q["slice_boundaries"]
        inside = [k for k in range(len(bnd) - 1) if bnd[k] >= a.z0 - 1e-6 and bnd[k + 1] <= a.z1 + 1e-6]
        partial = [k for k in range(len(bnd) - 1)
                   if k not in inside and bnd[k] < a.z1 and bnd[k + 1] > a.z0]
        rec = obj[inside].sum(0) if inside else np.zeros_like(t_slab)
        z_rec = (bnd[inside[0]], bnd[inside[-1] + 1]) if inside else (np.nan, np.nan)
        t_rec = rr.sample_truth(pot, dr_t, rows_A, cols_A, z_rec[0] - 1e-6, z_rec[1] - 1e-6, st, dr_q) * sig \
            if inside else t_slab
        c_slab, c_proj = rr.corr(rec, t_rec), rr.corr(rec, t_proj)
        in_rec = (xyz[:, 2] >= z_rec[0]) & (xyz[:, 2] < z_rec[1])
        label = run.replace("quantem_window_noiseless_", "")
        ax = axs[i]
        ax[0].imshow(t_rec, cmap="gray", extent=ext)             # truth over the same depth as the QuantEM slices
        overlay(ax[0], in_rec, f"truth σV, z {z_rec[0]:g}–{z_rec[1]:g} Å + atoms in it\n"
                               f"(open circles: columns with no atom in this depth range)")
        ax[1].imshow(rec, cmap="gray", extent=ext)
        overlay(ax[1], in_rec, f"{label}\nQuantEM slices {z_rec[0]:g}–{z_rec[1]:g} Å (summed {len(inside)}); "
                               f"corr vs own truth {c_slab:.2f}, vs projection {c_proj:.2f}"
                + (f"\npartly overlapping, left out: {[f'{bnd[k]:g}-{bnd[k + 1]:g}' for k in partial]}" if partial else ""))
        ax[2].imshow(obj.sum(0), cmap="gray", extent=ext)
        overlay(ax[2], np.ones(len(xyz), bool), f"QuantEM projection (all {len(bnd) - 1} slices) + all columns")
        if i == 0:
            ax[0].legend(loc="lower right", fontsize=6, framealpha=0.7)
        summary["runs"][run] = dict(slices=[[float(bnd[k]), float(bnd[k + 1])] for k in inside],
                                    corr_vs_own_truth=float(c_slab), corr_vs_projection=float(c_proj),
                                    layers_in_rec_z_A=sorted({round(float(z), 2) for z in xyz[in_rec & fov, 2]}))
    fig.suptitle(f"{a.specimen}: atoms (colour = atomic layer, z mod {period:.2f} Å) on the true and QuantEM potential, "
                 f"slab {tag} Å", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"atoms_slab_{tag}.png"), dpi=100)
    plt.close(fig)
    json.dump(summary, open(os.path.join(a.out, f"atoms_slab_{tag}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
