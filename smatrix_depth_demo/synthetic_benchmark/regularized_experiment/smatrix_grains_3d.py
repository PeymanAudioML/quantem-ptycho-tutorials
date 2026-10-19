"""3D grain map from the S-matrix recovery: which grain sits where, and where the boundaries are.

Input: the refocused S-matrix phase volume of a run (runs/.../refocused_phase.npy: the S-matrix refocused to depths
-0.2T..1.2T in 2 A steps, over the scanned field, 0.2 A pixels). Ground truth is used only to score and to colour.

1. Local lattice fingerprint at every voxel: |FFT| of a Hann-windowed tile (default 6.4 A) around it, on a lateral
   stride grid, DC and the lowest frequencies removed, unit-normalised. Grains differ by orientation, so their
   fingerprints differ; the depth position is the refocus depth.
2. Unsupervised: PCA + k-means (k = 3) over all voxels with 0 <= z <= T. No truth enters the clustering.
3. Scoring only: clusters are matched to the true grains by the best one-to-one assignment, then voxel accuracy, per-depth
   grain fractions and boundary depths are compared with the true grain map (construction rule of the specimen).
The best QuantEM reconstruction (--quantem-run; default arm F, 20 slices, depth TV w = 1000) goes through the same
pipeline: its slice containing each depth, cropped to the same field (same 0.2 A pixels).
The same pipeline on the TRUE potential (10 A running slabs at the same depths) gives the reference: what the
fingerprint can do with perfect depth information.

Run from synthetic_benchmark/:
  ../../.venv/bin/python regularized_experiment/smatrix_grains_3d.py --specimen P3_XYZ \
      --run smatrix_window_noiseless_s0_sfall_single_d1
"""
import argparse, itertools, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
os.chdir(BENCH)
import bench  # noqa: E402
from plot_fit_monitor import Truth  # noqa: E402
import step0_gt_axial_tv as s0  # noqa: E402

GRAIN_COL = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def fingerprints(vol, size, stride, kmin):
    """vol (Z, n, n) -> (Z, m, m, F) unit-normalised |FFT| of Hann tiles, plus tile centres (pixels)."""
    Z, n, _ = vol.shape
    starts = np.arange(0, n - size + 1, stride)
    win = np.outer(np.hanning(size), np.hanning(size))
    k = np.fft.fftfreq(size) * size
    keep = (np.hypot(k[:, None], k[None, :]) >= kmin).ravel()
    out = np.empty((Z, len(starts), len(starts), int(keep.sum())), np.float32)
    for i, a in enumerate(starts):
        for j, b in enumerate(starts):
            t = vol[:, a:a + size, b:b + size]
            t = (t - t.mean(axis=(1, 2), keepdims=True)) * win
            s = np.abs(np.fft.fft2(t)).reshape(Z, -1)[:, keep]
            out[:, i, j] = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-30)
    return out, starts + size / 2.0


def kmeans(X, k, seed=0, iters=100, restarts=10):
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.array([X[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def cluster(F, k, npc):
    X = F.reshape(-1, F.shape[-1]).astype(np.float64)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return kmeans(U[:, :npc] * S[:npc], k).reshape(F.shape[:-1])


def match(pred, true, k=3):
    """Best one-to-one cluster -> grain map (scoring/colouring only)."""
    best = None
    for perm in itertools.permutations(range(k)):
        acc = (np.array(perm)[pred] == true).mean()
        if best is None or acc > best[0]:
            best = (acc, perm)
    return np.array(best[1])[pred], float(best[0]), best[1]


def grain_fractions(lab):
    return np.stack([(lab == g).mean(axis=(1, 2)) for g in range(3)], 1)


def boundary_mask(lab):
    """Voxels whose label differs from any 6-neighbour."""
    b = np.zeros(lab.shape, bool)
    for ax in range(3):
        d = np.diff(lab, axis=ax) != 0
        sl0 = [slice(None)] * 3; sl0[ax] = slice(0, -1)
        sl1 = [slice(None)] * 3; sl1[ax] = slice(1, None)
        b[tuple(sl0)] |= d
        b[tuple(sl1)] |= d
    return b


def voxels3d(ax, lab, x, z, title, cut=True):
    """lab (Z, m, m) -> matplotlib voxel plot, depth downwards; front quadrant cut away to show the inside."""
    L = np.transpose(lab, (1, 2, 0))                      # (x, y, z)
    filled = np.ones(L.shape, bool)
    if cut:
        m = L.shape[0]
        filled[m // 2:, :m // 2, :] = False
    col = np.empty(L.shape + (4,))
    for g in range(3):
        rgb = matplotlib.colors.to_rgba(GRAIN_COL[g], 0.9)
        col[L == g] = rgb
    xe = np.r_[x - (x[1] - x[0]) / 2, x[-1] + (x[1] - x[0]) / 2]
    ze = np.r_[z - (z[1] - z[0]) / 2, z[-1] + (z[1] - z[0]) / 2]
    X, Y, Zz = np.meshgrid(xe, xe, ze, indexing="ij")
    ax.voxels(X, Y, Zz, filled, facecolors=col, edgecolor=None, linewidth=0)
    ax.set_zlim(ze[-1], ze[0])
    ax.set_xlabel("x (Å)", fontsize=7), ax.set_ylabel("y (Å)", fontsize=7), ax.set_zlabel("depth z (Å)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect((1, 1, 1.6))
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poly_si_3grain.yaml")
    ap.add_argument("--specimen", default="P3_XYZ")
    ap.add_argument("--run", default="smatrix_window_noiseless_s0_sfall_single_d1")
    ap.add_argument("--tile", type=int, default=32, help="fingerprint tile (pixels of 0.2 A)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--kmin", type=float, default=2.0, help="drop |k| < kmin (tile frequency units): DC / slow trends")
    ap.add_argument("--dz", type=float, default=4.0, help="depth step used (A); refocus grid is 2 A")
    ap.add_argument("--npc", type=int, default=10)
    ap.add_argument("--truth-slab", type=float, default=10.0)
    ap.add_argument("--quantem-run", default="quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ20_tvz1000")
    ap.add_argument("--quantem-label", default="QuantEM F, 20 slices, TV w=1000")
    ap.add_argument("--smatrix-label", default="S-matrix recovery")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "smatrix_grains_3d"))
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    tr = Truth(cfg, a.specimen)
    T, dr = tr.T, tr.setup.dr
    rd = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.run)
    s = json.load(open(os.path.join(rd, "summary.json")))
    zg = np.load(os.path.join(rd, "arrays.npz"))["depth_grid"]
    ph = np.load(os.path.join(rd, "refocused_phase.npy"), mmap_mode="r")
    c0, c1 = s["crop"]
    rows, cols = tr.region(s["s_grid_offset_px"], c0, c1 - c0)
    zsel = np.flatnonzero((zg >= 0) & (zg <= T))
    zsel = zsel[::max(1, int(round(a.dz / float(zg[1] - zg[0]))))]
    z = zg[zsel]
    vol_s = np.asarray(ph[zsel], np.float64)
    # true potential: running slabs of truth_slab A centred on the same depths
    vol_t = np.stack([tr.slab(zz - a.truth_slab / 2, zz + a.truth_slab / 2, rows, cols) for zz in z]).astype(np.float64)
    # best QuantEM run: the slice containing each depth, on the S-matrix field
    vols = [("S-matrix", vol_s)]
    if a.quantem_run:
        qd = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.quantem_run)
        q, qs = np.load(os.path.join(qd, "quantem.npz")), json.load(open(os.path.join(qd, "summary.json")))
        qb, dq, qo = q["slice_boundaries"], qs["dr_quantem"], qs["origin_A"]
        jr = np.round((rows * dr - qo[0]) / dq).astype(int)
        jc = np.round((cols * dr - qo[1]) / dq).astype(int)
        if abs(dq - dr) > 1e-4 or jr.min() < 0 or jr.max() >= q["object_potential"].shape[1]:
            raise ValueError("QuantEM grid does not cover the S-matrix field on the same pixels")
        kz = np.clip(np.searchsorted(qb, z, side="right") - 1, 0, len(qb) - 2)
        vols.append((a.quantem_label, q["object_potential"][kz][:, jr][:, :, jc].astype(np.float64)))
    vols.append(("truth potential", vol_t))
    # true grain labels at the tile centres
    s0.GT = os.path.join(bench.specimen_dir(cfg, a.specimen), "ground_truth")   # step0 defaults to P3_XYZ
    _, poly, pot, _, dr0 = s0.load()
    lab_full = s0.voxel_labels(poly, z, pot.shape[1], dr0)          # (Z, n, n) cell pixels
    lab_field = lab_full[:, rows][:, :, cols]
    os.makedirs(a.out, exist_ok=True)
    res, maps = {}, {}
    for name, vol in vols:
        F, centres = fingerprints(vol, a.tile, a.stride, a.kmin)
        ci = np.round(centres).astype(int)
        lab_true = lab_field[:, ci][:, :, ci]
        pred = cluster(F, 3, a.npc)
        pred_m, acc, perm = match(pred, lab_true)
        ft, fp = grain_fractions(lab_true), grain_fractions(pred_m)
        bands = np.arange(0, T + 1e-6, 40.0)
        band_acc = [float((pred_m[(z >= lo) & (z < hi)] == lab_true[(z >= lo) & (z < hi)]).mean())
                    for lo, hi in zip(bands[:-1], bands[1:])]
        res[name] = dict(voxel_accuracy=acc, band_accuracy_40A=band_acc, cluster_to_grain=list(map(int, perm)),
                         chance=float(max(np.bincount(lab_true.ravel(), minlength=3)) / lab_true.size))
        maps[name] = (pred_m, lab_true, ft, fp)
        print(f"{name}: voxel accuracy {acc:.3f} (majority-class {res[name]['chance']:.3f}); per 40 A band "
              + " / ".join(f"{v:.2f}" for v in band_acc), flush=True)
    x = (np.asarray(centres) * dr)
    lab_true, ft = maps["S-matrix"][1], maps["S-matrix"][2]
    names = [s0.ZONES[g] for g in range(3)]
    short = {"S-matrix": a.smatrix_label, "truth potential": f"method on TRUE potential ({a.truth_slab:g} Å slabs)"}
    panels = [("TRUE grains (construction rule)", lab_true)] + \
             [(f"{short.get(n, n)}\nvoxel accuracy {res[n]['voxel_accuracy']:.2f}", maps[n][0]) for n, _ in vols]
    styles = {"S-matrix": "--", "truth potential": ":"}

    # ---- figure 1: 3D voxels
    fig = plt.figure(figsize=(5.2 * len(panels), 7.5))
    for i, (ttl, lab) in enumerate(panels):
        voxels3d(fig.add_subplot(1, len(panels), i + 1, projection="3d"), lab, x, z, ttl)
    h = [plt.Rectangle((0, 0), 1, 1, color=GRAIN_COL[g]) for g in range(3)]
    fig.legend(h, [f"grain {names[g]}" for g in range(3)], loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{a.specimen}: 3D grain map over the scanned field ({x[0]:.1f}–{x[-1]:.1f} Å tile centres), front "
                 f"quarter cut away; S-matrix {a.run}" + (f"; QuantEM {a.quantem_run}" if a.quantem_run else ""),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(a.out, "grains_3d.png"), dpi=110)

    # ---- figure 2: sections and depth profiles
    fig, ax = plt.subplots(2, len(panels) + 1, figsize=(4.3 * (len(panels) + 1), 9.5), squeeze=False)
    cmap = ListedColormap(GRAIN_COL)
    mid = len(x) // 2
    ext = [x[0], x[-1], z[-1], z[0]]
    for r, (cut, lbl) in enumerate(((lambda L: L[:, mid, :], f"x–z at y = {x[mid]:.1f} Å"),
                                    (lambda L: L[:, :, mid], f"y–z at x = {x[mid]:.1f} Å"))):
        for c, (ttl, L) in enumerate(panels):
            ax[r, c].imshow(cut(L), cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", extent=ext, interpolation="nearest")
            ax[r, c].contour(x, z, cut(boundary_mask(L)).astype(float), levels=[0.5], colors="k", linewidths=0.6)
            ax[r, c].set_title(f"{ttl.splitlines()[0]}\n{lbl}", fontsize=8)
            ax[r, c].set_xlabel("lateral (Å)"), ax[r, c].set_ylabel("depth z (Å)")
    pr = [n for n, _ in vols if n != "truth potential"]
    for r, group in enumerate((pr, ["truth potential"])):
        a_ = ax[r, -1]
        for g in range(3):
            a_.plot(ft[:, g], z, "-", color=GRAIN_COL[g], lw=2.5, alpha=0.6, label=f"truth {names[g]}")
            for n in group:
                a_.plot(maps[n][3][:, g], z, styles.get(n, "-."), color=GRAIN_COL[g], lw=1.3,
                        label=f"{short.get(n, n).split(' (')[0][:22]} {names[g]}")
        a_.set_ylim(z[-1], z[0]), a_.set_xlim(0, 1), a_.set_xlabel("fraction of field"), a_.set_ylabel("depth z (Å)")
        a_.legend(fontsize=5.5), a_.grid(alpha=0.3)
        a_.set_title("grain share per depth (thick = truth)", fontsize=8)
    fig.suptitle(f"{a.specimen}: grains and boundaries (black) vs truth; colours = grain ({', '.join(names)}); accuracy "
                 "per 40 Å band: " + "; ".join(f"{short.get(n, n).split(' (')[0]} "
                                               + "/".join(f"{v:.2f}" for v in res[n]["band_accuracy_40A"]) for n, _ in vols),
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.out, "grains_sections.png"), dpi=100)
    json.dump(dict(run=a.run, specimen=a.specimen, params=vars(a), depths_A=z.tolist(), tile_centres_A=x.tolist(),
                   results=res), open(os.path.join(a.out, "grains_3d.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "grains_3d.npz"), z=z, x=x, truth=lab_true,
                        **{n.replace(" ", "_").replace(",", "").replace("=", ""): maps[n][0] for n, _ in vols})
    print(a.out)


if __name__ == "__main__":
    main()
