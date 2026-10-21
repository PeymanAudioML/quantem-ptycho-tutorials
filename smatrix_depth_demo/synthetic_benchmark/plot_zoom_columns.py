"""Zoomed QuantEM slices vs the true slab, with every atomic column circled and coloured by its Si [001] layer.

Upper row: QuantEM slice; lower row: true sigma*V over the same depth range.  Circles mark the projected column positions;
colour = atomic layer (z mod a), solid = that layer has atoms inside the slab, dotted = it has none there.

Run from synthetic_benchmark/, e.g.
  ../../.venv/bin/python plot_zoom_columns.py --config <cfg> --specimen k4_d0-d1-d2-d3 --run <quantem run> \
      --slices 8,9,0,4 --out <png>
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import bench
import run_realization as rr
from plot_atoms_slab import read_extxyz, layer_period, LAYER_COLORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--slices", default="8,9,0,4", type=lambda s: [int(x) for x in s.split(",")],
                    help="QuantEM slice indices to show")
    ap.add_argument("--half", type=int, default=60, help="half width of the zoom [px]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    sd = bench.specimen_dir(cfg, a.specimen)
    truth = json.load(open(os.path.join(sd, "ground_truth", "truth.json")))
    sig = truth["potential"]["sigma_rad_per_V_A"]
    xyz, _ = read_extxyz(os.path.join(sd, "ground_truth", "atoms.extxyz"))
    per, step = layer_period(xyz[:, 2])
    lay = np.round((xyz[:, 2] % per) / step).astype(int) % 4
    tr = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)
    pot, dr_t, st = tr["potential"]
    rd = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.run)
    s = json.load(open(os.path.join(rd, "summary.json")))
    q = np.load(os.path.join(rd, "quantem.npz"))
    obj, b = q["object_potential"], q["slice_boundaries"]
    dr, o = s["dr_quantem"], s["origin_A"]
    c, m = np.array(obj.shape[1:]) // 2, a.half
    r, cc = np.arange(c[0] - m, c[0] + m), np.arange(c[1] - m, c[1] + m)
    rA, cA = o[0] + r * dr, o[1] + cc * dr
    ext = [cA[0], cA[-1], rA[-1], rA[0]]
    inside = (xyz[:, 0] > rA[0]) & (xyz[:, 0] < rA[-1]) & (xyz[:, 1] > cA[0]) & (xyz[:, 1] < cA[-1])
    cols = {L: np.unique(np.round(xyz[inside & (lay == L)][:, :2], 2), axis=0) for L in range(4)}

    n = len(a.slices)
    fig, ax = plt.subplots(2, n, figsize=(4 * n, 10), squeeze=False)
    for j, k in enumerate(a.slices):
        t = rr.sample_truth(pot, dr_t, rA, cA, b[k] - 1e-6, b[k + 1] - 1e-6, st, dr) * sig
        inz = (xyz[:, 2] >= b[k]) & (xyz[:, 2] < b[k + 1])
        present = sorted(set(lay[inz].tolist()))
        absent = [L for L in range(4) if L not in present]
        for i, (img, nm) in enumerate([(obj[k][np.ix_(r, cc)], "QuantEM slice"), (t, "ground truth σV")]):
            axx = ax[i, j]
            axx.imshow(img, cmap="gray", extent=ext)
            for L in range(4):
                axx.scatter(cols[L][:, 1], cols[L][:, 0], s=60, facecolors="none", edgecolors=LAYER_COLORS[L], lw=1.2,
                            ls="-" if L in present else ":")
            axx.set_title(f"{nm} {b[k]:g}–{b[k + 1]:g} Å (max {img.max():.2f} rad)\n"
                          + (f"layers in slab: {present}; absent: {absent}" if absent else
                             f"layers in slab: all four {present}"), fontsize=9)
            if i == 1:
                axx.set_xlabel("y (Å)")
            axx.set_ylabel("x (Å)")
    handles = [Line2D([], [], marker="o", ls="none", mfc="none", mec=LAYER_COLORS[L], mew=1.5, ms=9,
                      label=f"layer {L}: atoms at z mod {per:.2f} Å = {L * step:.2f} Å") for L in range(4)]
    handles += [Line2D([], [], color="0.3", lw=1.5, ls="-", label="solid circle: this layer has atoms in the slab"),
                Line2D([], [], color="0.3", lw=1.5, ls=":", label="dotted circle: no atoms of this layer in the slab")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=True)
    label = a.run.replace("quantem_window_noiseless_", "")
    fig.suptitle(f"{a.specimen} / {label}: zoomed QuantEM slices (upper) vs the true potential of the same slab (lower)\n"
                 "Circles = projected Si atomic columns (true positions from atoms.extxyz). Each column is made of one "
                 f"[001] layer, repeating every {per:.2f} Å in depth; colour = layer", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(a.out, dpi=100)
    print(a.out)


if __name__ == "__main__":
    main()
