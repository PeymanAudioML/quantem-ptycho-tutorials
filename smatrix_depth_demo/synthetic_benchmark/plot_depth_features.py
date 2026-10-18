"""Depth output of an S-matrix run: per-feature depth response curves, feature depths, focus curve vs truth.

    python plot_depth_features.py --config configs/poly_si_3grain.yaml --specimen P3_XY \
        --runs smatrix_window_noiseless_s0_full80 smatrix_window_noiseless_s0_full80_single_d1

Reads results/smatrix/<config>/<specimen>/<run>/{summary.json, arrays.npz}; writes depth_features.png next to them.
One row per run:
  (a) every feature's response |arg(1 + r(z))| vs z (normalised to its peak; reliable ones solid), picked depth marked
  (b) histogram of picked depths with their uncertainties, and the focus-curve layers
  (c) S-matrix focus curve vs truth (potential variance per 2 A slice over the scanned field), both normalised
  (d) feature positions on the scan field, coloured by picked depth, over the true projected potential
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
from plot_fit_monitor import Truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    tr = Truth(cfg, a.specimen)
    T = tr.T
    layered = a.specimen.endswith("XYZ")  # P3_XYZ: grains stacked in depth, not columns
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    fig, ax = plt.subplots(len(a.runs), 4, figsize=(22, 4.6 * len(a.runs)), squeeze=False)
    for i, run in enumerate(a.runs):
        d = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen, run)
        with open(os.path.join(d, "summary.json")) as f:
            s = json.load(f)
        arr = np.load(os.path.join(d, "arrays.npz"))
        zg, curve = arr["depth_grid"], arr["focus_curve"]
        c0, c1 = s["crop"]
        rows, cols = tr.region(s["s_grid_offset_px"], c0, c1 - c0)
        cands = s["depth"]["candidates"]
        label = f"{run} ({len(s.get('datasets', [0, 1, 2]))} dataset{'s' if len(s.get('datasets', [0, 1, 2])) > 1 else ''})"
        cm = plt.cm.plasma
        for c in cands:
            p = np.asarray(c["profile"], float)
            p = p / max(p.max(), 1e-30)
            col = cm(np.clip(c["z"] / T, 0, 1))
            ax[i, 0].plot(zg, p, "-" if c["reliable"] else ":", color=col, lw=1.2 if c["reliable"] else 0.8, alpha=0.8)
            ax[i, 0].plot([c["z"]], [np.interp(c["z"], zg, p)], "o", color=col, ms=3)
        for x in (0, T):
            ax[i, 0].axvline(x, color="k", lw=0.8)
        n_rel = sum(c["reliable"] for c in cands)
        ax[i, 0].set_title(f"{label}\nfeature depth responses (normalised); {len(cands)} features, {n_rel} reliable "
                           "(solid) / dotted = not reliable", fontsize=8)
        ax[i, 0].set_xlabel("z (Å)")
        z = np.array([c["z"] for c in cands])
        u = np.array([c["uncertainty_A"] for c in cands])
        order = np.argsort(z)
        ax[i, 1].errorbar(z[order], np.arange(len(z)), xerr=u[order], fmt="o", ms=3, color="k", ecolor="0.7", lw=0.8)
        for v in s["depth"]["layers"]:
            ax[i, 1].axvline(v, color="tab:red", ls="--", lw=1)
        if layered:
            for zs, nm in ((40, "[001]"), (110, "[110]"), (170, "[111]")):
                ax[i, 1].axvline(zs, color="tab:green", lw=1.2, alpha=0.6)
                ax[i, 1].text(zs, len(z) - 0.5, nm, color="tab:green", fontsize=7, ha="center", va="bottom")
            ax[i, 1].axvspan(0, T, color="tab:green", alpha=0.08, label="specimen 0–T; green lines = true grain seed depths")
        else:
            ax[i, 1].axvspan(0, T, color="tab:green", alpha=0.08, label="true grains: all span 0–T")
        ax[i, 1].set_xlim(zg.min(), zg.max()), ax[i, 1].set_xlabel("picked depth z ± uncertainty (Å)")
        ax[i, 1].set_ylabel("feature (sorted by z)"), ax[i, 1].legend(fontsize=7, loc="lower right")
        ax[i, 1].set_title(f"feature depths; red dashed = focus-curve layers {s['depth']['layers']}", fontsize=8)
        prof = tr.depth_profile(rows, cols)
        nrm = lambda v: (v - v.min()) / max(np.ptp(v), 1e-30)
        ax[i, 2].plot(zg, nrm(curve), "k-", label="S-matrix focus curve")
        ax[i, 2].plot(tr.zc, nrm(prof), "g-", lw=0.8, label="truth: potential variance per 2 Å slice")
        inside = (zg >= 0) & (zg <= T)
        r = np.corrcoef(curve[inside], np.interp(zg[inside], tr.zc, prof))[0, 1]
        ax[i, 2].axvspan(zg.min(), 0, color="0.9"), ax[i, 2].axvspan(T, zg.max(), color="0.9")
        ax[i, 2].set_title(f"focus curve vs true depth profile (corr {r:.2f})", fontsize=8)
        ax[i, 2].set_xlabel("z (Å)"), ax[i, 2].legend(fontsize=7)
        proj = tr.slab(-1, T + 1, rows, cols)
        dr = tr.setup.dr
        ext = [cols[0] * dr, (cols[-1] + 1) * dr, (rows[-1] + 1) * dr, rows[0] * dr]
        ax[i, 3].imshow(proj, cmap="gray", extent=ext)
        sc = ax[i, 3].scatter([c["x"] for c in cands], [c["y"] for c in cands], c=z, cmap=cm, vmin=0, vmax=T,
                              s=[30 if c["reliable"] else 14 for c in cands], edgecolors="w", linewidths=0.4)
        fig.colorbar(sc, ax=ax[i, 3], shrink=0.8, label="picked z (Å)")
        ax[i, 3].set_title("feature positions on the true projected potential, coloured by depth", fontsize=8)
        ax[i, 3].set_xlabel("x (Å)"), ax[i, 3].set_ylabel("y (Å)")
    truth_txt = ("truth: grains stacked in depth, [001] top / [110] middle / [111] bottom, seeds at z 40 / 110 / 170 Å"
                 if layered else f"truth: grains are columns through the full {T:.0f} Å")
    fig.suptitle(f"{a.specimen}: S-matrix depth output ({truth_txt})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen, "depth_features.png")
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == "__main__":
    main()
