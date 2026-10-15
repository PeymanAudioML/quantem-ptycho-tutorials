"""Side-by-side comparison of QuantEM runs that plot_quantem.py has already evaluated (reads <run>/plots/metrics.json
and <run>/quantem.npz).

    python compare_quantem.py --config configs/poly_si_3grain.yaml --specimen P3_XY --out results/quantem_compare/P3_XY \
        --runs quantem_..._A_..._grid35 quantem_..._C_..._grid35 quantem_..._D_..._grid35

Writes into --out:
  compare_metrics.png   per-slab correlation vs depth (solid: own slab, dashed: whole projection), depth profiles,
                        loss curves, summary bars (projected corr, mean slab corr - mean projection corr, probe RMS)
  compare_slices.png    reconstructed slices at the same depths for every run, with the true slab
  COMPARE.md            table of the numbers
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
import run_realization as rr

LABELS = {"A": "A: uniform slices, flat start", "B": "B: uniform slices, S-informed start",
          "C": "C: S-informed slices, flat start", "D": "D: S-informed slices + start",
          "E": "E: S-informed slices, S-matrix object start"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", type=lambda s: [float(v) for v in s.split(",")], default=[25, 60, 100, 140, 175])
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    sd = bench.specimen_dir(cfg, a.specimen)
    os.makedirs(a.out, exist_ok=True)
    R = []
    for run in a.runs:
        d = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, run)
        s = json.load(open(os.path.join(d, "summary.json")))
        m = json.load(open(os.path.join(d, "plots", "metrics.json")))
        q = np.load(os.path.join(d, "quantem.npz"))
        R.append(dict(run=run, s=s, m=m, q=q, label=LABELS.get(s["arm"], s["arm"]) + f" ({s['planes']} planes)"))
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    cols = plt.cm.tab10.colors

    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    for i, r in enumerate(R):
        b = np.array(r["m"]["slice_boundaries_A"])
        zc = 0.5 * (b[1:] + b[:-1])
        ax[0, 0].plot(zc, r["m"]["slab_corr"], "o-", ms=3, color=cols[i], label=r["label"])
        ax[0, 0].plot(zc, r["m"]["slab_corr_vs_projection"], "--", color=cols[i], lw=0.8)
        obj = r["q"]["object_potential"]
        prof = obj.reshape(len(obj), -1).std(1) / np.diff(b)
        ax[0, 1].step(b, np.r_[prof, prof[-1]] / prof.max(), where="post", color=cols[i], label=r["label"])
        ax[0, 2].semilogy(np.arange(1, len(r["q"]["loss"]) + 1), r["q"]["loss"], color=cols[i], label=r["label"])
    ax[0, 0].set_xlabel("slab centre z (Å)"), ax[0, 0].set_ylabel("correlation")
    ax[0, 0].set_title("per-slab corr: solid = vs own true slab, dashed = vs whole projection"), ax[0, 0].legend(fontsize=7)
    for v in json.load(open(os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen,
                                         R[0]["s"]["smatrix_run"], "summary.json")))["depth"]["candidates"]:
        ax[0, 0].axvline(v["z"], color="0.8", lw=0.5, zorder=0)
    ax[0, 1].set_xlabel("depth z (Å)"), ax[0, 1].set_title("reconstruction: lateral std per Å (normalised)")
    ax[0, 1].legend(fontsize=7)
    ax[0, 2].set_xlabel("iteration"), ax[0, 2].set_title("QuantEM loss"), ax[0, 2].legend(fontsize=7)
    x = np.arange(len(R))
    names = [f"{r['s']['arm']}{r['s']['planes']}" for r in R]
    for j, (key, title) in enumerate((("proj", "projected potential corr vs truth"),
                                      ("sel", "depth selectivity = mean slab corr − mean projection corr"),
                                      ("probe", "final probe RMS phase error (rad)"))):
        v = [r["m"]["projected"]["corr"] if key == "proj" else
             r["m"]["mean_slab_corr"] - r["m"]["mean_proj_corr"] if key == "sel" else r["m"]["probe"]["final_rms_rad"]
             for r in R]
        ax[1, j].bar(x, v, color=[cols[i] for i in x])
        ax[1, j].set_xticks(x, names), ax[1, j].set_title(title), ax[1, j].axhline(0, color="k", lw=0.8)
        for xi, vi in zip(x, v):
            ax[1, j].text(xi, vi, f"{vi:.3f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=8)
    fig.suptitle(f"{a.specimen}: QuantEM, uniform vs S-matrix-informed slices (dataset {R[0]['s']['dataset']}, "
                 f"{R[0]['s']['iters']} iters, probe {R[0]['s']['probe_mode']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "compare_metrics.png"), dpi=90)
    plt.close(fig)

    # slices at common depths
    truth = json.load(open(os.path.join(sd, "ground_truth", "truth.json")))
    sig = truth["potential"]["sigma_rad_per_V_A"]
    tr = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)
    pot, dr_t, st = tr["potential"]
    fig, ax = plt.subplots(len(R) + 1, len(a.depths), figsize=(3.2 * len(a.depths), 3.3 * (len(R) + 1)), squeeze=False)
    for i, r in enumerate(R):
        s, b = r["s"], np.array(r["m"]["slice_boundaries_A"])
        dr_q, o = s["dr_quantem"], s["origin_A"]
        j0 = np.array(s["first_scan_pixel"]).astype(int)
        j1 = j0 + np.round((np.array(truth["scan"]["shape"]) - 1) * truth["scan"]["step_A"] / dr_q).astype(int) + 1
        crop = r["q"]["object_potential"][:, j0[0]:j1[0], j0[1]:j1[1]]
        for j, zz in enumerate(a.depths):
            k = int(np.clip(np.searchsorted(b, zz) - 1, 0, len(b) - 2))
            ax[i, j].imshow(crop[k], cmap="gray")
            ax[i, j].set_title(f"{s['arm']}: slab {b[k]:.0f}–{b[k + 1]:.0f} Å, corr {r['m']['slab_corr'][k]:.2f}", fontsize=8)
            if i == 0:
                rows_A = o[0] + np.arange(j0[0], j1[0]) * dr_q
                cols_A = o[1] + np.arange(j0[1], j1[1]) * dr_q
                t = rr.sample_truth(pot, dr_t, rows_A, cols_A, zz - 2.5, zz + 2.5, st, dr_q) * sig
                ax[-1, j].imshow(t, cmap="gray"), ax[-1, j].set_title(f"truth σV, {zz - 2.5:.1f}–{zz + 2.5:.1f} Å", fontsize=8)
    for a_ in ax.ravel():
        a_.axis("off")
    fig.suptitle(f"{a.specimen}: reconstructed slab containing each depth (rows = runs), truth 5 Å slab (bottom)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "compare_slices.png"), dpi=90)
    plt.close(fig)

    L = [f"# QuantEM: uniform vs S-matrix-informed slices, {a.specimen}", "",
         f"Dataset {R[0]['s']['dataset']}, {R[0]['s']['iters']} iterations, probe {R[0]['s']['probe_mode']}. "
         "Depth selectivity = mean slab corr (vs own true slab) − mean corr vs the whole projection; > 0 means the "
         "slices carry depth information.", "",
         "| run | planes | final loss | projected corr | slope | NRMSE | mean slab corr | mean proj corr | depth selectivity | "
         "depth profile corr | probe overlap | probe RMS (rad) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        m = r["m"]
        L.append(f"| {r['label']} | {m['planes']} | {m['final_loss']:.4g} | {m['projected']['corr']:.3f} | "
                 f"{m['projected']['slope']:.2f} | {m['projected']['nrmse_fit']:.2f} | {m['mean_slab_corr']:.3f} | "
                 f"{m['mean_proj_corr']:.3f} | {m['mean_slab_corr'] - m['mean_proj_corr']:+.3f} | "
                 f"{m['depth_profile_corr']:.2f} | {m['probe']['final_overlap']:.4f} | {m['probe']['final_rms_rad']:.3f} |")
    open(os.path.join(a.out, "COMPARE.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
