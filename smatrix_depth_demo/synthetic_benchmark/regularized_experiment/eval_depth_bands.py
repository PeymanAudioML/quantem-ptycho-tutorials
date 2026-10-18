"""Depth placement on a common footing for runs with different slice counts: 40 A depth bands.

Each reconstruction is summed into 5 bands of 40 A (0-40, ..., 160-200 A; whole slices, assigned by slice centre; 35
slices of 5.71 A and 20 slices of 10 A both fit exactly). The true potential is summed over the same bands (2 A
abTEM slices, exact). Over the scanned field:
  C[i, j] = corr(reconstructed band i, true band j)
  diag      = mean_i C[i, i]                               (each band vs its own true band)
  selectivity = mean_i (C[i, i] - mean_{j != i} C[i, j])   (0 = no depth information)
  hits      = number of bands whose best-matching true band is its own (of 5)
P3_XYZ grain content by band in the scanned field: [001] 1.00 / 0.73 / 0.15 / 0 / 0, [110] 0 / 0.27 / 0.80 / 0.47 / 0.05,
[111] 0 / 0 / 0.06 / 0.53 / 0.95.

Run from synthetic_benchmark/:
  ../../.venv/bin/python regularized_experiment/eval_depth_bands.py --specimen P3_XYZ --out <folder> \
      --runs <run>[=<label>],<run>[=<label>],...
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
os.chdir(BENCH)
import bench  # noqa: E402
import run_realization as rr  # noqa: E402


def load_run(cfg, sp, run, truth):
    d = os.path.join(BENCH, "runs", cfg["name"], sp, run)
    s = json.load(open(os.path.join(d, "summary.json")))
    q = np.load(os.path.join(d, "quantem.npz"))
    obj, bnd = q["object_potential"], q["slice_boundaries"]
    dr_q, o = s["dr_quantem"], s["origin_A"]
    j0 = np.array(s["first_scan_pixel"]).astype(int)
    j1 = j0 + np.round((np.array(truth["scan"]["shape"]) - 1) * truth["scan"]["step_A"] / dr_q).astype(int) + 1
    rows = o[0] + np.arange(j0[0], j1[0]) * dr_q
    cols = o[1] + np.arange(j0[1], j1[1]) * dr_q
    return s, obj[:, j0[0]:j1[0], j0[1]:j1[1]], bnd, rows, cols, dr_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poly_si_3grain.yaml")
    ap.add_argument("--specimen", default="P3_XYZ")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--band", type=float, default=40.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    sd = bench.specimen_dir(cfg, a.specimen)
    truth = json.load(open(os.path.join(sd, "ground_truth", "truth.json")))
    T, sig = truth["thickness_A"], truth["potential"]["sigma_rad_per_V_A"]
    pot, dr_t, st = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)["potential"]
    edges = np.arange(0, T + 1e-6, a.band)
    nb = len(edges) - 1
    os.makedirs(a.out, exist_ok=True)
    out, tb_cache = {}, {}
    entries = [(r.split("=")[0], r.split("=")[1] if "=" in r else r) for r in a.runs.split(",")]
    for run, label in entries:
        s, crop, bnd, rows, cols, dr_q = load_run(cfg, a.specimen, run, truth)
        zc = 0.5 * (bnd[1:] + bnd[:-1])
        rb = np.array([crop[(zc >= edges[i]) & (zc < edges[i + 1])].sum(0) for i in range(nb)])
        key = (len(rows), round(dr_q, 4))
        if key not in tb_cache:
            tb_cache[key] = np.array([rr.sample_truth(pot, dr_t, rows, cols, edges[i], edges[i + 1] - 1e-6, st, dr_q) * sig
                                      for i in range(nb)])
        tb = tb_cache[key]
        C = np.array([[rr.corr(rb[i], tb[j]) for j in range(nb)] for i in range(nb)])
        off = (C.sum(1) - np.diag(C)) / (nb - 1)
        m = dict(run=run, label=label, planes=int(s["planes"]), tv_weight_z=s.get("tv_weight_z", 0.0),
                 reported_loss=s["final_loss"], data_loss=s.get("final_data_loss", s["final_loss"]),
                 tv_loss=s.get("final_tv_loss", 0.0), projected_corr=rr.corr(crop.sum(0), tb.sum(0)),
                 band_diag=float(np.diag(C).mean()), band_selectivity=float((np.diag(C) - off).mean()),
                 band_hits=int((C.argmax(1) == np.arange(nb)).sum()), band_diag_each=np.diag(C).round(3).tolist(),
                 C=C.round(3).tolist())
        out[label] = m
        m["_rb"], m["_tb"] = rb, tb
    # table
    lines = [f"# Depth bands ({a.band:g} A) on {a.specimen}", "",
             "| run | planes | tv_weight_z | data loss | projected corr | band diag | selectivity | hits / %d | diag per band |" % nb,
             "|---|---|---|---|---|---|---|---|---|"]
    for lab, m in out.items():
        lines.append(f"| {lab} | {m['planes']} | {m['tv_weight_z']:g} | {m['data_loss']:.2f} | {m['projected_corr']:.3f} | "
                     f"{m['band_diag']:.3f} | {m['band_selectivity']:.3f} | {m['band_hits']} | "
                     f"{' / '.join(f'{v:.2f}' for v in m['band_diag_each'])} |")
    open(os.path.join(a.out, "depth_bands.md"), "w").write("\n".join(lines) + "\n")
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in out.items()},
              open(os.path.join(a.out, "depth_bands.json"), "w"), indent=1)
    print("\n".join(lines))
    # figure: per run, the 5 reconstructed bands and the correlation matrix; truth bands on top
    n = len(out)
    fig, ax = plt.subplots(n + 1, nb + 1, figsize=(2.2 * (nb + 1), 2.3 * (n + 1)), squeeze=False)
    first = next(iter(out.values()))
    for i in range(nb):
        ax[0, i].imshow(first["_tb"][i], cmap="gray")
        ax[0, i].set_title(f"truth {edges[i]:.0f}–{edges[i + 1]:.0f} Å", fontsize=8)
    ax[0, nb].axis("off")
    for r, (lab, m) in enumerate(out.items(), start=1):
        for i in range(nb):
            ax[r, i].imshow(m["_rb"][i], cmap="gray")
            ax[r, i].set_title(f"{lab}: corr {m['band_diag_each'][i]:.2f}", fontsize=7)
        im = ax[r, nb].imshow(np.array(m["C"]), vmin=0, vmax=1, cmap="viridis")
        ax[r, nb].set_title(f"C[recon, truth]; sel {m['band_selectivity']:.3f}", fontsize=7)
        ax[r, nb].set_xticks(range(nb)); ax[r, nb].set_yticks(range(nb)); ax[r, nb].tick_params(labelsize=6)
    for a_ in ax[:, :nb].ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{a.specimen}: reconstruction summed into {a.band:g} Å depth bands vs truth (scanned field); "
                 "right: corr of each reconstructed band (row) with each true band (column)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "depth_bands.png"), dpi=90)


if __name__ == "__main__":
    main()
