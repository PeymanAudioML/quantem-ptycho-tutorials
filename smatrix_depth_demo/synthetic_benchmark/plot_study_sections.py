"""All refocused S-matrix sections of a defocus study vs the true +-10 A slabs.

One row per fitted subset, the truth in the bottom row, one column per section (every 20 A, as in the study metrics).
Each panel title gives the correlation of that section with its own true slab.

    ../../.venv/bin/python plot_study_sections.py --name si_defocus_series_w100_window_noiseless

--quantem <subset tag>/<run> (repeatable) adds QuantEM rows: the reconstructed potential summed over the slices whose
centres lie in the same +-10 A slab, vs the truth sampled at the QuantEM pixels (0.1 A blur, as the S-matrix truth),
plus a truth row for the QuantEM field (it is not the same crop as the S-matrix field).  Units differ (S-matrix:
refocused phase; QuantEM: potential in rad), so each panel has its own grey scale; compare shapes and r.
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import defocus_study as ds

HERE = os.path.dirname(os.path.abspath(__file__))


def show(ax, img):
    lo, hi = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_xticks([]), ax.set_yticks([])


def quantem_sections(cfg_path, tag, run_dir, zs, blur_px=1.0):
    """QuantEM potential summed over each +-10 A slab and the matching truth at the QuantEM pixels."""
    from scipy.ndimage import gaussian_filter
    import bench
    import run_realization as rr
    cfg = bench.load_config(cfg_path)
    sd = bench.specimen_dir(cfg, tag)
    truth = json.load(open(os.path.join(sd, "ground_truth", "truth.json")))
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    q = np.load(os.path.join(run_dir, "quantem.npz"))
    obj, bnd = q["object_potential"], q["slice_boundaries"]
    dr, o = s["dr_quantem"], s["origin_A"]
    j0 = np.array(s["first_scan_pixel"]).astype(int)
    j1 = j0 + np.round((np.array(truth["scan"]["shape"]) - 1) * truth["scan"]["step_A"] / dr).astype(int) + 1
    crop = obj[:, j0[0]:j1[0], j0[1]:j1[1]].astype(float)
    rows_A, cols_A = o[0] + np.arange(j0[0], j1[0]) * dr, o[1] + np.arange(j0[1], j1[1]) * dr
    tr = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)
    pot, dr_t, st = tr["potential"]
    sig = truth["potential"]["sigma_rad_per_V_A"]
    zc = 0.5 * (bnd[1:] + bnd[:-1])
    secs, slabs = [], []
    for z in zs:
        secs.append(crop[(zc >= z - 10) & (zc < z + 10)].sum(0))
        t = rr.sample_truth(pot, dr_t, rows_A, cols_A, z - 10, z + 10, st, dr) * sig
        slabs.append(gaussian_filter(t, blur_px, mode="nearest"))
    C = np.array([[ds_corr(p, t) for t in slabs] for p in secs])
    return dict(secs=secs, slabs=slabs, C=C, planes=len(zc))


def ds_corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-30))


def scores(e):
    """Same section metrics as defocus_study.evaluate, from the section/slab stack of one row."""
    C = e["C"]
    proj = e.get("proj")
    if proj is None:
        proj = np.sum(e["slabs"], 0)
    pc = np.array([ds_corr(p, proj) for p in e["secs"]])
    d = np.diag(C)
    return dict(mean_slab_corr=float(d.mean()), mean_proj_corr=float(pc.mean()), selectivity=float(d.mean() - pc.mean()),
                depth_hits=int((C.argmax(1) == np.arange(len(d))).sum()), n_sections=len(d),
                slab_corr=d.round(3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="study name (folder in results/defocus_study/)")
    ap.add_argument("--tag", default="study", help="run tag used by defocus_study.py")
    ap.add_argument("--smatrix", action="append", default=[],
                    help="<subset tag>/<run> of an extra S-matrix fit (e.g. a joint S + probe run) to add as a row")
    ap.add_argument("--quantem", action="append", default=[],
                    help="<subset tag>/<run> of a QuantEM run to add as a row (repeatable)")
    ap.add_argument("--out", default=None, help="output png (default: <study>/sections_all_vs_truth.png)")
    a = ap.parse_args()

    out_dir = os.path.join(HERE, "results", "defocus_study", a.name)
    summary = json.load(open(os.path.join(out_dir, "summary.json")))
    rows = []
    for r in summary["results"]:
        sub, tag = r["datasets"], r["tag"]
        cfg_path = os.path.join(out_dir, "configs", f"{tag}.yaml")
        run_dir = os.path.join(HERE, "runs", f"defocus_study_{a.name}", tag, f"smatrix_window_noiseless_s0_{a.tag}")
        _, e = ds.evaluate(cfg_path, tag, run_dir, sub, True)
        rows.append(("+".join(f"def{d}" for d in sub), e))
    subsets = {r["tag"]: r["datasets"] for r in summary["results"]}
    for spec in a.smatrix:
        tag, run = spec.split("/", 1)
        sub = subsets[tag]
        run_dir = os.path.join(HERE, "runs", f"defocus_study_{a.name}", tag, run)
        r, e = ds.evaluate(os.path.join(out_dir, "configs", f"{tag}.yaml"), tag, run_dir, sub, True)
        st = json.load(open(os.path.join(run_dir, "summary.json")))
        c0 = np.array(st.get("start_coefficients", st["nominal"]))[:, 0]
        c10 = " / ".join(f"{d['C10']:.1f}" for d in r["fitted"].values())
        e["info"] = dict(run=run, loss=st["loss_train"], C10_start=c0.tolist(),
                         C10_fit=[d["C10"] for d in r["fitted"].values()])
        rows.append(("+".join(f"def{d}" for d in sub) + " S + probe\n"
                     + f"C10 start {' / '.join(f'{x:.0f}' for x in c0)} -> {c10}", e))

    zs = rows[0][1]["zs"]
    n = len(zs)
    qrows = []
    for spec in a.quantem:
        tag, run = spec.split("/", 1)
        e = quantem_sections(os.path.join(out_dir, "configs", f"{tag}.yaml"), tag,
                             os.path.join(HERE, "runs", f"defocus_study_{a.name}", tag, run), zs)
        arm = next((x for x in run.split("_") if len(x) == 1 and x.isupper()), "")
        probe = "probe trained" if "trained" in run else ("probe fixed" if "fixed" in run else "")
        sq = json.load(open(os.path.join(HERE, "runs", f"defocus_study_{a.name}", tag, run, "summary.json")))
        e["info"] = dict(run=run, loss=sq.get("final_loss", sq.get("loss")))
        qrows.append((f"QuantEM {arm}, {e['planes']} pl.\n{probe}", e))
    nr = len(rows) + 1 + len(qrows) + (1 if qrows else 0)
    fig, ax = plt.subplots(nr, n, figsize=(1.9 * n, 2.1 * nr), squeeze=False)
    for i, (label, e) in enumerate(rows):
        for j in range(n):
            show(ax[i, j], e["secs"][j])
            ax[i, j].set_title(f"z={zs[j]:.0f} A  r={e['C'][j, j]:.2f}", fontsize=7)
        ax[i, 0].set_ylabel(f"{label}\nmean r {np.diag(e['C']).mean():.3f}", fontsize=8)
    it = len(rows)
    for j in range(n):
        show(ax[it, j], rows[0][1]["slabs"][j])
        ax[it, j].set_title(f"truth {zs[j] - 10:.0f}-{zs[j] + 10:.0f} A", fontsize=7)
    ax[it, 0].set_ylabel("ground truth\n(+-10 A slab)" + ("\nS-matrix field" if qrows else ""), fontsize=8)
    for k, (label, e) in enumerate(qrows):
        i = it + 1 + k
        for j in range(n):
            show(ax[i, j], e["secs"][j])
            ax[i, j].set_title(f"z={zs[j]:.0f} A  r={e['C'][j, j]:.2f}", fontsize=7)
        ax[i, 0].set_ylabel(f"{label}\nmean r {np.diag(e['C']).mean():.3f}", fontsize=8)
    if qrows:
        for j in range(n):
            show(ax[-1, j], qrows[0][1]["slabs"][j])
            ax[-1, j].set_title(f"truth {zs[j] - 10:.0f}-{zs[j] + 10:.0f} A", fontsize=7)
        ax[-1, 0].set_ylabel("ground truth\n(+-10 A slab)\nQuantEM field", fontsize=8)
    head = f"{a.name}: refocused S-matrix phase at every section (rows: datasets used) vs the true slab"
    if qrows:
        head += ";\nQuantEM potential summed over the same 20 A slab vs the true slab in the QuantEM field (bottom)"
    fig.suptitle(head + "; r = correlation with the true slab at the same depth", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / fig.get_figheight()))
    path = a.out or os.path.join(out_dir, "sections_all_vs_truth.png")
    fig.savefig(path, dpi=110)
    print(path)
    table = [dict(row=lab.replace("\n", ", "), field="S-matrix" if k < len(rows) else "QuantEM",
                  **scores(e), **e.get("info", {}))
             for k, (lab, e) in enumerate(rows + qrows)]
    json.dump(dict(section_depths_A=list(map(float, zs)), rows=table), open(path[:-4] + "_metrics.json", "w"), indent=1)
    with open(path[:-4] + "_metrics.md", "w") as f:
        f.write("| row | field | mean slab r | mean proj r | selectivity | depth hits |\n|---|---|---|---|---|---|\n")
        for t in table:
            f.write(f"| {t['row']} | {t['field']} | {t['mean_slab_corr']:.3f} | {t['mean_proj_corr']:.3f} | "
                    f"{t['selectivity']:+.3f} | {t['depth_hits']} / {t['n_sections']} |\n")
    print(open(path[:-4] + "_metrics.md").read())


if __name__ == "__main__":
    main()
