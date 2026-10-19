"""How much does each extra defocus dataset improve the S-matrix + probe recovery?

Give the folder that holds your defocus datasets; everything else is read from the data:

    ../../.venv/bin/python defocus_study.py --data /path/to/my_sample        # contains def0/, def1/, def2/, ...

For k = 1, 2, ..., N datasets it fits the S-matrix and the probe aberrations jointly (run_benchmark.py smatrix, unchanged),
refocuses through depth, and - if a ground-truth folder is present - compares each fit with the truth.  Output:
results/defocus_study/<name>/ (summary.md, summary.json, figures).  See SMATRIX_JOINT_REFOCUS.md, section 7.

Steps
  1. read every def<d>/4dstem_quantem: energy, semiangle, aperture taper, nominal defocus, detector K and dk, scan
     positions (all datasets must share the microscope settings and the scan)
  2. choose the subsets: nested, starting from the reference dataset, each step adding the dataset whose focus is
     farthest from those already used (or --subsets)
  3. per subset: write a config, link the datasets into data/defocus_study_<name>/<subset>/..., run the fit
     (skipped if it already exists; --refit to redo)
  4. evaluate against the truth (optional) and write the report

The ground truth is read only in step 4, never by the fits.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
from types import SimpleNamespace

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run_realization as rr  # noqa: E402

PY = sys.executable


# ------------------------------------------------------------------------------------------------ 1. data
def find_datasets(data):
    ds = {}
    for p in sorted(glob.glob(os.path.join(data, "def*"))):
        m = re.fullmatch(r"def(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            ds[int(m.group(1))] = os.path.abspath(p)
    if not ds:
        raise SystemExit(f"no def0/, def1/, ... folders in {data}")
    return dict(sorted(ds.items()))


def read_metadata(ds, a):
    meta = {}
    for d, p in ds.items():
        x = rr.load_dataset(p, SimpleNamespace(energy=a.energy, alpha=a.alpha, defocus=None, taper=a.taper), d)
        meta[d] = dict(energy=x["energy"], alpha_mrad=x["alpha_mrad"], taper_mrad=x["taper_mrad"],
                       defocus=x["defocus"], K=int(x["data"].shape[-1]), dk=float(x["dk"][0]),
                       step=[float(v) for v in x["step"]], scan=list(x["data"].shape[:2]),
                       positions=x["positions_A"], cell=None if x["cell"] is None else [float(v) for v in x["cell"]])
    first = meta[min(meta)]
    for d, m in meta.items():
        for k in ("energy", "alpha_mrad", "taper_mrad", "K", "dk", "scan"):
            if np.any(np.abs(np.asarray(m[k], float) - np.asarray(first[k], float)) > 1e-6):
                raise SystemExit(f"def{d}: {k} = {m[k]} differs from def{min(meta)} ({first[k]}); all datasets must "
                                 f"share the microscope settings and the scan")
        if m["positions"] is None:
            raise SystemExit(f"def{d}: no scan positions (padded_scan_positions_A) in the metadata")
        if np.abs(m["positions"] - first["positions"]).max() > 1e-6:
            raise SystemExit(f"def{d}: scan positions differ from def{min(meta)}")
    return meta


def nested_order(focus, ref):
    """Reference first, then each time the dataset whose focus is farthest from all chosen ones."""
    order, rest = [ref], [d for d in focus if d != ref]
    while rest:
        nxt = max(rest, key=lambda d: (min(abs(focus[d] - focus[o]) for o in order), -d))
        order.append(nxt)
        rest.remove(nxt)
    return order


# ------------------------------------------------------------------------------------------------ 3. configs and links
def subset_tag(sub):
    return f"k{len(sub)}_" + "-".join(f"d{d}" for d in sub)


def write_config(path, name, tag, meta, sub, ref, T, dr, Y):
    m0 = meta[sub[0]]
    cfg = dict(
        version=1, name=name,
        microscope=dict(energy_ev=m0["energy"], alpha_mrad=m0["alpha_mrad"], aperture_taper_mrad=m0["taper_mrad"]),
        sampling=dict(dr_A=dr, K=m0["K"], Y=Y, scan_shape=m0["scan"]),
        specimen=dict(thickness_A=float(T)),
        defocus=dict(reference=sub.index(ref), nominal_C10=[-float(meta[d]["defocus"]) for d in sub],
                     source_datasets=list(sub)),
        specimens={tag: dict(group=tag, features=[])},
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# written by defocus_study.py: datasets {sub} of the study '{name}' (def<j> here = source def{sub})\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src):
            return
        if os.path.islink(dst):
            os.remove(dst)
        else:
            raise SystemExit(f"{dst} exists and is not a link; remove it first")
    os.symlink(src, dst)


def link_subset(spec_dir, ds, sub, a, truth):
    vdir = os.path.join(spec_dir, f"{a.variant}_{a.noise}")
    os.makedirs(vdir, exist_ok=True)
    for j, d in enumerate(sub):
        link(ds[d], os.path.join(vdir, f"def{j}"))
    if truth is None:
        return
    gdir = os.path.join(spec_dir, "ground_truth")                  # truth renumbered to the subset
    os.makedirs(gdir, exist_ok=True)
    for f in ("potential.zarr", "atoms.extxyz", "polycrystal.json"):
        if os.path.exists(os.path.join(truth, f)):
            link(os.path.join(truth, f), os.path.join(gdir, f))
    for j, d in enumerate(sub):
        if os.path.exists(os.path.join(truth, f"probe_d{d}.zarr")):
            link(os.path.join(truth, f"probe_d{d}.zarr"), os.path.join(gdir, f"probe_d{j}.zarr"))
    t = json.load(open(os.path.join(truth, "truth.json")))
    ab = t.get("aberrations", {})
    for k in ("nominal", "true", "focus_depth_nominal_A", "focus_depth_true_A"):
        if k in ab:
            ab[k] = [ab[k][d] for d in sub]
    ab["reference_dataset"] = 0 if ab.get("reference_dataset") is None else (
        sub.index(ab["reference_dataset"]) if ab["reference_dataset"] in sub else None)
    ab["source_datasets"] = list(sub)
    t["aberrations"] = ab
    tmp = os.path.join(gdir, "truth.json.tmp")
    json.dump(t, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(gdir, "truth.json"))


def run_fit(cfg_path, tag, run_dir, a, log_path):
    if os.path.exists(os.path.join(run_dir, "refocused_phase.npy")) and not a.refit:
        print(f"[{tag}] fit exists, skipped ({run_dir})", flush=True)
        return True
    cmd = [PY, "run_benchmark.py", "smatrix", "--config", cfg_path, "--specimen", tag, "--variant", a.variant,
           "--noise", a.noise, "--epochs", str(a.epochs), "--warmup", str(a.warmup), "--batch", str(a.batch),
           "--lr-S", str(a.lr_S), "--heldout", "0.0", "--dz", str(a.dz), "--tag", a.tag, "--device", a.device,
           "--allow-underdetermined", "--monitor-every", str(a.monitor_every), "--checkpoint-every", "5"]
    if a.lr_probe is not None:
        cmd += ["--lr-probe", str(a.lr_probe)]
    if os.path.exists(os.path.join(run_dir, "checkpoint.pt")) and not a.refit:
        cmd.append("--resume")
    print(f"[{tag}] {time.strftime('%H:%M:%S')} fitting: {' '.join(cmd)}", flush=True)
    with open(log_path, "a") as f:
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and os.path.exists(os.path.join(run_dir, "refocused_phase.npy"))
    print(f"[{tag}] {time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED, see ' + log_path}", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------ 4. evaluation
def evaluate(cfg_path, tag, run_dir, sub, has_truth):
    import bench
    from plot_fit_monitor import Truth, corr, potential_estimate
    from evaluate_reconstruction import true_probes_at_beams
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    arr = np.load(os.path.join(run_dir, "arrays.npz"))
    coeff = np.array(s["coefficients"])
    geo = {c["check"]: c["value"] for c in s["geometry"]}
    r = dict(tag=tag, datasets=list(sub), k=len(sub), loss_train=s["loss_train"], fit_time_h=s["fit_time_s"] / 3600,
             measurements_per_unknown=geo.get("measurements/unknowns"),
             fitted={f"def{d}": dict(zip(("C10", "C12a", "C12b"), coeff[j, :3].round(3).tolist())) for j, d in enumerate(sub)},
             nominal={f"def{d}": dict(zip(("C10", "C12a", "C12b"), np.array(s["nominal"])[j, :3].round(3).tolist()))
                      for j, d in enumerate(sub)},
             focus_layers=s["depth"]["layers"])
    if not has_truth:
        return r, None
    cfg = bench.load_config(cfg_path)
    tr = Truth(cfg, tag)
    gt = os.path.join(bench.specimen_dir(cfg, tag), "ground_truth")
    ky, kx = arr["beams_ky"], arr["beams_kx"]
    # probe: aberration error and RMS aperture phase error (piston and tilt removed) per dataset
    tp = true_probes_at_beams(gt, ky, kx, tr.setup.dr)
    perr = {}
    for j, d in enumerate(sub):
        e = dict(zip(("C10", "C12a", "C12b"), (coeff[j, :3] - tr.true[j][:3]).round(3).tolist()))
        if j < len(tp):
            e["rms_phase_rad"] = rr.probe_match(arr["probe_beams"][j], tp[j], ky, kx)[1]
        perr[f"def{d}"] = e
    r["truth_aberrations"] = {f"def{d}": dict(zip(("C10", "C12a", "C12b"), np.round(tr.true[j][:3], 3).tolist()))
                              for j, d in enumerate(sub)}
    r["probe_error"] = perr
    r["probe_rms_rad_max"] = max((e.get("rms_phase_rad", np.nan) for e in perr.values()), default=np.nan)
    # object: refocused phase sections vs the true potential
    c0, c1 = s["crop"]
    rows, cols = tr.region(s["s_grid_offset_px"], c0, c1 - c0)
    zg = arr["depth_grid"]
    ph = np.load(os.path.join(run_dir, "refocused_phase.npy"), mmap_mode="r")
    T = tr.T
    proj = tr.slab(-1, T + 1, rows, cols)
    imid = int(np.argmin(np.abs(zg - T / 2)))
    r["projected_corr"] = corr(potential_estimate(np.exp(1j * np.asarray(ph[imid], float))), proj)
    zs = np.arange(10.0, T, 20.0)                                   # sections every 20 A vs true +-10 A slabs
    secs = [np.asarray(ph[int(np.argmin(np.abs(zg - z)))], float) for z in zs]
    slabs = [tr.slab(z - 10, z + 10, rows, cols) for z in zs]
    C = np.array([[corr(p, t) for t in slabs] for p in secs])       # section i vs true slab j
    pc = np.array([corr(p, proj) for p in secs])
    best = C.argmax(1)
    r.update(section_depths_A=zs.tolist(), slab_corr=np.diag(C).round(4).tolist(),
             mean_slab_corr=float(np.diag(C).mean()), mean_proj_corr=float(pc.mean()),
             selectivity=float(np.diag(C).mean() - pc.mean()),
             depth_hits=int((best == np.arange(len(zs))).sum()), n_sections=len(zs),
             depth_error_A=float(np.abs(zs[best] - zs).mean()))
    inside = (zg >= 0) & (zg <= T)
    r["focus_curve_vs_truth_corr"] = corr(arr["focus_curve"][inside], np.interp(zg[inside], tr.zc, tr.depth_profile(rows, cols)))
    return r, dict(C=C, zs=zs, secs=secs, slabs=slabs, proj=proj)


def grain_map(cfg_path, tag, run, out, log_path):
    cmd = [PY, "regularized_experiment/smatrix_grains_3d.py", "--config", cfg_path, "--specimen", tag, "--run", run,
           "--quantem-run", "", "--smatrix-label", f"S-matrix ({tag})", "--out", out]
    with open(log_path, "a") as f:
        rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    js = os.path.join(out, "grains_3d.json")
    if rc or not os.path.exists(js):
        return None
    res = json.load(open(js))["results"]
    g = next(v for k, v in res.items() if k.startswith("S-matrix"))
    t = res.get("truth potential", {})
    return dict(accuracy=g["voxel_accuracy"], bands=g["band_accuracy_40A"], truth_limit=t.get("voxel_accuracy"),
                chance=g.get("chance"))


# ------------------------------------------------------------------------------------------------ report
def report(R, extra, out, a, meta, ds, order, has_truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    ks = [r["k"] for r in R]
    lab = [f"{r['k']}: " + "+".join(f"d{d}" for d in r["datasets"]) for r in R]
    L = [f"# Defocus study: {a.name}", "",
         f"Data: `{os.path.abspath(a.data)}`" + (f"; truth: `{a.truth}`" if has_truth else "; no ground truth (fits only)"),
         f"Fits: {a.epochs} epochs, warm-up {a.warmup}, batch {a.batch}, lr_S {a.lr_S}, all patterns, "
         f"trainable C10/C12a/C12b on every dataset except the reference (fixed: the gauge).", "",
         "| dataset | nominal defocus (A) = -C10 |", "|---|---|"]
    L += [f"| def{d}{' (reference)' if d == a.reference else ''} | {meta[d]['defocus']:g} |" for d in ds]
    L += ["", f"Order of adding datasets: {' -> '.join(f'def{d}' for d in order)}", "",
          "## Fitted probe aberrations (C10 / C12a / C12b, A)", "",
          "| subset | " + " | ".join(f"def{d}" for d in ds) + " | loss | meas./unknown | fit (h) |",
          "|" + "---|" * (len(ds) + 4)]
    for r in R:
        cells = []
        for d in ds:
            f = r["fitted"].get(f"def{d}")
            cells.append("" if f is None else f"{f['C10']:.2f} / {f['C12a']:.2f} / {f['C12b']:.2f}")
        L.append(f"| {lab[R.index(r)]} | " + " | ".join(cells) +
                 f" | {r['loss_train']:.3e} | {r['measurements_per_unknown']:.2f} | {r['fit_time_h']:.2f} |")
    if has_truth:
        tru = {}
        for r in R:
            tru.update(r["truth_aberrations"])
        L.append("| truth | " + " | ".join(
            f"{tru[f'def{d}']['C10']:.2f} / {tru[f'def{d}']['C12a']:.2f} / {tru[f'def{d}']['C12b']:.2f}"
            if f"def{d}" in tru else "" for d in ds) + " | | | |")
        g = any("grains" in r and r["grains"] for r in R)
        L += ["", "## Recovery vs truth", "",
              "| subset | probe RMS phase error, worst dataset (rad) | slab corr (mean) | projection corr (mean) | "
              "selectivity | depth hits | mean depth error (A) | focus curve vs truth | projected corr |"
              + (" grain accuracy |" if g else ""),
              "|" + "---|" * (9 + g)]
        for i, r in enumerate(R):
            L.append(f"| {lab[i]} | {r['probe_rms_rad_max']:.4f} | {r['mean_slab_corr']:.3f} | {r['mean_proj_corr']:.3f} | "
                     f"{r['selectivity']:+.3f} | {r['depth_hits']} / {r['n_sections']} | {r['depth_error_A']:.1f} | "
                     f"{r['focus_curve_vs_truth_corr']:.2f} | {r['projected_corr']:.3f} |"
                     + (f" {r['grains']['accuracy']:.3f} |" if g and r.get("grains") else (" |" if g else "")))
        L += ["", "Metrics (truth used here only):", "",
              "- **probe RMS phase error**: fitted vs true probe inside the aperture, piston and tilt removed; worst "
              "dataset of the subset. The reference dataset is fixed at its nominal value, so it is 0 if that is exact.",
              "- **slab corr**: refocused phase at z = 10, 30, ... A vs the true potential of the +-10 A slab around z "
              "(mean over sections). **projection corr**: the same sections vs the whole projected potential. "
              "**selectivity** = slab - projection: > 0 means the sections carry depth information beyond the projection.",
              "- **depth hits**: sections whose best-matching true slab is the right one. **mean depth error**: "
              "|z of best-matching slab - z| averaged over sections (20 A grid).",
              "- **focus curve vs truth**: correlation of the S-matrix focus curve with the true lateral-variance depth "
              "profile.  **projected corr**: mid-depth unwrapped phase vs the true projected potential.",
              ] + (["- **grain accuracy**: 3D grain map (regularized_experiment/smatrix_grains_3d.py) vs the true grain "
                    "layout; chance = majority class." ] if g else [])
    L += ["", "## Files", "", "- `metrics_vs_defocus.png`: metrics vs number of defocus datasets",
          "- `sections_vs_truth.png`: refocused S-matrix sections per subset (and truth)",
          "- `depth_confusion.png`: section vs true slab correlation matrices (diagonal = right depth)",
          "- `summary.json`: all numbers; `configs/`: generated configs; `logs/`: fit logs",
          "- fits: `runs/<config name>/<subset>/<run>/` (S.npy, refocused_phase.npy, summary.json)"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(name=a.name, data=os.path.abspath(a.data), truth=a.truth, reference=a.reference, order=order,
                   datasets={f"def{d}": {k: v for k, v in meta[d].items() if k != "positions"} for d in ds},
                   settings=dict(epochs=a.epochs, warmup=a.warmup, batch=a.batch, lr_S=a.lr_S, lr_probe=a.lr_probe,
                                 dz=a.dz), results=R),
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)

    if not has_truth:
        return
    # metrics vs k
    panels = [("probe_rms_rad_max", "probe RMS phase error, worst dataset (rad)"),
              ("mean_slab_corr", "slab corr (mean over depth)"), ("selectivity", "selectivity (slab - projection)"),
              ("depth_error_A", "mean depth error (A)"), ("focus_curve_vs_truth_corr", "focus curve vs truth")]
    if any(r.get("grains") for r in R):
        panels.append(("grains", "3D grain accuracy"))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4))
    for axi, (k, t) in zip(ax, panels):
        v = [(r[k]["accuracy"] if r.get(k) else np.nan) if k == "grains" else r[k] for r in R]
        axi.plot(ks, v, "o-", color="tab:blue")
        if k == "grains" and R[0].get("grains"):
            axi.axhline(R[0]["grains"]["truth_limit"], color="k", ls="--", lw=0.8, label="method on true potential")
            axi.axhline(R[0]["grains"]["chance"], color="0.6", ls=":", lw=0.8, label="chance")
            axi.legend(fontsize=7)
        axi.set_xticks(ks)
        axi.set_xlabel("number of defocus datasets")
        axi.set_title(t, fontsize=9)
        axi.grid(alpha=0.3)
    fig.suptitle(f"{a.name}: S-matrix + probe recovery vs number of defocus datasets ("
                 + "; ".join(lab) + ")", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "metrics_vs_defocus.png"), dpi=110)
    plt.close(fig)
    # sections
    first = extra[0]
    zi = list(range(0, len(first["zs"]), max(1, len(first["zs"]) // 5)))[:5]
    fig, ax = plt.subplots(len(R) + 1, len(zi), figsize=(2.6 * len(zi), 2.6 * (len(R) + 1)), squeeze=False)
    for row, (r, e) in enumerate(zip(R, extra)):
        for c, i in enumerate(zi):
            ax[row, c].imshow(e["secs"][i], cmap="gray")
            ax[row, c].set_title(f"{lab[row]}  z={e['zs'][i]:.0f} A\nslab corr {e['C'][i, i]:.2f}", fontsize=7)
    for c, i in enumerate(zi):
        ax[-1, c].imshow(first["slabs"][i], cmap="gray")
        ax[-1, c].set_title(f"truth, slab {first['zs'][i] - 10:.0f}-{first['zs'][i] + 10:.0f} A", fontsize=7)
    for x in ax.ravel():
        x.set_xticks([]), x.set_yticks([])
    fig.suptitle("refocused S-matrix phase sections per subset (rows) vs the true +-10 A slab (bottom)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sections_vs_truth.png"), dpi=100)
    plt.close(fig)
    # confusion
    fig, ax = plt.subplots(1, len(R), figsize=(3.8 * len(R) + 1, 3.6), squeeze=False, constrained_layout=True)
    for i, (r, e) in enumerate(zip(R, extra)):
        im = ax[0, i].imshow(e["C"], cmap="viridis", vmin=0, vmax=1, origin="upper",
                             extent=[e["zs"][0] - 10, e["zs"][-1] + 10, e["zs"][-1] + 10, e["zs"][0] - 10])
        ax[0, i].set_title(f"{lab[i]}\nhits {r['depth_hits']}/{r['n_sections']}, error {r['depth_error_A']:.0f} A", fontsize=8)
        ax[0, i].set_xlabel("true slab z (A)")
        ax[0, i].set_ylabel("S-matrix section z (A)")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label="correlation")
    fig.savefig(os.path.join(out, "depth_confusion.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with def0/, def1/, ... (each a QuantEM Dataset4dstem zarr, "
                                                  "directly or in def<d>/4dstem_quantem)")
    ap.add_argument("--truth", default=None, help="ground-truth folder (truth.json, potential.zarr, probe_d<d>.zarr, "
                                                  "atoms.extxyz); default: <data>/../ground_truth if it exists; 'none' = skip")
    ap.add_argument("--name", default=None, help="study name (default: <parent>_<data folder name>)")
    ap.add_argument("--reference", type=int, default=0, help="dataset whose aberrations are exactly known (gauge)")
    ap.add_argument("--subsets", default=None, help='e.g. "0;0,2;0,2,1" (default: nested from the reference)')
    ap.add_argument("--max-k", type=int, default=None, help="stop after this many datasets")
    ap.add_argument("--thickness", type=float, default=None, help="specimen thickness [A] (default: truth.json, "
                                                                   "else computational_cell_A[2] in the metadata)")
    ap.add_argument("--energy", type=float, default=None, help="override the metadata [eV]")
    ap.add_argument("--alpha", type=float, default=None, help="override the metadata [mrad]")
    ap.add_argument("--taper", type=float, default=None, help="override the metadata [mrad]")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-S", dest="lr_S", type=float, default=0.02)
    ap.add_argument("--lr-probe", type=float, default=None)
    ap.add_argument("--dz", type=float, default=2.0, help="refocus step [A]")
    ap.add_argument("--monitor-every", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", default="window", choices=["window", "pixelated"], help="label only")
    ap.add_argument("--noise", default="noiseless", choices=["noiseless", "poisson"], help="label only")
    ap.add_argument("--tag", default="study")
    ap.add_argument("--no-grains", action="store_true", help="skip the 3D grain map (only for 3-grain polycrystal truth)")
    ap.add_argument("--refit", action="store_true", help="redo fits that already exist")
    ap.add_argument("--report-only", action="store_true", help="no fitting; evaluate the fits that exist")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    a = ap.parse_args()

    a.data = os.path.abspath(a.data)
    ds = find_datasets(a.data)
    if a.truth is None:
        cand = os.path.join(os.path.dirname(a.data), "ground_truth")
        a.truth = cand if os.path.exists(os.path.join(cand, "truth.json")) else None
    elif a.truth.lower() == "none":
        a.truth = None
    else:
        a.truth = os.path.abspath(a.truth)
    has_truth = a.truth is not None
    if a.name is None:
        a.name = f"{os.path.basename(os.path.dirname(a.data))}_{os.path.basename(a.data)}"
    a.name = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name)
    meta = read_metadata(ds, a)
    if a.reference not in ds:
        raise SystemExit(f"--reference {a.reference}: no def{a.reference}")
    m0 = meta[a.reference]
    dr = 1.0 / (m0["K"] * m0["dk"])
    T = a.thickness
    if T is None and has_truth:
        T = json.load(open(os.path.join(a.truth, "truth.json"))).get("thickness_A")
    if T is None and m0["cell"] is not None:
        T = m0["cell"][2]
    if T is None:
        raise SystemExit("specimen thickness unknown: pass --thickness")
    Y = int(round(m0["cell"][0] / dr)) if m0["cell"] is not None else int(np.ceil(m0["positions"].max() / dr)) + m0["K"]
    focus = {d: meta[d]["defocus"] for d in ds}
    order = nested_order(focus, a.reference)
    if a.subsets:
        subsets = [[int(v) for v in s.split(",")] for s in a.subsets.split(";") if s.strip()]
    else:
        subsets = [order[:k] for k in range(1, len(order) + 1)]
    if a.max_k:
        subsets = [s for s in subsets if len(s) <= a.max_k]
    cfg_name = f"defocus_study_{a.name}"
    out = os.path.join(HERE, "results", "defocus_study", a.name)
    for sub_dir in ("configs", "logs"):
        os.makedirs(os.path.join(out, sub_dir), exist_ok=True)

    print(f"study '{a.name}': {len(ds)} datasets in {a.data}")
    for d in ds:
        print(f"  def{d}: nominal defocus {meta[d]['defocus']:g} A (C10 = {-meta[d]['defocus']:g})"
              + ("  <- reference" if d == a.reference else ""))
    print(f"  {m0['energy'] / 1e3:g} kV, alpha {m0['alpha_mrad']:g} mrad (taper {m0['taper_mrad']:g}), K {m0['K']}, "
          f"dk {m0['dk']:.5f} 1/A -> dr {dr:.4f} A, scan {m0['scan']} step {m0['step'][0]:g} A, thickness {T:g} A")
    print(f"  truth: {a.truth or 'none (fits and fitted aberrations only)'}")
    print(f"  subsets: {subsets}")
    print(f"  output: {out}")
    if a.dry_run:
        return

    R, extra = [], []
    for sub in subsets:
        bad = [d for d in sub if d not in ds]
        if bad:
            raise SystemExit(f"subset {sub}: no def{bad[0]}")
        ref = a.reference if a.reference in sub else sub[0]
        if len(sub) > 1 and a.reference not in sub:
            print(f"  warning: subset {sub} has no def{a.reference}; def{ref} is used as the fixed (gauge) dataset")
        tag = subset_tag(sub)
        cfg_path = write_config(os.path.join(out, "configs", f"{tag}.yaml"), cfg_name, tag, meta, sub, ref, T, dr, Y)
        spec_dir = os.path.join(HERE, "data", cfg_name, tag, tag)
        link_subset(spec_dir, ds, sub, a, a.truth)
        run = f"smatrix_{a.variant}_{a.noise}_s0_{a.tag}"
        run_dir = os.path.join(HERE, "runs", cfg_name, tag, run)
        log_path = os.path.join(out, "logs", f"{tag}.log")
        if not a.report_only:
            if not run_fit(cfg_path, tag, run_dir, a, log_path):
                continue
        if not os.path.exists(os.path.join(run_dir, "refocused_phase.npy")):
            print(f"[{tag}] no fit yet, skipped in the report")
            continue
        r, e = evaluate(cfg_path, tag, run_dir, sub, has_truth)
        if has_truth and not a.no_grains and os.path.exists(os.path.join(a.truth, "polycrystal.json")):
            r["grains"] = grain_map(cfg_path, tag, run, os.path.join(out, "grains", tag), log_path)
        R.append(r)
        extra.append(e)
        report(R, extra, out, a, meta, list(ds), order, has_truth)       # updated after every subset
        print(f"[{tag}] evaluated" + (f": probe RMS {r['probe_rms_rad_max']:.4f} rad, slab corr "
                                      f"{r['mean_slab_corr']:.3f}, depth hits {r['depth_hits']}/{r['n_sections']}"
                                      if has_truth else ""), flush=True)
    if R:
        print(f"report: {os.path.join(out, 'summary.md')}")


if __name__ == "__main__":
    main()
