"""Ground-truth evaluation of the single-focus S-matrix runs: nominal probe fixed vs probe updated jointly (experimental).

    python evaluate_single_focus.py --config configs/poly_si_3grain.yaml --specimen P3_XY --pattern "*_sf80*single_d1*"

Reads results/smatrix/<config>/<specimen>/<run>/{summary.json, arrays.npz} and runs/.../refocused_phase.npy; the ground
truth is read here only (never by the fits).  Writes results/single_focus_joint/<specimen>/:
  metrics.json         per run and per group (mean / std over runs)
  SINGLE_FOCUS.md      tables
  losses.png           running training loss, fresh full-training-set loss, held-out loss per epoch
  probe.png            C10 / C12a / C12b trajectories and probe RMS phase error per epoch; true and nominal values
  probe_phase.png      final aperture phase error vs the true probe (piston and tilt removed) per run
  projected.png        projected-potential estimate (unwrapped phase at mid-depth) vs the true projection
  depth.png            focus curve vs the true depth profile; candidate depths with uncertainties

Probe: complex overlap and RMS aperture phase error vs the true dataset probe at the S-matrix beams
(run_realization.probe_match).  Projected: the conversion used for the QuantEM handoff (phase of the S refocused to
T/2, 2-D unwrapped), correlation and least-squares slope vs sigma * projected V (Gaussian blur 0.2 A); the best
correlation over all refocus depths is also listed as a truth-selected diagnostic.  Depth: refocused phase sections
every 20 A vs the true +-10 A slab and vs the whole projection (selectivity = slab - projection); focus curve vs the
true lateral-variance depth profile; focus-curve layers and candidate depth clusters (P3_XY has columnar grains, so a
localised depth peak is not supported by the truth).
"""
import argparse, glob, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench
import run_realization as rr
import simple_joint as sj
from evaluate_reconstruction import true_probes_at_beams
from plot_fit_monitor import Truth, corr, potential_estimate

SEC = np.arange(10.0, 200.0, 20.0)


def fit_line(est, tru):
    A = np.stack([tru.ravel(), np.ones(tru.size)], 1)
    (sl, off), *_ = np.linalg.lstsq(A, est.ravel(), rcond=None)
    return dict(corr=corr(est, tru), slope=float(sl), offset=float(off),
                rmse_after_fit_rad=float(np.sqrt(((est - sl * tru - off) ** 2).mean())), truth_std_rad=float(tru.std()))


def group_of(s):
    if s.get("single_probe_mode") != "joint":
        return "fixed"
    return "joint" if not s.get("probe_init_override") else "joint_start"


def evaluate(cfg, specimen, run, tr, tp):
    d = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], specimen, run)
    s = json.load(open(os.path.join(d, "summary.json")))
    arr = np.load(os.path.join(d, "arrays.npz"))
    ds = s["datasets"][0]
    ky, kx = arr["beams_ky"], arr["beams_kx"]
    true_c = tr.true[ds]
    ptrue = tp[ds]
    coeff = np.array(s["coefficients"])[0]
    start = np.array(s.get("start_coefficients", s["nominal"]))[0]
    ov, rms = rr.probe_match(arr["probe_beams"][0], ptrue, ky, kx)
    ov0, rms0 = rr.probe_match(tr.probe(start), ptrue, ky, kx)
    traj = []
    for ep, c in zip(s["fit_trace"]["epoch"], s["fit_trace"]["coefficients"]):
        o, r = rr.probe_match(tr.probe(c[0]), ptrue, ky, kx)
        traj.append(dict(epoch=ep, C10=c[0][0], C12a=c[0][1], C12b=c[0][2], overlap=o, rms_rad=r))
    probe = dict(start=dict(zip(("C10", "C12a", "C12b"), start[:3].tolist())),
                 final=dict(zip(("C10", "C12a", "C12b"), coeff[:3].tolist())),
                 truth=dict(zip(("C10", "C12a", "C12b"), true_c[:3].tolist())),
                 error=dict(zip(("C10", "C12a", "C12b"), (coeff[:3] - true_c[:3]).tolist())),
                 start_overlap=ov0, start_rms_rad=rms0, overlap=ov, rms_rad=rms, trajectory=traj)
    # object: refocused phase (S-grid crop), the same conversion as the QuantEM handoff
    c0, c1 = s["crop"]
    rows, cols = tr.region(s["s_grid_offset_px"], c0, c1 - c0)
    zg = arr["depth_grid"]
    ph = np.load(os.path.join(bench.HERE, "runs", cfg["name"], specimen, run, "refocused_phase.npy"), mmap_mode="r")
    T = tr.T
    proj_t = tr.slab(-1, T + 1, rows, cols)
    imid = int(np.argmin(np.abs(zg - T / 2)))
    est = potential_estimate(np.exp(1j * np.asarray(ph[imid], float)))
    projected = fit_line(est, proj_t)
    zc_all = [corr(np.asarray(ph[i], float), proj_t) for i in range(len(zg))]
    ib = int(np.argmax(zc_all))
    projected.update(mid_depth_A=float(zg[imid]), best_depth_A=float(zg[ib]), best_depth_corr=float(zc_all[ib]),
                     corr_vs_depth=[float(v) for v in zc_all])
    secs = []
    for z in SEC:
        i = int(np.argmin(np.abs(zg - z)))
        p = np.asarray(ph[i], float)
        secs.append(dict(z=float(z), corr_slab=corr(p, tr.slab(z - 10, z + 10, rows, cols)), corr_proj=corr(p, proj_t)))
    inside = (zg >= 0) & (zg <= T)
    prof_t = np.interp(zg[inside], tr.zc, tr.depth_profile(rows, cols))
    curve = arr["focus_curve"]
    cands = s["depth"]["candidates"]
    z = np.array([c["z"] for c in cands])
    unc = np.array([c["uncertainty_A"] for c in cands])
    multi = np.array([len(c.get("z_peaks") or []) >= 2 for c in cands])
    depth = dict(section_corr=secs, mean_slab_corr=float(np.mean([x["corr_slab"] for x in secs])),
                 mean_proj_corr=float(np.mean([x["corr_proj"] for x in secs])),
                 focus_curve_vs_truth_corr=corr(curve[inside], prof_t),
                 truth_profile_cv=float(prof_t.std() / prof_t.mean()),
                 layers=s["depth"]["layers"], n_layers=len(s["depth"]["layers"]),
                 n_candidates=len(cands), n_reliable=int(sum(c["reliable"] for c in cands)),
                 z_median=float(np.median(z)) if len(z) else None,
                 z_quartiles=np.percentile(z, [25, 75]).tolist() if len(z) else None,
                 uncertainty_median_A=float(np.median(unc)) if len(unc) else None,
                 frac_multi_peak=float(multi.mean()) if len(z) else None,
                 n_shallow=int((z < 0.35 * T).sum()), n_middle=int(((z >= 0.35 * T) & (z <= 0.65 * T)).sum()),
                 n_deep=int((z > 0.65 * T).sum()),
                 depth_shift_from_C10_change_A=float(coeff[0] - start[0]))
    depth["selectivity"] = depth["mean_slab_corr"] - depth["mean_proj_corr"]
    sel = s["selection"]
    return dict(run=run, group=group_of(s), seed=s["seed"], mode=s.get("single_probe_mode"),
                start=probe["start"], best_epoch=sel["best_epoch"], selection=sel["method"],
                loss_train=s["loss_train"], loss_heldout=s["loss_heldout"],
                heldout_over_train=s["loss_heldout"] / s["loss_train"],
                evaluations=sel["evaluations"], running_loss=s["fit_trace"]["loss"],
                probe_update_check=s.get("probe_update_check"), fit_time_s=s["fit_time_s"],
                settings={k: s.get(k) for k in ("epochs", "warmup", "batch", "lr_S", "lr_probe", "mode", "trainable",
                                                "heldout_fraction", "split_seed", "n_train", "n_heldout")},
                probe=probe, projected=projected, depth=depth, _zg=zg, _curve=curve, _prof=(zg[inside], prof_t),
                _est=est, _proj=proj_t, _cands=cands, _ky=ky, _kx=kx, _psi=arr["probe_beams"][0], _ptrue=ptrue)


SCALARS = [("loss_train", lambda r: r["loss_train"]), ("loss_heldout", lambda r: r["loss_heldout"]),
           ("C10", lambda r: r["probe"]["final"]["C10"]), ("C12a", lambda r: r["probe"]["final"]["C12a"]),
           ("C12b", lambda r: r["probe"]["final"]["C12b"]), ("probe_overlap", lambda r: r["probe"]["overlap"]),
           ("probe_rms_rad", lambda r: r["probe"]["rms_rad"]), ("proj_corr", lambda r: r["projected"]["corr"]),
           ("proj_slope", lambda r: r["projected"]["slope"]),
           ("selectivity", lambda r: r["depth"]["selectivity"]),
           ("focus_vs_truth", lambda r: r["depth"]["focus_curve_vs_truth_corr"]),
           ("n_layers", lambda r: r["depth"]["n_layers"])]


def aperture_phase_error(psi, ref, ky, kx):
    w = np.abs(ref) ** 2
    d = np.angle(psi * np.conj(ref))
    d = np.angle(np.exp(1j * (d - np.angle((w * np.exp(1j * d)).sum()))))
    A = np.stack([np.ones_like(ky), ky, kx], 1) * np.sqrt(w)[:, None]
    c = np.linalg.lstsq(A, d * np.sqrt(w), rcond=None)[0]
    return np.angle(np.exp(1j * (d - (c[0] + c[1] * ky + c[2] * kx))))


def label(r):
    if r["group"] == "joint_start":
        st = ", ".join(f"{k} {v:g}" for k, v in r["start"].items() if abs(v) > 0 and not (k == "C10" and v == -100.0))
        return f"joint s{r['seed']} start {st}"
    return f"{r['group']} s{r['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--pattern", default="*_sf80*single_d1*")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    tr = Truth(cfg, a.specimen)
    gt = os.path.join(bench.specimen_dir(cfg, a.specimen), "ground_truth")
    base = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen)
    runs = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(base, a.pattern, "summary.json")))
    first = np.load(os.path.join(base, runs[0], "arrays.npz"))
    tp = true_probes_at_beams(gt, first["beams_ky"], first["beams_kx"], tr.setup.dr)
    R = [evaluate(cfg, a.specimen, r, tr, tp) for r in runs]
    order = {"fixed": 0, "joint": 1, "joint_start": 2}
    R.sort(key=lambda r: (order[r["group"]], r["seed"], r["run"]))
    out = os.path.join(bench.HERE, "results", "single_focus_joint", a.specimen)
    os.makedirs(out, exist_ok=True)
    groups = {}
    for g in order:
        rs = [r for r in R if r["group"] == g]
        if rs:
            groups[g] = {k: dict(mean=float(np.mean([f(r) for r in rs])), std=float(np.std([f(r) for r in rs])),
                                 min=float(np.min([f(r) for r in rs])), max=float(np.max([f(r) for r in rs])),
                                 n=len(rs)) for k, f in SCALARS}
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in R]
    json.dump(dict(runs=clean, groups=groups, specimen=a.specimen, truth_coefficients=tr.true.tolist()),
              open(os.path.join(out, "metrics.json"), "w"), indent=1)

    # ---- tables
    L = [f"# Single-focus S-matrix (dataset 1): fixed vs joint probe, {a.specimen}", "",
         "Ground truth used for evaluation only. Joint = experimental (probe phase not identifiable from one dataset).",
         "", "| run | best epoch | train loss | held-out loss | held-out / train | C10 | C12a | C12b | probe overlap | "
         "probe RMS (rad) | proj corr | slope | best-z corr (z) | slab / proj corr | selectivity | focus vs truth | "
         "layers (A) | candidates (reliable) | z median [IQR] | median unc. | multi-peak |",
         "|" + "---|" * 21]
    for r in R:
        p, pr, d = r["probe"], r["projected"], r["depth"]
        L.append(f"| {label(r)} | {r['best_epoch']} | {r['loss_train']:.3e} | {r['loss_heldout']:.3e} | "
                 f"{r['heldout_over_train']:.1f} | {p['final']['C10']:.2f} | {p['final']['C12a']:.2f} | "
                 f"{p['final']['C12b']:.2f} | {p['overlap']:.4f} | {p['rms_rad']:.3f} | {pr['corr']:.3f} | "
                 f"{pr['slope']:.3f} | {pr['best_depth_corr']:.3f} ({pr['best_depth_A']:.0f}) | "
                 f"{d['mean_slab_corr']:.3f} / {d['mean_proj_corr']:.3f} | {d['selectivity']:+.3f} | "
                 f"{d['focus_curve_vs_truth_corr']:.2f} | {', '.join(f'{v:.0f}' for v in d['layers'])} | "
                 f"{d['n_candidates']} ({d['n_reliable']}) | {d['z_median']:.0f} [{d['z_quartiles'][0]:.0f}, "
                 f"{d['z_quartiles'][1]:.0f}] | {d['uncertainty_median_A']:.0f} | {d['frac_multi_peak']:.2f} |")
    L += ["", f"True probe (dataset 1): C10 {tr.true[1][0]:.1f}, C12a {tr.true[1][1]:.1f}, C12b {tr.true[1][2]:.1f} A; "
          f"nominal C10 {tr.nominal[1][0]:.1f}.", "", "| group | n | " + " | ".join(k for k, _ in SCALARS) + " |",
          "|" + "---|" * (len(SCALARS) + 2)]
    for g, v in groups.items():
        L.append(f"| {g} | {v['loss_train']['n']} | " + " | ".join(
            f"{v[k]['mean']:.4g} ± {v[k]['std']:.2g}" for k, _ in SCALARS) + " |")
    open(os.path.join(out, "SINGLE_FOCUS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- figures
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    cmap = {"fixed": plt.cm.Blues, "joint": plt.cm.Reds, "joint_start": plt.cm.Greens}
    cnt = {g: 0 for g in order}
    col = {}
    for r in R:
        n = sum(1 for x in R if x["group"] == r["group"])
        col[r["run"]] = cmap[r["group"]](0.45 + 0.5 * cnt[r["group"]] / max(n, 1))
        cnt[r["group"]] += 1

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for r in R:
        c = col[r["run"]]
        ax[0].semilogy(np.arange(1, len(r["running_loss"]) + 1), r["running_loss"], color=c, label=label(r))
        ev = r["evaluations"]
        ax[1].semilogy([e["epoch"] for e in ev], [e["train"] for e in ev], color=c, label=label(r))
        ax[2].semilogy([e["epoch"] for e in ev], [e["heldout"] for e in ev], color=c, label=label(r))
    for x, t in zip(ax, ("running training loss (during each epoch)", "fresh full-training-set loss (selection)",
                         "held-out loss (never used for fitting or selection)")):
        x.axvspan(0.5, 8.5, color="0.92"), x.set_title(t), x.set_xlabel("epoch")
    ax[2].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(os.path.join(out, "losses.png"), dpi=90), plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    for r in R:
        t = r["probe"]["trajectory"]
        ep = [x["epoch"] for x in t]
        for j, k in enumerate(("C10", "C12a", "C12b")):
            ax[j].plot(ep, [x[k] for x in t], color=col[r["run"]], label=label(r))
        ax[3].plot(ep, [x["rms_rad"] for x in t], color=col[r["run"]], label=label(r))
    for j, k in enumerate(("C10", "C12a", "C12b")):
        ax[j].axhline(tr.true[1][j], color="k", ls="--", lw=1, label="truth")
        ax[j].axhline(tr.nominal[1][j], color="0.5", ls=":", lw=1, label="nominal")
        ax[j].set_title(f"{k} (A)"), ax[j].set_xlabel("epoch"), ax[j].axvspan(0.5, 8.5, color="0.92")
    ax[3].set_title("probe RMS phase error vs truth (rad)"), ax[3].set_xlabel("epoch")
    ax[0].legend(fontsize=7)
    fig.suptitle("probe coefficients per epoch (grey band: S-only warm-up); joint mode is experimental", fontsize=10)
    fig.tight_layout(), fig.savefig(os.path.join(out, "probe.png"), dpi=90), plt.close(fig)

    n = len(R)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for i, r in enumerate(R):
        e = aperture_phase_error(r["_psi"], r["_ptrue"], r["_ky"], r["_kx"])
        sc = ax[0, i].scatter(r["_kx"], r["_ky"], c=e, s=2, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax[0, i].set_aspect("equal"), ax[0, i].axis("off")
        ax[0, i].set_title(f"{label(r)}\nRMS {r['probe']['rms_rad']:.3f} rad", fontsize=8)
    fig.colorbar(sc, ax=ax[0, -1], fraction=0.05)
    fig.suptitle("final probe aperture phase − truth (rad, piston/tilt removed)", fontsize=10)
    fig.tight_layout(), fig.savefig(os.path.join(out, "probe_phase.png"), dpi=90), plt.close(fig)

    show = [r for r in R if r["seed"] == 0 and r["group"] in ("fixed", "joint")]
    fig, ax = plt.subplots(2, len(show) + 1, figsize=(4.2 * (len(show) + 1), 8.2), squeeze=False)
    t = show[0]["_proj"]
    ax[0, 0].imshow(t, cmap="gray"), ax[0, 0].set_title("truth σ·ΣV (rad)")
    ax[1, 0].axis("off")
    for i, r in enumerate(show, 1):
        p = r["projected"]
        ax[0, i].imshow(r["_est"], cmap="gray")
        ax[0, i].set_title(f"{label(r)}: corr {p['corr']:.3f}, slope {p['slope']:.3f}", fontsize=8)
        ax[1, i].plot(t.ravel()[::7], r["_est"].ravel()[::7], ",", alpha=0.4)
        xx = np.linspace(t.min(), t.max(), 2)
        ax[1, i].plot(xx, p["slope"] * xx + p["offset"], "r-")
        ax[1, i].set_xlabel("truth (rad)"), ax[1, i].set_ylabel("estimate (rad)")
    for x in ax[0]:
        x.axis("off")
    fig.suptitle("projected-potential estimate (unwrapped phase of the S refocused to T/2) vs truth", fontsize=10)
    fig.tight_layout(), fig.savefig(os.path.join(out, "projected.png"), dpi=90), plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    zt, pt_ = R[0]["_prof"]
    ax[0].plot(zt, pt_ / pt_.max(), "k-", lw=2, label="truth (lateral variance)")
    for r in R:
        inside = (r["_zg"] >= 0) & (r["_zg"] <= tr.T)
        c = r["_curve"][inside]
        ax[0].plot(r["_zg"][inside], c / c.max(), color=col[r["run"]], label=label(r))
        ax[1].plot(SEC, [x["corr_slab"] - x["corr_proj"] for x in r["depth"]["section_corr"]], "o-", ms=3,
                   color=col[r["run"]], label=label(r))
    ax[0].set_title("focus curve (phase variance) vs true depth profile"), ax[0].set_xlabel("z (A)")
    ax[0].legend(fontsize=7)
    ax[1].axhline(0, color="k", lw=0.8), ax[1].set_xlabel("z (A)")
    ax[1].set_title("section corr vs own ±10 Å slab − corr vs projection")
    for i, r in enumerate(R):
        z = np.array([c["z"] for c in r["_cands"]])
        u = np.array([c["uncertainty_A"] for c in r["_cands"]])
        ax[2].errorbar(z, i + np.linspace(-0.3, 0.3, len(z)), xerr=u, fmt=".", ms=3, lw=0.4, color=col[r["run"]])
    ax[2].set_yticks(range(len(R)), [label(r) for r in R], fontsize=7)
    ax[2].axvspan(0, tr.T, color="0.95"), ax[2].set_xlabel("candidate depth ± uncertainty (A)")
    ax[2].set_title("depth candidates (truth: columnar grains, no depth structure)")
    fig.tight_layout(), fig.savefig(os.path.join(out, "depth.png"), dpi=90), plt.close(fig)


if __name__ == "__main__":
    main()
