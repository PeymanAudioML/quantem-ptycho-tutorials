"""Tables and figures for the joint S-matrix / probe experiments (reads results_joint/<sample>/<config>/*.json)."""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_aberrations as pa
import joint_smatrix_probe as jp
from run_joint_probe import sample, NAMES

OUT = os.path.join(HERE, "results_joint")
FIG = os.environ.get("JOINT_FIG_DIR", os.path.join(OUT, "figures")); os.makedirs(FIG, exist_ok=True)
VAR = ["fixed", "joint", "oracle"]
COL = {"fixed": "#9aa5b1", "joint": "#1c4587", "oracle": "#e69138"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})


def load_all():
    runs = []
    for f in sorted(glob.glob(os.path.join(OUT, "*", "*", "*_seed*.json"))):
        r = json.load(open(f)); r["_file"] = f; r["_group"] = os.path.relpath(os.path.dirname(f), OUT)
        runs.append(r)
    return runs


def scalar_metrics(r):
    ev = r["eval"]; dm = ev["depth"]
    est, true, nom = np.array(ev["coeff_est"]), np.array(ev["coeff_true"]), np.array(r["nominal"])
    tr = [NAMES.index(n) for n in r["trainable"]] if r["variant"] == "joint" else [NAMES.index(n) for n in ("C10", "C12a", "C12b")]
    inj = np.linalg.norm((true - nom)[1:, tr])
    res = np.linalg.norm((est - true)[1:, tr])
    out = dict(final_loss=r["summary"]["final_loss"], probe_err=float(np.mean(ev["probe_wavefunction_error"][1:])),
               coeff_resid=float(res), coeff_injected=float(inj),
               frac_recovered=(1 - res / inj) if inj > 0 else float("nan"),
               dC10_d1=float(est[1, 0] - true[1, 0]), dC10_d2=float(est[2, 0] - true[2, 0]),
               S_err=ev["S_error"]["nrmse"], runtime=r["summary"]["runtime_s"], peak_mb=r["summary"]["peak_memory_mb"])
    if "mean_abs_depth_err" in dm:
        # FWHM recomputed from the stored axial curves with the corrected metric (contiguous main peak, interpolated
        # half-maximum crossings); runs made before the fix stored the first/last-above-half width on the 10 A grid
        C = np.abs(np.array(dm["curves"])); z = np.arange(C.shape[0]) * 10.0
        w = [jp.axial_fwhm(z, C[:, j]) for j in range(C.shape[1])]
        out.update(depth_err=dm["mean_abs_depth_err"], mean_r=dm["mean_r"], fwhm=float(np.mean([x[0] for x in w])),
                   fwhm_open=float(sum(x[1] for x in w)), crosstalk=dm["mean_crosstalk"], resolved=dm["n_resolved"])
    else:
        out.update(depth_err=dm["mean_depth_err"], mean_r=dm["mean_diag_r"], selectivity=dm["selectivity"])
    return out


def table(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["_group"], r["variant"]), []).append(scalar_metrics(r))
    rows = []
    for (grp, v), ms in sorted(groups.items(), key=lambda t: (t[0][0], VAR.index(t[0][1]) if t[0][1] in VAR else 9)):
        keys = ms[0].keys()
        agg = {k: (float(np.nanmean([m[k] for m in ms])), float(np.nanstd([m[k] for m in ms]))) for k in keys}
        rows.append(dict(group=grp, variant=v, n=len(ms), **agg))
    return rows


def fmt(ms, k, p=1):
    if k not in ms:
        return "–"
    m, s = ms[k]
    return f"{m:.{p}f}" if ms["n"] == 1 else f"{m:.{p}f} ± {s:.{p}f}"


def write_tables(rows):
    if os.environ.get("JOINT_FIG_DIR"):
        return
    lines = ["| group | variant | n | final loss | depth err (Å) | mean r | axial FWHM (Å) ¹ | cross-talk | probe err (d1,d2 mean) | "
             "coef. residual / injected (Å) | fraction recovered | ΔC10 d1 / d2 (Å) | S NRMSE | runtime (s) | peak MB |",
             "|" + "---|" * 16]
    for r in rows:
        lines.append(f"| {r['group']} | {r['variant']} | {r['n']} | {fmt(r,'final_loss',5)} | {fmt(r,'depth_err')} | {fmt(r,'mean_r',2)} | "
                     f"{fmt(r,'fwhm',0)} | {fmt(r,'crosstalk',2)} | {fmt(r,'probe_err',3)} | {fmt(r,'coeff_resid')} / {fmt(r,'coeff_injected')} | "
                     f"{fmt(r,'frac_recovered',2)} | {fmt(r,'dC10_d1')} / {fmt(r,'dC10_d2')} | {fmt(r,'S_err',3)} | {fmt(r,'runtime',0)} | {fmt(r,'peak_mb',0)} |")
    lines.append("\n¹ mean over layers of the contiguous main-peak FWHM with interpolated half-maximum crossings (10 Å "
                 "section grid); for layers whose response reaches the end of the 0–thickness depth range (e.g. 30 Å and "
                 "270 Å in a 300 Å slab) the width is a lower bound.")
    open(os.path.join(OUT, "summary_table.md"), "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(OUT, "summary_table.json"), "w"), indent=1)
    print("\n".join(lines))


def pick(runs, group, variant, seed=None):
    c = sorted([r for r in runs if r["_group"] == group and r["variant"] == variant], key=lambda r: r["seed"])
    if seed == 0 and c:            # "seed 0" panels: the first available noise realisation
        seed = c[0]["seed"]
    return [r for r in c if seed is None or r["seed"] == seed]


def first_seed(runs, group):
    return min([r["seed"] for r in runs if r["_group"] == group] or [0])


def fig_loss_and_coeffs(runs, group):
    rs = {v: pick(runs, group, v, 0) for v in VAR}
    if not rs["joint"]:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for v in VAR:
        if rs[v]:
            h = rs[v][0]["history"]
            ax[0].semilogy(h["s_iter"], h["s_loss"], "-", color=COL[v], label=v)
    ax[0].set_xlabel("S iteration"); ax[0].set_ylabel("relative amplitude loss (Eq. 12)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("training loss (seed 0)")
    nom = np.array(rs["joint"][0]["nominal"]); true = np.array(rs["joint"][0]["eval"]["coeff_true"])
    for r in pick(runs, group, "joint"):
        its = [c[0] for c in r["history"]["coeffs"]]; C = np.array([c[1] for c in r["history"]["coeffs"]])
        for d, ls in ((1, "-"), (2, "--")):
            ax[1].plot(its, C[:, d, 0] - nom[d, 0], ls, color="#1c4587", alpha=0.5, lw=1)
            for j, col in ((1, "#e69138"), (2, "#6aa84f")):
                ax[2].plot(its, C[:, d, j] - nom[d, j], ls, color=col, alpha=0.5, lw=1)
    for d, ls in ((1, "-"), (2, "--")):
        ax[1].axhline(true[d, 0] - nom[d, 0], color="k", ls=ls, lw=0.8)
        for j, col in ((1, "#e69138"), (2, "#6aa84f")):
            ax[2].axhline(true[d, j] - nom[d, j], color=col, ls=ls, lw=0.8)
    ax[1].set_title("ΔC10 vs iteration (solid d1, dashed d2; black = truth)"); ax[1].set_xlabel("S iteration"); ax[1].set_ylabel("Å")
    ax[2].set_title("ΔC12a (orange), ΔC12b (green); thin = truth"); ax[2].set_xlabel("S iteration"); ax[2].set_ylabel("Å")
    for a in ax[1:]: a.grid(alpha=0.3)
    fig.suptitle(f"{group}: loss and probe-aberration trajectories (all seeds for the joint runs)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"loss_coeffs_{group.replace('/', '_')}.png")); plt.close(fig)


def fig_coeff_bars(runs, groups):
    groups = [g for g in groups if pick(runs, g, "joint")]
    if not groups:
        return
    labs = ["d1 C10", "d1 C12a", "d1 C12b", "d2 C10", "d2 C12a", "d2 C12b"]
    idx = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    nc = min(3, len(groups)); nr = math.ceil(len(groups) / nc)
    fig, ax = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.4 * nr), squeeze=False)
    for a in ax.ravel()[len(groups):]:
        a.axis("off")
    for a, g in zip(ax.ravel(), groups):
        rj = pick(runs, g, "joint"); nom = np.array(rj[0]["nominal"]); true = np.array(rj[0]["eval"]["coeff_true"])
        est = np.array([np.array(r["eval"]["coeff_est"]) for r in rj])
        x = np.arange(len(idx))
        a.bar(x - 0.2, [true[d, j] - nom[d, j] for d, j in idx], 0.4, color="k", label="injected (truth)")
        a.bar(x + 0.2, [np.mean(est[:, d, j] - nom[d, j]) for d, j in idx], 0.4, color=COL["joint"],
              yerr=[np.std(est[:, d, j]) for d, j in idx], capsize=2, label="joint (mean ± std)")
        a.set_xticks(x); a.set_xticklabels(labs, rotation=45, fontsize=7); a.set_title(f"{g} (n={len(rj)})", fontsize=9); a.grid(axis="y", alpha=0.3)
        a.axhline(0, color="k", lw=0.5)
    ax[0, 0].set_ylabel("relative aberration (Å)"); ax[0, 0].legend(fontsize=7)
    fig.suptitle("Ground-truth vs recovered relative probe aberrations (reference dataset d0 fixed)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "aberrations_truth_vs_recovered.png")); plt.close(fig)


def fig_probe(runs, group):
    rj = pick(runs, group, "joint", 0)
    if not rj:
        return
    r = rj[0]; s = sample(r["sample"]); g = s["geom"]
    nom, true, est = np.array(r["nominal"]), np.array(r["eval"]["coeff_true"]), np.array(r["eval"]["coeff_est"])
    n = 101; k = np.linspace(-1, 1, n) * g.alpha / g.lam
    KY, KX = np.meshgrid(k, k, indexing="ij"); inside = KY ** 2 + KX ** 2 <= (g.alpha / g.lam) ** 2
    chi = lambda C: sum(C[j] * pa.basis_function(nm, KY, KX, g.lam) for j, nm in enumerate(NAMES))
    fig, ax = plt.subplots(2, 4, figsize=(12, 5.6))
    for row, d in enumerate((1, 2)):
        panels = [("injected error χ_true − χ_nominal", chi(true[d]) - chi(nom[d])),
                  ("recovered χ_joint − χ_nominal", chi(est[d]) - chi(nom[d])),
                  ("residual χ_joint − χ_true", chi(est[d]) - chi(true[d]))]
        vm = np.abs(panels[0][1][inside]).max()
        for c, (t, im) in enumerate(panels):
            a = ax[row, c]; im = np.where(inside, im, np.nan)
            h = a.imshow(im, cmap="RdBu_r", vmin=-vm, vmax=vm, extent=[-g.alpha * 1e3, g.alpha * 1e3] * 2); a.set_title(f"d{d}: {t}", fontsize=8)
            a.set_xlabel("θx (mrad)")
        fig.colorbar(h, ax=ax[row, 2], label="phase (rad)", shrink=0.8)
        # real-space probe at the entrance surface (true vs joint)
        A_t = pa.probe_on_grid(list(true[d]), NAMES, 128, 128, g.dr, g.lam, g.alpha)
        A_j = pa.probe_on_grid(list(est[d]), NAMES, 128, 128, g.dr, g.lam, g.alpha)
        p_t, p_j = np.fft.fftshift(np.fft.ifft2(A_t)), np.fft.fftshift(np.fft.ifft2(A_j))
        cc = slice(64 - 30, 64 + 30)
        a = ax[row, 3]; a.plot((np.arange(60) - 30) * g.dr, np.abs(p_t[64, cc]) / np.abs(p_t).max(), "k-", label="|probe| true")
        a.plot((np.arange(60) - 30) * g.dr, np.abs(p_j[64, cc]) / np.abs(p_t).max(), "--", color=COL["joint"], label="|probe| joint")
        a2 = a.twinx(); pd = np.angle(p_j[64, cc] * np.conj(p_t[64, cc])); pd[np.abs(p_t[64, cc]) < 0.1 * np.abs(p_t).max()] = np.nan
        a2.plot((np.arange(60) - 30) * g.dr, pd, ":", color="#cc0000")
        a2.set_ylabel("phase diff. (rad)", color="#cc0000"); a.set_xlabel("x (Å)"); a.legend(fontsize=6); a.set_title(f"d{d}: entrance-plane probe profile", fontsize=8)
    fig.suptitle(f"{group}, seed 0: probe aberration phase on the aperture and real-space probe (reference d0 fixed)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"probe_{group.replace('/', '_')}.png")); plt.close(fig)


def fig_smatrix(runs, group):
    have = {v: os.path.join(OUT, group, f"{v}_seed{first_seed(runs, group)}_beams.npz") for v in VAR}
    if not all(os.path.exists(f) for f in have.values()):
        return False
    D = {v: np.load(f) for v, f in have.items()}
    r = pick(runs, group, "joint", 0)[0]; g = sample(r["sample"])["geom"]; cr = g.crop()
    yy = (np.arange(cr[0].start, cr[0].stop) * g.dr)[:, None]; xx = (np.arange(cr[1].start, cr[1].stop) * g.dr)[None, :]
    by, bx = D["joint"]["by"], D["joint"]["bx"]
    fig, ax = plt.subplots(len(by), 4, figsize=(9.5, 2.3 * len(by)))
    for i in range(len(by)):
        carrier = np.exp(-2j * np.pi * (by[i] / g.LW * yy + bx[i] / g.LW * xx))
        St = D["joint"]["S_true"][i] * carrier
        for c, (lab, Sx) in enumerate([("truth", D["joint"]["S_true"][i])] + [(v, D[v]["S_rec"][i]) for v in VAR]):
            c_ = np.vdot(Sx, D["joint"]["S_true"][i]); Sx = Sx * (c_ / abs(c_))          # per-beam phase gauge
            im = np.angle(Sx * carrier * np.exp(-1j * np.angle(np.mean(St))))
            ax[i, c].imshow(im, cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
            if c > 0:
                e = np.linalg.norm(Sx - D["joint"]["S_true"][i]) / np.linalg.norm(D["joint"]["S_true"][i])
                ax[i, c].set_title(f"{lab}, beam ({by[i]},{bx[i]})\nNRMSE {e:.2f}", fontsize=7)
            else:
                ax[i, c].set_title(f"truth, beam ({by[i]},{bx[i]})", fontsize=7)
    fig.suptitle(f"{group}, seed 0: S-matrix beam phase (carrier removed, per-beam phase gauge aligned)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"smatrix_{group.replace('/', '_')}.png")); plt.close(fig)
    return True


def fig_sections(runs, group, zs=None):
    r = pick(runs, group, "joint", 0) or pick(runs, group, "fixed", 0)
    if not r:
        return
    r = r[0]; s = sample(r["sample"]); g = s["geom"]; cr = g.crop()
    if s["kind"] == "layers":
        zs = s["layer_z"]; truth = [L[cr] for L in s["layers"]]
    else:
        zs = zs or [25.0, 75.0, 125.0]; zc = (np.arange(s["slices"].shape[0]) + 0.5) * s["dz"]
        truth = [s["slices"][np.abs(zc - z) <= 20].sum(0)[cr] for z in zs]
    vs = [v for v in VAR if os.path.exists(os.path.join(OUT, group, f"{v}_seed{first_seed(runs, group)}_sections.npz"))]
    fig, ax = plt.subplots(1 + len(vs), len(zs), figsize=(2.6 * len(zs), 2.5 * (1 + len(vs))), squeeze=False)
    for j, (z, t) in enumerate(zip(zs, truth)):
        ax[0, j].imshow(t, cmap="magma"); ax[0, j].set_title(f"truth z={z:.0f} Å", fontsize=8)
        for i, v in enumerate(vs, start=1):
            d = np.load(os.path.join(OUT, group, f"{v}_seed{first_seed(runs, group)}_sections.npz")); iz = int(np.argmin(np.abs(d["depths"] - z)))
            im = d["phase"][iz]; ax[i, j].imshow(im - np.median(im), cmap="gray")
            ax[i, j].set_title(f"{v}: z={d['depths'][iz]:.0f}, r={jp.corr(im, t):+.2f}", fontsize=8)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{group}, seed 0: depth sections at the true depths (fixed = nominal miscalibrated probe, i.e. before probe refinement; joint = after)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"sections_{group.replace('/', '_')}.png")); plt.close(fig)


def fig_axial(runs, group):
    rs = {v: pick(runs, group, v, 0) for v in VAR}
    if not rs["fixed"] or "curves" not in rs["fixed"][0]["eval"]["depth"]:
        return
    s = sample(rs["fixed"][0]["sample"]); depths = np.arange(0.0, s["thick"] + 1, 10.0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for a, v in zip(ax, VAR):
        if not rs[v]:
            continue
        C = np.array(rs[v][0]["eval"]["depth"]["curves"])
        for j, z in enumerate(s["layer_z"]):
            a.plot(depths, C[:, j], color=plt.cm.tab10(j), label=f"layer z={z:.0f}"); a.axvline(z, color=plt.cm.tab10(j), ls="--", alpha=0.4)
        dm = rs[v][0]["eval"]["depth"]
        fw = np.mean([jp.axial_fwhm(depths, np.abs(C[:, j]))[0] for j in range(C.shape[1])])
        a.set_title(f"{v}: err {dm['mean_abs_depth_err']:.1f} Å, FWHM {fw:.0f} Å, cross-talk {dm['mean_crosstalk']:.2f}", fontsize=8)
        a.set_xlabel("section depth z (Å)"); a.grid(alpha=0.3)
    ax[0].set_ylabel("correlation with true layer"); ax[0].legend(fontsize=7)
    fig.suptitle(f"{group}, seed 0: axial response (correlation of each depth section with each true layer)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"axial_{group.replace('/', '_')}.png")); plt.close(fig)


def fig_depth_bars(rows):
    groups = sorted({r["group"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, (k, t) in zip(ax, [("depth_err", "mean depth error (Å) ↓"), ("probe_err", "probe-wavefunction error (d1,d2) ↓"),
                              ("final_loss", "final relative amplitude loss ↓")]):
        for i, v in enumerate(VAR):
            xs, ms, ss = [], [], []
            for j, gname in enumerate(groups):
                rr = [r for r in rows if r["group"] == gname and r["variant"] == v]
                if rr and k in rr[0]:
                    xs.append(j + (i - 1) * 0.27); ms.append(rr[0][k][0]); ss.append(rr[0][k][1])
            a.bar(xs, ms, 0.27, yerr=ss, color=COL[v], capsize=2, label=v)
        a.set_xticks(range(len(groups))); a.set_xticklabels(groups, rotation=35, fontsize=7, ha="right"); a.set_title(t, fontsize=9)
        a.grid(axis="y", alpha=0.3)
    ax[2].set_yscale("log"); ax[0].legend(fontsize=7)
    fig.suptitle("Fixed (miscalibrated) vs joint vs oracle probe: mean ± std over noise realisations where n > 1", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "depth_probe_summary.png")); plt.close(fig)


def fig_dose(rows):
    doses = [(2e5, "reduced_s2/B"), (2e4, "reduced_s2/B_dose2e4"), (2e3, "reduced_s2/B_dose2e3")]
    if not any(r["group"] == doses[-1][1] for r in rows):
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (k, t) in zip(ax, [("depth_err", "mean depth error (Å)"), ("frac_recovered", "fraction of injected aberration recovered"),
                              ("probe_err", "probe-wavefunction error (d1,d2)")]):
        for v in VAR:
            pts = [(d, r[k]) for d, gname in doses for r in rows if r["group"] == gname and r["variant"] == v and k in r]
            if pts:
                a.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], marker="o", color=COL[v], label=v, capsize=2)
        a.set_xscale("log"); a.set_xlabel("dose (e⁻ / pattern)"); a.set_title(t, fontsize=9); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("reduced_s2, config B: dose sensitivity (mean ± std over noise seeds)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dose_sensitivity.png")); plt.close(fig)


if __name__ == "__main__":
    runs = [r for r in load_all() if (os.environ.get("JOINT_INCLUDE_TUNE") or "_tune" not in r["_group"])
            and not r["_group"].startswith("sep")]          # two-layer benchmark: run_separation.py analyze
    rows = table(runs)
    write_tables(rows)
    fig_depth_bars(rows)
    fig_dose(rows)
    groups = sorted({r["_group"] for r in runs})
    fig_coeff_bars(runs, [g for g in groups if g.endswith("/B")])
    for g in groups:
        fig_loss_and_coeffs(runs, g); fig_probe(runs, g); fig_sections(runs, g); fig_axial(runs, g); fig_smatrix(runs, g)
    print("figures in", FIG)
