import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, "results_paper")
S = json.load(open(os.path.join(P, "summary.json")))
CFG = [("baseline_3layers", "3 layers", "results"), ("six_phi1p5", "6 layers\n1.5 rad", "results_six_phi1p5"),
       ("six_phi2p5", "6 layers\n2.5 rad", "results_six_phi2p5"), ("poly", "polycrystal", "results_poly")]
def val(c, k):
    r = S[c][k]
    if c == "poly": return r["mean_depth_err_A"], abs(r["mean_diag_r"]) if r["mean_diag_r"] > 0 else r["mean_diag_r"], r["frac_within_10A"]
    return r["mean_abs_depth_err"], r["mean_abs_r"], r["n_resolved"] / r["n_layers"]
def single(c, kind):
    ks = [k for k in S[c] if k.startswith(f"{kind} paper (df=")]
    v = np.array([val(c, k) for k in ks]); return v.mean(0), v.min(0), v.max(0)
groups = [("tcBF (paper), single df", lambda c: single(c, "tcbf"), "#9aa5b1"), ("acBF (paper), single df", lambda c: single(c, "acbf"), "#3d85c6"),
          ("acBF (paper), 3 df summed*", lambda c: (np.array(val(c, "acbf paper (3 df summed, extension)")),) * 3, "#1c4587"),
          ("S-matrix (3 df)", lambda c: (np.array(val(c, "S-matrix + refocusing")),) * 3, "#e69138")]
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
for a, (idx, ttl) in zip(ax, [(0, "mean depth error (Å) ↓"), (1, "mean correlation with truth ↑"), (2, "fraction resolved ↑")]):
    for i, (lab, f, col) in enumerate(groups):
        xs, ms, lo, hi = [], [], [], []
        for j, (c, _, _) in enumerate(CFG):
            mean, mn, mx = f(c); xs.append(j + (i - 1.5) * 0.2); ms.append(mean[idx]); lo.append(mean[idx] - mn[idx]); hi.append(mx[idx] - mean[idx])
        a.bar(xs, ms, 0.2, color=col, label=lab, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], capsize=2, error_kw=dict(lw=0.8))
    a.set_xticks(range(4)); a.set_xticklabels([x[1] for x in CFG], fontsize=7); a.set_title(ttl, fontsize=9); a.grid(axis="y", alpha=0.3)
ax[0].legend(fontsize=6.5)
fig.text(0.01, 0.005, "Bars: mean over the 3 single-defocus datasets (whiskers = min/max). *3-df sum is my extension, uses the same total data as the S-matrix.", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(P, "fig_pp1_summary.png")); plt.close(fig)

def imgs(c, rdir, picks, titles, dfidx):
    st = np.load(os.path.join(P, f"stacks_{c}.npz")); depths = st["depths"]
    keys = [(f"tcbf paper (df=%s)" , "tcBF (paper)"), ("acbf paper (df=%s)", "acBF (paper)")]
    cfgp = os.path.join(HERE, rdir, "config.json"); defoci = json.load(open(cfgp))["defoci"] if c == "poly" else [-200.0, -100.0, 0.0]
    dfs = f"{defoci[dfidx]:.0f}".replace(" ", "_")
    rows = [(f"tcbf_paper_(df={dfs})", f"tcBF (paper), df={dfs}"), (f"acbf_paper_(df={dfs})", f"acBF (paper), df={dfs}"),
            ("acbf_paper_(3_df_summed,_extension)", "acBF 3 df summed"), ("S-matrix_+_refocusing", "S-matrix (3 df)")]
    fig, ax = plt.subplots(1 + len(rows), len(picks), figsize=(2.3 * len(picks), 2.3 * (1 + len(rows))), squeeze=False)
    for j, (tr, ttl) in enumerate(zip(picks, titles)):
        ax[0, j].imshow(tr, cmap="magma"); ax[0, j].set_title(ttl, fontsize=8)
        for r, (k, lab) in enumerate(rows, start=1):
            stack = st[k]; cc = np.nan_to_num([np.corrcoef(stack[i].ravel(), tr.ravel())[0, 1] for i in range(len(depths))])
            ib = int(np.argmax(np.abs(cc))); ax[r, j].imshow(stack[ib] - np.median(stack[ib]), cmap="gray")
            ax[r, j].set_title(f"{lab}\nz={depths[ib]:.0f} r={cc[ib]:+.2f}", fontsize=6.5)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(P, f"fig_pp_images_{c}.png")); plt.close(fig)
CROP = np.s_[48:148, 48:148]
L6 = np.load(os.path.join(HERE, "results_six_phi1p5", "layers_and_data.npz"))["layers"]
imgs("six_phi1p5", "results_six_phi1p5", [L6[i][CROP] for i in (0, 2, 5)], ["ring (25 Å)", "triangle (125 Å)", "diamond (275 Å)"], 0)
LP = np.load(os.path.join(HERE, "results_poly", "layers_and_data.npz"))["layers"]; zc = (np.arange(LP.shape[0]) + 0.5) * 2.5
imgs("poly", "results_poly", [LP[np.abs(zc - z) <= 20].sum(0)[CROP] for z in (25, 75, 125)], ["slab z=25", "slab z=75", "slab z=125"], 0)
