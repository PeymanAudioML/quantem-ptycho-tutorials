import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(HERE, "results_quantem")
S = json.load(open(os.path.join(Q, "summary.json")))
CROP = np.s_[48:148, 48:148]
M = [("tcBF mine (3 df)", "tcBF (mine, 3 df)"), ("tcBF quantem (df=0)", "tcBF quantem (df=0)"),
     ("ACBF quantem (df=0)", "ACBF quantem (df=0)"), ("ACBF quantem (3 df averaged)", "ACBF quantem (3 df)"),
     ("S-matrix + refocusing", "S-matrix")]
cols = ["#9aa5b1", "#6fa8dc", "#3d85c6", "#1c4587", "#e69138"]
cfgs = ["baseline_3layers", "six_phi1p5", "poly"]
def get(c, k):
    r = S[c]["results"][k]
    if c == "poly": return r["mean_depth_err_A"], r["mean_diag_r"], r["frac_within_10A"]
    return r["mean_abs_depth_err"], r["mean_abs_r"], r["n_resolved"] / r["n_layers"]
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
for a, (idx, ttl) in zip(ax, [(0, "mean depth error (Å) ↓"), (1, "mean correlation with truth ↑"), (2, "fraction of layers/depths resolved ↑")]):
    for i, (k, lab) in enumerate(M):
        a.bar(np.arange(3) + (i - 2) * 0.16, [get(c, k)[idx] for c in cfgs], 0.16, color=cols[i], label=lab)
    a.set_xticks(range(3)); a.set_xticklabels(["3 layers", "6 overlapping layers", "polycrystal"], fontsize=8); a.set_title(ttl, fontsize=9); a.grid(axis="y", alpha=0.3)
ax[0].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(Q, "fig_q1_summary.png")); plt.close(fig)

def img_fig(cfg, rdir, picks, titles):
    st = np.load(os.path.join(Q, f"stacks_{cfg}.npz")); d = np.load(os.path.join(HERE, rdir, "layers_and_data.npz"))["layers"]
    depths = st["depths"]; truth = d[(slice(None),) + CROP]
    keys = {k: k.replace(" ", "_") for k, _ in M}
    fig, ax = plt.subplots(1 + len(M), len(picks), figsize=(2.3 * len(picks), 2.3 * (1 + len(M))), squeeze=False)
    for j, (tr, ttl) in enumerate(zip(picks, titles)):
        ax[0, j].imshow(tr, cmap="magma"); ax[0, j].set_title(ttl, fontsize=8)
        for r, (k, lab) in enumerate(M, start=1):
            stack = st[keys[k]]
            cc = [np.corrcoef(stack[i].ravel(), tr.ravel())[0, 1] for i in range(len(depths))]
            cc = list(np.nan_to_num(cc)); ib = int(np.argmax(np.abs(cc)))
            ax[r, j].imshow(stack[ib] - np.median(stack[ib]), cmap="gray"); ax[r, j].set_title(f"{lab}\nz={depths[ib]:.0f} r={cc[ib]:+.2f}", fontsize=6.5)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(Q, f"fig_q_images_{cfg}.png")); plt.close(fig)
img_fig("six_phi1p5", "results_six_phi1p5", [truth for truth in []] or [np.load(os.path.join(HERE, "results_six_phi1p5", "layers_and_data.npz"))["layers"][i][CROP] for i in (0, 2, 5)],
        ["ring (25 Å)", "triangle (125 Å)", "diamond (275 Å)"])
dz = 2.5
L = np.load(os.path.join(HERE, "results_poly", "layers_and_data.npz"))["layers"]; zc = (np.arange(L.shape[0]) + 0.5) * dz
img_fig("poly", "results_poly", [L[np.abs(zc - z) <= 20].sum(0)[CROP] for z in (25, 75, 125)], ["slab z=25", "slab z=75", "slab z=125"])
