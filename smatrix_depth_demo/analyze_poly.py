"""Score tcBF vs S-matrix refocusing on the polycrystal against the true depth-resolved structure."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import smatrix_depth_demo as m
import sweep_compare as sc

R = os.path.join(m.HERE, "results_poly")
cfg = json.load(open(os.path.join(R, "config.json")))
DZ, SLAB = cfg["dz"], 40.0
depths = np.arange(0.0, cfg["thickness"] + 1, 10.0)
sc.depths = depths
sc.zf_all = [-df for df in cfg["defoci"]]
d = np.load(os.path.join(R, "layers_and_data.npz")); s = np.load(os.path.join(R, "sections.npz"))
slices, dps = d["layers"], d["dps"]
zc = (np.arange(slices.shape[0]) + 0.5) * DZ
ext_c, ext = sc.ext_c, sc.ext

def blur(img, sig_A):
    q2 = (np.fft.fftfreq(m.Y, m.DR)[:, None] ** 2 + np.fft.fftfreq(m.X, m.DR)[None, :] ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * np.pi ** 2 * sig_A ** 2 * q2)))

slab = np.array([slices[np.abs(zc - z) <= SLAB / 2].sum(0) for z in depths])            # true depth-resolved structure
slab_b = np.array([blur(x, 0.8) for x in slab])                                          # limited to ~1.6 A scan sampling

stacks = {"tcBF (df=0 only)": sc.tcbf_sections(dps, [2]),
          "tcBF (3 defocus)": sc.tcbf_sections(dps, [0, 1, 2]),
          "S-matrix + refocusing": np.angle(s["ew"])}
nD = len(depths)

def Rmat(stack, truth):
    M = np.zeros((nD, nD))
    for i in range(nD):
        p = stack[i][ext_c, ext_c]
        for j in range(nD):
            M[i, j] = sc.corr(p, truth[j][ext_c, ext_c])
    return M

def metrics(M):
    A = np.abs(M); diag = np.diag(M)
    loc = np.array([abs(depths[np.argmax(A[i])] - depths[i]) for i in range(nD)])
    sel = []
    for i in range(nD):
        off = [A[i, j] for j in range(nD) if abs(depths[j] - depths[i]) >= 60]
        sel.append(A[i, i] - np.mean(off))
    return dict(mean_diag_r=float(diag.mean()), mean_abs_diag_r=float(np.abs(diag).mean()),
                mean_depth_err_A=float(loc.mean()), frac_within_10A=float((loc <= 10).mean()),
                selectivity=float(np.mean(sel)), n_inverted=int((diag < 0).sum()))

out, Ms = {}, {}
for k, st in stacks.items():
    Ms[k] = (Rmat(st, slab), Rmat(st, slab_b))
    out[k] = dict(full_band_truth=metrics(Ms[k][0]), scan_limited_truth=metrics(Ms[k][1]))
    for t in out[k]:
        v = out[k][t]
        print(f"{k:24s} [{t:18s}] mean r(diag) {v['mean_diag_r']:+.2f}  |depth err| {v['mean_depth_err_A']:5.1f} A  "
              f"within 10A {v['frac_within_10A']*100:3.0f}%  selectivity {v['selectivity']:+.2f}  inverted {v['n_inverted']}/{nD}")
json.dump(out, open(os.path.join(R, "poly_metrics.json"), "w"), indent=2)

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
# structure
fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.3))
ax[0].imshow(slab.sum(0)[ext_c, ext_c] if False else slices.sum(0)[ext_c, ext_c], cmap="magma", extent=ext)
ax[0].set_title("projected phase, all depths")
for a, z in zip(ax[1:], [25, 75, 125]):
    a.imshow(slab[int(z // 10)][ext_c, ext_c], cmap="magma", extent=ext); a.set_title(f"true slab around z={z} Å (±20)")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Polycrystalline CaAl₂Si₂O₈: {cfg['natoms']} atoms, 8 rotated grains, {cfg['thickness']:.0f} Å")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_p1_structure.png")); plt.close(fig)
# images
zs = [25, 75, 125]; ks = list(stacks)
fig, ax = plt.subplots(1 + len(ks), 3, figsize=(8.6, 11.5))
for j, z in enumerate(zs):
    iz = z // 10
    ax[0, j].imshow(slab_b[iz][ext_c, ext_c], cmap="magma", extent=ext); ax[0, j].set_title(f"truth slab z={z} (blurred 0.8 Å)", fontsize=8)
    for r, k in enumerate(ks, start=1):
        p = stacks[k][iz][ext_c, ext_c]; ax[r, j].imshow(p - np.median(p), cmap="gray", extent=ext)
        ax[r, j].set_title(f"{k}\nr={Ms[k][1][iz, iz]:+.2f}", fontsize=8)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_p2_images.png")); plt.close(fig)
# matrices
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
for a, k in zip(ax, ks):
    im = a.imshow(np.abs(Ms[k][1]), origin="lower", extent=[depths[0], depths[-1], depths[0], depths[-1]], vmin=0, vmax=1, cmap="viridis")
    a.plot([0, depths[-1]], [0, depths[-1]], "w--", lw=0.8); a.set_title(k, fontsize=9)
    a.set_xlabel("true slab depth (Å)"); a.set_ylabel("section depth (Å)")
fig.colorbar(im, ax=ax, label="|correlation|", shrink=0.8)
fig.savefig(os.path.join(R, "fig_p3_depth_matrices.png"), bbox_inches="tight"); plt.close(fig)
# bars
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for a, (key, t) in zip(ax, [("mean_abs_diag_r", "mean |r| at the right depth"), ("mean_depth_err_A", "mean depth error (Å)"), ("selectivity", "depth selectivity")]):
    a.bar(range(len(ks)), [out[k]["scan_limited_truth"][key] for k in ks], color=["tab:blue", "tab:orange", "tab:green"])
    a.set_xticks(range(len(ks))); a.set_xticklabels([k.replace(" + refocusing", "") for k in ks], fontsize=7, rotation=10); a.set_title(t, fontsize=9); a.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_p4_summary.png")); plt.close(fig)
