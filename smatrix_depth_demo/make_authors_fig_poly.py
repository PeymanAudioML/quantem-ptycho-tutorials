"""Figure: authors' unmodified GradDS code vs my code on the polycrystal."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); CROP = np.s_[48:148, 48:148]
A = np.load(os.path.join(H, "results_authors", "authors_full_sections_poly.npz")); M = np.load(os.path.join(H, "results_poly", "sections.npz"))
L = np.load(os.path.join(H, "results_poly", "layers_and_data.npz"))["layers"]; cfg = json.load(open(os.path.join(H, "results_poly", "config.json")))
depths = A["depths"]; pa = A["phase"]; pm = np.angle(M["ew"])[(slice(None),) + CROP]
zc = (np.arange(L.shape[0]) + 0.5) * cfg["dz"]; slab = [L[np.abs(zc - z) <= 20].sum(0)[CROP] for z in depths]
corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
fig = plt.figure(figsize=(11, 7.2)); gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.4])
for j, z in enumerate([25, 75, 125]):
    i = int(np.argmin(abs(depths - z)))
    for r, (lab, im) in enumerate([(f"true slab z={z}±20", slab[i]), (f"authors' code, z={depths[i]:.0f}", pa[i]), (f"my code, z={depths[i]:.0f}", pm[i])]):
        a = fig.add_subplot(gs[r, j]); a.imshow(im if r == 0 else im - np.median(im), cmap="magma" if r == 0 else "gray"); a.set_xticks([]); a.set_yticks([])
        a.set_title(lab + ("" if r == 0 else f"\nr(truth)={corr(im, slab[i]):+.2f}"), fontsize=8)
ax = fig.add_subplot(gs[:, 3])
ax.plot(depths, [corr(pa[i], slab[i]) for i in range(len(depths))], "o-", color="tab:blue", label="authors' code")
ax.plot(depths, [corr(pm[i], slab[i]) for i in range(len(depths))], "k--", label="my code")
ax.plot(depths, [corr(pa[i], pm[i]) for i in range(len(depths))], ":", color="tab:green", label="authors' vs mine (section-to-section)")
ax.set_xlabel("section depth z (Å)"); ax.set_ylabel("correlation"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
ax.set_title("correlation with the true slab at the same depth", fontsize=9)
fig.suptitle("Polycrystal CaAl₂Si₂O₈: authors' unmodified GradDS (mu=1, 10 it) vs my NumPy code (30 it)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(H, "results_authors", "fig_authors_vs_mine_poly.png"), dpi=130)
