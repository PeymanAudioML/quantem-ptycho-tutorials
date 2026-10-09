"""Figure: authors' unmodified GradDS code vs my re-implementation on the same 3-layer data."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); CROP = np.s_[48:148, 48:148]
A = np.load(os.path.join(H, "results_authors", "authors_full_sections.npz")); M = np.load(os.path.join(H, "results", "sections.npz"))
L = np.load(os.path.join(H, "results", "layers_and_data.npz"))["layers"]
depths = A["depths"]; pa = A["phase"]; pm = np.angle(M["ew"])[(slice(None),) + CROP]
corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
names, zs = ["ring", "square", "triangle"], [30, 150, 270]
fig = plt.figure(figsize=(11, 7.2)); gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.5])
for j, (n, z) in enumerate(zip(names, zs)):
    i = int(np.argmin(abs(depths - z)))
    for r, (lab, im) in enumerate([("truth", L[j][CROP]), (f"authors' code, z={depths[i]:.0f}", pa[i]), (f"my code, z={depths[i]:.0f}", pm[i])]):
        a = fig.add_subplot(gs[r, j]); a.imshow(im - np.median(im) if r else im, cmap="magma" if r == 0 else "gray"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{n}: {lab}" + ("" if r == 0 else f"\nr(truth)={corr(im, L[j][CROP]):+.2f}"), fontsize=8)
ax = fig.add_subplot(gs[:, 3])
for j, (n, c) in enumerate(zip(names, ["tab:red", "tab:green", "tab:orange"])):
    ax.plot(depths, [corr(pa[i], L[j][CROP]) for i in range(len(depths))], "-", color=c, lw=2, label=f"{n} (authors)")
    ax.plot(depths, [corr(pm[i], L[j][CROP]) for i in range(len(depths))], "--", color="k", lw=1)
ax.plot([], [], "k--", lw=1, label="my code (all layers)")
ax.set_xlabel("section depth z (Å)"); ax.set_ylabel("correlation with true layer"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
ax.set_title("authors' code (colour) vs my code (dashed)", fontsize=9)
fig.suptitle("Authors' unmodified GradDS (mu=1, 10 it) vs my NumPy re-implementation (25 it), same 3-layer data", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(H, "results_authors", "fig_authors_vs_mine.png"), dpi=130)
