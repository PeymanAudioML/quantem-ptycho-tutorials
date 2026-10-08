"""Figures for the S-matrix / depth-sectioning demo (reads results/*.npz)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import smatrix_depth_demo as m

OUT = m.OUT
d = np.load(os.path.join(OUT, "layers_and_data.npz"))
s = np.load(os.path.join(OUT, "sections.npz"))
layers, dps = d["layers"], d["dps"]
depths, loss = s["depths"], s["loss"]
lz = m.LAYER_Z
names = ["ring (z=30 Å)", "square lattice (z=150 Å)", "triangle (z=270 Å)"]
ext_c = slice(m.Y // 2 - 50, m.Y // 2 + 50)
ext = [0, 100 * m.DR, 100 * m.DR, 0]
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def metrics(EW):
    ph = np.angle(EW)
    C = np.zeros((len(depths), 3))
    for i in range(len(depths)):
        p = ph[i][ext_c, ext_c]
        for j in range(3):
            C[i, j] = corr(p, layers[j][ext_c, ext_c])
    return C


EW = s["ew"]
C = metrics(EW)
phase = np.angle(EW)

# ---- fig1: sample ---------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
for j in range(3):
    ax[j].imshow(layers[j][ext_c, ext_c], cmap="magma", extent=ext)
    ax[j].set_title(names[j]); ax[j].set_xlabel("x (Å)")
ax[0].set_ylabel("y (Å)")
ax[3].set_xlim(-0.2, 1.2); ax[3].set_ylim(m.THICK + 20, -20)
ax[3].axvline(0.5, color="k", lw=0.5)
for z, n in zip(lz, ["ring", "square", "triangle"]):
    ax[3].axhspan(z - 3, z + 3, xmin=0.2, xmax=0.8, color="tab:blue", alpha=0.7)
    ax[3].text(0.5, z - 8, n, ha="center", fontsize=9)
for df in m.DEFOCI:
    ax[3].plot([0.0], [-df], "v", color="tab:red")
    ax[3].text(0.05, -df + 4, f"df={df:.0f}", color="tab:red", fontsize=8)
ax[3].set_ylabel("depth z (Å)"); ax[3].set_xticks([]); ax[3].set_title("geometry (▼ probe foci)")
fig.suptitle("Test sample: three phase layers (300 kV, 20 mrad, 300 Å thick)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_sample.png")); plt.close(fig)

# ---- fig2: 4D-STEM data ---------------------------------------------------
c = m.NSCAN // 2
fig, ax = plt.subplots(2, 3, figsize=(10, 6.4))
for i, df in enumerate(m.DEFOCI):
    dp = np.fft.fftshift(dps[i, c, c])
    ax[0, i].imshow(np.log10(dp + 1e-5 * dp.max()), cmap="viridis")
    ax[0, i].set_title(f"diffraction pattern, df={df:.0f} Å"); ax[0, i].axis("off")
    pac = np.fft.fftshift(dps[i].sum((0, 1)))
    ax[1, i].imshow(np.log10(pac + 1e-5 * pac.max()), cmap="viridis")
    ax[1, i].set_title(f"PACBED, df={df:.0f} Å"); ax[1, i].axis("off")
fig.suptitle(f"Simulated 4D-STEM data: {m.NSCAN}×{m.NSCAN} scan × {m.K}×{m.K} detector × {len(m.DEFOCI)} defoci, {m.DOSE:g} e⁻/pattern")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_4dstem_data.png")); plt.close(fig)

# ---- fig3: loss -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.semilogy(np.arange(1, len(loss) + 1), loss, "o-")
ax.set_xlabel("amplitude-flow iteration"); ax.set_ylabel("relative amplitude loss")
ax.set_title("S-matrix recovery convergence"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_loss.png")); plt.close(fig)

# ---- fig4: depth montage --------------------------------------------------
sel = list(range(0, len(depths), 3))
fig, ax = plt.subplots(2, 6, figsize=(14, 5.2))
for a, i in zip(ax.ravel(), sel):
    p = phase[i][ext_c, ext_c]; p = p - np.median(p)
    a.imshow(p, cmap="gray", extent=ext)
    a.set_title(f"z = {depths[i]:.0f} Å"); a.set_xticks([]); a.set_yticks([])
    for z, col in zip(lz, ["tab:red", "tab:green", "tab:orange"]):
        if abs(depths[i] - z) <= 15:
            for sp in a.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(3)
for a in ax.ravel()[len(sel):]:
    a.axis("off")
fig.suptitle("Recovered S-matrix, refocused to depth z: phase sections (coloured frames = true layer depths)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_depth_sections.png")); plt.close(fig)

# ---- fig5: focus curves + best slices vs ground truth ----------------------
fig = plt.figure(figsize=(13, 6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
ax0 = fig.add_subplot(gs[0, :])
cols = ["tab:red", "tab:green", "tab:orange"]
for j in range(3):
    ax0.plot(depths, C[:, j], "-o", ms=3, color=cols[j], label=names[j])
    ax0.axvline(lz[j], color=cols[j], ls="--", alpha=0.5)
ax0.set_xlabel("section depth z (Å)"); ax0.set_ylabel("correlation with true layer")
ax0.legend(ncol=3, loc="upper center"); ax0.grid(alpha=0.3)
ax0.set_title("Depth information: each layer is sharpest at its true depth (dashed)")
res = {}
for j in range(3):
    ib = int(np.argmax(C[:, j]))
    plateau = C[:, j] >= C[ib, j] - 0.03          # centre of the in-focus plateau
    zc = float(np.sum(depths[plateau] * C[plateau, j]) / np.sum(C[plateau, j]))
    res[names[j]] = dict(true_z=lz[j], best_z=zc, peak_z=float(depths[ib]),
                         depth_of_field=float(depths[plateau].max() - depths[plateau].min()),
                         corr=float(C[ib, j]))
    a = fig.add_subplot(gs[1, j])
    p = phase[ib][ext_c, ext_c]
    a.imshow(p - np.median(p), cmap="gray", extent=ext)
    a.set_title(f"recovered @ z={depths[ib]:.0f} Å (r={C[ib, j]:.2f})"); a.set_xticks([]); a.set_yticks([])
a = fig.add_subplot(gs[1, 3]); a.axis("off")
a.text(0, 0.9, "Peak position vs truth", fontsize=11, weight="bold")
for k, (n, r) in enumerate(res.items()):
    a.text(0, 0.7 - 0.2 * k, f"{n.split(' (')[0]}: true {r['true_z']:.0f} Å → found {r['best_z']:.0f} Å", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_depth_localisation.png")); plt.close(fig)

# ---- fig6: recovered S-matrix ---------------------------------------------
Sm = np.load(os.path.join(OUT, "Smatrix_complex64.npy"), mmap_mode="r")
by, bx = s["beams_y"], s["beams_x"]
iy = np.arange(m.Y)[:, None] * m.DR; ix = np.arange(m.X)[None, :] * m.DR
sel_b = [int(np.argmin(by ** 2 + bx ** 2)), int(np.argmin((by - 8) ** 2 + bx ** 2)),
         int(np.argmin((by - 6) ** 2 + (bx + 8) ** 2))]
fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.6))
for k, b in enumerate(sel_b):
    ky, kx = by[b] / m.LW, bx[b] / m.LW
    S0 = np.asarray(Sm[b]) * np.exp(-2j * np.pi * (ky * iy + kx * ix))   # remove plane-wave carrier
    ax[0, k].imshow(np.abs(S0)[ext_c, ext_c], cmap="viridis", extent=ext, vmin=0)
    ax[0, k].set_title(f"|S_b|, beam ({ky*m.LAM*1e3:.1f}, {kx*m.LAM*1e3:.1f}) mrad")
    ax[1, k].imshow(np.angle(S0 * np.exp(-1j * np.angle(S0[ext_c, ext_c].mean())))[ext_c, ext_c],
                    cmap="RdBu_r", extent=ext, vmin=-0.6, vmax=0.6)
    ax[1, k].set_title("arg S_b (carrier removed)")
    for a in ax[:, k]: a.set_xlabel("x (Å)")
fig.suptitle(f"Recovered scattering matrix: {Sm.shape[0]} input beams × {m.Y}×{m.X} px (vacuum would be |S_b|=1, arg=0)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_smatrix_beams.png")); plt.close(fig)

# ---- GIF ------------------------------------------------------------------
frames = []
for i in range(len(depths)):
    p = phase[i][ext_c, ext_c]; p = p - np.median(p)
    lo, hi = -0.6, 0.6
    img = (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(img).resize((300, 300), Image.NEAREST)
    frames.append(im)
frames[0].save(os.path.join(OUT, "depth_sections.gif"), save_all=True,
               append_images=frames[1:] + frames[-2:0:-1], duration=150, loop=0)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(dict(final_loss=float(loss[-1]), first_loss=float(loss[0]), niter=len(loss),
                   localisation=res, depth_axis='from entrance surface'), f, indent=2)
print(json.dumps(res, indent=2))
