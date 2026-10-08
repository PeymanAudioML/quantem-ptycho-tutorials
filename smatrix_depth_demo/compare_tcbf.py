"""Fair comparison: tcBF / parallax (bright-field only) vs S-matrix recovery + refocusing.

Same simulated 4D-STEM data, same defocus values, same scoring (correlation of the
phase/contrast image at each depth with each true layer).
tcBF: every bright-field detector pixel k gives an image of the sample under a tilted
illumination; a layer at depth z appears shifted by  s*lambda*k*(z - z_f)  (z_f = probe focus
depth = -df).  Shift-and-sum at trial depth z -> depth-resolved tcBF image.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import smatrix_depth_demo as m

OUT = m.OUT
d = np.load(os.path.join(OUT, "layers_and_data.npz"))
s = np.load(os.path.join(OUT, "sections.npz"))
layers, dps = d["layers"], d["dps"]
depths, EW = s["depths"], s["ew"]
ext_c = slice(m.Y // 2 - 50, m.Y // 2 + 50)
ext = [0, 100 * m.DR, 100 * m.DR, 0]
names = ["ring (z=30)", "square (z=150)", "triangle (z=270)"]
zf_all = [-df for df in m.DEFOCI]
NC = m.Y // 4                                    # coarse (scan-sampled) grid, 48x48
c0 = (m.Y // 2 - (m.NSCAN // 2) * m.STEP) // 4   # scan origin on coarse grid
STEP_A = m.STEP * m.DR


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


# bright-field pixels (detector index n -> k = n / LW)
n = np.fft.fftfreq(m.K, 1.0 / m.K).astype(int)
ny, nx = np.meshgrid(n, n, indexing="ij")
kr = np.sqrt(ny ** 2 + nx ** 2) / m.LW
bf = kr <= 0.9 * m.ALPHA * m.KW                  # stay inside the disk edge
ky, kx = (ny[bf] / m.LW), (nx[bf] / m.LW)
print("BF pixels:", bf.sum())

qc = np.fft.fftfreq(NC, STEP_A)                  # coarse-grid spatial frequency (1/A)
QY, QX = qc[:, None], qc[None, :]


def bf_spectra(idf):
    """FFT of every BF image on the coarse (48x48) grid -> (nbf, 48, 48)."""
    imgs = dps[idf][..., bf]                     # (25,25,nbf)
    imgs = imgs - imgs.mean(axis=(0, 1), keepdims=True)   # remove per-pixel mean
    buf = np.zeros((imgs.shape[-1], NC, NC), np.float32)
    buf[:, c0:c0 + m.NSCAN, c0:c0 + m.NSCAN] = np.moveaxis(imgs, -1, 0)
    return np.fft.fft2(buf, axes=(-2, -1))


SP = {i: bf_spectra(i) for i in range(len(m.DEFOCI))}


def upsample(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.zeros((m.Y, m.X), complex)
    o = (m.Y - NC) // 2
    P[o:o + NC, o:o + NC] = F
    return np.real(np.fft.ifft2(np.fft.ifftshift(P))) * (m.Y / NC) ** 2


def sections(idfs, sign):
    out = np.zeros((len(depths), m.Y, m.X))
    for iz, z in enumerate(depths):
        acc = np.zeros((NC, NC), complex)
        for idf in idfs:
            sh_y = sign * m.LAM * ky * (z - zf_all[idf])    # Angstrom shifts per BF pixel
            sh_x = sign * m.LAM * kx * (z - zf_all[idf])
            # shift image back by -shift to align (Fourier shift theorem)
            ramp = np.exp(2j * np.pi * (QY[None] * sh_y[:, None, None] + QX[None] * sh_x[:, None, None]))
            acc += np.sum(SP[idf] * ramp, axis=0)
        out[iz] = upsample(np.real(np.fft.ifft2(acc)))
    return out


def curves(stack):
    C = np.zeros((len(depths), 3))
    for i in range(len(depths)):
        p = stack[i][ext_c, ext_c]
        for j in range(3):
            C[i, j] = corr(p, layers[j][ext_c, ext_c])
    return C


def summarise(C):
    r = {}
    for j in range(3):
        a = np.abs(C[:, j]); ib = int(np.argmax(a))
        pl = a >= a[ib] - 0.03
        r[names[j]] = dict(true_z=m.LAYER_Z[j], best_z=float(np.sum(depths[pl] * a[pl]) / np.sum(a[pl])),
                           peak_abs_corr=float(a[ib]), signed_corr_at_peak=float(C[ib, j]),
                           depth_of_field=float(depths[pl].max() - depths[pl].min()))
    return r


# Shift sign from geometry (not fitted): a ray through the focus (x0, z_f) with angle
# theta = lambda*k is at x0 + theta*(z - z_f) in plane z, so BF image k shows the layer
# displaced by +theta*(z - z_f); aligning needs the opposite shift -> sign = -1.
sign = -1

variants = {
    "tcBF, df=0 only (1/3 data)": [2],
    "tcBF, df=-200 only (1/3 data)": [0],
    "tcBF, all 3 defocus (same data)": [0, 1, 2],
}
res, stacks, Cs = {}, {}, {}
for k, idfs in variants.items():
    stacks[k] = sections(idfs, sign)
    Cs[k] = curves(stacks[k]); res[k] = summarise(Cs[k])
C_S = curves(np.angle(EW))
res["S-matrix + refocusing (all 3 defocus)"] = summarise(C_S)
Cs["S-matrix + refocusing (all 3 defocus)"] = C_S
stacks["S-matrix + refocusing (all 3 defocus)"] = np.angle(EW)
print(json.dumps(res, indent=1))

# ---------------- figures
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
keys = ["tcBF, df=0 only (1/3 data)", "tcBF, all 3 defocus (same data)", "S-matrix + refocusing (all 3 defocus)"]
fig, ax = plt.subplots(len(keys) + 1, 3, figsize=(10, 12))
for j in range(3):
    ax[0, j].imshow(layers[j][ext_c, ext_c], cmap="magma", extent=ext)
    ax[0, j].set_title(f"truth: {names[j]}"); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
for r, k in enumerate(keys, start=1):
    for j in range(3):
        a = np.abs(Cs[k][:, j]); ib = int(np.argmax(a)); zc = res[k][names[j]]["best_z"]
        iz = int(np.argmin(np.abs(depths - zc)))
        p = stacks[k][iz][ext_c, ext_c]; p = p - np.median(p)
        ax[r, j].imshow(p, cmap="gray", extent=ext)
        ax[r, j].set_title(f"{k.split(',')[0] if 'tcBF' in k else 'S-matrix'} {'(' + k.split(',')[1].strip() + ')' if 'tcBF' in k else ''}\n"
                           f"z={depths[iz]:.0f} Å, r={Cs[k][iz, j]:+.2f}", fontsize=8)
        ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
fig.suptitle("Same data, three reconstructions: each layer at the depth where it is best focused")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_compare_images.png")); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, k in zip(ax, keys):
    for j, col in enumerate(["tab:red", "tab:green", "tab:orange"]):
        a.plot(depths, Cs[k][:, j], "-o", ms=3, color=col, label=names[j])
        a.axvline(m.LAYER_Z[j], color=col, ls="--", alpha=0.4)
    a.set_title(k, fontsize=9); a.set_xlabel("section depth z (Å)"); a.grid(alpha=0.3)
ax[0].set_ylabel("correlation with true layer"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_compare_depth_curves.png")); plt.close(fig)
with open(os.path.join(OUT, "comparison_summary.json"), "w") as f:
    json.dump(dict(shift_sign=sign, results=res), f, indent=2)
