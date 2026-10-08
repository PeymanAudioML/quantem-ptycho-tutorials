"""Compare tcBF vs S-matrix refocusing across geometries (baseline + harder configs).
usage: python sweep_compare.py [results_dir ...]   (default: results results_six_phi1p5 results_six_phi2p5)
Both methods get the same data (all 3 defocus). Depth = centre of the contiguous in-focus
plateau of |correlation with the true layer|; contrast sign reported separately."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import smatrix_depth_demo as m

HERE = m.HERE
ext_c = slice(m.Y // 2 - 50, m.Y // 2 + 50)
ext = [0, 100 * m.DR, 100 * m.DR, 0]
NC = m.Y // 4
c0 = (m.Y // 2 - (m.NSCAN // 2) * m.STEP) // 4
STEP_A = m.STEP * m.DR
depths = np.arange(0.0, m.THICK + 1, 10.0)
zf_all = [-df for df in m.DEFOCI]

n = np.fft.fftfreq(m.K, 1.0 / m.K).astype(int)
ny, nx = np.meshgrid(n, n, indexing="ij")
bf = np.sqrt(ny ** 2 + nx ** 2) / m.LW <= 0.9 * m.ALPHA * m.KW
ky, kx = ny[bf] / m.LW, nx[bf] / m.LW
qc = np.fft.fftfreq(NC, STEP_A)
QY, QX = qc[:, None], qc[None, :]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def upsample(img):
    F = np.fft.fftshift(np.fft.fft2(img)); P = np.zeros((m.Y, m.X), complex)
    o = (m.Y - NC) // 2; P[o:o + NC, o:o + NC] = F
    return np.real(np.fft.ifft2(np.fft.ifftshift(P))) * (m.Y / NC) ** 2


def tcbf_sections(dps, idfs):
    sp = {}
    for i in idfs:
        imgs = dps[i][..., bf]; imgs = imgs - imgs.mean(axis=(0, 1), keepdims=True)
        buf = np.zeros((imgs.shape[-1], NC, NC), np.float32)
        buf[:, c0:c0 + m.NSCAN, c0:c0 + m.NSCAN] = np.moveaxis(imgs, -1, 0)
        sp[i] = np.fft.fft2(buf, axes=(-2, -1))
    out = np.zeros((len(depths), m.Y, m.X))
    for iz, z in enumerate(depths):
        acc = np.zeros((NC, NC), complex)
        for i in idfs:   # sign -1: geometric (ray through focus z_f is displaced by theta*(z-z_f))
            shy = -m.LAM * ky * (z - zf_all[i]); shx = -m.LAM * kx * (z - zf_all[i])
            ramp = np.exp(2j * np.pi * (QY[None] * shy[:, None, None] + QX[None] * shx[:, None, None]))
            acc += np.sum(sp[i] * ramp, axis=0)
        out[iz] = upsample(np.real(np.fft.ifft2(acc)))
    return out


def curves(stack, layers):
    C = np.zeros((len(depths), len(layers)))
    for i in range(len(depths)):
        p = stack[i][ext_c, ext_c]
        for j, L in enumerate(layers):
            C[i, j] = corr(p, L[ext_c, ext_c])
    return C


def plateau_depth(c):
    a = np.abs(c); ib = int(np.argmax(a)); thr = a[ib] - 0.03
    lo = hi = ib
    while lo > 0 and a[lo - 1] >= thr: lo -= 1
    while hi < len(a) - 1 and a[hi + 1] >= thr: hi += 1
    w = a[lo:hi + 1]
    return float(np.sum(depths[lo:hi + 1] * w) / np.sum(w)), ib, float(depths[hi] - depths[lo])


def summarise(C, lz):
    rows = []
    for j, z in enumerate(lz):
        zc, ib, dof = plateau_depth(C[:, j])
        rows.append(dict(true_z=z, found_z=zc, err=zc - z, r=float(C[ib, j]), dof=dof))
    err = np.array([r["err"] for r in rows]); rr = np.array([r["r"] for r in rows])
    return dict(layers=rows, mean_abs_depth_err=float(np.abs(err).mean()),
                max_abs_depth_err=float(np.abs(err).max()),
                mean_abs_r=float(np.abs(rr).mean()), n_contrast_inverted=int((rr < 0).sum()),
                n_resolved=int(((np.abs(err) <= 25) & (np.abs(rr) > 0.5)).sum()), n_layers=len(rows))


def analyse(rdir):
    cfgp = os.path.join(rdir, "config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp)); lz, motifs = cfg["layer_z"], cfg["motifs"]; label = cfg["name"]
        phi = cfg["phase_scale"]
    else:
        lz, motifs, label, phi = m.LAYER_Z, ["ring", "square", "triangle"], "baseline_3layers", 1.0
    d = np.load(os.path.join(rdir, "layers_and_data.npz")); s = np.load(os.path.join(rdir, "sections.npz"))
    layers, dps = d["layers"], d["dps"]
    stacks = {"tcBF (3 defocus)": tcbf_sections(dps, [0, 1, 2]),
              "tcBF (df=0 only)": tcbf_sections(dps, [2]),
              "S-matrix + refocusing": np.angle(s["ew"])}
    res, Cs = {}, {}
    for k, st in stacks.items():
        Cs[k] = curves(st, layers); res[k] = summarise(Cs[k], lz)
    return dict(label=label, phi=phi, lz=lz, motifs=motifs, layers=layers, stacks=stacks, Cs=Cs, res=res)


def figs(A, out):
    nL = len(A["lz"]); ks = ["tcBF (3 defocus)", "S-matrix + refocusing"]
    fig, ax = plt.subplots(3, nL, figsize=(2.1 * nL, 6.6), squeeze=False)
    for j in range(nL):
        ax[0, j].imshow(A["layers"][j][ext_c, ext_c], cmap="magma", extent=ext)
        ax[0, j].set_title(f"{A['motifs'][j]}\nz={A['lz'][j]:.0f}", fontsize=8)
        for r, k in enumerate(ks, start=1):
            row = A["res"][k]["layers"][j]; iz = int(np.argmin(np.abs(depths - row["found_z"])))
            p = A["stacks"][k][iz][ext_c, ext_c]
            ax[r, j].imshow(p - np.median(p), cmap="gray", extent=ext)
            ax[r, j].set_title(f"{'tcBF' if 'tcBF' in k else 'S-matrix'} z={depths[iz]:.0f}\nr={A['Cs'][k][iz, j]:+.2f}", fontsize=7)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{A['label']} (phase/atom = {A['phi']}): each layer at its best-focus depth", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"sweep_images_{A['label']}.png")); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for a, k in zip(ax, ks):
        for j in range(nL):
            a.plot(depths, A["Cs"][k][:, j], "-", lw=1.3, color=plt.cm.tab10(j), label=A["motifs"][j])
            a.axvline(A["lz"][j], color=plt.cm.tab10(j), ls="--", alpha=0.35)
        a.set_title(k, fontsize=9); a.set_xlabel("section depth z (Å)"); a.grid(alpha=0.3)
    ax[0].set_ylabel("correlation with true layer"); ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"sweep_curves_{A['label']}.png")); plt.close(fig)


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["results", "results_six_phi1p5", "results_six_phi2p5"]
    dirs = [d if os.path.isabs(d) else os.path.join(HERE, d) for d in dirs]
    out = os.path.join(HERE, "results_sweep"); os.makedirs(out, exist_ok=True)
    allres = {}
    for d in dirs:
        if not os.path.exists(os.path.join(d, "sections.npz")):
            print("skip (no sections):", d); continue
        A = analyse(d); figs(A, out); allres[A["label"]] = dict(phase_scale=A["phi"], methods=A["res"])
        print(A["label"])
        for k, v in A["res"].items():
            print(f"  {k:24s} resolved {v['n_resolved']}/{v['n_layers']}  mean|err| {v['mean_abs_depth_err']:5.1f} A  "
                  f"max|err| {v['max_abs_depth_err']:5.1f} A  mean|r| {v['mean_abs_r']:.2f}  inverted {v['n_contrast_inverted']}")
    json.dump(allres, open(os.path.join(out, "sweep_summary.json"), "w"), indent=2)
    labs = list(allres); ms = ["tcBF (df=0 only)", "tcBF (3 defocus)", "S-matrix + refocusing"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (key, ttl) in zip(ax, [("mean_abs_depth_err", "mean |depth error| (Å)"), ("mean_abs_r", "mean |correlation| with truth"),
                                  ("n_resolved", "layers correctly resolved")]):
        w = 0.27
        for i, mth in enumerate(ms):
            vals = [allres[l]["methods"][mth][key] for l in labs]
            a.bar(np.arange(len(labs)) + (i - 1) * w, vals, w, label=mth)
        a.set_xticks(range(len(labs))); a.set_xticklabels(labs, fontsize=7, rotation=10); a.set_title(ttl, fontsize=9); a.grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "sweep_summary.png")); plt.close(fig)
