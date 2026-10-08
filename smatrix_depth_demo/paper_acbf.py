"""Paper-faithful tcBF / acBF depth sectioning (Ma, Lee, Shi, Muller, Zeltmann, 'Parallax Depth Sectioning
and 3D Reconstruction in 4D-STEM'), implemented in NumPy from their equations, defocus-only aberration:

  D(r,Th)   = I(r,Th) - I0(Th)                       (Eq. 1; I0 = scan-mean)
  tcBF_z(q) = sum_Th  D~(q,Th) exp(-i 2pi q.Th (df - z))                         (Eq. 4)
  acBF_z(q) = sum_Th  D~(q,Th) exp(-i 2pi q.Th (df - z)) exp(-i arg PCTFres_z)   (Eq. 7)
  PCTFres   = i A(Th)/(2 Om0) [ A(Th-lam q) e^{-i pi lam (df-z) q^2} - A(Th+lam q) e^{+i pi lam (df-z) q^2} ]  (Eq. 5/6)
df = beam-crossover depth z_f = -(probe defocus used in my simulation).  Each defocus dataset is used on its own.
Detector handedness: data detector angle is mapped to Th = HAND * lam * k (HAND = +-1, fixed once on the 3-layer data)."""
import os, sys, json
import numpy as np
import smatrix_depth_demo as m
import sweep_compare as sc

HERE = m.HERE
NC = m.Y // 4
C0 = (m.Y // 2 - (m.NSCAN // 2) * m.STEP) // 4
STEP_A = m.STEP * m.DR
CROP = np.s_[48:148, 48:148]

n = np.fft.fftfreq(m.K, 1.0 / m.K).astype(int)
NY, NX = np.meshgrid(n, n, indexing="ij")
BF = (np.sqrt(NY ** 2 + NX ** 2) / m.LW) * m.LAM <= m.ALPHA + 1e-12       # |Theta| <= alpha
QC = np.fft.fftfreq(NC, STEP_A)
QY, QX = QC[:, None], QC[None, :]
Q2 = QY ** 2 + QX ** 2
OM0 = np.pi * m.ALPHA ** 2
# acBF output has the opposite overall sign to my object-phase convention (t = exp(+i*phi)) in EVERY structure and
# at every depth (uniform, not depth-dependent), i.e. the sign convention of the phase in Rose's PCTF. One global flip:
ACBF_SIGN = -1.0


def prep(dp4, hand):
    """FFT of background-subtracted virtual BF images on the padded scan grid; detector angles Th (nbf,2)."""
    imgs = dp4[..., BF]; imgs = imgs - imgs.mean(axis=(0, 1), keepdims=True)          # D = I - I0
    buf = np.zeros((imgs.shape[-1], NC, NC), np.float32)
    buf[:, C0:C0 + m.NSCAN, C0:C0 + m.NSCAN] = np.moveaxis(imgs, -1, 0)
    Dq = np.fft.fft2(buf, axes=(-2, -1))
    thy = hand * NY[BF] / m.LW * m.LAM; thx = hand * NX[BF] / m.LW * m.LAM
    ay = lambda s: (np.sqrt((thy[:, None, None] + s * m.LAM * QY[None]) ** 2 + (thx[:, None, None] + s * m.LAM * QX[None]) ** 2) <= m.ALPHA)
    Am, Ap = ay(-1.0), ay(+1.0)            # A(Th - lam q), A(Th + lam q)
    return Dq, thy, thx, Am, Ap


def upsample(img):
    F = np.fft.fftshift(np.fft.fft2(img)); P = np.zeros((m.Y, m.X), complex); o = (m.Y - NC) // 2
    P[o:o + NC, o:o + NC] = F
    return np.real(np.fft.ifft2(np.fft.ifftshift(P))) * (m.Y / NC) ** 2


def sections(prepped, zf, depths, kind):
    Dq, thy, thx, Am, Ap = prepped
    out = np.zeros((len(depths), 100, 100), np.float32)
    for i, z in enumerate(depths):
        dfz = zf - z
        reg = np.exp(-2j * np.pi * (thy[:, None, None] * QY[None] + thx[:, None, None] * QX[None]) * dfz)
        T = Dq * reg
        if kind == "acbf":
            ph = np.pi * m.LAM * dfz * Q2
            P = 1j / (2 * OM0) * (Am * np.exp(-1j * ph)[None] - Ap * np.exp(1j * ph)[None])
            mag = np.abs(P)
            corr = np.where(mag > 1e-12 * max(1.0, mag.max()), np.exp(-1j * np.angle(P)), 0.0)    # zero where no transfer
            T = T * corr
        J = np.real(np.fft.ifft2(T.sum(0))) * (ACBF_SIGN if kind == "acbf" else 1.0)
        out[i] = upsample(J)[CROP]
    return out


def curves(stack, layers_c):
    C = np.zeros((len(stack), len(layers_c)))
    for i in range(len(stack)):
        for j, L in enumerate(layers_c):
            C[i, j] = sc.corr(stack[i], L)
    return C


def layered_metrics(stack, layers_c, lz):
    r = sc.summarise(curves(stack, layers_c), lz)
    return dict(n_resolved=r["n_resolved"], n_layers=r["n_layers"], mean_abs_depth_err=r["mean_abs_depth_err"],
                max_abs_depth_err=r["max_abs_depth_err"], mean_abs_r=r["mean_abs_r"], n_contrast_inverted=r["n_contrast_inverted"],
                per_layer=[(x["true_z"], round(x["found_z"], 1), round(x["r"], 2)) for x in r["layers"]])


def poly_metrics(stack, slabs_c, depths):
    k = len(depths); M = np.array([[sc.corr(stack[i], slabs_c[j]) for j in range(k)] for i in range(k)])
    A = np.abs(M); diag = np.diag(M)
    loc = np.array([abs(depths[np.argmax(A[i])] - depths[i]) for i in range(k)])
    sel = [A[i, i] - np.mean([A[i, j] for j in range(k) if abs(depths[j] - depths[i]) >= 60]) for i in range(k)]
    return dict(mean_diag_r=float(diag.mean()), mean_depth_err_A=float(loc.mean()), frac_within_10A=float((loc <= 10).mean()),
                selectivity=float(np.mean(sel)), n_inverted=int((diag < 0).sum()), n_depths=k)


def setup(label, rdir):
    cfgp = os.path.join(rdir, "config.json"); cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    d = np.load(os.path.join(rdir, "layers_and_data.npz")); s = np.load(os.path.join(rdir, "sections.npz"))
    truth, dps = d["layers"], d["dps"]
    if label == "poly":
        defoci = cfg["defoci"]; depths = np.arange(0.0, cfg["thickness"] + 1, 10.0)
        zc = (np.arange(truth.shape[0]) + 0.5) * cfg["dz"]
        slabs_c = [truth[np.abs(zc - z) <= 20.0].sum(0)[CROP] for z in depths]
        score = lambda st: poly_metrics(st, slabs_c, depths)
    else:
        defoci = m.DEFOCI; depths = np.arange(0.0, 301.0, 10.0)
        lz = cfg.get("layer_z", m.LAYER_Z); layers_c = [L[CROP] for L in truth]
        sc.depths = depths
        score = lambda st: layered_metrics(st, layers_c, lz)
    return dps, defoci, depths, score, np.angle(s["ew"])[(slice(None),) + CROP]


if __name__ == "__main__":
    out = os.path.join(HERE, "results_paper"); os.makedirs(out, exist_ok=True)
    # --- one-time handedness check on the 3-layer data, df=0 dataset (crossover at the entrance surface)
    dps, defoci, depths, score, _ = setup("baseline_3layers", os.path.join(HERE, "results"))
    d0 = np.load(os.path.join(HERE, "results", "layers_and_data.npz"))["layers"]; layers_c = [L[CROP] for L in d0]
    tab = {}
    for hand in (+1, -1):
        pr = prep(dps[2], hand)
        for kind in ("tcbf", "acbf"):
            st = sections(pr, -defoci[2], depths, kind); C = curves(st, layers_c)
            tab[(hand, kind)] = float(np.abs(C).max(0).sum())
            print(f"hand={hand:+d} {kind}: sum peak|r|={tab[(hand, kind)]:.2f} peaks at {[int(depths[np.argmax(np.abs(C[:, j]))]) for j in range(3)]} (true 30/150/270)", flush=True)
    hand = max([(+1, "acbf"), (-1, "acbf")], key=lambda t: tab[t])[0]
    print("handedness chosen:", hand, flush=True)
    json.dump(dict(hand=hand, table={f"{k[0]}_{k[1]}": v for k, v in tab.items()}), open(os.path.join(out, "handedness.json"), "w"))
