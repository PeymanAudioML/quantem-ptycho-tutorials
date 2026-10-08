"""tcBF (parallax) -> ACBF (single-sideband) with quantem's DirectPtychography on the simulated data,
compared with the S-matrix refocusing.  Run inside an env with torch + quantem:  python run_quantem_acbf.py
Depth sections: C10 = s*(z - z_f) with z_f = -df (probe focus depth); s = +-1 calibrated once on the
3-layer data.  quantem result: real phase image on the 0.2 A grid (upsampling_factor=4) over the 20 A scan FOV."""
import os, sys, json
import numpy as np
import quantem as em
import smatrix_depth_demo as m
import sweep_compare as sc

HERE = m.HERE
CROP = np.s_[48:148, 48:148]          # global-grid pixels covered by the scan FOV (100x100 at 0.2 A)
sc.ext_c = slice(48, 148)
DK_MRAD = (1 / m.LW) * m.LAM * 1e3


def make_dp(dp4):
    arr = np.fft.fftshift(dp4, axes=(-2, -1)).astype(np.float32)
    ds = em.core.datastructures.Dataset4dstem.from_array(arr, sampling=(m.STEP * m.DR, m.STEP * m.DR, DK_MRAD, DK_MRAD),
                                                         units=("A", "A", "mrad", "mrad"))
    return em.diffractive_imaging.DirectPtychography.from_dataset4d(
        ds, energy=m.EV, semiangle_cutoff=m.ALPHA * 1e3, rotation_angle=0.0, aberration_coefs={"C10": 0.0},
        force_fitted_origin=(m.K // 2, m.K // 2), verbose=False)


def sections(dpt, zf, depths, kernel, sgn, transpose=False):
    out = np.zeros((len(depths), 100, 100), np.float32)
    for i, z in enumerate(depths):
        o = dpt.reconstruct(deconvolution_kernel=kernel, override_aberration_coefs={"C10": float(sgn * (z - zf))},
                            upsampling_factor=4, verbose=False).obj
        out[i] = o.T if transpose else o
    return out


def crop_truth(arr):
    return np.asarray(arr)[..., CROP[0], CROP[1]]


def curves_crop(stack, layers_c):
    C = np.zeros((len(stack), len(layers_c)))
    for i in range(len(stack)):
        for j, L in enumerate(layers_c):
            C[i, j] = sc.corr(stack[i], L)
    return C


def poly_metrics(stack, slabs_c, depths):
    n = len(depths); M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = sc.corr(stack[i], slabs_c[j])
    A = np.abs(M); diag = np.diag(M)
    loc = np.array([abs(depths[np.argmax(A[i])] - depths[i]) for i in range(n)])
    sel = [A[i, i] - np.mean([A[i, j] for j in range(n) if abs(depths[j] - depths[i]) >= 60]) for i in range(n)]
    return dict(mean_diag_r=float(diag.mean()), mean_depth_err_A=float(loc.mean()), frac_within_10A=float((loc <= 10).mean()),
                selectivity=float(np.mean(sel)), n_inverted=int((diag < 0).sum()), n_depths=n)


def layered_metrics(stack, layers_c, lz):
    C = curves_crop(stack, layers_c); r = sc.summarise(C, lz)
    return dict(n_resolved=r["n_resolved"], n_layers=r["n_layers"], mean_abs_depth_err=r["mean_abs_depth_err"],
                max_abs_depth_err=r["max_abs_depth_err"], mean_abs_r=r["mean_abs_r"], n_contrast_inverted=r["n_contrast_inverted"],
                per_layer=[(x["true_z"], round(x["found_z"], 1), round(x["r"], 2)) for x in r["layers"]])


def run_config(label, rdir, sgn, tr, out):
    cfgp = os.path.join(rdir, "config.json"); cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    d = np.load(os.path.join(rdir, "layers_and_data.npz")); s = np.load(os.path.join(rdir, "sections.npz"))
    dps, truth = d["dps"], d["layers"]
    poly = label == "poly"
    if poly:
        thick, defoci = cfg["thickness"], cfg["defoci"]; DZ = cfg["dz"]
        depths = np.arange(0.0, thick + 1, 10.0)
        zc = (np.arange(truth.shape[0]) + 0.5) * DZ
        slabs_c = [crop_truth(truth[np.abs(zc - z) <= 20.0].sum(0)) for z in depths]
        score = lambda st: poly_metrics(st, slabs_c, depths)
    else:
        defoci = m.DEFOCI; depths = np.arange(0.0, 301.0, 10.0)
        lz = cfg.get("layer_z", m.LAYER_Z); layers_c = crop_truth(truth)
        score = lambda st: layered_metrics(st, layers_c, lz)
    sc.depths = depths; sc.zf_all = [-df for df in defoci]
    stacks, fits = {}, {}
    acbf_all = np.zeros((len(depths), 100, 100), np.float32)
    for idf, df in enumerate(defoci):
        zf = -df; dpt = make_dp(dps[idf])
        tc = sections(dpt, zf, depths, "parallax", sgn, tr); ac = sections(dpt, zf, depths, "single-sideband", sgn, tr)
        stacks[f"tcBF quantem (df={df:.0f})"] = tc; stacks[f"ACBF quantem (df={df:.0f})"] = ac
        acbf_all += ac / len(defoci)
        # stage 1: fit defocus with the tcBF (parallax) kernel; stage 2: ACBF at the fitted value
        from quantem.diffractive_imaging.direct_ptychography import OptimizationParameter
        lo, hi = -zf - 50, (depths[-1] - zf) + 50
        dpt = dpt.grid_search_hyperparameters(aberration_coefs={"C10": OptimizationParameter(lo, hi, n_points=int((hi - lo) // 10) + 1)},
                                              deconvolution_kernel="parallax", verbose=False)
        c10 = float(dpt.aberration_coefs["C10"]); fits[f"df={df:.0f}"] = dict(fitted_C10=c10, implied_depth=sgn * c10 + zf)
        print(label, f"df={df:.0f}: tcBF-fitted C10={c10:.0f} A -> implied depth {sgn * c10 + zf:.0f} A", flush=True)
    stacks["ACBF quantem (3 df averaged)"] = acbf_all
    stacks["tcBF mine (3 df)"] = crop_truth(sc.tcbf_sections(dps, [0, 1, 2]))
    stacks["S-matrix + refocusing"] = crop_truth(np.angle(s["ew"]))
    res = {k: score(v) for k, v in stacks.items()}
    np.savez_compressed(os.path.join(out, f"stacks_{label}.npz"), depths=depths, **{k.replace(" ", "_"): v for k, v in stacks.items()})
    return dict(results=res, fits=fits)


if __name__ == "__main__":
    out = os.path.join(HERE, "results_quantem"); os.makedirs(out, exist_ok=True)
    cal = json.load(open(os.path.join(HERE, "results_quantem_calibration.json"))) if os.path.exists(os.path.join(HERE, "results_quantem_calibration.json")) else None
    sgn, tr = (cal["sign"], cal["transpose"]) if cal else (+1, False)
    allres = {}
    for label, rdir in [("baseline_3layers", "results"), ("six_phi1p5", "results_six_phi1p5"), ("poly", "results_poly")]:
        allres[label] = run_config(label, os.path.join(HERE, rdir), sgn, tr, out)
        json.dump(allres, open(os.path.join(out, "summary.json"), "w"), indent=2)
        print("=====", label)
        for k, v in allres[label]["results"].items():
            print(f"  {k:32s}", {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a != "per_layer"}, flush=True)
