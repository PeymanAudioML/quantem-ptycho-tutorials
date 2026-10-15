"""Evaluate saved reconstructions against the ground truth (the only place besides validate_forward that reads it).

    python evaluate_reconstruction.py smatrix --specimen A_z100            # every results/smatrix/<config>/<specimen>/*
    -> results/ground_truth_comparison/<config>/<specimen>/smatrix_<run>.json and SMATRIX_<specimen>.md

S-matrix (section 8 / 11):
  probe      per dataset: aberration errors (C10, C12a, C12b), complex overlap and RMS aperture phase error of the
             fitted probe vs the true probe at the S-matrix beams (piston and tilt removed, run_realization.probe_match);
             the reference dataset is fixed by the gauge and labelled as such, not as recovered
  features   candidates (found without truth) are matched to true Au atoms within --match-tol (lateral, A); a true
             feature is detected when any of its atoms is matched; unmatched candidates are false detections;
             feature depth = median z of its matched candidates; axial error e_z = z_hat - z_true
  reliability  per candidate: flag vs actual |e_z|
"""
import argparse, glob, json, os

import numpy as np

import bench


def true_probes_at_beams(gt, ky, kx, dr):
    import zarr
    out = []
    d = 0
    while os.path.isdir(os.path.join(gt, f"probe_d{d}.zarr")):
        F = zarr.open_array(os.path.join(gt, f"probe_d{d}.zarr", "array1"), mode="r")[...]
        N = F.shape
        out.append(F[np.round(ky * N[0] * dr).astype(int) % N[0], np.round(kx * N[1] * dr).astype(int) % N[1]])
        d += 1
    return np.array(out)


def match_3d(preds, truth, tol):
    """preds: list of dict(y, x, z).  A prediction is assigned to the true feature with an Au atom within `tol`
    laterally and the smallest |dz| among those; unassigned predictions are false detections.  Per true feature:
    detected, z_hat = median z of its predictions, e_z.  Ground truth is used here only."""
    per = {f["id"]: [] for f in truth["features"]}
    false = []
    for i, p in enumerate(preds):
        xy = np.array([p["y"], p["x"]])
        close = [f for f in truth["features"] if min(np.hypot(*(xy - np.array(q[:2]))) for q in f["positions"]) <= tol]
        if not close:
            false.append(i)
            continue
        f = min(close, key=lambda f: abs(p["z"] - f["z"]))
        per[f["id"]].append(i)
    rows = []
    for f in truth["features"]:
        zs = [preds[i]["z"] for i in per[f["id"]]]
        z_hat = float(np.median(zs)) if zs else None
        rows.append(dict(feature=f["id"], z_true=f["z"], detected=bool(zs), n_predictions=len(zs), z_hat=z_hat,
                         e_z=None if z_hat is None else z_hat - f["z"], prediction_z=zs))
    det = [r for r in rows if r["detected"]]
    ez = np.array([r["e_z"] for r in det]) if det else np.array([])
    return dict(features=rows, n_true=len(rows), n_detected=len(det), n_missed=len(rows) - len(det), n_false=len(false),
                false_predictions=[preds[i] for i in false], match_tol_A=tol,
                axial=dict(mae=float(np.abs(ez).mean()) if len(ez) else None,
                           median_ae=float(np.median(np.abs(ez))) if len(ez) else None,
                           rmse=float(np.sqrt((ez ** 2).mean())) if len(ez) else None,
                           bias=float(ez.mean()) if len(ez) else None))


def match_features(cands, truth, tol):
    atoms = [(f["id"], np.array(p[:2])) for f in truth["features"] for p in f["positions"]]
    per_feat = {f["id"]: [] for f in truth["features"]}
    false = []
    for i, c in enumerate(cands):
        xy = np.array([c["y"], c["x"]])
        dist = [(np.hypot(*(xy - p)), fid) for fid, p in atoms]
        dmin, fid = min(dist) if dist else (np.inf, None)
        if dmin <= tol:
            per_feat[fid].append((i, float(dmin)))
        else:
            false.append(i)
    rows = []
    for f in truth["features"]:
        m = per_feat[f["id"]]
        zs = [cands[i]["z"] for i, _ in m]
        z_hat = float(np.median(zs)) if zs else None
        rows.append(dict(feature=f["id"], z_true=f["z"], n_atoms=f["n_atoms"], detected=bool(m), n_candidates=len(m),
                         z_hat=z_hat, e_z=None if z_hat is None else z_hat - f["z"],
                         lateral_error_A=float(np.mean([d for _, d in m])) if m else None,
                         candidate_z=zs))
    return rows, false


def eval_smatrix(cfg, specimen, tol, log=print):
    import run_realization as rr
    sd = bench.specimen_dir(cfg, specimen)
    gt = os.path.join(sd, "ground_truth")
    truth = json.load(open(os.path.join(gt, "truth.json")))
    true = np.array(truth["aberrations"]["true"])
    ref = truth["aberrations"]["reference_dataset"]
    runs = sorted(glob.glob(os.path.join(bench.HERE, "results", "smatrix", cfg["name"], specimen, "*", "summary.json")))
    out = os.path.join(bench.HERE, "results", "ground_truth_comparison", cfg["name"], specimen)
    os.makedirs(out, exist_ok=True)
    rows = []
    for f in runs:
        s = json.load(open(f))
        arr = np.load(os.path.join(os.path.dirname(f), "arrays.npz"))
        ky, kx, psi = arr["beams_ky"], arr["beams_kx"], arr["probe_beams"]
        tp = true_probes_at_beams(gt, ky, kx, s["setup"]["dr"])
        coeff = np.array(s["coefficients"])
        probe = []
        for d in range(len(coeff)):
            ov, rms = rr.probe_match(psi[d], tp[d], ky, kx)
            a_est, a_true = np.abs(psi[d]), np.abs(tp[d])
            amp_err = float(np.sqrt(((a_est / np.linalg.norm(a_est) - a_true / np.linalg.norm(a_true)) ** 2).sum()))
            probe.append(dict(dataset=d, gauge_fixed=d == ref,
                              error_C10=float(coeff[d, 0] - true[d, 0]), error_C12a=float(coeff[d, 1] - true[d, 1]),
                              error_C12b=float(coeff[d, 2] - true[d, 2]), overlap=ov, rms_phase_rad=rms,
                              amplitude_rel_error=amp_err))
        cands = s["depth"]["candidates"]
        feats, false = match_features(cands, truth, tol)
        rel = [dict(candidate=i, reliable=c["reliable"], z=c["z"], uncertainty=c["uncertainty_A"],
                    abs_error=None) for i, c in enumerate(cands)]
        for fr in feats:
            for z in fr["candidate_z"]:
                for r in rel:
                    if r["z"] == z and r["abs_error"] is None:
                        r["abs_error"] = abs(z - fr["z_true"])
                        break
        det = [fr for fr in feats if fr["detected"]]
        ez = np.array([fr["e_z"] for fr in det]) if det else np.array([])
        res = dict(run=os.path.basename(os.path.dirname(f)), noise=s["noise"], variant=s["variant"], seed=s["seed"],
                   epochs=s["epochs"], loss_train=s["loss_train"], loss_heldout=s["loss_heldout"],
                   probe=probe, features=feats, n_true=len(feats), n_detected=len(det), n_missed=len(feats) - len(det),
                   n_false=len(false), match_tol_A=tol,
                   axial=dict(mae=float(np.abs(ez).mean()) if len(ez) else None,
                              median_ae=float(np.median(np.abs(ez))) if len(ez) else None,
                              rmse=float(np.sqrt((ez ** 2).mean())) if len(ez) else None,
                              bias=float(ez.mean()) if len(ez) else None),
                   reliability=rel, focus_curve_layers=s["depth"]["layers"],
                   matched_3d_primary=match_3d([dict(y=c["y"], x=c["x"], z=c["z"]) for c in cands], truth, tol),
                   matched_3d_all_peaks=match_3d([dict(y=c["y"], x=c["x"], z=z) for c in cands
                                                  for z in (c.get("z_peaks") or [c["z"]])], truth, tol))
        json.dump(res, open(os.path.join(out, f"smatrix_{res['run']}.json"), "w"), indent=1)
        rows.append(res)
        log(f"{res['run']}: loss train {res['loss_train']:.3e} / held-out {res['loss_heldout']:.3e}; probe " + "; ".join(
            f"d{p['dataset']}{'(gauge)' if p['gauge_fixed'] else ''} dC10 {p['error_C10']:+.2f} dC12a {p['error_C12a']:+.2f} "
            f"dC12b {p['error_C12b']:+.2f} ov {p['overlap']:.4f} {p['rms_phase_rad']:.3f} rad" for p in probe)
            + f"; features {res['n_detected']}/{res['n_true']} detected, {res['n_false']} false; e_z "
            + str([None if fr["e_z"] is None else round(fr["e_z"], 1) for fr in feats]))
    L = [f"# S-matrix stage vs ground truth: {specimen} ({cfg['name']})", "",
         f"True features: " + ", ".join(f"#{f['id']} z = {f['z']:g} A ({f['n_atoms']} Au)" for f in truth["features"])
         + f". Matching tolerance {tol} A (lateral, to any true Au atom). Dataset {ref} is the gauge reference: its "
           "probe is fixed at the (exact) nominal values, not recovered.", "",
         "| run | train loss | held-out loss | dataset | dC10 | dC12a | dC12b | overlap | RMS phase (rad) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for p in r["probe"]:
            L.append(f"| {r['run']} | {r['loss_train']:.3e} | {r['loss_heldout']:.3e} | {p['dataset']}"
                     f"{' (gauge)' if p['gauge_fixed'] else ''} | {p['error_C10']:+.2f} | {p['error_C12a']:+.2f} | "
                     f"{p['error_C12b']:+.2f} | {p['overlap']:.4f} | {p['rms_phase_rad']:.3f} |")
    L += ["", "| run | detected / true | missed | false | feature: z_true -> z_hat (e_z) | MAE | bias |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        fz = "; ".join(f"{fr['z_true']:g} -> " + ("missed" if fr["z_hat"] is None else f"{fr['z_hat']:.1f} ({fr['e_z']:+.1f})")
                       for fr in r["features"])
        mae = "-" if r["axial"]["mae"] is None else f"{r['axial']['mae']:.1f}"
        bias = "-" if r["axial"]["bias"] is None else f"{r['axial']['bias']:+.1f}"
        L.append(f"| {r['run']} | {r['n_detected']} / {r['n_true']} | {r['n_missed']} | {r['n_false']} | {fz} | {mae} | {bias} |")
    open(os.path.join(out, f"SMATRIX_{specimen}.md"), "w").write("\n".join(L) + "\n")
    return rows


def eval_quantem(cfg, specimen, tol, log=print):
    """QuantEM runs vs truth: projected potential, depth-resolved potential, feature depths (found from the
    reconstruction alone), probe."""
    import run_realization as rr
    import eval_ground_truth as eg
    setup = bench.make_setup(cfg)
    sd = bench.specimen_dir(cfg, specimen)
    gt = os.path.join(sd, "ground_truth")
    truth = json.load(open(os.path.join(gt, "truth.json")))
    T = truth["thickness_A"]
    sig = truth["potential"]["sigma_rad_per_V_A"]
    tr = rr.load_truth([os.path.join(sd, "window_noiseless", "def0")], lambda *_: None)
    pot, dr_t, st = tr["potential"]
    ky, kx = setup.beams()
    tp = true_probes_at_beams(gt, ky, kx, setup.dr)
    n_scan = truth["scan"]["shape"]
    step = truth["scan"]["step_A"]
    out = os.path.join(bench.HERE, "results", "ground_truth_comparison", cfg["name"], specimen)
    os.makedirs(out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(bench.HERE, "results", "quantem", cfg["name"], specimen, "*", "summary.json"))):
        s = json.load(open(f))
        tag = os.path.basename(os.path.dirname(f))
        npz = os.path.join(bench.HERE, "runs", cfg["name"], specimen, tag, "quantem.npz")
        if not os.path.exists(npz):
            log(f"{tag}: arrays missing ({npz}), skipped")
            continue
        q = np.load(npz)
        obj, bnd = q["object_potential"], q["slice_boundaries"]
        dr_q, o = s["dr_quantem"], s["origin_A"]
        p0 = np.array(s["first_scan_pixel"])
        j0 = p0.astype(int)
        j1 = j0 + np.round((np.array(n_scan) - 1) * step / dr_q).astype(int) + 1
        rows_A = o[0] + np.arange(j0[0], j1[0]) * dr_q
        cols_A = o[1] + np.arange(j0[1], j1[1]) * dr_q
        crop = obj[:, j0[0]:j1[0], j0[1]:j1[1]]
        t_proj = rr.sample_truth(pot, dr_t, rows_A, cols_A, 0, T + 1e-6, st, dr_q) * sig
        r_proj = crop.sum(0)
        proj = eg.fit_stats(r_proj, t_proj)
        k, fr = eg.frc(r_proj, t_proj, dr_q)
        above = np.flatnonzero((fr < 0.5) & (k > 0.1))
        proj["frc_half_crossing_inv_A"] = float(k[above[0]]) if len(above) else None
        proj["frc"] = dict(k=k.tolist(), frc=fr.tolist())
        # depth-resolved: the truth integrated over each reconstruction slab's physical interval
        slabs = []
        for kk in range(len(bnd) - 1):
            t_k = rr.sample_truth(pot, dr_t, rows_A, cols_A, bnd[kk] - 1e-6, bnd[kk + 1] - 1e-6, st, dr_q) * sig
            r_k = crop[kk]
            slabs.append(dict(z0=float(bnd[kk]), z1=float(bnd[kk + 1]), truth_phase_max=float(t_k.max()),
                              truth_std_per_A=float(t_k.std() / (bnd[kk + 1] - bnd[kk])),
                              recon_std_per_A=float(r_k.std() / (bnd[kk + 1] - bnd[kk])),
                              corr=float(rr.corr(r_k, t_k)) if t_k.std() > 1e-9 else None))
        tprof = np.array([x["truth_std_per_A"] for x in slabs])
        rprof = np.array([x["recon_std_per_A"] for x in slabs])
        depth_profile_corr = float(rr.corr(rprof, tprof)) if tprof.std() > 0 else None
        # features from the reconstruction alone: projected peaks -> column depth profiles -> all depth peaks
        zc = 0.5 * (bnd[1:] + bnd[:-1])
        pk = eg.detect_peaks(r_proj, dr_q, min_sep_A=1.0, rel=0.5)
        preds_primary, preds_all = [], []
        for (py, px) in pk:
            iy, ix = int(round(py / dr_q)), int(round(px / dr_q))
            col = crop[:, max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].mean(axis=(1, 2)) / np.diff(bnd)
            col = col - np.median(col)
            y, x = rows_A[0] + py, cols_A[0] + px
            zp = rr.local_peaks(zc, col, 4, frac=0.5, min_sep=0.5 * setup.lam / setup.alpha ** 2)
            zmain = float(zc[int(np.argmax(col))])
            preds_primary.append(dict(y=y, x=x, z=zmain))
            preds_all += [dict(y=y, x=x, z=float(z)) for z in (zp or [zmain])]
        probe = s.get("probe_mode")
        p_fin = rr.probe_match(rr.probe_at_beams(q["probe"], dr_q, ky, kx), tp[s["dataset"]], ky, kx)
        p_ini = rr.probe_match(rr.probe_at_beams(q["initial_probe"], dr_q, ky, kx), tp[s["dataset"]], ky, kx)
        res = dict(run=tag, arm=s["arm"], probe_mode=probe, oracle=s["oracle"], candidates=s["candidates"],
                   planes=s["planes"], final_loss=s["final_loss"], converged_rel_change=s["converged_rel_change_last10pct"],
                   projected=proj, slabs=slabs, depth_profile_corr=depth_profile_corr,
                   mean_slab_corr=float(np.mean([x["corr"] for x in slabs if x["corr"] is not None]))
                   if any(x["corr"] is not None for x in slabs) else None,
                   probe=dict(initial_overlap=p_ini[0], initial_rms_rad=p_ini[1], final_overlap=p_fin[0],
                              final_rms_rad=p_fin[1]),
                   features_primary=match_3d(preds_primary, truth, tol), features_all_peaks=match_3d(preds_all, truth, tol))
        json.dump(res, open(os.path.join(out, f"{tag}.json"), "w"), indent=1)
        rows.append(res)
        fp = res["features_all_peaks"]
        log(f"{tag}: loss {res['final_loss']:.4g}; projected corr {proj['corr']:.3f} slope {proj['slope']:.2f} NRMSE "
            f"{proj['nrmse_fit']:.2f}; slab corr {res['mean_slab_corr']}; probe rms {p_ini[1]:.3f} -> {p_fin[1]:.3f} rad; "
            f"features {fp['n_detected']}/{fp['n_true']} ({fp['n_false']} false), e_z "
            f"{[None if r['e_z'] is None else round(r['e_z'], 1) for r in fp['features']]}")
    L = [f"# QuantEM vs ground truth: {specimen} ({cfg['name']})", "",
         "Projected: correlation, fitted slope, normalised RMSE after the fit, FRC 0.5 crossing. Depth: truth integrated "
         "over each reconstruction slab's physical interval; slab corr = mean in-slab correlation over slabs containing "
         "potential; profile corr = correlation of lateral-std-per-A profiles. Features: found from the reconstruction "
         f"(projected peaks, then all column-profile peaks >= half max), matched in 3D (lateral tol {tol} A).", "",
         "| run | arm | probe | planes | loss | conv. | proj. corr | slope | NRMSE | FRC0.5 (1/A) | slab corr | profile corr | probe RMS rad (start -> end) | found/true (false) | e_z (A) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p, fp = r["projected"], r["features_all_peaks"]
        fz = ", ".join("missed" if x["e_z"] is None else f"{x['e_z']:+.1f}" for x in fp["features"])
        L.append(f"| {r['run']} | {r['arm']} | {r['probe_mode']}{' (oracle)' if r['oracle'] else ''} | {r['planes']} | "
                 f"{r['final_loss']:.4g} | {r['converged_rel_change']:.1e} | {p['corr']:.3f} | {p['slope']:.2f} | {p['nrmse_fit']:.2f} | "
                 f"{p['frc_half_crossing_inv_A'] or '-'} | {r['mean_slab_corr'] if r['mean_slab_corr'] is None else round(r['mean_slab_corr'], 3)} | "
                 f"{r['depth_profile_corr'] if r['depth_profile_corr'] is None else round(r['depth_profile_corr'], 3)} | "
                 f"{r['probe']['initial_rms_rad']:.3f} -> {r['probe']['final_rms_rad']:.3f} | "
                 f"{fp['n_detected']}/{fp['n_true']} ({fp['n_false']}) | {fz} |")
    open(os.path.join(out, f"QUANTEM_{specimen}.md"), "w").write("\n".join(L) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smatrix", "quantem"])
    ap.add_argument("--config", default=bench.DEFAULT_CONFIG)
    ap.add_argument("--specimen", default="A_z100")
    ap.add_argument("--match-tol", type=float, default=1.0, help="lateral matching tolerance [A]")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    if a.stage == "smatrix":
        eval_smatrix(cfg, a.specimen, a.match_tol)
    elif a.stage == "quantem":
        eval_quantem(cfg, a.specimen, a.match_tol)


if __name__ == "__main__":
    main()
