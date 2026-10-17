"""Benchmark reconstructions.  Reads only the datasets (nominal metadata), never ground_truth/: evaluation is separate
(evaluate_reconstruction.py).

    python run_benchmark.py smatrix --specimen A_z100 --device cuda                         # noiseless window data
    python run_benchmark.py smatrix --specimen A_z100 --noise poisson --seeds 0,1,2 --device cuda
    python run_benchmark.py smatrix --specimen A_z100 --epochs 2 --device cpu --tag smoke  # code-path check only
    python run_benchmark.py quantem --specimen A_z100 --device cuda                         # arms A-D, S-informed probe
    python run_benchmark.py quantem --specimen A_z100 --arms A,D --probes true_fixed,nominal_fixed,smatrix_fixed,\
        nominal_trained,smatrix_trained --device cuda                                       # probe separation

S-matrix stage (section 8): joint S-matrix + probe fit (simple_joint) on all three defoci by default, starting from the
nominal aberrations (dataset `reference` fixed: gauge). With --single-dataset, only --dataset is loaded and its
nominal probe stays fixed; the strict geometry check still applies. A random 10 % of scan positions is
held out; the training and held-out amplitude losses are both recorded.  Then refocusing over -0.2 T .. 1.2 T, lateral
feature detection (simple_joint.detect_features) and per-feature depth from the refocused response (peak of
|arg(1 + r(z))|, no templates, no truth), each candidate saved with its response profile, peak contrast, predicted
uncertainty and a reliability flag.  Outputs: runs/<config>/<specimen>/smatrix_<variant>_<noise>_s<seed>[_<tag>]/
(S is large and stays out of git; summary.json and small arrays are copied to results/smatrix/).
QuantEM --single-dataset fits only --dataset without S-matrix priors, using uniform slices and a nominal probe.

Single-focus probe refinement (experimental, opt-in): `smatrix --single-dataset --single-probe-mode joint` updates the
--trainable aberrations of the one dataset together with S after the warm-up (JointModel ref=None: no gauge dataset).
The default `fixed` keeps the nominal probe fixed.  One dataset cannot separate the probe phase from S (any per-beam
phase moves between them without changing a pattern), so the probe this reaches is selected by the start, the
optimiser and the schedule, not by the data.  --probe-init C10=-120,... sets the starting coefficients (absolute, A).
--select-every N picks the best epoch from a fresh full-training-set loss every N epochs (held-out loss is recorded
at the same epochs but never used for selection).  Output names end in `_single_d<i>_joint`.
"""
import argparse, json, math, os, shutil, time

import numpy as np
import torch

import bench

sj = bench.sj


def load_variant(cfg, name, variant, noise, datasets=None):
    """Patterns (D, Ny, Nx, K, K) in FFT order, scan positions [S-grid px, first at K/2], cell offset [px], metadata."""
    import zarr
    import run_realization as rr
    from types import SimpleNamespace
    setup = bench.make_setup(cfg)
    sd = bench.specimen_dir(cfg, name)
    dps, nominal_focus = [], []
    datasets = range(len(cfg["defocus"]["nominal_C10"])) if datasets is None else datasets
    for d in datasets:
        ddir = os.path.join(sd, f"{variant}_{noise}", f"def{d}")
        x = rr.load_dataset(ddir, SimpleNamespace(energy=None, alpha=None, defocus=None), d)
        if x["data"].shape[-1] != setup.K or abs(x["dk"][0] - 1 / (setup.K * setup.dr)) > 1e-9:
            raise ValueError(f"{ddir}: detector {x['data'].shape[-1]} px at {x['dk'][0]} 1/A does not match K = {setup.K}")
        cen = rr.centre_patterns(x["data"], x["zero"], 1)              # zero frequency -> K/2 (exact for 'window')
        dps.append(np.fft.ifftshift(cen, axes=(-2, -1)))
        nominal_focus.append(x["defocus"])
        pos_A = x["positions_A"]
    pos_px = np.round(pos_A / setup.dr).astype(int)
    if np.abs(pos_px * setup.dr - pos_A).max() > 1e-6:
        raise ValueError("scan positions are not on the S-grid")
    offset = pos_px[0, 0] - setup.K // 2                               # cell pixel of S-grid pixel 0 (per axis)
    return np.array(dps, np.float32), pos_px - offset, offset.astype(int), np.array(nominal_focus)


def _json(o):
    return o.item() if hasattr(o, "item") else str(o)


def heldout_mask(shape, frac, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros(int(np.prod(shape)), bool)
    m[rng.choice(m.size, int(round(frac * m.size)), replace=False)] = True
    return m.reshape(shape)


@torch.no_grad()
def amp_loss(model, amp, idx, batch):
    num = den = 0.0
    for b in torch.as_tensor(idx, device=amp.device).split(batch):
        Z = model(b)
        num += float(((Z.abs() - amp[b]) ** 2).sum())
        den += float((amp[b].double() ** 2).sum())
    return num / den


def smatrix(cfg, a, log=print):
    import run_realization as rr
    setup = bench.make_setup(cfg)
    T = float(cfg["specimen"]["thickness_A"])
    datasets = [a.dataset] if a.single_dataset else list(range(len(cfg["defocus"]["nominal_C10"])))
    dps, pos, offset, nominal_focus = load_variant(cfg, a.specimen, a.variant, a.noise, datasets=datasets)
    D, ny, nx = dps.shape[:3]
    Ys = int(pos.max() + setup.K - setup.K // 2)                      # minimal S grid covering every window
    setup = sj.Setup(energy_ev=setup.energy_ev, alpha=setup.alpha, dr=setup.dr, K=setup.K, Y=Ys, taper=setup.taper)
    nominal = np.zeros((D, len(sj.NAMES)))
    nominal[:, 0] = -nominal_focus                                     # metadata defocus = -C10 (nominal only)
    joint_single = a.single_dataset and a.single_probe_mode == "joint"
    ref = None if joint_single else (0 if a.single_dataset else int(cfg["defocus"]["reference"]))
    start = nominal.copy()
    for item in a.probe_init:
        start[:, sj.NAMES.index(item[0])] = item[1]
    if a.single_dataset and not joint_single:
        log(f"Single dataset {a.dataset}: the nominal probe is fixed; S and probe cannot be independently recovered.")
    if joint_single:
        log(f"Single dataset {a.dataset}, EXPERIMENTAL joint probe mode: {a.trainable} updated with S after "
            f"{a.warmup} warm-up epochs (lr_probe {a.lr_probe or 'fit default'}); the probe phase is not identifiable from one dataset.")
    if a.probe_init:
        log(f"starting probe coefficients {dict(zip(sj.NAMES, np.round(start[0], 3)))} (nominal "
            f"{dict(zip(sj.NAMES, np.round(nominal[0], 3)))})")
    checks = sj.check_geometry(setup, pos, nominal, T, abtem=True, strict=not a.allow_underdetermined, log=log)
    if a.allow_underdetermined and not all(c["ok"] for c in checks):
        log("geometry check failed but --allow-underdetermined is set: fitting anyway (expect a degraded S)")
    hold = heldout_mask((ny, nx), a.heldout, a.split_seed)
    amp, P, didx = sj.prepare_data(dps, pos, device=a.device)
    hold_j = np.tile(hold.ravel(), D)
    train_idx, test_idx = np.flatnonzero(~hold_j), np.flatnonzero(hold_j)
    results = []
    for seed in a.seeds:
        tag = f"smatrix_{a.variant}_{a.noise}_s{seed}" + (f"_{a.tag}" if a.tag else "")
        if a.single_dataset:
            tag += f"_single_d{a.dataset}" + ("_joint" if joint_single else "")
        out = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, tag)
        os.makedirs(out, exist_ok=True)
        torch.manual_seed(seed)
        model = sj.JointModel(setup, P, didx, start, trainable=tuple(a.trainable.split(",")), ref=ref, device=a.device)
        if a.init_noise > 0:                                           # sensitivity to initialisation
            g = torch.Generator(device="cpu").manual_seed(1000 + seed)
            with torch.no_grad():
                z = torch.randn(model.S.shape, generator=g, dtype=torch.float32) + 1j * torch.randn(model.S.shape, generator=g, dtype=torch.float32)
                model.S.add_((a.init_noise / math.sqrt(2) * z).to(model.S.device))
        amp_train = amp                                                # fit() sees only the training rows below
        t0 = time.time()
        evals = []
        select_fn = None
        if a.select_every > 0:
            def select_fn(m):
                tr_l = amp_loss(m, amp, train_idx, a.eval_batch)
                te_l = amp_loss(m, amp, test_idx, a.eval_batch) if len(test_idx) else float("nan")
                evals.append(dict(epoch=len(m.fit_trace["epoch"]) + 1, train=tr_l, heldout=te_l))
                return tr_l
        hist = fit_subset(model, amp_train, train_idx, a, seed, log, select_fn=select_fn,
                          checkpoint=os.path.join(out, "checkpoint.pt") if a.checkpoint_every > 0 else None,
                          on_epoch=make_monitor(out, setup, offset, T, a) if a.monitor_every > 0 else None)
        fit_time = time.time() - t0
        loss_train = amp_loss(model, amp, train_idx, a.batch)
        loss_test = amp_loss(model, amp, test_idx, a.batch) if len(test_idx) else float("nan")
        coeff = model.coefficients().detach().cpu().numpy()
        trace = getattr(model, "fit_trace", {})
        upd = None
        if joint_single:                                               # probe updates really happen after warm-up
            post = [i for i, e in enumerate(trace["epoch"]) if e > a.warmup]
            gn = [trace["probe_grad_norm_mean"][i] for i in post]
            moved = np.abs(np.array(trace["coefficients"])[post] - start[None]).max(axis=(1, 2)) if post else np.array([])
            upd = dict(epochs_with_probe_updates=len(post),
                       all_grad_finite=bool(post) and all(trace["probe_grad_finite"][i] for i in post),
                       grad_norm_mean_first=gn[0] if gn else None, grad_norm_mean_last=gn[-1] if gn else None,
                       grad_nonzero=bool(gn) and all(g is not None and g > 0 for g in gn),
                       max_coefficient_change_A=float(moved.max()) if len(moved) else 0.0,
                       final_change_A=dict(zip(sj.NAMES, (coeff[0] - start[0]).tolist())),
                       coefficients_changed=bool(len(moved) and moved.max() > 1e-6))
            upd["ok"] = bool(upd["all_grad_finite"] and upd["grad_nonzero"] and upd["coefficients_changed"])
            log(f"[{tag}] probe update check: {upd}")
        log(f"[{tag}] best epoch {getattr(model, 'best_epoch', len(hist))} used; {len(hist)} of {a.epochs} epochs {fit_time:.0f} s: train loss {loss_train:.3e}, held-out {loss_test:.3e}; C10 "
            f"{np.round(coeff[:, 0], 2).tolist()} C12a {np.round(coeff[:, 1], 2).tolist()} C12b {np.round(coeff[:, 2], 2).tolist()}")
        # refocus and depth candidates (no ground truth)
        dz = a.dz
        zg = np.arange(-0.2 * T, 1.2 * T + dz / 2, dz)
        vol = sj.refocus(model.S, setup, zg, device=a.device)
        c0, c1 = setup.K // 2, Ys - setup.K // 2 + 1
        curve = sj.FOCUS_METRICS["phase_var"](vol[:, c0:c1, c0:c1])
        inside = (zg >= 0) & (zg <= T)
        det, strength, thr = sj.detect_features(vol, setup, k=a.detect_k)
        cands = []
        for (y, x), st in zip(det[:a.max_features], strength[:a.max_features]):
            r = sj.feature_response(vol, sj.footprint(setup, [(y, x)]))
            prof = np.abs(np.angle(1 + r))
            pin = np.where(inside, prof, 0)
            i = int(np.argmax(pin))
            z = zg[i] + (rr.parabolic(pin, i) * dz if 0 < i < len(zg) - 1 else 0.0)
            out_of_slab = float(prof[~inside].max()) if (~inside).any() else 0.0
            contrast = float(pin[i] / max(np.median(pin[inside]), 1e-12))
            half = pin[i] / 2
            j0, j1 = i, i
            while j0 > 0 and pin[j0 - 1] >= half:
                j0 -= 1
            while j1 < len(pin) - 1 and pin[j1 + 1] >= half:
                j1 += 1
            fwhm = float(zg[j1] - zg[j0])
            near_surface = bool(z < 0.1 * T or z > 0.9 * T)
            peaks = rr.local_peaks(zg[inside], pin[inside], 4, frac=0.5, min_sep=0.5 * setup.lam / setup.alpha ** 2)
            cands.append(dict(
                y=float(y) + offset[0] * setup.dr, x=float(x) + offset[1] * setup.dr,    # cell frame [A], axis 0 first
                z=float(np.clip(z, 0, T)), amplitude=float(pin[i]), detection_strength=float(st), peak_contrast=contrast,
                response_fwhm_A=fwhm, uncertainty_A=float(max(fwhm / 2.355, dz)) if not near_surface else T / 4,
                peak_outside_slab=bool(out_of_slab > pin[i]), near_surface=near_surface,
                reliable=bool(contrast > a.reliable_contrast and not near_surface and out_of_slab <= pin[i]),
                z_peaks=[float(v) for v in peaks],                  # all depth hypotheses (>= half the main peak)
                profile=prof.astype(np.float32).tolist()))
        layers = rr.local_peaks(zg[inside], curve[inside], 5, frac=0.25, min_sep=0.5 * setup.lam / setup.alpha ** 2)
        log(f"[{tag}] {len(cands)} lateral features (threshold {thr:.3g}); depths "
            f"{np.round([c['z'] for c in cands], 1).tolist()} A; focus-curve layers {np.round(layers, 1).tolist()} A")
        # probes on the beams (for evaluation against the true probes)
        psi = model.probe().detach().cpu().numpy()
        np.save(os.path.join(out, "S.npy"), model.S.detach().cpu().numpy())
        np.save(os.path.join(out, "refocused_phase.npy"),
                np.angle(vol / vol.mean(axis=(1, 2), keepdims=True))[:, c0:c1, c0:c1].astype(np.float32))
        np.save(os.path.join(out, "refocused_amplitude.npy"), np.abs(vol)[:, c0:c1, c0:c1].astype(np.float32))
        np.savez(os.path.join(out, "arrays.npz"), depth_grid=zg, focus_curve=curve, loss=np.array(hist), probe_beams=psi,
                 beams_ky=setup.beams()[0], beams_kx=setup.beams()[1], heldout=hold)
        summ = dict(
            stage="smatrix", specimen=a.specimen, variant=a.variant, noise=a.noise, seed=seed, tag=a.tag, config=cfg["_path"],
            config_sha256=cfg["_sha256"], versions=bench.versions(), device=a.device,
            setup=dict(energy_ev=setup.energy_ev, alpha=setup.alpha, taper=setup.taper, dr=setup.dr, K=setup.K, Y=setup.Y),
            s_grid_offset_px=offset.tolist(), crop=[c0, c1], reference_dataset=None if ref is None else datasets[ref],
            datasets=datasets, single_dataset=a.single_dataset,
            trainable=a.trainable if (joint_single or not a.single_dataset) else "", epochs=a.epochs,
            single_probe_mode=a.single_probe_mode if a.single_dataset else None,
            experimental=("joint S + probe from one dataset: probe phase not identifiable; coefficient movement or "
                          "lower loss is not evidence of probe recovery") if joint_single else None,
            warmup=a.warmup, lr_probe=a.lr_probe if a.lr_probe is not None else (0.005 if a.mode == "simultaneous" else 0.05),
            probe_trained=bool(joint_single or not a.single_dataset), start_coefficients=start.tolist(),
            probe_init_override={k: v for k, v in a.probe_init}, probe_update_check=upd,
            selection=dict(method="fresh full-training-set amplitude loss" if a.select_every > 0 else
                           "running epoch loss", select_every=a.select_every, best_epoch=getattr(model, "best_epoch", None),
                           restore_best=not a.no_restore_best, evaluations=evals, patience=a.patience),
            fit_trace=trace,
            batch=a.batch, lr_S=a.lr_S, mode=a.mode, init_noise=a.init_noise, heldout_fraction=a.heldout,
            split_seed=a.split_seed, n_train=int(len(train_idx)), n_heldout=int(len(test_idx)),
            loss_train=loss_train, loss_heldout=loss_test, loss_history=list(map(float, hist)), fit_time_s=fit_time,
            names=sj.NAMES, nominal=nominal.tolist(), coefficients=coeff.tolist(), geometry=checks,
            depth=dict(dz=dz, layers=list(map(float, layers)), detection_threshold=thr, detect_k=a.detect_k,
                       reliable_contrast=a.reliable_contrast, candidates=cands,
                       method="peak of |arg(1 + feature_response(z))| inside 0..T, parabolic refinement; no templates"))
        json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=_json)
        dst = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen, tag)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(out, "summary.json"), dst)
        np.savez_compressed(os.path.join(dst, "arrays.npz"), **np.load(os.path.join(out, "arrays.npz")))
        results.append(out)
    return results

PROBE_MODES = {                  # name: (initial probe, trainable)
    "true_fixed": ("true", False),            # 1  oracle baseline (reads ground_truth/, labelled oracle)
    "nominal_fixed": ("nominal", False),      # 2
    "smatrix_fixed": ("smatrix", False),      # 3
    "nominal_trained": ("nominal", True),     # 4
    "smatrix_trained": ("smatrix", True),     # 5
}
ARMS = {"A": ("uniform", False), "B": ("uniform", True), "C": ("smatrix", False), "D": ("smatrix", True),
        "E": ("smatrix", "object"),   # E: initial object = the S-matrix's own reconstruction (smatrix_object_init)
        "F": ("uniform", "object")}   # F: as E on the uniform grid (single-focus handoff Q1/Q3)
E_SCALES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
E_SCALES_EXTRA = (8.0, 12.0, 16.0, 24.0, 32.0)   # tried in order only while the largest scale so far is the best


def smatrix_object_init(cfg, a, grid, shape, j0, T):
    """Initial QuantEM potential from the S-matrix reconstruction (arm E), before scaling: (N, *shape) rad.

    Lateral: the S-matrix projected-potential estimate = phase of the refocused wave at mid-depth over the scan field
    (runs/.../refocused_phase.npy, S-grid crop [K/2, Y - K/2], pixel 0 = first scan position = QuantEM pixel j0; same
    0.2 A pixel), 2-D unwrapped, minimum (1st percentile) set to 0, clipped >= 0 (QuantEM positivity).  Outside the scan
    field: edge values.  Depth: slab k gets the fraction of the S-matrix focus curve (arrays.npz) inside it.  The
    absolute scale is chosen afterwards from the data (QuantEM forward loss, E_SCALES), never from the ground truth."""
    from skimage.restoration import unwrap_phase
    run = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, a.smatrix_run)
    arr = np.load(os.path.join(run, "arrays.npz"))
    zg, curve = arr["depth_grid"], arr["focus_curve"]
    ph = np.load(os.path.join(run, "refocused_phase.npy"), mmap_mode="r")
    P = unwrap_phase(np.asarray(ph[int(np.argmin(np.abs(zg - T / 2)))], float))
    P = np.clip(P - np.percentile(P, 1), 0.0, None)
    zf = np.linspace(0.0, T, 4001)
    cf = np.interp(zf, zg, curve)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (cf[1:] + cf[:-1]) * np.diff(zf))])
    w = np.diff(np.interp(grid.boundaries, zf, cum)) / cum[-1]
    n = P.shape[0]
    lo = int(round(j0))
    if lo < 0 or lo + n > min(shape):
        raise ValueError(f"S-matrix field ({n} px at {lo}) does not fit the QuantEM object {shape}")
    full = np.pad(P, ((lo, shape[0] - lo - n), (lo, shape[1] - lo - n)), mode="edge")
    info = dict(source=os.path.relpath(run, bench.HERE), mid_depth_A=float(zg[int(np.argmin(np.abs(zg - T / 2)))]),
                projected_max_rad=float(P.max()), depth_weights=w.tolist())
    return w[:, None, None] * full[None], info


def smatrix_training_patterns(cfg, a, d0, pt, origin, dr_q):
    """QuantEM pattern indices of the S-matrix run's TRAINING scan positions (its held-out positions excluded), so the
    object scale is chosen without the S-matrix validation data.  None if the run held nothing out.  The QuantEM pattern
    order is matched to the scan grid through the positions themselves (checked to 1e-3 A)."""
    arr = np.load(os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen, a.smatrix_run, "arrays.npz"))
    hold = arr["heldout"] if "heldout" in arr.files else None
    if hold is None or not hold.any():
        return None
    grid_A = np.asarray(d0["positions_A"]).reshape(-1, 2)
    q_A = pt.dset.scan_positions_px.detach().cpu().numpy() * dr_q + np.asarray(origin)
    d = np.linalg.norm(q_A[:, None, :] - grid_A[None, :, :], axis=-1)
    j = d.argmin(axis=1)
    if d[np.arange(len(j)), j].max() > 1e-3 or len(set(j.tolist())) != len(j):
        raise ValueError("QuantEM scan positions do not map one-to-one onto the S-matrix scan grid")
    return np.flatnonzero(~hold.ravel()[j])


def _basis(setup):
    ky, kx = setup.beams()
    return sj.aberration_basis(torch.as_tensor(ky), torch.as_tensor(kx), setup.lam, sj.NAMES).numpy()


def probe_transfer_check(setup, p_quantem, dr_q, p_ref, target_beams, nominal):
    """QuantEM's initial probe vs the probe it was built from, at the S-matrix beams (no ground truth): overlap and
    RMS phase (piston/tilt removed), intensity relative to the nominal-probe reference QuantEM normalises to, and
    the defocus sign check: overlap with the target vs with the target's complex conjugate phase (a flipped C10 sign
    would swap them)."""
    import run_realization as rr
    if target_beams is None:
        return None
    ky, kx = setup.beams()
    q = rr.probe_at_beams(p_quantem, dr_q, ky, kx)
    t = np.asarray(target_beams)
    ov, rms = rr.probe_match(q, t, ky, kx)
    ov_flip, _ = rr.probe_match(q, np.abs(t) * np.exp(-1j * np.angle(t)), ky, kx)
    return dict(overlap=ov, rms_phase_rad=rms, overlap_sign_flipped=ov_flip, sign_ok=bool(ov > ov_flip),
                intensity_ratio_to_reference=float((np.abs(p_quantem) ** 2).sum() / (np.abs(p_ref) ** 2).sum()),
                nominal_C10=float(nominal[0]))


def quantem(cfg, a, log=print):
    """QuantEM multislice on one defocus dataset, arms A-D x probe modes, equal planes / iterations / optimiser."""
    import zarr
    import forward_check as fc
    import quantem_export as qe
    import run_realization as rr
    from types import SimpleNamespace
    from quantem.core.ml import OptimizerParams
    setup = bench.make_setup(cfg)
    T = float(cfg["specimen"]["thickness_A"])
    sd = bench.specimen_dir(cfg, a.specimen)
    qd = a.dataset
    ddir = os.path.join(sd, f"{a.variant}_{a.noise}", f"def{qd}")
    d0 = rr.load_dataset(ddir, SimpleNamespace(energy=None, alpha=None, defocus=None), qd)
    pdset = fc.setup_dataset(d0, setup.lam)
    if a.single_dataset:
        nominal_probe = np.zeros(len(sj.NAMES))
        nominal_probe[0] = -d0["defocus"]
        cands_all = []
    else:
        srun = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen, a.smatrix_run, "summary.json")
        sm = json.load(open(srun))
        datasets = sm.get("datasets", list(range(len(sm["coefficients"]))))
        if qd not in datasets:
            raise ValueError(f"dataset {qd} is not present in S-matrix run {a.smatrix_run}")
        if a.single_focus_handoff and list(datasets) != [qd]:
            raise SystemExit(f"--single-focus-handoff: S-matrix run {a.smatrix_run} used datasets {datasets}, not [{qd}]")
        probe_index = datasets.index(qd)
        smatrix_probe_beams = np.load(os.path.join(os.path.dirname(srun), "arrays.npz"))["probe_beams"][probe_index]
        nominal_probe = np.array(sm["nominal"])[probe_index]
        smatrix_probe = np.array(sm["coefficients"])[probe_index]
        cands_all = sm["depth"]["candidates"]
    cands = [c for c in cands_all if c["reliable"]] if a.candidates == "reliable" else cands_all
    dcands = [qe.DepthCandidate(y=c["y"], x=c["x"], modes=[(c["z"], max(c["uncertainty_A"], 2.0), 1.0)],
                                phase=float(np.clip(c["amplitude"], 0.05, 1.0))) for c in cands]
    dz_max, dz_min = a.dz_max or T / 12, a.dz_min or T / 40
    g_s = qe.slice_grid(dcands, T, dz_max=dz_max, dz_min=dz_min, extra_slices=a.extra_slices) if dcands else None
    n = g_s.num_slices if g_s is not None else int(math.ceil(T / dz_max))
    if a.num_slices is not None:
        n = a.num_slices
    g_u = qe.SliceGrid(boundaries=np.linspace(0.0, T, n + 1), thickness=T)
    g_s = g_s or g_u
    K = setup.K
    probe_arrays = {"nominal": bench.ad.entrance_probe(setup, nominal_probe)}
    if not a.single_dataset:
        probe_arrays["smatrix"] = bench.ad.entrance_probe(setup, smatrix_probe)
    if any(PROBE_MODES[m][0] == "true" for m in a.probes):
        import zarr                                            # ORACLE: the true probe, used only in mode true_fixed
        probe_arrays["true"] = zarr.open_array(os.path.join(sd, "ground_truth", f"probe_d{qd}.zarr", "array0"), mode="r")[...]
    tmp = fc.make_pt(pdset, d0, qe.to_quantem(np.full((2, 8, 8), 1e-3, np.float32),
                                             qe.SliceGrid(boundaries=np.array([0, T / 2, T]), thickness=T)),
                     fc.truth_probe_on_grid((probe_arrays["nominal"], setup.dr), (K, K), setup.dr, np.ones((K, K))), a.qdevice)
    shape = tuple(int(v) for v in tmp.obj_shape_full[1:])
    dr_q = float(np.asarray(tmp.sampling)[0])
    p0 = tmp.dset.scan_positions_px.detach().cpu().numpy().min(axis=0)
    pos0 = d0["positions_A"][0, 0]
    origin = (pos0[0] - p0[0] * dr_q, pos0[1] - p0[1] * dr_q)      # cell coordinate of object pixel 0 (validated)
    ref = np.asarray(fc.make_pt(pdset, d0, qe.to_quantem(np.full((2,) + shape, 1e-3, np.float32),
                                                         qe.SliceGrid(boundaries=np.array([0, T / 2, T]), thickness=T)),
                                fc.truth_probe_on_grid((probe_arrays["nominal"], setup.dr), (K, K), setup.dr,
                                                       np.ones((K, K))), a.qdevice).probe)[0]
    grid_label = "uniform" if a.single_dataset else "S-informed"
    log(f"QuantEM: dataset {qd} ({a.variant}, {a.noise}), object {shape} at {dr_q:.3f} A, {n} planes; candidates "
        f"{len(cands)}/{len(cands_all)} ({a.candidates}); {grid_label} slabs {np.round(g_s.slab_thicknesses, 1).tolist()} A")
    n_pat = len(tmp.dset.scan_positions_px)
    val_idx = fc.quantem_val_indices(n_pat, a.val_ratio) if a.val_ratio > 0 else np.zeros(0, int)
    if a.val_ratio > 0:
        log(f"QuantEM held-out patterns: {len(val_idx)}/{n_pat} (grid, every {round(1 / a.val_ratio)}th)")
    runs = []
    for arm in a.arms:
        slices, informed = ARMS[arm]
        grid = g_s if slices == "smatrix" else g_u
        for pm in a.probes:
            init, trained = PROBE_MODES[pm]
            for seed in a.seeds:
                tag = f"quantem_{a.variant}_{a.noise}_d{qd}_{arm}_{pm}_c{a.candidates}_s{seed}" + (f"_{a.tag}" if a.tag else "")
                if a.val_ratio > 0:
                    tag += f"_val{a.val_ratio:g}"
                if a.single_dataset:
                    tag += "_single"
                out = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, tag)
                os.makedirs(out, exist_ok=True)
                torch.manual_seed(seed)
                np.random.seed(seed)
                phi = np.full((grid.num_slices,) + shape, 1e-3)
                if informed is True and dcands:
                    phi += qe.initial_potential(dcands, grid, shape, dr_q, origin=origin)
                prb = fc.truth_probe_on_grid((probe_arrays[init], setup.dr), ref.shape, dr_q, ref)
                init_info = None
                if informed == "object":
                    if a.single_dataset:
                        raise SystemExit("arms E/F need an S-matrix run (not --single-dataset)")
                    base, init_info = smatrix_object_init(cfg, a, grid, shape, p0.min(), T)
                    subset = smatrix_training_patterns(cfg, a, d0, tmp, origin, dr_q)
                    if len(val_idx):                      # never score the scale on QuantEM's held-out patterns
                        subset = np.setdiff1d(np.arange(n_pat) if subset is None else subset, val_idx)
                    trial = {}
                    for sc in E_SCALES + E_SCALES_EXTRA:  # scale from the data: QuantEM loss of the initial object
                        if sc in E_SCALES_EXTRA and min(trial, key=trial.get) != max(trial):
                            break                         # extend the grid only while its top is the best
                        ptt = fc.make_pt(pdset, d0, qe.to_quantem((phi + sc * base).astype(np.float32), grid), prb,
                                         a.qdevice)
                        trial[sc] = fc.predict(ptt, np.ones(tuple(ptt.dset.detector_mask.shape)),
                                               subset=subset)["quantem_loss"]
                        del ptt
                    best = min(trial, key=trial.get)
                    init_info.update(scale_patterns="all patterns" if subset is None else
                                     "S-matrix training positions, minus QuantEM held-out" if len(val_idx) else
                                     "S-matrix training positions only", n_scale_patterns=None if subset is None else len(subset))
                    phi = phi + best * base
                    init_info.update(scale=best, scale_losses={str(k): v for k, v in trial.items()})
                    log(f"arm {arm}: S-matrix object init, loss by scale {({k: round(v, 3) for k, v in trial.items()})}; "
                        f"scale {best}")
                pt = fc.make_pt(pdset, d0, qe.to_quantem(phi.astype(np.float32), grid), prb, a.qdevice,
                                val_ratio=a.val_ratio)
                p_init = np.asarray(pt.probe)[0].copy()
                transfer = None if a.single_dataset else probe_transfer_check(setup, p_init, dr_q, ref,
                                                smatrix_probe_beams if init == "smatrix" else
                                                (setup.aperture(np.hypot(*setup.beams()))
                                                 * np.exp(-1j * (nominal_probe @ _basis(setup)))) if init == "nominal"
                                                else None, nominal_probe)
                log(f"probe transfer ({init}): {transfer}")
                cons = {"object": {"positivity": True}}
                warm = min(a.probe_warmup if trained else a.iters, a.iters)
                t0 = time.time()
                if warm > 0:
                    pt.reconstruct(num_iters=warm, reset=True, batch_size=a.qbatch, device=a.qdevice,
                                   optimizer_params={"object": OptimizerParams.Adam(lr=a.qlr)}, constraints=cons)
                if a.iters > warm:
                    pt.reconstruct(num_iters=a.iters - warm, reset=warm <= 0, batch_size=a.qbatch, device=a.qdevice,
                                   optimizer_params={"object": OptimizerParams.Adam(lr=a.qlr),
                                                     "probe": OptimizerParams.Adam(lr=a.probe_lr)}, constraints=cons)
                dt = time.time() - t0
                obj = pt.obj_model.obj.detach().cpu().numpy()
                losses = np.asarray(pt.iter_losses, float)
                val_losses = np.asarray(pt.val_iter_losses, float)
                probe = np.asarray(pt.probe)[0]
                np.savez_compressed(os.path.join(out, "quantem.npz"), object_potential=obj.astype(np.float32),
                                    initial_potential=phi.astype(np.float32), slice_boundaries=grid.boundaries,
                                    probe=probe.astype(np.complex64), initial_probe=p_init.astype(np.complex64), loss=losses,
                                    val_loss=val_losses, val_indices=val_idx)
                tail = losses[-max(5, len(losses) // 10):]
                summ = dict(stage="quantem", specimen=a.specimen, arm=arm, slices=slices, informed_potential=informed,
                            object_init=init_info, probe_transfer=transfer,
                            smatrix_datasets=None if a.single_dataset else list(datasets),
                            probe_mode=pm, probe_init=init, probe_trained=trained, oracle=init == "true",
                            candidates=a.candidates, n_candidates=len(cands), dataset=qd, variant=a.variant, noise=a.noise,
                            seed=seed, iters=a.iters, probe_warmup=warm, object_lr=a.qlr, probe_lr=a.probe_lr if trained else 0,
                            batch=a.qbatch, planes=grid.num_slices, slice_boundaries=grid.boundaries.tolist(),
                            object_shape=list(shape), dr_quantem=dr_q, first_scan_pixel=p0.tolist(), origin_A=list(origin),
                            final_loss=float(losses[-1]), loss_history=losses.tolist(),
                            val_ratio=a.val_ratio, val_mode="grid" if a.val_ratio > 0 else None, n_val=len(val_idx),
                            n_train=n_pat - len(val_idx),
                            final_val_loss=float(val_losses[-1]) if len(val_losses) else None,
                            best_val_iter=int(val_losses.argmin()) + 1 if len(val_losses) else None,
                            val_loss_history=val_losses.tolist(),
                            converged_rel_change_last10pct=float((tail.max() - tail.min()) / max(abs(tail.mean()), 1e-30)),
                            time_s=dt, smatrix_run=None if a.single_dataset else a.smatrix_run,
                            single_dataset=a.single_dataset, config=cfg["_path"], config_sha256=cfg["_sha256"],
                            note=("one defocus dataset; no S-matrix or ground-truth priors" if a.single_dataset else
                                  "one defocus dataset with S-matrix priors; "
                                  "installed QuantEM 0f60f90 has no multi-dataset reconstruction"))
                json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=_json)
                dst = os.path.join(bench.HERE, "results", "quantem", cfg["name"], a.specimen, tag)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(os.path.join(out, "summary.json"), dst)
                log(f"[{tag}] {grid.num_slices} planes, loss {losses[0]:.4g} -> {losses[-1]:.4g} "
                    f"(last-10% change {summ['converged_rel_change_last10pct']:.1e}), {dt:.0f} s"
                    + (f"; held-out {val_losses[-1]:.4g} (best at iteration {summ['best_val_iter']})"
                       if len(val_losses) else ""))
                runs.append(out)
    return runs


def make_monitor(out, setup, offset, T, a):
    """Per-epoch snapshot for live monitoring (plot_fit_monitor.py): loss, probe coefficients, and the S-matrix refocused
    through depth (focus curve on a 4 A grid, phase sections every 20 A over the scanned field, focus-curve layers).
    Small files (~0.5 MB) in <run>/monitor/epoch_NNN.npz; computed on the fit's device without gradients."""
    import run_realization as rr
    mdir = os.path.join(out, "monitor")
    os.makedirs(mdir, exist_ok=True)
    zg = np.arange(-0.2 * T, 1.2 * T + 1e-6, 4.0)
    zs = np.arange(10.0, T, 20.0)
    c0, c1 = setup.K // 2, setup.Y - setup.K // 2 + 1
    inside = (zg >= 0) & (zg <= T)

    def cb(model, ep, history):
        if ep % a.monitor_every and ep != a.epochs:
            return
        t0 = time.time()
        vol = sj.refocus(model.S, setup, np.r_[zg, zs])
        curve = sj.FOCUS_METRICS["phase_var"](vol[:len(zg), c0:c1, c0:c1])
        sec = np.angle(vol[len(zg):, c0:c1, c0:c1]).astype(np.float32)
        mid = vol[int(np.argmin(np.abs(zg - T / 2))), c0:c1, c0:c1].astype(np.complex64)   # projected-potential estimate
        layers = rr.local_peaks(zg[inside], curve[inside], 5, frac=0.25, min_sep=0.5 * setup.lam / setup.alpha ** 2)
        np.savez(os.path.join(mdir, f"epoch_{ep:03d}.npz"), epoch=ep, loss=np.array(history),
                 coefficients=model.coefficients().detach().cpu().numpy(),
                 nominal=model.nominal.detach().cpu().numpy(), names=np.array(sj.NAMES),
                 zg=zg, curve=curve, zs=zs, sections=sec, wave_mid=mid, layers=np.array(layers), crop=np.array([c0, c1]),
                 offset_px=np.asarray(offset), dr=setup.dr, K=setup.K, Y=setup.Y, thickness=T,
                 seconds=time.time() - t0)
        del vol
    return cb


def fit_subset(model, amp, idx, a, seed, log, checkpoint=None, on_epoch=None, select_fn=None):
    """simple_joint.fit restricted to the training patterns (held-out patterns never enter the optimisation)."""
    sub = torch.as_tensor(idx, device=amp.device)
    pos, didx = model.pos, model.didx
    model.pos, model.didx = pos[sub], didx[sub]
    try:
        return sj.fit(model, amp[sub], epochs=a.epochs, warmup=a.warmup, batch=a.batch, lr_S=a.lr_S, mode=a.mode,
                      lr_probe=a.lr_probe, select_fn=None if select_fn is None else _full_index(model, pos, didx, select_fn),
                      select_every=max(a.select_every, 1),
                      seed=seed, log=(lambda m: log("  " + m)) if a.verbose else (lambda m: None),
                      checkpoint=checkpoint, checkpoint_every=a.checkpoint_every, resume=a.resume,
                      patience=a.patience, min_rel_improvement=a.min_rel_improvement, on_epoch=on_epoch,
                      best_path=os.path.join(os.path.dirname(checkpoint), "best.pt") if checkpoint else None,
                      restore_best=not a.no_restore_best)
    finally:
        model.pos, model.didx = pos, didx


def _full_index(model, pos, didx, fn):
    """Wrap a selection function so it sees the model with every pattern indexed (fit_subset narrows them)."""
    def wrapped(m):
        sp, sd = m.pos, m.didx
        m.pos, m.didx = pos, didx
        try:
            return fn(m)
        finally:
            m.pos, m.didx = sp, sd
    return wrapped


def _probe_init(s):
    out = []
    for item in filter(None, s.split(",")):
        k, v = item.split("=")
        if k not in sj.NAMES:
            raise argparse.ArgumentTypeError(f"unknown aberration {k}")
        out.append((k, float(v)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["smatrix", "quantem"])
    ap.add_argument("--config", default=bench.DEFAULT_CONFIG)
    ap.add_argument("--specimen", default="A_z100")
    ap.add_argument("--variant", default="window", choices=["window", "pixelated"])
    ap.add_argument("--noise", default="noiseless", choices=["noiseless", "poisson"])
    ap.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")], default=[0])
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--single-dataset", action="store_true",
                    help="use only --dataset; QuantEM uses no S priors, S-matrix keeps the nominal probe fixed")
    ap.add_argument("--dataset", type=int, default=1, help="defocus dataset for QuantEM or single-dataset S recovery")
    ap.add_argument("--allow-underdetermined", action="store_true",
                    help="S-matrix: fit even if the geometry check fails (e.g. one defocus dataset); "
                         "the failed check is still logged and saved in summary.json")
    ap.add_argument("--single-probe-mode", default="fixed", choices=["fixed", "joint"],
                    help="smatrix --single-dataset: keep the nominal probe fixed (default) or update it jointly with S "
                         "(experimental: not identifiable from one dataset)")
    ap.add_argument("--probe-init", type=_probe_init, default=[],
                    help="starting aberrations for single-dataset joint runs, e.g. C10=-120,C12a=3 [A, absolute]")
    ap.add_argument("--lr-probe", type=float, default=None, help="probe learning rate (default: simple_joint.fit's)")
    ap.add_argument("--select-every", type=int, default=0,
                    help="pick the best epoch by a fresh full-training-set loss every N epochs (0 = running loss)")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-S", dest="lr_S", type=float, default=0.02)
    ap.add_argument("--mode", default="simultaneous", choices=["simultaneous", "alternating"])
    ap.add_argument("--trainable", default="C10,C12a,C12b")
    ap.add_argument("--init-noise", type=float, default=0.0, help="complex Gaussian added to the vacuum S (init test)")
    ap.add_argument("--heldout", type=float, default=0.1, help="fraction of scan positions held out")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="epochs between checkpoints (0 = none)")
    ap.add_argument("--monitor-every", type=int, default=0, help="epochs between monitor snapshots (0 = none)")
    ap.add_argument("--no-restore-best", action="store_true", help="keep the last epoch instead of the lowest-loss one")
    ap.add_argument("--resume", action="store_true", help="continue from the run directory's checkpoint.pt")
    ap.add_argument("--patience", type=int, default=None, help="early stopping: epochs without improvement (None = off)")
    ap.add_argument("--min-rel-improvement", type=float, default=0.01, help="early stopping threshold (relative)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--dz", type=float, default=2.0)
    ap.add_argument("--detect-k", type=float, default=4.0)
    ap.add_argument("--max-features", type=int, default=50)
    ap.add_argument("--reliable-contrast", type=float, default=3.0, help="peak / median response for 'reliable'")
    ap.add_argument("--threads", type=int, default=4)
    q = ap.add_argument_group("quantem")
    q.add_argument("--smatrix-run", default="smatrix_window_noiseless_s0", help="results/smatrix/<config>/<specimen>/<run>")
    q.add_argument("--num-slices", type=int, default=None, help="uniform slice count (arms A, B, F)")
    q.add_argument("--single-focus-handoff", action="store_true",
                   help="refuse --smatrix-run unless it was fitted on --dataset alone")
    q.add_argument("--arms", type=lambda s: s.split(","), default=None)
    q.add_argument("--probes", type=lambda s: s.split(","), default=None,
                   help=f"comma list of {list(PROBE_MODES)}")
    q.add_argument("--candidates", default="all", choices=["all", "reliable"], help="confidence filtering of S priors")
    q.add_argument("--iters", type=int, default=200)
    q.add_argument("--qlr", type=float, default=5e-3)
    q.add_argument("--probe-lr", type=float, default=1e-4)
    q.add_argument("--probe-warmup", type=int, default=40, help="object-only iterations before the probe is trained")
    q.add_argument("--qbatch", type=int, default=64)
    q.add_argument("--val-ratio", type=float, default=0.0,
                   help="QuantEM held-out fraction (grid: every k-th pattern, k = 1/ratio); adds _val<ratio> to the run name")
    q.add_argument("--dz-max", type=float, default=None)
    q.add_argument("--dz-min", type=float, default=None)
    q.add_argument("--extra-slices", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = bench.load_config(a.config)
    a.arms = a.arms if a.arms is not None else (["A"] if a.single_dataset else list("ABCD"))
    a.probes = a.probes if a.probes is not None else (["nominal_trained"] if a.single_dataset else ["smatrix_trained"])
    if (a.single_dataset or a.stage == "quantem") and not 0 <= a.dataset < len(cfg["defocus"]["nominal_C10"]):
        ap.error("--dataset is outside the configured defocus datasets")
    if a.single_dataset and a.stage == "quantem":
        if a.arms != ["A"] or any(probe not in ("nominal_fixed", "nominal_trained") for probe in a.probes):
            ap.error("single-dataset QuantEM requires --arms A and nominal_fixed/nominal_trained probes (no priors)")
    if not 0 <= a.val_ratio <= 0.5:
        ap.error("--val-ratio must be in [0, 0.5]")
    if a.num_slices is not None:
        if a.stage != "quantem" or a.num_slices < 1:
            ap.error("--num-slices must be positive and requires quantem")
    if a.single_probe_mode != "fixed" and not (a.stage == "smatrix" and a.single_dataset):
        ap.error("--single-probe-mode joint requires smatrix --single-dataset")
    if a.probe_init and a.single_probe_mode != "joint":
        ap.error("--probe-init requires --single-probe-mode joint")
    a.qdevice = a.device if a.device.startswith("cuda") else "cpu"    # ProbePixelated.from_array needs a torch name
    for m in getattr(a, "probes", []):
        if m not in PROBE_MODES:
            raise SystemExit(f"unknown probe mode {m}; choose from {list(PROBE_MODES)}")
    if a.stage == "smatrix":
        smatrix(cfg, a)
    elif a.stage == "quantem":
        quantem(cfg, a)


if __name__ == "__main__":
    main()
