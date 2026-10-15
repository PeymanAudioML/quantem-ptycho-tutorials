"""Benchmark reconstructions.  Reads only the datasets (nominal metadata), never ground_truth/: evaluation is separate
(evaluate_reconstruction.py).

    python run_benchmark.py smatrix --specimen A_z100 --device cuda                         # noiseless window data
    python run_benchmark.py smatrix --specimen A_z100 --noise poisson --seeds 0,1,2 --device cuda
    python run_benchmark.py smatrix --specimen A_z100 --epochs 2 --device cpu --tag smoke  # code-path check only
    python run_benchmark.py quantem --specimen A_z100 --device cuda                         # arms A-D, S-informed probe
    python run_benchmark.py quantem --specimen A_z100 --arms A,D --probes true_fixed,nominal_fixed,smatrix_fixed,\
        nominal_trained,smatrix_trained --device cuda                                       # probe separation

S-matrix stage (section 8): joint S-matrix + probe fit (simple_joint) on all three defoci, starting from the nominal
aberrations (dataset `reference` fixed: gauge).  A random 10 % of the scan positions (the same in every dataset) is
held out; the training and held-out amplitude losses are both recorded.  Then refocusing over -0.2 T .. 1.2 T, lateral
feature detection (simple_joint.detect_features) and per-feature depth from the refocused response (peak of
|arg(1 + r(z))|, no templates, no truth), each candidate saved with its response profile, peak contrast, predicted
uncertainty and a reliability flag.  Outputs: runs/<config>/<specimen>/smatrix_<variant>_<noise>_s<seed>[_<tag>]/
(S is large and stays out of git; summary.json and small arrays are copied to results/smatrix/).
"""
import argparse, json, math, os, shutil, time

import numpy as np
import torch

import bench

sj = bench.sj


def load_variant(cfg, name, variant, noise):
    """Patterns (D, Ny, Nx, K, K) in FFT order, scan positions [S-grid px, first at K/2], cell offset [px], metadata."""
    import zarr
    import run_realization as rr
    from types import SimpleNamespace
    setup = bench.make_setup(cfg)
    sd = bench.specimen_dir(cfg, name)
    dps, nominal_focus = [], []
    for d in range(len(cfg["defocus"]["nominal_C10"])):
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
    dps, pos, offset, nominal_focus = load_variant(cfg, a.specimen, a.variant, a.noise)
    D, ny, nx = dps.shape[:3]
    Ys = int(pos.max() + setup.K - setup.K // 2)                      # minimal S grid covering every window
    setup = sj.Setup(energy_ev=setup.energy_ev, alpha=setup.alpha, dr=setup.dr, K=setup.K, Y=Ys, taper=setup.taper)
    nominal = np.zeros((D, len(sj.NAMES)))
    nominal[:, 0] = -nominal_focus                                     # metadata defocus = -C10 (nominal only)
    ref = int(cfg["defocus"]["reference"])
    checks = sj.check_geometry(setup, pos, nominal, T, abtem=True, strict=True, log=log)
    hold = heldout_mask((ny, nx), a.heldout, a.split_seed)
    amp, P, didx = sj.prepare_data(dps, pos, device=a.device)
    hold_j = np.tile(hold.ravel(), D)
    train_idx, test_idx = np.flatnonzero(~hold_j), np.flatnonzero(hold_j)
    results = []
    for seed in a.seeds:
        tag = f"smatrix_{a.variant}_{a.noise}_s{seed}" + (f"_{a.tag}" if a.tag else "")
        out = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, tag)
        os.makedirs(out, exist_ok=True)
        torch.manual_seed(seed)
        model = sj.JointModel(setup, P, didx, nominal, trainable=tuple(a.trainable.split(",")), ref=ref, device=a.device)
        if a.init_noise > 0:                                           # sensitivity to initialisation
            g = torch.Generator(device="cpu").manual_seed(1000 + seed)
            with torch.no_grad():
                z = torch.randn(model.S.shape, generator=g, dtype=torch.float32) + 1j * torch.randn(model.S.shape, generator=g, dtype=torch.float32)
                model.S.add_((a.init_noise / math.sqrt(2) * z).to(model.S.device))
        amp_train = amp                                                # fit() sees only the training rows below
        t0 = time.time()
        hist = fit_subset(model, amp_train, train_idx, a, seed, log)
        fit_time = time.time() - t0
        loss_train = amp_loss(model, amp, train_idx, a.batch)
        loss_test = amp_loss(model, amp, test_idx, a.batch) if len(test_idx) else float("nan")
        coeff = model.coefficients().detach().cpu().numpy()
        log(f"[{tag}] {a.epochs} epochs {fit_time:.0f} s: train loss {loss_train:.3e}, held-out {loss_test:.3e}; C10 "
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
            s_grid_offset_px=offset.tolist(), crop=[c0, c1], reference_dataset=ref, trainable=a.trainable, epochs=a.epochs,
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
ARMS = {"A": ("uniform", False), "B": ("uniform", True), "C": ("smatrix", False), "D": ("smatrix", True)}


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
    srun = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen, a.smatrix_run, "summary.json")
    sm = json.load(open(srun))
    coeff_s = np.array(sm["coefficients"])
    nominal = np.array(sm["nominal"])
    cands_all = sm["depth"]["candidates"]
    cands = [c for c in cands_all if c["reliable"]] if a.candidates == "reliable" else cands_all
    dcands = [qe.DepthCandidate(y=c["y"], x=c["x"], modes=[(c["z"], max(c["uncertainty_A"], 2.0), 1.0)],
                                phase=float(np.clip(c["amplitude"], 0.05, 1.0))) for c in cands]
    dz_max, dz_min = a.dz_max or T / 12, a.dz_min or T / 40
    g_s = qe.slice_grid(dcands, T, dz_max=dz_max, dz_min=dz_min, extra_slices=a.extra_slices) if dcands else None
    n = g_s.num_slices if g_s is not None else int(math.ceil(T / dz_max))
    g_u = qe.SliceGrid(boundaries=np.linspace(0.0, T, n + 1), thickness=T)
    g_s = g_s or g_u
    K = setup.K
    probe_arrays = {"nominal": bench.ad.entrance_probe(setup, nominal[qd]),
                    "smatrix": bench.ad.entrance_probe(setup, coeff_s[qd])}
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
    log(f"QuantEM: dataset {qd} ({a.variant}, {a.noise}), object {shape} at {dr_q:.3f} A, {n} planes; candidates "
        f"{len(cands)}/{len(cands_all)} ({a.candidates}); S-informed slabs {np.round(g_s.slab_thicknesses, 1).tolist()} A")
    runs = []
    for arm in a.arms:
        slices, informed = ARMS[arm]
        grid = g_s if slices == "smatrix" else g_u
        for pm in a.probes:
            init, trained = PROBE_MODES[pm]
            for seed in a.seeds:
                tag = f"quantem_{a.variant}_{a.noise}_d{qd}_{arm}_{pm}_c{a.candidates}_s{seed}" + (f"_{a.tag}" if a.tag else "")
                out = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen, tag)
                os.makedirs(out, exist_ok=True)
                torch.manual_seed(seed)
                np.random.seed(seed)
                phi = np.full((grid.num_slices,) + shape, 1e-3)
                if informed and dcands:
                    phi += qe.initial_potential(dcands, grid, shape, dr_q, origin=origin)
                prb = fc.truth_probe_on_grid((probe_arrays[init], setup.dr), ref.shape, dr_q, ref)
                pt = fc.make_pt(pdset, d0, qe.to_quantem(phi.astype(np.float32), grid), prb, a.qdevice)
                p_init = np.asarray(pt.probe)[0].copy()
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
                probe = np.asarray(pt.probe)[0]
                np.savez_compressed(os.path.join(out, "quantem.npz"), object_potential=obj.astype(np.float32),
                                    initial_potential=phi.astype(np.float32), slice_boundaries=grid.boundaries,
                                    probe=probe.astype(np.complex64), initial_probe=p_init.astype(np.complex64), loss=losses)
                tail = losses[-max(5, len(losses) // 10):]
                summ = dict(stage="quantem", specimen=a.specimen, arm=arm, slices=slices, informed_potential=informed,
                            probe_mode=pm, probe_init=init, probe_trained=trained, oracle=init == "true",
                            candidates=a.candidates, n_candidates=len(cands), dataset=qd, variant=a.variant, noise=a.noise,
                            seed=seed, iters=a.iters, probe_warmup=warm, object_lr=a.qlr, probe_lr=a.probe_lr if trained else 0,
                            batch=a.qbatch, planes=grid.num_slices, slice_boundaries=grid.boundaries.tolist(),
                            object_shape=list(shape), dr_quantem=dr_q, first_scan_pixel=p0.tolist(), origin_A=list(origin),
                            final_loss=float(losses[-1]), loss_history=losses.tolist(),
                            converged_rel_change_last10pct=float((tail.max() - tail.min()) / max(abs(tail.mean()), 1e-30)),
                            time_s=dt, smatrix_run=a.smatrix_run, config=cfg["_path"], config_sha256=cfg["_sha256"],
                            note=("one defocus dataset (fewer measurements than the three-defocus S-matrix fit); "
                                  "installed QuantEM 0f60f90 has no multi-dataset reconstruction"))
                json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=_json)
                dst = os.path.join(bench.HERE, "results", "quantem", cfg["name"], a.specimen, tag)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(os.path.join(out, "summary.json"), dst)
                log(f"[{tag}] {grid.num_slices} planes, loss {losses[0]:.4g} -> {losses[-1]:.4g} "
                    f"(last-10% change {summ['converged_rel_change_last10pct']:.1e}), {dt:.0f} s")
                runs.append(out)
    return runs


def fit_subset(model, amp, idx, a, seed, log):
    """simple_joint.fit restricted to the training patterns (held-out patterns never enter the optimisation)."""
    sub = torch.as_tensor(idx, device=amp.device)
    pos, didx = model.pos, model.didx
    model.pos, model.didx = pos[sub], didx[sub]
    try:
        return sj.fit(model, amp[sub], epochs=a.epochs, warmup=a.warmup, batch=a.batch, lr_S=a.lr_S, mode=a.mode,
                      seed=seed, log=(lambda m: log("  " + m)) if a.verbose else (lambda m: None))
    finally:
        model.pos, model.didx = pos, didx


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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-S", dest="lr_S", type=float, default=0.02)
    ap.add_argument("--mode", default="simultaneous", choices=["simultaneous", "alternating"])
    ap.add_argument("--trainable", default="C10,C12a,C12b")
    ap.add_argument("--init-noise", type=float, default=0.0, help="complex Gaussian added to the vacuum S (init test)")
    ap.add_argument("--heldout", type=float, default=0.1, help="fraction of scan positions held out")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--dz", type=float, default=2.0)
    ap.add_argument("--detect-k", type=float, default=4.0)
    ap.add_argument("--max-features", type=int, default=50)
    ap.add_argument("--reliable-contrast", type=float, default=3.0, help="peak / median response for 'reliable'")
    ap.add_argument("--threads", type=int, default=4)
    q = ap.add_argument_group("quantem")
    q.add_argument("--smatrix-run", default="smatrix_window_noiseless_s0", help="results/smatrix/<config>/<specimen>/<run>")
    q.add_argument("--dataset", type=int, default=1, help="defocus dataset for QuantEM (default: middle)")
    q.add_argument("--arms", type=lambda s: s.split(","), default=list("ABCD"))
    q.add_argument("--probes", type=lambda s: s.split(","), default=["smatrix_trained"],
                   help=f"comma list of {list(PROBE_MODES)}")
    q.add_argument("--candidates", default="all", choices=["all", "reliable"], help="confidence filtering of S priors")
    q.add_argument("--iters", type=int, default=200)
    q.add_argument("--qlr", type=float, default=5e-3)
    q.add_argument("--probe-lr", type=float, default=1e-4)
    q.add_argument("--probe-warmup", type=int, default=40, help="object-only iterations before the probe is trained")
    q.add_argument("--qbatch", type=int, default=64)
    q.add_argument("--dz-max", type=float, default=None)
    q.add_argument("--dz-min", type=float, default=None)
    q.add_argument("--extra-slices", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = bench.load_config(a.config)
    a.qdevice = "gpu" if a.device.startswith("cuda") else "cpu"
    for m in getattr(a, "probes", []):
        if m not in PROBE_MODES:
            raise SystemExit(f"unknown probe mode {m}; choose from {list(PROBE_MODES)}")
    if a.stage == "smatrix":
        smatrix(cfg, a)
    elif a.stage == "quantem":
        quantem(cfg, a)


if __name__ == "__main__":
    main()
