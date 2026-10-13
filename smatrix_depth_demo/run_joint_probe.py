"""Reproducible experiments for joint S-matrix / probe-aberration reconstruction (see README).

  python run_joint_probe.py exp --sample reduced --config B --variants fixed,joint,oracle --seeds 0,1,2,3,4
  python run_joint_probe.py exp --sample layers3 --config A --variants fixed,joint

Samples: layers3 / six_phi1p5 / poly reuse the existing structures (results*/layers_and_data.npz); reduced is a smaller
3-layer random-atom sample (128 x 128 S grid, same 10.4 A scan field) used for multi-seed statistics; reduced_s{n} sets
the scan step to n pixels (0.2 A each; reduced = reduced_s4).
Configs: A = data simulated with the nominal probe; B = relative defocus + two-fold astigmatism errors on the
non-reference datasets; Bcoma = B plus axial coma; G = the same error on ALL datasets incl. the reference (gauge demo).
"""
import argparse, json, math, os, sys, time
from dataclasses import asdict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_aberrations as pa
import joint_smatrix_probe as jp

OUT = os.path.join(HERE, "results_joint")
CACHE = os.path.join(OUT, "cache")
NAMES = pa.DEFAULT_NAMES


# --------------------------------------------------------------------------------------------- samples
def sample(name):
    """Returns dict(geom, ops, thick, defoci, kind, truth...)."""
    if name == "layers3":
        d = np.load(os.path.join(HERE, "results", "layers_and_data.npz"))
        g = jp.STANDARD
        return dict(geom=g, ops=jp.ops_layers(d["layers"], [30.0, 150.0, 270.0], 300.0, g), thick=300.0,
                    defoci=[-200.0, -100.0, 0.0], kind="layers", layers=d["layers"], layer_z=[30.0, 150.0, 270.0])
    if name.startswith("six_"):
        rd = os.path.join(HERE, "results_" + name)
        d = np.load(os.path.join(rd, "layers_and_data.npz")); cfg = json.load(open(os.path.join(rd, "config.json")))
        g = jp.STANDARD
        return dict(geom=g, ops=jp.ops_layers(d["layers"], cfg["layer_z"], 300.0, g), thick=300.0,
                    defoci=[-200.0, -100.0, 0.0], kind="layers", layers=d["layers"], layer_z=cfg["layer_z"])
    if name.startswith("poly_abtem"):
        # polycrystal simulated with abTEM (make_poly_abtem.py); poly_abtem[D][_setup]: D defoci evenly spaced in -150..0 A
        # (D = 3 -> -150, -75, 0), setup from geom_spec.py (a20 default, a30 = 30 mrad).  Data come from the cache written there,
        # truth = abTEM projected potential per slice.  No ground-truth S-matrix (skip_true_S): scored by probe, loss and slab metrics.
        import geom_spec as gs
        D, setup = gs.parse_sample(name); sp = gs.spec(setup)
        defoci = [-150.0, -75.0, 0.0] if D == 3 else [float(v) for v in np.linspace(-150.0, 0.0, D)]
        sl = np.load(os.path.join(HERE, "results_poly", f"abtem_truth_slices{gs.suffix(setup)}.npz"))["slices"]
        g = jp.STANDARD if setup == "a20" else jp.Geometry(alpha=sp["alpha"], dr=sp["dr"], K=sp["K"], M=sp["M"], nscan=sp["nscan"], step=sp["step"])
        return dict(geom=g, ops=None, thick=150.0, defoci=defoci, kind="slabs", slices=sl, dz=2.5, skip_true_S=True)
    if name.startswith("sep"):
        # two-layer axial-resolution benchmark: the SAME atom pattern in two layers at 150 -/+ sep/2.
        # sep{sep}[x{mult}][_s{step}], e.g. sep40_s2.  sep = 0 is the coincident control: one layer at 150 A carrying
        # mult copies of the pattern phase (default 2 = same total scattering as the two-layer samples; x1 = one copy,
        # the single-layer response used for the superposition check).  Reduced geometry (128^2 grid, 10.4 A field).
        # optional suffix _a30: 30 mrad setup (0.16 A pixels, 80 px window, same 25.6 A box and ~10.4 A scan field), e.g. sep80_s3_a30
        a30 = name.endswith("_a30"); body = (name[:-4] if a30 else name)[3:]
        head = body.split("_s")[0]; step = int(body.split("_s")[1]) if "_s" in body else 2
        sep = float(head.split("x")[0]); mult = float(head.split("x")[1]) if "x" in head else (2.0 if sep == 0 else 1.0)
        g = jp.Geometry(alpha=0.030, dr=0.16, K=80, M=2, nscan=round(10.4 / (0.16 * step)), step=step) if a30 else jp.Geometry(M=2, nscan=52 // step, step=step)
        rng = np.random.default_rng(7)
        c = g.Y * g.dr / 2
        yy = (np.arange(g.Y) * g.dr)[:, None]; xx = (np.arange(g.Y) * g.dr)[None, :]
        pat = np.zeros((g.Y, g.Y))
        for p in c + rng.uniform(-4.0, 4.0, (10, 2)):
            pat += 0.6 * np.exp(-((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / (2 * 0.45 ** 2))
        lz = [150.0] if sep == 0 else [150.0 - sep / 2, 150.0 + sep / 2]
        layers = np.array([mult * pat] * len(lz))
        return dict(geom=g, ops=jp.ops_layers(layers, lz, 300.0, g), thick=300.0, defoci=[-200.0, -100.0, 0.0],
                    kind="layers", layers=layers, layer_z=lz, pattern=pat, sep=sep, mult=mult, depth_step=2.5,
                    skip_true_S=True)
    if name.startswith("reduced"):
        step = int(name.split("_s")[1]) if "_s" in name else 4        # scan step in pixels (0.2 A each)
        g = jp.Geometry(M=2, nscan=52 // step, step=step)            # same ~10.4 A field of view for every step
        rng = np.random.default_rng(2024)
        c = g.Y * g.dr / 2
        yy = (np.arange(g.Y) * g.dr)[:, None]; xx = (np.arange(g.Y) * g.dr)[None, :]
        layers = []
        for _ in range(3):
            ph = np.zeros((g.Y, g.Y))
            for p in c + rng.uniform(-4.0, 4.0, (6, 2)):
                ph += 1.0 * np.exp(-((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / (2 * 0.45 ** 2))
            layers.append(ph)
        return dict(geom=g, ops=jp.ops_layers(layers, [30.0, 150.0, 270.0], 300.0, g), thick=300.0,
                    defoci=[-200.0, -100.0, 0.0], kind="layers", layers=np.array(layers), layer_z=[30.0, 150.0, 270.0])
    raise ValueError(name)


def probe_config(defoci, config):
    """Nominal (D, P) and true (D, P) coefficients [A].  Reference dataset = 0."""
    nominal = pa.coefficient_matrix([{"C10": df} for df in defoci], NAMES)
    true = nominal.copy()
    def add(d, **kw):
        for k, v in kw.items():
            true[d, NAMES.index(k)] += v
    if config in ("B", "Bcoma"):
        a1, a2 = math.radians(30), math.radians(-50)
        add(1, C10=25.0, C12a=15.0 * math.cos(2 * a1), C12b=15.0 * math.sin(2 * a1))
        add(2, C10=-20.0, C12a=12.0 * math.cos(2 * a2), C12b=12.0 * math.sin(2 * a2))
        if config == "Bcoma":
            add(1, C21a=500.0); add(2, C21b=-400.0)
    elif config == "B7":                                   # B extended to up to 7 datasets (first two errors identical to B)
        dC10 = [25.0, -20.0, 15.0, -25.0, 20.0, -15.0]; mag = [15.0, 12.0, 10.0, 14.0, 11.0, 13.0]; ang = [30.0, -50.0, 70.0, -20.0, 10.0, -70.0]
        for d in range(1, len(defoci)):
            a = math.radians(ang[d - 1]); add(d, C10=dC10[d - 1], C12a=mag[d - 1] * math.cos(2 * a), C12b=mag[d - 1] * math.sin(2 * a))
    elif config == "G":                                   # common error on ALL datasets (not identifiable)
        for d in range(len(defoci)):
            add(d, C10=25.0)
    elif config != "A":
        raise ValueError(config)
    return nominal, true


def get_data(sname, config, seed, dose, device=None):
    """Noiseless data cached per (sample, config); Poisson noise drawn per seed."""
    os.makedirs(CACHE, exist_ok=True)
    dev = jp.pick_device(device) if isinstance(device, str) else device
    device = dev if dev is not None and dev.type != "cpu" else None     # GPU simulator only when a GPU is used
    s = sample(sname)
    nominal, true = probe_config(s["defoci"], config)
    f = os.path.join(CACHE, f"clean_{sname}_{config}.npz")
    if os.path.exists(f):
        dc = np.load(f); clean, coords = dc["dps"], dc["coords"]
    else:
        t0 = time.time()
        clean, coords = jp.simulate_4dstem(s["ops"], true, s["geom"], names=NAMES, device=device)
        np.savez(f, dps=clean, coords=coords)
        print(f"simulated {sname}/{config} in {time.time() - t0:.0f}s", flush=True)
    dps = jp.add_poisson(clean, dose, seed) if dose > 0 else clean
    return s, nominal, true, dps, coords


def get_true_S(sname):
    f = os.path.join(CACHE, f"Strue_{sname}.npy")
    if os.path.exists(f):
        return np.load(f, mmap_mode="r")
    s = sample(sname)
    S = jp.true_smatrix(s["ops"], s["geom"], s["thick"], reference="entrance").astype(np.complex64)
    np.save(f, S)
    return S


# --------------------------------------------------------------------------------------------- one reconstruction
def make_probe(geom, coeffs, trainable, ref=0, device="cpu"):
    by, bx = geom.beams()
    return pa.AberrationProbe(by / geom.LW, bx / geom.LW, geom.lam, geom.alpha, coeffs, names=NAMES,
                              trainable=trainable, ref_index=ref, norm=1 / math.sqrt(len(by) * geom.K ** 2), device=device)


def evaluate(S, probe, s, true, sname, depths, with_S_error=True, device="cpu"):
    g = s["geom"]
    cr = g.crop()
    t0 = time.time()
    ew = jp.depth_sections_general(S, g, depths, device=device)
    ph = np.angle(ew)[(slice(None),) + cr]
    if s["kind"] == "layers":
        dm = jp.layer_metrics(ph, [L[cr] for L in s["layers"]], s["layer_z"], depths)
    else:
        zc = (np.arange(s["slices"].shape[0]) + 0.5) * s["dz"]
        dm = jp.slab_metrics(ph, [s["slices"][np.abs(zc - z) <= 20.0].sum(0)[cr] for z in depths], depths)
    est = probe.coefficients().detach().cpu().numpy()
    tp = make_probe(g, true, [])
    perr = jp.probe_error(probe.beam_coefficients().detach().cpu().numpy(), tp.beam_coefficients().detach().cpu().numpy())
    out = dict(depth=dm, coeff_est=est.tolist(), coeff_true=true.tolist(), probe_wavefunction_error=perr.tolist(),
               sections_time_s=time.time() - t0)
    if with_S_error:
        St = get_true_S(sname)
        S_np = S.detach().cpu().numpy() if torch.is_tensor(S) else S
        out["S_error"] = jp.smatrix_error(S_np, np.asarray(St), g, z_out=np.arange(-60.0, 61.0, 10.0))
    return out, ph


def run_one(sname, config, variant, seed, sch_kw, dose, trainable, tag="", device="cpu", volume_step=0.0, save_S=False):
    s, nominal, true, dps, coords = get_data(sname, config, seed, dose, device=device)
    g = s["geom"]
    device = jp.pick_device(device) if isinstance(device, str) else device
    meas = jp.Measurements.from_dps(dps, coords, dose, device=device)
    init = true if variant == "oracle" else nominal
    probe = make_probe(g, init, trainable if variant == "joint" else [], device=device)
    sch = jp.Schedule(mode="joint" if variant == "joint" else "s_only", seed=seed, **sch_kw)
    print(f"== {sname} config {config} variant {variant} seed {seed}: {sch.n_s_iters()} S iterations", flush=True)
    rec = jp.JointReconstructor(meas, g, probe, sch, device=device, log=print)
    S_dev = rec.run()
    depths = np.arange(0.0, s["thick"] + 1e-6, s.get("depth_step", 10.0))
    ev, ph = evaluate(S_dev, probe, s, true, sname, depths, device=device, with_S_error=not s.get("skip_true_S"))
    S = S_dev.detach().cpu().numpy()
    res = dict(sample=sname, config=config, variant=variant, seed=seed, dose=dose, trainable=trainable,
               nominal=nominal.tolist(), names=NAMES, summary=rec.summary(), history=rec.history, eval=ev)
    d = os.path.join(OUT, sname, config + tag)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{variant}_seed{seed}")
    json.dump(res, open(base + ".json", "w"), indent=1, default=float)
    np.savez_compressed(base + "_sections.npz", phase=ph.astype(np.float32), depths=depths)
    if volume_step > 0:                     # complex refocused depth volume O_S(x, y, z) on the FULL S grid (not phase-only)
        zs = np.arange(0.0, s["thick"] + 1e-6, volume_step)
        O = jp.depth_sections_general(S_dev, g, zs, device=device)                # complex64 (Z, Y, X) = (depth, y, x)
        cr0 = g.crop()
        meta = dict(sample=sname, config=config, variant=variant, seed=seed, dose=dose, trainable=trainable, thick=s["thick"],
                    depth_step_A=volume_step, dr_A=g.dr, grid=g.Y, crop_y=[cr0[0].start, cr0[0].stop], crop_x=[cr0[1].start, cr0[1].stop],
                    scan_step_px=g.step, nscan=g.nscan, energy_eV=g.ev, alpha_rad=g.alpha, schedule=asdict(sch),
                    coeff_est=probe.coefficients().detach().cpu().numpy().tolist(), coeff_true=np.asarray(true).tolist(), coeff_names=NAMES,
                    note="O[z, y, x] = refocused complex exit-wave section of the recovered S-matrix, entrance-referenced; depth_step_A is a "
                         "computational sampling interval, not the axial resolution")
        np.savez_compressed(base + "_volume.npz", volume=O.astype(np.complex64), depths=zs, meta=json.dumps(meta, default=float))
        print(f"   saved complex volume {O.shape} at {volume_step} A steps -> {base}_volume.npz", flush=True)
    if save_S:
        np.save(os.path.join(CACHE, f"S_{sname}_{config}{tag}_{variant}_seed{seed}.npy"), S_dev.detach().cpu().numpy().astype(np.complex64))
    # a few S-matrix beams (recovered and ground truth, entrance-referenced) for response figures
    by, bx = g.beams()
    sel = [int(np.argmin(by ** 2 + bx ** 2))] + [int(np.argmin((by - a) ** 2 + (bx - b) ** 2)) for a, b in ((6, 0), (0, -9), (8, 8))]
    cr = g.crop()
    if not s.get("skip_true_S"):                  # benchmark samples skip the (costly) ground-truth S
        St = np.asarray(get_true_S(sname))
        np.savez_compressed(base + "_beams.npz", beams=np.array(sel), by=by[sel], bx=bx[sel],
                            S_rec=S[sel][(slice(None),) + cr], S_true=St[sel][(slice(None),) + cr])
    dm = ev["depth"]
    key = "mean_abs_depth_err" if "mean_abs_depth_err" in dm else "mean_depth_err"
    print(f"-> loss {rec.final_eval['rel_amplitude_loss']:.5f}, depth err {dm[key]:.1f} A, probe err {np.round(ev['probe_wavefunction_error'], 3)}, "
          + (f"S err {ev['S_error']['nrmse']:.3f} (z_out {ev['S_error']['z_out']:.0f}), " if "S_error" in ev else "")
          + f"runtime {rec.runtime:.0f}s, peak {rec.peak_mb:.0f} MB",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["exp"])
    ap.add_argument("--sample", default="reduced")
    ap.add_argument("--config", default="B")
    ap.add_argument("--variants", default="fixed,joint,oracle")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--dose", type=float, default=2e5)
    ap.add_argument("--trainable", default="C10,C12a,C12b")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, cuda, cuda:1, mps")
    ap.add_argument("--sched", default="{}", help="JSON dict of Schedule overrides")
    ap.add_argument("--volume-step", type=float, default=0.0, help="if > 0, also save the complex refocused volume at this depth step (A)")
    ap.add_argument("--save-S", action="store_true", help="save the recovered S-matrix (complex64 .npy, ~150 MB) in results_joint/cache/")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sch_kw = dict(warmup_s=5, cycles=8, s_per_cycle=3, p_per_cycle=2, final_s=3, final_p=1,
                  probe_lr=0.3, probe_fraction=0.34, grad_clip=1.0)
    sch_kw.update(json.loads(a.sched))
    for seed in [int(x) for x in a.seeds.split(",")]:
        for v in a.variants.split(","):
            run_one(a.sample, a.config, v, seed, sch_kw, a.dose, a.trainable.split(","), a.tag, device=a.device,
                    volume_step=a.volume_step, save_S=a.save_S)


if __name__ == "__main__":
    main()
