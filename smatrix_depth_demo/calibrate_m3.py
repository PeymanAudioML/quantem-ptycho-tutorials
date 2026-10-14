"""M3: realistic specimens.  A gold cluster at a known depth inside an amorphous-carbon support that fills the slab,
simulated with abTEM (real atomic potentials), reconstructed with the joint S + probe fit (GPU-ready).

Changes from M1, one at a time:
  * data      abTEM multislice through atoms (0.2 A pixels for the band limit) instead of Gaussian phase layers
  * specimen  3 Au atoms in a carbon matrix (~0.1 atoms/A^3 through the whole thickness)
  * templates DIFFERENTIAL: response(cluster at z0 in carbon) - response(same carbon, no cluster), environment A
  * tests     held-out cluster in a DIFFERENT carbon arrangement (environment B); fitted raw (as on real data) and,
              as an oracle, with environment B's own background subtracted
  * probe     reports the probe-estimate bias on this dense sample (M1's sparse sample biased one defocus by ~6 A)

    python calibrate_m3.py all --device cuda                    # 300 A (default): ~25 reconstructions
    python calibrate_m3.py all --device cuda --thickness 100    # cheap
    python calibrate_m3.py all --device cuda --thickness 600    # ~42x M1 per reconstruction: run deliberately
    python calibrate_m3.py analyze --thickness 300              # results_m3_T300/M3_RESULTS.md

Steps can be run separately (templates / test / pairs / analyze); finished reconstructions are skipped.
Needs abtem (pip install abtem).  abTEM's multislice runs on the CPU; the reconstruction uses --device.
"""
import argparse, json, os, time
import numpy as np
import torch

import simple_joint as sj
import abtem_data as ad

HERE = os.path.dirname(os.path.abspath(__file__))
DR, STEP = 0.2, 2                                           # 0.2 A pixels (abTEM band limit 33 mrad), 0.4 A scan step
GEOMETRY = {100: dict(K=32), 300: dict(K=60), 600: dict(K=108)}   # window >= probe cone (check_geometry)
TEMPLATE_SEED, TEST_SEED = 11, 22                           # gold-cluster arrangements
ENV_TEMPLATE, ENV_TEST = 1, 2                               # carbon arrangements
CARBON_DENSITY = 0.1                                        # atoms / A^3 (amorphous carbon ~2 g/cm^3)


# ------------------------------------------------------------------------------------------------ configuration
def config(T):
    """Setup, scan positions, nominal and true probe for thickness T (A).  Scan field = window size."""
    K = GEOMETRY[int(T)]["K"]
    setup = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=DR, K=K, Y=2 * K)
    g = np.arange(K // 2, setup.Y - K // 2 + 1, STEP)
    positions = np.stack(np.meshgrid(g, g, indexing="ij"), -1)
    nominal = np.zeros((3, len(sj.NAMES)))
    nominal[:, 0] = [-5 * T / 6, -T / 2, -T / 6]            # foci at 5T/6, T/2, T/6 (as -250/-150/-50 for 300 A)
    true = nominal.copy()
    true[1, :3] += [12.0, 6.0, -4.0]                        # same relative errors as M0-M2
    true[2, :3] += [-10.0, -3.0, 5.0]
    return setup, positions, nominal, true


def template_depths(T):
    return np.linspace(0.0, T, 13)                          # 25 A spacing at 300 A


def test_depths(T):
    return [round(0.1 * T), round(0.5 * T), round(0.9 * T)]


def pair_seps(T):
    return [d for d in (40.0, 60.0, 80.0, 120.0) if d < 0.9 * T]


def depth_grid(T):
    return np.arange(-0.2 * T, 1.2 * T + 1e-6, 5.0)


# ------------------------------------------------------------------------------------------------ specimen
def gold_cluster(setup, seed, n=3, spread=2.0, min_dist=2.5):
    """(n, 2) lateral (y, x) positions [A] of Au atoms near the field centre."""
    rng = np.random.default_rng(seed)
    c = setup.Y * setup.dr / 2
    pts = []
    while len(pts) < n:
        p = c + rng.uniform(-spread, spread, 2)
        if all(np.hypot(*(p - q)) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


def carbon(setup, T, seed, density=CARBON_DENSITY):
    """(N, 3) carbon positions (x, y, z) [A]: jittered cubic lattice at `density`, filling the box and 0..T."""
    a = density ** (-1 / 3)
    L = setup.Y * setup.dr
    rng = np.random.default_rng(seed)
    gx = np.arange(a / 2, L, a)
    gz = np.arange(a / 2, T, a)
    X, Yy, Z = np.meshgrid(gx, gx, gz, indexing="ij")
    p = np.stack([X.ravel(), Yy.ravel(), Z.ravel()], -1) + rng.uniform(-0.3 * a, 0.3 * a, (X.size, 3))
    p[:, :2] %= L
    p[:, 2] = np.clip(p[:, 2], 0.0, T)
    return p


def specimen(setup, T, cluster_xy, z_list, env_seed):
    """ase.Atoms: carbon environment `env_seed` plus the Au cluster at each depth in z_list (carbon within 2 A of
    any Au removed)."""
    from ase import Atoms
    C = carbon(setup, T, env_seed)
    au = np.array([(x, y, z) for z in z_list for (y, x) in cluster_xy]).reshape(-1, 3)
    if len(au):
        d = np.linalg.norm(C[:, None, :] - au[None, :, :], axis=-1).min(axis=1)
        C = C[d > 2.0]
    L = setup.Y * setup.dr
    return Atoms(symbols=["C"] * len(C) + ["Au"] * len(au), positions=np.vstack([C, au]) if len(au) else C,
                 cell=[L, L, T], pbc=[True, True, False])


# ------------------------------------------------------------------------------------------------ reconstructions
def out_dir(T):
    return os.path.join(HERE, f"results_m3_T{int(T)}")


def reconstruct(tag, atoms, noise_seed, a):
    T = a.thickness
    f = os.path.join(out_dir(T), f"S_{tag}.npy")
    if os.path.exists(f):
        return
    setup, positions, nominal, true = config(T)
    t0 = time.time()
    dps = ad.simulate_abtem(setup, atoms, true, positions, dose=a.dose, seed=noise_seed, log=lambda *_: None)
    t_sim = time.time() - t0
    amp, pos, didx = sj.prepare_data(dps, positions, device=a.device)
    model = sj.JointModel(setup, pos, didx, nominal, trainable=("C10", "C12a", "C12b"), device=a.device)
    hist = sj.fit(model, amp, epochs=a.epochs, batch=a.batch, log=lambda m: None)
    np.save(f, model.S.detach().cpu().numpy())
    json.dump(dict(tag=tag, thickness=T, noise_seed=noise_seed, loss=hist[-1], simulate_s=t_sim,
                   runtime_s=time.time() - t0, device=str(a.device), epochs=a.epochs,
                   coeff_err=(model.coefficients().detach().cpu().numpy() - true)[1:, :3].tolist()),
              open(f[:-4] + ".json", "w"), indent=1)
    print(f"{tag:24s} loss {hist[-1]:.6f}  simulate {t_sim:5.0f} s  total {time.time() - t0:6.0f} s", flush=True)


def run_templates(a):
    setup = config(a.thickness)[0]
    xy = gold_cluster(setup, TEMPLATE_SEED)
    reconstruct("tmpl_bg", specimen(setup, a.thickness, xy, [], ENV_TEMPLATE), 199, a)
    for i, z in enumerate(template_depths(a.thickness)):
        reconstruct(f"tmpl_z{z:.0f}", specimen(setup, a.thickness, xy, [z], ENV_TEMPLATE), 100 + i, a)


def run_test(a):
    setup = config(a.thickness)[0]
    xy = gold_cluster(setup, TEST_SEED)
    for n in (0, 1):
        reconstruct(f"test_bg_n{n}", specimen(setup, a.thickness, xy, [], ENV_TEST), 50 + n, a)
        for z in test_depths(a.thickness):
            reconstruct(f"test_z{z:.0f}_n{n}", specimen(setup, a.thickness, xy, [z], ENV_TEST), n, a)


def run_pairs(a):
    setup = config(a.thickness)[0]
    xy = gold_cluster(setup, TEST_SEED)
    c = a.thickness / 2
    for d in pair_seps(a.thickness):
        reconstruct(f"pair_d{d:.0f}", specimen(setup, a.thickness, xy, [c - d / 2, c + d / 2], ENV_TEST), 0, a)


# ------------------------------------------------------------------------------------------------ analysis
def analyze(a):
    T, dev = a.thickness, a.device
    out = out_dir(T)
    setup = config(T)[0]
    zt, zgrid = template_depths(T), depth_grid(T)
    tw = ad.layer_images(setup, [gold_cluster(setup, TEMPLATE_SEED)])[0]
    w = ad.layer_images(setup, [gold_cluster(setup, TEST_SEED)])[0]
    tw, w = tw / tw.max(), w / w.max()
    tests = [(z, n) for n in (0, 1) for z in test_depths(T)]
    tags = (["tmpl_bg"] + [f"tmpl_z{z:.0f}" for z in zt] + ["test_bg_n0", "test_bg_n1"]
            + [f"test_z{z:.0f}_n{n}" for z, n in tests] + [f"pair_d{d:.0f}" for d in pair_seps(T)])
    missing = [t for t in tags if not os.path.exists(os.path.join(out, f"S_{t}.npy"))]
    if missing:
        print("missing reconstructions (run templates / test / pairs first):", missing)
        return
    R = lambda t, wt: sj.feature_response(sj.refocus(np.load(os.path.join(out, f"S_{t}.npy")), setup, zgrid,
                                                     device=dev), wt)
    bg_t = R("tmpl_bg", tw)
    h = np.array([R(f"tmpl_z{z:.0f}", tw) - bg_t for z in zt])         # differential pipeline templates
    kw = dict(refine=zt[1] - zt[0], step=1.0, min_sep=10.0)
    bg = {n: R(f"test_bg_n{n}", w) for n in (0, 1)}
    raw = {(z, n): R(f"test_z{z:.0f}_n{n}", w) for z, n in tests}
    rep = {"thickness": T, "single": [], "negative_control": [], "pairs": []}
    for (z, n), r in raw.items():
        for kind, rr in (("raw", r), ("background-subtracted (oracle)", r - bg[n])):
            (zf,), res = sj.fit_depths_joint(rr, h, zt, 1, **kw)
            rep["single"].append(dict(response=kind, z0=z, noise=n, depth=zf, error=zf - z, residual=res))
    floor = 2.0 * max(s["residual"] for s in rep["single"] if s["response"] == "raw" and s["noise"] == 0)
    rep["floor"] = floor
    for z in test_depths(T):
        zs, fits = sj.select_n_features(raw[(z, 1)], h, zt, floor, max_features=2, **kw)
        rep["negative_control"].append(dict(z0=z, n_found=len(zs), depths=zs, res1=fits[1][1]))
    c = T / 2
    for d in pair_seps(T):
        zs, fits = sj.select_n_features(R(f"pair_d{d:.0f}", w), h, zt, floor, max_features=2, **kw)
        rep["pairs"].append(dict(sep=d, truth=[c - d / 2, c + d / 2], selected=zs, two_feature_fit=fits[2][0],
                                 res1=fits[1][1], res2=fits[2][1]))
    errs = np.array([json.load(open(os.path.join(out, f"S_{t}.json")))["coeff_err"] for t in tags])
    rep["probe_bias_mean_A"] = errs.mean(0).tolist()             # (dataset 1-2) x (C10, C12a, C12b)
    rep["probe_bias_absmax_A"] = np.abs(errs).max(0).tolist()
    json.dump(rep, open(os.path.join(out, "M3_report.json"), "w"), indent=1, default=float)
    write_markdown(rep, out)
    print(open(os.path.join(out, "M3_RESULTS.md")).read())


def write_markdown(rep, out):
    T = rep["thickness"]
    L = [f"# M3 results, {T:.0f} A slab (generated by calibrate_m3.py analyze)", "",
         "Au cluster in amorphous carbon, abTEM data, joint S + probe fit. Differential templates from carbon "
         "environment A; tests in environment B with a different cluster.", "",
         "## Single feature", "", "| response | z0 | noise | depth | error | residual |", "|---|---|---|---|---|---|"]
    L += [f"| {s['response']} | {s['z0']:.0f} | {s['noise']} | {s['depth']:.1f} | {s['error']:+.1f} | {s['residual']:.4f} |"
          for s in rep["single"]]
    L += ["", f"Residual floor (2 x max raw single-feature residual, noise 0): {rep['floor']:.4f}", "",
          "## Negative control (raw, noise 1)", "", "| z0 | features found | depths | res 1 |", "|---|---|---|---|"]
    L += [f"| {s['z0']:.0f} | {s['n_found']} | {np.round(s['depths'], 1).tolist()} | {s['res1']:.4f} |"
          for s in rep["negative_control"]]
    L += ["", f"## Two clusters at {T / 2:.0f} -/+ d/2 (raw)", "",
          "| d | truth | selected | 2-feature fit | res 1 | res 2 |", "|---|---|---|---|---|---|"]
    L += [f"| {p['sep']:.0f} | {p['truth']} | {np.round(p['selected'], 1).tolist()} | "
          f"{np.round(p['two_feature_fit'], 1).tolist()} | {p['res1']:.4f} | {p['res2']:.4f} |" for p in rep["pairs"]]
    m, x = np.array(rep["probe_bias_mean_A"]), np.array(rep["probe_bias_absmax_A"])
    L += ["", "## Probe estimate error over all runs (A; rows = datasets 1, 2; columns = C10, C12a, C12b)", "",
          f"- mean: {np.round(m, 1).tolist()}", f"- max |error|: {np.round(x, 1).tolist()}",
          "- M1 (sparse phase sample): dataset 2 C10 5.5-7.3 A off in every run"]
    open(os.path.join(out, "M3_RESULTS.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["templates", "test", "pairs", "analyze", "all"])
    ap.add_argument("--thickness", type=float, default=300.0, choices=[100.0, 300.0, 600.0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dose", type=float, default=1e5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(out_dir(a.thickness), exist_ok=True)
    setup, positions, nominal, _ = config(a.thickness)
    nb = len(setup.beams()[0])
    print(f"device {a.device}; {a.thickness:.0f} A slab; K {setup.K}, Y {setup.Y}, {nb} beams, "
          f"{3 * positions.shape[0] * positions.shape[1]} patterns per reconstruction "
          f"(~{nb * setup.K ** 2 * 3 * positions[..., 0].size / (473 * 1600 * 5043):.1f}x an M1 reconstruction)", flush=True)
    sj.check_geometry(setup, positions, nominal, a.thickness, abtem=True)
    if a.cmd in ("templates", "all"):
        run_templates(a)
    if a.cmd in ("test", "all"):
        run_test(a)
    if a.cmd in ("pairs", "all"):
        run_pairs(a)
    if a.cmd in ("analyze", "all"):
        analyze(a)


if __name__ == "__main__":
    main()
