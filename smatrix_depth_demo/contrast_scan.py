"""M3 follow-up: how much feature-to-background contrast do depth placement and feature counting need?

Grid over Au cluster size (atoms) x carbon support density (atoms/A^3) in a 100 A slab, abTEM data, joint fit, as in
calibrate_m3.py.  Per combination (12 reconstructions, ~15 min on 4 CPU cores):
    templates  cluster (seed 11) at 0/25/50/75/100 A in carbon environment A, plus environment A alone (differential)
    tests      held-out cluster (seed 22) at 15 / 50 / 85 A in environment B, plus environment B alone (oracle)
    pairs      two held-out clusters at 50 -/+ d/2, d = 40, 60
Reports per combination: raw and oracle depth errors, single-feature residuals, the pairs' one-feature residual
relative to the singles' (counting is possible when pairs clearly exceed singles), and the measured contrast
||response(feature) - response(background)|| / ||response(background)|| at the feature.

    python contrast_scan.py run --sizes 3 --densities 0,0.025,0.05,0.1
    python contrast_scan.py report                       # results_contrast/CONTRAST_RESULTS.md
"""
import argparse, json, os, time
import numpy as np
import torch

import simple_joint as sj
import abtem_data as ad
import calibrate_m3 as m3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_contrast")
T = 100.0
TEMPLATE_Z = np.linspace(0.0, T, 5)
TEST_Z = [15.0, 50.0, 85.0]
SEPS = [40.0, 60.0]


def combo_dir(n, rho):
    return os.path.join(OUT, f"au{n}_rho{rho:g}")


def cluster(setup, seed, n):
    # area grows with n; above 6 atoms random placement 2.5 A apart needs extra room (12 atoms do not fit in +-4 A)
    return m3.gold_cluster(setup, seed, n=n, spread=2.0 * np.sqrt(n / 3.0) * (1.0 if n <= 6 else 1.4))


def specimen(setup, xy, zs, env, rho):
    from ase import Atoms
    C = m3.carbon(setup, T, env, density=rho) if rho > 0 else np.zeros((0, 3))
    au = np.array([(x, y, min(z, T - m3.Z_EPS)) for z in zs for (y, x) in xy]).reshape(-1, 3)
    if len(au) and len(C):
        C = C[np.linalg.norm(C[:, None, :] - au[None, :, :], axis=-1).min(axis=1) > 2.0]
    L = setup.Y * setup.dr
    pos = np.vstack([C, au]) if len(au) else C          # never empty: background runs only exist for rho > 0
    return Atoms(symbols=["C"] * len(C) + ["Au"] * len(au), positions=pos, cell=[L, L, T], pbc=[True, True, False])


def reconstruct(d, tag, atoms, noise_seed, a):
    f = os.path.join(d, f"S_{tag}.npy")
    if os.path.exists(f):
        return
    setup, positions, nominal, true = m3.config(T)
    t0 = time.time()
    dps = ad.simulate_abtem(setup, atoms, true, positions, dose=a.dose, seed=noise_seed, log=lambda *_: None)
    amp, pos, didx = sj.prepare_data(dps, positions, device=a.device)
    model = sj.JointModel(setup, pos, didx, nominal, trainable=("C10", "C12a", "C12b"), device=a.device)
    hist = sj.fit(model, amp, epochs=a.epochs, log=lambda m: None)
    np.save(f, model.S.detach().cpu().numpy())
    json.dump(dict(tag=tag, loss=hist[-1], runtime_s=time.time() - t0,
                   coeff_err=(model.coefficients().detach().cpu().numpy() - true)[1:, :3].tolist()),
              open(f[:-4] + ".json", "w"), indent=1)
    print(f"{os.path.basename(d)} {tag:16s} loss {hist[-1]:.6f}  {time.time() - t0:5.0f} s", flush=True)


def run(a):
    setup = m3.config(T)[0]
    for n in a.sizes:
        for rho in a.densities:
            d = combo_dir(n, rho)
            os.makedirs(d, exist_ok=True)
            xt, xs = cluster(setup, m3.TEMPLATE_SEED, n), cluster(setup, m3.TEST_SEED, n)
            if rho > 0:
                reconstruct(d, "tmpl_bg", specimen(setup, xt, [], m3.ENV_TEMPLATE, rho), 199, a)
                reconstruct(d, "test_bg", specimen(setup, xs, [], m3.ENV_TEST, rho), 50, a)
            for i, z in enumerate(TEMPLATE_Z):
                reconstruct(d, f"tmpl_z{z:.0f}", specimen(setup, xt, [z], m3.ENV_TEMPLATE, rho), 100 + i, a)
            for z in TEST_Z:
                reconstruct(d, f"test_z{z:.0f}", specimen(setup, xs, [z], m3.ENV_TEST, rho), 0, a)
            for sep in SEPS:
                reconstruct(d, f"pair_d{sep:.0f}", specimen(setup, xs, [50 - sep / 2, 50 + sep / 2], m3.ENV_TEST, rho), 0, a)


def analyze_combo(n, rho, device):
    d = combo_dir(n, rho)
    setup = m3.config(T)[0]
    zg = m3.depth_grid(T)
    tw = ad.layer_images(setup, [cluster(setup, m3.TEMPLATE_SEED, n)])[0]
    w = ad.layer_images(setup, [cluster(setup, m3.TEST_SEED, n)])[0]
    tw, w = tw / tw.max(), w / w.max()
    tags = ([f"tmpl_z{z:.0f}" for z in TEMPLATE_Z] + [f"test_z{z:.0f}" for z in TEST_Z] + [f"pair_d{s:.0f}" for s in SEPS]
            + (["tmpl_bg", "test_bg"] if rho > 0 else []))
    if any(not os.path.exists(os.path.join(d, f"S_{t}.npy")) for t in tags):
        return None
    V = lambda t: sj.refocus(np.load(os.path.join(d, f"S_{t}.npy")), setup, zg, device=device)
    bt = sj.feature_response(V("tmpl_bg"), tw) if rho > 0 else 0.0
    bs = sj.feature_response(V("test_bg"), w) if rho > 0 else 0.0
    h = np.array([sj.feature_response(V(f"tmpl_z{z:.0f}"), tw) - bt for z in TEMPLATE_Z])
    kw = dict(refine=TEMPLATE_Z[1] - TEMPLATE_Z[0], step=1.0, min_sep=10.0, max_phase=np.pi / 4)
    row = dict(au=n, rho=rho, raw=[], oracle=[], res_single=[], contrast=[])
    for z in TEST_Z:
        r = sj.feature_response(V(f"test_z{z:.0f}"), w)
        (zr,), res = sj.fit_depths_joint(r, h, TEMPLATE_Z, 1, **kw)
        (zo,), _ = sj.fit_depths_joint(r - bs, h, TEMPLATE_Z, 1, **kw)
        row["raw"].append(zr - z); row["oracle"].append(zo - z); row["res_single"].append(res)
        row["contrast"].append(float(np.linalg.norm(r - bs) / np.linalg.norm(bs)) if rho > 0 else float("inf"))
    row["pairs"] = []
    for s in SEPS:
        r = sj.feature_response(V(f"pair_d{s:.0f}"), w)
        r1 = sj.fit_depths_joint(r, h, TEMPLATE_Z, 1, **kw)[1]
        zs, r2 = sj.fit_depths_joint(r, h, TEMPLATE_Z, 2, **kw)
        row["pairs"].append(dict(sep=s, res1=r1, res2=r2, two_fit=zs, truth=[50 - s / 2, 50 + s / 2],
                                 res1_over_single_max=r1 / max(row["res_single"])))
    row["probe_err_absmax"] = float(np.abs([json.load(open(os.path.join(d, f"S_{t}.json")))["coeff_err"]
                                            for t in tags]).max())
    return row


def report(a):
    rows = []
    for name in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        if name.startswith("au"):
            n, rho = int(name[2:].split("_")[0]), float(name.split("rho")[1])
            r = analyze_combo(n, rho, a.device)
            if r:
                rows.append(r)
    rows.sort(key=lambda r: (r["au"], r["rho"]))
    json.dump(rows, open(os.path.join(OUT, "contrast_report.json"), "w"), indent=1, default=float)
    L = ["# Contrast scan, 100 A slab (generated by contrast_scan.py report)", "",
         "Au cluster size x carbon density. Errors in A at true depths 15 / 50 / 85 A. Pair ratio = one-feature "
         "residual of the pair / largest single-feature residual (counting needs this clearly > 1).", "",
         "| Au atoms | carbon (atoms/A^3) | contrast (median) | raw errors | oracle errors | single res. (max) | "
         "pair ratio d=40 / 60 | pair 2-fit d=40 / 60 | probe max err |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['au']} | {r['rho']:g} | {np.median(r['contrast']):.2f} | {np.round(r['raw'], 1).tolist()} | "
                 f"{np.round(r['oracle'], 1).tolist()} | {max(r['res_single']):.4f} | "
                 + " / ".join(f"{p['res1_over_single_max']:.1f}" for p in r["pairs"]) + " | "
                 + " / ".join(str(np.round(p["two_fit"], 0).tolist()) for p in r["pairs"]) + f" | {r['probe_err_absmax']:.1f} |")
    open(os.path.join(OUT, "CONTRAST_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--sizes", type=lambda s: [int(v) for v in s.split(",")], default=[3, 6, 12])
    ap.add_argument("--densities", type=lambda s: [float(v) for v in s.split(",")], default=[0.0, 0.025, 0.05, 0.1])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dose", type=float, default=1e5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
