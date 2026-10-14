"""M1 pilot: depth templates built THROUGH the reconstruction pipeline, tested on held-out clusters (GPU-ready).

M0 showed that templates from the exact S place a single feature to within 4 A on recovered S, but leave a residual
floor ~20x higher than on exact S, so a two-feature model splits one cluster into two.  M1 builds the templates from
reconstructed S (same simulation + joint fit as the data) and checks:

  1. held-out accuracy   cluster 22 (not used for templates) at 30 / 150 / 270 A, two noise realisations each
  2. residual floor      single-feature residual with pipeline vs exact-S templates
  3. negative control    single clusters must stay single (floor calibrated on noise realisation 0, tested on 1)
  4. two features        two clusters in one column at 150 -/+ d/2, d = 40 / 80 / 120 A

Run everything on a GPU (one reconstruction is minutes there, ~1 h on 4 CPU cores):

    python calibrate_m1.py all --device cuda            # templates + test + pairs + analyze
    python calibrate_m1.py templates --device cuda       # or step by step; finished runs are skipped
    python calibrate_m1.py test --device cuda
    python calibrate_m1.py pairs --device cuda
    python calibrate_m1.py analyze                       # CPU is fine; writes results_m1/M1_RESULTS.md

Setup, clusters and acquisition are those of calibrate_depth.py (M0): 300 kV, 20 mrad, 0.3 A pixels, 300 A slab,
three defoci, relative probe errors, 0.3 A scan step, 1e5 e-/pattern, joint S + probe fit for 40 epochs.
"""
import argparse, json, os, time
import numpy as np
import torch

import simple_joint as sj
import demo_simple_joint as demo
import calibrate_depth as cd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_m1")
TEMPLATE_DEPTHS = np.arange(0.0, cd.THICK + 1e-6, 25.0)     # 13 templates; 25 A spacing interpolates exactly (M1 check)
TEST_DEPTHS = [30.0, 150.0, 270.0]
TEST_NOISE = [0, 1]                                          # noise realisation 0 calibrates the floor, 1 tests it
PAIR_SEPS = [40.0, 80.0, 120.0]


# ------------------------------------------------------------------------------------------------ reconstructions
def reconstruct(tag, layers, layer_z, noise_seed, args, out=None):
    """Simulate data for `layers` at `layer_z`, run the joint S + probe fit, save S and a JSON summary to `out`
    (default results_m1/). Skips if done."""
    f = os.path.join(out or OUT, f"S_{tag}.npy")
    if os.path.exists(f):
        return
    positions, nominal, true = cd.problem()
    t0 = time.time()
    dps = demo.simulate(cd.SETUP, layers, layer_z, cd.THICK, true, positions, dose=args.dose, seed=noise_seed,
                        device=args.device)
    amp, pos, didx = sj.prepare_data(dps, positions, device=args.device)
    model = sj.JointModel(cd.SETUP, pos, didx, nominal, trainable=("C10", "C12a", "C12b"), device=args.device)
    hist = sj.fit(model, amp, epochs=args.epochs, batch=args.batch, log=lambda m: None)
    np.save(f, model.S.detach().cpu().numpy())
    json.dump(dict(tag=tag, layer_z=list(map(float, layer_z)), noise_seed=noise_seed, loss=hist[-1],
                   runtime_s=time.time() - t0, device=str(args.device), epochs=args.epochs, batch=args.batch,
                   coeff_err=(model.coefficients().detach().cpu().numpy() - true)[1:, :3].tolist()),
              open(f[:-4] + ".json", "w"), indent=1)
    print(f"{tag:22s} loss {hist[-1]:.6f}  {time.time() - t0:6.0f} s", flush=True)


def run_templates(args):
    img, _ = cd.cluster(cd.TEMPLATE_SEED)
    for i, z in enumerate(TEMPLATE_DEPTHS):
        reconstruct(f"tmpl_z{z:.0f}", [img], [z], 100 + i, args)


def run_test(args):
    img, _ = cd.cluster(cd.TEST_SEED)
    for n in TEST_NOISE:
        for z in TEST_DEPTHS:
            reconstruct(f"test_z{z:.0f}_n{n}", [img], [z], n, args)


def run_pairs(args):
    img, _ = cd.cluster(cd.TEST_SEED)
    for d in PAIR_SEPS:
        reconstruct(f"pair_d{d:.0f}", [img, img], [150.0 - d / 2, 150.0 + d / 2], 0, args)


# ------------------------------------------------------------------------------------------------ analysis
def pipeline_templates(device=None):
    """(13, Nz) pipeline templates: loaded from results_m1/templates_pipeline.npz, else built from the template runs."""
    f = os.path.join(OUT, "templates_pipeline.npz")
    if os.path.exists(f):
        return np.load(f)["h"]
    _, tw = cd.cluster(cd.TEMPLATE_SEED)
    tags = [f"tmpl_z{z:.0f}" for z in TEMPLATE_DEPTHS]
    missing = [t for t in tags if not os.path.exists(os.path.join(OUT, f"S_{t}.npy"))]
    if missing:
        raise FileNotFoundError(f"run 'python calibrate_m1.py templates' first; missing {missing}")
    tr = responses(tags, tw, device)
    h = np.array([tr[t] for t in tags])
    np.savez(f, h=h, z0=TEMPLATE_DEPTHS, depths=cd.DEPTHS)
    return h


def responses(tags, weight, device, out=None):
    return {t: sj.feature_response(sj.refocus(np.load(os.path.join(out or OUT, f"S_{t}.npy")), cd.SETUP, cd.DEPTHS,
                                              device=device), weight) for t in tags}


def analyze(args):
    dev = args.device
    _, tw = cd.cluster(cd.TEMPLATE_SEED)
    img, w = cd.cluster(cd.TEST_SEED)
    missing = [t for t in [f"tmpl_z{z:.0f}" for z in TEMPLATE_DEPTHS]
               + [f"test_z{z:.0f}_n{n}" for n in TEST_NOISE for z in TEST_DEPTHS]
               + [f"pair_d{d:.0f}" for d in PAIR_SEPS] if not os.path.exists(os.path.join(OUT, f"S_{t}.npy"))]
    if missing:
        print("missing reconstructions (run templates / test / pairs first):", missing)
        return
    # template libraries on the same depth grid: pipeline (recovered S) and exact S (as in M0)
    H = {"pipeline": pipeline_templates(dev),
         "exact": np.array([sj.feature_response(sj.refocus(sj.exact_smatrix(cd.SETUP, [cd.cluster(cd.TEMPLATE_SEED)[0]],
                                                                            [z], cd.THICK), cd.SETUP, cd.DEPTHS,
                                                           device=dev), tw) for z in TEMPLATE_DEPTHS])}
    kw = dict(refine=TEMPLATE_DEPTHS[1] - TEMPLATE_DEPTHS[0], step=1.0)
    rs = responses([f"test_z{z:.0f}_n{n}" for n in TEST_NOISE for z in TEST_DEPTHS], w, dev)
    rp = responses([f"pair_d{d:.0f}" for d in PAIR_SEPS], w, dev)

    report = {"single": [], "floor": {}, "negative_control": {}, "pairs": [], "reference_plane": []}
    for lib, h in H.items():
        res0 = []
        for n in TEST_NOISE:
            for z in TEST_DEPTHS:
                (zf,), res = sj.fit_depths_joint(rs[f"test_z{z:.0f}_n{n}"], h, TEMPLATE_DEPTHS, 1, **kw)
                report["single"].append(dict(templates=lib, z0=z, noise=n, depth=zf, error=zf - z, residual=res))
                if n == TEST_NOISE[0]:
                    res0.append(res)
        floor = 2.0 * max(res0)                       # calibrated on noise realisation 0 only
        report["floor"][lib] = floor
        nc = []
        for z in TEST_DEPTHS:                         # negative control on the held-out realisation
            zs, _ = sj.select_n_features(rs[f"test_z{z:.0f}_n{TEST_NOISE[-1]}"], h, TEMPLATE_DEPTHS, floor,
                                         max_features=2, **kw)
            nc.append(dict(z0=z, n_found=len(zs), depths=zs))
        report["negative_control"][lib] = nc
        for d in PAIR_SEPS:
            truth = [150.0 - d / 2, 150.0 + d / 2]
            zs, fits = sj.select_n_features(rp[f"pair_d{d:.0f}"], h, TEMPLATE_DEPTHS, floor, max_features=2, **kw)
            report["pairs"].append(dict(templates=lib, sep=d, truth=truth, selected=zs, n_found=len(zs),
                                        two_feature_fit=fits[2][0], res1=fits[1][1], res2=fits[2][1]))
    dz = np.arange(-60.0, 60.0 + 1e-6, 5.0)
    for z in TEST_DEPTHS:
        S = np.load(os.path.join(OUT, f"S_test_z{z:.0f}_n0.npy"))
        report["reference_plane"].append(dict(z0=z, dz=sj.reference_plane(
            S, sj.exact_smatrix(cd.SETUP, [img], [z], cd.THICK).numpy(), cd.SETUP, dz, cd.CROP)[0]))
    json.dump(report, open(os.path.join(OUT, "M1_report.json"), "w"), indent=1, default=float)
    write_markdown(report)
    print(open(os.path.join(OUT, "M1_RESULTS.md")).read())


def write_markdown(rep):
    L = ["# M1 pilot results (generated by calibrate_m1.py analyze)", "",
         f"Templates every {TEMPLATE_DEPTHS[1] - TEMPLATE_DEPTHS[0]:.0f} A, 0-{cd.THICK:.0f} A; held-out cluster "
         f"{cd.TEST_SEED}; noise realisation {TEST_NOISE[0]} calibrates the floor, {TEST_NOISE[-1]} tests it.", "",
         "## Single feature: fitted depth error (A) and residual", "",
         "| templates | z0 | noise | depth | error | residual |", "|---|---|---|---|---|---|"]
    L += [f"| {r['templates']} | {r['z0']:.0f} | {r['noise']} | {r['depth']:.1f} | {r['error']:+.1f} | {r['residual']:.4f} |"
          for r in rep["single"]]
    L += ["", "## Residual floor (2 x max single-feature residual, noise realisation 0)", ""]
    L += [f"- {lib}: {f:.4f}" for lib, f in rep["floor"].items()]
    L += ["", "## Negative control: single clusters (held-out noise realisation), features found", "",
          "| templates | z0 | features found | depths |", "|---|---|---|---|"]
    L += [f"| {lib} | {r['z0']:.0f} | {r['n_found']} | {np.round(r['depths'], 1).tolist()} |"
          for lib, rows in rep["negative_control"].items() for r in rows]
    L += ["", "## Two clusters in one column at 150 -/+ d/2", "",
          "| templates | d | truth | selected | 2-feature fit | res 1 | res 2 |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['templates']} | {r['sep']:.0f} | {r['truth']} | {np.round(r['selected'], 1).tolist()} | "
          f"{np.round(r['two_feature_fit'], 1).tolist()} | {r['res1']:.4f} | {r['res2']:.4f} |" for r in rep["pairs"]]
    L += ["", "## Reference plane of recovered S (A, 5 A grid)", ""]
    L += [f"- z0 {r['z0']:.0f}: {r['dz']:+.0f}" for r in rep["reference_plane"]]
    open(os.path.join(OUT, "M1_RESULTS.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["templates", "test", "pairs", "analyze", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16, help="patterns per Adam step (16 = M0; keep for comparability)")
    ap.add_argument("--dose", type=float, default=1e5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    print(f"device {a.device}" + (f" ({torch.cuda.get_device_name(0)})" if str(a.device).startswith("cuda") else ""),
          flush=True)
    pos0, nom0, _ = cd.problem()
    sj.check_geometry(cd.SETUP, pos0, nom0, cd.THICK)
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
