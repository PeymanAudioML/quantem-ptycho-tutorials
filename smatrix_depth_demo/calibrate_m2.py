"""M2: one- versus two-feature fitting.  How close can two features be in depth and still be found as two, and how
often is a single feature wrongly split?  (GPU-ready; uses the M1 pipeline templates.)

Reconstructions (same setup and joint fit as M0/M1, held-out cluster 22):
  pairs    two identical clusters in one column at c -/+ d/2, for every separation d and centre c
  singles  one cluster at several depths, several noise realisations: realisation 0 calibrates the residual floor,
           the others are the negative controls (false-split rate)

Analysis, for pipeline templates and, as the noise-free ideal, the exact S of every configuration:
  separability   fraction of pairs found as two features with both depths within tolerance, per separation
  depth errors   of the two fitted features
  false splits   fraction of single clusters reported as two

    python calibrate_m1.py templates --device cuda    # M1 templates are required first
    python calibrate_m2.py all --device cuda          # pairs + singles + analyze
    python calibrate_m2.py analyze                    # results_m2/M2_RESULTS.md and M2_report.json

Lists are adjustable, e.g. --seps 20,40,60 --centres 150 --noise 0,1.  Finished reconstructions are skipped.
"""
import argparse, json, os
import numpy as np
import torch

import simple_joint as sj
import calibrate_depth as cd
import calibrate_m1 as m1

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_m2")


def floats(s):
    return [float(v) for v in s.split(",")]


def pair_tag(d, c, n):
    return f"pair_d{d:.0f}_c{c:.0f}_n{n}"


def single_tag(z, n):
    return f"single_z{z:.0f}_n{n}"


def noise_seed(tag):
    """Distinct, reproducible noise seed per run (M1 uses 0, 1 and 100+)."""
    return 1000 + int.from_bytes(tag.encode(), "little") % 1_000_000


def pair_depths(d, c):
    return [c - d / 2, c + d / 2]


def valid_pair(d, c):
    lo, hi = pair_depths(d, c)
    return lo >= 0 and hi <= cd.THICK


def run_pairs(a):
    img, _ = cd.cluster(cd.TEST_SEED)
    for n in a.noise:
        for c in a.centres:
            for d in a.seps:
                if valid_pair(d, c):
                    t = pair_tag(d, c, n)
                    m1.reconstruct(t, [img, img], pair_depths(d, c), noise_seed(t), a, out=OUT)


def run_singles(a):
    img, _ = cd.cluster(cd.TEST_SEED)
    for n in a.single_noise:
        for z in a.single_depths:
            t = single_tag(z, n)
            m1.reconstruct(t, [img], [z], noise_seed(t), a, out=OUT)


def exact_response(layer_z, w, device):
    img, _ = cd.cluster(cd.TEST_SEED)
    S = sj.exact_smatrix(cd.SETUP, [img] * len(layer_z), layer_z, cd.THICK)
    return sj.feature_response(sj.refocus(S, cd.SETUP, cd.DEPTHS, device=device), w)


def judge(found, truth, tol):
    """Two features found, matched in depth order, both within tol."""
    if len(found) != 2:
        return False, [None, None]
    err = [f - t for f, t in zip(sorted(found), truth)]
    return all(abs(e) <= tol for e in err), err


def analyze(a):
    dev = a.device
    _, w = cd.cluster(cd.TEST_SEED)
    grid = m1.TEMPLATE_DEPTHS
    kw = dict(refine=grid[1] - grid[0], step=1.0, min_sep=a.min_sep)
    H = {"pipeline": m1.pipeline_templates(dev)}
    _, tw = cd.cluster(cd.TEMPLATE_SEED)
    H["exact"] = np.array([sj.feature_response(sj.refocus(sj.exact_smatrix(
        cd.SETUP, [cd.cluster(cd.TEMPLATE_SEED)[0]], [z], cd.THICK), cd.SETUP, cd.DEPTHS, device=dev), tw) for z in grid])

    have = lambda t: os.path.exists(os.path.join(OUT, f"S_{t}.npy"))
    singles = [(z, n) for n in a.single_noise for z in a.single_depths if have(single_tag(z, n))]
    pairs = [(d, c, n) for n in a.noise for c in a.centres for d in a.seps if valid_pair(d, c) and have(pair_tag(d, c, n))]
    if not singles or not pairs:
        print("no reconstructions found; run 'pairs' and 'singles' first")
        return
    calib = [s for s in singles if s[1] == a.single_noise[0]]
    tests = [s for s in singles if s[1] != a.single_noise[0]]
    rs = m1.responses([single_tag(z, n) for z, n in singles], w, dev, out=OUT)
    rp = m1.responses([pair_tag(d, c, n) for d, c, n in pairs], w, dev, out=OUT)

    rep = {"settings": dict(seps=a.seps, centres=a.centres, noise=a.noise, single_depths=a.single_depths,
                            single_noise=a.single_noise, tol=a.tol, min_sep=a.min_sep), "pipeline": {}, "exact": {}}
    for lib, h in H.items():
        if lib == "pipeline":
            res1 = lambda t, r: sj.fit_depths_joint(r, h, grid, 1, **kw)[1]
            floor = a.floor_factor * max(res1(None, rs[single_tag(z, n)]) for z, n in calib)
            single_resp = {s: rs[single_tag(*s)] for s in tests}
            pair_resp = {p: rp[pair_tag(*p)] for p in pairs}
        else:                                         # noise-free ideal: exact S of the same configurations
            single_resp = {(z, 0): exact_response([z], w, dev) for z in a.single_depths}
            pair_resp = {(d, c, 0): exact_response(pair_depths(d, c), w, dev)
                         for d in a.seps for c in a.centres if valid_pair(d, c)}
            floor = a.floor_factor * max(sj.fit_depths_joint(r, h, grid, 1, **kw)[1] for r in single_resp.values())
        out = rep[lib]
        out["floor"] = floor
        out["singles"] = []
        for (z, n), r in single_resp.items():
            zs, fits = sj.select_n_features(r, h, grid, floor, max_features=2, **kw)
            out["singles"].append(dict(z0=z, noise=n, n_found=len(zs), depths=zs, res1=fits[1][1], res2=fits[2][1]))
        out["pairs"] = []
        for (d, c, n), r in pair_resp.items():
            zs, fits = sj.select_n_features(r, h, grid, floor, max_features=2, **kw)
            ok, err = judge(zs, pair_depths(d, c), max(a.tol, 0.25 * d))
            out["pairs"].append(dict(sep=d, centre=c, noise=n, truth=pair_depths(d, c), found=zs, n_found=len(zs),
                                     correct=ok, errors=err, two_feature_fit=fits[2][0], res1=fits[1][1], res2=fits[2][1]))
        out["false_split_rate"] = float(np.mean([s["n_found"] > 1 for s in out["singles"]])) if out["singles"] else None
        out["separability"] = {str(d): float(np.mean([p["correct"] for p in out["pairs"] if p["sep"] == d]))
                               for d in a.seps if any(p["sep"] == d for p in out["pairs"])}
    json.dump(rep, open(os.path.join(OUT, "M2_report.json"), "w"), indent=1, default=float)
    write_markdown(rep)
    print(open(os.path.join(OUT, "M2_RESULTS.md")).read())


def write_markdown(rep):
    st = rep["settings"]
    L = ["# M2 results (generated by calibrate_m2.py analyze)", "",
         f"Pairs of identical clusters in one column at c -/+ d/2; tolerance max({st['tol']:.0f} A, d/4). "
         "Pipeline = M1 templates on reconstructed data; exact = exact-S templates on exact S (noise-free ideal; its floor "
         "comes from the same single clusters, so its false-split rate is not a test).", ""]
    for lib in ("pipeline", "exact"):
        r = rep[lib]
        L += [f"## {lib}", "", f"Residual floor {r['floor']:.4f}; false-split rate on single clusters "
              f"{r['false_split_rate']:.2f} ({len(r['singles'])} controls).", "",
              "| separation d (A) | fraction found as two, both within tolerance |", "|---|---|"]
        L += [f"| {d} | {v:.2f} |" for d, v in r["separability"].items()]
        L += ["", "| d | centre | noise | truth | found | errors | res 1 | res 2 |", "|---|---|---|---|---|---|---|---|"]
        for p in sorted(r["pairs"], key=lambda p: (p["sep"], p["centre"], p["noise"])):
            err = "-" if p["errors"][0] is None else str(np.round(p["errors"], 1).tolist())
            L.append(f"| {p['sep']:.0f} | {p['centre']:.0f} | {p['noise']} | {p['truth']} | "
                     f"{np.round(p['found'], 1).tolist()} | {err} | {p['res1']:.4f} | {p['res2']:.4f} |")
        L += ["", "| single z0 | noise | features found | depths | res 1 |", "|---|---|---|---|---|"]
        L += [f"| {s['z0']:.0f} | {s['noise']} | {s['n_found']} | {np.round(s['depths'], 1).tolist()} | {s['res1']:.4f} |"
              for s in r["singles"]]
        L.append("")
    open(os.path.join(OUT, "M2_RESULTS.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "singles", "analyze", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seps", type=floats, default=floats("20,40,60,80,120"))
    ap.add_argument("--centres", type=floats, default=floats("100,150,200"))
    ap.add_argument("--noise", type=lambda s: [int(v) for v in s.split(",")], default=[0])
    ap.add_argument("--single-depths", type=floats, default=floats("60,100,150,200,240"))
    ap.add_argument("--single-noise", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2],
                    help="first realisation calibrates the floor, the rest are negative controls")
    ap.add_argument("--tol", type=float, default=10.0, help="depth tolerance [A] (or d/4 if larger)")
    ap.add_argument("--min-sep", type=float, default=10.0, help="smallest separation the two-feature fit may return")
    ap.add_argument("--floor-factor", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dose", type=float, default=1e5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    pos0, nom0, _ = cd.problem()
    sj.check_geometry(cd.SETUP, pos0, nom0, cd.THICK)
    if a.cmd in ("pairs", "all"):
        run_pairs(a)
    if a.cmd in ("singles", "all"):
        run_singles(a)
    if a.cmd in ("analyze", "all"):
        analyze(a)


if __name__ == "__main__":
    main()
