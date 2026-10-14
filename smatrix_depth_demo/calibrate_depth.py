"""M0: where does the depth bias come from, and where is the recovered S-matrix's reference plane?

One small cluster of phase 'atoms' at a known depth z0 in a 300 A slab, three ways:

    A  exact S (entrance-referenced)               -> readout error only (refocusing + depth estimator)
    B  S recovered from simulated data, true probe -> + S-recovery error
    C  S and probe recovered jointly               -> + probe-estimation error

Depth estimators compared:
    peak      depth of maximum correlation between the section phase and the true feature (the earlier readout)
    template  complex template fit (simple_joint.fit_depth) against templates of a DIFFERENT cluster built from exact S

    python calibrate_depth.py a                         # test A, 7 depths (minutes)
    python calibrate_depth.py fit --depths 30,150,270   # tests B and C: one reconstruction per depth and variant
    python calibrate_depth.py report                    # A/B/C table + reference plane of every recovered S

Outputs go to results_m0/ (S-matrices as .npy, summary as JSON).
"""
import argparse, json, os, time
import numpy as np
import torch

import simple_joint as sj
import demo_simple_joint as demo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_m0")
THICK = 300.0
SETUP = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.3, K=40, Y=80)
DEPTHS = np.arange(-60.0, THICK + 60.0 + 1e-6, 5.0)       # refocus grid: specimen plus margins
Z0_GRID = np.arange(0.0, THICK + 1e-6, 10.0)              # template depths
TEMPLATE_SEED, TEST_SEED = 11, 22                         # templates and tests use different clusters
CROP = (slice(SETUP.K // 2, SETUP.Y - SETUP.K // 2),) * 2  # scanned region


def cluster(seed, n_atoms=3, spread=2.0, phase=1.0, min_dist=1.5):
    """Phase image (Y, Y) of `n_atoms` Gaussian atoms within +-spread A of the centre, and its lateral weight."""
    rng = np.random.default_rng(seed)
    c = SETUP.Y * SETUP.dr / 2
    r = np.arange(SETUP.Y) * SETUP.dr
    pts = []
    while len(pts) < n_atoms:
        p = c + rng.uniform(-spread, spread, 2)
        if all(np.hypot(*(p - q)) >= min_dist for q in pts):
            pts.append(p)
    img = sum(np.exp(-((r[:, None] - y) ** 2 + (r[None, :] - x) ** 2) / (2 * 0.45 ** 2)) for y, x in pts)
    return phase * img, img / img.max()


def response_of(S, weight):
    return sj.feature_response(sj.refocus(S, SETUP, DEPTHS), weight)


def peak_depth(S, img):
    """Earlier readout: depth where the section phase correlates best with the true feature image."""
    return demo.layer_depths(sj.refocus(S, SETUP, DEPTHS), [img], DEPTHS, CROP)[0][0]


def templates():
    f = os.path.join(OUT, "templates.npz")
    if os.path.exists(f):
        return np.load(f)["h"]
    img, w = cluster(TEMPLATE_SEED)
    h = np.array([response_of(sj.exact_smatrix(SETUP, [img], [z0], THICK), w) for z0 in Z0_GRID])
    os.makedirs(OUT, exist_ok=True)
    np.savez(f, h=h, z0=Z0_GRID, depths=DEPTHS)
    return h


def test_a(depths):
    h = templates()
    img, w = cluster(TEST_SEED)
    rows = []
    for z0 in depths:
        S = sj.exact_smatrix(SETUP, [img], [z0], THICK)
        zt = sj.fit_depth(response_of(S, w), h, Z0_GRID)[0]
        rows.append(dict(z0=z0, peak=peak_depth(S, img), template=zt))
        print(f"A  z0 {z0:5.0f}   peak {rows[-1]['peak']:6.1f}   template {zt:6.1f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "test_a.json"), "w"), indent=1)


def problem():
    """Acquisition as in demo_simple_joint: 3 defoci, relative errors on datasets 1 and 2, 0.3 A scan step."""
    s0 = SETUP.K // 2
    g = np.arange(s0, SETUP.Y - s0 + 1, 1)
    positions = np.stack(np.meshgrid(g, g, indexing="ij"), -1)
    nominal = np.zeros((3, len(sj.NAMES)))
    nominal[:, 0] = [-250.0, -150.0, -50.0]
    true = nominal.copy()
    true[1, :3] += [12.0, 6.0, -4.0]
    true[2, :3] += [-10.0, -3.0, 5.0]
    return positions, nominal, true


def run_fits(depths, variants, epochs, dose, device):
    positions, nominal, true = problem()
    img, _ = cluster(TEST_SEED)
    for z0 in depths:
        dps = demo.simulate(SETUP, [img], [z0], THICK, true, positions, dose=dose, device=device)
        amp, pos, didx = sj.prepare_data(dps, positions, device=device)
        for v in variants:
            f = os.path.join(OUT, f"S_{v}_z{z0:.0f}.npy")
            if os.path.exists(f):
                continue
            init, train = (true, ()) if v == "B" else (nominal, ("C10", "C12a", "C12b"))
            model = sj.JointModel(SETUP, pos, didx, init, trainable=train, device=device)
            t0 = time.time()
            hist = sj.fit(model, amp, epochs=epochs, log=lambda m: None)
            np.save(f, model.S.detach().cpu().numpy())
            json.dump(dict(z0=z0, variant=v, loss=hist[-1], runtime=time.time() - t0,
                           coeff_err=(model.coefficients().detach().cpu().numpy() - true)[1:, :3].tolist()),
                      open(f[:-4] + ".json", "w"), indent=1)
            print(f"{v}  z0 {z0:5.0f}  loss {hist[-1]:.6f}  {time.time() - t0:.0f}s", flush=True)


def report():
    h = templates()
    img, w = cluster(TEST_SEED)
    rows = {r["z0"]: dict(A_peak=r["peak"], A_template=r["template"])
            for r in json.load(open(os.path.join(OUT, "test_a.json")))}
    dz = np.arange(-60.0, 60.0 + 1e-6, 5.0)
    for name in sorted(os.listdir(OUT)):
        if not (name.startswith("S_") and name.endswith(".npy")):
            continue
        v, z0 = name[2], float(name.split("_z")[1][:-4])
        S = np.load(os.path.join(OUT, name))
        r = rows.setdefault(z0, {})
        r[f"{v}_peak"] = peak_depth(S, img)
        r[f"{v}_template"] = sj.fit_depth(response_of(S, w), h, Z0_GRID)[0]
        r[f"{v}_ref_plane"] = sj.reference_plane(S, sj.exact_smatrix(SETUP, [img], [z0], THICK), SETUP, dz, CROP)[0]
    json.dump({str(k): v for k, v in rows.items()}, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    cols = ["A_peak", "A_template", "B_peak", "B_template", "B_ref_plane", "C_peak", "C_template", "C_ref_plane"]
    print("z0     " + "  ".join(f"{c:>11s}" for c in cols))
    for z0 in sorted(rows):
        print(f"{z0:5.0f}  " + "  ".join(f"{rows[z0][c]:11.1f}" if c in rows[z0] else f"{'-':>11s}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a", "fit", "report"])
    ap.add_argument("--depths", default="30,75,115,150,185,225,270")
    ap.add_argument("--variants", default="B,C")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dose", type=float, default=1e5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    depths = [float(d) for d in a.depths.split(",")]
    if a.cmd == "a":
        test_a(depths)
    elif a.cmd == "fit":
        run_fits(depths, a.variants.split(","), a.epochs, a.dose, a.device)
    else:
        report()


if __name__ == "__main__":
    main()
