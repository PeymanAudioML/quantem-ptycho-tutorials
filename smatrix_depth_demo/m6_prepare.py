"""M6 step 1 (environment with abtem + torch, e.g. equiformer_v3): simulate specimens, run the S-matrix stage and export
everything QuantEM needs to one .npz per specimen.  Step 2 is m6_quantem.py (QuantEM's .venv).

Per specimen (100 A slab, 300 kV, 20 mrad, 0.2 A pixels, 0.4 A scan step, three defoci):
  1. abTEM data for all three defoci (Au cluster at known depths, optional carbon matrix)
  2. joint S-matrix + probe fit (simple_joint)
  3. lateral candidates: detect_features on the refocused volume (no ground truth used)
  4. depth: complex template fit (differential pipeline templates, built here once per Au size / carbon density),
     amplitude-phase limit pi/4; one or two features (select_n_features); reliability from the residual:
     sigma = 8 A if residual <= flag else 30 A (contrast scan: residual > 0.03 always meant > 10 A error)
  5. export: dataset 1's patterns (fftshifted for QuantEM), geometry, the S-matrix probe estimate as QuantEM defocus
     (-C10, M5), candidates (y, x, depth modes, phase), and the truth for scoring only

    python m6_prepare.py --device cuda                  # all specimens -> results_m6/<name>.npz
    python m6_prepare.py --device cuda --only vac_z30   # one specimen
"""
import argparse, json, os, time
import numpy as np
import torch

import simple_joint as sj
import abtem_data as ad
import calibrate_m3 as m3
import contrast_scan as cs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_m6")
T = 100.0
CLUSTER_SEED, ENV_SEED = 33, 3          # held out: templates use cluster 11 / environment 1
QUANTEM_DATASET = 1                     # nominal C10 = -50 A (focus mid-slab)
SPECIMENS = {                           # name: (Au atoms, carbon density, Au depths)
    "vac_z30": (3, 0.0, [30.0]),
    "vac_z70": (3, 0.0, [70.0]),
    "vac_pair": (3, 0.0, [20.0, 80.0]),
    "c025_z50": (3, 0.025, [50.0]),
    "c025_z20": (3, 0.025, [20.0]),
    "c1_z50": (3, 0.1, [50.0]),          # dense carbon: negative control (detection / depth unreliable)
}
FLOOR = {0.0: 0.005}                     # single-feature residual floor: vacuum (M2: 0.0046); carbon uses RESIDUAL_FLAG
RESIDUAL_FLAG = 0.03


def templates(n, rho, a):
    """Differential pipeline templates (5 depths, 25 A apart) for this Au size / density, built once (contrast_scan)."""
    d = cs.combo_dir(n, rho)
    os.makedirs(d, exist_ok=True)
    setup = m3.config(T)[0]
    xt = cs.cluster(setup, m3.TEMPLATE_SEED, n)
    if rho > 0:
        cs.reconstruct(d, "tmpl_bg", cs.specimen(setup, xt, [], m3.ENV_TEMPLATE, rho), 199, a)
    for i, z in enumerate(cs.TEMPLATE_Z):
        cs.reconstruct(d, f"tmpl_z{z:.0f}", cs.specimen(setup, xt, [z], m3.ENV_TEMPLATE, rho), 100 + i, a)
    zg = m3.depth_grid(T)
    tw = sj.footprint(setup, xt)
    V = lambda t: sj.refocus(np.load(os.path.join(d, f"S_{t}.npy")), setup, zg, device=a.device)
    bg = sj.feature_response(V("tmpl_bg"), tw) if rho > 0 else 0.0
    return np.array([sj.feature_response(V(f"tmpl_z{z:.0f}"), tw) - bg for z in cs.TEMPLATE_Z])


def prepare(name, a):
    f = os.path.join(OUT, f"{name}.npz")
    if os.path.exists(f):
        print(f"{name}: exists, skipped", flush=True)
        return
    n, rho, z_au = SPECIMENS[name]
    setup, positions, nominal, true = m3.config(T)
    t0 = time.time()
    h = templates(n, rho, a)
    xy = cs.cluster(setup, CLUSTER_SEED, n)
    atoms = cs.specimen(setup, xy, z_au, ENV_SEED, rho)
    dps = ad.simulate_abtem(setup, atoms, true, positions, dose=a.dose, seed=7, log=lambda *_: None)
    amp, pos, didx = sj.prepare_data(dps, positions, device=a.device)
    model = sj.JointModel(setup, pos, didx, nominal, trainable=("C10", "C12a", "C12b"), device=a.device)
    sj.fit(model, amp, epochs=a.epochs, log=lambda m: None)
    coeff = model.coefficients().detach().cpu().numpy()
    zg = m3.depth_grid(T)
    vol = sj.refocus(model.S, setup, zg, device=a.device)
    det, strength, thr = sj.detect_features(vol, setup)
    cands, flag = [], None
    if len(det):
        r = sj.feature_response(vol, sj.footprint(setup, det))
        kw = dict(refine=cs.TEMPLATE_Z[1] - cs.TEMPLATE_Z[0], step=1.0, min_sep=10.0, max_phase=np.pi / 4)
        floor = FLOOR.get(rho, RESIDUAL_FLAG)
        zs, fits = sj.select_n_features(r, h, cs.TEMPLATE_Z, floor, max_features=2, min_amp_ratio=0.5, **kw)
        res = fits[len(zs)][1]
        flag = bool(res <= RESIDUAL_FLAG)
        sigma = 8.0 if flag else 30.0
        iz = [int(np.argmin(np.abs(zg - z))) for z in zs]
        sec = np.angle(vol[iz] / vol[iz].mean(axis=(1, 2), keepdims=True))
        px = np.round(det / setup.dr).astype(int)
        phase = float(np.clip(np.median(sec[:, px[:, 0], px[:, 1]]), 0.05, 1.0))
        cands = [dict(y=float(y), x=float(x), modes=[(float(z), sigma, 1.0 / len(zs)) for z in zs], phase=phase)
                 for y, x in det]
    data = np.fft.fftshift(dps[QUANTEM_DATASET], axes=(-2, -1)).astype(np.float32)
    np.savez_compressed(f, data=data, scan_step=STEP_A(setup), dk_mrad=1e3 * setup.lam / (setup.K * setup.dr),
                        energy=setup.energy_ev, alpha_mrad=1e3 * setup.alpha, K=setup.K, dr=setup.dr, thickness=T,
                        defocus_quantem=-coeff[QUANTEM_DATASET, 0], defocus_quantem_true=-true[QUANTEM_DATASET, 0],
                        candidates=json.dumps(cands), true_xy=xy, true_z=np.array(z_au), rho=rho, n_au=n,
                        detected=det, residual_ok=json.dumps(flag))
    zs_txt = sorted({m[0] for c in cands for m in c["modes"]}) if cands else []
    print(f"{name}: {len(det)} detected (true {n}), candidate depths {np.round(zs_txt, 1).tolist()} (true {z_au}), "
          f"reliable {flag}, probe C10 err {coeff[QUANTEM_DATASET, 0] - true[QUANTEM_DATASET, 0]:+.1f} A, "
          f"{time.time() - t0:.0f} s", flush=True)


def STEP_A(setup):
    return m3.STEP * setup.dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of specimen names (default: all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dose", type=float, default=1e5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    for name in (a.only.split(",") if a.only else SPECIMENS):
        prepare(name, a)


if __name__ == "__main__":
    main()
