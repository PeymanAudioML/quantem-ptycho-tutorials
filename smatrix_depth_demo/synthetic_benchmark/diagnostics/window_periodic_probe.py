"""D1: is the S-matrix model's error the periodic probe (beams on the 1/(K dr) grid)?"""
import sys, os, json, numpy as np, torch, zarr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench, validate_forward as vf
cfg = bench.load_config(); setup = bench.make_setup(cfg)
sd = os.path.join(bench.HERE, "data/au_single/A_z100"); gt = sd + "/ground_truth"
truth = json.load(open(gt + "/truth.json"))
V = zarr.open_array(gt + "/potential.zarr/array0", mode="r")[...]
phase = truth["potential"]["sigma_rad_per_V_A"] * V.astype(np.float64)
true = np.array(truth["aberrations"]["true"]); pos = bench.scan_positions(cfg).reshape(-1, 2)
meas = vf.load(sd, "window")
Y, K = setup.Y, setup.K
q = np.fft.fftfreq(Y, setup.dr); qy, qx = np.meshgrid(q, q, indexing="ij")
onK = (np.round(qy * K * setup.dr, 6) % 1 == 0) & (np.round(qx * K * setup.dr, 6) % 1 == 0)
kk = (np.arange(K) - K // 2) / (K * setup.dr); bf = (np.hypot(kk[:, None], kk[None, :]) <= setup.alpha / setup.lam).astype(float)
for d in range(3):
    p0 = bench.ad.entrance_probe(setup, true[d]).astype(np.complex128)
    F = np.fft.fft2(p0); pper = np.fft.ifft2(F * onK)            # same probe, only the K-grid Fourier components = periodic copies
    res = {}
    for nm, p in (("isolated (true)", p0), ("periodic K-grid", pper), ("vacuum isolated", None)):
        out = []
        for b0 in range(0, len(pos), 125):
            ch = pos[b0:b0 + 125]
            w0 = np.stack([np.roll(p0 if p is None else p, (int(r0), int(r1)), axis=(0, 1)) for r0, r1 in ch])
            ex = vf.multislice_torch(setup, phase * (0 if p is None else 1), w0, 2.0, True, "cpu").numpy()
            out.append(vf.window_patterns(setup, ex, ch))
        P = np.concatenate(out).reshape(meas[d].shape)
        s2 = (np.sqrt(P) * np.sqrt(meas[d])).sum() / P.sum()
        res[nm] = (vf.ldata(P, meas[d], bf)["L_data"], vf.ldata(P * s2**2, meas[d], bf)["L_data"])
    print(f"def{d}: " + "; ".join(f"{k}: L {v[0]:.2e} (scaled {v[1]:.2e})" for k, v in res.items()), flush=True)
