"""D4: periodic-window model error vs aperture edge (hard vs linear taper of width TAPER mrad), K = 48 and 96."""
import sys, os, json, numpy as np, abtem, torch
from ase import Atoms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench, validate_forward as vf
truth = json.load(open(os.path.join(bench.HERE, "data/benchmark_200A/au_single/A_z100/ground_truth/truth.json")))
true = np.array(truth["aberrations"]["true"])
Y = 192; setup = bench.sj.Setup(energy_ev=300e3, alpha=0.02, dr=0.2, K=48, Y=Y)
au = np.array(truth["features"][0]["positions"]) + [9.6, 9.6, 0]
atoms = Atoms("Au3", positions=au, cell=[Y * 0.2, Y * 0.2, 200.0], pbc=[True, True, False])
V = np.asarray(abtem.Potential(atoms, gpts=(Y, Y), slice_thickness=2.0).build().compute(progress_bar=False).array, np.float64)
phase = truth["potential"]["sigma_rad_per_V_A"] * V
c = int(round(au[:, :2].mean(0)[0] / 0.2))
pos = np.array([(c + i, c + j) for i in range(-12, 13, 6) for j in range(-12, 13, 6)])
q = np.fft.fftfreq(Y, 0.2); qy, qx = np.meshgrid(q, q, indexing="ij")
B = bench.sj.aberration_basis(torch.as_tensor(qy.ravel()), torch.as_tensor(qx.ravel()), setup.lam).numpy()
a_mrad = np.hypot(qy, qx) * setup.lam * 1e3
def probe(cf, taper):
    w = np.clip((20.0 - a_mrad) / taper + 0.5, 0, 1) if taper > 0 else (a_mrad <= 20.0).astype(float)
    p = np.fft.ifft2(w * np.exp(-1j * (cf @ B).reshape(Y, Y)))
    return p / np.sqrt((np.abs(p) ** 2).sum())
print("taper_mrad K | L(periodic vs truth) mean | Au signal mean | ratio | intensity outside window radius")
for taper in (0.0, 1.0, 2.0, 4.0):
    for K in (48, 96):
        onK = (np.abs(qy * K * 0.2 - np.round(qy * K * 0.2)) < 1e-6) & (np.abs(qx * K * 0.2 - np.round(qx * K * 0.2)) < 1e-6)
        sK = bench.sj.Setup(energy_ev=300e3, alpha=0.02, dr=0.2, K=K, Y=Y)
        per, sig, outside = [], [], []
        for d in range(3):
            p0 = probe(true[d], taper)
            r = np.hypot(*np.meshgrid(np.minimum(np.arange(Y), Y - np.arange(Y)), np.minimum(np.arange(Y), Y - np.arange(Y)), indexing="ij"))
            outside.append(float((np.abs(p0) ** 2)[r > K / 2].sum()))
            mk = lambda p, ph: vf.multislice_torch(setup, ph, np.stack([np.roll(p, (int(a), int(b)), axis=(0, 1)) for a, b in pos]), 2.0, True, "cpu").numpy()
            iso = vf.window_patterns(sK, mk(p0, phase), pos)
            per_ = vf.window_patterns(sK, mk(np.fft.ifft2(np.fft.fft2(p0) * onK), phase), pos)
            vac = vf.window_patterns(sK, mk(p0, 0 * phase), pos)
            s2 = (np.sqrt(per_) * np.sqrt(iso)).sum() / per_.sum()
            per.append(vf.ldata(per_ * s2 ** 2, iso)["L_data"]); sig.append(vf.ldata(vac, iso)["L_data"])
        print(f"{taper:4.1f} {K:3d} | {np.mean(per):.2e} | {np.mean(sig):.2e} | {np.mean(per) / np.mean(sig):6.2f} | {np.mean(outside):.4f}", flush=True)
