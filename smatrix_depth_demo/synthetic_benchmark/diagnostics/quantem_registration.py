"""D2: QuantEM truth-forward loss vs lateral shift of the truth object (registration check), window def1."""
import sys, os, json, numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench, run_realization as rr, forward_check as fc, quantem_export as qe
cfg = bench.load_config(os.path.join(bench.HERE, "configs", "benchmark_200A.yaml")); setup = bench.make_setup(cfg)
sd = os.path.join(bench.HERE, "data/benchmark_200A/au_single/A_z100"); truth = json.load(open(sd + "/ground_truth/truth.json"))
true = np.array(truth["aberrations"]["true"]); T = 200.0; d = 1; K = 48
ddir = f"{sd}/window_noiseless/def{d}"
d0 = rr.load_dataset(ddir, SimpleNamespace(energy=None, alpha=None, defocus=None), d)
pdset = fc.setup_dataset(d0, setup.lam)
tr = rr.load_truth([ddir], lambda *_: None); pot, dr_t, st = tr["potential"]; edges = np.r_[0, np.cumsum(st)]
sig = truth["potential"]["sigma_rad_per_V_A"]
g1 = qe.SliceGrid(boundaries=np.array([0.0, T / 2, T]), thickness=T)
p_true = bench.ad.entrance_probe(setup, true[d])
pt0 = fc.make_pt(pdset, d0, qe.to_quantem(np.full((2, 64, 64), 1e-3, np.float32), g1), fc.truth_probe_on_grid((p_true, 0.2), (K, K), 0.2, np.ones((K, K))), "cpu")
ref = np.asarray(pt0.probe)[0]; tp = fc.truth_probe_on_grid((p_true, 0.2), ref.shape, 0.2, ref)
print("QuantEM probe roi", ref.shape, "| scan px min", pt0.dset.scan_positions_px.min(0).values.tolist(), "max", pt0.dset.scan_positions_px.max(0).values.tolist())
print("QuantEM initial probe |p| argmax (centring):", np.unravel_index(np.argmax(np.abs(ref)), ref.shape))
pos0 = d0["positions_A"][0, 0]; Q = pt0.dset.targets.shape[-1]
kq = (np.arange(Q) - Q // 2) * float(d0["dk"][0]); bfq = (np.hypot(kq[:, None], kq[None, :]) <= setup.alpha / setup.lam).astype(np.float32)
g = qe.SliceGrid(boundaries=edges, thickness=T)
def L(shift):
    rows = pos0[0] + (np.arange(64) - 8 + shift[0]) * 0.2; cols = pos0[1] + (np.arange(64) - 8 + shift[1]) * 0.2
    phi = np.stack([rr.sample_truth(pot, dr_t, rows, cols, edges[k] - 1e-6, edges[k + 1] - 1e-6, st, 0.2) * sig for k in range(len(edges) - 1)]).astype(np.float32)
    return fc.predict(fc.make_pt(pdset, d0, qe.to_quantem(phi, g), tp, "cpu"), bfq)["L_data"]
vac = fc.predict(fc.make_pt(pdset, d0, qe.to_quantem(np.zeros((2, 64, 64), np.float32), g1), tp, "cpu"), bfq)["L_data"]
print(f"vacuum {vac:.3e}")
for s in [(0, 0), (24, 24), (-24, -24), (24, 0), (0, 24), (1, 0), (-1, 0), (0, 1), (0, -1), (12, 12), (-12, -12)]:
    print(s, f"{L(s):.3e}", flush=True)
