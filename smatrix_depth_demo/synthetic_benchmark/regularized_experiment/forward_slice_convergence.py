"""Forward-model error vs slice thickness in QuantEM's own multislice, on the true potential (no reconstruction).

QuantEM's forward model (quantem 0f60f90): N slices, transmission exp(i * phi_k) applied at plane k, Fresnel propagator
exp(-i pi lambda dz_k k^2) between planes k and k+1 (N - 1 propagators), no propagation after the last slice, no
anti-aliasing aperture.  Our runner puts plane k at slab entrance b_k with the probe defined at the entrance surface.

For N uniform slabs (boundaries snapped to the 2 A abTEM slices, so each slab holds whole slices and the propagation
distance matches its content), the true potential summed per slab is pushed through QuantEM's forward path with the true
probe and compared with the abTEM patterns (2 A slices, conventional multislice, 2/3 anti-aliasing):
  entrance:  probe at the entrance surface, as our runs do (first-order placement: each slab acts at its top)
  midpoint:  probe first propagated dz/2 into the specimen, so each slab acts at its centre.  With uniform slabs this
             is exactly symmetric (Strang) splitting: the final half step does not change the intensities.
  shared xm: the same N slab potentials, each spread evenly over m sub-steps of ~2 A (phi_k / m per sub-step): fine
             propagation with N independent maps (the "adaptive slicing" idea).
Metric: L_data = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas (validate_forward.py F3), and QuantEM's own loss.

Run from synthetic_benchmark/:
  ../../.venv/bin/python regularized_experiment/forward_slice_convergence.py --specimen P3_XYZ --dataset 1 --device cuda
"""
import argparse, json, os, sys, time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
os.chdir(BENCH)
import bench  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def fresnel(p, dr, lam, dz):
    k = [np.fft.fftfreq(n, dr) for n in p.shape]
    k2 = k[0][:, None] ** 2 + k[1][None] ** 2
    return np.fft.ifft2(np.fft.fft2(p) * np.exp(-1j * np.pi * lam * dz * k2)).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poly_si_3grain.yaml")
    ap.add_argument("--specimen", default="P3_XYZ")
    ap.add_argument("--dataset", type=int, default=1)
    ap.add_argument("--slabs", default="100,50,35,25,20,10,5,2,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "forward_slice_convergence"))
    a = ap.parse_args()
    import run_realization as rr
    import forward_check as fc
    import quantem_export as qe
    cfg = bench.load_config(a.config)
    setup = bench.make_setup(cfg)
    sd = bench.specimen_dir(cfg, a.specimen)
    truth = json.load(open(os.path.join(sd, "ground_truth", "truth.json")))
    sigma, T, d = truth["potential"]["sigma_rad_per_V_A"], truth["thickness_A"], a.dataset
    true = np.array(truth["aberrations"]["true"])
    ddir = os.path.join(sd, "window_noiseless", f"def{d}")
    d0 = rr.load_dataset(ddir, SimpleNamespace(energy=None, alpha=None, defocus=None), d)
    pdset = fc.setup_dataset(d0, setup.lam)
    K = setup.K
    g1 = qe.SliceGrid(boundaries=np.array([0.0, T / 2, T]), thickness=T)
    tmp = fc.make_pt(pdset, d0, qe.to_quantem(np.full((2, 8, 8), 1e-3, np.float32), g1),
                     np.zeros((K, K), np.complex64) + 1e-3, a.device)
    shape = tuple(int(v) for v in tmp.obj_shape_full[1:])
    pt0 = fc.make_pt(pdset, d0, qe.to_quantem(np.full((2,) + shape, 1e-3, np.float32), g1),
                     fc.truth_probe_on_grid((bench.ad.entrance_probe(setup, true[d]), setup.dr), (K, K), setup.dr,
                                            np.ones((K, K))), a.device)
    dr_q = float(np.asarray(pt0.sampling)[0])
    p0 = pt0.dset.scan_positions_px.detach().cpu().numpy().min(axis=0)
    pos0 = d0["positions_A"][0, 0]
    rows = pos0[0] + (np.arange(shape[0]) - p0[0]) * dr_q
    cols = pos0[1] + (np.arange(shape[1]) - p0[1]) * dr_q
    pot, dr_t, st = rr.load_truth([ddir], lambda *_: None)["potential"]
    edges_t = np.r_[0, np.cumsum(st)]
    ref = np.asarray(pt0.probe)[0]
    tp = fc.truth_probe_on_grid((bench.ad.entrance_probe(setup, true[d]), setup.dr), ref.shape, dr_q, ref)
    Q = pt0.dset.targets.shape[-1]
    kq = (np.arange(Q) - Q // 2) * float(d0["dk"][0])
    bfq = (np.hypot(kq[:, None], kq[None, :]) <= setup.alpha / setup.lam).astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    rows_out = []
    for n in [int(v) for v in a.slabs.split(",")]:
        b = edges_t[np.round(np.linspace(0, len(st), n + 1)).astype(int)]     # snapped to whole 2 A slices
        phi = np.stack([rr.sample_truth(pot, dr_t, rows, cols, b[k] - 1e-6, b[k + 1] - 1e-6, st, dr_q) * sigma
                        for k in range(n)]).astype(np.float32)
        g = qe.SliceGrid(boundaries=b, thickness=T)
        dz_mean = T / n
        # shared: each slab's potential spread evenly over sub-steps of ~2 A (finer propagation, same N maps)
        m = max(1, int(round((b[1] - b[0]) / 2.0)))
        bs = np.concatenate([np.linspace(b[k], b[k + 1], m + 1)[:-1] for k in range(n)] + [[T]])
        phis = np.repeat(phi / m, m, axis=0)
        cases = [("entrance", phi, g, tp), ("midpoint", phi, g, fresnel(tp, dr_q, setup.lam, float(b[1] - b[0]) / 2))]
        if m > 1:
            cases.append((f"shared x{m}", phis, qe.SliceGrid(boundaries=bs, thickness=T), tp))
        for place, ph_, g_, prb in cases:
            t0 = time.time()
            r = fc.predict(fc.make_pt(pdset, d0, qe.to_quantem(ph_, g_), prb, a.device), bfq)
            r.update(slabs=n, dz_A=dz_mean, placement=place)
            rows_out.append(r)
            print(f"N {n:3d} dz {dz_mean:6.2f} A {place:8s}: L_data {r['L_data']:.3e} (BF {r['L_data_bright_field']:.3e}, "
                  f"DF {r['L_data_dark_field']:.3e}), QuantEM loss {r['quantem_loss']:.2f}  [{time.time() - t0:.0f} s]",
                  flush=True)
    vac = fc.predict(fc.make_pt(pdset, d0, qe.to_quantem(np.zeros((2,) + shape, np.float32), g1), tp, a.device), bfq)
    json.dump(dict(specimen=a.specimen, dataset=d, energy_eV=float(d0["energy"]), wavelength_A=setup.lam,
                   dr_quantem_A=dr_q, rows=rows_out, vacuum=vac), open(os.path.join(a.out, "convergence.json"), "w"),
              indent=1, default=float)
    L = [f"# QuantEM forward error vs slice thickness: {a.specimen}, dataset {d} (true potential, true probe)", "",
         f"Vacuum (no specimen) reference: L_data {vac['L_data']:.3e}, QuantEM loss {vac['quantem_loss']:.1f}.", "",
         "| slabs | dz (Å) | placement | L_data | bright field | dark field | QuantEM loss |", "|---|---|---|---|---|---|---|"]
    for r in rows_out:
        L.append(f"| {r['slabs']} | {r['dz_A']:.2f} | {r['placement']} | {r['L_data']:.3e} | {r['L_data_bright_field']:.3e} | "
                 f"{r['L_data_dark_field']:.3e} | {r['quantem_loss']:.2f} |")
    open(os.path.join(a.out, "convergence.md"), "w").write("\n".join(L) + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for place, ls in (("entrance", "o-"), ("midpoint", "s--"), ("shared", "^-.")):
        rr_ = [r for r in rows_out if r["placement"] == place or (place == "shared" and r["placement"].startswith("shared"))]
        dzs = [r["dz_A"] for r in rr_]
        ax[0].loglog(dzs, [r["quantem_loss"] for r in rr_], ls, label=place)
        ax[1].loglog(dzs, [r["L_data_bright_field"] for r in rr_], ls, label=f"{place}, bright field")
        ax[1].loglog(dzs, [r["L_data_dark_field"] for r in rr_], ls, alpha=0.6, label=f"{place}, dark field")
    ax[0].axhline(vac["quantem_loss"], color="0.5", ls=":", label="vacuum")
    ax[0].set_xlabel("slice thickness dz (Å)"), ax[0].set_ylabel("QuantEM loss (true object, true probe)")
    ax[1].set_xlabel("slice thickness dz (Å)"), ax[1].set_ylabel("L_data")
    for x in ax:
        x.legend(fontsize=7), x.grid(alpha=0.3, which="both")
    fig.suptitle(f"{a.specimen} dataset {d}: QuantEM multislice error vs slice thickness (ground truth through QuantEM)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "convergence.png"), dpi=100)
    print("\n".join(L))


if __name__ == "__main__":
    main()
