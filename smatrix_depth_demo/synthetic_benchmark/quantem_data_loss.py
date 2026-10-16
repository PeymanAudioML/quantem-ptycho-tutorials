"""Relative amplitude loss of saved QuantEM reconstructions, on the S-matrix scale, so the two stages can be compared:

    L = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas      (forward_check.predict; = the S-matrix's sum(|Z|-a)^2 / sum a^2)

    python quantem_data_loss.py --config configs/poly_si_3grain.yaml --specimen P3_XY --runs <quantem run> ...

The object and probe saved in runs/<config>/<specimen>/<run>/quantem.npz are put back into QuantEM's forward model (no
optimisation) and every pattern of the run's dataset is predicted.  If the run's S-matrix held patterns out, L is also given
on its training and held-out scan positions (QuantEM itself trained on all patterns).  quantem_loss is QuantEM's own loss,
printed as a check against the run's final training loss.
"""
import argparse, json, os
from types import SimpleNamespace

import numpy as np

import bench
import forward_check as fc
import quantem_export as qe
import run_realization as rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    setup = bench.make_setup(cfg)
    T = float(cfg["specimen"]["thickness_A"])
    sd = bench.specimen_dir(cfg, a.specimen)
    rd = os.path.join(bench.HERE, "runs", cfg["name"], a.specimen)
    out = {}
    for run in a.runs:
        s = json.load(open(os.path.join(rd, run, "summary.json")))
        q = np.load(os.path.join(rd, run, "quantem.npz"))
        d0 = rr.load_dataset(os.path.join(sd, f"{s['variant']}_{s['noise']}", f"def{s['dataset']}"),
                             SimpleNamespace(energy=None, alpha=None, defocus=None), s["dataset"])
        pdset = fc.setup_dataset(d0, setup.lam)
        grid = qe.SliceGrid(boundaries=np.asarray(q["slice_boundaries"]), thickness=T)
        pt = fc.make_pt(pdset, d0, qe.to_quantem(q["object_potential"].astype(np.float32), grid), q["probe"], a.device)
        bf = np.ones(tuple(pt.dset.detector_mask.shape))
        res = dict(all=fc.predict(pt, bf)["L_data"], n_all=int(pt.dset.num_gpts))
        chk = fc.predict(pt, bf)["quantem_loss"]
        srun = s.get("smatrix_run")
        arr_f = os.path.join(bench.HERE, "results", "smatrix", cfg["name"], a.specimen, srun or "", "arrays.npz")
        if srun and os.path.exists(arr_f) and "heldout" in np.load(arr_f).files and np.load(arr_f)["heldout"].any():
            hold = np.load(arr_f)["heldout"].ravel()
            grid_A = np.asarray(d0["positions_A"]).reshape(-1, 2)
            q_A = pt.dset.scan_positions_px.detach().cpu().numpy() * s["dr_quantem"] + np.asarray(s["origin_A"])
            d = np.linalg.norm(q_A[:, None, :] - grid_A[None, :, :], axis=-1)
            j = d.argmin(axis=1)
            assert d[np.arange(len(j)), j].max() < 1e-3 and len(set(j.tolist())) == len(j)
            tr, te = np.flatnonzero(~hold[j]), np.flatnonzero(hold[j])
            res.update(smatrix_train=fc.predict(pt, bf, subset=tr)["L_data"], n_smatrix_train=len(tr),
                       smatrix_heldout=fc.predict(pt, bf, subset=te)["L_data"], n_smatrix_heldout=len(te))
        res.update(quantem_loss_check=chk, final_loss_reported=s["final_loss"], smatrix_run=srun)
        out[run] = res
        print(run, json.dumps(res))
    f = os.path.join(bench.HERE, "results", "single_focus_joint", a.specimen, "quantem_data_loss.json")
    json.dump(out, open(f, "w"), indent=1)
    print(f)


if __name__ == "__main__":
    main()
