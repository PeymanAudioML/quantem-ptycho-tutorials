"""M6 step 2 (QuantEM's environment, e.g. .venv/bin/python): four-arm QuantEM comparison on the specimens exported by
m6_prepare.py.  Needs only numpy, torch, quantem and quantem_export.py (pure numpy).

Arms, all with the same number of planes N, iterations, optimiser, constraints and probe (the S-matrix estimate):
    A  uniform slices,   flat potential          (baseline)
    B  uniform slices,   S-informed potential
    C  S-informed slices, flat potential
    D  S-informed slices, S-informed potential
"S-informed" = quantem_export.slice_grid / initial_potential from the exported candidates.  Without candidates
(detection failed) B = A and D = C, which is the honest outcome.

Scores per arm: final loss; depth of the reconstructed potential along the true Au columns, as the centroid of the
half-maximum region around the main peak inside each feature's window (windows split at the midpoints between the true depths), error vs truth;
runtime.  Every arm is run with --repeats seeds (torch / numpy seeded per run, so identical setups give identical
results); the report gives mean +- std over seeds and the B - A, C - A, D - A differences paired by seed.

    .venv/bin/python m6_quantem.py                       # all results_m6/*.npz, device auto
    .venv/bin/python m6_quantem.py --iters 200 --repeats 3 --only vac_z30
    -> results_m6/M6_RESULTS.md, M6_report.json

QuantEM conventions (M5, commit 0f60f90): probe defocus = -C10; object pixel = S-grid pixel - K/2 + padding;
constraints are a dict; positivity on, so potentials start at a small positive baseline.
"""
import argparse, glob, json, os, time, warnings
import numpy as np
import torch

import quantem_export as qe

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_m6")
PAD = 8
BASE = 1e-3


def load(f):
    z = np.load(f, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["candidates"] = [qe.DepthCandidate(y=c["y"], x=c["x"], modes=[tuple(m) for m in c["modes"]], phase=c["phase"])
                       for c in json.loads(str(d["candidates"]))]
    return d


def dataset(d):
    from quantem.core.datastructures import Dataset4dstem
    from quantem.diffractive_imaging import PtychographyDatasetRaster
    s, dk = float(d["scan_step"]), float(d["dk_mrad"])
    dset = Dataset4dstem.from_array(d["data"], name="m6", sampling=(s, s, dk, dk), units=("A", "A", "mrad", "mrad"))
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", probe_energy=float(d["energy"]), force_com_rotation=0,
                     plot_rotation=False, plot_com=False)
    return pdset


def ptycho(d, pdset, obj, device, seed=0):
    from quantem.diffractive_imaging import ProbePixelated, DetectorPixelated, Ptychography
    probe = ProbePixelated.from_params(num_probes=1, probe_params={
        "energy": float(d["energy"]), "defocus": float(d["defocus_quantem"]), "semiangle_cutoff": float(d["alpha_mrad"])})
    pt = Ptychography.from_models(dset=pdset, obj_model=obj, probe_model=probe, detector_model=DetectorPixelated(),
                                  device=device, rng=seed)              # rng drives QuantEM's batch shuffling
    pt.preprocess(obj_padding_px=(PAD, PAD), batch_size=64)
    return pt


def grids(d):
    """S-informed slice grid from the candidates and a uniform grid with the same number of planes."""
    T = float(d["thickness"])
    cands = d["candidates"]
    g_s = qe.slice_grid(cands, T, dz_max=8.0, dz_min=2.5, extra_slices=2.0) if cands else None
    n = g_s.num_slices if g_s is not None else int(np.ceil(T / 8.0))
    g_u = qe.SliceGrid(boundaries=np.linspace(0.0, T, n + 1), thickness=T)
    return g_u, (g_s if g_s is not None else g_u)


def run_arm(d, pdset, grid, informed, shape, a, seed):
    from quantem.core.ml import OptimizerParams
    torch.manual_seed(seed)
    np.random.seed(seed)
    K, dr = int(d["K"]), float(d["dr"])
    origin = (K // 2 - PAD) * dr
    phi = np.full((grid.num_slices,) + tuple(shape), BASE)
    if informed and d["candidates"]:
        phi += qe.initial_potential(d["candidates"], grid, shape, dr, origin=(origin, origin))
    pt = ptycho(d, pdset, qe.to_quantem(phi, grid), a.device, seed)
    t0 = time.time()
    pt.reconstruct(num_iters=a.iters, reset=True, batch_size=64, device=a.device,
                   optimizer_params={"object": OptimizerParams.Adam(lr=a.lr)}, constraints={"object": {"positivity": True}})
    obj = pt.obj_model.obj.detach().cpu().numpy()
    return obj, float(np.asarray(pt.iter_losses)[-1]), time.time() - t0


def score(d, grid, obj):
    """Depth of the potential along the true Au columns, per true feature: windows split at the midpoints between the
    true depths; in each, the centroid of the contiguous half-maximum region around the main peak (tails and spikes
    elsewhere do not pull it), and that region's width (FWHM)."""
    K, dr = int(d["K"]), float(d["dr"])
    px = np.round(np.asarray(d["true_xy"]) / dr).astype(int) - (K // 2 - PAD)
    prof = np.clip(np.mean([obj[:, y, x] for y, x in px], axis=0) / grid.slab_thicknesses, 0, None)
    zc = grid.planes + grid.slab_thicknesses / 2
    truth = np.sort(np.asarray(d["true_z"], float))
    edges = np.concatenate([[0.0], (truth[1:] + truth[:-1]) / 2, [grid.thickness]])
    zf = np.linspace(0.0, grid.thickness, 1001)
    pf = np.interp(zf, zc, prof)
    found, fwhm = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.flatnonzero((zf >= lo) & (zf <= hi))
        w = pf[m]
        if w.max() <= 0:
            found.append(float("nan")); fwhm.append(float("nan")); continue
        i = int(np.argmax(w)); half = 0.5 * w[i]
        j0, j1 = i, i                                    # contiguous half-maximum region around the main peak
        while j0 > 0 and w[j0 - 1] >= half: j0 -= 1
        while j1 < len(w) - 1 and w[j1 + 1] >= half: j1 += 1
        zz, ww = zf[m][j0:j1 + 1], w[j0:j1 + 1]
        found.append(float((zz * ww).sum() / ww.sum()))
        fwhm.append(float(zz[-1] - zz[0]))
    found = np.array(found)
    return dict(found=found.tolist(), errors=(found - truth).tolist(), fwhm=fwhm,
                profile=(prof / max(prof.max(), 1e-12)).tolist(), plane_centres=zc.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="gpu" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--repeats", type=int, default=3, help="seeds per arm (mean +- std, paired differences)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(OUT, "*.npz")))
    if a.only:
        files = [f for f in files if os.path.basename(f)[:-4] in a.only.split(",")]
    report = {}
    for f in files:
        name = os.path.basename(f)[:-4]
        d = load(f)
        pdset = dataset(d)
        g_u, g_s = grids(d)
        from quantem.diffractive_imaging import ObjectPixelated
        probe_pt = ptycho(d, pdset, ObjectPixelated.from_uniform(num_slices=g_u.num_slices,
                                                                  slice_thicknesses=float(g_u.slice_thicknesses[0]),
                                                                  obj_type="potential"), a.device)
        shape = tuple(int(v) for v in probe_pt.obj_shape_full[1:])
        rows = {}
        for arm, grid, informed in (("A uniform/flat", g_u, False), ("B uniform/S-pot", g_u, True),
                                    ("C S-slices/flat", g_s, False), ("D S-slices/S-pot", g_s, True)):
            runs = []
            for seed in range(a.repeats):
                obj, loss, t = run_arm(d, pdset, grid, informed, shape, a, seed)
                runs.append(dict(seed=seed, loss=loss, time_s=t, **score(d, grid, obj)))
            err = np.array([r["errors"] for r in runs]); loss = np.array([r["loss"] for r in runs])
            rows[arm] = dict(planes=grid.num_slices, runs=runs, err_mean=err.mean(0).tolist(), err_std=err.std(0).tolist(),
                             loss_mean=float(loss.mean()), loss_std=float(loss.std()))
            print(f"{name:10s} {arm:18s} planes {grid.num_slices:3d} loss {loss.mean():.4f}+-{loss.std():.4f} "
                  f"err {np.round(err.mean(0), 1).tolist()}+-{np.round(err.std(0), 1).tolist()} "
                  f"(true {np.asarray(d['true_z']).tolist()})", flush=True)
        base = rows["A uniform/flat"]
        for arm in ("B uniform/S-pot", "C S-slices/flat", "D S-slices/S-pot"):
            dl = [r["loss"] - b["loss"] for r, b in zip(rows[arm]["runs"], base["runs"])]
            de = [np.abs(r["errors"]).mean() - np.abs(b["errors"]).mean() for r, b in zip(rows[arm]["runs"], base["runs"])]
            rows[arm]["vs_A"] = dict(dloss_mean=float(np.mean(dl)), dloss_std=float(np.std(dl)),
                                     d_abs_err_mean=float(np.mean(de)), d_abs_err_std=float(np.std(de)))
        report[name] = dict(true_z=np.asarray(d["true_z"]).tolist(), rho=float(d["rho"]),
                            n_candidates=len(d["candidates"]), reliable=json.loads(str(d["residual_ok"])),
                            candidate_depths=sorted({m[0] for c in d["candidates"] for m in c.modes}), arms=rows)
    json.dump(report, open(os.path.join(OUT, "M6_report.json"), "w"), indent=1, default=float)
    L = ["# M6 results (generated by m6_quantem.py)", "",
         f"{a.iters} iterations, Adam lr {a.lr}, positivity on, probe fixed at the S-matrix estimate, {a.repeats} seeds "
         "per arm. Depth = centroid of the reconstructed potential along the true Au columns, per feature window.", "",
         "## Depth error per feature (A, mean +- std over seeds) and final loss (mean)", "",
         "| specimen | carbon | candidates (depths, reliable) | A uniform/flat | B uniform/S-pot | C S-slices/flat | "
         "D S-slices/S-pot |", "|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        cell = lambda k: (f"{np.round(r['arms'][k]['err_mean'], 1).tolist()} +- {np.round(r['arms'][k]['err_std'], 1).tolist()}"
                          f" / {r['arms'][k]['loss_mean']:.4f}")
        L.append(f"| {name} | {r['rho']:g} | {np.round(r['candidate_depths'], 1).tolist()}, {r['reliable']} | "
                 + " | ".join(cell(k) for k in r["arms"]) + " |")
    L += ["", "## Paired differences vs arm A (same seed): mean |depth error| change (A) and loss change, mean +- std", "",
          "| specimen | B - A | C - A | D - A |", "|---|---|---|---|"]
    for name, r in report.items():
        cell = lambda k: (f"{r['arms'][k]['vs_A']['d_abs_err_mean']:+.1f} +- {r['arms'][k]['vs_A']['d_abs_err_std']:.1f} A, "
                          f"{r['arms'][k]['vs_A']['dloss_mean']:+.4f} +- {r['arms'][k]['vs_A']['dloss_std']:.4f}")
        L.append(f"| {name} | " + " | ".join(cell(k) for k in ("B uniform/S-pot", "C S-slices/flat", "D S-slices/S-pot")) + " |")
    L += ["", "Negative = better than A. Same planes, iterations, optimiser and probe in all arms; with no candidates "
          "B = A and D = C by construction, so their spread there is pure run-to-run noise."]
    open(os.path.join(OUT, "M6_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
