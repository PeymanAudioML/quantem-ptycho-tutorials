"""M6 step 2 (QuantEM's environment, e.g. .venv/bin/python): four-arm QuantEM comparison on the specimens exported by
m6_prepare.py.  Needs only numpy, torch, quantem and quantem_export.py (pure numpy).

Arms, all with the same number of planes N, iterations, optimiser, constraints and probe (the S-matrix estimate):
    A  uniform slices,   flat potential          (baseline)
    B  uniform slices,   S-informed potential
    C  S-informed slices, flat potential
    D  S-informed slices, S-informed potential
"S-informed" = quantem_export.slice_grid / initial_potential from the exported candidates.  Without candidates
(detection failed) B = A and D = C, which is the honest outcome.

Scores per arm: final loss; depth of the reconstructed potential along the true Au columns (peak and centroid of each
feature, matched to the truth in depth order), error vs truth; runtime.

    .venv/bin/python m6_quantem.py                       # all results_m6/*.npz, device auto
    .venv/bin/python m6_quantem.py --iters 100 --only vac_z30
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


def ptycho(d, pdset, obj, device):
    from quantem.diffractive_imaging import ProbePixelated, DetectorPixelated, Ptychography
    probe = ProbePixelated.from_params(num_probes=1, probe_params={
        "energy": float(d["energy"]), "defocus": float(d["defocus_quantem"]), "semiangle_cutoff": float(d["alpha_mrad"])})
    pt = Ptychography.from_models(dset=pdset, obj_model=obj, probe_model=probe, detector_model=DetectorPixelated(),
                                  device=device)
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


def run_arm(d, pdset, grid, informed, shape, a):
    from quantem.core.ml import OptimizerParams
    K, dr = int(d["K"]), float(d["dr"])
    origin = (K // 2 - PAD) * dr
    phi = np.full((grid.num_slices,) + tuple(shape), BASE)
    if informed and d["candidates"]:
        phi += qe.initial_potential(d["candidates"], grid, shape, dr, origin=(origin, origin))
    pt = ptycho(d, pdset, qe.to_quantem(phi, grid), a.device)
    t0 = time.time()
    pt.reconstruct(num_iters=a.iters, reset=True, batch_size=64, device=a.device,
                   optimizer_params={"object": OptimizerParams.Adam(lr=a.lr)}, constraints={"object": {"positivity": True}})
    obj = pt.obj_model.obj.detach().cpu().numpy()
    return obj, float(np.asarray(pt.iter_losses)[-1]), time.time() - t0


def score(d, grid, obj):
    """Depth of the potential along the true Au columns: one peak per true feature, matched in depth order."""
    K, dr = int(d["K"]), float(d["dr"])
    px = np.round(np.asarray(d["true_xy"]) / dr).astype(int) - (K // 2 - PAD)
    prof = np.mean([obj[:, y, x] for y, x in px], axis=0) / grid.slab_thicknesses
    zc = grid.planes + grid.slab_thicknesses / 2
    truth = np.sort(np.asarray(d["true_z"], float))
    zf = np.linspace(0.0, grid.thickness, 601)
    pf = np.interp(zf, zc, prof)
    peaks = [i for i in range(1, len(pf) - 1) if pf[i] >= pf[i - 1] and pf[i] >= pf[i + 1]] + [0, len(pf) - 1]
    peaks = sorted(set(peaks), key=lambda i: -pf[i])[:len(truth)]
    found = np.sort(zf[peaks]) if peaks else np.full(len(truth), np.nan)
    cen = float((zc * np.clip(prof, 0, None)).sum() / max(np.clip(prof, 0, None).sum(), 1e-12))
    return dict(found=found.tolist(), errors=(found - truth).tolist(), centroid=cen,
                profile=(prof / max(prof.max(), 1e-12)).tolist(), plane_centres=zc.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="gpu" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
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
            obj, loss, t = run_arm(d, pdset, grid, informed, shape, a)
            sc = score(d, grid, obj)
            rows[arm] = dict(loss=loss, time_s=t, planes=grid.num_slices, **sc)
            print(f"{name:10s} {arm:18s} planes {grid.num_slices:3d} loss {loss:.4f} depth {np.round(sc['found'], 1).tolist()} "
                  f"(true {np.asarray(d['true_z']).tolist()}) err {np.round(sc['errors'], 1).tolist()}  {t:.0f} s", flush=True)
        report[name] = dict(true_z=np.asarray(d["true_z"]).tolist(), rho=float(d["rho"]),
                            n_candidates=len(d["candidates"]), reliable=json.loads(str(d["residual_ok"])),
                            candidate_depths=sorted({m[0] for c in d["candidates"] for m in c.modes}), arms=rows)
    json.dump(report, open(os.path.join(OUT, "M6_report.json"), "w"), indent=1, default=float)
    L = ["# M6 results (generated by m6_quantem.py)", "",
         f"{a.iters} iterations, Adam lr {a.lr}, positivity on, probe fixed at the S-matrix estimate. Depth errors in A "
         "(reconstructed potential along the true Au columns vs truth).", "",
         "| specimen | carbon | candidates (depths, reliable) | A uniform/flat | B uniform/S-pot | C S-slices/flat | "
         "D S-slices/S-pot |", "|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        cell = lambda k: (f"{np.round(r['arms'][k]['errors'], 1).tolist()} / {r['arms'][k]['loss']:.4f}")
        L.append(f"| {name} | {r['rho']:g} | {np.round(r['candidate_depths'], 1).tolist()}, {r['reliable']} | "
                 + " | ".join(cell(k) for k in r["arms"]) + " |")
    L += ["", "Cells: depth errors per true feature / final loss. Same planes, iterations and probe in all four arms."]
    open(os.path.join(OUT, "M6_RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
