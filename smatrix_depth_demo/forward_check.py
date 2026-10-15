"""Forward-model check: do an object + probe explain the measured diffraction?  (No optimisation.)

    python forward_check.py --data data/realization_0/4dstem_quantem_noisefree --truth data/realization_0 \
        --runs results_real0_2slice:baseline --out results_real0_2slice/ground_truth_comparison

For each configuration the patterns are predicted with QuantEM's own forward path, exactly as in its reconstruct loop
(dset.forward -> probe_model.forward -> obj_model.forward -> forward_operator -> detector_model.forward), at every scan
position, and compared with the measured patterns QuantEM fits (dset.targets, amplitudes):

    L_data = sum_{R,q} (sqrt(I_pred) - sqrt(I_meas))^2 / sum_{R,q} I_meas

also split into bright field (|q| <= alpha) and dark field, and QuantEM's own loss (error_estimate, l2_amplitude) for
reference to the training loss.  Configurations:
    recon object + recon probe (each run), recon object + truth probe,
    truth object (8 x 0.75 A slabs = the simulation's slicing) + truth probe   <- model floor
    truth object summed into the run's slabs (e.g. 2 x 3 A) + truth probe        <- cost of the coarse slicing
    truth object projected into 1 slab + truth probe,  truth object + nominal probe,  vacuum + truth probe (reference)
Truth object: sigma * V_k from potential.zarr (V*A per slice, abTEM transmission exp(i sigma V)), Gaussian-smoothed to the
QuantEM pixel and sampled at the QuantEM pixel centres (cell coordinate = scan origin + (j - p0) * dr).
"""
import argparse, json, os
from types import SimpleNamespace
import numpy as np
import torch

import run_realization as rr
import quantem_export as qe

PAD = rr.PAD


def setup_dataset(d0, lam):
    from quantem.core.datastructures import Dataset4dstem
    from quantem.diffractive_imaging import PtychographyDatasetRaster
    x = rr.centre_patterns(d0["data"], d0["zero"], 1)
    x, _ = rr.bin_patterns(x, 1)
    s, dk = float(d0["step"][0]), float(d0["dk"][0])
    dset = Dataset4dstem.from_array(x, name="fc", sampling=(s, s, 1e3 * lam * dk, 1e3 * lam * dk),
                                    units=("A", "A", "mrad", "mrad"))
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", probe_energy=d0["energy"], force_com_rotation=0,
                     force_com_transpose=False, plot_rotation=False, plot_com=False)
    return pdset


def make_pt(pdset, d0, obj_model, probe_array, device):
    from quantem.diffractive_imaging import ProbePixelated, DetectorPixelated, Ptychography
    probe = ProbePixelated.from_array(np.asarray(probe_array, np.complex64)[None], device=device,
                                      probe_params={"energy": d0["energy"], "semiangle_cutoff": d0["alpha_mrad"]})
    pt = Ptychography.from_models(dset=pdset, obj_model=obj_model, probe_model=probe, detector_model=DetectorPixelated(),
                                  device=device, rng=0)
    pt.preprocess(obj_padding_px=(PAD, PAD), batch_size=64)
    return pt


@torch.no_grad()
def predict(pt, bf_mask, batch=64):
    n = pt.dset.num_gpts
    num = den = num_bf = den_bf = 0.0
    q_loss = []
    for i0 in range(0, n, batch):
        idx = np.arange(i0, min(n, i0 + batch))
        patch_indices, _, frac, descan = pt.dset.forward(idx, pt.obj_padding_px)
        probes = pt.probe_model.forward(frac)
        patches = pt.obj_model.forward(patch_indices)
        _, overlap = pt.forward_operator(patches, probes, descan)
        I = pt.detector_model.forward(overlap)
        loss, targets = pt.error_estimate(I, idx, loss_type="l2_amplitude")
        q_loss.append(float(loss) * len(idx))
        m = pt.dset.detector_mask
        a_pred = torch.sqrt(I + 1e-12) * m
        a_meas = targets * m
        r = (a_pred - a_meas) ** 2
        num += float(r.sum()); den += float((a_meas ** 2).sum())
        bf = torch.as_tensor(bf_mask, device=r.device)
        num_bf += float((r * bf).sum()); den_bf += float(((a_meas ** 2) * bf).sum())
    return dict(L_data=num / den, L_data_bright_field=num_bf / den_bf,
                L_data_dark_field=(num - num_bf) / max(den - den_bf, 1e-30),
                dark_field_fraction_of_measured_intensity=(den - den_bf) / den,
                quantem_loss=sum(q_loss) / n)


def truth_probe_on_grid(tp, shape, dr_q, ref_probe):
    """Truth probe (corner-centred, pixel s_t) -> QuantEM probe grid (corner-centred, pixel dr_q), same total intensity
    as ref_probe.  Exact decimation when dr_q is a multiple of s_t (the probe is band-limited far below Nyquist)."""
    arr, s_t = np.asarray(tp[0]), float(tp[1])
    f = dr_q / s_t
    if abs(f - round(f)) > 1e-6:
        raise ValueError(f"QuantEM pixel {dr_q} is not a multiple of the truth probe pixel {s_t}")
    f = int(round(f))
    iy = (np.round(np.fft.fftfreq(shape[0]) * shape[0]).astype(int) * f) % arr.shape[0]
    ix = (np.round(np.fft.fftfreq(shape[1]) * shape[1]).astype(int) * f) % arr.shape[1]
    p = arr[np.ix_(iy, ix)].astype(np.complex64)
    return p * np.sqrt((np.abs(ref_probe) ** 2).sum() / (np.abs(p) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run_dir:arm (first one sets the geometry)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    args = SimpleNamespace(energy=None, alpha=None, defocus=None, scan_roi=None)
    d0 = rr.load_dataset(a.data, args, 0)
    T = float(d0["cell"][-1])
    sm = rr.load_smatrix(a.runs[0].split(":")[0])
    lam = sm["setup"].lam
    tr = rr.load_truth([a.truth], lambda *_: None)
    sig = rr.sigma_interaction(d0["energy"])
    pdset = setup_dataset(d0, lam)
    pos0 = d0["positions_A"][0, 0]

    runs = []
    for spec in a.runs:
        rd, arm = spec.split(":")
        o = os.path.join(rd, f"quantem_{arm}")
        if not os.path.exists(os.path.join(o, "object_potential.npy")):
            print(f"missing: {o}")
            continue
        runs.append(dict(name=f"{os.path.basename(os.path.normpath(rd))}/{arm}", obj=np.load(os.path.join(o, "object_potential.npy")),
                         bnd=np.load(os.path.join(o, "slice_boundaries.npy")), probe=np.load(os.path.join(o, "probe_real_space.npy")),
                         p_init=np.load(os.path.join(o, "initial_probe_real_space.npy")),
                         loss=np.load(os.path.join(o, "loss.npy")), info=json.load(open(os.path.join(o, "arm.json")))))
    shape = runs[0]["obj"].shape[1:]
    ref_probe = runs[0]["probe"]

    # geometry (pixel size, first scan pixel) from a throw-away model
    pt0 = make_pt(pdset, d0, qe.to_quantem(np.full((2,) + shape, 1e-3, np.float32),
                                           qe.SliceGrid(boundaries=np.array([0, T / 2, T]), thickness=T)), ref_probe, a.device)
    dr_q = float(np.asarray(pt0.sampling)[0])
    p0 = pt0.dset.scan_positions_px.detach().cpu().numpy().min(axis=0)
    Q = pt0.dset.targets.shape[-1]
    dk = float(d0["dk"][0])
    qy, qx = np.meshgrid((np.arange(Q) - Q // 2) * dk, (np.arange(Q) - Q // 2) * dk, indexing="ij")
    bf = (np.hypot(qy, qx) <= 1e-3 * d0["alpha_mrad"] / lam).astype(np.float32)
    rows = pos0[0] + (np.arange(shape[0]) - p0[0]) * dr_q
    cols = pos0[1] + (np.arange(shape[1]) - p0[1]) * dr_q
    pot, dr_t, st = tr["potential"]
    edges = np.r_[0, np.cumsum(st)]

    def truth_obj(bnd):
        return np.stack([rr.sample_truth(pot, dr_t, rows, cols, bnd[k] - 1e-6, bnd[k + 1] - 1e-6, st, dr_q) * sig
                         for k in range(len(bnd) - 1)]).astype(np.float32)

    tp = truth_probe_on_grid(tr["probes"][0], ref_probe.shape, dr_q, ref_probe)
    ov_tp = rr.probe_match(rr.probe_at_beams(tp, dr_q, sm["ky"], sm["kx"]),
                           rr.probe_at_beams(tr["probes"][0][0], tr["probes"][0][1], sm["ky"], sm["kx"]), sm["ky"], sm["kx"])
    configs = []
    for r in runs:
        g = qe.SliceGrid(boundaries=r["bnd"], thickness=T)
        configs.append((f"{r['name']}: recon object + recon probe", r["obj"], g, r["probe"]))
        configs.append((f"{r['name']}: recon object + truth probe", r["obj"], g, tp))
    configs.append(("truth object, 8 x 0.75 A slabs + truth probe (model floor)", truth_obj(edges),
                    qe.SliceGrid(boundaries=edges, thickness=T), tp))
    configs.append(("truth object, 2 x 3 A slabs + truth probe", truth_obj(np.array([0, T / 2, T])),
                    qe.SliceGrid(boundaries=np.array([0, T / 2, T]), thickness=T), tp))
    configs.append(("truth object, 1 slab (projected) + truth probe", truth_obj(np.array([0, T])),
                    qe.SliceGrid(boundaries=np.array([0.0, T]), thickness=T), tp))
    configs.append(("truth object, 8 slabs + nominal probe (QuantEM start)", truth_obj(edges),
                    qe.SliceGrid(boundaries=edges, thickness=T), runs[0]["p_init"]))
    configs.append(("vacuum (zero potential) + truth probe (reference)", np.zeros((2,) + shape, np.float32),
                    qe.SliceGrid(boundaries=np.array([0, T / 2, T]), thickness=T), tp))
    res = {}
    for name, phi, g, probe in configs:
        try:
            pt = make_pt(pdset, d0, qe.to_quantem(phi, g), probe, a.device)
            res[name] = predict(pt, bf)
            res[name]["slabs"] = int(len(g.boundaries) - 1)
            print(f"{name}: L_data {res[name]['L_data']:.4f}, QuantEM loss {res[name]['quantem_loss']:.3f}", flush=True)
        except Exception as e:                                   # report, do not hide
            res[name] = dict(error=f"{type(e).__name__}: {e}")
            print(f"{name}: FAILED {e}", flush=True)
    out = dict(geometry=dict(dr_quantem=dr_q, p0=p0.tolist(), scan_origin_A=list(map(float, pos0)), object_shape=list(shape),
                             bright_field_radius_inv_A=float(1e-3 * d0["alpha_mrad"] / lam),
                             truth_probe_resampled_vs_truth=dict(overlap=ov_tp[0], rms_phase=ov_tp[1])),
               configs=res, final_training_loss={r["name"]: float(r["loss"][-1]) for r in runs})
    json.dump(out, open(os.path.join(a.out, "forward_check.json"), "w"), indent=1, default=float)
    L = ["# Forward-model check (QuantEM forward path, no optimisation)", "",
         "L_data = Σ(√I_pred − √I_meas)² / Σ I_meas over all scan positions and detector pixels (measured = the noise-free "
         "patterns as QuantEM sees them).  Bright field = |q| ≤ α/λ.  QuantEM loss = its own `error_estimate` "
         "(l2_amplitude), comparable with the training loss.", "",
         f"Truth probe resampled to the QuantEM grid: overlap {ov_tp[0]:.4f}, RMS phase {ov_tp[1]:.3f} rad vs the original.",
         f"Measured dark-field share of the intensity: "
         f"{next(v for v in res.values() if 'dark_field_fraction_of_measured_intensity' in v)['dark_field_fraction_of_measured_intensity']:.3f}.", "",
         "| configuration | slabs | L_data | bright field | dark field | QuantEM loss |", "|---|---|---|---|---|---|"]
    for n, v in res.items():
        if "error" in v:
            L.append(f"| {n} | – | failed: {v['error']} | | | |")
        else:
            L.append(f"| {n} | {v['slabs']} | {v['L_data']:.4f} | {v['L_data_bright_field']:.4f} | {v['L_data_dark_field']:.4f} | "
                     f"{v['quantem_loss']:.3f} |")
    L += ["", "Final training loss (last iteration, includes batch averaging): " +
          ", ".join(f"{k} {v:.3f}" for k, v in out["final_training_loss"].items())]
    open(os.path.join(a.out, "FORWARD_CHECK.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
