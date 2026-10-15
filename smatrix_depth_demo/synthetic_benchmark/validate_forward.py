"""Forward-model consistency of a generated specimen, before any reconstruction (no optimisation, ground truth only).

    python validate_forward.py --specimen A_z100 [--device cpu]   -> results/forward_validation/<specimen>/

Measured = the generated noiseless patterns (window and pixelated variants).  Predictions from the TRUE potential and
TRUE probes through three independent forward models:
  F1  plain multislice in torch from the saved potential (transmission exp(i sigma V), Fresnel propagator), with and
      without abTEM's 2/3-Nyquist anti-aliasing aperture: checks units, orientation, slicing and probe placement.
  F2  S-matrix model (simple_joint.JointModel with the true aberrations) using a numerically computed reference
      S-matrix: every beam's plane wave through the same multislice (exit-referenced, so the window crop is exact),
      and entrance-referenced (the gauge refocus() assumes).
  F3  QuantEM's own forward path (forward_check.predict) with the true potential sampled on its object grid and the
      true probe, for every defocus dataset; vacuum and nominal-probe references.
Metric: L_data = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas over all patterns and pixels (F2 after one fitted
global scale, reported); bright-field / dark-field split.  Nothing is changed to make the numbers smaller.
"""
import argparse, json, math, os, sys, time
from types import SimpleNamespace

import numpy as np
import torch

import bench

sj = bench.sj


def load(spec_dir, variant, noise="noiseless"):
    import zarr
    out = []
    for d in range(3):
        p = os.path.join(spec_dir, f"{variant}_{noise}", f"def{d}", "4dstem_quantem", "_array")
        out.append(zarr.open_array(p, mode="r")[...])
    return np.array(out)


def ldata(Ip, Im, bf=None):
    r = (np.sqrt(np.clip(Ip, 0, None)) - np.sqrt(np.clip(Im, 0, None))) ** 2
    out = dict(L_data=float(r.sum() / Im.sum()))
    if bf is not None:
        out.update(L_bright_field=float((r * bf).sum() / (Im * bf).sum()),
                   L_dark_field=float((r * (1 - bf)).sum() / max((Im * (1 - bf)).sum(), 1e-30)),
                   dark_field_fraction=float((Im * (1 - bf)).sum() / Im.sum()))
    return out


def multislice_torch(setup, phase, waves, dz, antialias, device):
    """waves (n, Y, Y) complex at the entrance; phase (N, Y, Y) = sigma V; slice k: transmit, then propagate dz."""
    Y = setup.Y
    q = torch.as_tensor(np.fft.fftfreq(Y, setup.dr), device=device)
    q2 = q[:, None] ** 2 + q[None, :] ** 2
    P = torch.exp(-1j * math.pi * setup.lam * dz * q2)
    if antialias:
        P = P * (q2 <= (2.0 / 3.0 * 0.5 / setup.dr) ** 2)
    t = torch.exp(1j * torch.as_tensor(phase, device=device, dtype=torch.float64))
    w = torch.as_tensor(waves, device=device, dtype=torch.complex128)
    for k in range(len(phase)):
        w = torch.fft.ifft2(torch.fft.fft2(w * t[k]) * P)
    return w


def window_patterns(setup, exit_waves, pos):
    h = setup.K // 2
    out = []
    for w, (r0, r1) in zip(exit_waves, pos):
        out.append(np.fft.fftshift(np.abs(np.fft.fft2(w[r0 - h:r0 + h, r1 - h:r1 + h], norm="ortho")) ** 2))
    return np.array(out)


def run(cfg, name, device="cpu", log=print, stride=3, batch=4):
    import zarr
    setup = bench.make_setup(cfg)
    spec = cfg["specimens"][name]
    sd = bench.specimen_dir(cfg, name)
    gt = os.path.join(sd, "ground_truth")
    truth = json.load(open(os.path.join(gt, "truth.json")))
    V = zarr.open_array(os.path.join(gt, "potential.zarr", "array0"), mode="r")[...]
    sigma = truth["potential"]["sigma_rad_per_V_A"]
    phase = sigma * V.astype(np.float64)
    dz = float(np.diff(truth["potential"]["slice_boundaries_A"])[0])
    true = np.array(truth["aberrations"]["true"])
    nominal = np.array(truth["aberrations"]["nominal"])
    pos = bench.scan_positions(cfg)
    flat = pos.reshape(-1, 2)
    meas = dict(window=load(sd, "window"), pixelated=load(sd, "pixelated"))
    K = setup.K
    kk = (np.arange(K) - K // 2) / (K * setup.dr)
    bf = (np.hypot(kk[:, None], kk[None, :]) <= setup.alpha / setup.lam).astype(float)
    res = dict(specimen=name, config_sha256=cfg["_sha256"], versions=bench.versions())

    # ---------------------------------------------------------------------------------------- F1 torch multislice
    t0 = time.time()
    f1 = {}
    pred1 = {}
    sub = np.zeros(pos.shape[:2], bool)
    sub[::stride, ::stride] = True
    subflat = flat[sub.ravel()]
    msub = meas["window"][:, sub]
    for aa in (True, False, "vacuum"):
        P = []
        for d in range(len(true)):
            p0 = bench.ad.entrance_probe(setup, true[d]).astype(np.complex128)
            Pd = []
            for b0 in range(0, len(subflat), 32):
                ch = subflat[b0:b0 + 32]
                w0 = np.stack([np.roll(p0, (int(r0), int(r1)), axis=(0, 1)) for r0, r1 in ch])
                ex = multislice_torch(setup, phase * (aa != "vacuum"), w0, dz, aa is not False, device).cpu().numpy()
                Pd.append(window_patterns(setup, ex, ch))
            P.append(np.concatenate(Pd))
        P = np.array(P)
        key = {True: "antialias_2/3", False: "no_antialias", "vacuum": "vacuum (Au signal reference)"}[aa]
        pred1[key] = P
        f1[key] = dict(all=ldata(P, msub, bf), per_dataset=[ldata(P[d], msub[d], bf) for d in range(len(P))],
                       max_abs_rel=float(np.abs(P - msub).max() / msub.max()))
    f1["positions_used"] = int(len(subflat))
    res["F1_torch_multislice_vs_abtem_window"] = f1
    log(f"F1 torch multislice vs abTEM (window, {len(subflat)} positions): " + ", ".join(
        f"{k} L_data {v['all']['L_data']:.2e}" for k, v in f1.items() if isinstance(v, dict)) + f" ({time.time() - t0:.0f} s)")
    best = min(("antialias_2/3", "no_antialias"), key=lambda k: f1[k]["all"]["L_data"])
    res["au_signal_L"] = f1["vacuum (Au signal reference)"]["all"]["L_data"]

    # ---------------------------------------------------------------------------------------- F2 S-matrix model
    t0 = time.time()
    ky, kx = setup.beams()
    r = np.arange(setup.Y) * setup.dr
    f2 = {}
    for ref_plane in ("exit", "entrance"):
        S0 = (np.exp(2j * np.pi * ky[:, None, None] * r[None, :, None]) * np.exp(2j * np.pi * kx[:, None, None] * r[None, None, :]))
        S = []
        for b0 in range(0, len(S0), 32):
            S.append(multislice_torch(setup, phase, S0[b0:b0 + 32], dz, best == "antialias_2/3", device).cpu().numpy()
                     .astype(np.complex64))
        S = np.concatenate(S)
        del S0
        if ref_plane == "entrance":
            q = np.fft.fftfreq(setup.Y, setup.dr)
            S = np.fft.ifft2(np.fft.fft2(S) * np.exp(1j * np.pi * setup.lam * truth["thickness_A"] * (q[:, None] ** 2 + q[None, :] ** 2)))
        m = sj.JointModel(setup, flat, np.zeros(len(flat), int), true, trainable=(), device=device)
        with torch.no_grad():
            m.S.copy_(torch.as_tensor(S.astype(np.complex64)))
            pred = []
            for d in range(len(true)):
                m.didx[:] = d
                Z = torch.cat([m(idx) for idx in torch.arange(len(flat)).split(batch)])
                pred.append(np.fft.fftshift((Z.abs() ** 2).cpu().numpy(), axes=(-2, -1)).reshape(*pos.shape[:2], K, K))
        pred = np.array(pred)
        out = {}
        for variant in ("window", "pixelated"):
            Im = meas[variant]
            s = float((np.sqrt(pred) * np.sqrt(Im)).sum() / (pred.sum()))      # least-squares amplitude scale
            out[variant] = dict(amplitude_scale=s, **ldata(pred * s ** 2, Im, bf),
                                per_dataset=[ldata(pred[d] * s ** 2, Im[d], bf) for d in range(len(pred))])
        f2[ref_plane] = out
        if ref_plane == "exit":
            pred2 = pred
    res["F2_smatrix_model"] = dict(reference_S=f"multislice of {len(ky)} plane waves ({best})", **f2)
    log(f"F2 S-matrix model: exit-ref vs window L {f2['exit']['window']['L_data']:.2e}, vs pixelated "
        f"{f2['exit']['pixelated']['L_data']:.2e}; entrance-ref vs window {f2['entrance']['window']['L_data']:.2e} "
        f"({time.time() - t0:.0f} s)")

    # ---------------------------------------------------------------------------------------- F3 QuantEM forward
    t0 = time.time()
    f3 = {}
    try:
        import run_realization as rr
        import forward_check as fc
        import quantem_export as qe
        for variant in ("window", "pixelated"):
            f3[variant] = {}
            for d in range(len(true)):
                ddir = os.path.join(sd, f"{variant}_noiseless", f"def{d}")
                d0 = rr.load_dataset(ddir, SimpleNamespace(energy=None, alpha=None, defocus=None), d)
                pdset = fc.setup_dataset(d0, setup.lam)
                T = truth["thickness_A"]
                g1 = qe.SliceGrid(boundaries=np.array([0.0, T / 2, T]), thickness=T)
                probe_nom = bench.ad.entrance_probe(setup, nominal[d])
                tmp = fc.make_pt(pdset, d0, qe.to_quantem(np.full((2, 8, 8), 1e-3, np.float32), g1),
                                 np.zeros((K, K), np.complex64) + 1e-3, device)
                pt0 = fc.make_pt(pdset, d0, qe.to_quantem(np.full((2,) + tuple(int(v) for v in tmp.obj_shape_full[1:]), 1e-3,
                                                                  np.float32), g1),
                                 fc.truth_probe_on_grid((bench.ad.entrance_probe(setup, true[d]), setup.dr), (K, K), setup.dr,
                                                        np.ones((K, K))), device)
                shape = tuple(int(v) for v in pt0.obj_shape_full[1:])
                dr_q = float(np.asarray(pt0.sampling)[0])
                p0 = pt0.dset.scan_positions_px.detach().cpu().numpy().min(axis=0)
                pos0 = d0["positions_A"][0, 0]
                rows = pos0[0] + (np.arange(shape[0]) - p0[0]) * dr_q
                cols = pos0[1] + (np.arange(shape[1]) - p0[1]) * dr_q
                tr = rr.load_truth([ddir], lambda *_: None)
                pot, dr_t, st = tr["potential"]
                edges = np.r_[0, np.cumsum(st)]

                def truth_obj(bnd):
                    return np.stack([rr.sample_truth(pot, dr_t, rows, cols, bnd[k] - 1e-6, bnd[k + 1] - 1e-6, st, dr_q) * sigma
                                     for k in range(len(bnd) - 1)]).astype(np.float32)

                ref = np.asarray(pt0.probe)[0]
                tp = fc.truth_probe_on_grid((bench.ad.entrance_probe(setup, true[d]), setup.dr), ref.shape, dr_q, ref)
                npb = fc.truth_probe_on_grid((probe_nom, setup.dr), ref.shape, dr_q, ref)
                Q = pt0.dset.targets.shape[-1]
                kq = (np.arange(Q) - Q // 2) * float(d0["dk"][0])
                bfq = (np.hypot(kq[:, None], kq[None, :]) <= setup.alpha / setup.lam).astype(np.float32)
                full = variant == "window" and d == 1
                cfgs = [("truth potential (100 x 2 A) + true probe", truth_obj(edges), qe.SliceGrid(boundaries=edges, thickness=T), tp),
                        ("truth potential (100 x 2 A) + nominal probe", truth_obj(edges), qe.SliceGrid(boundaries=edges, thickness=T), npb),
                        ("truth projected (1 slab) + true probe", truth_obj(np.array([0.0, T])), qe.SliceGrid(boundaries=np.array([0.0, T]), thickness=T), tp),
                        ("vacuum + true probe", np.zeros((2,) + shape, np.float32), g1, tp)]
                if not full:
                    cfgs = [cfgs[0], cfgs[3]]
                if variant == "pixelated" and d != 1:
                    continue
                f3[variant][f"def{d}"] = {}
                for nm, phi, g, prb in cfgs:
                    pt = fc.make_pt(pdset, d0, qe.to_quantem(phi, g), prb, device)
                    f3[variant][f"def{d}"][nm] = fc.predict(pt, bfq)
                log(f"F3 QuantEM {variant} def{d}: " + "; ".join(f"{nm.split(' +')[0][:22]}+{nm.split('+ ')[1][:7]} "
                                                                f"L {v['L_data']:.2e}" for nm, v in f3[variant][f'def{d}'].items()))
        res["F3_quantem_forward"] = dict(geometry=dict(dr_quantem=dr_q, first_scan_pixel=p0.tolist(), object_shape=list(shape)), **f3)
    except Exception as e:
        import traceback
        res["F3_quantem_forward"] = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        log(f"F3 QuantEM forward FAILED: {e}")
    log(f"F3 ({time.time() - t0:.0f} s)")
    return res, dict(window=meas["window"], pixelated=meas["pixelated"], F1=pred1[best], F2=pred2, stride=stride)


def report(res, arrays, out, setup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = [f"# Forward-model consistency: {res['specimen']}", "",
         "L_data = sum (sqrt(I_pred) - sqrt(I_meas))^2 / sum I_meas over all scan positions and pixels "
         "(0 = identical). Measured = generated noiseless patterns. Ground truth only; nothing fitted except F2's single "
         "global amplitude scale.", "",
         "## F1: independent torch multislice vs abTEM (window variant)", "",
         "| propagator | L_data | bright field | dark field | max abs diff / max |", "|---|---|---|---|---|"]
    L.insert(2, f"**Au signal** (vacuum vs measured): L = {res['au_signal_L']:.2e}. A usable forward model must be well below it.")
    for k, v in res["F1_torch_multislice_vs_abtem_window"].items():
        if not isinstance(v, dict):
            continue
        a = v["all"]
        L.append(f"| {k} | {a['L_data']:.2e} | {a['L_bright_field']:.2e} | {a['L_dark_field']:.2e} | {v['max_abs_rel']:.2e} |")
    f2 = res["F2_smatrix_model"]
    L += ["", f"## F2: S-matrix forward model with a numerically computed reference S ({f2['reference_S']})", "",
          "| S reference plane | measured variant | amplitude scale | L_data | bright field | dark field |", "|---|---|---|---|---|---|"]
    for rp in ("exit", "entrance"):
        for v in ("window", "pixelated"):
            x = f2[rp][v]
            L.append(f"| {rp} | {v} | {x['amplitude_scale']:.4f} | {x['L_data']:.2e} | {x['L_bright_field']:.2e} | {x['L_dark_field']:.2e} |")
    f3 = res["F3_quantem_forward"]
    L += ["", "## F3: QuantEM forward path (truth potential sampled on QuantEM's grid, true probe)", ""]
    if "error" in f3:
        L.append(f"FAILED: {f3['error']}")
    else:
        L += [f"QuantEM grid: {f3['geometry']['dr_quantem']:.3f} A pixels, object {f3['geometry']['object_shape']}, first scan "
              f"pixel {f3['geometry']['first_scan_pixel']}.", "",
              "| variant | dataset | configuration | L_data | bright field | dark field | QuantEM loss |", "|---|---|---|---|---|---|---|"]
        for v in ("window", "pixelated"):
            for d, cf in f3[v].items():
                for nm, x in cf.items():
                    L.append(f"| {v} | {d} | {nm} | {x['L_data']:.2e} | {x['L_data_bright_field']:.2e} | "
                             f"{x['L_data_dark_field']:.2e} | {x['quantem_loss']:.4f} |")
    open(os.path.join(out, "FORWARD_VALIDATION.md"), "w").write("\n".join(L) + "\n")
    # figure: measured vs predicted, residuals (dataset 1, centre position)
    c = tuple(s // 2 for s in arrays["window"].shape[1:3])
    stride = arrays["stride"]
    ci = (c[0] // stride) * ((arrays["window"].shape[2] - 1) // stride + 1) + c[1] // stride
    c = ((c[0] // stride) * stride, (c[1] // stride) * stride)
    fig, ax = plt.subplots(3, 4, figsize=(13, 9.5))
    for d in range(3):
        m = arrays["window"][d][c]
        f1 = arrays["F1"][d][ci]
        p = arrays["pixelated"][d][c]
        f2 = arrays["F2"][d][c] * res["F2_smatrix_model"]["exit"]["window"]["amplitude_scale"] ** 2
        for j, (img, t) in enumerate([(np.log10(m + 1e-9), f"measured (window) def{d}"),
                                      (np.sqrt(f2) - np.sqrt(m), "F2 S-matrix - measured (sqrt I)"),
                                      (np.sqrt(p) - np.sqrt(m), "pixelated - window (sqrt I)"),
                                      (np.sqrt(f1) - np.sqrt(m), "F1 torch - measured (sqrt I)")]):
            if j == 0:
                im = ax[d, j].imshow(img, cmap="magma", vmin=-6, vmax=np.log10(m.max()))
            else:
                v = max(np.abs(img).max(), 1e-12)
                im = ax[d, j].imshow(img, cmap="RdBu_r", vmin=-v, vmax=v)
            ax[d, j].set_title(t, fontsize=9)
            ax[d, j].set_xticks([]); ax[d, j].set_yticks([])
            fig.colorbar(im, ax=ax[d, j], fraction=0.046)
    fig.suptitle(f"{res['specimen']}: centre scan position, three defoci", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forward_residuals.png"), dpi=150, facecolor="white")
    plt.close(fig)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=bench.DEFAULT_CONFIG)
    ap.add_argument("--specimen", default="A_z100")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stride", type=int, default=3, help="F1 uses every stride-th scan position per axis")
    ap.add_argument("--batch", type=int, default=4, help="patterns per S-matrix forward batch (memory)")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    res, arrays = run(cfg, a.specimen, a.device, stride=a.stride, batch=a.batch)
    out = os.path.join(bench.HERE, "results", "forward_validation", cfg["name"], a.specimen)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "forward_validation.json"), "w"), indent=1, default=float)
    print(report(res, arrays, out, bench.make_setup(cfg)))


if __name__ == "__main__":
    main()
