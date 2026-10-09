"""Run the authors' UNMODIFIED GradDS code (AmpflowS.py, Optical_sectioning.py; Brown et al., arXiv:2011.07652)
on my simulated 3-layer data and compare with my NumPy re-implementation.

usage (needs numpy, torch, h5py, tqdm, matplotlib):
    python check_authors_code.py AUTHORS_CODE_DIR SHIM_DIR {refocus|equiv|full}

AUTHORS_CODE_DIR contains the GradDS/ package; SHIM_DIR contains a numpy-backed `cupy` stand-in.
Compatibility shims applied (authors' code itself untouched): cupy -> numpy, removed numpy aliases
(np.float/np.int/np.complex) restored, legacy torch.fft(x, ndim, normalized)/torch.ifft re-implemented.
  refocus : authors' depth_section_reconstruction vs mine, both applied to MY recovered S-matrix
  equiv   : authors' vs my amplitude-flow update after a few iterations with the equivalent step size
  full    : authors' recovery (their defaults) + their refocusing vs mine, scored against the true layers
"""
import os, sys, json, time
import numpy as np

authors_dir, shim_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, shim_dir)
import numpy.ma, matplotlib                                                    # import before patching numpy aliases
np.float, np.int, np.complex = float, int, complex                             # aliases removed in numpy >= 1.24 (np.bool exists again)
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
_fftmod = torch.fft


def _legacy(kind):
    def f(x, signal_ndim, normalized=False):
        c = torch.view_as_complex(x.contiguous())
        dims = tuple(range(-signal_ndim, 0))
        fn = _fftmod.fftn if kind == "fft" else _fftmod.ifftn
        return torch.view_as_real(fn(c, dim=dims, norm="ortho" if normalized else "backward"))
    return f


torch.fft, torch.ifft = _legacy("fft"), _legacy("ifft")                       # torch < 1.8 API used by the authors
sys.path.insert(0, authors_dir)
import GradDS                                                                  # authors' package
import smatrix_depth_demo as m
m.MU = 60.0                                                                    # the step size used for all my published runs

OUT = os.path.join(HERE, "results_authors"); os.makedirs(OUT, exist_ok=True)
DATASET = os.environ.get("CHECK_DATASET", "3layer")                            # "3layer" (default) or "poly"
base = os.path.join(HERE, "results_poly" if DATASET == "poly" else "results")
if DATASET == "poly":
    cfg = json.load(open(os.path.join(base, "config.json"))); m.DEFOCI = cfg["defoci"]
d = np.load(os.path.join(base, "layers_and_data.npz")); dps, layers = d["dps"], d["layers"]
sec = np.load(os.path.join(base, "sections.npz")); depths = sec["depths"]
CROP = np.s_[48:148, 48:148]
K = m.K


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def authors_inputs():
    dc = np.fft.fftshift(dps, axes=(-2, -1)).astype(np.float32).copy()       # authors expect a centred diffraction origin
    dims = np.array([m.NSCAN * m.STEP * m.DR, m.NSCAN * m.STEP * m.DR, K / m.LW, K / m.LW])   # [A, A, 1/A, 1/A]
    idx = np.arange(m.NSCAN)
    sc_ = np.zeros((len(m.DEFOCI), m.NSCAN, m.NSCAN, 2))
    c0 = (m.Y // 2 - (m.NSCAN // 2) * m.STEP) // m.STEP                         # scan origin in scan pixels (=12)
    sc_[..., 0] = (c0 + idx)[None, :, None]; sc_[..., 1] = (c0 + idx)[None, None, :]
    return dc, dims, sc_


def authors_refocus(S, beams_index, smat_dims, ts):
    return GradDS.depth_section_reconstruction(list(map(float, ts)), np.asarray(beams_index).T, m.LAM, S, smat_dims,
                                               device=torch.device("cpu"))


if mode == "refocus":
    S = np.load(os.path.join(base, "Smatrix_complex64.npy"))                      # my recovered S-matrix
    by, bx = m.beam_list(); beams = (M := m.M) * np.stack([by, bx])               # authors' beam index = M * window index
    for sgn, name in ((-1.0, "t = -z"), (+1.0, "t = +z")):
        t0 = time.time(); EW = authors_refocus(S, beams, np.array([m.Y * m.DR, m.X * m.DR]), sgn * depths)
        mine = m.depth_sections(S, (by, bx), depths)
        cs = [corr(np.angle(EW[i])[CROP], np.angle(mine[i])[CROP]) for i in range(len(depths))]
        print(f"{name}: correlation of authors' vs my refocused PHASE per depth: min {min(cs):.4f}  mean {np.mean(cs):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if sgn == -1.0:
            np.save(os.path.join(OUT, "refocus_authors_on_my_S.npy"), np.angle(EW)[(slice(None),) + CROP])
            json.dump(dict(corr_per_depth=cs), open(os.path.join(OUT, "refocus_equiv.json"), "w"))

elif mode == "equiv":
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    dc, dims, sc_ = authors_inputs(); nscan = dc.shape[0] * dc.shape[1] * dc.shape[2]
    by, bx = m.beam_list(); nb = len(by)
    cover = np.zeros((m.Y, m.X)); c0 = m.Y // 2 - (m.NSCAN // 2) * m.STEP
    for iy in range(m.NSCAN):
        for ix in range(m.NSCAN):
            for _ in m.DEFOCI:
                cover[c0 + iy * m.STEP - K // 2: c0 + iy * m.STEP + K // 2, c0 + ix * m.STEP - K // 2: c0 + ix * m.STEP + K // 2] += 1
    ncov = np.median(cover[cover > 0.5 * cover.max()])
    mu_authors = m.MU * nscan / (nb * ncov)                                       # step equivalence derived from the two update rules
    print(f"my MU={m.MU} <-> authors' mu={mu_authors:.4f}  (nscan={nscan}, nbeams={nb}, overlap={ncov:.0f})", flush=True)
    t0 = time.time()
    S_a, beams_index, smat_dims, _ = GradDS.reconstruct_Smatrix_from_datacube(
        dc.copy(), dims, m.DEFOCI, m.ALPHA * 1e3, m.EV, mu=mu_authors, niterations=niter, nchunks=20, padding=1.5,
        scan_coordinates=sc_, stream_datacube=True, report_memory=False, report_loss=False)
    print(f"authors' recovery: {niter} it in {time.time()-t0:.0f}s, S {S_a.shape}, grid {smat_dims}", flush=True)
    m.NITER = niter
    S_m, (by2, bx2), _ = m.reconstruct(dps, np.stack(np.meshgrid(c0 + m.STEP * np.arange(m.NSCAN), c0 + m.STEP * np.arange(m.NSCAN), indexing="ij"), -1), print)
    # match beams, convert normalisation (authors' |S| = 1/K, mine = 1)
    key_a = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(*beams_index))}
    order = [key_a[(int(m.M * a), int(m.M * b))] for a, b in zip(by2, bx2)]
    S_a = S_a[order] * K
    S0 = np.asarray((np.exp(2j * np.pi * (by2 / m.LW)[:, None] * (np.arange(m.Y)[None, :] * m.DR))[:, :, None]
                     * np.exp(2j * np.pi * (bx2 / m.LW)[:, None] * (np.arange(m.X)[None, :] * m.DR))[:, None, :]))
    rel = np.linalg.norm(S_a - S_m) / np.linalg.norm(S_m - S0)
    cc = [abs(np.vdot((S_a[b] - S0[b]).ravel(), (S_m[b] - S0[b]).ravel())) / (np.linalg.norm(S_a[b] - S0[b]) * np.linalg.norm(S_m[b] - S0[b]) + 1e-30) for b in range(nb)]
    print(f"after {niter} iterations: ||S_authors - S_mine|| / ||S_mine - S_init|| = {rel:.4f};  per-beam corr of the UPDATES: mean {np.mean(cc):.4f} min {np.min(cc):.4f}")
    json.dump(dict(niter=niter, mu_authors=mu_authors, rel_diff=float(rel), update_corr_mean=float(np.mean(cc)), update_corr_min=float(np.min(cc))), open(os.path.join(OUT, f"equiv_{niter}it.json"), "w"))

elif mode == "full":
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    dc, dims, sc_ = authors_inputs()
    t0 = time.time()
    S_a, beams_index, smat_dims, Loss = GradDS.reconstruct_Smatrix_from_datacube(
        dc, dims, m.DEFOCI, m.ALPHA * 1e3, m.EV, mu=1.0, niterations=niter, nchunks=30, padding=1.5,
        scan_coordinates=sc_, stream_datacube=True, report_memory=False, report_loss=True)
    print(f"authors' recovery (mu=1.0, {niter} it): {time.time()-t0:.0f}s, their loss {np.round(Loss, 4)}", flush=True)
    np.save(os.path.join(OUT, f"S_authors_mu1_{DATASET}.npy"), S_a)
    EW = authors_refocus(S_a, beams_index, smat_dims, -depths)
    ph = np.angle(EW)[(slice(None),) + CROP]
    mine = np.angle(sec["ew"])[(slice(None),) + CROP]
    out = {}
    if DATASET == "poly":
        import paper_acbf as pa
        zc = (np.arange(layers.shape[0]) + 0.5) * cfg["dz"]
        slabs_c = [layers[np.abs(zc - z) <= 20.0].sum(0)[CROP] for z in depths]
        for name, st in (("authors", ph), ("mine", mine)):
            out[name] = pa.poly_metrics(st, slabs_c, depths)
            r = out[name]
            print(f"{name}: depth err {r['mean_depth_err_A']:.1f} A, mean r {r['mean_diag_r']:+.2f}, within 10 A {r['frac_within_10A']*100:.0f}%, selectivity {r['selectivity']:+.2f}, inverted {r['n_inverted']}/{r['n_depths']}")
    else:
        import sweep_compare as sc
        sc.depths = depths; sc.ext_c = slice(48, 148)
        lz = m.LAYER_Z; layers_c = [L[CROP] for L in layers]
        for name, st in (("authors", ph), ("mine", mine)):
            C = np.array([[sc.corr(st[i], L) for L in layers_c] for i in range(len(depths))])
            r = sc.summarise(C, lz)
            out[name] = dict(depth_err=r["mean_abs_depth_err"], max_err=r["max_abs_depth_err"], mean_abs_r=r["mean_abs_r"], resolved=r["n_resolved"], found=[round(x["found_z"], 1) for x in r["layers"]])
            print(f"{name}: depth err {r['mean_abs_depth_err']:.1f} A (max {r['max_abs_depth_err']:.1f}), mean|r| {r['mean_abs_r']:.2f}, resolved {r['n_resolved']}/3, found {out[name]['found']}")
    cs = [corr(ph[i], mine[i]) for i in range(len(depths))]
    print("per-depth correlation of sections, authors' vs mine:", np.round(cs, 2).tolist(), flush=True)
    out["per_depth_corr"] = cs
    # score BOTH final S-matrices with MY forward model (relative amplitude loss on all 1875 patterns)
    by, bx = m.beam_list(); nb = len(by); kby, kbx = by / m.LW, bx / m.LW; k2 = kby ** 2 + kbx ** 2
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(*beams_index))}
    S_a_m = S_a[[key[(int(m.M * a), int(m.M * b))] for a, b in zip(by, bx)]] * K
    S_m = np.load(os.path.join(base, "Smatrix_complex64.npy"))
    c0 = m.Y // 2 - (m.NSCAN // 2) * m.STEP
    pats = dps.reshape(-1, K, K); amp = np.sqrt(pats / pats.sum((1, 2)).max())
    def loss_of(Sx):
        tot = 0.0; n = 0
        for idf in range(3):
            chi = -np.pi * m.LAM * k2 * m.DEFOCI[idf]
            for a in range(m.NSCAN):
                for b in range(m.NSCAN):
                    ry, rx = c0 + a * m.STEP, c0 + b * m.STEP
                    il = np.exp(1j * (chi - 2 * np.pi * (kby * ry + kbx * rx) * m.DR)) / np.sqrt(nb * K * K)
                    z = np.tensordot(il, Sx[:, ry - K // 2: ry + K // 2, rx - K // 2: rx + K // 2], axes=(0, 0))
                    tot += np.sum((np.abs(np.fft.fft2(z, norm="ortho")) - amp[n]) ** 2); n += 1
        return tot / np.sum(amp ** 2)
    S0 = (np.exp(2j*np.pi*kby[:, None]*(np.arange(m.Y)[None, :]*m.DR))[:, :, None] * np.exp(2j*np.pi*kbx[:, None]*(np.arange(m.X)[None, :]*m.DR))[:, None, :]).astype(np.complex64)
    for name, Sx in (("vacuum start", S0), ("authors' final S", S_a_m), ("my final S (25 it)", S_m)):
        out["loss_" + name] = float(loss_of(Sx)); print(f"my-metric relative amplitude loss, {name}: {out['loss_' + name]:.5f}", flush=True)
    json.dump(out, open(os.path.join(OUT, f"full_{niter}it_{DATASET}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(OUT, f"authors_full_sections_{DATASET}.npz"), depths=depths, phase=ph, loss=Loss)
