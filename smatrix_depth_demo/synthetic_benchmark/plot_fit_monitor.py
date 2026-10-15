"""Live ground-truth comparison of a running joint S-matrix + probe fit (run_benchmark.py smatrix --monitor-every 1).

    python plot_fit_monitor.py --config configs/poly_si_3grain.yaml --specimen P3_XY \
        --run runs/poly_si_3grain/P3_XY/smatrix_window_noiseless_s0_full80 --log smatrix_P3_XY_full80.log

Every --every seconds, for each new <run>/monitor/epoch_NNN.npz, compares with the ground truth and redraws
  <run>/monitor/progress.png   loss; probe corrections vs truth; probe RMS phase error; object correlation per depth;
                               depth selectivity; focus curve vs the true depth profile
  <run>/monitor/sections.png   latest refocused phase sections vs the true potential in a slab around the same depth
  <run>/monitor/metrics.csv    one row per snapshot
Object comparison: the refocused S-matrix phase is contrast, not a potential, so only correlations are reported.  Truth
slab at depth z = sigma * sum of the true potential slices with centres in [z - 10, z + 10) A, Gaussian blur 0.2 A;
"projection" = the whole 0..T sum (control: a section that matches the projection as well as its slab carries no depth
information).  Depth profile truth = variance of the true potential per 2 A slice over the scanned field.
  <run>/monitor/potential.png  S-matrix projected-potential estimate vs the true projected potential, in rad:
                               phase of the refocused wave at mid-depth (all beams demodulated and summed; for a thin
                               object this equals the transmission function, so its phase is sigma * V projected),
                               2-D unwrapped, global phase offset removed (least squares).  An approximation for a thick
                               specimen (tilt averaging, multiple scattering) - not a reconstruction.
"""
import argparse, csv, glob, json, os, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import bench
import run_realization as rr
import simple_joint as sj

SLAB = 10.0


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-30))


class Truth:
    def __init__(self, cfg, name):
        import zarr
        self.cfg = cfg
        self.setup = bench.make_setup(cfg)
        gt = os.path.join(bench.specimen_dir(cfg, name), "ground_truth")
        t = json.load(open(os.path.join(gt, "truth.json")))
        self.V = zarr.open_array(os.path.join(gt, "potential.zarr", "array0"), mode="r")[...].astype(np.float32)
        self.sigma = t["potential"]["sigma_rad_per_V_A"]
        b = np.asarray(t["potential"]["slice_boundaries_A"])
        self.zc = 0.5 * (b[1:] + b[:-1])
        self.T = t["thickness_A"]
        self.true = np.array(t["aberrations"]["true"])
        self.nominal = np.array(t["aberrations"]["nominal"])
        self.ref = t["aberrations"]["reference_dataset"]
        from ase.io import read
        at = read(os.path.join(gt, "atoms.extxyz"))
        self.P = at.positions
        self.G = at.arrays.get("grain")
        ky, kx = self.setup.beams()
        self.ky, self.kx = ky, kx
        import torch
        self.basis = sj.aberration_basis(torch.as_tensor(ky), torch.as_tensor(kx), self.setup.lam, sj.NAMES).numpy()
        self.amp = self.setup.aperture(np.hypot(ky, kx))
        self.cache = {}

    def probe(self, coeff):
        return self.amp * np.exp(-1j * (np.asarray(coeff) @ self.basis))

    def region(self, offset, c0, n):
        """Cell pixel indices (rows, cols) of S-grid crop [c0, c0 + n)."""
        r = (offset[0] + c0 + np.arange(n)) % self.V.shape[1]
        c = (offset[1] + c0 + np.arange(n)) % self.V.shape[2]
        return r, c

    def slab(self, z0, z1, rows, cols):
        key = (round(z0, 3), round(z1, 3), rows[0], cols[0], len(rows))
        if key not in self.cache:
            sel = (self.zc >= z0) & (self.zc < z1)
            img = self.V[sel].sum(0) if sel.any() else np.zeros(self.V.shape[1:], np.float32)
            img = gaussian_filter(img * self.sigma, 1.0, mode="wrap")
            self.cache[key] = img[np.ix_(rows, cols)]
        return self.cache[key]

    def depth_profile(self, rows, cols):
        sub = self.V[:, rows][:, :, cols]
        return sub.reshape(len(sub), -1).var(axis=1)

    def boundary_depths(self, lo, hi, d_max=2.0):
        """z of atoms within d_max of another grain inside the lateral box [lo, hi] (cell frame); None if no grains."""
        if self.G is None:
            return None
        import polysi
        L = self.V.shape[1] * self.setup.dr
        d = polysi.distance_to_other_grains(self.P, self.G, L, self.T)
        m = np.all((self.P[:, :2] >= lo) & (self.P[:, :2] <= hi), axis=1) & (d < d_max)
        return self.P[m, 2]


def evaluate(tr, f):
    z = np.load(f)
    ep = int(z["epoch"])
    c0, c1 = map(int, z["crop"])
    n = c1 - c0
    rows, cols = tr.region(z["offset_px"], c0, n)
    coeff = z["coefficients"]
    row = dict(epoch=ep, loss=float(z["loss"][-1]))
    for d in range(len(coeff)):
        if d == tr.ref:
            continue
        for j, nm in enumerate(("C10", "C12a", "C12b")):
            row[f"d{d}_{nm}_err"] = float(coeff[d, j] - tr.true[d, j])
        ov, rms = rr.probe_match(tr.probe(coeff[d]), tr.probe(tr.true[d]), tr.ky, tr.kx)
        row[f"d{d}_probe_overlap"], row[f"d{d}_probe_rms_rad"] = ov, rms
    proj = tr.slab(-1, tr.T + 1, rows, cols)
    cs, cp = [], []
    for k, zz in enumerate(z["zs"]):
        s = z["sections"][k]
        cs.append(corr(s, tr.slab(zz - SLAB, zz + SLAB, rows, cols)))
        cp.append(corr(s, proj))
        row[f"corr_slab_z{zz:.0f}"], row[f"corr_proj_z{zz:.0f}"] = cs[-1], cp[-1]
    row["corr_slab_mean"], row["corr_proj_mean"] = float(np.mean(cs)), float(np.mean(cp))
    row["depth_selectivity"] = float(np.mean(np.array(cs) - np.array(cp)))
    zg, curve = z["zg"], z["curve"]
    inside = (zg >= 0) & (zg <= tr.T)
    prof = np.interp(zg[inside], tr.zc, tr.depth_profile(rows, cols))
    row["focus_curve_vs_truth_corr"] = corr(curve[inside], prof)
    row["layers"] = " ".join(f"{v:.0f}" for v in z["layers"])
    if "wave_mid" in z.files:
        est, tru = potential_estimate(z["wave_mid"]), tr.slab(-1, tr.T + 1, rows, cols)
        b = float((tru - est).mean())
        est = est + b                                          # global phase offset (least squares)
        A = np.stack([tru.ravel(), np.ones(tru.size)], 1)
        (sl, off), *_ = np.linalg.lstsq(A, est.ravel(), rcond=None)
        row.update(pot_corr=corr(est, tru), pot_rmse_rad=float(np.sqrt(((est - tru) ** 2).mean())),
                   pot_slope=float(sl), pot_rmse_after_fit_rad=float(np.sqrt(((est - sl * tru - off) ** 2).mean())),
                   pot_truth_std_rad=float(tru.std()))
    return row, z, rows, cols


def potential_estimate(w):
    """Unwrapped phase [rad] of the refocused S-matrix wave (projected-potential estimate, up to a constant)."""
    from skimage.restoration import unwrap_phase
    return unwrap_phase(np.angle(w)).astype(np.float64)


def draw(tr, rows_all, last, out, title):
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    ep = np.array([r["epoch"] for r in rows_all])
    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    ax[0, 0].semilogy(ep, [r["loss"] for r in rows_all], "k.-")
    ax[0, 0].axvspan(0.5, 8.5, color="0.9", label="warm-up (S only)")
    ax[0, 0].set_title("training loss"), ax[0, 0].set_xlabel("epoch"), ax[0, 0].legend(fontsize=8)
    cols = plt.cm.tab10.colors
    ds = sorted({int(k[1]) for k in rows_all[0] if re.match(r"d\d_C10_err", k)})
    for j, nm in enumerate(("C10", "C12a", "C12b")):
        a = ax[0, 1 + j]
        for d in ds:
            a.plot(ep, [r[f"d{d}_{nm}_err"] for r in rows_all], ".-", color=cols[d], label=f"dataset {d}")
        a.axhline(0, color="k", lw=0.8)
        a.set_title(f"{nm} error = fit − truth (Å); nominal start = {tr.nominal[ds[0], j] - tr.true[ds[0], j]:+.0f} / "
                    + " / ".join(f"{tr.nominal[d, j] - tr.true[d, j]:+.0f}" for d in ds[1:]), fontsize=9)
        a.set_xlabel("epoch"), a.legend(fontsize=8)
    for d in ds:
        ax[1, 0].plot(ep, [r[f"d{d}_probe_rms_rad"] for r in rows_all], ".-", color=cols[d], label=f"dataset {d}")
    ax[1, 0].set_title("probe RMS phase error vs true probe (rad)"), ax[1, 0].set_xlabel("epoch"), ax[1, 0].legend(fontsize=8)
    zs = last["zs"]
    cm = plt.cm.viridis(np.linspace(0, 1, len(zs)))
    for k, zz in enumerate(zs):
        ax[1, 1].plot(ep, [r[f"corr_slab_z{zz:.0f}"] for r in rows_all], "-", color=cm[k], label=f"z {zz:.0f} Å")
    ax[1, 1].plot(ep, [r["corr_proj_mean"] for r in rows_all], "k--", label="vs projection (mean)")
    ax[1, 1].set_title(f"object: corr(refocused section, true ±{SLAB:g} Å slab)"), ax[1, 1].set_xlabel("epoch")
    ax[1, 1].legend(fontsize=6, ncol=2)
    ax[1, 2].plot(ep, [r["depth_selectivity"] for r in rows_all], "k.-")
    ax[1, 2].axhline(0, color="0.5", lw=0.8)
    ax[1, 2].set_title("depth selectivity = mean(corr slab − corr projection)"), ax[1, 2].set_xlabel("epoch")
    zg, curve = last["zg"], last["curve"]
    inside = (zg >= 0) & (zg <= tr.T)
    prof = tr.depth_profile(*last["_rc"])
    nrm = lambda v: (v - v.min()) / max(np.ptp(v), 1e-30)
    ax[1, 3].plot(zg, nrm(curve), "k-", label=f"S-matrix focus curve (epoch {int(last['epoch'])})")
    ax[1, 3].plot(tr.zc, nrm(prof), "g-", label="truth: potential variance per slice")
    for v in last["layers"]:
        ax[1, 3].axvline(v, color="k", ls=":", lw=1)
    bd = last.get("_bd")
    if bd is not None and len(bd):
        h, e = np.histogram(bd, bins=np.arange(0, tr.T + 4, 4))
        ax[1, 3].bar(0.5 * (e[1:] + e[:-1]), h / max(h.max(), 1), width=4, color="tab:red", alpha=0.3,
                     label="truth: grain-boundary atoms (z)")
    ax[1, 3].axvspan(zg.min(), 0, color="0.9"), ax[1, 3].axvspan(tr.T, zg.max(), color="0.9")
    ax[1, 3].set_title(f"depth: focus curve vs truth (corr {rows_all[-1]['focus_curve_vs_truth_corr']:.2f}); "
                       f"layers {rows_all[-1]['layers'] or '-'}", fontsize=9)
    ax[1, 3].set_xlabel("depth z (Å)"), ax[1, 3].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "progress.png"), dpi=95)
    plt.close(fig)
    # sections
    pick = np.linspace(0, len(zs) - 1, 5).round().astype(int)
    fig, ax = plt.subplots(2, 5, figsize=(19, 8))
    rws, cls = last["_rc"]
    for j, k in enumerate(pick):
        zz = zs[k]
        s = last["sections"][k]
        t = tr.slab(zz - SLAB, zz + SLAB, rws, cls)
        ax[0, j].imshow(s, cmap="gray"), ax[0, j].set_title(f"S-matrix phase, z {zz:.0f} Å (epoch {int(last['epoch'])})",
                                                            fontsize=8)
        ax[1, j].imshow(t, cmap="gray"), ax[1, j].set_title(f"truth σV, {zz - SLAB:.0f}–{zz + SLAB:.0f} Å; corr "
                                                            f"{rows_all[-1][f'corr_slab_z{zz:.0f}']:.2f}", fontsize=8)
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle(title + " — refocused sections (top) vs true potential slabs (bottom), scanned field", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sections.png"), dpi=95)
    plt.close(fig)
    # projected-potential estimate vs truth (rad)
    if "wave_mid" in last:
        rws, cls = last["_rc"]
        tru = tr.slab(-1, tr.T + 1, rws, cls)
        est = potential_estimate(last["wave_mid"])
        est = est + float((tru - est).mean())
        r = rows_all[-1]
        fig, ax = plt.subplots(1, 4, figsize=(21, 5))
        vmin, vmax = np.percentile(tru, [0.5, 99.5])
        for a_, img, t in ((ax[0], est, f"S-matrix estimate (rad), epoch {int(last['epoch'])}"),
                           (ax[1], tru, "truth σ·ΣV (rad), blur 0.2 Å")):
            im = a_.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
            a_.set_title(t), a_.axis("off")
            fig.colorbar(im, ax=a_, shrink=0.8)
        dmax = np.percentile(np.abs(est - tru), 99)
        im = ax[2].imshow(est - tru, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        ax[2].set_title(f"estimate − truth (rad): RMSE {r['pot_rmse_rad']:.3f}, slope {r['pot_slope']:.2f}"), ax[2].axis("off")
        fig.colorbar(im, ax=ax[2], shrink=0.8)
        ee = [x for x in rows_all if "pot_corr" in x]
        ax[3].plot([x["epoch"] for x in ee], [x["pot_corr"] for x in ee], "k.-", label="correlation")
        ax3 = ax[3].twinx()
        ax3.plot([x["epoch"] for x in ee], [x["pot_rmse_rad"] for x in ee], "r.-", label="RMSE (rad)")
        ax3.axhline(ee[-1]["pot_truth_std_rad"], color="r", ls=":", lw=1, label="truth std (rad)")
        ax[3].set_xlabel("epoch"), ax[3].set_ylabel("correlation"), ax3.set_ylabel("RMSE (rad)", color="r")
        ax[3].legend(loc="lower left", fontsize=8), ax3.legend(loc="upper right", fontsize=8)
        ax[3].set_title("projected potential vs truth, per epoch")
        fig.suptitle(title + " — S-matrix projected-potential estimate (phase of the refocused wave at mid-depth, "
                     "unwrapped; an approximation for 200 Å, not a reconstruction)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "potential.png"), dpi=95)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", default=None, help="fit log; the loop stops when its summary line appears")
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    tr = Truth(cfg, a.specimen)
    mdir = os.path.join(a.run, "monitor")
    done, rows_all, bd = {}, [], "unset"
    while True:
        files = sorted(glob.glob(os.path.join(mdir, "epoch_*.npz")))
        new = [f for f in files if f not in done]
        for f in new:
            try:
                row, z, rws, cls = evaluate(tr, f)
            except Exception as e:                      # file still being written
                print(f"skip {f}: {e}", flush=True)
                continue
            done[f] = row
            last = dict(z)
            last["_rc"] = (rws, cls)
            if isinstance(bd, str):
                lo = (rws[0] * tr.setup.dr, cls[0] * tr.setup.dr)
                hi = (rws[-1] * tr.setup.dr, cls[-1] * tr.setup.dr)
                bd = tr.boundary_depths(np.array(lo), np.array(hi))
            last["_bd"] = bd
        if new and done:
            rows_all = sorted(done.values(), key=lambda r: r["epoch"])
            with open(os.path.join(mdir, "metrics.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows_all[-1].keys()))
                w.writeheader()
                w.writerows(rows_all)
            draw(tr, rows_all, last, mdir, f"{a.specimen}: {os.path.basename(a.run)} — epoch {rows_all[-1]['epoch']}")
        finished = a.log and os.path.exists(a.log) and any(re.search(r"^\[smatrix\S*\] .*\d+ of \d+ epochs", l) for l in open(a.log))
        if a.once or (finished and not new):
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
