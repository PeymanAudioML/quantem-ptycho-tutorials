"""Step 0 of the regularization plan: is axial TV defensible on the TRUE potential of P3_XYZ?

Ground truth only (no reconstruction).  For QuantEM-like slabs of the true potential, measure the slab-to-slab change
|r(z_{k+1}) - r(z_k)| at every lateral position, split into
  within  : both slabs lie entirely in the same grain at that position
  boundary: the two slabs belong (by majority) to different grains
for several representations r of the slab:
  raw          slab phase sigma*V per A (no lateral filtering)
  lp<s>        the same, Gaussian low-passed laterally with sigma s A (removes atomic contrast, keeps coarse contrast)
  spectrum     local orientation fingerprint: |FFT| of 6.4 A Hann-windowed tiles (stride 1.6 A), unit-normalised;
               step = L2 distance between the fingerprints of neighbouring slabs
Axial TV penalises the sum of these steps.  It is defensible on a representation only if boundary steps are clearly
larger than within-grain steps (AUC near 1) and boundary pairs carry a large share of the total TV.

Slabs (--source atoms, default): every atom whose centre lies in the slab, rendered as a Gaussian (sigma 0.35 A) on the
0.2 A grid and scaled so the total projection matches the abTEM potential (same atom-to-slice rule as abTEM, at any slab
thickness).  --source slices re-bins the 2 A abTEM slices instead; that is exact only for slab thicknesses that are
multiples of 2 A, because the 2 A slices alternate ~2x in mean potential (atomic planes fall unevenly into 2 A bins), so
mixing 2 and 3 of them per 5.71 A slab adds steps that are not in the specimen.

Grain labels: the construction rule of polysi.build (nearest seed instance in the metric (1, 1, 0.05), lateral
periodic images), evaluated on a 1 A depth grid at every pixel; checked against the grain ids of the atoms in
atoms.extxyz.  Slab label = majority over the 1 A levels in the slab; purity = fraction of levels with that label.

    ../../.venv/bin/python regularized_experiment/step0_gt_axial_tv.py        (from synthetic_benchmark/)
Outputs: regularized_experiment/results/step0_gt_axial_tv/ (metrics.json, summary.md, *.png)
"""
import argparse, itertools, json, os
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
GT = os.path.join(BENCH, "data", "poly_si_3grain", "poly_si_3grain_xyz", "P3_XYZ", "ground_truth")
OUT = os.path.join(HERE, "results", "step0_gt_axial_tv")
ZONES = {0: "[001]", 1: "[110]", 2: "[111]"}


def load():
    truth = json.load(open(os.path.join(GT, "truth.json")))
    poly = json.load(open(os.path.join(GT, "polycrystal.json")))
    pot = zarr.open_array(os.path.join(GT, "potential.zarr", "array0"), mode="r")[...].astype(np.float64)
    sig = truth["potential"]["sigma_rad_per_V_A"]
    bnd = np.asarray(truth["potential"]["slice_boundaries_A"], float)
    dr = float(truth["potential"]["checks"]["sampling"][0])
    return truth, poly, pot * sig, bnd, dr


def seed_instances(poly):
    L = poly["cell_A"][0]
    inst, owner = [], []
    for g in poly["grains"]:
        for a, b in itertools.product((-1, 0, 1), repeat=2):
            inst.append(np.asarray(g["seed"], float) + [a * L, b * L, 0.0])
            owner.append(g["id"])
    return np.array(inst), np.array(owner), np.asarray(poly["metric"], float)


def labels_at(points, inst, owner, w):
    out = np.empty(len(points), np.int8)
    for c in range(0, len(points), 400000):
        p = points[c:c + 400000]
        d2 = (((p[:, None, :] - inst[None]) ** 2) * w).sum(-1)
        out[c:c + 400000] = owner[np.argmin(d2, axis=1)]
    return out


def voxel_labels(poly, nz_centres, n, dr):
    """(Z, n, n) grain id at every voxel centre (axis 0 = ase x, axis 1 = ase y, pixel i at i*dr)."""
    inst, owner, w = seed_instances(poly)
    x = np.arange(n) * dr
    X, Y = np.meshgrid(x, x, indexing="ij")
    lab = np.empty((len(nz_centres), n, n), np.int8)
    for k, z in enumerate(nz_centres):
        pts = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
        lab[k] = labels_at(pts, inst, owner, w).reshape(n, n)
    return lab


def read_atoms():
    f = open(os.path.join(GT, "atoms.extxyz"))
    nat = int(f.readline())
    f.readline()
    rows = [f.readline().split() for _ in range(nat)]
    P = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    G = np.array([int(r[4]) for r in rows])
    return P, G


def check_labels(poly, P, G):
    """Fraction of atoms whose own grain id equals the voxel-rule label at their position (expect ~1)."""
    inst, owner, w = seed_instances(poly)
    return float((labels_at(P, inst, owner, w) == G).mean())


def render(P, n, dr, sigma, scale):
    """Projected image of atoms P (x, y) on the periodic n x n grid: bilinear deposit, then a Gaussian of sigma A."""
    L = n * dr
    u, v = (P[:, 0] % L) / dr, (P[:, 1] % L) / dr
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    img = np.zeros((n, n))
    for di, dj, wt in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
        np.add.at(img, ((i0 + di) % n, (j0 + dj) % n), wt)
    return scale * gaussian_filter(img, sigma / dr, mode="wrap")


def slab_values(source, edges, P, phase, bnd_sub, n, dr, sigma, scale):
    """Slab phase per A for each slab [edges[k], edges[k+1])."""
    vals = []
    for k in range(len(edges) - 1):
        if source == "atoms":
            m = (P[:, 2] >= edges[k]) & (P[:, 2] < edges[k + 1])
            vals.append(render(P[m], n, dr, sigma, scale) / (edges[k + 1] - edges[k]))
        else:
            zc = 0.5 * (bnd_sub[1:] + bnd_sub[:-1])
            sel = (zc >= edges[k]) & (zc < edges[k + 1])
            vals.append(phase[sel].sum(0) / np.diff(bnd_sub)[sel].sum())
    return np.array(vals)


def slab_labels(lab_fine, z_fine, edges):
    """Majority grain and purity per slab and pixel from the 1 A label grid."""
    labs, pur = [], []
    for k in range(len(edges) - 1):
        L = lab_fine[(z_fine >= edges[k]) & (z_fine < edges[k + 1])]
        counts = np.stack([(L == g).sum(0) for g in range(3)])
        labs.append(counts.argmax(0).astype(np.int8))
        pur.append(counts.max(0) / len(L))
    return np.array(labs), np.array(pur)


def classes(labs, pur):
    """Masks for neighbouring slab pairs (k, k+1): within (both pure, same grain) and boundary (majority differs)."""
    within = (pur[:-1] == 1) & (pur[1:] == 1) & (labs[:-1] == labs[1:])
    boundary = labs[:-1] != labs[1:]
    return within, boundary


def stats(step, within, boundary, rng, grain_of_within=None):
    a, b = step[within], step[boundary]
    sa = a if len(a) <= 200000 else rng.choice(a, 200000, replace=False)
    sb = b if len(b) <= 200000 else rng.choice(b, 200000, replace=False)
    auc = float(mannwhitneyu(sb, sa, alternative="two-sided").statistic / (len(sa) * len(sb))) if len(a) and len(b) else np.nan
    tot = float(step[within | boundary].sum())
    out = dict(n_within=int(len(a)), n_boundary=int(len(b)), mean_within=float(a.mean()), mean_boundary=float(b.mean()),
               ratio_means=float(b.mean() / max(a.mean(), 1e-30)),
               auc_boundary_gt_within=auc,
               boundary_share_of_pairs=float(len(b) / max(len(a) + len(b), 1)),
               boundary_share_of_tv=float(b.sum() / max(tot, 1e-30)))
    if grain_of_within is not None:
        out["mean_within_by_grain"] = {ZONES[g]: float(a[grain_of_within == g].mean())
                                         for g in range(3) if (grain_of_within == g).any()}
    return out


def tile_spectra(img, size, stride):
    """Unit-normalised |FFT| of Hann-windowed size x size tiles on a stride grid (periodic): (ny, nx, size*size)."""
    n = img.shape[0]
    win = np.outer(np.hanning(size), np.hanning(size))
    pad = np.pad(img, ((0, size), (0, size)), mode="wrap")
    starts = np.arange(0, n, stride)
    out = np.empty((len(starts), len(starts), size * size), np.float32)
    for i, a in enumerate(starts):
        for j, b in enumerate(starts):
            t = pad[a:a + size, b:b + size]
            s = np.abs(np.fft.fft2((t - t.mean()) * win)).ravel()
            out[i, j] = s / max(np.linalg.norm(s), 1e-30)
    return out, starts


def tile_labels(labs, pur, starts, size):
    """Per slab and tile: majority grain over the tile's pixels and the fraction of pure pixels of that grain."""
    n = labs.shape[1]
    Lp = np.pad(labs, ((0, 0), (0, size), (0, size)), mode="wrap")
    Pp = np.pad(pur, ((0, 0), (0, size), (0, size)), mode="wrap")
    tl = np.empty((labs.shape[0], len(starts), len(starts)), np.int8)
    tp = np.empty(tl.shape)
    for i, a in enumerate(starts):
        for j, b in enumerate(starts):
            L = Lp[:, a:a + size, b:b + size].reshape(labs.shape[0], -1)
            P = Pp[:, a:a + size, b:b + size].reshape(labs.shape[0], -1)
            cnt = np.stack([((L == g) & (P == 1)).sum(1) for g in range(3)], 1)
            tl[:, i, j] = cnt.argmax(1)
            tp[:, i, j] = cnt.max(1) / L.shape[1]
    return tl, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-slabs", default="100,50,35,25,20")
    ap.add_argument("--lowpass", default="0.5,1,2,3", help="lateral Gaussian sigmas [A]")
    ap.add_argument("--tile", type=float, default=6.4, help="spectrum tile size [A]")
    ap.add_argument("--stride", type=float, default=1.6, help="spectrum tile stride [A]")
    ap.add_argument("--source", default="atoms", choices=["atoms", "slices"])
    ap.add_argument("--atom-sigma", type=float, default=0.35, help="Gaussian width of a rendered atom [A]")
    ap.add_argument("--out", default=None, help="output folder (default results/step0_gt_axial_tv[_slices])")
    a = ap.parse_args()
    global OUT
    OUT = a.out or (OUT if a.source == "atoms" else OUT + "_slices")
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    truth, poly, phase, bnd, dr = load()
    T, n = truth["thickness_A"], phase.shape[1]
    scan = truth["scan"]
    f0 = scan["positions_px_first"][0] * dr
    f1 = f0 + (scan["shape"][0] - 1) * scan["step_A"]
    field = slice(int(round(f0 / dr)), int(round(f1 / dr)) + 1)
    print(f"truth: {phase.shape} slices of {np.diff(bnd)[0]:.1f} A, {dr} A pixels; scan field {f0:.1f}-{f1:.1f} A")

    z_fine = np.arange(0.5, T, 1.0)
    lab_fine = voxel_labels(poly, z_fine, n, dr)
    P, G = read_atoms()
    nat = len(P)
    agree = check_labels(poly, P, G)
    frac = [float((lab_fine == g).mean()) for g in range(3)]
    print(f"label check: {agree:.4f} of {nat} atoms carry the voxel-rule grain; volume fractions {np.round(frac, 3)}")
    proj_true = phase.sum(0)
    scale = 1.0
    proj_atoms = render(P, n, dr, a.atom_sigma, scale)
    scale = proj_true.sum() / proj_atoms.sum()
    c = np.corrcoef(proj_true.ravel(), proj_atoms.ravel())[0, 1]
    print(f"atom rendering: sigma {a.atom_sigma} A, scale {scale:.4g}; corr with abTEM projection {c:.3f}")

    ns = [int(v) for v in a.n_slabs.split(",")]
    sigmas = [float(v) for v in a.lowpass.split(",")]
    size, stride = int(round(a.tile / dr)), int(round(a.stride / dr))
    metrics = dict(source=a.source, atom_sigma_A=a.atom_sigma, atom_render_corr_with_abtem_projection=float(c),
                   label_check_atom_agreement=agree, n_atoms=nat, grain_volume_fraction=dict(zip(ZONES.values(), frac)),
                   truth_slice_A=float(np.diff(bnd)[0]), pixel_A=dr, scan_field_A=[f0, f1], tile_A=size * dr,
                   stride_A=stride * dr, results={})
    keep = {}
    for ns_ in ns:
        edges = np.linspace(0, T, ns_ + 1)
        vals = slab_values(a.source, edges, P, phase, bnd, n, dr, a.atom_sigma, scale)
        labs, pur = slab_labels(lab_fine, z_fine, edges)
        within, boundary = classes(labs, pur)
        gw = labs[:-1][within]
        res = {}
        reps = {"raw": vals}
        for s in sigmas:
            reps[f"lp{s:g}"] = np.array([gaussian_filter(v, s / dr, mode="wrap") for v in vals])
        for name, r in reps.items():
            step = np.abs(np.diff(r, axis=0))
            res[name] = stats(step, within, boundary, rng, gw)
            fw, fb = within[:, field, field], boundary[:, field, field]
            res[name]["scan_field"] = stats(step[:, field, field], fw, fb, rng)
            if ns_ == 35 and name in ("raw", "lp2"):
                keep[name] = (r, step)
        # local orientation fingerprint
        spec = [tile_spectra(v, size, stride) for v in vals]
        starts = spec[0][1]
        sp = np.array([s[0] for s in spec])
        sstep = np.linalg.norm(np.diff(sp, axis=0), axis=-1)
        tl, tp = tile_labels(labs, pur, starts, size)
        tw = (tp[:-1] >= 0.9) & (tp[1:] >= 0.9) & (tl[:-1] == tl[1:])
        tb = tl[:-1] != tl[1:]
        res["spectrum"] = stats(sstep, tw, tb, rng, tl[:-1][tw])
        if ns_ == 35:
            keep.update(spectrum=(sstep, tl, starts), spec_masks=(tw, tb), vals=vals, labs=labs, pur=pur, edges=edges, within=within,
                        boundary=boundary)
        metrics["results"][str(ns_)] = dict(slab_A=T / ns_, **res)
        print(f"{ns_:3d} slabs ({T / ns_:.2f} A): " + "  ".join(
            f"{k} AUC {v['auc_boundary_gt_within']:.3f} ratio {v['ratio_means']:.2f} TVshare "
            f"{v['boundary_share_of_tv']:.2f}/{v['boundary_share_of_pairs']:.2f}" for k, v in res.items()))
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    plots(metrics, keep, ns, sigmas, dr, T, n, field)
    summary(metrics, ns)


def plots(m, keep, ns, sigmas, dr, T, n, field):
    reps = ["raw"] + [f"lp{s:g}" for s in sigmas] + ["spectrum"]
    thick = [T / v for v in ns]
    R = m["results"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for r in reps:
        ax[0].plot(thick, [R[str(v)][r]["auc_boundary_gt_within"] for v in ns], "o-", label=r)
        ax[1].plot(thick, [R[str(v)][r]["ratio_means"] for v in ns], "o-", label=r)
        ax[2].plot(thick, [R[str(v)][r]["boundary_share_of_tv"] for v in ns], "o-", label=r)
    ax[2].plot(thick, [R[str(v)]["raw"]["boundary_share_of_pairs"] for v in ns], "k--", label="share of pairs (raw)")
    ax[0].axhline(0.5, color="0.6", lw=0.8)
    for a_, t in zip(ax, ["AUC: P(boundary step > within-grain step)", "mean boundary step / mean within-grain step",
                          "share of total axial TV from boundary pairs"]):
        a_.set_xlabel("slab thickness (Å)"), a_.set_title(t, fontsize=10), a_.axvline(200 / 35, color="0.8", ls=":")
    ax[1].set_yscale("log"), ax[1].axhline(1, color="0.6", lw=0.8)
    ax[0].legend(fontsize=8)
    ax[2].legend(fontsize=7)
    fig.suptitle("P3_XYZ ground truth: does the slab-to-slab change mark grain boundaries? (dotted = 35 slabs, QuantEM)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "separability_vs_slab.png"), dpi=130)
    plt.close(fig)

    r35 = R["35"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a_, name in zip(ax, ["raw", "lp2", "spectrum"]):
        if name == "spectrum":
            sstep, _, _ = keep["spectrum"]
            tw, tb = keep["spec_masks"]
            bins = np.linspace(0, sstep.max(), 100)
            a_.hist(sstep[tw], bins=bins, density=True, alpha=0.6, label=f"within grain (n={tw.sum()})")
            a_.hist(sstep[tb], bins=bins, density=True, alpha=0.6, label=f"boundary (n={tb.sum()})")
            a_.legend(fontsize=8)
            a_.set_title(f"spectrum, 35 slabs: fingerprint distance\nAUC {r35[name]['auc_boundary_gt_within']:.3f}, "
                         f"mean within {r35[name]['mean_within']:.3g} / boundary {r35[name]['mean_boundary']:.3g}",
                         fontsize=9)
        else:
            _, step = keep[name]
            w, b = step[keep["within"]], step[keep["boundary"]]
            bins = np.linspace(0, np.percentile(np.r_[w, b], 99.5), 100)
            a_.hist(w, bins=bins, density=True, alpha=0.6, label=f"within grain (n={len(w)})")
            a_.hist(b, bins=bins, density=True, alpha=0.6, label=f"boundary (n={len(b)})")
            a_.set_title(f"{name}, 35 slabs: |Δz| (rad/Å)\nAUC {r35[name]['auc_boundary_gt_within']:.3f}, "
                         f"mean ratio {r35[name]['ratio_means']:.2f}", fontsize=9)
            a_.legend(fontsize=8)
        a_.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "step_histograms_35slabs.png"), dpi=130)
    plt.close(fig)

    # x-z section through the scan-field centre
    iy = (field.start + field.stop) // 2
    vals, labs, edges = keep["vals"], keep["labs"], keep["edges"]
    raw_step, lp_step = keep["raw"][1], keep["lp2"][1]
    sstep, tl, starts = keep["spectrum"]
    jt = int(np.argmin(np.abs(starts + int(round(3.2 / dr)) - iy)))
    x = np.arange(n) * dr
    ext = [0, n * dr, T, 0]
    fig, ax = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    panels = [(labs[:, :, iy], "grain (0 [001], 1 [110], 2 [111]) per slab", "viridis"),
              (vals[:, :, iy], "true slab phase per Å (35 slabs)", "gray"),
              (raw_step[:, :, iy], "|Δz| raw", "magma"),
              (lp_step[:, :, iy], "|Δz| low-passed σ = 2 Å", "magma")]
    for a_, (img, t, cm) in zip(ax, panels):
        vmax = np.percentile(img, 99.5) if "Δz" in t else None
        a_.imshow(img, aspect="auto", extent=ext, cmap=cm, interpolation="nearest", vmax=vmax)
        a_.contour(x, 0.5 * (edges[1:] + edges[:-1]), labs[:, :, iy], levels=[0.5, 1.5], colors="c", linewidths=0.8)
        a_.set_title(t, fontsize=9), a_.set_ylabel("z (Å)")
        a_.axvspan(field.start * dr, field.stop * dr, color="w", alpha=0.08)
    xs = (starts + int(round(3.2 / dr))) * dr
    ax[4].imshow(sstep[:, :, jt], aspect="auto", extent=[xs[0], xs[-1], T, 0], cmap="magma", interpolation="nearest")
    ax[4].contour(x, 0.5 * (edges[1:] + edges[:-1]), labs[:, :, iy], levels=[0.5, 1.5], colors="c", linewidths=0.8)
    ax[4].set_title("spectrum step (6.4 Å tiles)", fontsize=9), ax[4].set_ylabel("z (Å)"), ax[4].set_xlabel("x (Å)")
    fig.suptitle(f"x–z section at y = {iy * dr:.1f} Å (scan-field centre; field shaded), cyan = true grain boundaries")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xz_section_35slabs.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for g in ZONES.values():
        ax[0].plot(thick, [R[str(v)]["raw"]["mean_within_by_grain"].get(g, np.nan) for v in ns], "o-", label=g)
        ax[1].plot(thick, [R[str(v)]["lp2"]["mean_within_by_grain"].get(g, np.nan) for v in ns], "o-", label=g)
    for a_, t in zip(ax, ["raw", "low-passed σ = 2 Å"]):
        a_.plot(thick, [R[str(v)][t.split()[0] if t == "raw" else "lp2"]["mean_boundary"] for v in ns], "k--",
                label="boundary (all)")
        a_.set_xlabel("slab thickness (Å)"), a_.set_ylabel("mean |Δz| (rad/Å)"), a_.set_yscale("log")
        a_.set_title(f"within-grain step by grain orientation: {t}", fontsize=10), a_.legend(fontsize=8)
        a_.axvline(200 / 35, color="0.8", ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "within_grain_by_orientation.png"), dpi=130)
    plt.close(fig)


def summary(m, ns):
    lines = ["# Step 0 metrics (generated by step0_gt_axial_tv.py)", "",
             f"Slab source: {m['source']} (atom sigma {m['atom_sigma_A']} A; rendered projection vs abTEM projection "
             f"corr {m['atom_render_corr_with_abtem_projection']:.3f}).", "",
             f"Label check: {m['label_check_atom_agreement']:.4f} of {m['n_atoms']} atoms agree with the voxel rule. "
             f"Grain volume fractions: {m['grain_volume_fraction']}.", "",
             "| slabs | slab (Å) | representation | AUC | mean within | mean boundary | ratio of means | boundary share of TV | "
             "boundary share of pairs | AUC scan field |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in ns:
        R = m["results"][str(v)]
        for k, r in R.items():
            if k == "slab_A":
                continue
            sf = r.get("scan_field", {}).get("auc_boundary_gt_within", float("nan"))
            lines.append(f"| {v} | {R['slab_A']:.2f} | {k} | {r['auc_boundary_gt_within']:.3f} | {r['mean_within']:.3g} | "
                         f"{r['mean_boundary']:.3g} | {r['ratio_means']:.2f} | {r['boundary_share_of_tv']:.3f} | "
                         f"{r['boundary_share_of_pairs']:.3f} | {sf:.3f} |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
