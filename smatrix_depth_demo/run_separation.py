"""Two-layer axial-resolution benchmark for the S-matrix depth sections (fixed / joint / oracle probe).

Two layers with the SAME atom pattern (10 Gaussian atoms, 0.6 rad) at z = 150 -/+ sep/2 inside a 300 A slab; sep = 0
is a single layer (axial point-spread response).  Data: independent multislice simulation (simulate_4dstem), config B
probe errors, 0.4 A scan step (reduced_s2 geometry), Poisson noise 2e5 e-/pattern.  The axial response r(z) is the
correlation of the phase of each depth section (2.5 A apart) with the pattern.  Two layers count as separated when r(z)
has two local maxima with a dip between them; "Rayleigh-like" when the dip is <= 0.81 of the lower maximum.

  python run_separation.py run [--seps 0,20,40,60,80,120,160] [--variants fixed,joint,oracle] [--device auto]
  python run_separation.py analyze          # results_joint/separation_summary.{md,json}, figures/separation_*.png
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results_joint")
VAR = ["fixed", "joint", "oracle"]
COL = {"fixed": "#9aa5b1", "joint": "#1c4587", "oracle": "#e69138"}
SCHED = dict(warmup_s=5, cycles=8, s_per_cycle=3, p_per_cycle=2, final_s=3, final_p=1, mu=60.0,
             probe_lr=0.3, probe_fraction=0.34, grad_clip=1.0)          # identical to the other experiments


def run(seps, variants, seed, device, step):
    import torch
    import run_joint_probe as R
    torch.set_num_threads(os.cpu_count())
    for sep in seps:
        for v in variants:
            R.run_one(f"sep{sep:g}_s{step}", "B", v, seed, SCHED, 2e5, ["C10", "C12a", "C12b"], device=device)


def peaks(z, r):
    """Local maxima of r (plateaus count once), sorted by height."""
    idx = [i for i in range(1, len(r) - 1) if r[i] >= r[i - 1] and r[i] > r[i + 1]]
    if r[0] > r[1]:
        idx.insert(0, 0)
    if r[-1] > r[-2]:
        idx.append(len(r) - 1)
    return sorted(idx, key=lambda i: -r[i])


def analyze_curve(z, r, layer_z):
    import joint_smatrix_probe as jp
    out = dict(true_sep=float(layer_z[-1] - layer_z[0]) if len(layer_z) > 1 else 0.0)
    pk = peaks(z, r)
    if len(layer_z) == 1:
        w, open_ = jp.axial_fwhm(z, r)
        out.update(fwhm=w, fwhm_open=open_, peak_z=float(z[pk[0]]))
        return out
    # two highest maxima that are at least 5 A apart
    p1 = pk[0]; p2 = next((i for i in pk[1:] if abs(z[i] - z[p1]) >= 5.0), None)
    if p2 is None:
        out.update(n_peaks=1, separated=False, rayleigh=False, dip_ratio=1.0, est_sep=0.0, peak_z=[float(z[p1])])
        return out
    a, b = sorted((p1, p2))
    dip = float(r[a:b + 1].min()); low = float(min(r[a], r[b]))
    near = all(min(abs(z[i] - t) for t in layer_z) <= max(10.0, out["true_sep"] / 4) for i in (a, b))
    out.update(n_peaks=2, dip_ratio=dip / low if low > 0 else 1.0, est_sep=float(z[b] - z[a]),
               peak_z=[float(z[a]), float(z[b])], near_truth=bool(near))
    out["separated"] = bool(out["dip_ratio"] < 0.99 and near)
    out["rayleigh"] = bool(out["dip_ratio"] <= 0.81 and near)
    return out


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "sep*_s*", "B", "*_seed*.json"))):
        r = json.load(open(f))
        dm = r["eval"]["depth"]
        C = np.array(dm["curves"])
        z = np.arange(C.shape[0]) * (2.5 if "sep" in r["sample"] else 10.0)
        layer_z = [L["true_z"] for L in dm["layers"]]
        res = analyze_curve(z, C[:, 0], layer_z)                      # both layers have the same pattern
        res.update(sample=r["sample"], variant=r["variant"], seed=r["seed"], z=z.tolist(), r=C[:, 0].tolist(),
                   layer_z=layer_z, final_loss=r["summary"]["final_loss"],
                   probe_err=float(np.mean(r["eval"]["probe_wavefunction_error"][1:])))
        rows.append(res)
    return rows


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load()
    if not rows:
        print("no separation runs found"); return
    seps = sorted({r["true_sep"] for r in rows})
    lines = ["| separation (Å) | variant | peaks | dip / lower peak | separated (two maxima at the layers) | Rayleigh-like (dip ≤ 0.81) | "
             "estimated separation (Å) | peak depths (Å) | single-layer FWHM (Å) | probe err |", "|" + "---|" * 10]
    for sp in seps:
        for v in VAR:
            for r in [r for r in rows if r["true_sep"] == sp and r["variant"] == v]:
                if sp == 0:
                    lines.append(f"| 0 (single layer) | {v} | – | – | – | – | – | {r['peak_z']:.1f} | "
                                 f"{r['fwhm']:.0f}{' (lower bound)' if r['fwhm_open'] else ''} | {r['probe_err']:.3f} |")
                else:
                    lines.append(f"| {sp:g} | {v} | {r['n_peaks']} | {r['dip_ratio']:.3f} | {'yes' if r['separated'] else 'no'} | "
                                 f"{'yes' if r['rayleigh'] else 'no'} | {r['est_sep']:.1f} | "
                                 f"{', '.join(f'{p:.1f}' for p in r['peak_z'])} | – | {r['probe_err']:.3f} |")
    open(os.path.join(OUT, "separation_summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump([{k: v for k, v in r.items() if k not in ("z", "r")} for r in rows],
              open(os.path.join(OUT, "separation_summary.json"), "w"), indent=1)
    print("\n".join(lines))

    fig, ax = plt.subplots(1, len(seps), figsize=(2.9 * len(seps), 3.2), sharey=True, squeeze=False)
    for a, sp in zip(ax[0], seps):
        for v in VAR:
            for r in [r for r in rows if r["true_sep"] == sp and r["variant"] == v]:
                a.plot(r["z"], r["r"], color=COL[v], label=v, lw=1.2)
                lz = r["layer_z"]
        for t in lz:
            a.axvline(t, color="k", ls="--", lw=0.7)
        a.set_title("single layer" if sp == 0 else f"Δz = {sp:g} Å", fontsize=9); a.set_xlabel("section depth z (Å)")
        a.set_xlim(0, 300); a.grid(alpha=0.3)
    ax[0, 0].set_ylabel("correlation with the layer pattern"); ax[0, 0].legend(fontsize=7)
    fig.suptitle("Two identical layers at 150 ∓ Δz/2: axial response of the S-matrix depth sections (dashed = true depths)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "separation_axial.png")); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for v in VAR:
        rr = sorted([r for r in rows if r["variant"] == v and r["true_sep"] > 0], key=lambda r: r["true_sep"])
        if rr:
            ax[0].plot([r["true_sep"] for r in rr], [r["dip_ratio"] for r in rr], "o-", color=COL[v], label=v)
            ax[1].plot([r["true_sep"] for r in rr], [r["est_sep"] for r in rr], "o-", color=COL[v], label=v)
    ax[0].axhline(0.81, color="k", ls=":", lw=0.8); ax[0].set_ylabel("dip / lower peak (1 = no dip)")
    ax[1].plot([0, max(seps)], [0, max(seps)], "k:", lw=0.8); ax[1].set_ylabel("separation of the two maxima (Å)")
    for a in ax:
        a.set_xlabel("true layer separation (Å)"); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "separation_summary.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seps", default="0,20,40,60,80,120,160")
    ap.add_argument("--variants", default="fixed,joint,oracle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.cmd == "run":
        run([float(x) for x in a.seps.split(",")], a.variants.split(","), a.seed, a.device, a.step)
    else:
        analyze()


if __name__ == "__main__":
    main()
