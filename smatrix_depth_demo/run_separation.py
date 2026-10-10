"""Two-layer axial-resolution benchmark for S-matrix depth sectioning.

Samples (run_joint_probe.sample): the SAME atom pattern (10 Gaussian atoms, 0.6 rad) in two layers at
z = 150 -/+ dz/2 inside a 300 A slab (sep{dz}), plus two controls at z = 150:
  sep0    one layer with TWICE the pattern phase (same total scattering as the two-layer samples): must not be
          reported as two layers;
  sep0x1  one layer with the single pattern phase: the single-layer response used for the superposition check
          and as the fitting template.
Data: independent multislice simulation (simulate_4dstem), config B data, 0.4 A scan step (reduced_s2 geometry),
Poisson noise 2e5 e-/pattern, sections every 2.5 A.  The pilot uses the TRUE probe ("oracle") only, so probe estimation
cannot affect the result.

Signal: linear projection of each depth-section phase onto the pattern,
    s(z) = <phi_z - mean, p - mean> / <p - mean, p - mean>
(fixed normalisation, so responses of independent layers add; Pearson correlation is reported for reference only).

Decision rules (fixed before running):
  two peaks       two distinct interior local maxima of s(z) (the two highest are used)
  localised       sorted peaks matched one-to-one to the sorted true depths, each within dz/4
  Rayleigh-like   localised and valley / lower peak <= 0.81
  control         the sep0 (double-strength) response, tested with the same dz, must not pass "localised"
  model fit       one- vs two-layer least-squares fits of shifted single-layer templates (+ offset).  The template
                  has its own noise, so BIC is calibrated with the control: model-fittable if
                  dBIC(test) - dBIC(control) > 10, |dz_fit - dz| <= dz/4, the 1-sigma profile half-width <= dz/4
                  (statistical only: correlated depth samples and template noise make it optimistic) and the
                  control's own fitted separation is < dz - dz/4
Outcome per dz: resolved / model-fittable / not resolved / inconclusive (control fails, or fit prefers two layers
for the control too).  One seed = pilot; a separation that passes must be repeated with two more seeds.

  python run_separation.py run --seps 0,0x1,20,40,60 --variants oracle --seed 0 --step 2 --device auto
  python run_separation.py analyze [--variant oracle] [--step 2]
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results_joint")
SCHED = dict(warmup_s=5, cycles=8, s_per_cycle=3, p_per_cycle=2, final_s=3, final_p=1, mu=60.0,
             probe_lr=0.3, probe_fraction=0.34, grad_clip=1.0)          # identical to the other experiments
RAYLEIGH = 0.81
DBIC = 10.0


def run(seps, variants, seed, device, step):
    import torch
    import run_joint_probe as R
    torch.set_num_threads(os.cpu_count())
    for sep in seps:
        for v in variants:
            R.run_one(f"sep{sep}_s{step}", "B", v, seed, SCHED, 2e5, ["C10", "C12a", "C12b"], device=device)


# ------------------------------------------------------------------------------------------------ signals
def section_signals(sname, sections_file):
    """Linear projection s(z) and Pearson r(z) of every depth-section phase onto the layer pattern."""
    import run_joint_probe as R
    s = R.sample(sname)
    d = np.load(sections_file)
    ph, z = d["phase"].astype(np.float64), d["depths"]
    p = s["pattern"][s["geom"].crop()]; p = p - p.mean()
    lin, pear = [], []
    for f in ph:
        f = np.angle(np.exp(1j * (f - np.angle(np.mean(np.exp(1j * f))))))   # remove the global phase, no wrapping
        f = f - f.mean()
        lin.append(float((f * p).sum() / (p * p).sum()))
        pear.append(float((f * p).sum() / np.sqrt((f * f).sum() * (p * p).sum())))
    return z, np.array(lin), np.array(pear), s


def noise_sigma(sig):
    """Robust white-noise estimate from second differences (MAD)."""
    d2 = np.diff(sig, 2)
    return float(np.median(np.abs(d2 - np.median(d2))) / 0.6745 / np.sqrt(6.0))


def prominent_maxima(sig, min_prom):
    """Interior local maxima whose topographic prominence is >= min_prom, highest first."""
    mx = [i for i in range(1, len(sig) - 1) if sig[i] >= sig[i - 1] and sig[i] > sig[i + 1]]
    out = []
    for i in mx:
        l = i
        while l > 0 and sig[l - 1] <= sig[i]:
            l -= 1
        r = i
        while r < len(sig) - 1 and sig[r + 1] <= sig[i]:
            r += 1
        left_base = sig[l:i + 1].min() if l > 0 else sig[:i + 1].min()
        right_base = sig[i:r + 1].min() if r < len(sig) - 1 else sig[i:].min()
        if sig[i] - max(left_base, right_base) >= min_prom:
            out.append(i)
    return sorted(out, key=lambda i: -sig[i])


def classify(z, sig, layer_z):
    """Peak / localisation / Rayleigh-like tests for two layers at layer_z (sorted).  Maxima must be prominent by at
    least max(3 sigma_noise, 2 % of the signal maximum) so that noise ripples are not counted as layers."""
    dz = layer_z[1] - layer_z[0]
    mx = prominent_maxima(sig, max(3 * noise_sigma(sig), 0.02 * float(np.abs(sig).max())))
    out = dict(n_maxima=len(mx), two_peaks=len(mx) >= 2, localised=False, rayleigh=False, valley_ratio=None,
               peak_z=[float(z[i]) for i in mx[:2]])
    if len(mx) < 2:
        return out
    a, b = sorted(mx[:2])
    valley = float(sig[a:b + 1].min()); low = float(min(sig[a], sig[b]))
    out["valley_ratio"] = valley / low if low > 0 else None
    out["peak_z"] = [float(z[a]), float(z[b])]
    out["localised"] = bool(abs(z[a] - layer_z[0]) <= dz / 4 and abs(z[b] - layer_z[1]) <= dz / 4)
    out["rayleigh"] = bool(out["localised"] and out["valley_ratio"] is not None and out["valley_ratio"] <= RAYLEIGH)
    return out


def shifted(z, template, shift):
    """template(z - shift) by linear interpolation (edge values held)."""
    return np.interp(z - shift, z, template)


def fit_models(z, sig, template, z_t=150.0, window=(20.0, 280.0), grid=1.0):
    """One-layer vs two-layer fit of shifted copies of the single-layer template (free amplitudes + offset)."""
    m = (z >= window[0]) & (z <= window[1])
    y = sig[m]; n = len(y)
    pos = np.arange(window[0], window[1] + grid / 2, grid)
    T = np.array([shifted(z, template, p - z_t)[m] for p in pos])          # (P, n)
    one = np.ones(n)
    best1 = (np.inf, None)
    for i, t in enumerate(T):
        A = np.stack([t, one], 1); c, *_ = np.linalg.lstsq(A, y, rcond=None); r = float(((A @ c - y) ** 2).sum())
        if r < best1[0]:
            best1 = (r, pos[i])
    rss2 = np.full((len(pos), len(pos)), np.inf)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            A = np.stack([T[i], T[j], one], 1); c, *_ = np.linalg.lstsq(A, y, rcond=None)
            rss2[i, j] = float(((A @ c - y) ** 2).sum())
    i2, j2 = np.unravel_index(np.argmin(rss2), rss2.shape)
    r2 = float(rss2[i2, j2])
    bic1 = n * np.log(best1[0] / n) + 3 * np.log(n)
    bic2 = n * np.log(r2 / n) + 5 * np.log(n)
    # profile over the separation for a 1-sigma interval (Delta chi^2 = 1 with sigma^2 from the two-layer residual)
    sig2 = r2 / (n - 5)
    seps = np.round(pos[None, :] - pos[:, None], 6)
    prof = {}
    for d in np.unique(seps[seps > 0]):
        prof[d] = float(rss2[seps == d].min())
    ok = [d for d, r in prof.items() if (r - r2) / sig2 <= 1.0]
    return dict(z1_one=float(best1[1]), dz_fit=float(pos[j2] - pos[i2]), z_fit=[float(pos[i2]), float(pos[j2])],
                dBIC=float(bic1 - bic2), ci=[float(min(ok)), float(max(ok))], rss1=best1[0], rss2=r2, n=n)


# ------------------------------------------------------------------------------------------------ analysis
def load_runs(variant, step):
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, f"sep*_s{step}", "B", f"{variant}_seed*.json"))):
        r = json.load(open(f))
        z, lin, pear, s = section_signals(r["sample"], f.replace(".json", "_sections.npz"))
        runs[(r["sample"], r["seed"])] = dict(z=z, s=lin, r=pear, layer_z=s["layer_z"], sep=s["sep"], mult=s["mult"],
                                              final_loss=r["summary"]["final_loss"])
    return runs


def analyze(variant, step):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = load_runs(variant, step)
    seeds = sorted({k[1] for k in runs})
    report, lines = [], []
    lines += [f"Two-layer benchmark, variant = {variant}, scan step = {0.2 * step:.1f} Å, seeds = {seeds} "
              f"({'pilot' if len(seeds) < 3 else 'confirmation'})", "",
              "| Δz (Å) | seed | maxima | peak depths (Å) | valley / lower peak | localised (±Δz/4, 1:1) | Rayleigh ≤ 0.81 | "
              "control falsely localised | superposition rel. error | superposition predicts Rayleigh | "
              "2-layer fit Δz (Å) [1σ, statistical only] | ΔBIC (1 vs 2) | control ΔBIC / Δz_fit | outcome |", "|" + "---|" * 14]
    for seed in seeds:
        ctrl = runs.get((f"sep0_s{step}", seed)); single = runs.get((f"sep0x1_s{step}", seed))
        if single is None:
            print(f"seed {seed}: sep0x1 (single-layer template) missing"); continue
        z, s1 = single["z"], single["s"]
        cfit = fit_models(z, ctrl["s"], s1) if ctrl else None
        lin_err = (float(np.linalg.norm(ctrl["s"] - 2 * s1) / np.linalg.norm(ctrl["s"])) if ctrl else None)
        for (sname, sd), rr in sorted(runs.items(), key=lambda kv: kv[1]["sep"]):
            if sd != seed or rr["sep"] == 0:
                continue
            dz, lz = rr["sep"], rr["layer_z"]
            c = classify(z, rr["s"], lz)
            cc = classify(z, ctrl["s"], lz) if ctrl else None
            pred = shifted(z, s1, -dz / 2) + shifted(z, s1, dz / 2)
            sup_err = float(np.linalg.norm(rr["s"] - pred) / np.linalg.norm(rr["s"]))
            cp = classify(z, pred, lz)
            fit = fit_models(z, rr["s"], s1)
            half = 0.5 * (fit["ci"][1] - fit["ci"][0])
            # the template carries its own noise, so BIC alone over-prefers two layers: calibrate against the control
            ctrl_dbic = cfit["dBIC"] if cfit else 0.0
            fittable = (fit["dBIC"] - ctrl_dbic > DBIC and abs(fit["dz_fit"] - dz) <= dz / 4 and half <= dz / 4)
            ctrl_bad = cc is not None and cc["localised"]
            ctrl_fit_bad = cfit is not None and cfit["dz_fit"] >= dz - dz / 4
            if c["rayleigh"] and not ctrl_bad:
                outcome = "resolved"
            elif ctrl_bad or (fittable and ctrl_fit_bad):
                outcome = "inconclusive"
            elif fittable:
                outcome = "model-fittable"
            else:
                outcome = "not resolved"
            row = dict(dz=dz, seed=seed, classify=c, control=cc, superposition_rel_err=sup_err, superposition_classify=cp,
                       fit=fit, control_fit=cfit, outcome=outcome)
            report.append(row)
            vr = "–" if c["valley_ratio"] is None else f"{c['valley_ratio']:.3f}"
            lines.append(f"| {dz:g} | {seed} | {c['n_maxima']} | {', '.join(f'{p:.1f}' for p in c['peak_z']) or '–'} | {vr} | "
                         f"{'yes' if c['localised'] else 'no'} | {'yes' if c['rayleigh'] else 'no'} | "
                         f"{'YES' if ctrl_bad else 'no'} | {sup_err:.3f} | {'yes' if cp['rayleigh'] else 'no'} | "
                         f"{fit['dz_fit']:.1f} [{fit['ci'][0]:.1f}, {fit['ci'][1]:.1f}] | {fit['dBIC']:.1f} | "
                         f"{cfit['dBIC']:.1f} / {cfit['dz_fit']:.1f} | **{outcome}** |")
        lines += ["", f"Seed {seed} controls: single-layer peak at {z[np.argmax(s1)]:.1f} Å (true 150), "
                  f"FWHM {__import__('joint_smatrix_probe').axial_fwhm(z, s1)[0]:.1f} Å; "
                  f"double-strength control vs 2 × single-layer response: rel. difference {lin_err:.3f} "
                  "(0 = perfectly linear in the layer phase)." if ctrl else ""]
    md = "\n".join(lines) + "\n"
    open(os.path.join(OUT, f"separation_{variant}.md"), "w").write(md)
    json.dump(report, open(os.path.join(OUT, f"separation_{variant}.json"), "w"), indent=1, default=float)
    print(md)

    # figure: measured response, superposition prediction and fits, per separation, plus the controls (first seed)
    seed = seeds[0]
    tests = sorted([rr for (sn, sd), rr in runs.items() if sd == seed and rr["sep"] > 0], key=lambda rr: rr["sep"])
    single = runs[(f"sep0x1_s{step}", seed)]; ctrl = runs.get((f"sep0_s{step}", seed))
    z, s1 = single["z"], single["s"]
    fig, ax = plt.subplots(1, len(tests) + 1, figsize=(3.6 * (len(tests) + 1), 3.3), squeeze=False)
    a = ax[0, 0]
    a.plot(z, s1, color="#444", label="single layer (×1)")
    if ctrl:
        a.plot(z, ctrl["s"], color="#c00", label="coincident control (×2)")
        a.plot(z, 2 * s1, ":", color="#c00", label="2 × single layer")
    a.axvline(150, color="k", ls="--", lw=0.7); a.set_title("controls at z = 150 Å", fontsize=9)
    a.set_ylabel("projection of section phase on pattern"); a.legend(fontsize=6)
    for a, rr in zip(ax[0, 1:], tests):
        dz = rr["sep"]; row = next(r for r in report if r["dz"] == dz and r["seed"] == seed)
        pred = shifted(z, s1, -dz / 2) + shifted(z, s1, dz / 2)
        a.plot(z, rr["s"], color="#1c4587", lw=1.6, label="measured")
        a.plot(z, pred, "--", color="#e69138", label="superposition of single-layer responses")
        for t in rr["layer_z"]:
            a.axvline(t, color="k", ls="--", lw=0.7)
        for zf in row["fit"]["z_fit"]:
            a.axvline(zf, color="#6aa84f", ls=":", lw=1.0)
        a.set_title(f"Δz = {dz:g} Å: {row['outcome']}\nfit Δz = {row['fit']['dz_fit']:.1f} Å, ΔBIC {row['fit']['dBIC']:.0f}", fontsize=8)
        a.legend(fontsize=6)
    for a in ax[0]:
        a.set_xlim(40, 260); a.set_xlabel("section depth z (Å)"); a.grid(alpha=0.3)
    fig.suptitle(f"Two identical layers, {variant} probe, {0.2 * step:.1f} Å scan step, seed {seed} "
                 "(dashed black = true depths, dotted green = two-layer fit)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", f"separation_{variant}.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--seps", default="0,0x1,20,40,60", help="sample tokens: 0 (x2 control), 0x1 (single layer), dz in A")
    ap.add_argument("--variants", default="oracle")
    ap.add_argument("--variant", default="oracle", help="analyze: which variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.seps.split(","), a.variants.split(","), a.seed, a.device, a.step)
    else:
        analyze(a.variant, a.step)


if __name__ == "__main__":
    main()
