"""Reconstruction and transfer function before vs after hyperparameter fitting.

After a fit, DirectPtychography.reconstruct() uses the optimized aberrations / rotation
(HyperparameterState.current_aberrations prefers them), so the same call gives a new result.

    python visualizations/after_tuning.py      # run from repo root, ~5 min on CPU
"""

import numpy as np
import matplotlib.pyplot as plt
import optuna

# Reuses the simulation block of white_noise_steps.py (data4d, potential, aberrations, ...).
exec(open("visualizations/white_noise_steps.py").read().split("# ================================================================ step 1")[0])
from quantem.diffractive_imaging import DirectPtychography
from quantem.diffractive_imaging.direct_ptychography import OptimizationParameter

optuna.logging.set_verbosity(optuna.logging.WARNING)
ds = em.core.datastructures.Dataset4dstem.from_array(
    data4d, sampling=[sampling] * 2 + [reciprocal_sampling] * 2, units=["A", "A", "A^-1", "A^-1"])
make = lambda ab, rot: DirectPtychography.from_dataset4d(
    ds, energy=energy, semiangle_cutoff=k_probe * wavelength * 1e3,
    rotation_angle=rot, aberration_coefs=ab, verbose=False)
zero = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 else float("nan")


def fmt_r(r):
    return "n/a: image is all zero" if np.isnan(r) else f"{r:.2f}"


def describe(dp):
    st = dp.hyperparameter_state
    ab = st.current_aberrations()
    rot = st.current_rotation_angle()
    return (f"C10={ab.get('C10', 0):.0f} Å, C12={ab.get('C12', 0):.0f} Å, "
            f"φ12={np.rad2deg(ab.get('phi12', 0)):.1f}°, rot={(np.rad2deg(rot) + 180) % 360 - 180:.1f}°")


rows = []

d = make(aberrations, rotation_angle_rad)
rows.append(("Ground-truth parameters", d))

d = make(aberrations, -13.0)
rows.append(("Notebook as written (rot = −13 rad)", d))

d = make(zero, rotation_angle_rad)
rows.append(("Before fitting: aberrations = 0", d))

d = make(aberrations, rotation_angle_rad).optimize_hyperparameters(
    aberration_coefs={"C10": OptimizationParameter(0, 500), "C12": OptimizationParameter(0, 500),
                      "phi12": OptimizationParameter(-0.5, 0.5)},
    rotation_angle=OptimizationParameter(-0.5, 0.5),
    n_trials=200, deconvolution_kernel="parallax", sampler=optuna.samplers.TPESampler(seed=0))
rows.append(("After Optuna (200 trials)", d))

d = make(aberrations, rotation_angle_rad).fit_hyperparameters_cross_correlation(
    alignment_method="reference", bin_factors=(1, 1, 1), aberration_coefs=zero, rotation_angle=0)
rows.append(("After cross-correlation fit (from 0)", d))

d = make(aberrations, rotation_angle_rad).fit_hyperparameters_least_squares(
    aberration_coefs=zero, cartesian_basis="low_order", q_signal_weight=0.0, num_q_modes=12)
rows.append(("After least-squares fit (from 0)", d))

ext = [-k_max, k_max, k_max, -k_max]
fig, axs = plt.subplots(len(rows), 4, figsize=(13, 3.25 * len(rows)),
                        gridspec_kw={"wspace": 0.08, "hspace": 0.45})
for i, (label, dp) in enumerate(rows):
    ssb = dp.reconstruct(deconvolution_kernel="ssb", verbose=False).obj
    obf = dp.reconstruct(deconvolution_kernel="obf", verbose=False).obj
    prl = dp.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
    T = lambda r: np.fft.fftshift(np.abs(np.fft.fft2(r - r.mean())))
    panels = [
        (ssb, f"SSB recon  r = {fmt_r(corr(ssb, potential))}", "gray", None),
        (T(ssb), "|T| SSB", "magma", ext),
        (T(obf), f"|T| OBF  (recon r = {fmt_r(corr(obf, potential))})", "magma", ext),
        (T(prl), f"|T| parallax  (r = {fmt_r(corr(prl, potential))})", "magma", ext),
    ]
    for ax, (img, title, cmap, extent) in zip(axs[i], panels):
        kw = dict(vmin=0, vmax=1) if cmap == "magma" else {}
        ax.imshow(img, cmap=cmap, extent=extent, **kw)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axs[i, 0].text(-0.06, 0.5, label, transform=axs[i, 0].transAxes, rotation=90,
                   ha="right", va="center", fontsize=10, fontweight="bold")
    axs[i, 0].text(0.0, -0.1, describe(dp), transform=axs[i, 0].transAxes, fontsize=8.5,
                   color="#52514e", va="top")
    print(f"{label:40s} {describe(dp):60s} r_ssb={corr(ssb, potential):.3f}")

fig.suptitle("Reconstruction and transfer function |T(q)| before vs after hyperparameter fitting\n"
             "truth: C10=200 Å, C12=100 Å, φ12=11°, rot=−13°", fontsize=12.5, fontweight="bold", y=0.925)
fig.savefig("visualizations/white_noise_steps/after_tuning.png", dpi=130, bbox_inches="tight",
            facecolor="#fcfcfb")
print("saved visualizations/white_noise_steps/after_tuning.png")
