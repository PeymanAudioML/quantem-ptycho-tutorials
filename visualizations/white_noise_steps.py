"""
Step-by-step figures for the white-noise-object direct-ptychography tutorial.

Re-runs simulation_notebooks/white-noise-object.ipynb (numpy forward model) and
white_noise_object_phase_retrieval.ipynb (quantem DirectPtychography), saving one
PNG per step to visualizations/white_noise_steps/.

    python visualizations/white_noise_steps.py            # runs on CPU in ~1-2 min
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.patches import Circle
from scipy.ndimage import rotate

import quantem as em
from quantem.diffractive_imaging import DirectPtychography
from quantem.diffractive_imaging.direct_ptychography import OptimizationParameter

OUT = Path(__file__).parent / "white_noise_steps"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "font.size": 9.5,
        "lines.linewidth": 2,
    }
)
PHASE_CMAP = "gray"  # object phase: standard microscopy grayscale
MAG_CMAP = "magma"  # intensities / magnitudes: perceptually uniform, dark -> light
CYC_CMAP = "twilight"  # wrapped phases (cyclic quantity)


def imshow(ax, img, title, cmap=PHASE_CMAP, **kw):
    im = ax.imshow(img, cmap=cmap, **kw)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def cbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    if label:
        cb.set_label(label, color=INK_2, fontsize=8.5)


def caption(fig, text, y=0.01):
    fig.text(0.5, y, text, ha="center", va="bottom", color=INK_2, fontsize=9, wrap=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


# ================================================================ simulation (same as notebook)
np.random.seed(0)
n = 96
k_max, k_probe = 2, 1  # 1/A
energy = 300e3
wavelength = em.core.utils.utils.electron_wavelength_angstrom(energy)
sampling = 1 / k_max / 2  # A
reciprocal_sampling = 2 * k_max / n  # 1/A
phi0 = 1
aberrations = {"C10": 200, "C12": 100, "phi12": np.deg2rad(11)}
rotation_angle_deg = -13
# quantem expects radians (the tutorial notebook passes -13, which quantem reads as -13 rad)
rotation_angle_rad = np.deg2rad(rotation_angle_deg)


def white_noise_object_2D(n, phi0):
    evenQ = n % 2 == 0
    pos_ind = np.arange(1, (n if evenQ else n + 1) // 2)
    neg_ind = np.flip(np.arange(n // 2 + 1, n))
    arr = np.random.randn(n, n)
    arr[pos_ind[:, None], pos_ind[None, :]] = -arr[neg_ind[:, None], neg_ind[None, :]]
    arr[pos_ind[:, None], neg_ind[None, :]] = -arr[neg_ind[:, None], pos_ind[None, :]]
    arr[0, pos_ind] = -arr[0, neg_ind]
    arr[pos_ind, 0] = -arr[neg_ind, 0]
    if evenQ:
        arr[n // 2, :] = 0
        arr[:, n // 2] = 0
    arr[0, 0] = 0
    arr = np.exp(2j * np.pi * arr) * phi0
    return np.fft.ifft2(arr).real


potential = white_noise_object_2D(n, phi0)
complex_obj = np.exp(1j * potential)

kx = ky = np.fft.fftfreq(n, sampling)
k2 = kx[:, None] ** 2 + ky[None, :] ** 2
k = np.sqrt(k2)
phi = np.arctan2(ky[None, :], kx[:, None])
aperture = np.sqrt(np.clip((k_probe - k) / reciprocal_sampling + 0.5, 0, 1))


def prepare_probe(ab):
    chi = k2 * wavelength * np.pi * ab.get("C10", 0.0)
    chi += k2 * wavelength * np.pi * ab.get("C12", 0.0) * np.cos(2 * (phi - ab.get("phi12", 0.0)))
    pf = aperture * np.exp(-1j * chi)
    pf /= np.sqrt(np.sum(np.abs(pf) ** 2))
    return chi, pf, np.fft.ifft2(pf) * n


chi, probe_fourier, probe = prepare_probe(aberrations)
_, _, probe_ideal = prepare_probe({})

# raster scan, 1 px step; probe is stored centred at the array origin
xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
positions = np.stack((xx.ravel(), yy.ravel()), -1)
idx = np.fft.fftfreq(n, d=1 / n).astype(int)
row = (positions[:, 0, None, None] + idx[None, :, None]) % n
col = (positions[:, 1, None, None] + idx[None, None, :]) % n
obj_patches = complex_obj[row, col]
exit_waves = obj_patches * probe
intensities = np.abs(np.fft.fft2(exit_waves)) ** 2

rotated = np.fft.ifftshift(
    rotate(np.fft.fftshift(intensities, axes=(-1, -2)), -rotation_angle_deg,
           axes=(-1, -2), reshape=False, order=3),
    axes=(-1, -2),
)
data4d = np.fft.fftshift(rotated.reshape((n, n, n, n)), axes=(-1, -2))
data4d_unrot = np.fft.fftshift(intensities.reshape((n, n, n, n)), axes=(-1, -2))

sh = np.fft.fftshift
ext_k = [-k_max, k_max, k_max, -k_max]  # detector extent in 1/A (rows down)
ext_r = [0, n * sampling, n * sampling, 0]  # real-space extent in A

# ================================================================ step 1: object
F = np.fft.fft2(potential)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4), gridspec_kw={"wspace": 0.45})
im = imshow(axs[0], potential, "Object phase  φ(r)", extent=ext_r)
cbar(fig, im, axs[0], "rad")
axs[0].set_xticks([0, 12, 24]); axs[0].set_xlabel("Å")
im = imshow(axs[1], sh(np.abs(F)), "|FFT(φ)|  — flat: every frequency equal",
            cmap=MAG_CMAP, extent=ext_k, vmin=0)
cbar(fig, im, axs[1])
im = imshow(axs[2], sh(np.angle(F)), "arg FFT(φ)  — random phases", cmap=CYC_CMAP, extent=ext_k)
cbar(fig, im, axs[2], "rad")
fig.suptitle("Step 1 · White-noise test object", y=1.02)
caption(fig, "Constant |FFT| means any loss of contrast at a frequency in a reconstruction "
             "is the method's fault — this is why the object is 'white'.", y=-0.06)
save(fig, "step1_object.png")

# ================================================================ step 2: probe
mask = aperture > 0
chi_wrapped = np.where(mask, np.angle(np.exp(-1j * chi)), np.nan)
fig, axs = plt.subplots(1, 4, figsize=(15.5, 4))
im = imshow(axs[0], sh(aperture), "Aperture A(k)  (k < 1 Å⁻¹)", cmap=MAG_CMAP, extent=ext_k)
cbar(fig, im, axs[0])
im = imshow(axs[1], sh(chi_wrapped), "Aberration phase −χ(k)\nC10=200 Å defocus + C12=100 Å stig",
            cmap=CYC_CMAP, extent=ext_k)
cbar(fig, im, axs[1], "rad (wrapped)")
im = imshow(axs[2], sh(np.abs(probe_ideal)), "|probe(r)|  no aberrations", cmap=MAG_CMAP, extent=ext_r)
cbar(fig, im, axs[2])
im = imshow(axs[3], sh(np.abs(probe)), "|probe(r)|  with aberrations", cmap=MAG_CMAP, extent=ext_r)
cbar(fig, im, axs[3])
fig.suptitle("Step 2 · Electron probe = IFFT[ A(k) · exp(−iχ(k)) ]", y=1.03)
caption(fig, "Defocus spreads the probe over several Å; astigmatism stretches it into a line. "
             "The reconstruction must undo this blur.", y=-0.06)
save(fig, "step2_probe.png")

# ================================================================ step 3: forward model
p = 48 * n + 48  # centre scan position
mean_I = intensities.mean(0)
fig, axs = plt.subplots(1, 5, figsize=(19.5, 4))
im = imshow(axs[0], sh(np.angle(obj_patches[p])), "Object phase  arg O(r − rₚ)\n(shifted to scan position)", extent=ext_r)
cbar(fig, im, axs[0], "rad")
im = imshow(axs[1], sh(np.abs(exit_waves[p])), "|exit wave|  = |O · P|", cmap=MAG_CMAP, extent=ext_r)
cbar(fig, im, axs[1])
ew_phase = np.where(np.abs(exit_waves[p]) > 0.05 * np.abs(exit_waves[p]).max(),
                    np.angle(exit_waves[p]), np.nan)
im = imshow(axs[2], sh(ew_phase), "arg exit wave  (lost by the camera)", cmap=CYC_CMAP, extent=ext_r)
cbar(fig, im, axs[2], "rad")
im = imshow(axs[3], sh(intensities[p]) ** 0.5, "Diffraction  I(k) = |FFT(O·P)|²\n(shown as √I)",
            cmap=MAG_CMAP, extent=ext_k)
cbar(fig, im, axs[3])
dI = sh(intensities[p] - mean_I)
v = np.abs(dI).max()
im = imshow(axs[4], dI, "I(k) − mean I(k)\nthe object signal", cmap="RdBu_r", extent=ext_k, vmin=-v, vmax=v)
cbar(fig, im, axs[4])
fig.suptitle("Step 3 · Forward model at one scan position", y=1.03)
caption(fig, "The camera records only intensity. A weak object barely changes I(k), so the information "
             "is the small speckle left after subtracting the mean (right panel).", y=-0.06)
save(fig, "step3_forward_model.png")

# step 3b: overlap of scan positions
fig, ax = plt.subplots(figsize=(5.2, 5.2))
imshow(ax, potential, "", extent=ext_r)
probe_r = 0.5 * (k_probe * wavelength * aberrations["C10"]) * 2  # defocus disk radius (A)
for i, (r0, c0) in enumerate([(30, 30), (30, 42), (42, 30), (42, 42)]):
    ax.add_patch(Circle((c0 * sampling, r0 * sampling), probe_r, fill=False,
                        ec=SERIES[i], lw=2))
    ax.plot(c0 * sampling, r0 * sampling, "o", ms=5, color=SERIES[i], mec=SURFACE, mew=1.5)
gx = np.arange(0, n, 6) * sampling
ax.plot(*np.meshgrid(gx, gx), ".", ms=1.5, color="#c3c2b7")
ax.set_title("Step 3b · Scan: 96×96 positions, 0.25 Å step\n"
             "4 example probe footprints (every 12th position)")
save(fig, "step3b_scan_overlap.png")

# ================================================================ step 4: rotation + dataset
vbf_mask = (np.sqrt((np.arange(n)[:, None] - n / 2) ** 2 + (np.arange(n)[None, :] - n / 2) ** 2)
            < k_probe / reciprocal_sampling)
vbf = (data4d * vbf_mask).sum((-1, -2))
vdf = (data4d * ~vbf_mask).sum((-1, -2))
fig, axs = plt.subplots(1, 4, figsize=(15.5, 4))
for ax, arr, t in [(axs[0], data4d_unrot, "I − mean, before rotation"),
                  (axs[1], data4d, "I − mean, after −13° rotation")]:
    dI = arr[48, 48] - arr.mean((0, 1))
    v = np.abs(dI).max()
    imshow(ax, dI, t, cmap="RdBu_r", extent=ext_k, vmin=-v, vmax=v)
for ax in axs[:2]:
    ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color="#c3c2b7", lw=0.8)
imshow(axs[2], data4d.mean((0, 1)).clip(0) ** 0.5, "Mean pattern  (bright-field disk)", cmap=MAG_CMAP, extent=ext_k)
imshow(axs[3], vbf, "Virtual bright-field image\n(sum inside disk) — blurry", extent=ext_r)
fig.suptitle("Step 4 · Detector rotation and the saved 4D dataset  (96, 96, 96, 96)", y=1.03)
caption(fig, "Real cameras are rotated relative to the scan; −13° is baked in so the "
             "reconstruction has to know (or find) it.", y=-0.06)
save(fig, "step4_rotation_dataset.png")

# ================================================================ step 5: quantem direct ptychography
dataset = em.core.datastructures.Dataset4dstem.from_array(
    data4d,
    sampling=[sampling, sampling, reciprocal_sampling, reciprocal_sampling],
    units=["A", "A", "A^-1", "A^-1"],
)
semiangle_cutoff = k_probe * wavelength * 1e3
make = lambda **kw: DirectPtychography.from_dataset4d(
    dataset, energy=energy, semiangle_cutoff=semiangle_cutoff, verbose=False, **kw
)
dp = make(rotation_angle=rotation_angle_rad, aberration_coefs=aberrations)

names = ["single-sideband", "optimum bright-field", "matched filter", "parallax", "iCOM"]
recons = dp._reconstruct_all_permutations(verbose=False)


def corr(a, b):
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    return float((a * b).mean())


fig, axs = plt.subplots(1, 6, figsize=(18, 3.6))
imshow(axs[0], potential, "Ground truth")
for ax, name, r in zip(axs[1:], names, recons):
    imshow(ax, r, f"{name}\nr = {corr(r, potential):.2f}")
fig.suptitle("Step 5 · Direct ptychography with the true aberrations — five deconvolution kernels", y=1.06)
caption(fig, "r = Pearson correlation with ground truth. All are band-limited (blurrier than truth); "
             "iCOM recovers only the lowest frequencies, so it barely correlates with a white object.", y=-0.08)
save(fig, "step5_reconstructions.png")

# ================================================================ step 6: transfer functions
ffts = [sh(np.abs(np.fft.fft2(r - r.mean()))) for r in recons]
fig = plt.figure(figsize=(18, 7.6))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.35)
for i, (name, f) in enumerate(zip(names, ffts)):
    ax = fig.add_subplot(gs[0, i])
    imshow(ax, f / f.max(), name, cmap=MAG_CMAP, extent=ext_k, vmin=0, vmax=1)

kr = np.sqrt(sh(k2)).ravel()
bins = np.arange(0, k_max + 1e-9, reciprocal_sampling)
which = np.digitize(kr, bins)
ax = fig.add_subplot(gs[1, 1:4])
for i, (name, f) in enumerate(zip(names, ffts)):
    prof = np.array([f.ravel()[which == b].mean() for b in range(1, len(bins))])
    prof /= prof.max()
    centres = bins[:-1] + reciprocal_sampling / 2
    ax.plot(centres, prof, color=SERIES[i], label=name)
ax.axvline(k_probe, color=INK_2, lw=1, ls="--")
ax.axvline(2 * k_probe, color=INK_2, lw=1, ls="--")
ax.text(k_probe, 1.07, " aperture k", color=INK_2, fontsize=8)
ax.text(2 * k_probe, 1.07, " 2k = info limit", color=INK_2, fontsize=8, ha="right")
ax.set_xlim(0, k_max); ax.set_ylim(0, 1.12)
ax.set_xlabel("spatial frequency |k|  (Å⁻¹)"); ax.set_ylabel("radial mean |FFT|  (normalised)")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.32, 1))
fig.suptitle("Step 6 · Transfer functions — |FFT(reconstruction)| of a white object", y=0.98)
save(fig, "step6_transfer_functions.png")

# ================================================================ step 7: why calibration matters
cases = [
    ("true aberrations + rotation", {}),
    ("defocus ignored (C10 = 0)", {"override_aberration_coefs": {**aberrations, "C10": 0}}),
    ("stig ignored (C12 = 0)", {"override_aberration_coefs": {**aberrations, "C12": 0}}),
    ("rotation ignored (0°)", {"override_rotation_angle": 0}),
]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
for ax, (title, kw) in zip(axs, cases):
    r = dp.reconstruct(deconvolution_kernel="ssb", verbose=False, **kw).obj
    imshow(ax, r, f"{title}\nr = {corr(r, potential):.2f}")
fig.suptitle("Step 7 · Single-sideband reconstruction with wrong microscope parameters", y=1.06)
caption(fig, "Wrong aberrations or rotation scramble the deconvolution — so they must be measured.",
        y=-0.05)
save(fig, "step7_wrong_parameters.png")

# ================================================================ step 8: parallax shifts
bf_idx = np.argwhere(vbf_mask)[::7]
ref = data4d[:, :, n // 2, n // 2]
Fref = np.conj(np.fft.fft2(ref - ref.mean()))
shifts = []
for i, j in bf_idx:
    img = data4d[:, :, i, j]
    cc = np.fft.ifft2(np.fft.fft2(img - img.mean()) * Fref).real
    s = np.array(np.unravel_index(np.argmax(cc), cc.shape), float)
    s[s > n / 2] -= n
    shifts.append(s * sampling)
shifts = np.array(shifts)
qy = (bf_idx[:, 0] - n / 2) * reciprocal_sampling
qx = (bf_idx[:, 1] - n / 2) * reciprocal_sampling

fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for ax, (i, j), lab in zip(axs[:2], [(n // 2, n // 2 - 18), (n // 2, n // 2 + 18)], ["left", "right"]):
    imshow(ax, data4d[:, :, i, j][24:72, 24:72], f"Image from one detector pixel ({lab} of disk)")
ax = axs[2]
ax.add_patch(Circle((0, 0), k_probe, fill=False, ec=GRID, lw=1.5))
ax.quiver(qx, qy, shifts[:, 1], -shifts[:, 0], color=SERIES[0], angles="xy", scale_units="xy",
          scale=30, width=0.005)
ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(1.15, -1.15)
ax.set_xlabel("detector kx (Å⁻¹)"); ax.set_ylabel("detector ky (Å⁻¹)")
ax.set_title("Measured image shift per detector pixel\n(arrows ∝ ∇χ: defocus → radial, stig → skewed)")
for s in ax.spines.values():
    s.set_visible(False)
fig.suptitle("Step 8 · Parallax: each bright-field pixel sees a shifted image — "
             "the shifts encode the aberrations", y=1.04)
caption(fig, "Cross-correlation and least-squares fits read C10, C12, φ12 and rotation off this "
             "vector field — no initial guess needed.", y=-0.05)
save(fig, "step8_parallax_shifts.png")

# ================================================================ step 9: hyperparameter recovery
results = {}
d = make(rotation_angle=rotation_angle_rad, aberration_coefs=aberrations)
d = d.optimize_hyperparameters(
    aberration_coefs={"C10": OptimizationParameter(0, 500), "C12": OptimizationParameter(0, 500),
                      "phi12": OptimizationParameter(-0.5, 0.5)},
    rotation_angle=OptimizationParameter(-0.5, 0.5),  # radians
    n_trials=200, deconvolution_kernel="parallax", sampler=optuna.samplers.TPESampler(seed=0),
)
results["Optuna (200 trials, ranges)"] = d.hyperparameter_state

d = make(rotation_angle=rotation_angle_rad, aberration_coefs=aberrations)
d = d.fit_hyperparameters_cross_correlation(
    alignment_method="reference", bin_factors=(1, 1, 1),
    aberration_coefs={"C10": 0, "C12": 0, "phi12": 0}, rotation_angle=0,
)
results["Cross-correlation (from 0)"] = d.hyperparameter_state

d = make(rotation_angle=rotation_angle_rad, aberration_coefs=aberrations)
d = d.fit_hyperparameters_least_squares(
    aberration_coefs={"C10": 0, "C12": 0, "phi12": 0}, cartesian_basis="low_order",
    q_signal_weight=0.0, num_q_modes=12,
)
results["Least squares (from 0)"] = d.hyperparameter_state

params = [("C10", "defocus C10 (Å)", aberrations["C10"]),
          ("C12", "astigmatism C12 (Å)", aberrations["C12"]),
          ("phi12", "stig angle φ12 (°)", np.rad2deg(aberrations["phi12"])),
          ("rot", "rotation (°)", rotation_angle_deg)]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
methods = list(results)
for ax, (key, label, truth) in zip(axs, params):
    vals = []
    for m_i, m in enumerate(methods):
        st = results[m]
        if key == "rot":
            val = st.optimized_rotation_angle
            if val is not None:
                val = np.rad2deg(val)
        else:
            val = st.optimized_aberrations.get(key)
            if key == "phi12" and val is not None:
                val = np.rad2deg(val)
        if val is None:
            ax.text(truth, m_i, "  not fitted", va="center", color=INK_2, fontsize=8.5)
            continue
        ax.plot(val, m_i, "o", ms=10, color=SERIES[m_i], mec=SURFACE, mew=2)
        ax.annotate(f"{val:.1f}", (val, m_i), xytext=(0, -13), textcoords="offset points",
                    ha="center", color=INK_2, fontsize=8.5)
        vals.append(val)
    ax.axvline(truth, color=INK, lw=1.2, ls="--")
    vals.append(truth)
    ax.set_title(f"{label}\ntrue = {truth:g}")
    lo, hi = min(vals), max(vals)
    pad = max(0.15 * (hi - lo), 2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(len(methods) - 0.5, -0.7)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(range(len(methods)), methods)
fig.suptitle("Step 9 · Recovering the microscope parameters from the data (dashed = truth)", y=1.08)
save(fig, "step9_hyperparameter_fit.png")

for m, st in results.items():
    print(m, st.optimized_aberrations, st.optimized_rotation_angle)
