"""
How to use quantEM's iterative ptychography API, end to end, on CPU.

    python examples/ptycholite_example.py

1. Simulate a small 4D-STEM dataset with abTEM (MoS2 monolayer with sulfur vacancies), so no
   downloads are needed, and keep the true projected phase for comparison.
2. Wrap it in a quantEM Dataset4dstem (detector sampling must be in A^-1).
3. Reconstruct three ways:
     A. PtychoLite                     - one-call constructor, simple learning-rate arguments
     B. PtychoLiteDIP                  - same, with U-Net (deep image prior) object and probe
     C. Ptychography.from_models(...)  - the full modular API: explicit object / probe / detector
                                         models, optimizer and scheduler objects, constraints,
                                         Poisson loss
4. Save, reload and continue.
5. Plot everything against the ground truth.

Figures go to examples/figures/. Runtime: a few minutes on a laptop CPU.
"""

import time
from pathlib import Path

import abtem
import ase
import ase.build
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch

import quantem as em
from quantem.core.ml import OptimizerParams, SchedulerParams
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    PtychoLite,
    PtychoLiteDIP,
    Ptychography,
    PtychographyDatasetRaster,
)

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
SAVE = HERE / "outputs"
SAVE.mkdir(exist_ok=True)
torch.manual_seed(0)
rng = np.random.default_rng(0)

# ============================================================== 1. simulate a dataset with abTEM
ENERGY = 80e3          # eV
SEMIANGLE = 20.0       # mrad
DEFOCUS = 100.0        # A  (abTEM convention: positive = underfocus, C10 = -defocus)
STEP = 0.5             # A, scan step
MAX_ANGLE = 60.0       # mrad, detector radius kept
DOSE = 1e5             # e/A^2

atoms = ase.build.mx2("MoS2", vacuum=2.0)
atoms = abtem.orthogonalize_cell(atoms) * (8, 5, 1)          # ~25 x 27 A, periodic
s_sites = np.where(atoms.numbers == 16)[0]
del atoms[rng.choice(s_sites, size=12, replace=False)]        # a few S vacancies to look for

potential = abtem.Potential(atoms, sampling=0.05, slice_thickness=1.0)
probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=SEMIANGLE, defocus=DEFOCUS)
probe.grid.match(potential)
scan = abtem.GridScan(start=(0, 0), end=atoms.cell.lengths()[:2], sampling=STEP, endpoint=False)
detector = abtem.PixelatedDetector(max_angle=MAX_ANGLE)

cache = SAVE / "mos2_simulation.npz"                          # simulation takes ~3-4 min on CPU
if cache.exists():
    z = np.load(cache)
    intens, gt_phase, angular_sampling = z["intens"], z["gt_phase"], float(z["angular_sampling"])
else:
    t0 = time.time()
    measurement = probe.scan(potential=potential, scan=scan, detectors=detector).compute()
    intens = np.asarray(measurement.array, dtype=np.float32)      # (scan_x, scan_y, kx, ky), centred
    angular_sampling = measurement.angular_sampling[0]            # mrad / pixel
    print(f"simulated {intens.shape} in {time.time()-t0:.0f} s")
    # ground truth: projected phase  phi = sigma * integral V dz
    gt_phase = np.asarray(potential.build().project().array) * abtem.core.energy.energy2sigma(ENERGY)
    np.savez(cache, intens=intens, gt_phase=gt_phase, angular_sampling=angular_sampling)

# The DIP U-Net (3 down-sampling levels) needs detector sizes divisible by 2**3 = 8,
# so crop the patterns to a centred 72 x 72 (still +-56 mrad).
N_DET = 72
cr, cc = intens.shape[-2] // 2, intens.shape[-1] // 2
intens = intens[..., cr - N_DET // 2:cr + N_DET // 2, cc - N_DET // 2:cc + N_DET // 2]

# Poisson noise: electrons per pattern = dose * step^2
noisy = rng.poisson(intens / intens.sum((-1, -2), keepdims=True) * DOSE * STEP**2).astype(np.float32)

# ============================================================== 2. quantEM dataset (A^-1 on the detector)
wavelength = abtem.core.energy.energy2wavelength(ENERGY)
dk = angular_sampling * 1e-3 / wavelength                         # mrad -> A^-1
dset = em.core.datastructures.Dataset4dstem.from_array(
    noisy, sampling=[STEP, STEP, dk, dk], units=["A", "A", "A^-1", "A^-1"],
)
print(dset)

# ============================================================== 3A. PtychoLite (simplest)
pdset = PtychographyDatasetRaster.from_dataset4dstem(dset, verbose=False)
pdset.preprocess(com_fit_function="constant", plot_rotation=False, plot_com=False)   # COM, rotation, normalise

lite = PtychoLite.from_dataset(
    dset=pdset,
    obj_type="pure_phase",             # O = exp(i*phi): only the phase is optimised
    num_slices=1,
    num_probes=1,
    energy=ENERGY,
    defocus=DEFOCUS,                   # initial probe guess, refined during reconstruction
    semiangle_cutoff=SEMIANGLE,
    obj_padding_px=(8, 8),
    device="cpu",
    verbose=False,
    rng=0,
)
t0 = time.time()
lite.reconstruct(num_iters=40, reset=True, lr_obj=5e-2, lr_probe=5e-2,
                 batch_size=256, scheduler_type="plateau", verbose=False)
print(f"PtychoLite: 40 iters in {time.time()-t0:.0f} s, loss {lite.iter_losses[-1]:.4g}")

# ============================================================== 3B. PtychoLiteDIP (U-Net prior)
# from_ptychography copies the setup (dataset, probe settings) but RESETS the object to uniform,
# so run a few pixelated iterations on the copy before handing it to the DIP.
start = PtychoLite.from_ptychography(ptycho=lite)
start.reconstruct(num_iters=10, lr_obj=5e-2, lr_probe=5e-2, batch_size=256, verbose=False)
# Builds CNN2d U-Nets for object and probe and pretrains them to reproduce the pixelated result.
dip = PtychoLiteDIP.from_ptycholite(ptycholite=start, pretrain_iters=100, device="cpu", verbose=False)
t0 = time.time()
dip.reconstruct(num_iters=20, reset=True, lr_obj=5e-4, lr_probe=5e-4,   # ~100x smaller lr for network weights
                batch_size=256, scheduler_type="plateau", verbose=False)
print(f"PtychoLiteDIP: 20 iters in {time.time()-t0:.0f} s, loss {dip.iter_losses[-1]:.4g}")

# ============================================================== 3C. full modular API
obj_model = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1, rng=0)
probe_model = ProbePixelated.from_params(
    probe_params={"energy": ENERGY, "defocus": DEFOCUS, "semiangle_cutoff": SEMIANGLE})
full = Ptychography.from_models(
    dset=pdset, obj_model=obj_model, probe_model=probe_model,
    detector_model=DetectorPixelated(), device="cpu", verbose=False, rng=0,
)
full.preprocess(obj_padding_px=(8, 8), plot_rotation=False, plot_com=False)
t0 = time.time()
full.reconstruct(
    num_iters=40,
    reset=True,
    optimizer_params={"object": OptimizerParams.Adam(lr=5e-2), "probe": OptimizerParams.Adam(lr=5e-2)},
    scheduler_params={"object": SchedulerParams.Plateau(factor=0.5), "probe": SchedulerParams.Plateau(factor=0.5)},
    constraints={"object": {"tv_weight_xy": 1e-3}, "probe": {"center_probe": True}},
    loss_type="poisson",               # counting-noise likelihood instead of the default l2 on amplitudes
    batch_size=256,
)
print(f"Ptychography.from_models: 40 iters in {time.time()-t0:.0f} s, loss {full.iter_losses[-1]:.4g}")

# ============================================================== 4. save / load / continue
path = SAVE / "mos2_ptycholite.zip"
lite.save(path, mode="o")
reloaded = PtychoLite.from_file(path, dset=pdset)      # the dataset is passed back in separately
reloaded.reconstruct(num_iters=5, verbose=False)
print(f"reloaded + 5 more iters, loss {reloaded.iter_losses[-1]:.4g}")


# ============================================================== 5. figures
def phase_of(p):
    o = np.asarray(p.obj_cropped)
    o = o[0] if o.ndim == 3 else o
    return np.angle(o) if np.iscomplexobj(o) else o


def aligned_r(rec, ref):
    """Correlation with ground truth after resampling to the same grid and best periodic shift."""
    ref = ndi.zoom(ref, (rec.shape[0] / ref.shape[0], rec.shape[1] / ref.shape[1]), order=1)
    a = (rec - rec.mean()) / rec.std(); b = (ref - ref.mean()) / ref.std()
    cc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    s = np.unravel_index(np.argmax(cc), cc.shape)
    return float((np.roll(a, (-s[0], -s[1]), (0, 1)) * b).mean())


recs = {"PtychoLite (40 it)": lite, "PtychoLiteDIP (10 pix + 20 DIP it)": dip, "from_models, Poisson + TV (40 it)": full}
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
axs[0].imshow(ndi.gaussian_filter(gt_phase, 4).T, cmap="magma")       # 0.2 A blur so atoms are visible
axs[0].set_title("ground truth projected phase")
for ax, (name, p) in zip(axs[1:], recs.items()):
    ph = phase_of(p)
    ax.imshow(ph.T, cmap="magma")
    ax.set_title(f"{name}\nr = {aligned_r(ph, gt_phase):.2f}")
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"MoS₂ with S vacancies — {DOSE:.0e} e⁻/Å², {STEP} Å step, 80 keV / 20 mrad / {DEFOCUS:.0f} Å defocus",
             fontweight="bold")
fig.savefig(FIG / "reconstructions.png", dpi=130, bbox_inches="tight")

fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for name, p in list(recs.items())[:2]:
    L = np.asarray(p.iter_losses, float)
    axs[0].plot(np.arange(1, len(L) + 1), L, lw=2, label=name)
axs[0].set_title("L2 amplitude loss (PtychoLite default)"); axs[0].legend(frameon=False)
L = np.asarray(full.iter_losses, float)
axs[1].plot(np.arange(1, len(L) + 1), L, lw=2, color="C2")
axs[1].set_title("Poisson negative log-likelihood (from_models; lower = better, can be < 0)")
for ax in axs:
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
fig.savefig(FIG / "loss_curves.png", dpi=130, bbox_inches="tight")

fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
pr = lite.probe[0]                                       # numpy, shape (num_probes, H, W)
pr = np.fft.fftshift(pr) if np.abs(pr[0, 0]) > np.abs(pr[pr.shape[0] // 2, pr.shape[1] // 2]) else pr
axs[0].imshow(np.abs(pr) ** 2, cmap="magma"); axs[0].set_title("refined probe |P(r)|²")
axs[1].imshow(np.sqrt(noisy.mean((0, 1))), cmap="magma"); axs[1].set_title("mean diffraction pattern (√I)")
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(FIG / "probe_and_pattern.png", dpi=130, bbox_inches="tight")
print("figures saved to", FIG)
