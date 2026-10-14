"""M5 dry run (CPU): abTEM data of 3 Au atoms at 50 A in a 100 A slab, QuantEM multislice from uniform slices vs
from the quantem_export slice grid + initial potential. Needs abtem and quantem in one environment.
    python m5_dry_run.py   ->  prints loss and Au-column depth for both arms (see results_m5/M5_NOTES.md)"""
import os, sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simple_joint as sj, abtem_data as ad, calibrate_m3 as m3, contrast_scan as cs, quantem_export as qe
from quantem.core.datastructures import Dataset4dstem
from quantem.core.ml import OptimizerParams
from quantem.diffractive_imaging import ObjectPixelated, ProbePixelated, DetectorPixelated, Ptychography, PtychographyDatasetRaster
torch.manual_seed(0)
T = 100.0
setup, positions, nominal, true = m3.config(T)
xy = cs.cluster(setup, m3.TEST_SEED, 3); z_au = 50.0
atoms = cs.specimen(setup, xy, [z_au], 2, 0.0)
c10 = true[1, 0]                                     # -50 + 12 = -38 A: focus 38 A inside
dps = ad.simulate_abtem(setup, atoms, np.array([[c10] + [0.0] * 7]), positions, dose=None, log=lambda *_: None)[0]
data = np.fft.fftshift(dps, axes=(-2, -1)).astype(np.float32)
dk = 1e3 * setup.lam / (setup.K * setup.dr)
dset = Dataset4dstem.from_array(data, name="sim", sampling=(2 * setup.dr,) * 2 + (dk, dk), units=("A", "A", "mrad", "mrad"))
pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
pdset.preprocess(com_fit_function="constant", probe_energy=setup.energy_ev, force_com_rotation=0, plot_rotation=False, plot_com=False)
PAD = 8
def run(obj_model, defocus, iters=30):
    probe = ProbePixelated.from_params(num_probes=1, probe_params={"energy": setup.energy_ev, "defocus": defocus, "semiangle_cutoff": 20.0})
    pt = Ptychography.from_models(dset=pdset, obj_model=obj_model, probe_model=probe, detector_model=DetectorPixelated(), device="cpu")
    pt.preprocess(obj_padding_px=(PAD, PAD), batch_size=64)
    pt.reconstruct(num_iters=iters, reset=True, batch_size=64, device="cpu",
                   optimizer_params={"object": OptimizerParams.Adam(lr=2e-3)})
    return pt
import time
ORIGIN = (setup.K // 2 - PAD) * setup.dr                    # S-grid coordinate of QuantEM object pixel 0
true_px = np.round(xy / setup.dr).astype(int) - (setup.K // 2 - PAD)

def build(arm):
    if arm == "uniform":
        planes = np.linspace(0.0, T - 5.0, 20)               # 20 slabs of 5 A
        phi = np.full((len(planes), 48, 48), 1e-3)
        return ObjectPixelated.from_array(phi.astype(np.float32), slice_thicknesses=5.0, obj_type="potential"), planes, 5.0
    cand = [qe.DepthCandidate(y=y, x=x, modes=[(z_au, 8.0, 1.0)], phase=0.3) for y, x in xy]
    g = qe.slice_grid(cand, T, dz_max=8.0, dz_min=2.5, extra_slices=2.0)
    phi = qe.initial_potential(cand, g, (48, 48), setup.dr, origin=(ORIGIN, ORIGIN)) + 1e-3
    return qe.to_quantem(phi, g), g.planes, g.slab_thicknesses

def depth_profile(pt, planes, thick):
    o = pt.obj_model.obj.detach().cpu().numpy()
    prof = np.mean([o[:, y, x] for y, x in true_px], axis=0) / np.asarray(thick)   # potential per A at the Au columns
    zc = np.asarray(planes) + np.asarray(thick) / 2
    return zc, prof, float(zc[np.argmax(prof)]), float((zc * np.clip(prof, 0, None)).sum() / max(np.clip(prof, 0, None).sum(), 1e-12))

for arm in ("uniform", "export"):
    obj, planes, thick = build(arm)
    probe = ProbePixelated.from_params(num_probes=1, probe_params={"energy": setup.energy_ev, "defocus": -c10, "semiangle_cutoff": 20.0})
    pt = Ptychography.from_models(dset=pdset, obj_model=obj, probe_model=probe, detector_model=DetectorPixelated(), device="cpu")
    pt.preprocess(obj_padding_px=(PAD, PAD), batch_size=64)
    t0 = time.time()
    pt.reconstruct(num_iters=40, reset=True, batch_size=64, device="cpu",
                   optimizer_params={"object": OptimizerParams.Adam(lr=5e-3)}, constraints={"object": {"positivity": True}})
    zc, prof, zmax, zcen = depth_profile(pt, planes, thick)
    L = np.asarray(pt.iter_losses)
    print(f"{arm:8s}: {len(planes)} planes, loss {L[0]:.4f} -> {L[-1]:.4f}, Au column peak at {zmax:.1f} A, centroid {zcen:.1f} A "
          f"(true {z_au}), {time.time() - t0:.0f} s", flush=True)
    print("          profile:", np.round(prof / max(prof.max(), 1e-12), 2).tolist())
