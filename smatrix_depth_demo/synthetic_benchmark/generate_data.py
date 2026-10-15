"""Generate benchmark specimens: atoms -> abTEM potential -> three-defocus 4D-STEM (two detector variants, noiseless and
Poisson) + complete ground truth, in the realization layout read by run_realization.py.

    python generate_data.py --specimens A_z100                 # one specimen
    python generate_data.py --specimens all                    # every specimen in the config
    python generate_data.py --specimens A_z100 --overwrite

Layout (data/<config name>/<group>/<specimen>/):
    ground_truth/   atoms.extxyz (feature id per atom, -1 = carbon), potential.zarr (V*A per slice), probe_d<i>.zarr
                    (true entrance probe, real + Fourier), truth.json (features, aberrations nominal/true, reference,
                    slicing, sigma, checks, versions, seeds, config hash)
    <variant>_<noise>/def<i>/   4dstem_quantem/ (Dataset4dstem: _array (Ny, Nx, K, K) centred, _sampling, metadata with
                    NOMINAL defocus only), and links probe.zarr / potential.zarr / atoms.extxyz -> ground_truth
        variants: window (S-matrix model), pixelated (abTEM PixelatedDetector + pixel integration)
        noise:    noiseless (intensity fraction per pixel), poisson (counts, expectation = fraction x dose)
Ground truth never enters the datasets' metadata.
"""
import argparse, json, os, shutil, time

import numpy as np

import bench
import check_geometry


def generate(cfg, name, overwrite=False, workers=None, log=print):
    setup = bench.make_setup(cfg)
    spec = cfg["specimens"][name]
    root = bench.specimen_dir(cfg, name)
    if os.path.exists(os.path.join(root, "ground_truth", "truth.json")) and not overwrite:
        log(f"{name}: exists ({root}), skipped (--overwrite to regenerate)")
        return root
    if os.path.exists(root):
        shutil.rmtree(root)
    t0 = time.time()
    geo, _ = check_geometry.run(cfg, log=lambda m: None)
    if not geo["ok"]:
        raise SystemExit(f"geometry check failed for {cfg['_path']}; not generating")
    pos = bench.scan_positions(cfg)
    nominal, true, ref = bench.coefficients(cfg)
    T = float(cfg["specimen"]["thickness_A"])
    atoms, feats = bench.build_specimen(cfg, setup, name)
    V, bounds, sigma, potential, pchecks = bench.build_potential(cfg, setup, atoms)
    log(f"{name}: {len(atoms)} atoms ({sum(f['n_atoms'] for f in feats)} Au in {len(feats)} features), potential "
        f"{V.shape} verified ({time.time() - t0:.0f} s)")
    sim = bench.simulate(setup, atoms, float(cfg["specimen"]["potential_slice_A"]), true, pos, workers=workers, log=log)

    gt = os.path.join(root, "ground_truth")
    os.makedirs(gt)
    from ase.io import write as ase_write
    ase_write(os.path.join(gt, "atoms.extxyz"), atoms, format="extxyz")
    bench.write_potential(os.path.join(gt, "potential.zarr"), V, bounds, setup, sigma)
    for d in range(len(true)):
        bench.write_probe(os.path.join(gt, f"probe_d{d}.zarr"), bench.ad.entrance_probe(setup, true[d]), setup,
                          true[d], nominal[d])
    step = cfg["sampling"]["scan_step_px"] * setup.dr
    dk = 1.0 / (setup.K * setup.dr)
    positions_A = pos * setup.dr                                  # cell frame, axis-0 coordinate first
    seeds = {}
    dose = float(cfg["noise"]["dose_per_pattern"])
    for vi, variant in enumerate(cfg["detector"]["variants"]):
        for noise in ("noiseless", "poisson"):
            for d in range(len(true)):
                I = sim[variant][d]
                if noise == "poisson":
                    seed = int(cfg["noise"]["seed"]) + 1000 * vi + d
                    seeds[f"{variant}/def{d}"] = seed
                    data, units = bench.poisson(I, dose, seed), "electron counts"
                else:
                    seed, data, units = None, I, "fraction of incident electrons per pixel"
                ddir = os.path.join(root, f"{variant}_{noise}", f"def{d}")
                meta = dict(label=f"{name} {variant} {noise} dataset {d}", variant=variant, noise=noise,
                            signal_units=units, dose_per_pattern=dose if noise == "poisson" else None, noise_seed=seed,
                            integration=("point samples of FFT_ortho of the K x K exit-wave window" if variant == "window"
                                         else f"abTEM PixelatedDetector(max_angle=None) / Y^2, {sim['bin']}x{sim['bin']} "
                                              "pixel sum"),
                            dataset_index=d, reference_dataset=ref, specimen=name, config=cfg["_path"],
                            specimen_sha256=bench.specimen_sha256(cfg, name),
                            config_sha256=cfg["_sha256"], generated=bench.now())
                bench.write_dataset(os.path.join(ddir, "4dstem_quantem"), data, step, dk, sim["zero_pixel"][variant],
                                    positions_A, [setup.Y * setup.dr] * 2 + [T], setup.energy_ev, 1e3 * setup.alpha,
                                    -nominal[d, 0], meta, taper_mrad=1e3 * setup.taper)
                bench.link(os.path.join(gt, f"probe_d{d}.zarr"), os.path.join(ddir, "probe.zarr"))
                bench.link(os.path.join(gt, "potential.zarr"), os.path.join(ddir, "potential.zarr"))
                bench.link(os.path.join(gt, "atoms.extxyz"), os.path.join(ddir, "atoms.extxyz"))
    truth = dict(
        specimen=name, group=spec["group"], config=cfg["_path"], config_sha256=cfg["_sha256"],
        specimen_sha256=bench.specimen_sha256(cfg, name), generated=bench.now(),
        versions=bench.versions(), thickness_A=T, cell_A=[setup.Y * setup.dr] * 2 + [T],
        axis_order="axis 0 = ase x, axis 1 = ase y; z = depth below the entrance",
        features=feats, n_features=len(feats), n_au=int(sum(f["n_atoms"] for f in feats)),
        n_carbon=int((np.asarray(atoms.get_chemical_symbols()) == "C").sum()),
        carbon=spec.get("carbon"), potential=dict(slice_boundaries_A=bounds.tolist(), units="V A per slice",
                                                  sigma_rad_per_V_A=sigma, checks=pchecks,
                                                  abtem="Potential(atoms, gpts=(Y, Y), slice_thickness), defaults"),
        aberrations=dict(names=bench.sj.NAMES, nominal=nominal.tolist(), true=true.tolist(), reference_dataset=ref,
                         focus_depth_nominal_A=(-nominal[:, 0]).tolist(), focus_depth_true_A=(-true[:, 0]).tolist(),
                         gauge="dataset `reference` is exact; the S-matrix fit keeps it fixed and learns the others"),
        scan=dict(positions_px_first=pos[0, 0].tolist(), step_A=step, shape=list(pos.shape[:2]),
                  positions_A="ground_truth datasets: _metadata/simulation/padded_scan_positions_A"),
        detector=dict(K=setup.K, dk_inv_A=dk, zero_pixel=sim["zero_pixel"], pixel_bin=sim["bin"],
                      abtem_pixelated_vs_fft_max_rel_diff=sim["pixelated_vs_fft_check"]),
        noise=dict(dose_per_pattern=dose, seeds=seeds),
        captured_intensity=dict(window=float(sim["window"].sum(axis=(-2, -1)).mean()),
                                pixelated=float(sim["pixelated"].sum(axis=(-2, -1)).mean())),
        geometry=dict(measurements_per_unknown=geo["measurements_per_unknown"], beams=geo["beams"], ok=geo["ok"]))
    json.dump(truth, open(os.path.join(gt, "truth.json"), "w"), indent=1)
    log(f"{name}: written to {os.path.relpath(root, bench.PARENT)} ({time.time() - t0:.0f} s); mean captured intensity "
        f"window {truth['captured_intensity']['window']:.4f}, pixelated {truth['captured_intensity']['pixelated']:.4f}")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=bench.DEFAULT_CONFIG)
    ap.add_argument("--specimens", default="A_z100", help="comma list or 'all'")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: one per defocus)")
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    names = list(cfg["specimens"]) if a.specimens == "all" else a.specimens.split(",")
    for n in names:
        generate(cfg, n, a.overwrite, a.workers)


if __name__ == "__main__":
    main()
