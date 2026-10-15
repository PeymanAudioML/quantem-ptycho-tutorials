"""Geometry and resource check of a benchmark configuration (run before generating data; generate_data.py calls it).

    python check_geometry.py [--config configs/benchmark_200A.yaml]   -> results/geometry/<name>.json, .md

simple_joint.check_geometry(..., abtem=True, strict=True) plus the benchmark's own checks:
  scan step represented as an exact whole number of S-grid pixels; detector size and reciprocal pixel consistent
  between the S-matrix window (1 / (K dr)) and the pixelated variant (Y/K binning of the 1 / (Y dr) grid).
Raises on any failure; there is no override.
"""
import argparse, json, math, os

import numpy as np

import bench

def run(cfg, log=print):
    setup = bench.make_setup(cfg)
    pos = bench.scan_positions(cfg)
    nominal, true, ref = bench.coefficients(cfg)
    T = float(cfg["specimen"]["thickness_A"])
    s = cfg["sampling"]
    lines = []
    checks = bench.sj.check_geometry(setup, pos, nominal, T, abtem=True, strict=False, log=lambda m: lines.append(m))
    step_A = s["scan_step_px"] * setup.dr
    extra = [
        dict(check="scan step = whole S-grid pixels", value=step_A / setup.dr, limit=float(s["scan_step_px"]),
             ok=abs(step_A / setup.dr - round(step_A / setup.dr)) < 1e-9, note=f"{step_A:.3f} A = {step_A / setup.dr:g} px"),
        dict(check="pixelated detector = window detector", value=float(setup.Y % setup.K), limit=0.0,
             ok=setup.Y % setup.K == 0,
             note=f"window dk = 1/(K dr) = {1 / (setup.K * setup.dr):.5f} 1/A; pixelated: {setup.Y // setup.K}x"
                  f"{setup.Y // setup.K} bins of 1/(Y dr) = {1 / (setup.Y * setup.dr):.5f} 1/A"),
        dict(check="true aberrations inside window", value=0.0, limit=0.0, ok=True, note="see below"),
    ]
    # the window check above uses the nominal defoci; repeat it with the true ones
    tc = bench.sj.check_geometry(setup, pos, true, T, abtem=True, strict=False, log=lambda m: None)
    win_true = next(c for c in tc if c["check"] == "window size")
    extra[-1].update(value=win_true["value"], limit=win_true["limit"], ok=win_true["ok"], note="with the TRUE defoci")
    extra = [dict(c, value=float(c["value"]), limit=float(c["limit"]), ok=bool(c["ok"])) for c in extra]
    allc = checks + extra
    ny, nx = pos.shape[:2]
    D = len(nominal)
    B = len(setup.beams()[0])
    J = D * ny * nx
    cover = np.zeros((setup.Y, setup.Y), bool)
    for y, x in pos.reshape(-1, 2):
        cover[y - setup.K // 2:y + setup.K - setup.K // 2, x - setup.K // 2:x + setup.K - setup.K // 2] = True
    unknown_reals = 2 * B * int(cover.sum())
    meas = J * setup.K ** 2
    res = dict(
        energy_eV=setup.energy_ev, wavelength_A=setup.lam, alpha_mrad=1e3 * setup.alpha, dr_A=setup.dr, K=setup.K,
        Y=setup.Y, field_of_view_A=setup.Y * setup.dr, window_A=setup.K * setup.dr, scan_shape=[ny, nx],
        scan_step_A=step_A, scanned_extent_A=[(ny - 1) * step_A, (nx - 1) * step_A], datasets=D, reference_dataset=ref,
        patterns=J, detector_pixels=setup.K ** 2, reciprocal_pixel_inv_A=1 / (setup.K * setup.dr),
        beams=B, s_matrix_pixels_covered=int(cover.sum()), measurements=meas, unknown_reals=unknown_reals,
        measurements_per_unknown=meas / unknown_reals, depth_resolution_lambda_over_alpha2_A=setup.lam / setup.alpha ** 2,
        abtem_band_limit_mrad=1e3 * setup.lam / (3 * setup.dr), thickness_A=T,
        potential_slices=int(math.ceil(T / cfg["specimen"]["potential_slice_A"])),
        memory_MB=dict(
            patterns_per_variant=J * setup.K ** 2 * 4 / 1e6,
            s_matrix_complex64=B * setup.Y ** 2 * 8 / 1e6,
            s_matrix_with_adam_state=3 * B * setup.Y ** 2 * 8 / 1e6,
            forward_window_gather_per_16_patterns=B * 16 * setup.K ** 2 * 8 / 1e6,
            potential=math.ceil(T / cfg["specimen"]["potential_slice_A"]) * setup.Y ** 2 * 4 / 1e6),
        checks=allc, ok=all(c["ok"] for c in allc), config=cfg["_path"], config_sha256=cfg["_sha256"])
    return res, lines


def markdown(res):
    L = [f"# Geometry check: {res['config']}", "",
         f"{res['energy_eV'] / 1e3:.0f} kV (lambda {res['wavelength_A']:.5f} A), {res['alpha_mrad']:g} mrad; "
         f"dr {res['dr_A']} A, window K = {res['K']} ({res['window_A']:.1f} A), grid Y = {res['Y']} "
         f"({res['field_of_view_A']:.1f} A); scan {res['scan_shape'][0]} x {res['scan_shape'][1]} at "
         f"{res['scan_step_A']:.2f} A ({res['scanned_extent_A'][0]:.1f} A field); {res['datasets']} defoci, reference "
         f"dataset {res['reference_dataset']}.", "",
         f"Patterns {res['patterns']}; S-matrix beams {res['beams']}; S pixels covered {res['s_matrix_pixels_covered']}; "
         f"measurements {res['measurements']:,} vs unknown reals {res['unknown_reals']:,} -> "
         f"**{res['measurements_per_unknown']:.3f} measurements per unknown**. Depth resolution lambda/alpha^2 = "
         f"{res['depth_resolution_lambda_over_alpha2_A']:.1f} A; abTEM band limit {res['abtem_band_limit_mrad']:.1f} mrad.", "",
         "| check | value | limit | ok | note |", "|---|---|---|---|---|"]
    L += [f"| {c['check']} | {c['value']:.4g} | {c['limit']:.4g} | {'yes' if c['ok'] else '**NO**'} | {c['note']} |"
          for c in res["checks"]]
    L += ["", "Memory (MB): " + ", ".join(f"{k} {v:.1f}" for k, v in res["memory_MB"].items()), "",
          f"Overall: {'PASS' if res['ok'] else 'FAIL'}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=bench.DEFAULT_CONFIG)
    a = ap.parse_args()
    cfg = bench.load_config(a.config)
    res, lines = run(cfg)
    out = os.path.join(bench.HERE, "results", "geometry")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"{cfg['name']}.json"), "w"), indent=1)
    md = markdown(res)
    open(os.path.join(out, f"{cfg['name']}.md"), "w").write(md)
    print(md)
    for m in lines:
        print(m)
    if not res["ok"]:
        raise SystemExit("geometry check FAILED")


if __name__ == "__main__":
    main()
