"""Demo for simple_joint.py: simulate 4D-STEM data of a 3-layer sample with a miscalibrated probe, jointly recover the
S-matrix and the probe aberrations, and refocus through the depth.

    python demo_simple_joint.py                    # small CPU demo (a few minutes)
    python demo_simple_joint.py --device cuda      # same on a GPU

With your own data, replace `simulate(...)` by your patterns (D, Ny, Nx, K, K), scan positions (Ny, Nx, 2) in S-grid
pixels and nominal aberrations (D, P); the rest of `main()` stays the same.
"""
import argparse, math, time
import numpy as np
import torch

import simple_joint as sj


# ------------------------------------------------------------------------------------------------ test data
def make_layers(setup, n_layers=3, atoms=6, seed=1):
    """Phase images of `n_layers` layers of Gaussian 'atoms' near the centre of the field."""
    rng = np.random.default_rng(seed)
    r = np.arange(setup.Y) * setup.dr
    c = setup.Y * setup.dr / 2
    layers = []
    for _ in range(n_layers):
        ph = np.zeros((setup.Y, setup.Y))
        for y0, x0 in c + rng.uniform(-3.5, 3.5, (atoms, 2)):
            ph += 1.0 * np.exp(-((r[:, None] - y0) ** 2 + (r[None, :] - x0) ** 2) / (2 * 0.45 ** 2))
        layers.append(ph)
    return np.array(layers)


def simulate(setup, layers, layer_z, thickness, coeffs, positions, dose=None, seed=0, device="cpu"):
    """Multislice 4D-STEM: probe (aperture + aberrations `coeffs` per dataset) at each scan position, propagate to each
    layer, multiply by exp(i phase), propagate to the exit surface, crop the K x K window, |FFT|^2.
    Returns intensities (D, Ny, Nx, K, K)."""
    dev = torch.device(device)
    Y, K = setup.Y, setup.K
    q = torch.as_tensor(np.fft.fftfreq(Y, setup.dr), dtype=torch.float64, device=dev)
    qy, qx = q[:, None].expand(Y, Y), q[None, :].expand(Y, Y)
    basis = sj.aberration_basis(qy.reshape(-1), qx.reshape(-1), setup.lam).reshape(len(sj.NAMES), Y, Y)
    aperture = (qy ** 2 + qx ** 2) <= (setup.alpha / setup.lam) ** 2
    zs = [0.0] + list(layer_z) + [thickness]
    prop = lambda dz: torch.exp(-1j * math.pi * setup.lam * dz * (qy ** 2 + qx ** 2)).to(torch.complex64)
    props = [prop(zs[i + 1] - zs[i]) for i in range(len(zs) - 1)]
    trans = [torch.exp(1j * torch.as_tensor(L, device=dev)).to(torch.complex64) for L in layers]
    pos = torch.as_tensor(np.asarray(positions).reshape(-1, 2), device=dev)
    ar, win = torch.arange(Y, device=dev), torch.arange(-(K // 2), K - K // 2, device=dev)
    out = []
    for c in np.asarray(coeffs, float):
        chi = torch.einsum("p,pyx->yx", torch.as_tensor(c, device=dev), basis)
        p0 = torch.fft.ifft2(aperture * torch.exp(-1j * chi))
        p0 = (p0 / p0.abs().pow(2).sum().sqrt()).to(torch.complex64)
        I = []
        for chunk in pos.split(64):
            ry, rx = chunk[:, 0], chunk[:, 1]
            psi = p0[((ar - ry[:, None]) % Y)[:, :, None], ((ar - rx[:, None]) % Y)[:, None, :]]   # probe at r
            for i in range(len(trans)):
                psi = torch.fft.ifft2(torch.fft.fft2(psi) * props[i]) * trans[i]
            psi = torch.fft.ifft2(torch.fft.fft2(psi) * props[-1])
            w = psi[torch.arange(len(chunk), device=dev)[:, None, None], (ry[:, None] + win)[:, :, None],
                    (rx[:, None] + win)[:, None, :]]
            I.append(torch.fft.fft2(w, norm="ortho").abs() ** 2)
        out.append(torch.cat(I).cpu().numpy())
    dps = np.array(out, np.float32).reshape(len(coeffs), *np.asarray(positions).shape[:2], K, K)
    if dose:
        dps = np.random.default_rng(seed).poisson(dps * dose).astype(np.float32) / dose
    return dps


def layer_depths(volume, layers, depths, crop):
    """For each true layer, the depth where the refocused phase correlates best with it (and that correlation)."""
    out = []
    for L in layers:
        Lc = L[crop] - L[crop].mean()
        r = []
        for sec in np.angle(volume[(slice(None),) + crop]):
            s = sec - sec.mean()
            r.append(float((s * Lc).sum() / np.sqrt((s * s).sum() * (Lc * Lc).sum())))
        r = np.array(r)
        out.append((float(depths[np.argmax(r)]), float(r.max())))
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--step", type=int, default=2, help="scan step in pixels (smaller = more overlap = better probe)")
    ap.add_argument("--dose", type=float, default=1e5, help="electrons per pattern (0 = noiseless)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    setup = sj.Setup(energy_ev=300e3, alpha=20e-3, dr=0.3, K=32, Y=64)
    # scan: whole windows inside the S grid
    s0 = setup.K // 2
    grid = np.arange(s0, setup.Y - s0 + 1, a.step)
    positions = np.stack(np.meshgrid(grid, grid, indexing="ij"), -1)
    # sample: 3 layers at 30 / 100 / 170 A in a 200 A slab
    layers, layer_z, thickness = make_layers(setup), [30.0, 100.0, 170.0], 200.0
    # three datasets at set defoci; datasets 1 and 2 have unknown defocus + astigmatism errors
    nominal = np.zeros((3, len(sj.NAMES)))
    nominal[:, 0] = [-100.0, -50.0, 0.0]
    true = nominal.copy()
    true[1, :3] += [12.0, 6.0, -4.0]           # dC10, dC12a, dC12b [A]
    true[2, :3] += [-10.0, -3.0, 5.0]

    t0 = time.time()
    dps = simulate(setup, layers, layer_z, thickness, true, positions, dose=a.dose, device=a.device)
    print(f"simulated {dps.shape} in {time.time() - t0:.0f}s")

    amp, pos, didx = sj.prepare_data(dps, positions, device=a.device)
    results = {}
    for name, init_probe, train in (("fixed nominal probe", nominal, ()),
                                     ("joint (probe refined)", nominal, ("C10", "C12a", "C12b")),
                                     ("true probe (reference)", true, ())):
        print(f"\n=== {name}")
        model = sj.JointModel(setup, pos, didx, init_probe, trainable=train, device=a.device)
        t0 = time.time()
        hist = sj.fit(model, amp, epochs=a.epochs, log=lambda m: print(m) if name.startswith("joint") else None)
        est = model.coefficients().detach().cpu().numpy()
        depths = np.arange(0.0, thickness + 1e-6, 5.0)
        vol = sj.refocus(model.S, setup, depths)
        crop = (slice(s0, setup.Y - s0), slice(s0, setup.Y - s0))
        found = layer_depths(vol, layers, depths, crop)
        results[name] = (hist[-1], est, found, time.time() - t0)

    print("\n================ summary")
    print("true relative errors [A] (C10, C12a, C12b):", (true - nominal)[1:, :3].tolist())
    for name, (loss, est, found, t) in results.items():
        err = np.abs(est - true)[1:, :3]
        print(f"{name:24s} loss {loss:.6f}  max |C - C_true| {err.max():5.2f} A  "
              f"layer depths {[f'{z:.0f}' for z, _ in found]} (true {layer_z})  r {[f'{r:.2f}' for _, r in found]}  {t:.0f}s")


if __name__ == "__main__":
    main()
