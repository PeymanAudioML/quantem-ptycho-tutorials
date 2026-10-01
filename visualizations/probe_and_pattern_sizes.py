"""
Size and shape of the probe (real space) and the diffraction pattern (reciprocal space)
for the white-noise-object tutorial, with dimensions annotated.

    python visualizations/probe_and_pattern_sizes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

OUT = Path(__file__).parent / "white_noise_steps"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.titlesize": 13, "figure.titleweight": "bold",
})

# ---------------------------------------------------------------- parameters (same as notebook)
n = 96
k_max, k_probe = 2.0, 1.0                      # 1/A
energy = 300e3                                  # eV
V = energy
wavelength = 12.2643 / np.sqrt(V * (1 + 0.978476e-6 * V))   # relativistic, A
sampling = 1 / k_max / 2                        # 0.25 A / px
dk = 2 * k_max / n                              # 0.0417 1/A / px
alpha_mrad = k_probe * wavelength * 1e3
ab = {"C10": 200.0, "C12": 100.0, "phi12": np.deg2rad(11)}

kx = np.fft.fftfreq(n, sampling)
k2 = kx[:, None] ** 2 + kx[None, :] ** 2
kk = np.sqrt(k2)
phi = np.arctan2(kx[None, :], kx[:, None])
aperture = np.sqrt(np.clip((k_probe - kk) / dk + 0.5, 0, 1))


def probe(ab):
    chi = np.pi * wavelength * k2 * (ab.get("C10", 0) + ab.get("C12", 0) * np.cos(2 * (phi - ab.get("phi12", 0))))
    pf = aperture * np.exp(-1j * chi)
    pf /= np.sqrt((np.abs(pf) ** 2).sum())
    return np.fft.fftshift(np.fft.ifft2(pf) * n)


P_ideal, P = probe({}), probe(ab)
I_ideal, I_probe = np.abs(P_ideal) ** 2, np.abs(P) ** 2

# one diffraction pattern from the white-noise object (seeded, same construction as the notebook)
np.random.seed(0)
arr = np.random.randn(n, n)
pos, neg = np.arange(1, n // 2), np.flip(np.arange(n // 2 + 1, n))
arr[pos[:, None], pos[None, :]] = -arr[neg[:, None], neg[None, :]]
arr[pos[:, None], neg[None, :]] = -arr[neg[:, None], pos[None, :]]
arr[0, pos] = -arr[0, neg]; arr[pos, 0] = -arr[neg, 0]
arr[n // 2, :] = 0; arr[:, n // 2] = 0; arr[0, 0] = 0
pot = np.fft.ifft2(np.exp(2j * np.pi * arr)).real
obj = np.exp(1j * pot)
pattern = np.fft.fftshift(np.abs(np.fft.fft2(obj * np.fft.ifftshift(P))) ** 2)

# ---------------------------------------------------------------- measured sizes
x = (np.arange(n) - n // 2) * sampling          # A, probe centred at 0


def moments(I):
    """Return centre, FWHM-equivalent major/minor widths (A) and orientation (deg) from 2nd moments."""
    X, Y = np.meshgrid(x, x, indexing="xy")
    w = I / I.sum()
    cx, cy = (w * X).sum(), (w * Y).sum()
    cxx = (w * (X - cx) ** 2).sum(); cyy = (w * (Y - cy) ** 2).sum(); cxy = (w * (X - cx) * (Y - cy)).sum()
    evals, evecs = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    s_minor, s_major = np.sqrt(evals)
    ang = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    return (cx, cy), 2.355 * s_major, 2.355 * s_minor, ang


def fwhm_1d(profile, coords):
    half = profile.max() / 2
    above = np.where(profile >= half)[0]
    return coords[above[-1]] - coords[above[0]] + (coords[1] - coords[0])


c0, maj, mnr, ang = moments(I_probe)
fwhm_ideal = fwhm_1d(I_ideal[n // 2], x)
airy_fwhm = 0.51 * wavelength / (alpha_mrad * 1e-3)       # analytic, A
defocus_diam = 2 * ab["C10"] * alpha_mrad * 1e-3          # geometric defocus disk, A
r90 = np.sort(I_probe.ravel())[::-1]
area90 = (np.cumsum(r90) < 0.9 * r90.sum()).sum() * sampling ** 2

print(f"wavelength            {wavelength:.5f} A   semiangle {alpha_mrad:.2f} mrad")
print(f"probe array           {n}x{n} px = {n*sampling:.0f} x {n*sampling:.0f} A  ({sampling} A/px)")
print(f"ideal probe FWHM      {fwhm_ideal:.2f} A (pixel-limited), analytic Airy {airy_fwhm:.2f} A")
print(f"aberrated probe FWHM  {maj:.1f} A x {mnr:.1f} A, long axis at {ang:.0f} deg")
print(f"90%-energy area       {area90:.1f} A^2 (equiv. diameter {2*np.sqrt(area90/np.pi):.1f} A)")
print(f"geometric defocus disk diameter {defocus_diam:.1f} A")
print(f"pattern array         {n}x{n} px, {dk:.4f} 1/A/px, spans +-{k_max} 1/A (+-{k_max*wavelength*1e3:.1f} mrad)")
print(f"BF disk               radius {k_probe} 1/A = {k_probe/dk:.0f} px = {alpha_mrad:.1f} mrad")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(15, 10.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
ext_r = [x[0] - sampling / 2, x[-1] + sampling / 2] * 2
ext_r = [ext_r[0], ext_r[1], ext_r[1], ext_r[0]]
kc = (np.arange(n) - n // 2) * dk
ext_k = [kc[0] - dk / 2, kc[-1] + dk / 2, kc[-1] + dk / 2, kc[0] - dk / 2]


def style(ax):
    for s in ax.spines.values():
        s.set_visible(False)


# (a) ideal probe, zoomed
ax = fig.add_subplot(gs[0, 0])
z = 3.0
ax.imshow(I_ideal, cmap="magma", extent=ext_r)
ax.set_xlim(-z, z); ax.set_ylim(z, -z)
ax.annotate("", xy=(-fwhm_ideal / 2, 1.6), xytext=(fwhm_ideal / 2, 1.6),
            arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
ax.text(0, 2.2, f"FWHM ≈ {airy_fwhm:.2f} Å", color="white", ha="center", fontsize=9)
ax.set_title("(a) Probe without aberrations  |P(r)|²\n(zoom 6 × 6 Å; only ~2 px wide at 0.25 Å/px)")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); style(ax)

# (b) aberrated probe, full field, with moment ellipse
ax = fig.add_subplot(gs[0, 1])
ax.imshow(I_probe, cmap="magma", extent=ext_r)
ax.add_patch(Ellipse(c0, maj, mnr, angle=ang, fill=False, ec=AQUA, lw=2))
ax.add_patch(Circle((0, 0), defocus_diam / 2, fill=False, ec="white", lw=1, ls="--"))
ax.text(0, -9.5, f"FWHM ellipse: {maj:.1f} × {mnr:.1f} Å\nlong axis at {ang:.0f}°",
        color=AQUA, ha="center", fontsize=9, fontweight="bold")
ax.text(0, 9.8, f"dashed: defocus-only disk Ø {defocus_diam:.1f} Å", color="white", ha="center", fontsize=8.5)
ax.set_title(f"(b) Probe with aberrations  |P(r)|²\nfull array {n}×{n} px = {n*sampling:.0f}×{n*sampling:.0f} Å")
ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); style(ax)

# (c) probe line profiles
ax = fig.add_subplot(gs[0, 2])
t = np.linspace(-10, 10, 401)
from scipy.ndimage import map_coordinates
for a_deg, lab, col in [(ang, f"long axis ({ang:.0f}°)", BLUE), (ang + 90, "short axis", ORANGE)]:
    a = np.radians(a_deg)
    px = (c0[0] + t * np.cos(a)) / sampling + n // 2
    py = (c0[1] + t * np.sin(a)) / sampling + n // 2
    prof = map_coordinates(I_probe, [py, px], order=1)
    ax.plot(t, prof / I_probe.max(), color=col, lw=2, label=lab)
ax.axhline(0.5, color=INK_2, lw=1, ls=":")
ax.text(9.8, 0.52, "half max", color=INK_2, fontsize=8, ha="right")
ax.set_xlabel("distance from probe centre (Å)"); ax.set_ylabel("|P|² (normalised)")
ax.set_title("(c) Aberrated probe line profiles")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="center right", bbox_to_anchor=(1.02, 0.72))

# (d) diffraction pattern, full detector
ax = fig.add_subplot(gs[1, 0])
ax.imshow(np.sqrt(pattern), cmap="magma", extent=ext_k)
ax.add_patch(Circle((0, 0), k_probe, fill=False, ec=AQUA, lw=2))
ax.annotate("", xy=(0, 0), xytext=(k_probe * np.cos(np.pi / 4), -k_probe * np.sin(np.pi / 4)),
            arrowprops=dict(arrowstyle="<->", color=AQUA, lw=1.5))
ax.text(0.85, -0.95, f"r = {k_probe:.0f} Å⁻¹\n= {k_probe/dk:.0f} px\n= {alpha_mrad:.1f} mrad",
        color=AQUA, fontsize=8.5, fontweight="bold")
ax.add_patch(Rectangle((ext_k[0], ext_k[3]), 2 * k_max, 2 * k_max, fill=False, ec="white", lw=1, ls="--"))
ax.text(0, 1.85, f"detector edge ±{k_max:.0f} Å⁻¹ (±{k_max*wavelength*1e3:.0f} mrad)", color="white",
        ha="center", fontsize=8.5)
ax.set_title(f"(d) Diffraction pattern  I(k)  (√ shown)\n{n}×{n} px, {dk:.4f} Å⁻¹/px")
ax.set_xlabel("kx (Å⁻¹)"); ax.set_ylabel("ky (Å⁻¹)"); style(ax)

# (e) signal inside the disk
ax = fig.add_subplot(gs[1, 1])
mean_disk = aperture.max() ** 2
dI = pattern - np.fft.fftshift(np.abs(np.fft.fft2(np.fft.ifftshift(P))) ** 2)
v = np.abs(dI).max()
ax.imshow(dI, cmap="RdBu_r", vmin=-v, vmax=v, extent=ext_k)
ax.add_patch(Circle((0, 0), k_probe, fill=False, ec=INK_2, lw=1, ls="--"))
ax.set_xlim(-1.3, 1.3); ax.set_ylim(1.3, -1.3)
ax.set_title("(e) Object signal: I(k) − |A(k)|²\n(zoom on the bright-field disk)")
ax.set_xlabel("kx (Å⁻¹)"); ax.set_ylabel("ky (Å⁻¹)"); style(ax)

# (f) radial profile of pattern
ax = fig.add_subplot(gs[1, 2])
cut = pattern[n // 2]
ax.plot(kc, cut / cut.max(), color=BLUE, lw=2)
for s in (-k_probe, k_probe):
    ax.axvline(s, color=AQUA, lw=1.5, ls="--")
ax.text(k_probe + 0.05, 0.55, f"disk edge\n±{k_probe:.0f} Å⁻¹\n({alpha_mrad:.1f} mrad)", color=INK_2, fontsize=8.5)
ax.set_xlim(-k_max, k_max)
ax.set_xlabel("kx (Å⁻¹)"); ax.set_ylabel("I (normalised)")
ax.set_title("(f) Horizontal cut through the pattern")
secax = ax.secondary_xaxis("top", functions=(lambda k: k * wavelength * 1e3, lambda m: m / (wavelength * 1e3)))
secax.set_xlabel("scattering angle (mrad)", color=INK_2)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("right",):
    ax.spines[s].set_visible(False)

fig.suptitle("Probe (real space) and diffraction pattern (reciprocal space): size and shape", y=0.995)
fig.text(0.5, 0.005,
         f"300 keV, λ = {wavelength:.4f} Å, semiangle α = {alpha_mrad:.1f} mrad, C10 = 200 Å, C12 = 100 Å @ 11°.  "
         "Probe and pattern are Fourier pairs: a 1 Å⁻¹ aperture gives a ~0.5 Å ideal probe; "
         "aberrations spread it over several Å but leave the disk size unchanged.",
         ha="center", color=INK_2, fontsize=9)
fig.savefig(OUT / "probe_and_pattern_sizes.png", dpi=150, bbox_inches="tight")
print("saved", OUT / "probe_and_pattern_sizes.png")
