"""
Teaching figures for the apoferritin 4D-STEM tutorials.

Follows simulation_notebooks/apoferritin-pdb-dataset.ipynb (PDB -> ice-embedded potential ->
multislice 4D-STEM) and apoF_dataset_phase_retrieval.ipynb (dose -> direct ptychography),
saving one figure per step to visualizations/apoferritin_steps/.

Needs abtem, gemmi, ase and data/8rqb.pdb1 (RCSB 8RQB, "Biological Assembly 1", PDB format).

    python visualizations/apoferritin_steps.py                 # 3x3 particles, as in the notebook
    APOF_GRID=2 python visualizations/apoferritin_steps.py      # 2x2 particles (enough for the 48x48 crop)

The simulated dataset and ground-truth projected potentials are cached in data/, so a second run
only redraws the figures.
"""

import os
import itertools
import time
from pathlib import Path

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).parent / "apoferritin_steps"
OUT.mkdir(parents=True, exist_ok=True)
PDB = Path(os.environ.get("APOF_PDB", DATA / "8rqb.pdb1"))
GRID = int(os.environ.get("APOF_GRID", 3))
TAG = os.environ.get("APOF_TAG", "")
DSET = DATA / f"apoF_4mrad_1.5um-df_10A-step{TAG}.zip"
GT = DATA / f"apoF_projected_potentials{TAG}.npy"

# ---------------------------------------------------------------- style
SURFACE, INK, INK_2, GRID_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID_C,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.titlesize": 13, "figure.titleweight": "bold", "lines.linewidth": 2,
})


def clean(ax, ticks=False):
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def cbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    if label:
        cb.set_label(label, color=INK_2, fontsize=8.5)


def caption(fig, text, y=-0.04):
    fig.text(0.5, y, text, ha="center", va="top", color=INK_2, fontsize=9.5, wrap=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name, flush=True)


def lineplot(ax):
    ax.grid(color=GRID_C, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- notebook parameters
energy = 300e3
semiangle = 4.0          # mrad
defocus = 1.5e4          # A  (abTEM convention: defocus = -C10)
cell = 256.0             # A
num_probes_per_particle = 24
bin_factor = 2
rotation_deg = 15        # rotation applied to the simulated patterns
dose = 100               # e/A^2 used in the reconstruction notebook

import abtem  # noqa: E402
import ase  # noqa: E402
import gemmi  # noqa: E402
import quantem as em  # noqa: E402

wavelength = abtem.core.energy.energy2wavelength(energy)
sigma = abtem.core.energy.energy2sigma(energy)   # interaction constant, rad / (V A)
probe_diameter = 2 * defocus * semiangle * 1e-3  # geometric defocus disk, A

# ================================================================ 1. PDB -> atoms
structure = gemmi.read_pdb(str(PDB))
xyz_list = []
for n_asu in range(len(structure)):
    model = structure[n_asu]
    residues = [res for ch in model for res in ch]
    xyz_list.append(np.array([[*atom.pos.tolist(), atom.element.atomic_number]
                              for res in residues for atom in res]))
xyz = np.concatenate(xyz_list)
atoms = ase.Atoms(numbers=xyz[:, 3], positions=xyz[:, :3], cell=[cell] * 3)
atoms.center()
pos, Z = atoms.positions, atoms.numbers
radius = np.linalg.norm(pos - pos.mean(0), axis=1)
n_chains = sum(len(structure[i]) for i in range(len(structure)))
elements, counts = np.unique(Z, return_counts=True)
symbols = [ase.data.chemical_symbols[z] for z in elements]
print(f"{len(structure)} models, {n_chains} chains, {len(atoms)} atoms, outer radius {np.percentile(radius, 99):.0f} A")

fig = plt.figure(figsize=(15, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.3)
colors = {6: "#52514e", 7: SERIES[0], 8: SERIES[1], 16: SERIES[3]}
order = np.argsort(pos[:, 2])
for i, (a, b, lab) in enumerate([(0, 1, "top view (x–y)"), (0, 2, "side view (x–z)")]):
    ax = fig.add_subplot(gs[i])
    o = np.argsort(pos[:, 3 - a - b])
    ax.scatter(pos[o, a], pos[o, b], s=0.15, c=[colors.get(z, SERIES[4]) for z in Z[o]], alpha=0.6, lw=0)
    ax.set_aspect("equal"); ax.set_xlim(0, cell); ax.set_ylim(cell, 0)
    ax.set_title(f"Atoms, {lab}"); ax.set_xlabel("Å"); clean(ax, ticks=True)
ax = fig.add_subplot(gs[2])
shell = np.abs(pos[:, 2] - pos[:, 2].mean()) < 8
ax.scatter(pos[shell, 0], pos[shell, 1], s=0.4, c=[colors.get(z, SERIES[4]) for z in Z[shell]], lw=0)
r_out = np.percentile(radius, 99); r_in = np.percentile(radius, 3)
for r, t in [(r_out, f"outer Ø ≈ {2*r_out/10:.0f} nm"), (r_in, f"cavity Ø ≈ {2*r_in/10:.0f} nm")]:
    ax.add_patch(Circle(pos.mean(0)[:2], r, fill=False, ec=SERIES[2], lw=1.3, ls="--"))
ax.text(pos.mean(0)[0], pos.mean(0)[1] + r_out + 14, f"outer Ø ≈ {2*r_out/10:.0f} nm,  cavity Ø ≈ {2*r_in/10:.0f} nm",
        ha="center", color=SERIES[2], fontsize=8.5, fontweight="bold")
ax.set_aspect("equal"); ax.set_xlim(0, cell); ax.set_ylim(cell, 0)
ax.set_title("16 Å thick central cut: a hollow cage"); ax.set_xlabel("Å"); clean(ax, ticks=True)
ax = fig.add_subplot(gs[3])
ax.barh(symbols, counts, color=[colors.get(z, SERIES[4]) for z in elements], height=0.6)
for y, c in enumerate(counts):
    ax.text(c, y, f" {c:,}", va="center", fontsize=8.5, color=INK_2)
ax.set_title("Atoms by element"); ax.set_xlim(0, counts.max() * 1.35); lineplot(ax); ax.grid(False)
ax.tick_params(axis="y", length=0)
fig.suptitle(f"Step 1 · Apoferritin (PDB 8RQB): {n_chains} protein chains, {len(atoms):,} atoms", y=1.03)
caption(fig, "The PDB file stores one asymmetric unit; looping over the 'biological assembly' copies "
             "rebuilds the full 24-subunit cage. Only light atoms (C, N, O, S): a very weak electron scatterer.")
save(fig, "step01_atoms.png")

# ================================================================ 2. potential in vacuum
potential_vac = abtem.Potential(atoms, slice_thickness=1, sampling=(1, 1)).build(lazy=False)
vac_xyz = np.ascontiguousarray(potential_vac.array.transpose(1, 2, 0))   # (x, y, z), V*A per 1 A slice
proj_vac = vac_xyz.sum(-1)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.35})
im = axs[0].imshow(proj_vac.T * sigma, cmap="magma", extent=[0, cell, cell, 0])
axs[0].set_title("Projected potential × σ\n= phase shift of the beam (rad)"); cbar(fig, im, axs[0], "rad")
im = axs[1].imshow(vac_xyz[:, :, vac_xyz.shape[2] // 2].T, cmap="magma", extent=[0, cell, cell, 0],
                   vmax=np.percentile(vac_xyz, 99.9))
axs[1].set_title("Central 1 Å slice of the 3D potential"); cbar(fig, im, axs[1], "V·Å")
mid = proj_vac.shape[1] // 2
axs[2].plot(np.arange(proj_vac.shape[0]), proj_vac[:, mid] * sigma, color=SERIES[0])
axs[2].set_xlabel("x (Å)"); axs[2].set_ylabel("phase (rad)"); axs[2].set_title("Line profile through the centre")
lineplot(axs[2])
for ax in axs[:2]:
    ax.set_xlabel("Å"); clean(ax, ticks=True)
fig.suptitle(f"Step 2 · Electrostatic potential of the protein in vacuum (1 Å grid, σ = {sigma*1e3:.2f} mrad/(V·Å))",
             y=1.04)
caption(fig, "Electrons are deflected by the electrostatic potential V(r). For a thin sample the beam picks up a "
             "phase φ(x,y) = σ·∫V dz — this phase image is what ptychography tries to recover.")
save(fig, "step02_vacuum_potential.png")

# ================================================================ 3. ice embedding
def solvent_density_function(r, r1=0.5, r2=1.7, r3=0.7, a2=0.2, a3=-0.15, s1=1, s2=1.77, s3=1.06):
    """Shang & Sigworth continuum water model, J. Struct. Biol. 2012."""
    return (0.5 + 0.5 * scipy.special.erf((r - r1) / np.sqrt(2) / s1)
            + a2 * np.exp(-(r - r2) ** 2 / 2 / s2 ** 2) + a3 * np.exp(-(r - r3) ** 2 / 2 / s3 ** 2))


filtered = scipy.ndimage.gaussian_filter(vac_xyz, 5)
protein_surface = filtered < 2
protein_distance = scipy.ndimage.distance_transform_edt(protein_surface)
solvent_density = solvent_density_function(protein_distance)
embedded_xyz = solvent_density * 3.6 + vac_xyz
embedded = scipy.ndimage.zoom(embedded_xyz, 3 / 2)          # 2/3 A voxels, as in the notebook
res = embedded.shape[0]
pixel_size = cell / res
bin_factor_z = res // 32
print("embedded volume", embedded.shape, "pixel", pixel_size)

zc = vac_xyz.shape[2] // 2
fig = plt.figure(figsize=(16, 4.4))
gs = fig.add_gridspec(1, 4, wspace=0.35)
ax = fig.add_subplot(gs[0])
r = np.linspace(0, 8, 400)
ax.plot(r, solvent_density_function(r), color=SERIES[0])
ax.axhline(1, color=INK_2, lw=1, ls=":"); ax.text(7.9, 1.03, "bulk water", ha="right", color=INK_2, fontsize=8.5)
ax.set_xlabel("distance from protein surface (Å)"); ax.set_ylabel("water density / bulk")
ax.set_title("Shang–Sigworth water model\n(hydration shell bump near 1.7 Å)"); lineplot(ax)
for i, (img, title, cm, lab) in enumerate([
        (protein_distance[:, :, zc], "Distance to protein surface", "viridis", "Å"),
        (solvent_density[:, :, zc] * 3.6, "Ice potential = 3.6 V × density", "magma", "V"),
        (embedded_xyz[:, :, zc], "Protein + ice (central slice)", "magma", "V")]):
    ax = fig.add_subplot(gs[i + 1])
    im = ax.imshow(img.T, cmap=cm, extent=[0, cell, cell, 0],
                   vmax=np.percentile(img, 99.5) if i == 2 else None)
    ax.set_title(title); ax.set_xlabel("Å"); clean(ax, ticks=True); cbar(fig, im, ax, lab)
fig.suptitle("Step 3 · Embedding the protein in vitreous ice", y=1.04)
caption(fig, "Real cryo samples are frozen in glassy ice. Ice has its own mean potential (~3.6 V), so the protein "
             "only stands out by a small difference — the main reason cryo images have such low contrast.")
save(fig, "step03_ice_embedding.png")

# ================================================================ 4. slicing for multislice
binned = embedded.reshape(res, res, res // bin_factor_z, bin_factor_z).sum(-1)     # (x, y, 32)
n_slices = binned.shape[2]
dz = pixel_size * bin_factor_z
fig, axs = plt.subplots(1, 5, figsize=(17, 3.9), gridspec_kw={"wspace": 0.12})
for ax, k in zip(axs[:4], [4, 12, 16, 24]):
    im = ax.imshow(binned[:, :, k].T, cmap="magma", extent=[0, cell, cell, 0])
    ax.set_title(f"slice {k} of {n_slices}  (z ≈ {k*dz:.0f} Å)"); clean(ax)
im = axs[4].imshow(binned.sum(-1).T * sigma * pixel_size, cmap="magma", extent=[0, cell, cell, 0])
axs[4].set_title("sum of all slices\n(projected phase, rad)"); clean(axs[4]); cbar(fig, im, axs[4], "rad")
fig.suptitle(f"Step 4 · Cutting the {res}³ volume into {n_slices} slices of {dz:.0f} Å for multislice simulation",
             y=1.07)
caption(fig, "Multislice: the beam is multiplied by each slice's phase, then propagated (Fresnel) 8 Å to the next. "
             "This captures thickness effects that a single projection would miss.", y=-0.03)
save(fig, "step04_slices.png")

# ================================================================ 5. probe
potential_arr = abtem.PotentialArray(binned.transpose(2, 0, 1), slice_thickness=dz, sampling=(pixel_size, pixel_size))
probe = abtem.Probe(energy=energy, semiangle_cutoff=semiangle, defocus=defocus)
probe.match_grid(potential_arr)
probe_wave = probe.build(lazy=False).array          # abTEM centres the probe in the cell
P_int = np.abs(probe_wave) ** 2
probe0 = abtem.Probe(energy=energy, semiangle_cutoff=semiangle, defocus=0)
probe0.match_grid(potential_arr)
P0_int = np.abs(probe0.build(lazy=False).array) ** 2
ext = [-cell / 2, cell / 2, cell / 2, -cell / 2]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={"wspace": 0.35})
im = axs[0].imshow(P0_int.T, cmap="magma", extent=ext)
axs[0].set_xlim(-20, 20); axs[0].set_ylim(20, -20)
axs[0].set_title(f"In-focus probe |ψ|²\n(zoom ±20 Å; FWHM ≈ {0.51*wavelength/(semiangle*1e-3):.1f} Å)")
im = axs[1].imshow(P_int.T, cmap="magma", extent=ext)
axs[1].add_patch(Circle((0, 0), probe_diameter / 2, fill=False, ec=SERIES[2], lw=1.5, ls="--"))
axs[1].set_title(f"Probe at 1.5 µm defocus |ψ|²\ndisk Ø = 2·Δf·α ≈ {probe_diameter:.0f} Å")
mid = P_int.shape[1] // 2
xx = (np.arange(P_int.shape[0]) - P_int.shape[0] // 2) * pixel_size
axs[2].plot(xx, P_int[:, mid] / P_int.max(), color=SERIES[0])
for s in (-1, 1):
    axs[2].axvline(s * probe_diameter / 2, color=SERIES[2], lw=1.2, ls="--")
axs[2].set_xlabel("x (Å)"); axs[2].set_ylabel("|ψ|² (norm.)"); axs[2].set_title("Line profile: flat top + Fresnel fringes")
lineplot(axs[2])
ang = np.fft.fftshift(np.fft.fftfreq(res, pixel_size)) * wavelength * 1e3
Pk = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe_wave)))
phase_k = np.where(np.abs(Pk) > 0.05 * np.abs(Pk).max(), np.angle(Pk), np.nan)
im = axs[3].imshow(phase_k.T, cmap="twilight", extent=[ang[0], ang[-1], ang[-1], ang[0]])
axs[3].set_xlim(-5, 5); axs[3].set_ylim(5, -5)
axs[3].set_title(f"Probe in Fourier space: phase −χ(k)\naperture α = {semiangle:.0f} mrad"); axs[3].set_xlabel("mrad")
cbar(fig, im, axs[3], "rad")
for ax in (axs[0], axs[1]):
    ax.set_xlabel("Å"); clean(ax, ticks=True)
clean(axs[3], ticks=True)
fig.suptitle("Step 5 · The electron probe: 300 keV, 4 mrad aperture, 1.5 µm defocus", y=1.06)
caption(fig, f"λ = {wavelength:.4f} Å. A small aperture plus large defocus gives a broad (~{probe_diameter:.0f} Å) "
             "probe — gentle, low-resolution illumination like cryo-EM, and huge overlap between scan positions.")
save(fig, "step05_probe.png")

# ================================================================ 6. random orientations (ground truth)
def rotate_xyz_volume(volume_array, rot_matrix, order=3):
    tf = np.asarray(rot_matrix.T)
    shape = np.asarray(volume_array.shape)
    in_center = (shape - 1) / 2
    offset = in_center - tf @ in_center
    return scipy.ndimage.affine_transform(volume_array, tf, offset=offset, order=order, cval=3.6)


def rotated_slices(seed):
    rot = scipy.spatial.transform.Rotation.random(random_state=seed).as_matrix()
    vol = rotate_xyz_volume(embedded, rot)
    return vol.reshape(res, res, res // bin_factor_z, bin_factor_z).sum(-1).transpose(2, 0, 1)


grid_scan = abtem.GridScan((0, 0), (cell, cell), gpts=(num_probes_per_particle + 1,) * 2, endpoint=True)
step = grid_scan.sampling[0]

if DSET.exists() and GT.exists():
    print("using cached", DSET.name, GT.name)
    gt_tiles = np.load(GT)
    dataset_full = em.core.io.load(str(DSET))
    big = np.asarray(dataset_full.array)
    det_sampling = dataset_full.sampling[2]
else:
    detector = abtem.PixelatedDetector(max_angle="cutoff")
    n_pos = GRID * num_probes_per_particle + 1
    big = None
    gt_tiles = np.zeros((GRID, GRID, res, res), np.float32)
    for ix, iy in itertools.product(range(GRID), range(GRID)):
        t0 = time.time()
        seed = ix * 3 + iy                                  # same seeds as the notebook's 3x3 loop
        slices = rotated_slices(seed)
        gt_tiles[ix, iy] = slices.sum(0) * pixel_size * sigma   # projected phase (rad)
        pot = abtem.PotentialArray(slices, slice_thickness=dz, sampling=(pixel_size, pixel_size))
        pr = abtem.Probe(energy=energy, semiangle_cutoff=semiangle, defocus=defocus)
        pr.match_grid(pot)
        meas = pr.scan(potential=pot, scan=grid_scan, detectors=detector, lazy=False).array
        meas = scipy.ndimage.rotate(meas, rotation_deg, reshape=False, axes=(-2, -1), order=1)
        sx, sy, qx, qy = meas.shape
        meas = meas.reshape(sx, sy, qx // bin_factor, bin_factor, qy // bin_factor, bin_factor).sum((3, 5))
        if big is None:
            big = np.zeros((n_pos, n_pos) + meas.shape[2:], np.float32)
        p = num_probes_per_particle
        big[ix * p:(ix + 1) * p + 1, iy * p:(iy + 1) * p + 1] = meas
        print(f"particle {ix},{iy} (seed {seed}) simulated in {time.time()-t0:.0f} s", flush=True)
    det_sampling = pr.angular_sampling[0] * bin_factor
    dataset_full = em.core.datastructures.Dataset4dstem.from_array(
        big, sampling=[step, step, det_sampling, det_sampling], units=["A", "A", "mrad", "mrad"])
    dataset_full.save(str(DSET), mode="o")
    np.save(GT, gt_tiles)

GRID = gt_tiles.shape[0]
gt_mosaic = np.block([[gt_tiles[i, j] for j in range(GRID)] for i in range(GRID)])  # (x, y), 2/3 A pixels
L = GRID * cell
fig, axs = plt.subplots(1, 2, figsize=(12.5, 6), gridspec_kw={"width_ratios": [1, 1.05]})
ax = axs[0]
im = ax.imshow(gt_mosaic.T, cmap="magma", extent=[0, L, L, 0])
for i in range(1, GRID):
    ax.axvline(i * cell, color="white", lw=0.8, ls=":"); ax.axhline(i * cell, color="white", lw=0.8, ls=":")
for i, j in itertools.product(range(GRID), range(GRID)):
    ax.text(i * cell + 8, j * cell + 20, f"seed {i*3+j}", color="white", fontsize=8)
ax.set_title(f"Ground truth: projected phase of {GRID}×{GRID} randomly oriented particles")
ax.set_xlabel("Å"); clean(ax, ticks=True); cbar(fig, im, ax, "rad")
ax = axs[1]
crop = slice(int(cell * 0.2 / pixel_size), int(cell * 0.8 / pixel_size))
im = ax.imshow((gt_tiles[0, 0] - np.median(gt_tiles[0, 0]))[crop, crop].T, cmap="magma",
               extent=[cell * 0.2, cell * 0.8, cell * 0.8, cell * 0.2])
ax.set_title("Particle 'seed 0', zoom, ice background subtracted"); ax.set_xlabel("Å")
clean(ax, ticks=True); cbar(fig, im, ax, "rad")
fig.suptitle("Step 6 · Random orientations: each particle is rotated in 3D, as on a real cryo-EM grid", y=1.0)
caption(fig, "Each tile is simulated in its own 256 Å box, so dotted lines are seams between independent "
             "simulations. This ground truth is not stored in the notebook's dataset; it is saved here for comparison.",
        y=0.02)
save(fig, "step06_orientations_ground_truth.png")

# ================================================================ 7. scan geometry
fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(gt_tiles[0, 0].T, cmap="gray", extent=[0, cell, cell, 0])
g = np.arange(num_probes_per_particle + 1) * step
gx, gy = np.meshgrid(g, g)
ax.plot(gx, gy, ".", ms=3, color=SERIES[3])
for k, (cx, cy) in enumerate([(8 * step, 8 * step), (9 * step, 8 * step), (16 * step, 16 * step)]):
    ax.add_patch(Circle((cx, cy), probe_diameter / 2, fill=False, ec=SERIES[k], lw=2))
    ax.plot(cx, cy, "o", ms=6, color=SERIES[k], mec="white")
ax.annotate("", xy=(8 * step, 3 * step), xytext=(9 * step, 3 * step),
            arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
ax.text(8.5 * step, 2.2 * step, f"step {step:.2f} Å", color="white", ha="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, cell); ax.set_ylim(cell, 0); ax.set_xlabel("Å"); clean(ax, ticks=True)
ax.set_title(f"Step 7 · Scan: 25×25 positions per particle, step {step:.2f} Å\n"
             f"probe Ø ≈ {probe_diameter:.0f} Å ≈ {probe_diameter/step:.0f} steps → ~{100*(1-step/probe_diameter):.0f}% "
             "linear overlap between neighbours")
save(fig, "step07_scan_geometry.png")

# ================================================================ 8. diffraction patterns
det_n = big.shape[-1]
dext = [-det_n / 2 * det_sampling, det_n / 2 * det_sampling] * 2
dext = [dext[0], dext[1], dext[1], dext[0]]
mean_pat = big.mean((0, 1))
centre = (int(0.5 * cell / step), int(0.5 * cell / step))
edge = (1, 1)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.3), gridspec_kw={"wspace": 0.3})
im = axs[0].imshow(mean_pat ** 0.5, cmap="magma", extent=dext)
axs[0].add_patch(Circle((0, 0), semiangle, fill=False, ec=SERIES[2], lw=1.5, ls="--"))
axs[0].set_title(f"Mean pattern (√I)\ndisk radius α = {semiangle:.0f} mrad = {semiangle/det_sampling:.0f} px")
for ax, (p, lab) in zip(axs[1:3], [(centre, "probe on the particle"), (edge, "probe on ice")]):
    d = big[p] - mean_pat
    v = np.abs(big[centre] - mean_pat).max()
    ax.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v, extent=dext)
    ax.set_xlim(-6, 6); ax.set_ylim(6, -6)
    ax.set_title(f"I − mean, {lab}\n(scan {p[0]},{p[1]}; zoom ±6 mrad)")
rr = np.hypot(*np.meshgrid(np.arange(det_n) - det_n / 2, np.arange(det_n) - det_n / 2)) * det_sampling
prof = [mean_pat[(rr >= a) & (rr < a + det_sampling)].mean() for a in np.arange(0, rr.max(), det_sampling)]
axs[3].semilogy(np.arange(len(prof)) * det_sampling, prof, color=SERIES[0])
axs[3].axvline(semiangle, color=SERIES[2], lw=1.2, ls="--")
axs[3].set_xlabel("scattering angle (mrad)"); axs[3].set_ylabel("mean intensity (log)")
axs[3].set_ylim(np.max(prof) * 1e-5, np.max(prof) * 3)
axs[3].set_title("Radial profile: almost all signal inside α"); lineplot(axs[3])
for ax in axs[:3]:
    ax.set_xlabel("mrad"); clean(ax, ticks=True)
fig.suptitle(f"Step 8 · Diffraction patterns: {det_n}×{det_n} px at {det_sampling:.3f} mrad/px (after 2× binning), "
             f"rotated {rotation_deg}°", y=1.06)
caption(fig, "Patterns look almost identical everywhere: the protein changes the bright-field disk by only a few "
             "percent. The red/blue speckle (pattern minus mean) is the signal ptychography uses.", y=-0.03)
save(fig, "step08_diffraction_patterns.png")

# ================================================================ 9. final 4D dataset
yy, xx2 = np.meshgrid(np.arange(det_n) - det_n / 2, np.arange(det_n) - det_n / 2, indexing="ij")
bf_mask = np.hypot(xx2, yy) * det_sampling < semiangle
vbf = (big * bf_mask).sum((-1, -2))
com_x = (big * xx2).sum((-1, -2)) / big.sum((-1, -2))
com_y = (big * yy).sum((-1, -2)) / big.sum((-1, -2))
n_pos = big.shape[0]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"wspace": 0.3})
for ax, img, t in [(axs[0], vbf, "Virtual bright-field image\n(sum inside the disk)"),
                   (axs[1], com_x - com_x.mean(), "Centre-of-mass shift, x\n(∝ phase gradient)"),
                   (axs[2], com_y - com_y.mean(), "Centre-of-mass shift, y")]:
    im = ax.imshow(img, cmap="gray", extent=[0, n_pos * step, n_pos * step, 0])
    ax.set_title(t); ax.set_xlabel("Å"); clean(ax, ticks=True)
fig.suptitle(f"Step 9 · The saved 4D dataset: shape {tuple(big.shape)} "
             f"= (scan x, scan y, detector, detector)", y=1.05)
caption(fig, "Simple 'virtual detector' images are featureless: at 1.5 µm defocus and with weak phase contrast, "
             "the particles are nearly invisible without phase retrieval.", y=-0.02)
save(fig, "step09_dataset_virtual_images.png")

# ================================================================ 10. dose and Poisson noise
crop_n = 48
ds_crop = big[:crop_n, :crop_n]
rng = np.random.default_rng(2025)
e_per_probe = lambda d: d * step ** 2
doses = [np.inf, 1000, 100, 10]
fig, axs = plt.subplots(2, 4, figsize=(16, 7.6), gridspec_kw={"hspace": 0.35, "wspace": 0.25})
for k, dd in enumerate(doses):
    pat = ds_crop[24, 24]
    noisy = pat if np.isinf(dd) else rng.poisson(pat * e_per_probe(dd))
    tot = pat.sum() * (1 if np.isinf(dd) else e_per_probe(dd))
    axs[0, k].imshow(noisy, cmap="magma", extent=dext)
    axs[0, k].set_xlim(-7, 7); axs[0, k].set_ylim(7, -7)
    axs[0, k].set_title("no noise (infinite dose)" if np.isinf(dd)
                        else f"{dd:g} e⁻/Å²  →  ~{tot:,.0f} e⁻ per pattern")
    full = ds_crop if np.isinf(dd) else rng.poisson(ds_crop * e_per_probe(dd))
    vb = (full * bf_mask).sum((-1, -2))
    axs[1, k].imshow(vb, cmap="gray")
    axs[1, k].set_title("virtual BF" + ("" if np.isinf(dd) else f", {dd:g} e⁻/Å²"))
    clean(axs[0, k], ticks=True); axs[0, k].set_xlabel("mrad"); clean(axs[1, k])
n_bf = bf_mask.sum()
fig.suptitle(f"Step 10 · Electron dose: Poisson counting noise (notebook uses {dose} e⁻/Å² on a {crop_n}×{crop_n} crop)",
             y=0.99)
caption(fig, f"Electrons per pattern = dose × step² = {dose} × {step:.2f}² ≈ {e_per_probe(dose):,.0f}, spread over "
             f"~{n_bf:,} bright-field pixels ≈ {e_per_probe(dose)/n_bf:.1f} counts/pixel. Proteins are destroyed by "
             "higher doses, so the data is intrinsically noisy.", y=0.03)
save(fig, "step10_dose_noise.png")

# ================================================================ 11-13. reconstruction
from quantem.diffractive_imaging import DirectPtychography  # noqa: E402


def make_dp(arr):
    ds = em.core.datastructures.Dataset4dstem.from_array(
        np.asarray(arr, np.float32), sampling=[step, step, det_sampling, det_sampling],
        units=["A", "A", "mrad", "mrad"])
    return DirectPtychography.from_dataset4d(
        ds, energy=energy, semiangle_cutoff=semiangle, rotation_angle=np.deg2rad(-rotation_deg),
        aberration_coefs={"defocus": defocus}, verbose=False)


def gt_on_grid(shape):
    """Ground-truth projected phase of the cropped field, resampled to a reconstruction grid."""
    n_px = int(round(crop_n * step / pixel_size))
    g = gt_mosaic[:n_px, :n_px]
    return scipy.ndimage.zoom(g, (shape[0] / g.shape[0], shape[1] / g.shape[1]), order=1)


def aligned_r(rec, ref, smooth_A=3.0):
    """Correlation with ground truth after the same 3 A Gaussian smoothing and best integer shift."""
    sig = smooth_A / (crop_n * step / rec.shape[0])
    a = scipy.ndimage.gaussian_filter(rec, sig); b = scipy.ndimage.gaussian_filter(ref, sig)
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    cc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    sh = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    a = np.roll(a, (-sh[0], -sh[1]), axis=(0, 1))
    return float((a * b).mean())


noisy_crop = rng.poisson(ds_crop * e_per_probe(dose)).astype(np.float32)
dp_clean, dp_noisy = make_dp(ds_crop), make_dp(noisy_crop)

# 11. upsampling
fig, axs = plt.subplots(1, 4, figsize=(17, 4.5), gridspec_kw={"wspace": 0.12})
ext_c = [0, crop_n * step, crop_n * step, 0]
g4 = None
for ax, (dp, up, t) in zip(axs[:3], [(dp_clean, 1, "parallax, no upsampling"),
                                     (dp_clean, 4, "parallax, upsampling ×4"),
                                     (dp_noisy, 4, f"parallax ×4, {dose} e⁻/Å²")]):
    rec = dp.reconstruct(deconvolution_kernel="prlx", upsampling_factor=up, verbose=False).obj
    ref = gt_on_grid(rec.shape)
    g4 = ref
    ax.imshow(rec, cmap="gray", extent=ext_c)
    ax.set_title(f"{t}\n{rec.shape[0]}×{rec.shape[1]} px ({crop_n*step/rec.shape[0]:.1f} Å/px), "
                 f"r = {aligned_r(rec, ref):.2f}")
    clean(ax)
axs[3].imshow(g4, cmap="gray", extent=ext_c); axs[3].set_title("ground truth (projected phase)"); clean(axs[3])
fig.suptitle("Step 11 · Parallax (tilt-corrected bright field) with upsampling", y=1.06)
caption(fig, f"Each bright-field pixel sees the sample shifted by Δf·θ (up to {defocus*semiangle*1e-3:.0f} Å). "
             f"Those sub-step shifts let parallax rebuild detail finer than the {step:.1f} Å scan step. "
             "r = correlation with ground truth (both smoothed by 3 Å, after alignment).", y=-0.03)
save(fig, "step11_parallax_upsampling.png")

# 12. five kernels
names = ["single-sideband", "optimum bright-field", "matched filter", "parallax", "iCOM"]
recs = dp_noisy._reconstruct_all_permutations(verbose=False, upsampling_factor=2)
fig, axs = plt.subplots(2, 5, figsize=(18, 7.6), gridspec_kw={"hspace": 0.3, "wspace": 0.12})
for k, (nm, rc) in enumerate(zip(names, recs)):
    ref = gt_on_grid(rc.shape)
    axs[0, k].imshow(rc, cmap="gray", extent=ext_c)
    axs[0, k].set_title(f"{nm}\nr = {aligned_r(rc, ref):.2f}"); clean(axs[0, k])
    F = np.fft.fftshift(np.abs(np.fft.fft2(rc - rc.mean())))
    axs[1, k].imshow(F ** 0.5, cmap="magma"); axs[1, k].set_title("√|FFT|"); clean(axs[1, k])
fig.suptitle(f"Step 12 · Five direct-ptychography kernels at {dose} e⁻/Å² (upsampling ×2)", y=0.99)
caption(fig, "Top: reconstructions (r vs ground truth, 3 Å smoothing). Bottom: Fourier amplitudes; defocus rings "
             "(zeros of sin χ) are dense because Δf is huge. iCOM scores well on r only because it keeps the "
             "particles' overall mass (lowest frequencies) — it shows no internal detail.",
        y=0.04)
save(fig, "step12_five_kernels.png")

# 13. dose sweep
sweep = [10, 100, 1000, np.inf]
fig, axs = plt.subplots(2, 4, figsize=(16, 8), gridspec_kw={"hspace": 0.3, "wspace": 0.12})
rs = {"optimum bright-field": [], "parallax": []}
for k, dd in enumerate(sweep):
    arr = ds_crop if np.isinf(dd) else rng.poisson(ds_crop * e_per_probe(dd)).astype(np.float32)
    dp = make_dp(arr)
    for row, (kern, nm) in enumerate([("obf", "optimum bright-field"), ("prlx", "parallax")]):
        rc = dp.reconstruct(deconvolution_kernel=kern, upsampling_factor=2, verbose=False).obj
        r = aligned_r(rc, gt_on_grid(rc.shape)); rs[nm].append(r)
        axs[row, k].imshow(rc, cmap="gray", extent=ext_c)
        axs[row, k].set_title(f"{nm}, " + ("no noise" if np.isinf(dd) else f"{dd:g} e⁻/Å²") + f"\nr = {r:.2f}")
        clean(axs[row, k])
fig.suptitle("Step 13 · How much dose do we need? Reconstruction quality vs electron dose", y=0.98)
caption(fig, "More electrons → less Poisson noise → clearer particles. Biology sets the budget: apoferritin "
             "tolerates only tens of e⁻/Å² before radiation damage, so low-dose methods matter.", y=0.04)
save(fig, "step13_dose_sweep.png")
print("r vs dose", dict(zip([str(d) for d in sweep], zip(*rs.values()))))
