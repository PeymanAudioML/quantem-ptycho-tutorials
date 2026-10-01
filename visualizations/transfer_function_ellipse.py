"""Transfer functions with rotation passed as -13 (notebook, read as radians) vs deg2rad(-13), with predicted sin(chi)=0 ellipses.

    python visualizations/transfer_function_ellipse.py   # run from repo root
"""
import numpy as np, matplotlib.pyplot as plt
# Reuses the simulation block of white_noise_steps.py.
src = open("visualizations/white_noise_steps.py").read().split("# ================================================================ step 1")[0]
exec(src)
from quantem.diffractive_imaging import DirectPtychography
ds = em.core.datastructures.Dataset4dstem.from_array(data4d, sampling=[sampling]*2+[reciprocal_sampling]*2, units=["A","A","A^-1","A^-1"])
names = ["single-sideband","optimum-bright-field","matched-filter","parallax-imaging","center-of-mass-imaging"]
rows = [("rotation_angle = -13  (as in notebook → read as −13 rad)", -13.0),
        ("rotation_angle = deg2rad(-13)  (correct)", np.deg2rad(-13))]
# predicted first zero of sin chi(q): chi = pi*lam*q^2*(C10 + C12 cos 2(theta-phi12)) = pi
th = np.linspace(0, 2*np.pi, 721)
Ceff = aberrations["C10"] + aberrations["C12"]*np.cos(2*(th - aberrations["phi12"]))
q1 = np.sqrt(1/(wavelength*Ceff)); q2 = np.sqrt(2/(wavelength*Ceff))
print("first zero along phi12: %.3f 1/A, perpendicular: %.3f 1/A, ratio %.2f" % (q1.min(), q1.max(), q1.max()/q1.min()))
fig, axs = plt.subplots(2, 5, figsize=(17, 7.6))
ext = [-k_max, k_max, k_max, -k_max]
for r, (lab, rot) in enumerate(rows):
    d = DirectPtychography.from_dataset4d(ds, energy=energy, semiangle_cutoff=k_probe*wavelength*1e3, rotation_angle=rot, aberration_coefs=aberrations, verbose=False)
    recons = d._reconstruct_all_permutations(verbose=False)
    for c, (nm, rc) in enumerate(zip(names, recons)):
        ax = axs[r, c]
        ax.imshow(np.fft.fftshift(np.abs(np.fft.fft2(rc))), cmap="magma", vmin=0, vmax=1, extent=ext)
        if c == 3:
            # overlay predicted zeros in (row=qx, col=qy) display coordinates
            for qq, ls in [(q1, "-"), (q2, "--")]:
                ax.plot(qq*np.sin(th), qq*np.cos(th), color="#1baf7a", lw=1.2, ls=ls)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(nm, fontsize=10)
    axs[r, 0].set_ylabel(lab, fontsize=9.5)
fig.suptitle("|FFT(reconstruction)| — top: notebook as written, bottom: rotation in radians.\n"
             "Green on parallax: predicted zeros of sin χ(q) = 0 (χ = π solid, 2π dashed)", fontsize=12, fontweight="bold")
fig.savefig("visualizations/white_noise_steps/transfer_function_ellipse.png", dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
