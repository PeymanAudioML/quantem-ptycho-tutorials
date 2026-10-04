"""
Slice explorer: see how a multislice reconstruction changes from slice to slice.

Works on a quantEM reconstruction object (Ptychography / PtychoLite, any object model) or on a
plain numpy array of shape (num_slices, H, W).

    from tools.slice_explorer import as_slice_stack, plot_montage, plot_depth_profiles, \
        plot_cross_sections, slice_slider, save_gif, summarize

    stack = as_slice_stack(ptycho_dip_free)        # or as_slice_stack(array, dz=1.0, pixel=0.2)
    summarize(stack)                                # table: per-slice mean / std / integral / change
    plot_montage(stack)                             # every slice, one shared colour scale
    plot_depth_profiles(stack)                      # how the potential is distributed along z
    plot_cross_sections(stack)                      # side views (x-z and y-z) at physical aspect
    slice_slider(stack)                             # interactive slider (needs an interactive backend)
    save_gif(stack, "slices.gif")                   # animation stepping through depth

Command line (arrays saved with np.save(path, ptycho.obj_cropped)):

    python tools/slice_explorer.py recon.npy --dz 1.0 --pixel 0.157 --out slice_figs/
    python tools/slice_explorer.py --demo           # synthetic two-layer example, no data needed
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------------------------


@dataclass
class SliceStack:
    """A real-valued 3D reconstruction, axis 0 = depth (beam direction)."""

    values: np.ndarray          # (S, H, W), real
    dz: np.ndarray              # (S,) slice thickness in Angstrom
    pixel: tuple[float, float]  # (row, col) Angstrom per pixel
    label: str                  # what the values mean, used on colour bars

    @property
    def n(self) -> int:
        return self.values.shape[0]

    @property
    def z_edges(self) -> np.ndarray:
        """Slice boundaries along z, starting at 0 (top of the slab)."""
        return np.concatenate([[0.0], np.cumsum(self.dz)])

    @property
    def z_centers(self) -> np.ndarray:
        e = self.z_edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def extent_xy(self) -> list[float]:
        h, w = self.values.shape[1:]
        return [0, w * self.pixel[1], h * self.pixel[0], 0]


def as_slice_stack(
    obj,
    dz: float | np.ndarray | None = None,
    pixel: float | tuple[float, float] | None = None,
    label: str | None = None,
) -> SliceStack:
    """
    Build a SliceStack from a quantEM reconstruction or a (S, H, W) array.

    For quantEM objects the thicknesses, pixel size and object type are read from the object.
    Complex objects ("pure_phase", "complex") are converted to their phase in radians;
    "potential" objects are used as they are.
    """
    if hasattr(obj, "obj_cropped"):  # quantEM Ptychography / PtychoLite
        arr = np.asarray(obj.obj_cropped)
        obj_type = getattr(obj, "obj_type", "")
        if dz is None:
            t = np.asarray(obj.slice_thicknesses, float)
            dz = t if t.size else 1.0
        if pixel is None:
            pixel = tuple(float(x) for x in np.asarray(obj.sampling, float)[:2])
        default_label = "potential" if obj_type == "potential" else "phase (rad)"
    else:
        arr = np.asarray(obj)
        default_label = "value"
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"expected (num_slices, H, W), got shape {arr.shape}")
    if np.iscomplexobj(arr):
        arr = np.angle(arr)
        default_label = "phase (rad)"

    dz_arr = np.broadcast_to(np.asarray(1.0 if dz is None else dz, float), (arr.shape[0],)).copy()
    if pixel is None:
        pixel = 1.0
    px = (float(pixel), float(pixel)) if np.isscalar(pixel) else (float(pixel[0]), float(pixel[1]))
    return SliceStack(arr.astype(np.float64), dz_arr, px, label or default_label)


# ---------------------------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------------------------


def slice_statistics(stack: SliceStack) -> dict[str, np.ndarray]:
    """Per-slice numbers that describe how the reconstruction varies with depth."""
    v = stack.values
    flat = v.reshape(stack.n, -1)
    mean_slice = v.mean(0)
    diff = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))  # same quantity tv_weight_z penalises
    ref = (mean_slice - mean_slice.mean()).ravel()
    corr = np.array([
        np.corrcoef(f - f.mean(), ref)[0, 1] if f.std() > 0 and ref.std() > 0 else np.nan
        for f in flat
    ])
    return {
        "z": stack.z_centers,
        "mean": flat.mean(1),
        "std": flat.std(1),
        "min": flat.min(1),
        "max": flat.max(1),
        "integral": flat.sum(1) * stack.pixel[0] * stack.pixel[1],  # value * Angstrom^2
        "diff_to_next": np.append(diff, np.nan),  # mean |V[k+1] - V[k]|
        "corr_with_mean": corr,
    }


def slice_correlation_matrix(stack: SliceStack) -> np.ndarray:
    """Pearson correlation between every pair of slices (1 everywhere = identical slices)."""
    flat = stack.values.reshape(stack.n, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.corrcoef(flat), nan=0.0)


def summarize(stack: SliceStack, show: bool = True) -> dict[str, np.ndarray]:
    """Print a table of per-slice statistics and return them."""
    s = slice_statistics(stack)
    if show:
        print(f"{stack.n} slices, {stack.values.shape[1]}x{stack.values.shape[2]} px, "
              f"pixel {stack.pixel[0]:.3g} x {stack.pixel[1]:.3g} A, "
              f"total thickness {stack.z_edges[-1]:.3g} A, quantity: {stack.label}")
        head = f"{'slice':>5} {'z (A)':>7} {'mean':>10} {'std':>10} {'max':>10} {'integral':>11} {'|dV| next':>10} {'corr':>6}"
        print(head)
        for k in range(stack.n):
            d = s["diff_to_next"][k]
            print(f"{k:5d} {s['z'][k]:7.2f} {s['mean'][k]:10.4g} {s['std'][k]:10.4g} {s['max'][k]:10.4g} "
                  f"{s['integral'][k]:11.4g} {'' if np.isnan(d) else f'{d:10.4g}':>10} {s['corr_with_mean'][k]:6.2f}")
    return s


# ---------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------


def _clim(stack: SliceStack, percentile: tuple[float, float]) -> tuple[float, float]:
    lo, hi = np.percentile(stack.values, percentile)
    return float(lo), float(hi)


def _scalebar(ax, stack: SliceStack, length: float | None = None):
    h, w = stack.values.shape[1:]
    fov = w * stack.pixel[1]
    if length is None:
        length = 10 ** math.floor(math.log10(fov / 4))
        length = length * (5 if fov / 4 / length >= 5 else 2 if fov / 4 / length >= 2 else 1)
    x0, y0 = 0.05 * fov, 0.93 * h * stack.pixel[0]
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 - 0.025 * h * stack.pixel[0], f"{length:g} Å", color="white",
            ha="center", va="bottom", fontsize=8)


def plot_montage(
    stack: SliceStack,
    ncols: int | None = None,
    shared_scale: bool = True,
    percentile: tuple[float, float] = (0.5, 99.8),
    cmap: str = "magma",
    crop: tuple[slice, slice] | None = None,
    title: str | None = None,
):
    """
    One panel per slice. With shared_scale=True every panel uses the same colour range, so brightness
    differences between panels are real differences in the reconstruction.
    crop = (row_slice, col_slice) zooms into a region, e.g. (slice(10, 110), slice(10, 110)).
    """
    view = stack if crop is None else SliceStack(
        stack.values[:, crop[0], crop[1]], stack.dz, stack.pixel, stack.label)
    n = view.n
    ncols = ncols or math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    vmin, vmax = _clim(view, percentile)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.9 * ncols + 0.9, 2.9 * nrows + 0.7), squeeze=False,
                            gridspec_kw={"wspace": 0.04, "hspace": 0.22})
    for k, ax in enumerate(axs.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        kw = dict(vmin=vmin, vmax=vmax) if shared_scale else {}
        im = ax.imshow(view.values[k], cmap=cmap, extent=view.extent_xy, **kw)
        ax.set_title(f"slice {k}   z = {view.z_centers[k]:.1f} Å", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            _scalebar(ax, view)
    if shared_scale:
        cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
        cb.set_label(f"{view.label}  (shared scale)")
    fig.suptitle(title or f"All {n} slices of the reconstruction (the beam enters at slice 0)",
                 fontweight="bold")
    return fig


def plot_depth_profiles(stack: SliceStack, title: str | None = None):
    """Four views of how the reconstruction changes along z."""
    s = slice_statistics(stack)
    z, edges = s["z"], stack.z_edges
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8), gridspec_kw={"hspace": 0.38, "wspace": 0.3})

    ax = axs[0, 0]
    ax.bar(z, s["integral"], width=stack.dz * 0.9, color="#2a78d6")
    ax.set_xlabel("depth z (Å)"); ax.set_ylabel(f"integral of {stack.label} (× Å²)")
    ax.set_title("How much is in each slice\n(sum over the slice; flat = identical slices)")

    ax = axs[0, 1]
    ax.plot(z, s["mean"], "o-", color="#2a78d6", label="mean")
    ax.plot(z, s["max"], "s--", color="#eb6834", label="maximum")
    ax.fill_between(z, s["mean"] - s["std"], s["mean"] + s["std"], color="#2a78d6", alpha=0.18,
                    label="mean ± std")
    ax.set_xlabel("depth z (Å)"); ax.set_ylabel(stack.label)
    ax.set_title("Level and spread per slice"); ax.legend(frameon=False, fontsize=8)

    ax = axs[1, 0]
    if stack.n > 1:
        zi = 0.5 * (z[:-1] + z[1:])
        ax.bar(zi, s["diff_to_next"][:-1], width=0.9 * 0.5 * (stack.dz[:-1] + stack.dz[1:]), color="#1baf7a")
    ax.set_xlabel("depth z of the interface (Å)"); ax.set_ylabel(f"mean |ΔV| between neighbours")
    ax.set_title("Change between neighbouring slices\n(the quantity tv_weight_z penalises)")

    ax = axs[1, 1]
    c = slice_correlation_matrix(stack)
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r", extent=[edges[0], edges[-1], edges[-1], edges[0]])
    ax.set_xlabel("depth z (Å)"); ax.set_ylabel("depth z (Å)")
    ax.set_title("Slice-to-slice correlation\n(all red = identical slices)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for ax in axs.ravel():
        ax.grid(alpha=0.25)
    fig.suptitle(title or "Depth profile of the reconstruction", fontweight="bold")
    return fig


def plot_cross_sections(
    stack: SliceStack,
    row: int | None = None,
    col: int | None = None,
    percentile: tuple[float, float] = (0.5, 99.8),
    cmap: str = "magma",
    title: str | None = None,
):
    """
    Projection (sum over slices) with a chosen row/column, plus the side views through them.
    Default row/col = the brightest point of the projection. The depth axis is drawn at the same
    Angstrom scale as the lateral axes, so a thin slab looks thin.
    """
    proj = stack.values.sum(0)
    if row is None or col is None:
        row, col = np.unravel_index(np.argmax(proj), proj.shape)
    h, w = proj.shape
    zt = stack.z_edges[-1]
    vmin, vmax = _clim(stack, percentile)

    fig = plt.figure(figsize=(14, 5.2), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2])

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(proj, cmap=cmap, extent=stack.extent_xy)
    ax.axhline(row * stack.pixel[0] + 0.5 * stack.pixel[0], color="#1baf7a", lw=1.5)
    ax.axvline(col * stack.pixel[1] + 0.5 * stack.pixel[1], color="#eda100", lw=1.5)
    ax.set_title("Projection (sum over all slices)")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[1])
    xz = stack.values[:, row, :]  # (S, W)
    im = ax.imshow(xz, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest",
                   extent=[0, w * stack.pixel[1], zt, 0])
    for e in stack.z_edges[1:-1]:
        ax.axhline(e, color="white", lw=0.4, alpha=0.4)
    ax.set_title(f"Side view x–z  (green line, y = {row * stack.pixel[0]:.1f} Å)")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("depth z (Å)")

    ax = fig.add_subplot(gs[2])
    yz = stack.values[:, :, col]  # (S, H)
    im = ax.imshow(yz, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest",
                   extent=[0, h * stack.pixel[0], zt, 0])
    for e in stack.z_edges[1:-1]:
        ax.axhline(e, color="white", lw=0.4, alpha=0.4)
    ax.set_title(f"Side view y–z  (yellow line, x = {col * stack.pixel[1]:.1f} Å)")
    ax.set_xlabel("y (Å)"); ax.set_ylabel("depth z (Å)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=stack.label)

    fig.suptitle(title or "Where in depth does the potential sit?", fontweight="bold")
    return fig


def slice_slider(
    stack: SliceStack,
    percentile: tuple[float, float] = (0.5, 99.8),
    cmap: str = "magma",
    shared_scale: bool = True,
):
    """
    Interactive viewer: drag the slider to step through depth while a marker moves along the
    depth profile. In Jupyter use an interactive backend first (`%matplotlib widget`).
    Keep a reference to the returned figure (and slider) or it will stop responding.
    """
    s = slice_statistics(stack)
    vmin, vmax = _clim(stack, percentile)
    fig, (ax_im, ax_pr) = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
    plt.subplots_adjust(bottom=0.2, wspace=0.28)

    kw = dict(vmin=vmin, vmax=vmax) if shared_scale else {}
    im = ax_im.imshow(stack.values[0], cmap=cmap, extent=stack.extent_xy, **kw)
    ax_im.set_xlabel("x (Å)"); ax_im.set_ylabel("y (Å)")
    fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.03, label=stack.label)
    title = ax_im.set_title("")

    ax_pr.plot(s["z"], s["integral"], "o-", color="#2a78d6")
    marker, = ax_pr.plot([s["z"][0]], [s["integral"][0]], "o", ms=13, mfc="none", mec="#eb6834", mew=2)
    ax_pr.set_xlabel("depth z (Å)"); ax_pr.set_ylabel(f"integral of {stack.label} (× Å²)")
    ax_pr.set_title("Total in each slice"); ax_pr.grid(alpha=0.25)

    def show(k: int):
        k = int(k)
        im.set_data(stack.values[k])
        if not shared_scale:
            im.set_clim(*np.percentile(stack.values[k], percentile))
        title.set_text(f"slice {k} / {stack.n - 1}    z = {stack.z_centers[k]:.2f} Å    "
                       f"mean {s['mean'][k]:.3g}   max {s['max'][k]:.3g}")
        marker.set_data([s["z"][k]], [s["integral"][k]])
        fig.canvas.draw_idle()

    ax_s = fig.add_axes([0.2, 0.06, 0.6, 0.04])
    slider = Slider(ax_s, "slice", 0, stack.n - 1, valinit=0, valstep=1, valfmt="%d")
    slider.on_changed(show)
    show(0)
    fig._slice_slider = slider  # keep alive
    return fig


def save_gif(stack: SliceStack, path: str | Path, fps: float = 3, cmap: str = "magma",
             percentile: tuple[float, float] = (0.5, 99.8), shared_scale: bool = True) -> Path:
    """Animation stepping through depth (needs Pillow, which matplotlib already depends on)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    vmin, vmax = _clim(stack, percentile)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    kw = dict(vmin=vmin, vmax=vmax) if shared_scale else {}
    im = ax.imshow(stack.values[0], cmap=cmap, extent=stack.extent_xy, **kw)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=stack.label)
    title = ax.set_title("")

    def update(k):
        im.set_data(stack.values[k])
        title.set_text(f"slice {k}   z = {stack.z_centers[k]:.1f} Å")
        return im, title

    anim = FuncAnimation(fig, update, frames=stack.n, blit=False)
    path = Path(path)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def explore(stack: SliceStack, out_dir: str | Path | None = None, show: bool = False) -> list[Path]:
    """Run the table and all static plots; save them to out_dir if given."""
    summarize(stack)
    figs = {
        "slices_montage": plot_montage(stack),
        "slices_depth_profile": plot_depth_profiles(stack),
        "slices_cross_sections": plot_cross_sections(stack),
    }
    saved = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            p = out_dir / f"{name}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight")
            saved.append(p)
            print("saved", p)
    if show:
        plt.show()
    else:
        for fig in figs.values():
            plt.close(fig)
    return saved


# ---------------------------------------------------------------------------------------------
# demo + command line
# ---------------------------------------------------------------------------------------------


def synthetic_stack(n_slices: int = 12, size: int = 96, dz: float = 1.0, pixel: float = 0.25,
                    seed: int = 0) -> SliceStack:
    """Two atomic layers at different depths on a vacuum background, plus a little noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] * pixel
    vol = np.zeros((n_slices, size, size))

    def layer(shift, period=3.0):
        v = np.zeros((size, size))
        for i in range(-2, int(size * pixel / period) + 3):
            for j in range(-2, int(size * pixel / period) + 3):
                cx, cy = (i + 0.5 * (j % 2)) * period + shift[0], j * period * 0.866 + shift[1]
                v += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.35 ** 2))
        return v

    z1, z2 = int(n_slices * 0.3), int(n_slices * 0.7)
    for z, amp, shift in [(z1, 1.0, (0.0, 0.0)), (z2, 0.7, (1.0, 0.6))]:
        for dzz, w in [(-1, 0.3), (0, 1.0), (1, 0.3)]:  # a little blur in depth
            if 0 <= z + dzz < n_slices:
                vol[z + dzz] += amp * w * layer(shift)
    vol += 0.03 * rng.standard_normal(vol.shape)
    return SliceStack(vol, np.full(n_slices, dz), (pixel, pixel), "potential (arb.)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("array", nargs="?", help=".npy / .npz file holding an array of shape (slices, H, W)")
    p.add_argument("--key", help="array name inside an .npz file")
    p.add_argument("--dz", type=float, default=1.0, help="slice thickness in Angstrom (default 1)")
    p.add_argument("--pixel", type=float, default=1.0, help="Angstrom per pixel (default 1)")
    p.add_argument("--label", default=None, help="what the values are, e.g. 'potential (V·Å)'")
    p.add_argument("--out", default="slice_figs", help="folder for the figures")
    p.add_argument("--gif", action="store_true", help="also write an animated GIF")
    p.add_argument("--show", action="store_true", help="open windows (and the slider) instead of only saving")
    p.add_argument("--demo", action="store_true", help="use a synthetic two-layer stack")
    a = p.parse_args()

    if a.demo:
        stack = synthetic_stack()
    elif a.array:
        data = np.load(a.array)
        arr = data[a.key or data.files[0]] if isinstance(data, np.lib.npyio.NpzFile) else data
        stack = as_slice_stack(arr, dz=a.dz, pixel=a.pixel, label=a.label)
    else:
        p.error("give an array file or use --demo")

    saved = explore(stack, a.out, show=a.show)
    if a.gif:
        saved.append(save_gif(stack, Path(a.out) / "slices.gif"))
        print("saved", saved[-1])
    if a.show:
        slice_slider(stack)
        plt.show()


if __name__ == "__main__":
    main()
