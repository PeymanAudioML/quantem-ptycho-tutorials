"""Plot saved atomic ground truth without running simulations or using the GPU."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantem-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ase.io import read


def plot_structure(ground_truth, out):
    ground_truth, out = Path(ground_truth), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atoms = read(ground_truth / "atoms.extxyz")
    p = atoms.positions
    grain = atoms.arrays["grain"]
    info = json.loads((ground_truth / "polycrystal.json").read_text())
    cell = np.asarray(info["cell_A"])
    ids = np.unique(grain)
    colors = {g: plt.get_cmap("tab10")(i) for i, g in enumerate(ids)}
    labels = {}
    for g in ids:
        record = next(r for r in info["grains"] if r["id"] == g)
        zone = " ".join(map(str, record["zone"]))
        labels[g] = f"Grain {g + 1}: [{zone}], {np.sum(grain == g):,} atoms"
    legend = [Line2D([], [], marker="o", linestyle="", color=colors[g], label=labels[g]) for g in ids]
    title = f"{ground_truth.parent.name} • {len(p):,} Si atoms • {len(ids)} grains"

    def scatter(ax, mask, axes=(0, 1), size=2, alpha=.65):
        for g in ids:
            q = p[mask & (grain == g)]
            ax.scatter(q[:, axes[0]], q[:, axes[1]], s=size, c=[colors[g]],
                       alpha=alpha, linewidths=0, rasterized=True)
        ax.set(xlim=(0, cell[axes[0]]), ylim=(0, cell[axes[1]]),
               xlabel=f"{'xyz'[axes[0]]} (Å)", ylabel=f"{'xyz'[axes[1]]} (Å)")
        ax.set_aspect("equal")

    fig, axs = plt.subplots(1, 3, figsize=(13, 9), constrained_layout=True)
    for ax, axes, name in zip(axs, [(0, 1), (0, 2), (1, 2)],
                              ["x–y projection (along beam)", "x–z projection", "y–z projection"]):
        scatter(ax, np.ones(len(p), bool), axes, size=1, alpha=.35)
        ax.set_title(name)
    fig.suptitle(title + "\nAll atoms projected; overlaps obscure atoms behind them")
    fig.legend(handles=legend, loc="outside lower center", ncol=len(ids))
    fig.savefig(out / "atomic_projections_2d.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 11), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for g in ids:
        q = p[grain == g]
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=1, c=[colors[g]], alpha=.35, linewidths=0)
    ax.set(xlabel="x (Å)", ylabel="y (Å)", zlabel="Depth z (Å)",
           xlim=(0, cell[0]), ylim=(0, cell[1]), zlim=(0, cell[2]))
    ax.set_box_aspect(cell.copy())
    ax.view_init(elev=17, azim=-55)
    ax.set_title(title + "\n3D atomic coordinates (true cell proportions)")
    fig.legend(handles=legend, loc="outside lower center")
    fig.savefig(out / "atomic_structure_3d.png", dpi=180)
    plt.close(fig)

    edges = np.linspace(0, cell[2], 9)
    fig, axs = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
    for ax, lo, hi in zip(axs.flat, edges[:-1], edges[1:]):
        mask = (p[:, 2] >= lo) & (p[:, 2] < hi)
        scatter(ax, mask, size=3, alpha=.6)
        ax.set_title(f"{lo:g} ≤ z < {hi:g} Å\n{mask.sum():,} atoms")
    fig.suptitle(title + "\nDepth slabs: all atoms in each 25 Å slab, projected onto x–y")
    fig.legend(handles=legend, loc="outside lower center", ncol=len(ids))
    fig.savefig(out / "atomic_depth_slabs.png", dpi=180)
    plt.close(fig)

    centers = [r["seed"][2] for r in info["grains"]]
    fig, axs = plt.subplots(1, len(centers), figsize=(5 * len(centers), 6),
                            constrained_layout=True, squeeze=False)
    for ax, z in zip(axs.flat, centers):
        mask = (p[:, 2] >= z - 2) & (p[:, 2] < z + 2)
        scatter(ax, mask, size=10, alpha=.9)
        ax.set_title(f"{z - 2:g} ≤ z < {z + 2:g} Å\n{mask.sum():,} atoms")
    fig.suptitle(title + "\nThin 4 Å slices centered on the saved grain-seed depths")
    fig.legend(handles=legend, loc="outside lower center", ncol=len(ids))
    fig.savefig(out / "atomic_thin_slices.png", dpi=180)
    plt.close(fig)
    for path in sorted(out.glob("*.png")):
        print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ground_truth", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plot_structure(args.ground_truth, args.out)
