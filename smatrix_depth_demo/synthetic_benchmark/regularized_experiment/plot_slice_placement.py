"""Schematic of the three forward-model cases in forward_slice_convergence.py (entrance, midpoint, shared).

Example: a 40 A piece of specimen cut into 2 slices of 20 A. Writes
results/forward_slice_convergence/slice_placement_schematic.png. Loss numbers are read from convergence.json.

Run from synthetic_benchmark/:  ../../.venv/bin/python regularized_experiment/plot_slice_placement.py
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "forward_slice_convergence")

DZ, NS = 20.0, 2                       # slice thickness, number of slices in the sketch
COLS = ["tab:blue", "tab:orange"]      # one colour per potential map


def loss(n, place):
    try:
        rows = json.load(open(os.path.join(OUT, "convergence.json")))["rows"]
        return next(r["quantem_loss"] for r in rows if r["slabs"] == n and r["placement"].startswith(place))
    except Exception:
        return float("nan")


def panel(ax, title, planes, start, subtitle):
    """planes: list of (z, map index, line width, alpha); start: z where the probe is defined (before any propagation)."""
    T = DZ * NS
    rng = np.random.default_rng(3)
    for k in range(NS):                                   # slices and the atoms inside them
        ax.axhspan(k * DZ, (k + 1) * DZ, color=COLS[k], alpha=0.10)
        za = rng.uniform(k * DZ + 1, (k + 1) * DZ - 1, 9)
        ax.scatter(rng.uniform(0.15, 0.55, 9), za, s=22, color="0.35", zorder=3)
        ax.text(0.03, k * DZ + DZ / 2, f"slice {k + 1}\n(map {k + 1})", color=COLS[k], fontsize=8, va="center")
    for z in np.arange(0, T + 1e-6, DZ):
        ax.axhline(z, color="0.6", lw=0.6, ls=":")
    # probe
    ax.annotate("", xy=(0.82, start), xytext=(0.82, -9), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.text(0.84, -7, "probe in", fontsize=8, va="center")
    # transmission planes
    for z, m, lw, al in planes:
        ax.plot([0.6, 0.98], [z, z], color=COLS[m], lw=lw, alpha=al, solid_capstyle="butt")
    # propagation steps
    zs = [start] + [p[0] for p in planes]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        if z1 - z0 > 1e-6:
            ax.annotate("", xy=(0.7, z1), xytext=(0.7, z0), arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.0))
    ax.set_ylim(T + 6, -12)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, T + 1, 10))
    ax.set_ylabel("depth z (Å)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.5, T + 4.5, subtitle, ha="center", va="top", fontsize=8, transform=ax.transData)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(14, 6.4), sharey=True)
    # entrance: map k applied at the top of slice k, probe defined at z = 0
    panel(ax[0], "entrance (our runs now)",
          [(k * DZ, k, 4, 1.0) for k in range(NS)], 0.0,
          f"map at slice TOP; step {DZ:g} Å\n10 × 20 Å: loss {loss(10, 'entrance'):.1f}")
    # midpoint: probe first propagated DZ/2, map k applied at the centre of slice k (= Strang splitting)
    panel(ax[1], "midpoint (= symmetric / Strang)",
          [(k * DZ + DZ / 2, k, 4, 1.0) for k in range(NS)], 0.0,
          f"map at slice CENTRE; first step {DZ / 2:g} Å, then {DZ:g} Å\n10 × 20 Å: loss {loss(10, 'midpoint'):.1f}")
    # shared: each map spread over DZ/2A sub-planes, phi/m each, propagation every 2 A
    m = int(DZ / 2)
    panel(ax[2], "shared (fine steps, same maps)",
          [(k * DZ + j * 2.0, k, 1.6, 0.9) for k in range(NS) for j in range(m)], 0.0,
          f"each map split into {m} sub-planes of φ/{m}; step 2 Å\n10 × 20 Å: loss {loss(10, 'shared'):.1f}")
    # legend
    h = [plt.Line2D([], [], color="k", lw=4, label="transmission plane: wave × exp(iφ), φ = map summed over the slice"),
         plt.Line2D([], [], color="tab:green", marker=">", lw=1, label="free-space (Fresnel) propagation"),
         plt.Line2D([], [], color="0.35", marker="o", lw=0, label="true atoms (at many depths inside a slice)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False)
    ref = loss(100, "entrance")
    fig.suptitle("How each case places a slice's potential in QuantEM's multislice (sketch: 2 slices of 20 Å). "
                 f"Losses: true potential at 10 slices of 20 Å; reference 100 slices of 2 Å = {ref:.1f}", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(OUT, "slice_placement_schematic.png"), dpi=110)
    print(os.path.join(OUT, "slice_placement_schematic.png"))


if __name__ == "__main__":
    main()
