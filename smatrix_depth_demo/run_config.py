"""Run the S-matrix pipeline on a harder multi-layer geometry.
usage: python run_config.py NAME PHASE_SCALE   (6 overlapping layers, 50 A spacing)
Outputs to results_NAME/ . Env MU / NITER / DOSE as in smatrix_depth_demo.py."""
import os, sys, json
import numpy as np
import smatrix_depth_demo as m

name, scale = sys.argv[1], float(sys.argv[2])
Z = [25.0, 75.0, 125.0, 175.0, 225.0, 275.0]
MOTIFS = ["ring", "square", "triangle", "plus", "X", "diamond"]


def build():
    c = m.Y * m.DR / 2
    P = {}
    ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    P["ring"] = [(c + 6 * np.sin(a), c + 6 * np.cos(a)) for a in ang] + [(c, c)]
    P["square"] = [(c + 2.6 * i, c + 2.6 * j) for i in range(-2, 3) for j in range(-2, 3)]
    v = np.array([[-6.5, -6.0], [6.5, -6.0], [0.0, 6.5]]); tri = []
    for a in range(3):
        for t in np.linspace(0, 1, 6, endpoint=False):
            p = v[a] * (1 - t) + v[(a + 1) % 3] * t; tri.append((c + p[0], c + p[1]))
    P["triangle"] = tri
    P["plus"] = [(c + 2.6 * k, c) for k in range(-2, 3)] + [(c, c + 2.6 * k) for k in (-2, -1, 1, 2)]
    P["X"] = [(c + 2.4 * k, c + 2.4 * k) for k in range(-2, 3)] + [(c + 2.4 * k, c - 2.4 * k) for k in (-2, -1, 1, 2)]
    d = np.array([[0, -6.5], [6.5, 0], [0, 6.5], [-6.5, 0]]); dia = []
    for a in range(4):
        for t in np.linspace(0, 1, 4, endpoint=False):
            p = d[a] * (1 - t) + d[(a + 1) % 4] * t; dia.append((c + p[1], c + p[0]))
    P["diamond"] = dia
    return [m.atoms_to_phase(P[k], [scale] * len(P[k])) for k in MOTIFS]


m.OUT = os.path.join(m.HERE, f"results_{name}")
os.makedirs(m.OUT, exist_ok=True)
m.LAYER_Z = Z
m.build_layers = build
json.dump(dict(name=name, phase_scale=scale, layer_z=Z, motifs=MOTIFS,
               mu=m.MU, niter=m.NITER, dose=m.DOSE), open(os.path.join(m.OUT, "config.json"), "w"))
m.main()
