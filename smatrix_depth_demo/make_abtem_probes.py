"""Step 1 for make_poly_abtem.py: entrance probes (D, Y, Y) [row=y, col=x], unit norm, for a probe-error config of run_joint_probe.py
(A = nominal, B = relative defocus + astigmatism errors on datasets 1, 2, ...).   usage: python make_abtem_probes.py [config=B] [n_defoci=3] [setup=a20]   (setups: geom_spec.py)
-> results_poly/abtem_probes_{config}.npz  (probes, coeff_true, coeff_nominal)"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_joint_probe as r, joint_smatrix_probe as jp
import geom_spec as gs
config = sys.argv[1] if len(sys.argv) > 1 else "B"
setup = sys.argv[3] if len(sys.argv) > 3 else "a20"; sp = gs.spec(setup)
D = int(sys.argv[2]) if len(sys.argv) > 2 else 3          # number of defocus datasets, evenly spaced in -150..0 A (3 -> -150, -75, 0)
g = jp.Geometry(alpha=sp["alpha"], dr=sp["dr"], K=sp["K"], M=sp["M"], nscan=sp["nscan"], step=sp["step"])
DEFOCI = [float(v) for v in np.linspace(-150.0, 0.0, D)]
nominal, true = r.probe_config(DEFOCI, config)
probes = []
for d in range(len(true)):
    p = np.fft.ifft2(jp.probe_on_grid(list(true[d]), r.NAMES, g.Y, g.Y, g.dr, g.lam, g.alpha))
    probes.append((p / np.sqrt(np.sum(np.abs(p) ** 2))).astype(np.complex64))
np.savez(os.path.join(HERE, "results_poly", f"abtem_probes_{config}{gs.suffix(setup)}.npz"), probes=np.array(probes), coeff_true=true, coeff_nominal=nominal)
print("saved probes", np.array(probes).shape, "true coeffs (A):\n", np.round(true, 1))
