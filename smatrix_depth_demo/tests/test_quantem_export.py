"""Tests for quantem_export: slice grid bounds, conservation of projected phase, QuantEM hand-off."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quantem_export as qe                              # noqa: E402

CANDS = [qe.DepthCandidate(y=12.0, x=12.0, modes=[(30.0, 6.0, 1.0)]),
         qe.DepthCandidate(y=10.0, x=14.0, modes=[(150.0, 8.0, 0.6), (190.0, 8.0, 0.4)], phase=0.8),
         qe.DepthCandidate(y=14.0, x=10.0, modes=[(270.0, 6.0, 1.0)])]


@pytest.mark.parametrize("dz_max,dz_min", [(20.0, 4.0), (10.0, 2.0), (30.0, 6.0)])
def test_slice_grid_bounds_and_coverage(dz_max, dz_min):
    g = qe.slice_grid(CANDS, 300.0, dz_max=dz_max, dz_min=dz_min)
    t = g.slab_thicknesses
    assert g.boundaries[0] == 0.0 and g.boundaries[-1] == 300.0
    assert t.min() >= dz_min - 1e-9 and t.max() <= dz_max + 1e-9
    assert len(g.slice_thicknesses) == g.num_slices - 1
    # denser near candidates than far from them
    near = t[np.abs(g.planes + t / 2 - 150.0) < 15].mean()
    far = t[np.abs(g.planes + t / 2 - 90.0) < 15].mean()
    assert near < far


def test_initial_potential_conserves_projected_phase_and_places_depth():
    g = qe.slice_grid(CANDS, 300.0)
    phi = qe.initial_potential(CANDS, g, (80, 80), 0.3)
    proj = phi.sum(0)
    assert abs(proj[40, 40] - 1.0) < 1e-6                 # candidate 0 at (12, 12) A = pixel (40, 40), phase 1 rad
    assert abs(proj[int(10 / 0.3 + 0.5), int(14 / 0.3 + 0.5)] - 0.8) < 0.05
    centres = g.planes + g.slab_thicknesses / 2
    prof = phi[:, 40, 40]                                  # depth profile of candidate 0 peaks near 30 A
    assert abs(centres[np.argmax(prof / g.slab_thicknesses)] - 30.0) < 5.0


def test_to_quantem_accepts_nonuniform_grid():
    pytest.importorskip("quantem")
    g = qe.slice_grid(CANDS, 300.0)
    obj = qe.to_quantem(qe.initial_potential(CANDS, g, (64, 64), 0.3), g)
    assert obj.num_slices == g.num_slices and obj.obj_type == "potential"
    np.testing.assert_allclose(obj.slice_thicknesses.cpu().numpy(), g.slice_thicknesses, atol=1e-4)
    assert tuple(obj._initial_obj.shape) == (g.num_slices, 64, 64)
