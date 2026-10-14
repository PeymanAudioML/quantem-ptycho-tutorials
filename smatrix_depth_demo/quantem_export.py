"""Turn S-matrix depth candidates into a QuantEM multislice starting point (plan stage 6 / milestone M5).

    cands  = [DepthCandidate(y=12.1, x=10.4, modes=[(148.0, 6.0, 1.0)], phase=1.0), ...]   # from template fitting
    grid   = slice_grid(cands, thickness=300.0, dz_max=20.0, dz_min=4.0)                     # nonuniform planes
    phi    = initial_potential(cands, grid, shape=(Ny, Nx), sampling=0.3)                     # (N, Ny, Nx) rad
    obj    = to_quantem(phi, grid)                                                            # ObjectPixelated

Conventions
  * Depth z = 0 is the specimen entrance surface, z = T the exit (as in simple_joint).  QuantEM's first plane is placed
    at z = 0, so a probe defocus given to QuantEM is referenced to the same entrance surface (C10 < 0 = focus inside).
  * Slab k covers [b_k, b_{k+1}); its transmission plane sits at b_k.  QuantEM gets the N - 1 propagation distances
    b_{k+1} - b_k between consecutive planes.  The last slab's thickness is not propagated; that only changes the phase
    of the exit wave's far field, not the measured intensities.
  * obj_type="potential" in QuantEM is the phase per slice in radians (transmission exp(1j * obj)), i.e.
    sigma_e * integral of V over the slab.  initial_potential returns exactly that: each candidate's total projected
    phase `phase` (rad, at its peak), spread laterally as a Gaussian and in depth by its support p(z).
  * Arrays are (slice, rows = y, cols = x).  Lateral positions are in A in the target object grid's frame; pass
    `origin` if the candidate coordinates come from a different frame (e.g. the S-matrix grid).
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DepthCandidate:
    """One lateral feature with its depth support p(z) = sum_k w_k N(z; mu_k, sigma_k^2) (A), truncated to 0..T."""
    y: float
    x: float
    modes: list                      # [(mu, sigma, weight), ...]; several modes = competing depth hypotheses
    phase: float = 1.0               # projected phase of the feature at its centre [rad], summed over depth
    lateral_sigma: float = 0.45      # Gaussian width of the feature [A]
    extra: dict = field(default_factory=dict)


def _norm_cdf(z, mu, sigma):
    return 0.5 * (1.0 + np.vectorize(math.erf)((np.asarray(z, float) - mu) / (sigma * math.sqrt(2.0))))


def support_mass(c, a, b, thickness):
    """Fraction of candidate c's depth support inside [a, b), renormalised to the specimen 0..T."""
    inside = sum(w * (_norm_cdf(thickness, m, s) - _norm_cdf(0.0, m, s)) for m, s, w in c.modes)
    part = sum(w * (_norm_cdf(b, m, s) - _norm_cdf(a, m, s)) for m, s, w in c.modes)
    return part / max(inside, 1e-30)


@dataclass
class SliceGrid:
    boundaries: np.ndarray           # (N + 1,) slab boundaries b_0 = 0 ... b_N = T
    thickness: float

    @property
    def planes(self):
        """(N,) transmission-plane depths b_0 .. b_{N-1}."""
        return self.boundaries[:-1]

    @property
    def slab_thicknesses(self):
        return np.diff(self.boundaries)

    @property
    def slice_thicknesses(self):
        """(N - 1,) propagation distances between consecutive planes, as QuantEM expects."""
        return np.diff(self.boundaries)[:-1]

    @property
    def num_slices(self):
        return len(self.boundaries) - 1


def slice_grid(candidates, thickness, dz_max=20.0, dz_min=4.0, extra_slices=4.0):
    """Nonuniform slab boundaries over 0..T from the density rho(z) = 1/dz_max + extra_slices * sum_i p_i(z),
    capped at 1/dz_min.  Every candidate adds about `extra_slices` slabs around its depth support; the baseline keeps
    every slab <= dz_max everywhere, so regions without candidates are still sampled."""
    z = np.linspace(0.0, thickness, 4001)
    rho = np.full_like(z, 1.0 / dz_max)
    for c in candidates:
        inside = sum(w * (_norm_cdf(thickness, m, s) - _norm_cdf(0.0, m, s)) for m, s, w in c.modes)
        p = sum(w * np.exp(-0.5 * ((z - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)) for m, s, w in c.modes)
        rho += extra_slices * p / max(inside, 1e-30)
    rho = np.minimum(rho, 1.0 / dz_min)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(z))])
    n = max(1, int(math.ceil(cum[-1])))
    b = np.interp(np.linspace(0.0, cum[-1], n + 1), cum, z)
    b[0], b[-1] = 0.0, thickness
    while len(b) > 2 and np.diff(b).min() < dz_min - 1e-9:      # rounding can leave a slab just under dz_min:
        k = int(np.argmin(np.diff(b)))                          # merge it with its thinner neighbour
        if k == 0:
            b = np.delete(b, 1)
        elif k == len(b) - 2:
            b = np.delete(b, k)
        else:
            b = np.delete(b, k if b[k] - b[k - 1] < b[k + 2] - b[k + 1] else k + 1)
    return SliceGrid(boundaries=b, thickness=thickness)


def initial_potential(candidates, grid, shape, sampling, origin=(0.0, 0.0), baseline=0.0):
    """(N, Ny, Nx) initial QuantEM potential (phase per slab, rad).

    Each candidate contributes phase * G((y, x) - (y_i, x_i)) * (fraction of its depth support in slab k), with G a
    unit-peak Gaussian; `baseline` [rad per A] adds a uniform phase for material no candidate accounts for.  The sum
    over slabs of a candidate's contribution equals its projected phase map (depth support conserved)."""
    Ny, Nx = shape
    yy = (np.arange(Ny) * sampling + origin[0])[:, None]
    xx = (np.arange(Nx) * sampling + origin[1])[None, :]
    b = grid.boundaries
    phi = np.zeros((grid.num_slices, Ny, Nx))
    for c in candidates:
        lat = c.phase * np.exp(-((yy - c.y) ** 2 + (xx - c.x) ** 2) / (2 * c.lateral_sigma ** 2))
        mass = np.array([support_mass(c, b[k], b[k + 1], grid.thickness) for k in range(grid.num_slices)])
        phi += mass[:, None, None] * lat[None]
    phi += baseline * grid.slab_thicknesses[:, None, None]
    return phi


def to_quantem(phi, grid, device="cpu"):
    """QuantEM ObjectPixelated (obj_type="potential") initialised with phi on `grid`'s nonuniform planes.

    QuantEM requires phi's lateral shape to equal the reconstruction's object shape (the dataset's obj_shape_full plus
    the object padding, see ObjectPixelated.from_array): render phi with initial_potential on that grid (shape,
    sampling = the dataset's object sampling, origin = where the candidates' frame sits in it).  The model only adopts
    the array once the Ptychography object is set up, so obj.obj reads (N, 1, 1) until then."""
    from quantem.diffractive_imaging import ObjectPixelated
    thick = grid.slice_thicknesses
    return ObjectPixelated.from_array(initial_obj=np.asarray(phi, np.float32),
                                      slice_thicknesses=thick.tolist() if len(thick) > 1 else
                                      (float(thick[0]) if len(thick) else None),
                                      obj_type="potential", device=device)
