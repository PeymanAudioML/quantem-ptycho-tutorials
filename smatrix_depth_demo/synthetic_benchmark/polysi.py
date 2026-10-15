"""Polycrystalline silicon specimens: periodic Voronoi grains of rotated diamond-cubic Si.

Construction (all lengths in A; lab frame: axis 0 = ase x, axis 1 = ase y, z = depth below the entrance, beam along +z):
  * Grain g has a seed s_g (lab) and a rotation R_g (lab = R_g @ crystal): crystal direction `zone` -> lab +z, then an
    in-plane rotation `inplane_deg` about the beam.  R_g is saved explicitly.
  * Grain cells: periodic Voronoi in x, y (cell L x L) with the metric `metric` = (1, 1, 0) for columnar grains
    (P3-XY: boundaries parallel to the beam) or (1, 1, 1) for 3-D grains (P3-XYZ).  Each grain's lattice is generated on
    the tile s_g +- L/2 laterally and kept where the nearest seed instance (over all seeds and their 3 x 3 lateral
    images) is s_g itself, then wrapped into [0, L).  The union tiles the periodic cell exactly and every grain's
    lattice is continuous inside its own cell (no wraparound seam inside a grain).
  * Pairs closer than `d_min` (periodic in x, y) across different cells are pruned: the atom farther from its own seed
    is removed, repeated until no pair is left.  Unrelaxed: no interatomic potential is applied (do not read the
    boundaries as equilibrated structures).
  * Per atom: grain id, distance to the nearest Voronoi bisector in the metric (`dist_gb`, a lower bound for the
    distance to the grain boundary), and whether it is a self-boundary (cell of s_g meeting an image of s_g).
"""
import itertools, math
import numpy as np

A_SI = 5.431
BASIS = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                  [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])     # diamond, conventional cell


def rotation(zone, inplane_deg, ref=(1, 0, 0)):
    """Proper rotation R (lab = R @ crystal) taking the crystal direction `zone` to lab +z, then rotating by
    inplane_deg about lab z."""
    z = np.asarray(zone, float)
    z /= np.linalg.norm(z)
    r = np.asarray(ref, float)
    if abs(r @ z) > 0.9:
        r = np.array([0.0, 1.0, 0.0])
    x = r - (r @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R0 = np.stack([x, y, z])                       # rows: crystal-frame components of the lab axes
    t = math.radians(inplane_deg)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    R = Rz @ R0
    assert abs(np.linalg.det(R) - 1) < 1e-9
    return R


def _instances(seeds, L):
    """Seed instances: (n_seeds * 9, 3) positions with lateral images, owner index, image shift."""
    pos, owner, shift = [], [], []
    for i, s in enumerate(seeds):
        for a, b in itertools.product((-1, 0, 1), repeat=2):
            pos.append(s + np.array([a * L, b * L, 0.0]))
            owner.append(i)
            shift.append((a, b))
    return np.array(pos), np.array(owner), np.array(shift)


def _nearest(p, inst, w):
    d2 = (((p[:, None, :] - inst[None, :, :]) ** 2) * w).sum(-1)
    return d2


def build(L, T, grains, metric=(1, 1, 1), d_min=2.0, z_eps=1e-3, seed=0):
    """grains: list of dict(seed=[x, y, z], zone=[u, v, w], inplane_deg=float).  Returns positions (N, 3), grain id (N,),
    dist_gb (N,), self_boundary (N,) bool, and an info dict (rotations, seeds, pruning and density statistics)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    w = np.asarray(metric, float)
    seeds = np.array([g["seed"] for g in grains], float)
    inst, owner, shift = _instances(seeds, L)
    P, G, DG, SB, DS, U = [], [], [], [], [], []
    Rs, offsets = [], []
    for g, gr in enumerate(grains):
        R = rotation(gr["zone"], gr.get("inplane_deg", 0.0))
        off = rng.uniform(0, A_SI, 3)              # lattice translation (reproducible)
        Rs.append(R)
        offsets.append(off)
        lo = seeds[g] - np.array([L / 2, L / 2, 0.0])
        lo[2] = 0.0
        hi = lo + np.array([L, L, T])
        corners = np.array(list(itertools.product(*zip(lo, hi))))
        cc = (corners - off) @ R                   # crystal coords of the tile corners (R orthogonal: R^T v = v @ R)
        nmin = np.floor(cc.min(0) / A_SI).astype(int) - 1
        nmax = np.ceil(cc.max(0) / A_SI).astype(int) + 1
        n = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(nmin, nmax)], indexing="ij"), -1).reshape(-1, 3)
        frac = (n[:, None, :] + BASIS[None]).reshape(-1, 3)
        p = frac * A_SI @ R.T + off                # lab = R @ crystal + off
        keep = np.all((p >= lo) & (p < hi), axis=1) & (p[:, 2] >= z_eps) & (p[:, 2] <= T - z_eps)
        p = p[keep]
        for c0 in range(0, len(p), 200000):        # nearest seed instance must be (g, image 0)
            q = p[c0:c0 + 200000]
            d2 = _nearest(q, inst, w)
            k = np.argmin(d2, axis=1)
            mine = (owner[k] == g) & (shift[k] == 0).all(1)
            q = q[mine]
            d2 = d2[mine]
            ka = k[mine]
            a = inst[ka]
            # distance to the nearest bisector (metric), over all other instances
            num = d2 - d2[np.arange(len(d2)), ka][:, None]
            den = 2 * np.sqrt((((inst[None] - a[:, None]) ** 2) * w).sum(-1))
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.where(den > 0, num / den, np.inf)
            dist[np.arange(len(d2)), ka] = np.inf
            j = np.argmin(dist, axis=1)
            U.append(q.copy())                     # unwrapped (inside the grain's own tile)
            q = q.copy()
            q[:, :2] %= L
            P.append(q)
            G.append(np.full(len(q), g))
            DG.append(dist[np.arange(len(q)), j])
            SB.append(owner[j] == g)               # nearest bisector is with an image of the same grain
            DS.append(np.sqrt(d2[np.arange(len(q)), ka]))
    P, G, DG, SB, DS, U = map(np.concatenate, (P, G, DG, SB, DS, U))
    n0 = len(P)
    # prune cross-grain pairs closer than d_min (periodic in x, y)
    removed = 0
    for _ in range(20):
        box = [L, L, 10 * T + 10]
        tree = cKDTree(np.c_[P[:, :2] % L, P[:, 2]], boxsize=box)
        pairs = tree.query_pairs(d_min, output_type="ndarray")
        if len(pairs) == 0:
            break
        # across a boundary: different grains, or the same grain only close through the periodic wrap (self-boundary)
        cross = (G[pairs[:, 0]] != G[pairs[:, 1]]) | (np.linalg.norm(U[pairs[:, 0]] - U[pairs[:, 1]], axis=1) > d_min)
        pairs = pairs[cross]
        if len(pairs) == 0:
            break
        drop = np.where(DS[pairs[:, 0]] > DS[pairs[:, 1]], pairs[:, 0], pairs[:, 1])
        keep = np.ones(len(P), bool)
        keep[np.unique(drop)] = False
        removed += int((~keep).sum())
        P, G, DG, SB, DS, U = P[keep], G[keep], DG[keep], SB[keep], DS[keep], U[keep]
    tree = cKDTree(np.c_[P[:, :2] % L, P[:, 2]], boxsize=[L, L, 10 * T + 10])
    dnn, _ = tree.query(np.c_[P[:, :2] % L, P[:, 2]], k=2)
    nn = dnn[:, 1]
    vol = L * L * T
    bulk = 8 / A_SI ** 3
    info = dict(
        cell_A=[L, L, T], metric=list(map(float, w)), d_min_A=d_min, relaxed=False, lattice_constant_A=A_SI,
        n_atoms=int(len(P)), n_generated=int(n0), n_pruned=int(removed),
        density_per_A3=float(len(P) / vol), bulk_density_per_A3=float(bulk),
        nearest_neighbour_A=dict(min=float(nn.min()), p01=float(np.percentile(nn, 1)), median=float(np.median(nn)),
                                 bulk=float(A_SI * math.sqrt(3) / 4)),
        grains=[dict(id=g, seed=list(map(float, seeds[g])), zone=list(grains[g]["zone"]),
                     inplane_deg=float(grains[g].get("inplane_deg", 0.0)), rotation_lab_from_crystal=Rs[g].tolist(),
                     lattice_offset_A=offsets[g].tolist(), n_atoms=int((G == g).sum()),
                     n_self_boundary_atoms_within_3A=int(((G == g) & SB & (DG < 3)).sum()))
                for g in range(len(grains))],
        boundary=dict(n_atoms_within_1A=int((DG < 1).sum()), n_atoms_within_3A=int((DG < 3).sum())))
    return P, G, DG, SB, info


def density_map(P, L, T, cube=6.0):
    """Atom counts in cubes of side ~cube (lateral periodic); returns relative density (count / bulk expectation)."""
    nx = max(1, int(round(L / cube)))
    nz = max(1, int(round(T / cube)))
    H, _ = np.histogramdd(np.c_[P[:, 0] % L, P[:, 1] % L, P[:, 2]], bins=(nx, nx, nz), range=((0, L), (0, L), (0, T)))
    expect = 8 / A_SI ** 3 * (L / nx) * (L / nx) * (T / nz)
    return H / expect


def field_coverage(P, G, DG, n_grains, lo, hi):
    """Fraction of atoms of each grain inside the lateral box [lo, hi) (scanned field), and boundary atoms there."""
    m = np.all((P[:, :2] >= lo) & (P[:, :2] < hi), axis=1)
    tot = m.sum()
    return dict(n_atoms_in_field=int(tot),
                grain_fraction_in_field=[float(((G == g) & m).sum() / max(tot, 1)) for g in range(n_grains)],
                boundary_atoms_within_2A_in_field=int((m & (DG < 2)).sum()))


def distance_to_other_grains(P, G, L, T):
    """Euclidean distance [A] from each atom to the nearest atom of a different grain (lateral periodic).  Use this,
    not dist_gb, for boundary masks: dist_gb is measured in the construction metric (z weighted for P3-XYZ)."""
    from scipy.spatial import cKDTree
    Q = np.c_[P[:, :2] % L, P[:, 2]]
    out = np.full(len(P), np.inf)
    for g in np.unique(G):
        other = cKDTree(Q[G != g], boxsize=[L, L, 10 * T + 10])
        m = G == g
        out[m] = other.query(Q[m], k=1)[0]
    return out
