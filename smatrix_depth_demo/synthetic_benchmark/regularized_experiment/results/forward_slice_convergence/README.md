# QuantEM forward-model error vs slice thickness (P3_XYZ, dataset 1)

Ground truth only, no reconstruction. Code: [`../../forward_slice_convergence.py`](../../forward_slice_convergence.py)
(2026-10-18, GPU, under 1 min). Numbers: [`convergence.md`](convergence.md), [`convergence.json`](convergence.json);
plot: [`convergence.png`](convergence.png).

The true potential, summed into N uniform slabs, is pushed through QuantEM's own forward path with the true probe. The
result is compared with the abTEM patterns (2 Å slices). The loss is QuantEM's own loss, so it can be compared directly
with training losses.

## How QuantEM propagates (quantem 0f60f90)

- **Scheme:** conventional first-order multislice: transmit exp(iφ_k) at plane k, then apply the Fresnel propagator
  exp(−iπλ dz_k k²) to the next plane.
- **Steps:** N slices use N − 1 propagators. There is no half step at entry or exit, and no anti-aliasing aperture.
  - Code: `ptychography_base.overlap_projection` and `probe_models._compute_propagator_arrays`.
- **Planes in our runner:** plane k sits at the slab entrance b_k, and the probe is defined at the entrance surface
  (`quantem_export.py` docstring). Each slab's potential therefore acts at its top surface.
- **Slab content:** each slab is one 2D transmission of the potential summed over its depth. There is no propagation
  inside a slab.

## Cases

| case | what it is |
|---|---|
| entrance | as our runs do: plane at slab top, probe at the entrance surface |
| midpoint | probe first propagated dz/2, so each slab acts at its centre. For uniform slabs this is exactly symmetric (Strang) splitting, because the last half step does not change intensities |
| shared ×m | the same N slab potentials, each spread evenly over m sub-steps of about 2 Å: finer propagation, the same N maps |

## Results (QuantEM loss; vacuum 583)

| slabs | dz | entrance | midpoint | shared |
|---|---|---|---|---|
| 100 | 2 Å | 17.9 | 18.2 | — |
| 35 | 5.7 Å | 20.9 | 20.4 | 19.0 |
| 20 | 10 Å | 24.3 | 21.3 | 18.2 |
| 10 | 20 Å | 92.5 | 75.9 | 16.6 |
| 5 | 40 Å | 288 | 220 | 23.4 |

## Findings

1. **The floor is not the slicing.** At 2 Å slices the loss is still 17.9. An independent torch multislice with
   abTEM's 2/3 anti-aliasing reaches L_data 2.1e-3 vs 6.4e-3 here (`../../../results/forward_validation/`). The rest
   comes from resampling to QuantEM's 0.2 Å grid and from having no anti-aliasing aperture.
2. **There are two separate slicing errors.**
   - **(a) Plane placement:** a dz/2 defocus offset. It affects only the bright field, and midpoint (Strang) placement
     removes it completely: bright field stays at 2.6e-3 up to 20 Å. A trained probe can also absorb it through C10.
   - **(b) No propagation inside a slab:** this affects only the dark field. Strang splitting does not fix it. It is
     small up to 10 Å and blows up between 10 and 20 Å (dark field 0.07 → 0.35).
3. **Fine propagation with shared maps removes (b).** Ten maps of 20 Å, each propagated in 2 Å sub-steps, fit as well as
   100 true 2 Å slices (16.6 vs 17.9). Forward accuracy then no longer limits the number of potential maps, down to
   about 20 Å maps. The even spreading is an assumption, but it costs little here because the depth resolution is
   about 49 Å.
4. **The reconstructions are already below the true object's loss.** The 20-slice arm F runs reach data loss 15.9–17.1
   (`../tvz_sweep/`). The true object at 20 entrance-placed slices scores 24.3. The trained probe and the object
   therefore absorb about 7 loss units of forward-model error. A lower data loss does not mean a more correct depth
   structure below about 18–24.
