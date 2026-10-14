# M0 results: source of the depth bias and the depth reference

Script: `calibrate_depth.py` (`a`, `fit`, `report`). Setup: 300 kV, 20 mrad, dr = 0.3 Å, K = 40, Y = 80 (473 beams),
300 Å slab, one cluster of 3 Gaussian phase atoms (1 rad each) at z0, defoci −250 / −150 / −50 Å, relative probe
errors on datasets 1 and 2 (as in `demo_simple_joint.py`), 0.3 Å scan step, 10⁵ e⁻/pattern, 40 epochs. Templates:
exact entrance-referenced S of a *different* cluster every 10 Å. Depth convention: z = 0 entrance surface.

| z0 (Å) | A peak | A template | B peak | B template | B ref. plane | C peak | C template | C ref. plane |
|---|---|---|---|---|---|---|---|---|
| 30  | 30  | 30  | 20  | 28  | +5 | 15  | 27  | 0 |
| 75  | 75  | 75  | –   | –   | –  | –   | –   | – |
| 115 | 115 | 115 | –   | –   | –  | –   | –   | – |
| 150 | 150 | 150 | 145 | 149 | +5 | 135 | 146 | 0 |
| 185 | 185 | 185 | –   | –   | –  | –   | –   | – |
| 225 | 225 | 225 | –   | –   | –  | –   | –   | – |
| 270 | 270 | 270 | 270 | 273 | +5 | 260 | 268 | 0 |

A = exact S; B = S recovered with the true probe; C = S and probe recovered jointly. Peak = depth of maximum
correlation of the section phase with the true feature (5 Å grid); template = `simple_joint.fit_depth` (1 Å step);
reference plane = `simple_joint.reference_plane`, propagation distance mapping the recovered S onto the exact S
(5 Å grid).

## Conclusions

1. **Readout.** For an isolated feature, refocusing plus either readout is exact on exact S (A). The −10 / −5 / +5 Å
   offsets seen earlier on the exact S of the 3-layer sample are cross-talk between overlapping axial responses: a
   joint 3-layer template fit gives 32 / 152 / 268 Å (true 30 / 150 / 270).
2. **S recovery.** It distorts the depth response enough to shift the peak readout by up to 10 Å (B). The complex
   template fit stays within 1–3 Å.
3. **Probe estimation.** It adds up to 5 Å more to the peak readout (C) and 1–4 Å to the template fit, although this
   sparse sample leaves dataset 2's defocus 6–7 Å off (the 3-layer sample: ≤ 1.4 Å).
4. **Reference plane.** The recovered S stays at the entrance plane (0 to +5 Å, within the 5 Å search step). No gauge
   drift to correct in this setup; keep measuring it in M1.
5. **Model mismatch.** With exact-S templates, recovered-S responses leave a single-feature residual of 0.033–0.052,
   versus ≤ 0.002 on exact S. Good enough to place one known feature, but a two-feature model then fits the mismatch
   and splits one cluster into two (10 + 44 Å at z0 = 30). Counting features needs templates and a residual floor
   built through the same reconstruction pipeline (M1/M2).

Total bias with template fitting: ≤ 4 Å at 30 / 150 / 270 Å in all three tests (one noise realisation each).
