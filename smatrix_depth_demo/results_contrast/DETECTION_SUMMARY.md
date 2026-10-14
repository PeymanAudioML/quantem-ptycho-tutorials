# Lateral feature detection (simple_joint.detect_features), tested on the contrast-scan reconstructions

Strength map = max over depth of |phase| of the normalised refocused sections; peaks above median + k * MAD inside the
scanned region (6.4 A field, 100 A slab, 0.4 A scan step, 40 epochs). Match = within 0.6 A of a true Au atom.

| specimen | Au atoms found (k = 6) | false peaks |
|---|---|---|
| vacuum, 3 / 6 Au | 9 / 9, 18 / 18 | 0 |
| vacuum, 12 Au | 9 / 36 (most atoms lie outside the 6.4 A scanned field) | 0 |
| carbon 0.025-0.1, all sizes | 0-6 per combination (0-33 %) | 0 (k = 6); 23 in total at k = 4 |

- In carbon, Au atoms are not distinct in the reconstructed phase: percentile of the map at the true Au positions
  91-99 (3 Au, 0.025), 4-100 (3 Au, 0.1), 34-100 (6 Au, 0.05), even in the section at the feature's own depth.
- Cluster-level detection (1.5 A smoothing) puts the strongest peak on the cluster in 31 / 36, but carbon-only data
  (no gold) also peak at the field centre in 18 / 18: it finds the best-reconstructed centre, not the gold.
- Consequence: at this geometry the S-matrix stage cannot locate heavy atoms laterally inside a carbon matrix; the
  carbon depth fits (M3, contrast scan) used the true footprint. Detection is usable for isolated features.
- Untested levers: larger scanned field (context, less centre bias), more epochs, smaller scan step, higher dose.
