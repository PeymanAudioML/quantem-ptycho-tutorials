# Examples

## `ptycholite_example.py`: how to use quantEM's iterative ptychography API

Self-contained, CPU-only. It simulates its own data with abTEM, so it needs no downloads.

```bash
uv pip install abtem          # not in pyproject.toml
python examples/ptycholite_example.py
```

1. **Simulate** a MoS₂ monolayer with sulfur vacancies (80 keV, 20 mrad, 100 Å defocus, 0.5 Å step,
   1e5 e⁻/Å² Poisson noise). The simulation (~3.5 min) is cached in `examples/outputs/`.
2. **Reconstruct three ways**
   - `PtychoLite.from_dataset(...).reconstruct(lr_obj=..., lr_probe=...)`: the one-call API
   - `PtychoLiteDIP.from_ptycholite(...)`: U-Net (deep image prior) object and probe
   - `Ptychography.from_models(dset, ObjectPixelated, ProbePixelated, DetectorPixelated)`: the full
     modular API with `OptimizerParams`/`SchedulerParams`, constraints and a Poisson loss
3. **Save, reload, continue**: `save()`, `PtychoLite.from_file(path, dset=...)`
4. **Figures** in `examples/figures/`: reconstructions vs ground truth, loss curves, refined probe.

Pitfalls this example works around (pinned quantEM commit):
- **The DIP needs detector sizes divisible by 2^3 = 8** (3 U-Net levels), so the patterns are cropped to 72 × 72.
- **`PtychoLite.from_ptychography(...)` resets the object to uniform.** Run a few pixelated iterations on
  the copy before `PtychoLiteDIP.from_ptycholite`, or the U-Net is pretrained on a blank image.
- **Constraints are plain dicts here**, e.g. `{"object": {"tv_weight_xy": 1e-3}}`. `PtychoObjConstraintParams`,
  used in `ptycho_iter_03_constraints.ipynb`, is not exported by this version.
- **The Poisson loss is a negative log-likelihood and can be negative.** Don't normalise it by its first value.

## `tools/slice_explorer.py`: look at a multislice reconstruction slice by slice

```python
from tools.slice_explorer import as_slice_stack, save_gif, save_html_viewer
stack = as_slice_stack(ptycho_dip_free)               # or as_slice_stack(array, dz=1.0, pixel=0.157)
save_gif(stack, "slices.gif")                         # one frame per slice
save_gif(stack, "change.gif", mode="diff")            # each frame = slice k minus slice k-1
save_html_viewer(stack, "slices.html")                # slider, play button, change view; opens in any browser
```
`examples/figures/slice_explorer/` holds the three outputs for a 6-slice test reconstruction of simulated MoS₂
(not experimental data). The HTML file is self-contained, so it also works offline.
