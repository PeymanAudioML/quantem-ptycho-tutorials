#!/usr/bin/env bash
# Progress of scripts/run_single_focus_joint.sh (its Python output is buffered, so single_focus_joint.log lags).
# Run from synthetic_benchmark/:  bash scripts/sf_progress.sh
cd "$(dirname "$0")/.."
date -u +"now %H:%M:%S UTC"
echo "--- queue steps started / finished"; grep "^===" single_focus_joint.log | cut -c5-130
echo "--- S-matrix monitor snapshots (every 10 epochs): epoch, running loss, C10 C12a C12b, focus-curve layers"
../../.venv/bin/python - <<'PY'
import glob, os, time, numpy as np
for d in sorted(glob.glob("runs/poly_si_3grain/P3_XY/smatrix_*sf80*")):
    fs = sorted(glob.glob(d + "/monitor/epoch_*.npz"))
    done = os.path.exists(d + "/summary.json")
    if not fs:
        print(os.path.basename(d), "started, no snapshot yet"); continue
    z = np.load(fs[-1])
    print(f"{os.path.basename(d):55s} epoch {int(z['epoch']):2d}/80 {'DONE' if done else ''} "
          f"loss {float(z['loss'][-1]):.5f}  C {np.round(z['coefficients'][0, :3], 2).tolist()}  "
          f"layers {z['layers'].tolist()}  ({time.strftime('%H:%M', time.gmtime(os.path.getmtime(fs[-1])))})")
PY
echo "--- QuantEM runs finished"; ls results/quantem/poly_si_3grain/P3_XY 2>/dev/null | grep sfQ || echo "(none yet)"
