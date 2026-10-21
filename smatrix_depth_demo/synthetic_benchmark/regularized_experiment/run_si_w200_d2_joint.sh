#!/usr/bin/env bash
# Single-dataset (def2, 80 A) S-matrix fit with the probe (C10, C12a, C12b) updated jointly with S (no fixed gauge).
# Same settings as the defocus study's k1_d2 fit (80 epochs, warm-up 8, batch 8, lr_S 0.02, all patterns), except the
# probe is trainable.  Two starts: nominal C10 = -80 A (= truth here) and a wrong C10 = -100 A (can the data correct it?).
# Waits for the QuantEM chain (PID given as $1) because both need the GPU.
set -u
PY=../../.venv/bin/python
CFG=results/defocus_study/si_defocus_series_w200_window_noiseless/configs/k1_d2.yaml
COMMON="--config $CFG --specimen k1_d2 --single-dataset --dataset 0 --single-probe-mode joint --allow-underdetermined \
  --epochs 80 --warmup 8 --batch 8 --lr-S 0.02 --heldout 0.0 --dz 2.0 --checkpoint-every 5 --device cuda"
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
run $PY run_benchmark.py smatrix $COMMON --tag jointC80
run $PY run_benchmark.py smatrix $COMMON --probe-init C10=-100 --tag jointC100
echo "=== $(date -Is) all done"
