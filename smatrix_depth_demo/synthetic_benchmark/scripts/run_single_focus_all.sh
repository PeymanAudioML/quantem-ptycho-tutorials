#!/usr/bin/env bash
# Single-focus S-matrix with joint probe on ALL patterns of P3_XY dataset 1 (no held-out split), plots, then the QuantEM handoff:
# object, probe and (arm E) slice grid from this S-matrix.  Same settings as run_single_focus_joint.sh except --heldout 0.
# Run from synthetic_benchmark/:  nohup bash scripts/run_single_focus_all.sh > single_focus_all.log 2>&1 &
set -u
PY=../../.venv/bin/python
CFG=configs/poly_si_3grain.yaml
S=smatrix_window_noiseless_s0_sfall_single_d1_joint
BND=quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfQ     # uniform 35-slab boundaries for the S-matrix slice plot
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }

run $PY run_benchmark.py smatrix --config $CFG --specimen P3_XY --single-dataset --dataset 1 --allow-underdetermined \
    --epochs 80 --warmup 8 --batch 16 --lr-S 0.02 --mode simultaneous --heldout 0.0 --split-seed 0 \
    --select-every 1 --checkpoint-every 10 --monitor-every 10 --device cuda --verbose \
    --seeds 0 --tag sfall --single-probe-mode joint --trainable C10,C12a,C12b --lr-probe 0.005
run $PY plot_smatrix_slices.py --config $CFG --specimen P3_XY --run $S --quantem-run $BND
run $PY run_benchmark.py quantem --config $CFG --specimen P3_XY --dataset 1 --smatrix-run $S --single-focus-handoff \
    --arms E,F --probes smatrix_trained --num-slices 35 --iters 200 --probe-warmup 40 \
    --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --tag sfallQ --device cuda
for arm in E F; do
  run $PY plot_quantem.py --config $CFG --specimen P3_XY --run quantem_window_noiseless_d1_${arm}_smatrix_trained_call_s0_sfallQ
done
echo "=== $(date -Is) all done"
