#!/usr/bin/env bash
# Single-focus S-matrix: fixed vs joint probe (P3_XY dataset 1), then the QuantEM handoff arms Q0-Q3.
# Every S-matrix run: same data, held-out split (10 %, split seed 0), vacuum S start, 80 epochs, 8 warm-up epochs,
# simultaneous updates, lr_S 0.02 (cosine), batch 16, best epoch by a fresh full-training-set loss every epoch.
# Joint runs: C10, C12a, C12b trained with lr_probe 0.005 (simple_joint.fit's simultaneous default).
# Run from synthetic_benchmark/:  nohup bash scripts/run_single_focus_joint.sh > single_focus_joint.log 2>&1 &
set -u
PY=../../.venv/bin/python
CFG=configs/poly_si_3grain.yaml
COMMON="smatrix --config $CFG --specimen P3_XY --single-dataset --dataset 1 --allow-underdetermined
        --epochs 80 --warmup 8 --batch 16 --lr-S 0.02 --mode simultaneous --heldout 0.1 --split-seed 0
        --select-every 1 --checkpoint-every 10 --monitor-every 10 --device cuda --verbose"
JOINT="--single-probe-mode joint --trainable C10,C12a,C12b --lr-probe 0.005"
S0=smatrix_window_noiseless_s0_sf80_single_d1_joint

run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }

run $PY run_benchmark.py $COMMON --seeds 0 --tag sf80 $JOINT
run $PY run_benchmark.py $COMMON --seeds 0 --tag sf80
# QuantEM handoff from the joint seed-0 run: Q0 = A x nominal, Q1 = F x nominal, Q2 = A x smatrix, Q3 = F x smatrix
run $PY run_benchmark.py quantem --config $CFG --specimen P3_XY --dataset 1 --smatrix-run $S0 --single-focus-handoff \
    --arms A,F --probes nominal_trained,smatrix_trained --num-slices 35 --iters 200 --probe-warmup 40 \
    --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --tag sfQ --device cuda
run $PY run_benchmark.py $COMMON --seeds 1,2 --tag sf80 $JOINT
run $PY run_benchmark.py $COMMON --seeds 1,2 --tag sf80
# ambiguity: joint recovery from other starting probes (seed 0, starts chosen without the ground truth)
run $PY run_benchmark.py $COMMON --seeds 0 --tag sf80_c10m130 $JOINT --probe-init C10=-130
run $PY run_benchmark.py $COMMON --seeds 0 --tag sf80_c10m70 $JOINT --probe-init C10=-70
run $PY run_benchmark.py $COMMON --seeds 0 --tag sf80_astig $JOINT --probe-init C12a=5,C12b=5
echo "=== $(date -Is) all done"
