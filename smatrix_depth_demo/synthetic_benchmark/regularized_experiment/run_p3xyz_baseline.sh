#!/usr/bin/env bash
# P3_XYZ (3 grains stacked in depth) baseline, no regularization: dataset 1 only (one defocus), ALL 2809 patterns.
#   1. QuantEM arm A: uniform 35 slices, flat start, nominal probe trained after 40 iters (no S-matrix)
#   2. S-matrix fit on dataset 1, probe fixed at nominal, --heldout 0 (all patterns), 80 epochs
#   3. S-matrix refocused slices plot (on arm A's 35 slab boundaries)
#   4. QuantEM hand-off: arm E (S-informed slice grid + S object start), arm F (uniform 35 slices + S object start)
#   5. plot_quantem for A, E, F; copy summaries + plots to regularized_experiment/results/p3xyz_baseline/
# Run from synthetic_benchmark/:  nohup bash regularized_experiment/run_p3xyz_baseline.sh > regularized_experiment/p3xyz_baseline.log 2>&1 &
set -u
PY=../../.venv/bin/python
CFG=configs/poly_si_3grain.yaml
SP=P3_XYZ
RUNS=runs/poly_si_3grain/$SP
OUT=regularized_experiment/results/p3xyz_baseline
A=quantem_window_noiseless_d1_A_nominal_trained_call_s0_xyz_single
S=smatrix_window_noiseless_s0_sfall_single_d1
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }

run $PY run_benchmark.py quantem --config $CFG --specimen $SP --single-dataset --dataset 1 \
    --arms A --probes nominal_trained --num-slices 35 --iters 200 --probe-warmup 40 \
    --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --tag xyz --device cuda
run $PY plot_quantem.py --config $CFG --specimen $SP --run $A
run $PY run_benchmark.py smatrix --config $CFG --specimen $SP --single-dataset --dataset 1 --allow-underdetermined \
    --epochs 80 --warmup 8 --batch 16 --lr-S 0.02 --mode simultaneous --heldout 0.0 --split-seed 0 \
    --select-every 1 --checkpoint-every 10 --monitor-every 10 --device cuda --verbose --seeds 0 --tag sfall
run $PY plot_smatrix_slices.py --config $CFG --specimen $SP --run $S --quantem-run $A
run $PY run_benchmark.py quantem --config $CFG --specimen $SP --dataset 1 --smatrix-run $S --single-focus-handoff \
    --arms E,F --probes smatrix_trained --num-slices 35 --iters 200 --probe-warmup 40 \
    --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --tag sfallQ --device cuda
for arm in E F; do
  run $PY plot_quantem.py --config $CFG --specimen $SP --run quantem_window_noiseless_d1_${arm}_smatrix_trained_call_s0_sfallQ
done
mkdir -p $OUT
for r in $A $S quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ; do
  mkdir -p $OUT/$r
  cp $RUNS/$r/summary.json $OUT/$r/ 2>/dev/null
  cp -r $RUNS/$r/plots $OUT/$r/ 2>/dev/null
done
echo "=== $(date -Is) all done"
