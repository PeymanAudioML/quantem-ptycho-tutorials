#!/usr/bin/env bash
# C1' sweep: QuantEM built-in depth TV (tv_weight_z) on P3_XYZ, arm F (S-matrix object start, uniform slices),
# 20 slices of 10 A, dataset 1 only, all patterns.  Weights 0 (20-slice control), 10, 100, 1000.
# Then plot_quantem per run and the 40 A depth-band comparison against the 35-slice baseline (A, E, F).
# Run from synthetic_benchmark/:  nohup bash regularized_experiment/run_tvz_sweep.sh > regularized_experiment/tvz_sweep.log 2>&1 &
set -u
PY=../../.venv/bin/python
CFG=configs/poly_si_3grain.yaml
SP=P3_XYZ
S=smatrix_window_noiseless_s0_sfall_single_d1
RUNS=runs/poly_si_3grain/$SP
OUT=regularized_experiment/results/tvz_sweep
W="0 10 100 1000"
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }

run $PY regularized_experiment/run_quantem_tv.py --tv-z $(echo $W | tr ' ' ',') -- --config $CFG --specimen $SP --dataset 1 \
    --smatrix-run $S --single-focus-handoff --arms F --probes smatrix_trained --num-slices 20 --iters 200 --probe-warmup 40 \
    --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --tag sfallQ20 --device cuda
mkdir -p $OUT
LIST=quantem_window_noiseless_d1_A_nominal_trained_call_s0_xyz_single=A_35sl,quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ=E_35sl,quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ=F_35sl
for w in $W; do
  r=quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ20_tvz$w
  run $PY plot_quantem.py --config $CFG --specimen $SP --run $r
  mkdir -p $OUT/$r && cp $RUNS/$r/summary.json $OUT/$r/ && cp -r $RUNS/$r/plots $OUT/$r/
  LIST=$LIST,$r=F_20sl_tvz$w
done
run $PY regularized_experiment/eval_depth_bands.py --specimen $SP --out $OUT --runs $LIST
echo "=== $(date -Is) all done"
