#!/usr/bin/env bash
# Depth TV at 35 slices (5.71 A): arm F, tv_weight_z = 1000 (same weight as the best 20-slice run) and 1750 (1000 x 10/5.71,
# compensating the smaller per-slice potential).  Same settings as run_tvz_sweep.sh otherwise.  Then plots and the 40 A
# band comparison with the 35-slice baseline and the 20-slice runs.
# Run from synthetic_benchmark/:  nohup bash regularized_experiment/run_tvz35.sh > regularized_experiment/tvz35.log 2>&1 &
set -u
PY=../../.venv/bin/python
CFG=configs/poly_si_3grain.yaml
SP=P3_XYZ
S=smatrix_window_noiseless_s0_sfall_single_d1
RUNS=runs/poly_si_3grain/$SP
OUT=regularized_experiment/results/tvz_35sl
W="1000 1750"
R=quantem_window_noiseless_d1_F_smatrix_trained_call_s0
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }

run $PY regularized_experiment/run_quantem_tv.py --tv-z $(echo $W | tr ' ' ',') -- --config $CFG --specimen $SP --dataset 1 \
    --smatrix-run $S --single-focus-handoff --arms F --probes smatrix_trained --num-slices 35 --iters 200 --probe-warmup 40 \
    --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --tag sfallQ35 --device cuda
mkdir -p $OUT
LIST=${R}_sfallQ=F_35sl_tvz0,${R}_sfallQ20_tvz0=F_20sl_tvz0,${R}_sfallQ20_tvz1000=F_20sl_tvz1000
for w in $W; do
  r=${R}_sfallQ35_tvz$w
  run $PY plot_quantem.py --config $CFG --specimen $SP --run $r
  mkdir -p $OUT/$r && cp $RUNS/$r/summary.json $OUT/$r/ && cp -r $RUNS/$r/plots $OUT/$r/
  LIST=$LIST,$r=F_35sl_tvz$w
done
run $PY regularized_experiment/eval_depth_bands.py --specimen $SP --out $OUT --runs $LIST
echo "=== $(date -Is) all done"
