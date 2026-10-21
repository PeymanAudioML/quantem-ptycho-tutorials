#!/usr/bin/env bash
# QuantEM on Si [001] (iman_data w200, window noiseless), dataset def2 (80 A defocus), started from the 4-defocus
# S-matrix fit of the defocus study: object init = S-matrix projected estimate (arms E, F), probe = S-matrix fitted probe.
#   E: S-informed slice grid;  F: 32 uniform slices (5 A).  Probe fixed and trained.  No TV.
# Run from synthetic_benchmark/:
#   nohup bash regularized_experiment/run_si_w200_quantem.sh > regularized_experiment/si_w200_quantem.log 2>&1 &
set -u
PY=../../.venv/bin/python
CFG=results/defocus_study/si_defocus_series_w200_window_noiseless/configs/k4_d0-d1-d2-d3.yaml
SP=k4_d0-d1-d2-d3
S=smatrix_window_noiseless_s0_study
RUNS=runs/defocus_study_si_defocus_series_w200_window_noiseless/$SP
COMMON="--config $CFG --specimen $SP --dataset 2 --smatrix-run $S --probes smatrix_fixed,smatrix_trained --iters 200 \
  --probe-warmup 40 --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --device cuda"
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run $PY run_benchmark.py quantem $COMMON --arms F --num-slices 32 --tag k4Q32
run $PY run_benchmark.py quantem ${COMMON/smatrix_fixed,/} --arms E --tag k4QE   # E: trained probe only (fixed dropped)
for r in $RUNS/quantem_*_k4Q32 $RUNS/quantem_*_k4QE; do
  [ -f $r/summary.json ] && run $PY plot_quantem.py --config $CFG --specimen $SP --run $(basename $r)
done
echo "=== $(date -Is) all done"
