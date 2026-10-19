#!/usr/bin/env bash
# P3_XYZ with all three defocus datasets in the S-matrix fit, set up like P3_XY full80 (80 epochs, warm-up 8, lr_S 0.02,
# batch 16, no held-out data, dataset 0 = gauge, C10/C12 of datasets 1-2 learned).  QuantEM is still trained on
# dataset 1 only, as in every earlier run:
#   F, 20 slices, tv_weight_z 0 and 1000 (as the 1-defocus TV sweep)   tag full80Q20
#   E, 35 slices, S-informed grid (as P3_XY grid35)                    tag full80Q35
# Then plots, 40 A depth bands vs the 1-defocus runs, and the 3D grain maps.
# Run from synthetic_benchmark/:
#   nohup bash regularized_experiment/run_p3xyz_3defocus.sh > regularized_experiment/p3xyz_3defocus.log 2>&1 &
set -u
PY=../../.venv/bin/python
CFG=configs/poly_si_3grain.yaml
SP=P3_XYZ
S=smatrix_window_noiseless_s0_full80
RUNS=runs/poly_si_3grain/$SP
OUT=regularized_experiment/results/p3xyz_3defocus
QARGS="--config $CFG --specimen $SP --dataset 1 --smatrix-run $S --probes smatrix_trained --iters 200 --probe-warmup 40 \
  --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --device cuda"
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }
mkdir -p $OUT

# 1. S-matrix fit on all three defocus datasets (~3 h on the T4)
run $PY run_benchmark.py smatrix --config $CFG --specimen $SP --epochs 80 --heldout 0.0 --monitor-every 10 --tag full80 \
    --device cuda
[ -f $RUNS/$S/refocused_phase.npy ] || { echo "S-matrix fit failed"; exit 1; }
mkdir -p $OUT/$S && cp $RUNS/$S/summary.json $RUNS/$S/arrays.npz $OUT/$S/
run $PY plot_fit_monitor.py --config $CFG --specimen $SP --run $S --once
cp $RUNS/$S/monitor/*.png $OUT/$S/ 2>/dev/null
run $PY plot_depth_features.py --config $CFG --specimen $SP --runs smatrix_window_noiseless_s0_sfall_single_d1 $S
cp results/smatrix/poly_si_3grain/$SP/depth_features.png $OUT/depth_features_1_vs_3_defocus.png

# 2. QuantEM on dataset 1 from the 3-defocus S-matrix
run $PY regularized_experiment/run_quantem_tv.py --tv-z 0,1000 -- $QARGS --arms F --num-slices 20 --tag full80Q20
run $PY run_benchmark.py quantem $QARGS --arms E --num-slices 35 --tag full80Q35
F0=quantem_window_noiseless_d1_F_smatrix_trained_call_s0_full80Q20_tvz0
F1=quantem_window_noiseless_d1_F_smatrix_trained_call_s0_full80Q20_tvz1000
E35=quantem_window_noiseless_d1_E_smatrix_trained_call_s0_full80Q35
for r in $F0 $F1 $E35; do
  run $PY plot_quantem.py --config $CFG --specimen $SP --run $r
  mkdir -p $OUT/$r && cp $RUNS/$r/summary.json $OUT/$r/ && cp -r $RUNS/$r/plots $OUT/$r/
done
run $PY plot_smatrix_slices.py --config $CFG --specimen $SP --run $S --quantem-run $E35
cp results/single_focus_joint/$SP/smatrix_slices_$S.png $OUT/ 2>/dev/null

# 3. depth bands: 3-defocus vs 1-defocus (same arms and settings)
run $PY regularized_experiment/eval_depth_bands.py --specimen $SP --out $OUT --runs \
quantem_window_noiseless_d1_A_nominal_trained_call_s0_xyz_single=A_35sl_noS,\
quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ20_tvz0=F20_w0_1def,\
$F0=F20_w0_3def,\
quantem_window_noiseless_d1_F_smatrix_trained_call_s0_sfallQ20_tvz1000=F20_w1000_1def,\
$F1=F20_w1000_3def,\
quantem_window_noiseless_d1_E_smatrix_trained_call_s0_sfallQ=E35_1def,\
$E35=E35_3def

# 4. 3D grain maps (S-matrix + one QuantEM run each)
run $PY regularized_experiment/smatrix_grains_3d.py --specimen $SP --run $S --smatrix-label "S-matrix (3 defocus)" \
    --quantem-run $F1 --quantem-label "QuantEM F, 20 slices, TV w=1000 (3-def S)" --out $OUT/grains_3d_F20_w1000
run $PY regularized_experiment/smatrix_grains_3d.py --specimen $SP --run $S --smatrix-label "S-matrix (3 defocus)" \
    --quantem-run $E35 --quantem-label "QuantEM E, 35 slices (3-def S)" --out $OUT/grains_3d_E35
run $PY regularized_experiment/plot_grains_overlay.py --specimen $SP --run $S --grains $OUT/grains_3d_F20_w1000 \
    --quantem-run $F1 --quantem-label "QuantEM F w=1000"
echo "=== $(date -Is) all done"
