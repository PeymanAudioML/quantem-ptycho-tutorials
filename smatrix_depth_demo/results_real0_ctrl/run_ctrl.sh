#!/bin/bash
cd /mnt/data/quantem-ptycho-tutorials/smatrix_depth_demo
D=data/realization_0/4dstem_quantem_noisefree
C="--data $D --reuse-smatrix --device cuda --force --probe-model parametric --dr 0.4 --extra-slices 0 --arms baseline --iters 200"
../.venv/bin/python run_realization.py $C --out results_real0_ctrl/rr_2s_fixed --dz-max 3 --probe-lr 0 > results_real0_ctrl/rr_2s_fixed.log 2>&1
../.venv/bin/python run_realization.py $C --out results_real0_ctrl/rr_2s_train --dz-max 3 > results_real0_ctrl/rr_2s_train.log 2>&1
../.venv/bin/python run_realization.py $C --out results_real0_ctrl/rr_8s_fixed --dz-max 0.75 --probe-lr 0 > results_real0_ctrl/rr_8s_fixed.log 2>&1
echo ALLDONE > results_real0_ctrl/done.txt
