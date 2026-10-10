#!/bin/sh
# Dose sensitivity (runs after run_joint_followup.sh): reduced_s2 config B at 2e3 and 2e4 e-/pattern, seeds 0,1.
cd "$(dirname "$0")"
while [ ! -f results_joint/logs/followup_done.txt ]; do sleep 60; done
for D in 2e4 2e3; do
  python run_joint_probe.py exp --sample reduced_s2 --config B --variants fixed,joint,oracle --seeds 0,1 --dose $D --tag _dose$D \
    > results_joint/logs/reduced_s2_B_dose$D.log 2>&1
done
echo DOSE_DONE > results_joint/logs/dose_done.txt
