#!/bin/bash
# Trains the DPCC diffusion models used in the paper (5 seeds, batch 32, 1e5 steps) into logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion/<seed>. Run from ./dpcc.
export PYTHONPATH=$PWD MUJOCO_GL=${MUJOCO_GL:-egl} DPCC_TRAIN_STEPS=${STEPS:-1e5} DPCC_BATCH_SIZE=${BATCH:-32} DPCC_LOGBASE=${LOGBASE:-logs}
mkdir -p $DPCC_LOGBASE
for s in ${SEEDS:-0 1 2 3 4}; do
  DPCC_SEEDS=$s python -u scripts/train.py > $DPCC_LOGBASE/train_seed$s.log 2>&1 &
done
wait
