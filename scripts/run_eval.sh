#!/bin/bash
# Closed-loop evaluation of one config. Run from ./dpcc.
# Usage: bash scripts/tca_dpc/run_eval.sh [config.yaml] [model_logbase]
#   config        default config/tca_dpc/tca_dpc.yaml
#   model_logbase folder holding avoiding-d3il/diffusion/... (trained models), default ../models (shipped)
set -e
CFG=${1:-config/tca_dpc/tca_dpc.yaml}; MODELS=${2:-../models}
OUT=logs_eval/$(basename "$CFG" .yaml)        # private log base per config, models shared via symlink
mkdir -p $OUT/avoiding-d3il && ln -sfn "$(realpath $MODELS)/avoiding-d3il/diffusion" $OUT/avoiding-d3il/diffusion
export PYTHONPATH=$PWD MUJOCO_GL=${MUJOCO_GL:-egl} MPLBACKEND=Agg DPCC_LOGBASE=$OUT DPCC_EVAL_CONFIG=$CFG
python -u scripts/eval.py
