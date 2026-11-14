#!/bin/bash
# Closed-loop evaluation of one config. Run from ./dpcc.
# Usage: bash scripts/tca_dpc/run_eval.sh [config.yaml] [models]
#   config        default config/tca_dpc/tca_dpc.yaml
#   models        folder with one subfolder per seed (0-4), default ../models (shipped);
#                 after train_models.sh use logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion
set -e
CFG=${1:-config/tca_dpc/tca_dpc.yaml}; MODELS=${2:-../models}
OUT=logs_eval/$(basename "$CFG" .yaml)        # private log base per config, models shared via symlink
mkdir -p $OUT/avoiding-d3il/diffusion && ln -sfn "$(realpath $MODELS)" $OUT/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion
export PYTHONPATH=$PWD MUJOCO_GL=${MUJOCO_GL:-egl} MPLBACKEND=Agg DPCC_LOGBASE=$OUT DPCC_EVAL_CONFIG=$CFG
python -u scripts/eval.py
