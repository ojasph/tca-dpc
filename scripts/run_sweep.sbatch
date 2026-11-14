#!/bin/bash
#SBATCH --job-name=tca-dpc
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs_triton/%x_%A_%a.out
# SLURM array over a sweep folder: task i runs line i of $SWEEP_DIR/jobs.txt. Submit from ./dpcc, e.g.
#   python scripts/tca_dpc/paper_sweeps/make_sweep_v1.py
#   mkdir -p logs_triton && sbatch --array=1-$(wc -l < config/tca_dpc/sweeps/v1/jobs.txt) \
#       --export=ALL,SWEEP_DIR=config/tca_dpc/sweeps/v1 scripts/tca_dpc/run_sweep.sbatch
# Each task writes to logs_sweep/<sweep>/<job> (paths saved by default), which analysis/ reads.
D=${SWEEP_DIR:?set SWEEP_DIR}; MODELS=${MODELS:-../models}
J=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $D/jobs.txt)
LB=logs_sweep/$(basename $D)/$J
mkdir -p $LB/avoiding-d3il/diffusion && ln -sfn "$(realpath $MODELS)" $LB/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion
export PYTHONPATH=$PWD MUJOCO_GL=egl MPLBACKEND=Agg OMP_NUM_THREADS=4 DPCC_LOGBASE=$LB DPCC_EVAL_CONFIG=$D/$J.yaml DPCC_SAVE_PATHS=${DPCC_SAVE_PATHS:-1}
echo "job=$J $(date)"
python -u scripts/eval.py
echo "exit=$? $(date)"
