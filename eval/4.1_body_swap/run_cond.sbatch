#!/bin/bash
#SBATCH --job-name=cond
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/cond_%x_%j.log
# Fixed pipeline 2026-10-26: fixture copy + robosuite EGL make_current patch + per-step dump.
export TMPDIR=$HOME/tmp; export HF_HOME=$HOME/hf_cache
export MUJOCO_GL=egl; unset DISPLAY; export TOKENIZERS_PARALLELISM=false
export ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1
export DUMP_DIR=$PROJ_ROOT/runs/$NOTE
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate oft
cd $PROJ_ROOT/openvla-oft
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) NOTE=$NOTE BODY=${BODY:-SRC} HIDE_ARM=${HIDE_ARM:-0} TRIALS=$TRIALS EP_OFFSET=${EP_OFFSET:-0} start=$(date)"
python -X faulthandler experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --task_suite_name libero_spatial \
  --num_trials_per_task $TRIALS \
  --run_id_note $NOTE
echo "exit=$? end=$(date)"
