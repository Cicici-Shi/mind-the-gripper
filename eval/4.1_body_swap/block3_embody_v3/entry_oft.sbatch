#!/bin/bash
#SBATCH --job-name=v3_entry_oft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Entry check: run_libero_eval with EMBODY_DIR=embody_v3, 1 episode (task 0, episode 0) per new
# condition. Dumps go to out/entry_oft/<NOTE> (never $PROJ_ROOT/runs).
export TMPDIR=$HOME/tmp HF_HOME=$HOME/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1; unset DISPLAY
export EMBODY_DIR=$PROJ_ROOT/embody_v3 MAX_TASKS=1 EP_OFFSET=0
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/openvla-oft
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
for spec in "p6j3 PandaL3" "k3 Kinova3" "rt PandaRT" "g_r Panda TWIN_RENDER=1 TWIN_BODY=PandaR85" \
            "grpkp100_tw PandaR85 GRP_KP=100 TWIN_RENDER=1 TWIN_CAMS=wrist" "grpkp100_ta PandaR85 GRP_KP=100 TWIN_RENDER=1 TWIN_CAMS=agent"; do
  set -- $spec; note=$1; body=$2; shift 2
  echo "=== $note BODY=$body $* $(date)"
  env "$@" BODY=$body DUMP_DIR=$PROJ_ROOT/embody_v3/out/entry_oft/$note python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial --num_trials_per_task 1 --run_id_note v3entry_$note \
    --local_log_dir $PROJ_ROOT/embody_v3/out/entry_oft/logs 2>&1 | grep -E "BODY_ALIGN|TWIN_T0|Success|success rate|Error|Traceback|error" | head -20
done
echo end $(date)
