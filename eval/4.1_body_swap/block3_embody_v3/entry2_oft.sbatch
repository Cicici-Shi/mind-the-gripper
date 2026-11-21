#!/bin/bash
#SBATCH --job-name=v3_entry2_oft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.log
# Entry check 2 (patched eval copy run_libero_eval_v3test.py): new spatial cells + libero_object 7 cells,
# 1 episode each (task 0, episode 0). Dumps to out/entry2_oft/<suite>_<cell> (never $PROJ_ROOT/runs).
export TMPDIR=$HOME/tmp HF_HOME=$HOME/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1; unset DISPLAY
export EMBODY_DIR=$PROJ_ROOT/embody_v3 MAX_TASKS=1 EP_OFFSET=0 COPY_GRIP_CMD=1
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/openvla-oft
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
while read suite note body extra; do
  [ -z "$suite" ] && continue
  echo "=== $suite $note BODY=$body $extra $(date)"
  b=""; [ "$body" != SRC ] && b="BODY=$body"
  env $extra $b DUMP_DIR=$PROJ_ROOT/embody_v3/out/entry2_oft/${suite}_$note python experiments/robot/libero/run_libero_eval_v3test.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name $suite --num_trials_per_task 1 --run_id_note v3entry2_$note \
    --local_log_dir $PROJ_ROOT/embody_v3/out/entry2_oft/logs 2>&1 | grep -E "BODY_ALIGN|TWIN_T0|Success:|CONTACT_LOG|Traceback|Error:|error:" | grep -v EGL | head -12
done <<LIST
libero_spatial g_r_sunk Panda TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk
libero_spatial ur5e_nc UR5e NO_FOREARM_CONTACT=1
libero_spatial rt2 PandaRT2 X=0
libero_object src SRC X=0
libero_object s Panda X=0
libero_object t0 Panda TWIN_RENDER=1
libero_object ur5e UR5e X=0
libero_object k3 Kinova3 X=0
libero_object g PandaR85 GRP_KP=100
libero_object g_t PandaR85 GRP_KP=100 TWIN_RENDER=1
LIST
echo end $(date)
