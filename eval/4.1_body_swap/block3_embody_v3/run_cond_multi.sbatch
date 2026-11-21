#!/bin/bash
#SBATCH --job-name=condm
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=24
#SBATCH --mem=110G
#SBATCH --time=02:59:00
#SBATCH --output=logs/cond_%x_%j.log
# Block 3 (2026-10-31): several OFT shards of one cell in ONE job, each a separate process (own seed-7 start, so each
# shard's scenes are exactly those of a ~/run_cond.sbatch shard), sharing the GPU (OFT bf16 ~16 GB each: 3-4 fit on an
# 80 GB card, 2 on an a100-40 slice). Starts are 20 s apart so the EVAL txt log names (second resolution) differ.
# Env: NOTE BODY TRIALS SHARDS="5 10 15" SUITE=libero_spatial|libero_object EMBODY_DIR COPY_GRIP_CMD SKIP_DONE + cell vars.
export TMPDIR=$HOME/tmp; export HF_HOME=$HOME/hf_cache
export MUJOCO_GL=egl; unset DISPLAY; export TOKENIZERS_PARALLELISM=false
export ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1
SUITE=${SUITE:-libero_spatial}
if [ "$SUITE" = libero_object ]; then export DUMP_DIR=$PROJ_ROOT/runs/obj_$NOTE; else export DUMP_DIR=$PROJ_ROOT/runs/$NOTE; fi
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate oft
cd $PROJ_ROOT/openvla-oft
echo "node=$(hostname) gpu=$NUS_GPU NOTE=$NOTE BODY=${BODY:-SRC} SUITE=$SUITE TRIALS=$TRIALS SHARDS=$SHARDS DUMP_DIR=$DUMP_DIR start=$(date)"
for off in $SHARDS; do
  EP_OFFSET=$off python -X faulthandler experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name $SUITE --num_trials_per_task $TRIALS --run_id_note $NOTE \
    > $HOME/logs/condm_${NOTE}_${SUITE}_e${off}_${SLURM_JOB_ID}.log 2>&1 &
  sleep 20
done
wait
for off in $SHARDS; do echo "shard $off: $(grep -c 'Starting episode' $HOME/logs/condm_${NOTE}_${SUITE}_e${off}_${SLURM_JOB_ID}.log) episodes run, tracebacks $(grep -c Traceback $HOME/logs/condm_${NOTE}_${SUITE}_e${off}_${SLURM_JOB_ID}.log)"; done
echo "end=$(date)"
