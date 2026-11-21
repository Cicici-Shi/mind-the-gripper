#!/bin/bash
#SBATCH --job-name=condm_g_r
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=02:59:00
#SBATCH --output=logs/cond_%x_%j.log
# Block 3 G_R full shards (amendment 2, ruling 2 "submission"): in ONE job, (a) tools/gr_t0check.py once per shard
# offset, in parallel, on the same node/GPU (records go into the cell average); (b) the OFT eval shards exactly as
# tools/run_cond_multi.sbatch (separate process per shard, 20 s apart). Env: NOTE=g_r BODY TRIALS SHARDS + cell vars.
export TMPDIR=$HOME/tmp; export HF_HOME=$HOME/hf_cache
export MUJOCO_GL=egl; unset DISPLAY; export TOKENIZERS_PARALLELISM=false
export ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1
V=$PROJ_ROOT/embody_v3
export DUMP_DIR=$PROJ_ROOT/runs/$NOTE
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate oft
echo "node=$(hostname) gpu=$NUS_GPU NOTE=$NOTE BODY=${BODY:-SRC} TRIALS=$TRIALS SHARDS=$SHARDS start=$(date)"
mkdir -p $V/out/gr_t0
for off in $SHARDS; do
  ( cd $V && OPENBLAS_NUM_THREADS=1 python -X faulthandler tools/gr_t0check.py --offsets $off \
      --out $V/out/gr_t0/oft_${SLURM_JOB_ID}_e${off}.jsonl ) > $V/out/gr_t0/oft_${SLURM_JOB_ID}_e${off}.log 2>&1 &
done
cd $PROJ_ROOT/openvla-oft
for off in $SHARDS; do
  EP_OFFSET=$off python -X faulthandler experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial --num_trials_per_task $TRIALS --run_id_note $NOTE \
    > $HOME/logs/condm_${NOTE}_libero_spatial_e${off}_${SLURM_JOB_ID}.log 2>&1 &
  sleep 20
done
wait
for off in $SHARDS; do echo "shard $off: $(grep -c 'Starting episode' $HOME/logs/condm_${NOTE}_libero_spatial_e${off}_${SLURM_JOB_ID}.log) episodes, tracebacks $(grep -c Traceback $HOME/logs/condm_${NOTE}_libero_spatial_e${off}_${SLURM_JOB_ID}.log); gr_t0check records $(wc -l < $V/out/gr_t0/oft_${SLURM_JOB_ID}_e${off}.jsonl)"; done
echo "end=$(date)"
