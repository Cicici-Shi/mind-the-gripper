#!/bin/bash
#SBATCH --job-name=zoo_oft
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# vla_zoo (2026-11-03): OpenVLA-OFT with an INDEPENDENT single-suite checkpoint, same evaluation code as the published
# runs (run_libero_eval_zoo.py = byte copy of openvla-oft/experiments/robot/libero/run_libero_eval.py, sha256 cf64a47b...),
# but cwd, logs, rollouts, dumps, HF cache all under $PROJ_ROOT/vla_zoo. Pattern of embody_v3/tools/run_cond_multi.sbatch.
# Env: MODEL (oft_sp|oft_obj) CKPT (HF id) SUITE NOTE (cell) BODY (empty = native) TRIALS SHARDS="0 5" + cell vars
export TMPDIR=$HOME/tmp HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1; unset DISPLAY
export EMBODY_DIR=${EMBODY_DIR:-$PROJ_ROOT/embody_v3} COPY_GRIP_CMD=${COPY_GRIP_CMD:-1} PYTHONPATH=$PROJ_ROOT/openvla-oft
SUITE=${SUITE:-libero_spatial}
Z=$PROJ_ROOT/vla_zoo
export DUMP_DIR=$Z/runs/$MODEL/$NOTE
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -z "$NUS_GPU" ] && { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
mkdir -p $DUMP_DIR $Z/logs/oft_eval; cd $Z/oft
echo "node=$(hostname) gpu=$NUS_GPU MODEL=$MODEL CKPT=$CKPT NOTE=$NOTE BODY=${BODY:-SRC} SUITE=$SUITE TRIALS=$TRIALS SHARDS=$SHARDS DUMP_DIR=$DUMP_DIR start=$(date)"
( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $Z/logs/thr_$SLURM_JOB_ID.tsv 2>&1 &
LPID=$!
for off in $SHARDS; do
  EP_OFFSET=$off python -X faulthandler run_libero_eval_zoo.py --pretrained_checkpoint $CKPT \
    --task_suite_name $SUITE --num_trials_per_task $TRIALS --run_id_note ${MODEL}_$NOTE --local_log_dir $Z/logs/oft_eval \
    > $Z/logs/oft_${MODEL}_${NOTE}_${SUITE}_e${off}_${SLURM_JOB_ID}.log 2>&1 &
  sleep 20
done
while :; do pids=$(jobs -p | grep -v "^$LPID$"); [ -z "$pids" ] && break; wait -n $pids 2>/dev/null || true; done
kill $LPID
for off in $SHARDS; do f=$Z/logs/oft_${MODEL}_${NOTE}_${SUITE}_e${off}_${SLURM_JOB_ID}.log; echo "shard $off: $(grep -c "Starting episode" $f) episodes, tracebacks $(grep -c Traceback $f)"; done
echo "end=$(date)"
