#!/bin/bash
#SBATCH --job-name=toyc
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# TOY1: several (cell, shard) OFT processes on ONE GPU. Env: JOBS="cell:offset cell:offset ..." (TRIALS=5 each, the
# published shard structure, EP_OFFSET 40 / 45), OUTROOT (default $PROJ_ROOT/runs_toy), MAX_TASKS (smoke only),
# NTRIALS (default 5; smoke 1). One process per item, starts 20 s apart; GPU util / memory / CPU% every 30 s to
# logs/thr_<job>.tsv; per-process wall time and episode count at the end.
export TMPDIR=$HOME/tmp HF_HOME=$HOME/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
T=$PROJ_ROOT/embody_toy; export EMBODY_DIR=$T PYTHONPATH=$PROJ_ROOT/openvla-oft
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T; source tools/toy_cells.sh
OUTROOT=${OUTROOT:-$PROJ_ROOT/runs_toy}; NTRIALS=${NTRIALS:-5}
mkdir -p $T/logs/eval
echo "node=$(hostname) gpu=$NUS_GPU cpus=$SLURM_CPUS_PER_TASK JOBS=$JOBS OUTROOT=$OUTROOT NTRIALS=$NTRIALS MAX_TASKS=${MAX_TASKS:-} start=$(date)"
PIDS=(); LOGS=(); T0=$(date +%s)
for it in $JOBS; do
  c=${it%%:*}; off=${it#*:}; v=$(cell_vars $c) || exit 2
  L=$T/logs/toyc_${c}_e${off}_${SLURM_JOB_ID}.log
  echo "launch $c e$off: $v"
  env $v DUMP_DIR=$OUTROOT/$c EP_OFFSET=$off ${MAX_TASKS:+MAX_TASKS=$MAX_TASKS} python -X faulthandler oft/run_libero_eval_toy.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial --num_trials_per_task $NTRIALS --run_id_note ${c}_e${off} --local_log_dir $T/logs/eval > $L 2>&1 &
  PIDS+=($!); LOGS+=($L); sleep 20
done
TH=$T/logs/thr_${SLURM_JOB_ID}.tsv; echo -e "t_s\tgpu_util\tgpu_mem_MiB\tcpu_pct_sum\tepisodes_started" > $TH
( while true; do alive=0; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive = 1 ] || break
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' '\t')
    c=$(ps -o pcpu= -p $(IFS=,; echo "${PIDS[*]}") 2>/dev/null | awk '{s+=$1} END {print s+0}')
    e=$(cat "${LOGS[@]}" 2>/dev/null | grep -c 'Starting episode')
    echo -e "$(( $(date +%s) - T0 ))\t$g\t$c\t$e" >> $TH; sleep 30; done ) &
wait "${PIDS[@]}"
i=0; for it in $JOBS; do L=${LOGS[$i]}; echo "$it: episodes $(grep -c 'Starting episode' $L) successes_logged=hidden tracebacks $(grep -c Traceback $L) wall_s=$(( $(date -r $L +%s) - T0 ))"; i=$((i+1)); done
echo "end=$(date)"
