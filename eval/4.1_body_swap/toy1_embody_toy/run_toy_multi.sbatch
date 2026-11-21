#!/bin/bash
#SBATCH --job-name=toym
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=24
#SBATCH --mem=110G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# TOY1 (2026-11-03), after embody_v3/tools/run_cond_multi.sbatch: several OFT processes of one cell on ONE GPU, each its own
# seed-7 process, starts 20 s apart. Code: $PROJ_ROOT/embody_toy (EMBODY_DIR) + oft/run_libero_eval_toy.py; outputs only
# under $PROJ_ROOT/embody_toy (dumps out/runs/<NOTE>, eval txt logs logs/eval, rollouts/).
# Env: NOTE BODY TRIALS (episodes per shard, default 5 = published shard structure) PROCS="off:keep off:keep ..."
#   (off = EP_OFFSET; keep = TOY_EP_KEEP list or "all"; e.g. PROCS="0:0,1 0:2,3 5:5,6 5:7,8" = 8 episodes per task on
#   4 processes with the published scenes) + cell vars (GRP_KP TWIN_RENDER TOY_* ...). QUERY_DUMP=1 stores raw chunks.
#   GPU type is chosen at submission (--gres=gpu:<type>:1); every ~30 s GPU util / memory and CPU% of the processes
#   are logged to logs/thr_<NOTE>_<job>.tsv for the per-card throughput table.
export TMPDIR=$HOME/tmp; export HF_HOME=$HOME/hf_cache
export MUJOCO_GL=egl; unset DISPLAY; export TOKENIZERS_PARALLELISM=false
export ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1
T=$PROJ_ROOT/embody_toy
export EMBODY_DIR=$T PYTHONPATH=$PROJ_ROOT/openvla-oft${PYTHONPATH:+:$PYTHONPATH}
export DUMP_DIR=$T/out/runs/$NOTE
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
TRIALS=${TRIALS:-5}; PROCS=${PROCS:-"0:all"}
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate oft
cd $T; mkdir -p $T/logs/eval $DUMP_DIR
echo "node=$(hostname) gpu=$NUS_GPU cpus=$SLURM_CPUS_PER_TASK NOTE=$NOTE BODY=${BODY:-SRC} TRIALS=$TRIALS PROCS=$PROCS DUMP_DIR=$DUMP_DIR start=$(date)"
env | grep -E '^(TWIN_|TOY_|GRP_|COPY_GRIP|QUERY_DUMP)' | sort
PIDS=(); i=0; T0=$(date +%s)
for p in $PROCS; do
  off=${p%%:*}; keep=${p#*:}; [ "$keep" = all ] && keep=""
  L=$T/logs/toym_${NOTE}_p${i}_${SLURM_JOB_ID}.log
  EP_OFFSET=$off TOY_EP_KEEP=$keep python -X faulthandler oft/run_libero_eval_toy.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial --num_trials_per_task $TRIALS --run_id_note $NOTE --local_log_dir $T/logs/eval \
    > $L 2>&1 &
  PIDS+=($!); i=$((i+1)); sleep 20
done
TH=$T/logs/thr_${NOTE}_${SLURM_JOB_ID}.tsv
echo -e "t_s\tgpu_util\tgpu_mem_MiB\tcpu_pct_sum\tdumps" > $TH
( while kill -0 ${PIDS[0]} 2>/dev/null || pgrep -f "run_id_note $NOTE" >/dev/null; do
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' '\t')
    c=$(ps -o pcpu= --ppid $(IFS=,; echo "${PIDS[*]}") -p $(IFS=,; echo "${PIDS[*]}") 2>/dev/null | awk '{s+=$1} END {print s+0}')
    echo -e "$(( $(date +%s) - T0 ))\t$g\t$c\t$(ls $DUMP_DIR | wc -l)" >> $TH; sleep 30; done ) &
wait "${PIDS[@]}"
T1=$(date +%s)
i=0; for p in $PROCS; do L=$T/logs/toym_${NOTE}_p${i}_${SLURM_JOB_ID}.log
  echo "proc $i ($p): $(grep -c 'Starting episode' $L) episodes, tracebacks $(grep -c Traceback $L)"; i=$((i+1)); done
echo "wall_s=$((T1-T0)) dumps=$(ls $DUMP_DIR | wc -l) end=$(date)"
