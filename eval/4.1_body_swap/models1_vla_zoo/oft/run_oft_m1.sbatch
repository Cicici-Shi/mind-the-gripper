#!/bin/bash
#SBATCH --job-name=m1_oft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=05:59:00
#SBATCH --output=logs/%x_%j.log
# models1 (protocol_models1.md 1, 2, 7.4), 2026-11-04: OFTi in-process eval on one WHOLE a100-40 card (node15; node23
# slices excluded), one python process per "cell:offset" in RUNS (2 per card). run_libero_eval_m1.py is a byte copy
# of $PROJ_ROOT/embody_toy/oft/run_libero_eval_toy.py (sha256 a65fc6d0...; the zoo copy of the published script has no
# twin_cam_err in its dump, protocol 7.4 fallback), every toy switch unset. Cell switches from tools/zoo_cells.sh.
# Env: MODEL (oft_sp|oft_obj) CKPT SUITE RUNS="cell:off cell:off ..." (run NPAR=2 at a time) (object cells named obj_*), TRIALS (5)
set -u
export TMPDIR=$HOME/tmp HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export M1_FASTSYNC=${M1_FASTSYNC:-1} ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 SKIP_DONE=1; unset DISPLAY
export EMBODY_DIR=$PROJ_ROOT/embody_toy PYTHONPATH=$PROJ_ROOT/openvla-oft
Z=$PROJ_ROOT/vla_zoo
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -z "$NUS_GPU" ] && { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
mkdir -p $Z/logs/oft_eval; cd $Z/oft
echo "node=$(hostname) gpu=$NUS_GPU MODEL=$MODEL CKPT=$CKPT SUITE=$SUITE RUNS=$RUNS TRIALS=${TRIALS:-5} start=$(date)"
( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $Z/logs/thr_$SLURM_JOB_ID.tsv 2>&1 &
LPID=$!
NPAR=${NPAR:-2}   # processes on the card at once; RUNS longer than NPAR are queued (pool)
for r in $RUNS; do
  while [ $(jobs -p | grep -v "^$LPID$" | wc -l) -ge $NPAR ]; do wait -n $(jobs -p | grep -v "^$LPID$") 2>/dev/null || true; done
  cell=${r%%:*}; off=${r##*:}
  set -- $($Z/tools/zoo_cells.sh vars $cell); body=$1; shift
  [ "$body" = SRC ] && body=""
  mkdir -p $Z/runs/$MODEL/$cell
  echo "launch $cell e$off BODY=${body:-SRC} $*"
  env -u BODY ${body:+BODY=$body} "$@" DUMP_DIR=$Z/runs/$MODEL/$cell EP_OFFSET=$off \
    python -X faulthandler launch_m1.py --pretrained_checkpoint $CKPT \
    --task_suite_name $SUITE --num_trials_per_task ${TRIALS:-5} --run_id_note ${MODEL}_${cell}_e$off --local_log_dir $Z/logs/oft_eval \
    > $Z/logs/m1_${MODEL}_${cell}_e${off}_${SLURM_JOB_ID}.log 2>&1 &
  sleep 20
done
while :; do pids=$(jobs -p | grep -v "^$LPID$"); [ -z "$pids" ] && break; wait -n $pids 2>/dev/null || true; done
kill $LPID
for r in $RUNS; do cell=${r%%:*}; off=${r##*:}; f=$Z/logs/m1_${MODEL}_${cell}_e${off}_${SLURM_JOB_ID}.log
  echo "$cell e$off: starting=$(grep -c "Starting episode" $f) tracebacks=$(grep -c Traceback $f)"; done
echo "end=$(date)"
