#!/bin/bash
#SBATCH --job-name=pf_oft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=05:59:00
#SBATCH --output=logs/%x_%j.log
# physfix (protocol_physfix.md 0dd4193c..., 1-2, 7.1, 7.4), 2026-11-06: OFT / OFTi in-process eval on one WHOLE a100-40
# card (node15), NPAR=2 processes at a time from the RUNS queue, EMBODY_DIR=$PROJ_ROOT/embody_next, cell switches from
# next/cells_next.sh, launcher next/pf/launch_pf.py (blind: no success lines, no videos; online BADQACC; contact log).
# Env: MODEL (pf_oftc_sp|pf_oftc_obj|pf_ofti_sp|pf_ofti_obj) CKPT SUITE RUNS="cell:off ..." TRIALS (5) HFH (HF_HOME)
set -u
export TMPDIR=$HOME/tmp HF_HOME=${HFH:-$PROJ_ROOT/vla_zoo/hf_cache} MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export M1_FASTSYNC=1 ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 SKIP_DONE=1; unset DISPLAY
export EMBODY_DIR=$PROJ_ROOT/embody_next PYTHONPATH=$PROJ_ROOT/openvla-oft
Z=$PROJ_ROOT/vla_zoo; P=$Z/next/pf
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -z "$NUS_GPU" ] && { echo "no GPU visible on $(hostname)"; exit 3; }
case "$NUS_GPU" in *A100*40GB*) ;; *) echo "wrong card $NUS_GPU"; exit 3;; esac
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
mkdir -p $Z/logs/oft_eval_pf; cd $Z/oft
echo "node=$(hostname) gpu=$NUS_GPU MODEL=$MODEL CKPT=$CKPT SUITE=$SUITE RUNS=$RUNS TRIALS=${TRIALS:-5} start=$(date)"
echo "sha launch_pf=$(sha256sum $P/launch_pf.py|cut -c1-16) eval=$(sha256sum run_libero_eval_m1.py|cut -c1-16) cells=$(sha256sum $Z/next/cells_next.sh|cut -c1-16) body_env=$(sha256sum $EMBODY_DIR/embody/body_env.py|cut -c1-16) robots=$(sha256sum $EMBODY_DIR/embody/robots.py|cut -c1-16) fastsync=$(sha256sum $Z/common/m1_fastsync.py|cut -c1-16)"
( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 60; done ) > $Z/logs/thr_pf_$SLURM_JOB_ID.tsv 2>&1 &
LPID=$!
NPAR=${NPAR:-2}
for r in $RUNS; do
  while [ $(jobs -p | grep -v "^$LPID$" | wc -l) -ge $NPAR ]; do wait -n $(jobs -p | grep -v "^$LPID$") 2>/dev/null || true; done
  cell=${r%%:*}; off=${r##*:}
  set -- $(bash $Z/next/cells_next.sh vars $cell); body=$1; shift
  [ "$body" = SRC ] && { echo "SRC not allowed here"; continue; }
  mkdir -p $Z/runs/$MODEL/$cell
  echo "launch $cell e$off BODY=$body $*"
  env -u BODY -u R85_ARMATURE -u GRP_KP -u GRP_MAP -u TWIN_RENDER -u TWIN_CAM -u TWIN_CAMS -u TWIN_CAM_SYNC -u TWIN_BODY -u TOY_PARTS \
    BODY=$body "$@" DUMP_DIR=$Z/runs/$MODEL/$cell EP_OFFSET=$off \
    python -X faulthandler $P/launch_pf.py --pretrained_checkpoint $CKPT \
    --task_suite_name $SUITE --num_trials_per_task ${TRIALS:-5} --run_id_note ${MODEL}_${cell}_e$off --local_log_dir $Z/logs/oft_eval_pf \
    > $Z/logs/pf_${MODEL}_${cell}_e${off}_${SLURM_JOB_ID}.log 2>&1 &
  sleep 20
done
while :; do pids=$(jobs -p | grep -v "^$LPID$"); [ -z "$pids" ] && break; wait -n $pids 2>/dev/null || true; done
kill $LPID
for r in $RUNS; do cell=${r%%:*}; off=${r##*:}; f=$Z/logs/pf_${MODEL}_${cell}_e${off}_${SLURM_JOB_ID}.log
  echo "$cell e$off: starting=$(grep -c "Starting episode" $f) tracebacks=$(grep -c Traceback $f) outcome_lines=$(grep -ci "success" $f)"; done
echo "end=$(date)"
