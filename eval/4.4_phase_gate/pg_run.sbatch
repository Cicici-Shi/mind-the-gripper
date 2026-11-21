#!/bin/bash
#SBATCH --job-name=pg_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# phasegate cells. phasegate-confirm copy (2026-11-10): OUTROOT defaults to $T/out/confirm (protocol_phasegate_confirm.md);
# window post added (complement of approach); card check by memory (A100 labels are unreliable, see below).
# One shard = up to 8 OFT processes on one h200, up to 4 on one a100-80 or up to 2 on one a100-40 (CARD). Physical body is always the stock Panda.
# Env: MODEL (oft|ofti) SUITE (spatial|object) JOBS="mode:window ..." (mode look|erase, window none|approach|grasp|
# transport|all|post); EP_OFFSET (default 0) NTRIALS (default 5) -> keys = tasks 0..9 x episodes EP_OFFSET..+NTRIALS-1;
# SKIP_DONE / MAX_TASKS passthrough; OUTROOT. Dumps: $OUTROOT/${MODEL}_${SUITE}/${mode}_${window}/.
# CARD (default h200): h200 -> at most 8 processes, default in-process GL allow-list (H200);
#   a100-80 -> at most 4 processes (~16 GB each), PG_GL_ALLOW=A100; a100-40 -> at most 2 (40 GB holds no more),
#   PG_GL_ALLOW=A100. Submit with the card on the command line, which overrides the template's gres / exclude (the
#   template excludes the A100 nodes); node39 (ARM64, x86 conda fails with Exec format error) and node29 (H100, render
#   crashes) stay excluded:
#   sbatch --gres=gpu:a100-80:1 --exclude=node39,node38 --export=ALL,CARD=a100-80,... tools/pg_run.sbatch
#   sbatch --gres=gpu:a100-40:1 --exclude=node39,node38 --export=ALL,CARD=a100-40,... tools/pg_run.sbatch
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
T=$PROJ_ROOT/phasegate_confirm; export EMBODY_DIR=$T PYTHONPATH=$PROJ_ROOT/openvla-oft
export NUS_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, /|/')"
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
CARD=${CARD:-h200}
# card check: family by name, memory at least the requested size (MiB). A node labelled a100-40 may carry an 80 GB card
# (node25, 2026-11-10); that passes for CARD=a100-40 and keeps MAXP=2 (the size that was requested).
case "$CARD" in
  h200) MAXP=8; NEED=139000; FAM=H200; unset PG_GL_ALLOW ;;
  a100-80) MAXP=4; NEED=79000; FAM=A100; export PG_GL_ALLOW=A100 ;;
  a100-40) MAXP=2; NEED=39000; FAM=A100; export PG_GL_ALLOW=A100 ;;
  *) echo "bad CARD $CARD"; exit 4 ;;
esac
[[ "$NUS_GPU" == *"$FAM"* ]] && [ "${GPU_MIB:-0}" -ge "$NEED" ] 2>/dev/null || { echo "card mismatch: CARD=$CARD needs $FAM with >= $NEED MiB, got $NUS_GPU ($GPU_MIB MiB)"; exit 4; }
NJ=$(echo $JOBS | wc -w); [ "$NJ" -le "$MAXP" ] || { echo "CARD=$CARD takes at most $MAXP processes, JOBS has $NJ"; exit 4; }
case "$MODEL:$SUITE" in
  oft:spatial|oft:object) CKPT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10; export HF_HOME=$HOME/hf_cache ;;
  ofti:spatial) CKPT=moojink/openvla-7b-oft-finetuned-libero-spatial; export HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache ;;
  ofti:object) CKPT=moojink/openvla-7b-oft-finetuned-libero-object; export HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache ;;
  *) echo "bad MODEL:SUITE"; exit 4 ;;
esac
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T
BLOCK=${MODEL}_${SUITE}; OUTROOT=${OUTROOT:-$T/out/confirm}; NTRIALS=${NTRIALS:-5}; OFF=${EP_OFFSET:-0}
mkdir -p $T/logs/eval
echo "node=$(hostname) gpu=$NUS_GPU card=$CARD maxp=$MAXP gl_allow=${PG_GL_ALLOW:-H200(default)} block=$BLOCK ckpt=$CKPT JOBS=$JOBS EP_OFFSET=$OFF NTRIALS=$NTRIALS MAX_TASKS=${MAX_TASKS:-} start=$(date)"
echo "sha256: $(sha256sum embody/phasegate.py embody/body_env.py oft/run_libero_eval_pg.py tools/pg_run.sbatch | awk '{print $1" "$2}' | tr '\n' ' ')"
echo "outroot=$OUTROOT gpu_mib=$GPU_MIB need_mib=$NEED"
PIDS=(); LOGS=()
for it in $JOBS; do
  mode=${it%%:*}; win=${it#*:}; c=${mode}_${win}
  case "$mode" in
    look) EX="TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk" ;;
    erase) EX="" ;;
    *) echo "bad mode $mode"; exit 4 ;;
  esac
  L=$T/logs/pg_${BLOCK}_${c}_e${OFF}_${SLURM_JOB_ID}.log; mkdir -p $OUTROOT/$BLOCK/$c
  echo "launch $c e$OFF"
  env BODY=Panda COPY_GRIP_CMD=1 PG_MODE=$mode PG_WINDOW=$win $EX DUMP_DIR=$OUTROOT/$BLOCK/$c EP_OFFSET=$OFF \
    ${SKIP_DONE:+SKIP_DONE=$SKIP_DONE} ${MAX_TASKS:+MAX_TASKS=$MAX_TASKS} \
    python -X faulthandler oft/run_libero_eval_pg.py --pretrained_checkpoint $CKPT \
    --task_suite_name libero_$SUITE --num_trials_per_task $NTRIALS --run_id_note pg_${BLOCK}_${c}_e${OFF} \
    --local_log_dir $T/logs/eval > $L 2>&1 &
  PIDS+=($!); LOGS+=($L); sleep 20
done
wait "${PIDS[@]}"
i=0; for it in $JOBS; do L=${LOGS[$i]}; c=${it%%:*}_${it#*:}; echo "$it: started $(grep -c "Starting episode" $L) dumps $(ls $OUTROOT/$BLOCK/$c/*.npz 2>/dev/null | wc -l) episode_errors $(grep -c "Episode error" $L) dump_errors $(grep -c "Dump error" $L) phasegate_lines $(grep -c "PHASEGATE:" $L)"; i=$((i+1)); done
echo "end=$(date)"
