#!/bin/bash
#SBATCH --job-name=mic_c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-confirm template (protocol_maskinfo_confirm.md v1 §2, §5, §7 item 5; copy of the frozen
# tools/mi_cells_run.sbatch e394f270...): h200 ONLY, one shard = 8 processes on one card; block = ${MODEL}_${SUITE}_c30;
# eval script oft/run_libero_eval_mi_c.py (writes mi_cell); EP_OFFSET 30 / 35 (smoke: the already-read keys).
# The EGL GL_RENDERER must contain H200 (else exit 5). Physical Panda (BODY=Panda COPY_GRIP_CMD=1 ORIGINAL_ONLY=1) +
# MI_CELL. Env: MODEL (oft|ofti) SUITE (spatial|object) JOBS="cell:offset ..." CARD (h200); NTRIALS (5); SKIP_DONE
# passthrough; MAX_TASKS (smoke only); OUTROOT.
# Never prints success, per-process wall clock or query counts.
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
T=$PROJ_ROOT/maskinfo_confirm; export EMBODY_DIR=$T PYTHONPATH=$PROJ_ROOT/openvla-oft
export NUS_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, /|/')"
MEM=$(echo "$NUS_GPU" | grep -o "[0-9]* MiB" | grep -o "[0-9]*")
case "$CARD" in
  h200) [[ "$NUS_GPU" == *H200* ]] || { echo "card mismatch: $NUS_GPU"; exit 4; } ;;
  *) echo "CARD must be h200 (protocol_maskinfo_confirm.md: h200 only)"; exit 4 ;;
esac
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); echo "DRIVER=$DRV"
if [ -n "${EXPECT_DRIVER:-}" ] && [ "$DRV" != "$EXPECT_DRIVER" ]; then echo "DRIVER changed: $DRV != $EXPECT_DRIVER (redo the bit-exact check first)"; exit 7; fi
GLR=$(source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T; python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-)
echo "GL_RENDERER=$GLR"
[[ "$GLR" == *H200* ]] || { echo "GL_RENDERER is not H200: $GLR"; exit 5; }
case "$MODEL:$SUITE" in
  oft:spatial|oft:object) CKPT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10; export HF_HOME=$HOME/hf_cache ;;
  ofti:spatial) CKPT=moojink/openvla-7b-oft-finetuned-libero-spatial; export HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache ;;
  ofti:object) CKPT=moojink/openvla-7b-oft-finetuned-libero-object; export HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache ;;
  *) echo "bad MODEL:SUITE"; exit 4 ;;
esac
export MI_BLOCK=${MODEL}_${SUITE}_c30 MI_PSET=$T/lists/mi_pset.json
eval "$($T/tools/mi_cells.sh params)"   # MI_LAG_STEPS MI_OUTLINE_W MI_PROTECT (frozen values)
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T
OUTROOT=${OUTROOT:-$T/out/runs}; NTRIALS=${NTRIALS:-5}
mkdir -p $T/logs/eval
echo "node=$(hostname) gpu=$NUS_GPU card=$CARD block=$MI_BLOCK ckpt=$CKPT JOBS=$JOBS lag=$MI_LAG_STEPS width=$MI_OUTLINE_W protect=$MI_PROTECT NTRIALS=$NTRIALS MAX_TASKS=${MAX_TASKS:-} start=$(date)"
echo "sha256: $(sha256sum embody/maskinfo.py embody/maskrepl.py embody/body_env.py oft/run_libero_eval_mi_c.py tools/mi_cells.sh lists/mi_pset.json | awk '{print $1" "$2}' | tr '\n' ' ')"
PIDS=(); LOGS=()
for it in $JOBS; do
  c=${it%%:*}; off=${it#*:}
  L=$T/logs/mic_${MI_BLOCK}_${c}_e${off}_${SLURM_JOB_ID}.log; mkdir -p $OUTROOT/$MI_BLOCK/$c
  echo "launch $c e$off"
  env BODY=Panda COPY_GRIP_CMD=1 MI_CELL=$c DUMP_DIR=$OUTROOT/$MI_BLOCK/$c EP_OFFSET=$off \
    MR_METRICS=$OUTROOT/$MI_BLOCK/$c/maskmetrics_e${off}_${SLURM_JOB_ID}.jsonl ${MAX_TASKS:+MAX_TASKS=$MAX_TASKS} \
    python -X faulthandler oft/run_libero_eval_mi_c.py --pretrained_checkpoint $CKPT \
    --task_suite_name libero_$SUITE --num_trials_per_task $NTRIALS --run_id_note mi_${MI_BLOCK}_${c}_e${off} \
    --local_log_dir $T/logs/eval > $L 2>&1 &
  PIDS+=($!); LOGS+=($L); sleep 20
done
wait "${PIDS[@]}"
i=0; for it in $JOBS; do L=${LOGS[$i]}; echo "$it: started $(grep -c "Starting episode" $L) episode_errors $(grep -c "Episode error" $L) dump_errors $(grep -c "Dump error" $L) tracebacks $(grep -c Traceback $L)"; i=$((i+1)); done
echo "end=$(date)"
