#!/bin/bash
#SBATCH --job-name=pg_pi_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# phasegate cells for the pi family, EXPLORATION tier (not pre-registered; small n; used to choose what to confirm).
# Adapted from $PROJ_ROOT/maskinfo_pi/tools/mi_pi_run.sbatch: ONE card, one fixed-noise openpi server
# (pi/serve_policy_mi.py, XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC) + one blind client (pi/zoo_client_pg.py) per entry of
# JOBS, all on the same card. Physical body is always the stock Panda (BODY=Panda COPY_GRIP_CMD=1). Noise key and
# MEMFRAC from tools/mi_pi_params.sh (unchanged copy of the maskinfo-pi file), so cells of a block share the noise per key.
# Env (as tools/pg_run.sbatch): MODEL (pi05|pi0) SUITE (spatial|object) JOBS="mode:window ..." (mode look|erase,
# window none|approach|grasp|transport|all); EP_OFFSET (default 0) NTRIALS (default 5) -> keys = tasks x episodes
# EP_OFFSET..+NTRIALS-1; MAX_TASKS (tasks 0..MAX_TASKS-1, default all); SKIP_DONE passthrough; OUTROOT.
# Dumps: $OUTROOT/${MODEL}_${SUITE}/${mode}_${window}/.
# CARD (default h200): h200, a100-80 or a100-40 (card and EGL GL_RENDERER asserted, PG_GL_ALLOW set accordingly); for
#   an A100 submit with sbatch --gres=gpu:a100-80:1 (or a100-40:1) --exclude=node39,node38 --export=ALL,CARD=...
#   (overrides the template's gres / exclude; node39 is ARM64, node29 is H100). At most 10 clients per job (one per cell).
# Never prints success, per-episode wall clock, steps or query counts.
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 HIDE_ARM=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache
export OPENPI_DATA_HOME=$HOME/.cache/openpi; unset DISPLAY
T=$PROJ_ROOT/phasegate_confirm; cd $T
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
export NUS_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, /|/')"
CARD=${CARD:-h200}
case "$CARD" in
  h200) export PG_GL_ALLOW=H200; [[ "$NUS_GPU" == *H200* ]] || { echo "card mismatch: CARD=$CARD $NUS_GPU"; exit 4; } ;;
  a100-80) export PG_GL_ALLOW=A100; [[ "$NUS_GPU" == *A100*80GB* ]] || { echo "card mismatch: CARD=$CARD $NUS_GPU"; exit 4; } ;;
  a100-40) export PG_GL_ALLOW=A100; [[ "$NUS_GPU" == *A100*40GB* ]] || { echo "card mismatch: CARD=$CARD $NUS_GPU"; exit 4; } ;;
  *) echo "bad CARD $CARD"; exit 4 ;;
esac
NJ=$(echo $JOBS | wc -w); [ "$NJ" -le 10 ] || { echo "at most 10 clients per job, JOBS has $NJ"; exit 4; }
source $HOME/miniforge3/etc/profile.d/conda.sh
GLR=$(conda activate oftpi; python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-)
echo "GL_RENDERER=$GLR"
[[ "$GLR" == *"$PG_GL_ALLOW"* ]] || { echo "ASSERT GL_RENDERER is not $PG_GL_ALLOW: $GLR"; exit 5; }
eval "$(bash $T/tools/mi_pi_params.sh)"  # unchanged copy of $PROJ_ROOT/maskinfo_pi/tools/mi_pi_params.sh
case "$MODEL:$SUITE" in
  pi05:spatial|pi05:object) CFG=pi05_libero; CKPT=$PROJ_ROOT/pi05/ckpt/pi05_libero ;;
  pi0:spatial|pi0:object) CFG=pi0_libero; CKPT=$PROJ_ROOT/vla_zoo/openpi_ckpt/pi0_libero ;;
  *) echo "bad MODEL:SUITE $MODEL:$SUITE"; exit 4 ;;
esac
export MI_MODEL_TAG=$MODEL EMBODY_DIR=$T COPY_GRIP_CMD=1
BLOCK=${MODEL}_${SUITE}; OUTROOT=${OUTROOT:-$T/out/runs}; NTRIALS=${NTRIALS:-5}; OFF=${EP_OFFSET:-0}
TASKS=all; [ -n "${MAX_TASKS:-}" ] && TASKS=$(seq -s, 0 $((MAX_TASKS - 1)))
PORT=$((20000 + SLURM_JOB_ID % 20000)); RID=$SLURM_JOB_ID; mkdir -p $T/logs/clients $T/logs/server $OUTROOT
echo "node=$(hostname) gpu=$NUS_GPU card=$CARD gl_allow=$PG_GL_ALLOW block=$BLOCK cfg=$CFG ckpt=$CKPT JOBS=$JOBS EP_OFFSET=$OFF NTRIALS=$NTRIALS TASKS=$TASKS OUTROOT=$OUTROOT start=$(date)"
echo "params: noise_master=$MI_NOISE_MASTER memfrac=$MEMFRAC"
echo "sha256: $(sha256sum embody/phasegate.py embody/body_env.py pi/zoo_client_pg.py pi/serve_policy_mi.py tools/mi_pi_params.sh tools/pg_pi_run.sbatch | awk '{print substr($1,1,16)" "$2}' | tr '\n' ' ')"
SLOG=$T/logs/server/server_pg_${BLOCK}_$RID.log
( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC exec .venv/bin/python $T/pi/serve_policy_mi.py \
    --config $CFG --dir $CKPT --port $PORT --state-log $T/logs/server/state_pg_${BLOCK}_$RID.jsonl ) > $SLOG 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:$PORT/healthz >/dev/null 2>&1; do
  sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died"; grep -m5 -A3 "Traceback\|Error" $SLOG; exit 1; }; done
export NUS_SRV_HOST=$(hostname) NUS_SRV_GPU="$NUS_GPU"
echo "server up :$PORT $(date)"; grep -m1 "^MI_SERVER_CFG" $SLOG | cut -c1-400; grep -m1 "^MI_STATE_PATH" $SLOG
( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $T/logs/server/gpu_pg_$RID.tsv 2>&1 &
LPID=$!
conda activate oftpi
PIDS=(); LOGS=()
for it in $JOBS; do
  mode=${it%%:*}; win=${it#*:}; c=${mode}_${win}
  case "$mode" in
    look) EX="TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk" ;;
    erase) EX="" ;;
    *) echo "bad mode $mode"; exit 4 ;;
  esac
  L=$T/logs/clients/pg_${BLOCK}_${c}_e${OFF}_${RID}.log; D=$OUTROOT/$BLOCK/$c; mkdir -p $D
  echo "launch $c e$OFF"
  env BODY=Panda PG_MODE=$mode PG_WINDOW=$win $EX DUMP_DIR=$D ${SKIP_DONE:+SKIP_DONE=$SKIP_DONE} \
    python -X faulthandler pi/zoo_client_pg.py --proto openpi --blind --body Panda --model $CFG --suite libero_$SUITE \
    --host 127.0.0.1 --port $PORT --trials $NTRIALS --ep-offset $OFF --tasks $TASKS \
    --out $D/blind_e${OFF}_${RID}.jsonl > $L 2>&1 &
  PIDS+=($!); LOGS+=($L); sleep 3
done
wait "${PIDS[@]}"
i=0; for it in $JOBS; do L=${LOGS[$i]}; c=${it%%:*}_${it#*:}; echo "$it: episodes_logged $(grep -c '^EP ' $L) ep_error_not_none $(grep '^EP ' $L | grep -vc 'error=None') dumps $(ls $OUTROOT/$BLOCK/$c/t*_e*_i*.npz 2>/dev/null | wc -l) tracebacks $(grep -c Traceback $L) asserts $(grep -c 'AssertionError\|ASSERT GL_RENDERER' $L) gl_inproc $(grep -m1 '^PG_GL_RENDERER' $L | cut -c16-) dump_errors $(grep -c DUMP_ERROR $L) phasegate_lines $(grep -c 'PHASEGATE:' $L) pg_cfg $(grep -c '^PG_CFG' $L)"; i=$((i+1)); done
echo "server tracebacks $(grep -c Traceback $SLOG) $(grep -m1 '^MI_IMAGE_MASK' $SLOG) state_log_lines $(wc -l < $T/logs/server/state_pg_${BLOCK}_$RID.jsonl)"
kill $SPID $LPID 2>/dev/null; echo "end=$(date)"
