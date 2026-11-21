#!/bin/bash
#SBATCH --job-name=mipi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi job template (protocol_maskinfo_pi.md; Astra rulings 2026-11-08). ONE whole h200: one openpi policy
# server (pi/serve_policy_mi.py, fixed noise, XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC) + one blind client per entry
# of JOBS, all on the same card. The card (nvidia-smi) and the EGL GL_RENDERER of the client env are asserted to be
# H200 BEFORE anything runs. Physical Panda (BODY=Panda COPY_GRIP_CMD=1), parameters from tools/mi_pi_params.sh.
# Env: MODEL (pi0|pi05) SUITE (spatial|object) JOBS="cell:offset[:tag] ..." NTRIALS (5) TASKS (all) OUTROOT (out/runs)
#      SKIP_DONE passthrough (non-formal roots only); POST_PROBE=1 (smoke only) runs tools/state_probe_pi.py after the clients.
# Never prints success, per-episode wall clock, steps or query counts.
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 HIDE_ARM=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache
export OPENPI_DATA_HOME=$HOME/.cache/openpi MI_GL_EXPECT=H200; unset DISPLAY
T=$PROJ_ROOT/maskinfo_pi; cd $T
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
export NUS_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, /|/')"
[[ "$NUS_GPU" == *H200* ]] || { echo "ASSERT card is not H200: $NUS_GPU"; exit 4; }
source $HOME/miniforge3/etc/profile.d/conda.sh
GLR=$(conda activate oftpi; python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-)
echo "GL_RENDERER=$GLR"
[[ "$GLR" == *H200* ]] || { echo "ASSERT GL_RENDERER is not H200: $GLR"; exit 5; }
eval "$(bash $T/tools/mi_pi_params.sh)"
case "$MODEL:$SUITE" in
  pi05:spatial|pi05:object) CFG=pi05_libero; CKPT=$PROJ_ROOT/pi05/ckpt/pi05_libero ;;
  pi0:spatial|pi0:object) CFG=pi0_libero; CKPT=$PROJ_ROOT/vla_zoo/openpi_ckpt/pi0_libero ;;
  *) echo "bad MODEL:SUITE $MODEL:$SUITE"; exit 4 ;;
esac
export MI_MODEL_TAG=$MODEL MI_BLOCK=${MODEL}_${SUITE} MI_PSET=$T/lists/mi_pset.json EMBODY_DIR=$T COPY_GRIP_CMD=1
OUTROOT=${OUTROOT:-$T/out/runs}; NTRIALS=${NTRIALS:-5}; TASKS=${TASKS:-all}; CLIENT_ARGS=${CLIENT_ARGS:-}
mkdir -p $OUTROOT; FORMAL=0; [ "$(realpath -m $OUTROOT)" = "$(realpath -m $T/out/runs)" ] && FORMAL=1
if [ $FORMAL = 1 ]; then  # formal output: frozen set intact, no debug switches (review 11-08 items 1, 11)
  [ -f $T/out/FROZEN ] && sha256sum -c --quiet $T/out/FROZEN || { echo "ASSERT formal run needs an intact out/FROZEN"; exit 4; }
  [ -z "${DEBUG_STATE_TRANSFORM:-}" ] || { echo "ASSERT debug state transform not allowed for out/runs"; exit 4; }
  [ -z "$CLIENT_ARGS" ] || { echo "ASSERT CLIENT_ARGS (debug only) not allowed for out/runs"; exit 4; }
  [ -z "${POST_PROBE:-}" ] || { echo "ASSERT POST_PROBE (smoke only) not allowed for out/runs"; exit 4; }
  # one shard = one server process: never splice a shard across processes (protocol 2.4 / 6). Names only, nothing opened.
  [ -z "${SKIP_DONE:-}" ] || { echo "ASSERT SKIP_DONE not allowed for out/runs (incomplete shard: tools/rerun_shard_mi_pi.sh)"; exit 4; }
  for it in $JOBS; do IFS=: read -r c off _t <<< "$it"
    for e in $(seq $off $((off + NTRIALS - 1))); do
      compgen -G "$OUTROOT/$MI_BLOCK/$c/t*_e${e}_i*.npz" >/dev/null && { echo "ASSERT $MI_BLOCK/$c ep $e already has output: move the whole shard with tools/rerun_shard_mi_pi.sh"; exit 4; }
    done
  done
fi
PORT=$((20000 + (SLURM_JOB_ID + ${PORT_OFFSET:-0}) % 20000)); RID=${SLURM_JOB_ID}${PORT_OFFSET:+_p$PORT_OFFSET}; mkdir -p $T/logs/clients $T/logs/server
echo "node=$(hostname) gpu=$NUS_GPU block=$MI_BLOCK cfg=$CFG ckpt=$CKPT JOBS=$JOBS NTRIALS=$NTRIALS TASKS=$TASKS OUTROOT=$OUTROOT CLIENT_ARGS=$CLIENT_ARGS start=$(date)"
echo "params: every=$MI_QUERY_EVERY lag=$MI_LAG_STEPS width=$MI_OUTLINE_W protect=$MI_PROTECT prep=$MI_PREP noise_master=$MI_NOISE_MASTER memfrac=$MEMFRAC"
echo "sha256: $(sha256sum embody/maskinfo.py embody/maskrepl.py embody/maskinfo_pi.py embody/body_env.py pi/zoo_client_mi.py pi/serve_policy_mi.py tools/mi_pi_params.sh tools/mi_pi_run.sbatch lists/mi_pset.json | awk '{print substr($1,1,16)" "$2}' | tr '\n' ' ')"
SLOG=$T/logs/server/server_${MI_BLOCK}_$RID.log
( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC exec .venv/bin/python $T/pi/serve_policy_mi.py \
    --config $CFG --dir $CKPT --port $PORT --state-log $T/logs/server/state_${MI_BLOCK}_$RID.jsonl \
    ${DEBUG_STATE_TRANSFORM:+--debug-state-transform} ) > $SLOG 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:$PORT/healthz >/dev/null 2>&1; do
  sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died"; grep -m5 -A3 "Traceback\|Error" $SLOG; exit 1; }; done
export NUS_SRV_HOST=$(hostname) NUS_SRV_GPU="$NUS_GPU"
echo "server up :$PORT $(date)"; grep -m1 "^MI_SERVER_CFG" $SLOG; grep -m1 "^MI_STATE_PATH" $SLOG
( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $T/logs/server/gpu_$RID.tsv 2>&1 &
LPID=$!
conda activate oftpi
PIDS=(); LOGS=()
for it in $JOBS; do
  IFS=: read -r c off tag <<< "$it"   # optional tag (smoke only): dumps go to ${OUTROOT}_<tag>/ (e.g. rerun)
  [ -n "$tag" ] && [ $FORMAL = 1 ] && { echo "ASSERT tagged entry $it not allowed for out/runs"; exit 4; }
  R=$OUTROOT; [ -n "$tag" ] && R=${OUTROOT}_$tag
  L=$T/logs/clients/c_${MI_BLOCK}_${c}_e${off}${tag:+_$tag}_${RID}.log; D=$R/$MI_BLOCK/$c; mkdir -p $D
  env MI_CELL=$c DUMP_DIR=$D ${SKIP_DONE:+SKIP_DONE=$SKIP_DONE} \
    python -X faulthandler pi/zoo_client_mi.py --proto openpi --blind --body Panda --model $CFG --suite libero_$SUITE \
    --host 127.0.0.1 --port $PORT --trials $NTRIALS --ep-offset $off --tasks $TASKS \
    --out $D/blind_e${off}_${RID}.jsonl ${CLIENT_ARGS:-} > $L 2>&1 &
  PIDS+=($!); LOGS+=($L); sleep 3
done
wait "${PIDS[@]}"
i=0; for it in $JOBS; do L=${LOGS[$i]}; echo "$it: episodes_logged $(grep -c '^EP ' $L) ep_error_not_none $(grep '^EP ' $L | grep -vc 'error=None') tracebacks $(grep -c Traceback $L) asserts $(grep -c 'AssertionError\|LAG_ASSERT\|ASSERT GL_RENDERER' $L) gl_inproc $(grep -m1 '^MI_GL_RENDERER' $L | cut -c16-) dump_errors $(grep -c DUMP_ERROR $L) mi_cfg $(grep -c '^MI_CFG' $L)"; i=$((i+1)); done
if [ "${POST_PROBE:-0}" = 1 ]; then
  f=$(ls $OUTROOT/$MI_BLOCK/C0/t*_e*_i*.npz 2>/dev/null | head -1)
  exp=used; [ "$MODEL" = pi05 ] && exp=unused
  [ -n "$f" ] && python -I tools/state_probe_pi.py --port $PORT --npz $f --suite libero_$SUITE --expect-state $exp
fi
echo "server tracebacks $(grep -c Traceback $SLOG) $(grep -m1 '^MI_IMAGE_MASK' $SLOG) state_log_lines $(wc -l < $T/logs/server/state_${MI_BLOCK}_$RID.jsonl)"
kill $SPID $LPID 2>/dev/null; echo "end=$(date)"
