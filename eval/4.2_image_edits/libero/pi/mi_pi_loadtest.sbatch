#!/bin/bash
#SBATCH --job-name=mipi_load
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi sustained load test (global rule: each level held 60-90 s, effective throughput + errors). Same card,
# server, client and hooks as tools/mi_pi_run.sbatch. Per level N in LEVELS: N blind clients (cells cycle C0..C8, keys
# Spatial ep 40-49 = outside the formal keys, task i mod 10), wait until all N are stepping, WARM s more, then a
# WINDOW s measurement: total executed control steps / s over all clients and new errors (tools/tput_agg.py). The
# clients are then killed and their dumps deleted unread. Nothing per episode is printed.
# Env: MODELS ("pi05 pi0"), LEVELS ("4 8 12 16"), WINDOW (75), WARM (20)
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 HIDE_ARM=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache
export OPENPI_DATA_HOME=$HOME/.cache/openpi MI_GL_EXPECT=H200; unset DISPLAY
T=$PROJ_ROOT/maskinfo_pi; cd $T
export NUS_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, /|/')"
[[ "$NUS_GPU" == *H200* ]] || { echo "ASSERT card is not H200: $NUS_GPU"; exit 4; }
source $HOME/miniforge3/etc/profile.d/conda.sh
GLR=$(conda activate oftpi; python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-)
echo "GL_RENDERER=$GLR"; [[ "$GLR" == *H200* ]] || { echo "ASSERT GL_RENDERER is not H200: $GLR"; exit 5; }
eval "$(bash $T/tools/mi_pi_params.sh)"
export MI_PSET=$T/lists/mi_pset.json EMBODY_DIR=$T COPY_GRIP_CMD=1
MODELS=${MODELS:-"pi05 pi0"}; LEVELS=${LEVELS:-"4 8 12 16"}; WINDOW=${WINDOW:-75}; WARM=${WARM:-20}
SCR=$T/out/loadtest_scratch_$SLURM_JOB_ID; LG=$T/logs/load_$SLURM_JOB_ID; mkdir -p $LG
echo "node=$(hostname) gpu=$NUS_GPU cpus=$SLURM_CPUS_PER_TASK MODELS=$MODELS LEVELS=$LEVELS WINDOW=$WINDOW start=$(date)"
for MODEL in $MODELS; do
  case $MODEL in pi05) CFG=pi05_libero; CKPT=$PROJ_ROOT/pi05/ckpt/pi05_libero ;; pi0) CFG=pi0_libero; CKPT=$PROJ_ROOT/vla_zoo/openpi_ckpt/pi0_libero ;; esac
  PORT=$((20000 + (SLURM_JOB_ID * 7 + ${#MODEL}) % 20000)); SLOG=$LG/server_$MODEL.log
  ( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC exec .venv/bin/python $T/pi/serve_policy_mi.py \
      --config $CFG --dir $CKPT --port $PORT --state-log $LG/state_$MODEL.jsonl ) > $SLOG 2>&1 &
  SPID=$!
  until curl -sf http://127.0.0.1:$PORT/healthz >/dev/null 2>&1; do
    sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died"; exit 1; }; done
  export NUS_SRV_HOST=$(hostname) NUS_SRV_GPU="$NUS_GPU" MI_BLOCK=${MODEL}_spatial MI_MODEL_TAG=$MODEL
  conda activate oftpi
  for N in $LEVELS; do
    PIDS=(); TF=()
    for i in $(seq 0 $((N - 1))); do
      c=C$((i % 9)); tf=$LG/tput_${MODEL}_n${N}_$i.txt; TF+=($tf)
      env MI_CELL=$c DUMP_DIR=$SCR/$MODEL/n$N/$i MI_TPUT_FILE=$tf \
        python -X faulthandler pi/zoo_client_mi.py --proto openpi --blind --body Panda --model $CFG --suite libero_spatial \
        --host 127.0.0.1 --port $PORT --trials 10 --ep-offset 40 --tasks $((i % 10)) \
        --out $SCR/$MODEL/n$N/blind_$i.jsonl > $LG/c_${MODEL}_n${N}_$i.log 2>&1 &
      PIDS+=($!)
    done
    w=0; until python3 tools/tput_agg.py --ready $N "${TF[@]}"; do sleep 5; w=$((w + 5)); [ $w -gt 600 ] && { echo "LOADTEST $MODEL n=$N not all clients stepping after 600 s"; break; }; done
    sleep $WARM; t0=$(date +%s.%N)
    ( for k in $(seq 1 $((WINDOW / 15))); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; sleep 15; done ) > $LG/gpu_${MODEL}_n$N.txt &
    sleep $WINDOW; sleep 6; t1=$(python3 -c "import time; print(time.time() - 6)")
    echo "LOADTEST model=$MODEL n=$N $(python3 tools/tput_agg.py $t0 $t1 "${TF[@]}") gpu_util_mean=$(awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n}' $LG/gpu_${MODEL}_n$N.txt) tracebacks=$(cat $LG/c_${MODEL}_n${N}_*.log | grep -c Traceback)"
    kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null
    rm -rf $SCR/$MODEL/n$N   # dumps deleted unread
  done
  conda deactivate
  echo "server $MODEL tracebacks $(grep -c Traceback $SLOG)"
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
done
rm -rf $SCR; echo "end=$(date)"
