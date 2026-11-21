#!/bin/bash
#SBATCH --job-name=mipi_det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi determinism probe (see tools/determinism_probe_pi.py): for each model four server processes on this h200:
# cold, warm, debug (--debug-state-transform), load (8 real blind clients on Spatial ep 40-49, dumps deleted unread).
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 HIDE_ARM=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache
export OPENPI_DATA_HOME=$HOME/.cache/openpi MI_GL_EXPECT=H200; unset DISPLAY
T=$PROJ_ROOT/maskinfo_pi; cd $T
export NUS_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, /|/')"
[[ "$NUS_GPU" == *H200* ]] || { echo "ASSERT card is not H200: $NUS_GPU"; exit 4; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oftpi
GLR=$(python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-); echo "GL_RENDERER=$GLR"; [[ "$GLR" == *H200* ]] || exit 5
eval "$(bash tools/mi_pi_params.sh)"
export MI_PSET=$T/lists/mi_pset.json EMBODY_DIR=$T COPY_GRIP_CMD=1 NUS_SRV_GPU="$NUS_GPU" NUS_SRV_HOST=$(hostname)
O=$T/out/det_probe_$SLURM_JOB_ID; mkdir -p $O/res $O/logs
i=0
for MODEL in pi05 pi0; do
  case $MODEL in pi05) CFG=pi05_libero; CKPT=$PROJ_ROOT/pi05/ckpt/pi05_libero ;; pi0) CFG=pi0_libero; CKPT=$PROJ_ROOT/vla_zoo/openpi_ckpt/pi0_libero ;; esac
  NPZ=$(ls $T/out/smoke_runs_v2/${MODEL}_spatial/C0/t0_e40_i*.npz | head -1)
  for mode in cold warm debug load; do
    i=$((i + 1)); PORT=$((20000 + (SLURM_JOB_ID + 100 + i) % 20000)); D=""; [ $mode = debug ] && D="--debug-state-transform"
    ( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC exec .venv/bin/python $T/pi/serve_policy_mi.py \
        --config $CFG --dir $CKPT --port $PORT --state-log $O/logs/state_${MODEL}_$mode.jsonl $D ) > $O/logs/server_${MODEL}_$mode.log 2>&1 &
    SPID=$!
    until curl -sf http://127.0.0.1:$PORT/healthz >/dev/null 2>&1; do sleep 3; kill -0 $SPID 2>/dev/null || { echo "server died"; exit 1; }; done
    PIDS=()
    if [ $mode = load ]; then
      TF=()
      for k in 0 1 2 3 4 5 6 7; do
        tf=$O/logs/tput_${MODEL}_$k.txt; TF+=($tf)
        env MI_CELL=C$k DUMP_DIR=$O/scratch/$MODEL/$k MI_TPUT_FILE=$tf MI_BLOCK=${MODEL}_spatial MI_MODEL_TAG=$MODEL \
          python -X faulthandler pi/zoo_client_mi.py --proto openpi --blind --body Panda --model $CFG --suite libero_spatial \
          --host 127.0.0.1 --port $PORT --trials 10 --ep-offset 40 --tasks $k --out $O/scratch/blind_${MODEL}_$k.jsonl > $O/logs/c_${MODEL}_$k.log 2>&1 &
        PIDS+=($!)
      done
      mkdir -p $O/scratch/$MODEL
      w=0; until python3 tools/tput_agg.py --ready 8 "${TF[@]}"; do sleep 5; w=$((w + 5)); [ $w -gt 300 ] && { echo "clients not stepping"; break; }; done
      sleep 15
    fi
    python -I tools/determinism_probe_pi.py --mode $mode --port $PORT --npz $NPZ --out $O/res/${MODEL}_$mode.npz
    [ $mode = load ] && echo "load clients tracebacks $(cat $O/logs/c_${MODEL}_*.log | grep -c Traceback) gpu_util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
    [ ${#PIDS[@]} -gt 0 ] && { kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; }
    echo "$MODEL $mode server tracebacks $(grep -c Traceback $O/logs/server_${MODEL}_$mode.log) image_mask_lines $(grep -c '^MI_IMAGE_MASK' $O/logs/server_${MODEL}_$mode.log)"
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  done
done
rm -rf $O/scratch
python -I tools/determinism_probe_pi.py --compare $O/res | tee $O/compare.txt
echo "end=$(date)"
