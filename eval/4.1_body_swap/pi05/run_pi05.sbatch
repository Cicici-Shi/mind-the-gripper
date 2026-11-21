#!/bin/bash
#SBATCH --job-name=pi05_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x_%j.log
# One job: openpi policy server (uv venv, background) + LIBERO client (conda oftpi), then state-sensitivity.
# PHASES: any of smoke,base,sens (comma list). TRIALS per task for base (default 2).
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache PIP_CACHE_DIR=$HOME/.cache/pip
export OPENPI_DATA_HOME=$HOME/.cache/openpi
export MUJOCO_GL=egl; unset DISPLAY
P=$PROJ_ROOT/pi05; CKPT=$P/ckpt/pi05_libero; OP=$P/openpi
PHASES=${PHASES:-smoke,base,sens}; TRIALS=${TRIALS:-2}
PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) port=$PORT phases=$PHASES start=$(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh

start_server() {
  cd $OP
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python scripts/serve_policy.py --port $PORT \
     policy:checkpoint --policy.config=pi05_libero --policy.dir=$CKPT > $P/logs/server_$SLURM_JOB_ID.log 2>&1 &
  SPID=$!
  T0=$(date +%s)
  until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex((\"127.0.0.1\",$PORT)))" 2>/dev/null; do
    sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -50 $P/logs/server_$SLURM_JOB_ID.log; exit 1; }
  done
  echo "server up after $(( $(date +%s) - T0 )) s"
}

if [[ $PHASES == *smoke* || $PHASES == *base* ]]; then
  start_server
  conda activate oftpi
  cd $P
  if [[ $PHASES == *smoke* ]]; then
    T0=$(date +%s)
    python pi05_libero_eval.py --port $PORT --tasks 0 --trials 1 --out $P/results_smoke.jsonl --dump-obs $P/obs_dump_smoke
    echo "smoke exit=$? wall=$(( $(date +%s) - T0 )) s"
  fi
  if [[ $PHASES == *base* ]]; then
    T0=$(date +%s)
    python pi05_libero_eval.py --port $PORT --trials $TRIALS --out $P/results_src${TRIALS}0.jsonl --dump-obs $P/obs_dump
    echo "base exit=$? wall=$(( $(date +%s) - T0 )) s"
  fi
  conda deactivate
  kill $SPID; wait $SPID 2>/dev/null
fi
if [[ $PHASES == *sens* ]]; then
  cd $OP
  .venv/bin/python $P/state_sensitivity.py $CKPT $P/obs_dump $P/state_sensitivity.jsonl 12
  echo "sens exit=$?"
fi
echo "end=$(date)"
