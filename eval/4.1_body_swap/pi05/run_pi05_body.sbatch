#!/bin/bash
#SBATCH --job-name=pi05_body
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:50:00
#SBATCH --output=logs/%x_%j.log
# Server + client; CONDS="name:BODY:HIDE ..." run sequentially, each TRIALS per task from EP_OFFSET.
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache PIP_CACHE_DIR=$HOME/.cache/pip
export OPENPI_DATA_HOME=$HOME/.cache/openpi MUJOCO_GL=egl PYTHONFAULTHANDLER=1; unset DISPLAY
P=$PROJ_ROOT/pi05; CKPT=$P/ckpt/pi05_libero; OP=$P/openpi
PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) CONDS=$CONDS TRIALS=$TRIALS EP_OFFSET=$EP_OFFSET start=$(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh
cd $OP
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=pi05_libero --policy.dir=$CKPT > $P/logs/server_$SLURM_JOB_ID.log 2>&1 &
SPID=$!
until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; done
conda activate oftpi; cd $P
for c in $CONDS; do
  IFS=: read name body hide <<< "$c"
  export HIDE_ARM=$hide
  bodyarg=""; [ "$body" != "SRC" ] && bodyarg="--body $body"
  python -X faulthandler pi05_body_eval.py --port $PORT --trials $TRIALS --ep-offset $EP_OFFSET $bodyarg --out $P/results_${name}.jsonl
  echo "$name exit=$?"
done
kill $SPID; echo "end=$(date)"
