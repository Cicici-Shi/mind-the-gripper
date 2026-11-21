#!/bin/bash
#SBATCH --job-name=zoo_srv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# vla_zoo (2026-11-03): policy server ONLY (no rendering -> may run on H100 node29 nodes). Writes "host port" to
# $PROJ_ROOT/vla_zoo/servers/$MODEL.addr once listening; render clients (run_zoo.sbatch with SERVER=remote:$MODEL)
# on other nodes connect over TCP. Same SERVER syntax as run_zoo.sbatch.
set -u
export TMPDIR=$HOME/tmp XDG_CACHE_HOME=$HOME/.cache HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache OPENPI_DATA_HOME=$HOME/.cache/openpi PYTHONDONTWRITEBYTECODE=1
Z=$PROJ_ROOT/vla_zoo; mkdir -p $Z/servers
PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) MODEL=$MODEL SERVER=$SERVER start=$(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh
IFS=: read -r kind a b c <<< "$SERVER"
if [ "$kind" = openpi ]; then
  cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.5} .venv/bin/python scripts/serve_policy.py --port $PORT \
      policy:checkpoint --policy.config=$a --policy.dir=$b &
else
  conda activate $a
  python $Z/common/zoo_server.py --adapter $b --port $PORT --adapter-args ${c//,/ } &
fi
SPID=$!
until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; done
echo "$(hostname) $PORT $SLURM_JOB_ID $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > $Z/servers/$MODEL.addr; echo "listening $(date)"
( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $Z/logs/thr_srv_$SLURM_JOB_ID.tsv 2>&1 &
wait $SPID
