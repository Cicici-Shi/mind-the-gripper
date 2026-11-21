#!/bin/bash
#SBATCH --job-name=zoo_srvN
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
# models1 (2026-11-04): NSRV independent policy-server processes of the SAME model on one GPU (ports PORT..PORT+NSRV-1),
# for single-step models whose throughput is bound by one server process (OpenVLA: batch-1 autoregressive, ~170 ms per
# query, one H100 mostly idle). Writes $PROJ_ROOT/vla_zoo/servers/${MODEL}_<i>.addr "host port jobid gpu"; clients use
# SERVER=remote:${MODEL}_<i>. Same SERVER syntax as run_server.sbatch (zoo:<env>:<adapter>[:k=v,...] only).
set -u
export TMPDIR=$HOME/tmp XDG_CACHE_HOME=$HOME/.cache HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache PYTHONDONTWRITEBYTECODE=1
Z=$PROJ_ROOT/vla_zoo; mkdir -p $Z/servers
PORT=$((20000 + SLURM_JOB_ID % 20000)); GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "node=$(hostname) gpu=$GPU MODEL=$MODEL NSRV=$NSRV SERVER=$SERVER start=$(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh
IFS=: read -r kind a b c <<< "$SERVER"
conda activate $a
PIDS=""
for i in $(seq 0 $((NSRV - 1))); do
  python $Z/common/zoo_server.py --adapter $b --port $((PORT + i)) --adapter-args ${c//,/ } > $Z/logs/srv_${MODEL}_${SLURM_JOB_ID}_$i.log 2>&1 &
  PIDS="$PIDS $!"; sleep 30
done
for i in $(seq 0 $((NSRV - 1))); do
  until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$((PORT + i)))))" 2>/dev/null; do sleep 5; kill -0 $(echo $PIDS | cut -d" " -f$((i + 1))) 2>/dev/null || { echo "server $i died"; exit 1; }; done
  echo "$(hostname) $((PORT + i)) $SLURM_JOB_ID $GPU" > $Z/servers/${MODEL}_$i.addr; echo "server $i listening $(date)"
done
( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $Z/logs/thr_srv_$SLURM_JOB_ID.tsv 2>&1 &
wait $PIDS
