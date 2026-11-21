#!/bin/bash
#SBATCH --job-name=nz_srv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
# next (2026-11-06): copy of vla_zoo/src_next/run_servers2.sbatch (several policy servers on ONE GPU). Diffs: names must
# start with nz_ (addr files servers/nz_*.addr never collide with models1 / src_next), logs in logs/next/.
# SERVERS="nz_a=SERVER1|nz_b=SERVER2"; SERVER = openpiz:<config>:<ckpt> (z-score wrapper src_next/serve_policy_zscore.py,
# REQUIRED for pi0_fast_libero, openpi #849) | openpi:<config>:<ckpt> | zoo:<conda env>:<adapter.py>[:k=v,...].
# Addr file: "host port jobid gpu". Ends when any server exits or at the time limit (clients scancel it when done).
set -u
export TMPDIR=$HOME/tmp XDG_CACHE_HOME=$HOME/.cache HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache OPENPI_DATA_HOME=$HOME/.cache/openpi PYTHONDONTWRITEBYTECODE=1
Z=$PROJ_ROOT/vla_zoo; LG=$Z/logs/next; mkdir -p $Z/servers $LG
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "node=$(hostname) gpu=$GPU SERVERS=$SERVERS start=$(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh
i=0; PIDS=""
IFS="|" read -ra SV <<< "$SERVERS"
for spec in "${SV[@]}"; do
  name=${spec%%=*}; srv=${spec#*=}; PORT=$((20000 + (SLURM_JOB_ID * 3 + i) % 20000)); i=$((i + 1))
  case "$name" in nz_*) ;; *) echo "name must start with nz_: $name"; exit 2;; esac
  IFS=: read -r kind a b c <<< "$srv"
  if [ "$kind" = openpiz ]; then   # z-score wrapper (pi0-FAST LIBERO, openpi #849)
    ( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.5} exec .venv/bin/python $PROJ_ROOT/vla_zoo/src_next/serve_policy_zscore.py --port $PORT \
        policy:checkpoint --policy.config=$a --policy.dir=$b ) > $LG/server_${name}_$SLURM_JOB_ID.log 2>&1 &
  elif [ "$kind" = openpi ]; then
    ( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.5} exec .venv/bin/python scripts/serve_policy.py --port $PORT \
        policy:checkpoint --policy.config=$a --policy.dir=$b ) > $LG/server_${name}_$SLURM_JOB_ID.log 2>&1 &
  else
    ( conda activate $a; exec python $Z/common/zoo_server.py --adapter $b --port $PORT --adapter-args ${c//,/ } ) > $LG/server_${name}_$SLURM_JOB_ID.log 2>&1 &
  fi
  SPID=$!; PIDS="$PIDS $SPID"
  until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex((\"127.0.0.1\",$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo "server $name died"; tail -30 $LG/server_${name}_$SLURM_JOB_ID.log; exit 1; }; done
  echo "$(hostname) $PORT $SLURM_JOB_ID $GPU" > $Z/servers/$name.addr; echo "listening $name :$PORT $(date)"
done
( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $LG/thr_srv_$SLURM_JOB_ID.tsv 2>&1 &
wait -n $PIDS; echo "a server exited $(date)"; kill $PIDS 2>/dev/null; exit 0
