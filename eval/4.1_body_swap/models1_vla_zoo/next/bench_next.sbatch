#!/bin/bash
#SBATCH --job-name=nz_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.log
# next (2026-11-06): server capacity on one h100-47 card. For each config "model:NSRV" in CONFIGS, start NSRV server
# processes of that model on this GPU (same launch lines as run_server_next.sbatch), run loadgen_next.py (real
# requests from the blind smoke dumps, localhost, tiers TIERS x DUR s), kill the servers, next config.
# Output: runs/next_smoke/bench/bench_$SLURM_JOB_ID.jsonl ; GPU util/mem every 10 s in logs/next/thr_bench_<job>.tsv
set -u
export TMPDIR=$HOME/tmp XDG_CACHE_HOME=$HOME/.cache HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache OPENPI_DATA_HOME=$HOME/.cache/openpi PYTHONDONTWRITEBYTECODE=1
Z=$PROJ_ROOT/vla_zoo; N=$Z/next; LG=$Z/logs/next; B=$Z/runs/next_smoke/bench; mkdir -p $LG $B
CONFIGS=${CONFIGS:-"pi0fast:1 pi0fast:2 pi0fast:3 smolvla:1 smolvla:2 smolvla:4"}
TIERS=${TIERS:-1,2,4,8,12,16}; DUR=${DUR:-75}
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; echo "node=$(hostname) gpu=$GPU CONFIGS=$CONFIGS TIERS=$TIERS DUR=$DUR start=$(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh
( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 10; done ) > $LG/thr_bench_$SLURM_JOB_ID.tsv 2>&1 &
MON=$!
base=$((20000 + (SLURM_JOB_ID * 7) % 19000))
for cfg in $CONFIGS; do
  m=${cfg%%:*}; ns=${cfg#*:}; PIDS=""; PORTS=""
  for i in $(seq 0 $((ns - 1))); do
    port=$((base + i)); PORTS="$PORTS,$port"
    if [ $m = pi0fast ]; then
      ( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=$(python3 -c "print(round(0.85/$ns,3))") exec .venv/bin/python $Z/src_next/serve_policy_zscore.py --port $port \
          policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=$Z/openpi_ckpt/pi0_fast_libero ) > $LG/bench_srv_${m}_${ns}_${i}_$SLURM_JOB_ID.log 2>&1 &
    else
      ( conda activate zoo_lerobot; exec python $Z/common/zoo_server.py --adapter $Z/common/adapter_lerobot.py --port $port --adapter-args repo=HuggingFaceVLA/smolvla_libero ) > $LG/bench_srv_${m}_${ns}_${i}_$SLURM_JOB_ID.log 2>&1 &
    fi
    PIDS="$PIDS $!"
  done
  ok=1
  for i in $(seq 0 $((ns - 1))); do
    port=$((base + i)); pid=$(echo $PIDS | cut -d" " -f$((i + 1)))
    until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$port)))" 2>/dev/null; do sleep 5; kill -0 $pid 2>/dev/null || { echo "CONFIG $cfg server $i died"; tail -5 $LG/bench_srv_${m}_${ns}_${i}_$SLURM_JOB_ID.log; ok=0; break; }; done
  done
  if [ $ok = 1 ]; then
    echo "CONFIG $cfg up $(date)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
    conda activate oftpi
    python $N/loadgen_next.py --ports ${PORTS#,} --inputs $Z/runs/next_smoke/$m/inputs --tiers $TIERS --dur $DUR --label $cfg --out $B/bench_$SLURM_JOB_ID.jsonl
    conda deactivate
  fi
  kill $PIDS 2>/dev/null; sleep 5; kill -9 $PIDS 2>/dev/null; sleep 10
  base=$((base + 10))
done
kill $MON; echo "end $(date)"
