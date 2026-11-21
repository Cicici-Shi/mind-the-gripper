#!/bin/bash
#SBATCH --job-name=pi05_multi
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# Block 3 (2026-10-31): one pi0.5 server, several clients (cells x shards) in one job. Same client and
# server commands as $PROJ_ROOT/pi05/run_pi05_v2.sbatch; only the client list differs.
# Env: LIST = file with lines "NAME BODY SUITE OFFSETS [VAR=VAL ...]" (BODY=SRC for native; OFFSETS comma list),
#      EMBODY_DIR, COPY_GRIP_CMD. Output $PROJ_ROOT/pi05/v2/<NAME>_e<off>.jsonl (suite libero_object: obj_<NAME>_e<off>.jsonl).
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache PIP_CACHE_DIR=$HOME/.cache/pip
export OPENPI_DATA_HOME=$HOME/.cache/openpi MUJOCO_GL=egl PYTHONFAULTHANDLER=1 HIDE_ARM=0 OPENBLAS_NUM_THREADS=1; unset DISPLAY
P=$PROJ_ROOT/pi05; CKPT=$P/ckpt/pi05_libero; OP=$P/openpi
PORT=$((20000 + SLURM_JOB_ID % 20000))
export NUS_HOST=$(hostname) NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "node=$NUS_HOST gpu=$NUS_GPU LIST=$LIST EMBODY_DIR=$EMBODY_DIR COPY_GRIP_CMD=$COPY_GRIP_CMD start=$(date)"; cat $LIST
source $HOME/miniforge3/etc/profile.d/conda.sh
if [ -n "$PRECHECK_GR" ]; then  # G_R shard 0 (amendment 1, section 6): ruling-2 step-0 check in this job, same node/GPU
  conda activate oft; mkdir -p $PROJ_ROOT/embody_v3/out/gr_t0
  ( cd $PROJ_ROOT/embody_v3 && ORIGINAL_ONLY=1 python -X faulthandler tools/gr_t0check.py --offsets $PRECHECK_GR \
      --out $PROJ_ROOT/embody_v3/out/gr_t0/pi05_${SLURM_JOB_ID}.jsonl ) > $PROJ_ROOT/embody_v3/out/gr_t0/pi05_${SLURM_JOB_ID}.log 2>&1
  echo "gr_t0check exit=$? $(date)"; conda deactivate
fi
if [ -n "$PRECHECK_GR_PAR" ]; then  # G_R full shards (amendment 2): one gr_t0check per offset, in parallel with the clients
  mkdir -p $PROJ_ROOT/embody_v3/out/gr_t0
  for off in $PRECHECK_GR_PAR; do
    ( conda activate oft; cd $PROJ_ROOT/embody_v3 && ORIGINAL_ONLY=1 OPENBLAS_NUM_THREADS=1 python -X faulthandler tools/gr_t0check.py --offsets $off \
        --out $PROJ_ROOT/embody_v3/out/gr_t0/pi05_${SLURM_JOB_ID}_e${off}.jsonl ) > $PROJ_ROOT/embody_v3/out/gr_t0/pi05_${SLURM_JOB_ID}_e${off}.log 2>&1 &
  done
fi
cd $OP
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=pi05_libero --policy.dir=$CKPT > $P/logs/server_$SLURM_JOB_ID.log 2>&1 &
SPID=$!
until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; done
conda activate oftpi; cd $P
mkdir -p $P/v2
while read name body suite offs extra; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  bodyarg=""; [ "$body" != "SRC" ] && bodyarg="--body $body"
  pre=""; [ "$suite" = "libero_object" ] && pre="obj_"
  for off in ${offs//,/ }; do
    env $extra python -X faulthandler pi05_body_eval.py --suite $suite --port $PORT --trials ${TRIALS:-5} --ep-offset $off --skip-done $bodyarg \
      --out $P/v2/${pre}${name}_e${off}.jsonl > $P/logs/v2_${pre}${name}_${SLURM_JOB_ID}_e$off.log 2>&1 &
  done
done < $LIST
# wait for every background job except the server (clients and, if any, gr_t0check); never call a bare "wait"
# (an empty pid list would make it wait for the server forever: <job> / <job> hung this way on 11/03)
while :; do pids=$(jobs -p | grep -v "^$SPID$"); [ -z "$pids" ] && break; wait -n $pids 2>/dev/null || true; done
echo "clients done $(date)"; kill $SPID; echo "end=$(date)"
