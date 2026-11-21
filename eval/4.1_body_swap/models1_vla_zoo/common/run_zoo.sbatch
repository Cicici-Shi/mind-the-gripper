#!/bin/bash
#SBATCH --job-name=zoo
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# vla_zoo (2026-11-03): one policy server + several zoo_client.py processes in one job.
# Pattern copied from $PROJ_ROOT/embody_v3/tools/run_pi05_multi.sbatch (read-only reference).
# Env:
#   MODEL   model name (output dir $PROJ_ROOT/vla_zoo/runs/$MODEL/)
#   SERVER  "openpi:<config>:<ckpt dir>"  -> $PROJ_ROOT/pi05/openpi/.venv serve_policy (read-only use)
#           "zoo:<conda env>:<adapter.py>[:k=v,k=v]" -> vla_zoo/common/zoo_server.py in that env
#   PROTO   openpi | raw          REPLAN (optional, else server metadata)
#   LIST    file with lines "NAME BODY SUITE OFFSETS [VAR=VAL ...]" (BODY=SRC for native), as embody_v3 lists
#   TRIALS  episodes per task per offset (default 5)
set -u
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache PIP_CACHE_DIR=$HOME/.cache/pip HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache
export OPENPI_DATA_HOME=$HOME/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 HIDE_ARM=0 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into embody_v3 / pi05 / openvla-oft
export EMBODY_DIR=${EMBODY_DIR:-$PROJ_ROOT/embody_v3} COPY_GRIP_CMD=${COPY_GRIP_CMD:-1}
unset DISPLAY
Z=$PROJ_ROOT/vla_zoo; OUT=$Z/runs/$MODEL; mkdir -p $OUT $Z/logs
PORT=$((20000 + SLURM_JOB_ID % 20000))
export NUS_HOST=$(hostname) NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -z "$NUS_GPU" ] && { echo "no GPU visible on $NUS_HOST"; exit 3; }
echo "node=$NUS_HOST gpu=$NUS_GPU MODEL=$MODEL SERVER=$SERVER PROTO=$PROTO LIST=$LIST EMBODY_DIR=$EMBODY_DIR COPY_GRIP_CMD=$COPY_GRIP_CMD start=$(date)"; cat $LIST
source $HOME/miniforge3/etc/profile.d/conda.sh
IFS=: read -r kind a b c <<< "$SERVER"
HOST=127.0.0.1
if [ "$kind" = remote ]; then   # server runs in another job (run_server.sbatch): SERVER=remote:<MODEL addr name>
  # wait for the server job ($SRVJOB) to write its address
  # models1: SRVN=n -> n server processes ${a}_0..${a}_{n-1} (run_server_multi.sbatch), clients round-robin over them
  if [ -n "${SRVN:-}" ]; then SRVS=$(seq -f "${a}_%g" 0 $((SRVN - 1))); else SRVS=$a; fi
  for sname in $SRVS; do
    until [ -f $Z/servers/$sname.addr ] && [ "$(awk '{print $3}' $Z/servers/$sname.addr)" = "${SRVJOB:-any}" -o -z "${SRVJOB:-}" ]; do sleep 10; done
  done
  read -r HOST PORT _SJ SRVGPU < $Z/servers/$(echo $SRVS | cut -d" " -f1).addr
  export NUS_SRV_HOST=$HOST NUS_SRV_GPU="$SRVGPU"
  sleep infinity & 
elif [ "$kind" = openpi ]; then
  ( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.5} .venv/bin/python scripts/serve_policy.py --port $PORT \
      policy:checkpoint --policy.config=$a --policy.dir=$b ) > $Z/logs/server_${MODEL}_$SLURM_JOB_ID.log 2>&1 &
else
  conda activate $a
  python $Z/common/zoo_server.py --adapter $b --port $PORT --adapter-args ${c//,/ } > $Z/logs/server_${MODEL}_$SLURM_JOB_ID.log 2>&1 &
  conda deactivate
fi
SPID=$!
until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('$HOST',$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo server died; tail -50 $Z/logs/server_${MODEL}_$SLURM_JOB_ID.log; exit 1; }; done
[ "$kind" = remote ] || export NUS_SRV_HOST=$NUS_HOST NUS_SRV_GPU="$NUS_GPU"
echo "server up $HOST:$PORT srv_gpu=$NUS_SRV_GPU $(date)"
( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"; sleep 30; done ) > $Z/logs/thr_$SLURM_JOB_ID.tsv 2>&1 &
LPID=$!
conda activate oftpi; cd $Z
NCL=0
while read name body suite offs extra; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  bodyarg=""; [ "$body" != "SRC" ] && bodyarg="--body $body"
  pre=""; [ "$suite" = "libero_object" ] && pre="obj_"
  rp=""; [ -n "${REPLAN:-}" ] && rp="--replan $REPLAN"
  for off in ${offs//,/ }; do
    if [ -n "${SRVN:-}" ]; then read -r HOST PORT _SJ SRVGPU < $Z/servers/${a}_$((NCL % SRVN)).addr; export NUS_SRV_HOST=$HOST; fi
    NCL=$((NCL + 1))
    env $extra python -X faulthandler $Z/common/zoo_client.py --proto $PROTO --model $MODEL $rp --suite $suite --host $HOST --port $PORT \
      --trials ${TRIALS:-5} --ep-offset $off --skip-done $bodyarg ${CLIENT_ARGS:-} \
      --out $OUT/${pre}${name}_e${off}.jsonl > $Z/logs/c_${MODEL}_${pre}${name}_${SLURM_JOB_ID}_e$off.log 2>&1 &
  done
done < $LIST
while :; do pids=$(jobs -p | grep -v -e "^$SPID$" -e "^$LPID$"); [ -z "$pids" ] && break; wait -n $pids 2>/dev/null || true; done
echo "clients done $(date)"; kill $SPID $LPID; echo "end=$(date)"
