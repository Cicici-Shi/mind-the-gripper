#!/bin/bash
#SBATCH --job-name=nz_cli
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# next (2026-11-06): render clients for the next model batch, BodyEnv cells, against a REMOTE server started by
# next/run_server_next.sbatch. Derived from common/run_zoo.sbatch (frozen) remote branch; diffs: client is
# next/zoo_client_next.py, BLIND=1 (default) passes --blind (no outcome / length proxy in record or log),
# EMBODY_DIR default $PROJ_ROOT/embody_toy, output $OUTROOT/$MODEL (default runs/next_smoke), logs logs/next/,
# DUMP=1 pickles the requests at queries 0,10 to $OUT/inputs/<cell>/ (load-test replay), M1_FASTSYNC=1 default.
# Env: MODEL, SRV (addr name, e.g. nz_pi0fast), SRVJOB (server job id), PROTO openpi|raw, LIST (cells_next.sh list
# output: "NAME BODY SUITE OFFSETS VAR=VAL..."; tokens starting with -- go to the client), TRIALS (default 1).
set -u
export TMPDIR=$HOME/tmp XDG_CACHE_HOME=$HOME/.cache HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 HIDE_ARM=0 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into embody_toy / pi05
export EMBODY_DIR=${EMBODY_DIR:-$PROJ_ROOT/embody_toy} M1_FASTSYNC=${M1_FASTSYNC:-1}
unset DISPLAY
Z=$PROJ_ROOT/vla_zoo; N=$Z/next; OUT=$Z/${OUTROOT:-runs/next_smoke}/$MODEL; LG=$Z/logs/next; mkdir -p $OUT $LG
export NUS_HOST=$(hostname) NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -z "$NUS_GPU" ] && { echo "no GPU visible on $NUS_HOST"; exit 3; }
echo "node=$NUS_HOST gpu=$NUS_GPU MODEL=$MODEL SRV=$SRV SRVJOB=${SRVJOB:-} PROTO=$PROTO LIST=$LIST EMBODY_DIR=$EMBODY_DIR BLIND=${BLIND:-1} start=$(date)"; cat $LIST
until [ -f $Z/servers/$SRV.addr ] && [ "$(awk '{print $3}' $Z/servers/$SRV.addr)" = "${SRVJOB:-any}" -o -z "${SRVJOB:-}" ]; do sleep 10; done
read -r HOST PORT _SJ SRVGPU < $Z/servers/$SRV.addr
export NUS_SRV_HOST=$HOST NUS_SRV_GPU="$SRVGPU"
echo "server $HOST:$PORT srv_gpu=$SRVGPU $(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate oftpi; cd $Z
bl=""; [ "${BLIND:-1}" = 1 ] && bl="--blind"
while read name body suite offs extra; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  bodyarg=""; [ "$body" != "SRC" ] && bodyarg="--body $body"
  cargs=""; envx=""; for t in $extra; do case "$t" in --*) cargs="$cargs ${t/=/ }";; *) envx="$envx $t";; esac; done
  pre=""; [ "$suite" = "libero_object" ] && pre="obj_"; [ "$suite" = "libero_goal" ] && pre="goal_"; [ "$suite" = "libero_10" ] && pre="l10_"
  dmp=""; [ "${DUMP:-0}" = 1 ] && dmp="--dump-inputs $OUT/inputs/${pre}${name}"
  for off in ${offs//,/ }; do
    tag=${pre}${name}_e${off}${cargs:+_$(echo $cargs | tr -dc 'a-z0-9')}
    env $envx python -X faulthandler $N/zoo_client_next.py --proto $PROTO --model $MODEL --suite $suite --host $HOST --port $PORT \
      --trials ${TRIALS:-1} --ep-offset $off --skip-done $bodyarg $bl $dmp ${CLIENT_ARGS:-} $cargs \
      --out $OUT/${tag}.jsonl > $LG/c_${MODEL}_${tag}_${SLURM_JOB_ID}.log 2>&1 &
  done
done < $LIST
wait
echo "clients done $(date)"
