#!/bin/bash
#SBATCH --job-name=pi05_obj
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# One condition per job: its own pi0.5 server and 4 clients, one per OFT shard (episodes 0-4, 5-9,
# 10-14, 15-19 of every task, tasks in order), so each (task, episode) scene is built from the same
# np.random stream position as in the OFT runs. --skip-done makes a rerun fill holes only.
# Server and clients live and die together.
# Env: NAME BODY HIDE [EMBODY_DIR GRP_KP RECOLOR_ARM]
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache PIP_CACHE_DIR=$HOME/.cache/pip
export OPENPI_DATA_HOME=$HOME/.cache/openpi MUJOCO_GL=egl PYTHONFAULTHANDLER=1 HIDE_ARM=$HIDE; unset DISPLAY
P=$PROJ_ROOT/pi05; CKPT=$P/ckpt/pi05_libero; OP=$P/openpi
PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) NAME=$NAME BODY=$BODY HIDE=$HIDE EMBODY_DIR=$EMBODY_DIR GRP_KP=$GRP_KP RECOLOR_ARM=$RECOLOR_ARM start=$(date)"
source $HOME/miniforge3/etc/profile.d/conda.sh
cd $OP
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=pi05_libero --policy.dir=$CKPT > $P/logs/server_$SLURM_JOB_ID.log 2>&1 &
SPID=$!
until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; done
conda activate oftpi; cd $P
bodyarg=""; [ "$BODY" != "SRC" ] && bodyarg="--body $BODY"
mkdir -p $P/v2
for off in ${OFFSETS:-0 5 10 15}; do
  python -X faulthandler pi05_body_eval.py --suite libero_object --port $PORT --trials ${TRIALS:-5} --ep-offset $off --skip-done $bodyarg --out $P/v2/obj_${NAME}_e${off}.jsonl > $P/logs/v2_obj_${NAME}_${SLURM_JOB_ID}_e$off.log 2>&1 &
done
wait $(jobs -p | grep -v "^$SPID$")
echo "clients done $(date)"; kill $SPID; echo "end=$(date)"
