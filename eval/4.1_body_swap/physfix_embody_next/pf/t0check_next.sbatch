#!/bin/bash
#SBATCH --job-name=pf_t0chk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.log
# physfix 7.2 step-0 consistency on H200 (t0check_next.py asserts GL_RENDERER contains H200, exit 2 otherwise).
# Shard-0 keys (ep 0-4) of both suites, 5 processes per suite split by task.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1 EMBODY_DIR=$PROJ_ROOT/embody_next; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
P=$PROJ_ROOT/vla_zoo/next/pf; O=$PROJ_ROOT/vla_zoo/out/pf/t0check_$SLURM_JOB_ID; mkdir -p $O; cd $P
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) cells=$(sha256sum ../cells_next.sh|cut -c1-16) t0=$(sha256sum t0check_next.py|cut -c1-16) robots=$(sha256sum $EMBODY_DIR/embody/robots.py|cut -c1-16) body_env=$(sha256sum $EMBODY_DIR/embody/body_env.py|cut -c1-16)"
for s in libero_spatial libero_object; do for tk in 0,1 2,3 4,5 6,7 8,9; do
  python t0check_next.py --suite $s --tasks $tk --out $O/${s}_${tk/,/}.jsonl > $O/${s}_${tk/,/}.log 2>&1 &
  sleep 5; done; done
wait
rc=0; grep -l "exit 2" $O/*.log && rc=2
cat $O/libero_spatial_*.jsonl > $O/t0_spatial.jsonl; cat $O/libero_object_*.jsonl > $O/t0_object.jsonl
python t0check_next_gatejson.py --spatial $O/t0_spatial.jsonl --object $O/t0_object.jsonl --regr-ok ${REGR_OK:-0} --out $O/gate_physfix_t0.json || rc=$?
echo end $(date) rc=$rc; exit $rc
