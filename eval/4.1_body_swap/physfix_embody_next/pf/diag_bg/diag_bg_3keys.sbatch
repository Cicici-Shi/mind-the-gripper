#!/bin/bash
#SBATCH --job-name=pf_bgdiag3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.log
# t0check background diagnosis, all 50 keys per suite (ep 0-4 x 10 tasks, same split / RNG windows as t0check_next.sbatch).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1 EMBODY_DIR=$PROJ_ROOT/embody_next; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
P=$PROJ_ROOT/vla_zoo/next/pf/diag_bg; O=$P/out3_$SLURM_JOB_ID; mkdir -p $O; cd $P
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) body_env=$(sha256sum $EMBODY_DIR/embody/body_env.py|cut -c1-16) diag=$(sha256sum diag_bg.py|cut -c1-16)"
for s in libero_spatial; do for tk in 2,3 4,5; do
  python diag_bg.py --suite $s --tasks $tk --trials 5 --save 1 --outdir $O > $O/${s}_${tk/,/}.log 2>&1 &
  sleep 5; done; done
wait
grep -c "DIAG_BG done" $O/*.log; grep -l "Traceback\|exit 2" $O/*.log
echo end $(date)
