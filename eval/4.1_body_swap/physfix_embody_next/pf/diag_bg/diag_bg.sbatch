#!/bin/bash
#SBATCH --job-name=pf_bgdiag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
# diagnosis of t0check <job> agent_background != 0 (no policy, no outcomes). Same env as t0check_next.sbatch.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1 EMBODY_DIR=$PROJ_ROOT/embody_next; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
P=$PROJ_ROOT/vla_zoo/next/pf/diag_bg; O=$P/out_$SLURM_JOB_ID; mkdir -p $O; cd $P
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) body_env=$(sha256sum $EMBODY_DIR/embody/body_env.py|cut -c1-16)"
python diag_bg.py --suite libero_object --tasks 4 --trials 1 --outdir $O > $O/obj_4.log 2>&1 &
sleep 5
python diag_bg.py --suite libero_object --tasks 0,1 --trials 2 --outdir $O > $O/obj_01.log 2>&1 &
sleep 5
python diag_bg.py --suite libero_spatial --tasks 0,1 --trials 3 --outdir $O > $O/sp_01.log 2>&1 &
wait
grep -h "Traceback\|exit 2\|DIAG_BG done" $O/*.log
echo end $(date)
