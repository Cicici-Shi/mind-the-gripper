#!/bin/bash
#SBATCH --job-name=pf_handfeas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 EMBODY_DIR=$PROJ_ROOT/embody_next; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
O=$PROJ_ROOT/vla_zoo/out/pf/handfeas_$SLURM_JOB_ID; mkdir -p $O; cd $PROJ_ROOT/vla_zoo/next/pf
echo node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
python hand_feas_pf.py --suite libero_spatial --out $O/spatial.jsonl --png $O/png > $O/spatial.log 2>&1 &
python hand_feas_pf.py --suite libero_object --out $O/object.jsonl --png $O/png > $O/object.log 2>&1 &
wait; echo end $(date)
