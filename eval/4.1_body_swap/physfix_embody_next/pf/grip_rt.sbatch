#!/bin/bash
#SBATCH --job-name=pf_grip_rt
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp MUJOCO_GL=disable ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/vla_zoo/next/pf; python grip_sweep_pf.py --code $PROJ_ROOT/embody_next --out $PROJ_ROOT/vla_zoo/out/pf/grip_sweep_rt.json --tasks 0
