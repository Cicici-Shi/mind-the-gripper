#!/bin/bash
#SBATCH --job-name=rs_proto
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.log
# rescue outline prototype (no policy, no outcome). Not h200.
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/rescue/proto
nvidia-smi --query-gpu=name --format=csv,noheader
python outline_proto.py --out $PROJ_ROOT/rescue/proto/out_$SLURM_JOB_ID ${PROTO_ARGS:-}
