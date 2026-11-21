#!/bin/bash
#SBATCH -p normal
#SBATCH --job-name=d3_erratum
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/d3_erratum_%j.log
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
source ~/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/embody_v3
python -u tools/d3_erratum_check.py 2>&1 | grep -v "robosuite WARNING"
