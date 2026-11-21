#!/bin/bash
#SBATCH --job-name=ik_diag_m1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ik_diag_m1_%j.log
# Read-only twin-IK diagnostic (tools/ik_diag_m1.py; Fable ruling 2026-11-05 3.4 / 4.4). CPU only, no GPU.
# Usage: sbatch $PROJ_ROOT/vla_zoo/tools/ik_diag_m1.sbatch            (all three groups, complete shards only)
Z=$PROJ_ROOT/vla_zoo
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disable TMPDIR=$HOME/tmp
echo "host $(hostname)  $(date)"
sha256sum $Z/tools/ik_diag_m1.py $Z/tools/ik_diag_m1.sbatch
python $Z/tools/ik_diag_m1.py --groups ${DIAG_GROUPS:-oft_obj,openvla,pi0}
