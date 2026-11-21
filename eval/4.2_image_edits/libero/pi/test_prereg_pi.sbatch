#!/bin/bash
#SBATCH --job-name=mipi_tprereg
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x_%j.log
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oftpi
T=$PROJ_ROOT/maskinfo_pi; export TMPDIR=$HOME/tmp PYTHONDONTWRITEBYTECODE=1; cd $T/analysis
sha256sum prereg_maskinfo_pi.py test_prereg_maskinfo_pi.py ref/protocol_maskinfo_pi.md
python -m unittest test_prereg_maskinfo_pi > $T/logs/tprereg_full_$SLURM_JOB_ID.txt 2>&1; echo "unittest exit=$?"; tail -4 $T/logs/tprereg_full_$SLURM_JOB_ID.txt
