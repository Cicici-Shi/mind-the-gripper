#!/bin/bash
#SBATCH --job-name=mic_tprereg
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/maskinfo_confirm/analysis && export TMPDIR=$HOME/tmp
sha256sum prereg_maskinfo_confirm.py test_prereg_maskinfo_confirm.py ref/protocol_maskinfo_confirm.md ../lists/first_round_maskinfo.json
python -m unittest test_prereg_maskinfo_confirm > $PROJ_ROOT/maskinfo_confirm/logs/tprereg_full_$SLURM_JOB_ID.txt 2>&1; tail -5 $PROJ_ROOT/maskinfo_confirm/logs/tprereg_full_$SLURM_JOB_ID.txt
