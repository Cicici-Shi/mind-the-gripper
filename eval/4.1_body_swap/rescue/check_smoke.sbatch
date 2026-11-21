#!/bin/bash
#SBATCH --job-name=rs_chk
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.log
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
python $PROJ_ROOT/rescue/check_smoke.py
