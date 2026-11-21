#!/bin/bash
#SBATCH --job-name=mic_drive
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
bash $PROJ_ROOT/maskinfo_confirm/tools/mic_autodrive.sh
