#!/bin/bash
#SBATCH --job-name=m1_driver
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/m1_driver_%j.log
# models1 driver tick (tools/m1_driver.py), then resubmit itself in 20 min until 2026-11-09 12:00 SGT.
# Stop: touch $PROJ_ROOT/vla_zoo/out/m1_driver.STOP
Z=$PROJ_ROOT/vla_zoo
[ -f $Z/out/m1_driver.STOP ] && { echo stopped; exit 0; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
export PYTHONDONTWRITEBYTECODE=1
python $Z/tools/m1_driver.py
if [ $(date +%s) -lt $(date -d "2026-11-09 12:00" +%s) ]; then
  sbatch --begin=now+20minutes $Z/tools/m1_driver.sbatch
fi
