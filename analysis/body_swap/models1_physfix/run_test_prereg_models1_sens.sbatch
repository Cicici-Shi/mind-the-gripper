#!/bin/bash
#SBATCH --job-name=m1_sens_test
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# Synthetic-data tests for prereg_models1_sens.py (no models1 data read). TMPDIR in $HOME (/tmp quota 10 MB).
set -u
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $HOME/tmp/m1sens_test; export TMPDIR=$HOME/tmp/m1sens_test
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/vla_zoo/analysis
sha256sum prereg_models1.py prereg_models1_sens.py prereg_models1_sens.diff test_prereg_models1.py test_prereg_models1_sens.py make_prereg_models1_sens.py
python test_prereg_models1_sens.py
echo "exit $? end $(date)"
