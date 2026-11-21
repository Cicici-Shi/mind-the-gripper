#!/bin/bash
#SBATCH --job-name=pf_test
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# Synthetic-data tests for prereg_physfix.py (no pf_ / models1 data read). TMPDIR in $HOME (/tmp quota small).
set -u
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $HOME/tmp/physfix_test; export TMPDIR=$HOME/tmp/physfix_test
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/vla_zoo/analysis
sha256sum prereg_models1.py prereg_physfix.py test_prereg_physfix.py run_test_prereg_physfix.sbatch
python test_prereg_physfix.py
echo "exit $? end $(date)"
