#!/bin/bash
#SBATCH --job-name=pf_prereg_test
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# synthetic-data tests of prereg_physfix.py rev3 (no pf_ data read)
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/vla_zoo/analysis
echo "node=$(hostname) start=$(date) prereg=$(sha256sum prereg_physfix.py|cut -c1-16) test=$(sha256sum test_prereg_physfix.py|cut -c1-16)"
python test_prereg_physfix.py 2>&1 | tail -60
echo "end=$(date)"
