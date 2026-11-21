#!/bin/bash
#SBATCH --job-name=pi05_clone
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp PIP_CACHE_DIR=$HOME/.cache/pip XDG_CACHE_HOME=$HOME/.cache CONDA_PKGS_DIRS=$HOME/miniforge3/pkgs
source $HOME/miniforge3/etc/profile.d/conda.sh
echo "node=$(hostname) start=$(date)"
conda create -y -n oftpi --clone oft 2>&1 | tail -5
conda activate oftpi
pip install --no-deps -e $PROJ_ROOT/pi05/openpi/packages/openpi-client 2>&1 | tail -2
pip install "websockets>=11.0" msgpack dm-tree 2>&1 | tail -2
python -c "import mujoco, robosuite, openpi_client, websockets, numpy; print(mujoco.__version__, robosuite.__version__, numpy.__version__, websockets.__version__)"
pip list 2>/dev/null | grep -iE "^(mujoco|robosuite|numpy|torch|websockets|msgpack) "
echo "exit=$? end=$(date)"
