#!/bin/bash
#SBATCH --job-name=pi05_install
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv PIP_CACHE_DIR=$HOME/.cache/pip XDG_CACHE_HOME=$HOME/.cache
export UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python
export PATH=$HOME/.local/bin:$PATH GIT_LFS_SKIP_SMUDGE=1
cd $PROJ_ROOT/pi05/openpi
echo "node=$(hostname) start=$(date)"
curl -s -o /dev/null -w "internet: %{http_code}\n" --max-time 10 https://storage.googleapis.com/
uv sync 2>&1 | tail -40
uv pip install -e . 2>&1 | tail -5
.venv/bin/python -c "import jax, openpi, sys; print(\"python\", sys.version); print(\"jax\", jax.__version__)"
echo "exit=$? end=$(date)"
