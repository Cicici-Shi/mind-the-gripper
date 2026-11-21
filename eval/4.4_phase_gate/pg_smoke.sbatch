#!/bin/bash
#SBATCH --job-name=pg_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.log
# phasegate smoke: no policy, no outcome. Unit tests of the gate logic, then tools/pg_smoke.py (render checks).
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
T=$PROJ_ROOT/phasegate_confirm
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date)"
echo "GL $(python tools/gl_renderer.py 2>&1 | grep '^GL_RENDERER')"
sha256sum embody/phasegate.py embody/body_env.py oft/run_libero_eval_pg.py tools/pg_smoke.py
python analysis/test_phasegate.py || exit 1
python analysis/test_pg_readout.py || exit 1
python tools/pg_smoke.py $T/out/smoke_${SLURM_JOB_ID}
echo "exit=$? end=$(date)"
