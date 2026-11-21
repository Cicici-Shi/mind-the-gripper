#!/bin/bash
#SBATCH -J b2s_install
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/logs/install_%j.log
set -x
export TMPDIR=$HOME/tmp; mkdir -p $TMPDIR
source ~/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q bench2_simpler || conda create -y -n bench2_simpler python=3.10
conda activate bench2_simpler
which python; python -V
cd $PROJ_ROOT/bench2_simpler/SimplerEnv
pip install numpy==1.24.4
pip install -e ManiSkill2_real2sim
pip install -e .
pip install mediapy imageio
python -c "import sapien, mani_skill2_real2sim, simpler_env; print('sapien', sapien.__version__)"
pip list 2>/dev/null | grep -i -E "sapien|numpy|gymnasium|torch|opencv|transforms3d"
