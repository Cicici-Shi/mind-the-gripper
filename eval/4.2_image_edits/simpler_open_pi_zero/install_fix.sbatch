#!/bin/bash
#SBATCH -J b2s_fix
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/logs/install_fix_%j.log
# 第一次安装后两处问题：opencv 5 把 numpy 拉到 2.2.6；新版 setuptools 没有 pkg_resources（sapien 2.2.2 要用）
export TMPDIR=$HOME/tmp
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bench2_simpler
pip install "setuptools<70" "numpy==1.24.4" "opencv-python==4.10.0.84" 2>&1 | tail -3
python -c "import numpy, cv2, sapien, mani_skill2_real2sim, simpler_env; print('ok numpy', numpy.__version__, 'cv2', cv2.__version__, 'sapien', sapien.__version__)"
pip check 2>&1 | head
