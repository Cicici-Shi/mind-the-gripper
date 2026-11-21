#!/bin/bash
#SBATCH -J b2c_install
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/logs/install_%j.log
set -x
export TMPDIR=$HOME/tmp; mkdir -p $TMPDIR
export PYTHONDONTWRITEBYTECODE=1
source ~/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q '^bench2_cogact ' || conda create -y -n bench2_cogact python=3.10
conda activate bench2_cogact
which python; python -V
R=$PROJ_ROOT/bench2_cogact
# SimplerEnv：从官方 clone 只读复制一份（CogACT README 指定官方 main），要往 policies 里放 sim_cogact、改 main_inference.py
[ -d $R/SimplerEnv ] || cp -a $PROJ_ROOT/bench2_simpler/SimplerEnv $R/SimplerEnv
find $R/SimplerEnv -name "*.egg-info" -maxdepth 3 -prune -exec rm -rf {} \;
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e $R/CogACT
pip install -e $R/SimplerEnv/ManiSkill2_real2sim
pip install -e $R/SimplerEnv
pip install mediapy imageio "setuptools<70" "numpy==1.24.4" "opencv-python==4.10.0.84"
pip list 2>/dev/null | grep -i -E "^(torch|transformers|tokenizers|timm|tensorflow|numpy|sapien|opencv|protobuf|sentencepiece|openvla|dlimp|setuptools|gymnasium|transforms3d) "
pip check 2>&1 | head -20
cd $R
python -c "import numpy, cv2, torch, sapien, mani_skill2_real2sim, simpler_env, transformers, timm; print('ok numpy', numpy.__version__, 'torch', torch.__version__, 'tf', transformers.__version__, 'timm', timm.__version__, 'sapien', sapien.__version__)"
python -c "import sys; sys.path.insert(0,'$R/CogACT'); from vla import load_vla; print('vla import ok')"
