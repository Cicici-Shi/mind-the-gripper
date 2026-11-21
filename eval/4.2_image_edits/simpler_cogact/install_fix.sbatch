#!/bin/bash
#SBATCH -J b2c_fix
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/logs/install_fix_%j.log
# 首装后 import vla 失败：最新 tensorflow-metadata 要 protobuf>=5（runtime_version），tensorflow 2.15 要 protobuf<5
export TMPDIR=$HOME/tmp PYTHONDONTWRITEBYTECODE=1 HF_HOME=$PROJ_ROOT/bench2_cogact/hf
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_cogact
pip install "tensorflow-metadata==1.15.0" 2>&1 | tail -2
pip list 2>/dev/null | grep -i -E "^(protobuf|tensorflow-metadata|tensorflow-datasets|tensorflow) "
pip check
R=$PROJ_ROOT/bench2_cogact; cd $R
python -c "import sys; sys.path.insert(0,'$R/CogACT'); from vla import load_vla; from sim_cogact import CogACTInference; import simpler_env, sapien; print('vla import ok')" 2>&1 | grep -v -E "cuda_|TensorRT|cpu_feature|AVX2|NVML"
python $R/make_llama_stub.py 2>&1 | grep -v -E "cuda_|TensorRT|cpu_feature|AVX2|NVML"
echo "sha256 expected 1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0"
sha256sum $R/ckpt/CogACT-Base/checkpoints/CogACT-Base.pt
