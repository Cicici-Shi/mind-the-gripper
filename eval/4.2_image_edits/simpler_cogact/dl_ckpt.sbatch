#!/bin/bash
#SBATCH -J b2c_dl
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/logs/dl_%j.log
# 下载 CogACT-Base（公开、MIT、不需要授权）到 $PROJ_ROOT/bench2_cogact/ckpt/CogACT-Base/
set -e
D=$PROJ_ROOT/bench2_cogact/ckpt/CogACT-Base; mkdir -p $D/checkpoints; cd $D
B=https://huggingface.co/CogACT/CogACT-Base/resolve/main
df -h ~ | tail -1
for f in config.json dataset_statistics.json README.md; do curl -sSfL -o $f $B/$f; done
curl -sSfL -C - -o checkpoints/CogACT-Base.pt $B/checkpoints/CogACT-Base.pt
ls -la $D $D/checkpoints
python3 -c "import json;print(json.load(open('config.json'))['vla'])" || cat config.json
