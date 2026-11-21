#!/bin/bash
#SBATCH -J b3_proj
#SBATCH -p gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/proj_%j.log
# 相机参数与末端参考点投影核查：回放参考记录（不跑策略），Titan RTX。
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
python proj_check.py --block B2 --keys keysets/dev_B2.json --ref ../b2/out_calib --out out_proj --queries 0,3,6,9,12 2>&1 | grep -v "$NOISE"
python proj_check.py --block F1 --keys keysets/dev_F1.json --ref ../b2/out_calib --out out_proj --queries 0,5,10,20,30 2>&1 | grep -v "$NOISE"
