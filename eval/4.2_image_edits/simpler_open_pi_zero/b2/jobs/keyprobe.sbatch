#!/bin/bash
#SBATCH -J b2_keyprobe2
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/keyprobe_%j.log
# 键核查（协议 v1.1，接手方）：四块全部正式键与校准键各 reset 两遍，解析 G 与保护名单，核实际 episode_id / urdf。不跑模型、不看成败。
source $PROJ_ROOT/bench2_simpler/b2/jobs/common.sh
python probe_keys.py --out keyprobe_v11.json${KP_BLOCKS:+ --blocks $KP_BLOCKS} 2>&1 | grep -v "$NOISE"
