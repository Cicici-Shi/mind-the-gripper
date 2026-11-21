#!/bin/bash
#SBATCH -J b3_obchk
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/obchk_%j.log
# v3 §3 任务物平移合成三条放弃条件：校准键（calib10）原图格参考轨迹回放，不跑策略
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
P=()
for b in B1 B2; do python cue_obcheck.py replay --block $b --keys ../b2/keysets/calib10_$b.json --ref ../b2/out_calib --out out_obcheck > logs/obchk_${SLURM_JOB_ID}_$b.log 2>&1 & P+=($!); done
waitall
python cue_obcheck.py summarize --root out_obcheck
grep -l Traceback logs/obchk_${SLURM_JOB_ID}_*.log || echo "no Traceback"
