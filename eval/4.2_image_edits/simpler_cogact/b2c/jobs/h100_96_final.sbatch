#!/bin/bash
#SBATCH -J b2c_h96f
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=15
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/h96f_%j.log
# H100 NVL（h100-96）同卡 3 个进程，定稿代码（keys.py 已去掉卡型分组，与 A100 冒烟同一哈希；Fractal 的 L 按校准为 8）：
#   进程 1：定稿冒烟 B1、B2 -> out_smoke_h100_v2；进程 2：定稿冒烟 F1、F2（--lag 8）-> out_smoke_h100_v2；
#   进程 3：补跑校准 C0 第一遍 B1 -> out_calib_h100（作业 <job> 里这一个进程载入时被主机内存 OOM 杀掉）。
# 进程错开载入：等前一个打出 model loaded 再起下一个（同时载入三个时 160G 被 OOM 杀过）。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
gpu_sampler logs/h96f_${SLURM_JOB_ID}_gpu.csv & GS=$!
waitload() { for i in $(seq 1 120); do grep -q "model loaded" $1 2>/dev/null && return; kill -0 $2 2>/dev/null || return; sleep 5; done; }
L1=logs/h96f_${SLURM_JOB_ID}_smoke_B1.log
( for b in B1 B2; do python eval_views.py --block $b --cells C0,C1,C3,C4,C5 --keys keysets/smoke_$b.json --out out_smoke_h100_v2 \
    --save-frames 0,40 --frames-dir out_smoke_h100_v2/frames --skip-done --blind-success 2>&1 | grep -v "$NOISE" > logs/h96f_${SLURM_JOB_ID}_smoke_$b.log; done ) & P1=$!
sleep 5; waitload $L1 $P1
L2=logs/h96f_${SLURM_JOB_ID}_smoke_F1.log
( for b in F1 F2; do python eval_views.py --block $b --cells C0,C1,C3,C4,C5 --lag 8 --keys keysets/smoke_$b.json --out out_smoke_h100_v2 \
    --save-frames 0,40 --frames-dir out_smoke_h100_v2/frames --skip-done --blind-success 2>&1 | grep -v "$NOISE" > logs/h96f_${SLURM_JOB_ID}_smoke_$b.log; done ) & P2=$!
sleep 5; waitload $L2 $P2
( python eval_views.py --block B1 --cells C0 --keys keysets/calib10_B1.json --out out_calib_h100 --skip-done 2>&1 | grep -v "$NOISE" > logs/h96f_${SLURM_JOB_ID}_calib_B1.log ) & P3=$!
wait $P1; echo "p1 exit $?"; wait $P2; echo "p2 exit $?"; wait $P3; echo "p3 exit $?"
kill $GS
grep -h "model loaded\|ERROR\|finished\|out of memory" logs/h96f_${SLURM_JOB_ID}_*.log
echo "gpu mem/util peak: $(sort -t, -k1 -n logs/h96f_${SLURM_JOB_ID}_gpu.csv | tail -1)"
