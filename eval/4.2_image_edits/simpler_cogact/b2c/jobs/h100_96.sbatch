#!/bin/bash
#SBATCH -J b2c_h96
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/h96_%j.log
# H100 NVL（h100-96）上同卡 3 个进程（实测显存与吞吐）：
#   进程 1：冒烟 B1、B2（2 键 x 五格，盲成败）-> out_smoke_h100
#   进程 2：冒烟 F1、F2 -> out_smoke_h100
#   进程 3：校准 C0 第一遍，四块各 10 个校准键 -> out_calib_h100（A/A 的第二遍在 h100-47 上跑，同一型号 H100 NVL）
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
gpu_sampler logs/h96_${SLURM_JOB_ID}_gpu.csv & GS=$!
run() { tag=$1; shift; for b in "$@"; do echo "== $tag $b $(date +%T)"; eval "$CMD" 2>&1 | grep -v "$NOISE" > logs/h96_${SLURM_JOB_ID}_${tag}_${b}.log; done; }
( CMD='python eval_views.py --block $b --cells C0,C1,C3,C4,C5 --keys keysets/smoke_$b.json --out out_smoke_h100 --save-frames 0,40 --frames-dir out_smoke_h100/frames --skip-done --blind-success'; run smoke B1 B2 ) & P1=$!
sleep 20
( CMD='python eval_views.py --block $b --cells C0,C1,C3,C4,C5 --keys keysets/smoke_$b.json --out out_smoke_h100 --skip-done --blind-success'; run smoke F1 F2 ) & P2=$!
sleep 20
( CMD='python eval_views.py --block $b --cells C0 --keys keysets/calib10_$b.json --out out_calib_h100 --skip-done'; run calib B1 B2 F1 F2 ) & P3=$!
wait $P1; echo "p1 exit $?"; wait $P2; echo "p2 exit $?"; wait $P3; echo "p3 exit $?"
kill $GS
grep -h "model loaded\|ERROR\|finished\|OutOfMemory\|CUDA out of memory" logs/h96_${SLURM_JOB_ID}_*.log
echo "gpu mem/util peak: $(sort -t, -k1 -n logs/h96_${SLURM_JOB_ID}_gpu.csv | tail -1)"
