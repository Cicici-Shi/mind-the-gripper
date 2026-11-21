#!/bin/bash
#SBATCH --job-name=toy_cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# TOY1 CPU checks: (1) protocol 8.3 physics invariance: HIDE_GRIPPER=all replay == plain replay (qpos per step),
# G (grpkp100 dumps, kp100 map) and S (sham_cg dumps), first 10 episodes of shard 0 (task 0..9, ep 0);
# (2) 8.5 GRP_MAP self-check line; (3) decision (c) pad width vs map.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
T=$PROJ_ROOT/embody_toy; O=$T/out/cpu_$SLURM_JOB_ID; mkdir -p $O; cd $T
R="python tools/regress_img.py --render 0 --code $T --eps 0-0"
G="GRP_KP=100 GRP_MAP=kp100 COPY_GRIP_CMD=1"
env $G $R --cond grpkp100 --body PandaR85 --out $O/g > $O/g.log 2>&1 &
env $G HIDE_GRIPPER=all $R --cond grpkp100 --body PandaR85 --out $O/g_hid > $O/g_hid.log 2>&1 &
env COPY_GRIP_CMD=1 $R --cond sham_cg --body Panda --out $O/s > $O/s.log 2>&1 &
env COPY_GRIP_CMD=1 HIDE_GRIPPER=all $R --cond sham_cg --body Panda --out $O/s_hid > $O/s_hid.log 2>&1 &
python tools/pad_width.py > $O/pad_width.txt 2>&1 &
wait
echo "== G vs G_hid"; python tools/regress_img_cmp.py $O/g $O/g_hid | tail -1
echo "== S vs S_hid"; python tools/regress_img_cmp.py $O/s $O/s_hid | tail -1
grep -h "GRP_MAP selfcheck" $O/g.log | head -1; grep -rh "hidden_gripper_geoms" $O/g_hid/result.jsonl | head -c 600; echo
grep -v -i "warn\|gym\|migration\|upgrade\|^$" $O/pad_width.txt
echo end
