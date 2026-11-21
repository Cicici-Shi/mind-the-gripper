#!/bin/bash
#SBATCH --job-name=toy_rchk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# TOY1 (2026-11-03): (a) regression embody_v3 vs embody_toy with all TOY_* off (qpos + image md5 per step), with an
# old-vs-old repeat as the render-noise baseline and TOY_PIXSTATS=1 (must not change images); (b) smoke screenshots.
# embody_v3 is only read (PYTHONDONTWRITEBYTECODE=1: no __pycache__ written there).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
T=$PROJ_ROOT/embody_toy; V=$PROJ_ROOT/embody_v3; O=$T/out/rchk_$SLURM_JOB_ID; mkdir -p $O; cd $T
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
TASKS=${TASKS:-0,2}; EPS=${EPS:-0-1}
R="python tools/regress_img.py --render 1 --tasks $TASKS --eps $EPS --save-img 40"
G="GRP_KP=100 COPY_GRIP_CMD=1"
if [ "${DO_REG:-1}" = 1 ]; then
for spec in "grpkp100|PandaR85|$G" "grpkp100_twin|PandaR85|$G TWIN_RENDER=1" "grpkp100_tw|PandaR85|$G TWIN_RENDER=1 TWIN_CAMS=wrist" \
            "sham_twin|Panda|TWIN_RENDER=1" "g_r|Panda|TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk COPY_GRIP_CMD=1"; do
  IFS='|' read c b v <<< "$spec"
  env $v $R --code $V --cond $c --body $b --out $O/reg/${c}_old > $O/reg_${c}_old.log 2>&1 &
  env $v $R --code $V --cond $c --body $b --out $O/reg/${c}_old2 > $O/reg_${c}_old2.log 2>&1 &
  env $v $R --code $T --cond $c --body $b --out $O/reg/${c}_new > $O/reg_${c}_new.log 2>&1 &
  case $c in grpkp100|grpkp100_twin) env $v TOY_PIXSTATS=1 $R --code $T --cond $c --body $b --out $O/reg/${c}_newpix > $O/reg_${c}_newpix.log 2>&1 & ;; esac
done
wait
for c in grpkp100 grpkp100_twin grpkp100_tw sham_twin g_r; do
  for x in old2 new newpix; do [ -d $O/reg/${c}_$x ] && { echo "== $c old vs $x"; python tools/regress_img_cmp.py $O/reg/${c}_old $O/reg/${c}_$x; }; done
done
fi
if [ "${DO_SMOKE:-1}" = 1 ]; then
S="python tools/smoke_toy.py --body PandaR85 --cond grpkp100 --tasks ${STASKS:-0,2} --eps ${SEPS:-0} --out $O/smoke --shots $O/shots"
B="$G TOY_PIXSTATS=1"
for spec in "G|$B" "G_camback|$B TOY_WRIST_CAM=back" "GT|$B TWIN_RENDER=1" "GT_camphys|$B TWIN_RENDER=1 TOY_WRIST_CAM=phys" \
            "hand_ik|$B TWIN_RENDER=1 TOY_PARTS=hand" "hand_flange|$B TWIN_RENDER=1 TOY_PARTS=hand TOY_TWIN_POSE=flange" \
            "fingers|$B TWIN_RENDER=1 TOY_PARTS=fingers" "fingers_keep|$B TWIN_RENDER=1 TOY_PARTS=fingers TOY_PHYS_GRIPPER=keep" \
            "palm|$B TWIN_RENDER=1 TOY_PARTS=palm" "palm_ik|$B TWIN_RENDER=1 TOY_PARTS=palm TOY_TWIN_POSE=ik" \
            "palm_keep|$B TWIN_RENDER=1 TOY_PARTS=palm TOY_PHYS_GRIPPER=keep" \
            "fingers_tipp5|$B TWIN_RENDER=1 TOY_PARTS=fingers TOY_TIP_SHIFT_MM=0,5,0" "GT_tipp5|$B TWIN_RENDER=1 TOY_TIP_SHIFT_MM=0,5,0"; do
  IFS='|' read n v <<< "$spec"
  env $v $S --name $n > $O/smoke_$n.log 2>&1 &
done
wait
grep -h -c Traceback $O/smoke_*.log | paste -sd' '
fi
echo "end=$(date)"
