#!/bin/bash
#SBATCH --job-name=ovr_diag
#SBATCH -p normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ovr_diag_%j.log
# After the re-run clients (afterany): reach_diag_m1 verify + diag on the re-run dumps (via reach_diag_ovla_rerun.py,
# reach_diag_m1.py unchanged), then the comparison / reachability summary (no outcomes printed).
set -u
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
export EMBODY_DIR=$PROJ_ROOT/embody_toy
Z=$PROJ_ROOT/vla_zoo; O=$Z/out/ovla_rerun/main
source ~/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $HOME/tmp
sha256sum $Z/tools/reach_diag_m1.py $Z/tools/reach_diag_ovla_rerun.py $Z/tools/ovla_rerun_compare.py $EMBODY_DIR/embody/body_env.py $EMBODY_DIR/embody/align.py $Z/out/ovla_rerun/keys.json
ls $O/g_tw_fc/*.npz | wc -l
T=$SLURM_JOB_ID
python -u $Z/tools/reach_diag_ovla_rerun.py --rerun-dir $O verify --model openvla_rerun --suite libero_spatial --cells g_tw_fc --offsets 0 --verify-keys 13 --tag $T 2>&1 \
  | grep -v "robosuite WARNING" | grep -vE '^\{' > $O/reach_verify_$T.log
python -u $Z/tools/reach_diag_ovla_rerun.py --rerun-dir $O diag --model openvla_rerun --suite libero_spatial --cells g_tw_fc --offsets 0 --tag $T 2>&1 \
  | grep -v "robosuite WARNING" | grep -vE '^\{' > $O/reach_diag_$T.log
tail -3 $O/reach_verify_$T.log $O/reach_diag_$T.log
python -u $Z/tools/ovla_rerun_compare.py main --rerun-dir $O --verify $O/reach_diag/verify_openvla_rerun_$T.json \
  --diag $O/reach_diag/diag_openvla_rerun_$T.json | tee $O/compare_$T.txt
sha256sum $O/compare_$T.txt $O/reach_diag/*_$T.json
