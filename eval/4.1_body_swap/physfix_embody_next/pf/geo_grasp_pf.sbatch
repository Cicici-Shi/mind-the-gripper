#!/bin/bash
#SBATCH --job-name=pf_geo
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# physfix armature rulings Q2 (2026-11-07): geometric calibration sweep + pre-registered outcome-blind grasp gate
# (criteria in the header of geo_grasp_pf.py). Physics only, embody_next code unchanged, no policy, no pf_ outcome.
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disable ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
P=$PROJ_ROOT/vla_zoo/next/pf; O=$PROJ_ROOT/vla_zoo/out/pf/geo_$SLURM_JOB_ID; mkdir -p $O/j $O/logs
echo "node=$(hostname) start=$(date) script=$(sha256sum $P/geo_grasp_pf.py | cut -c1-16) body_env=$(sha256sum $PROJ_ROOT/embody_next/embody/body_env.py | cut -c1-16) robots=$(sha256sum $PROJ_ROOT/embody_next/embody/robots.py | cut -c1-16)"
L=$O/jobs.txt; : > $L
for v in panda a_kp100 kp100 kp20; do
  for s in libero_spatial libero_object; do for t in 0 5; do echo "sweep $s $t 0 $v" >> $L; done; done
  for t in 0 1 2 3 4 5 6 7 8 9; do for i in 0 1 2; do echo "grasp libero_object $t $i $v" >> $L; done; done
  for t in 0 2 3 6; do for i in 0 1 2; do echo "grasp libero_spatial $t $i $v" >> $L; done; done
done
echo "lines $(wc -l < $L)"
export P O
cat $L | xargs -P 30 -L 1 bash -c 'python -X faulthandler $P/geo_grasp_pf.py --mode $0 --suite $1 --task $2 --init $3 --variant $4 --out $O/j/$0_$1_$2_$3_$4.json > $O/logs/$0_$1_$2_$3_$4.log 2>&1'
echo "tracebacks: $(grep -l Traceback $O/logs/*.log | wc -l) json: $(ls $O/j | wc -l)"
python $P/geo_grasp_pf.py --mode summ --indir $O/j --out $O/summary.json
echo end $(date)
