#!/bin/bash
#SBATCH --job-name=pf_grasp
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# physfix rev3 7.1 grasp plausibility (CPU, state reconstruction from npz qpos, never reads success). Two columns:
# kp20 (G'', GTfc'' of both checkpoints) and old_normal (published obj_g minus D keys).
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disable ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
Z=$PROJ_ROOT/vla_zoo; P=$Z/next/pf; O=$Z/out/pf/grasp_$SLURM_JOB_ID; mkdir -p $O/k $O/logs
echo "node=$(hostname) start=$(date) script=$(sha256sum $P/grasp_plaus_pf.py|cut -c1-16) body_env=$(sha256sum $PROJ_ROOT/embody_next/embody/body_env.py|cut -c1-16) robots=$(sha256sum $PROJ_ROOT/embody_next/embody/robots.py|cut -c1-16)"
L=$O/jobs.txt; : > $L
for t in 0 1 2 3 4 5 6 7 8 9; do
  echo "old_obj_g $PROJ_ROOT/runs/obj_g $t" >> $L
  for ck in oftc:OFT ofti:OFTi; do d=${ck%%:*}; n=${ck##*:}
    echo "${n}_G2 $Z/runs/pf_${d}_obj/pf_obj_grpkp20 $t" >> $L
    echo "${n}_GTfc2 $Z/runs/pf_${d}_obj/pf_obj_g_tfc_kp20 $t" >> $L
  done
done
export P O
cat $L | xargs -P 30 -L 1 bash -c 'python $P/grasp_plaus_pf.py --mode key --cell $1 --task $2 --out $O/k/$0_t$2.json > $O/logs/$0_t$2.log 2>&1'
echo "tracebacks: $(grep -l Traceback $O/logs/*.log | wc -l) json: $(ls $O/k | wc -l)"
python $P/grasp_plaus_pf.py --mode summ --indir $O/k --out $O/grasp_physfix.json
echo "end=$(date) sha=$(sha256sum $O/grasp_physfix.json | cut -c1-16)"
