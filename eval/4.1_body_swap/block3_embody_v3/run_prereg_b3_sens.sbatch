#!/bin/bash
#SBATCH -p normal
#SBATCH --job-name=b3_obj_sens
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prereg_b3_sens_%j.log
# Fable add1 §2.3 item 4: post-hoc robustness re-run of the PUBLISHED block-3 object readout with the physics-breakdown
# keys removed from every OFT object cell. Same defaults as tools/run_prereg_final.sbatch (a2 --suite object).
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/embody_v3/analysis
EX=$PROJ_ROOT/vla_zoo/out/physics_breakdown_published.json
for o in results_block3_object_sens_physics.txt results_block3_object_sens_noexcl.txt; do [ -e $o ] && { echo "refuse: $o exists"; exit 1; }; done
sha256sum prereg_block3.py prereg_block3_a2.py prereg_block3_a2_sens.py prereg_block3_a2_sens.diff ../tools/make_a2_sens.py $EX results_block3_object.txt
python prereg_block3_a2_sens.py --suite object --out results_block3_object_sens_noexcl.txt > /dev/null 2> sens_noexcl.err; echo "sens no-exclude exit=$?"
python $HOME/tmp/prereg_b3_sens_test/cmp_sens.py results_block3_object.txt results_block3_object_sens_noexcl.txt; echo "reproduces published (minus first line / sha section): exit=$?"
python prereg_block3_a2_sens.py --suite object --exclude-keys $EX --out results_block3_object_sens_physics.txt > /dev/null 2> sens_physics.err; echo "sens physics exit=$?"
chmod a-w results_block3_object_sens_physics.txt results_block3_object_sens_noexcl.txt
sha256sum results_block3_object_sens_physics.txt results_block3_object_sens_noexcl.txt
echo DONE
