#!/bin/bash
#SBATCH -p normal
#SBATCH --job-name=prereg_b3_final
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/prereg_final_%j.log
# Amendment 2 section 5: frozen script (literal) and a2 copy back to back in ONE job; nobody reads either before both are written.
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1
source ~/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/embody_v3/analysis
sha256sum prereg_block3.py prereg_block3_a2.py ../protocol_block3*.md
python prereg_block3.py --out results_block3_literal.txt > /dev/null 2> literal.err; echo "literal exit=$?"
python prereg_block3_a2.py --out results_block3_final.txt > /dev/null 2> final.err; echo "a2 exit=$?"
python prereg_block3_a2.py --suite object --out results_block3_object.txt > /dev/null 2> object.err; echo "a2 object exit=$?"
python $HOME/tmp/prereg_b3_test/cmp_estimands.py results_block3_literal.txt results_block3_final.txt
echo DONE
