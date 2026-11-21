#!/bin/bash
#SBATCH --job-name=pf_cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x_%j.log
# physfix pre-freeze CPU checks (protocol_physfix.md 7.1 (ii)/(iii), 5.3; Astra rulings 3), no policy, no outcome:
#  1. xml_pf.py: MJCF + compiled-model diff, R85_ARMATURE off vs 1e-3, Panda and PandaR85, spatial + object
#  2. grip_sweep_pf.py: whole open-close cycle of the Robotiq proprio / twin-finger map under armature (free air)
#  3. outcome-blind replay re-check of the failure rate with the embody_next XML implementation: diag_next/phys/phys_replay.py
#     (read-only, --code embody_next, variant base, R85_ARMATURE=1e-3 exported), published obj_g / obj_g_t (object) and
#     grpkp100 (spatial), every dumped key, seeds 0,1 (eps 1e-6) = the WORKLOG_vlazoo 14 design.
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disable ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
N=$PROJ_ROOT/embody_next; P=$PROJ_ROOT/vla_zoo/next/pf; O=$PROJ_ROOT/vla_zoo/out/pf/cpu_$SLURM_JOB_ID; mkdir -p $O/replay $O/logs
PR=$PROJ_ROOT/diag_next/phys/phys_replay.py
echo "node=$(hostname) start=$(date) robots=$(sha256sum $N/embody/robots.py | cut -c1-16) body_env=$(sha256sum $N/embody/body_env.py | cut -c1-16) phys_replay=$(sha256sum $PR | cut -c1-16)"
python -X faulthandler $P/xml_pf.py --code $N --out $O/xml_pf.json > $O/logs/xml.log 2>&1 &
python -X faulthandler $P/grip_sweep_pf.py --code $N --out $O/grip_sweep.json > $O/logs/grip.log 2>&1 &
J=$O/jobs.txt; : > $J
for spec in obj_g:object obj_g_t:object grpkp100:spatial; do c=${spec%%:*}; s=${spec##*:}
  for t in 0 1 2 3 4 5 6 7 8 9; do
    ks=$(ls $PROJ_ROOT/runs/$c | sed -n "s/^t${t}_e\([0-9]*\)_i.*/${t}:\1/p" | sort -t: -k2 -n | paste -sd, -)
    [ -n "$ks" ] && echo "$c|$ks|$s" >> $J
  done; done
echo "replay lines $(wc -l < $J)"
export O PR N
cat -n $J | xargs -P 28 -L 1 bash -c 'IFS="|" read -r c k s <<< "$1"; R85_ARMATURE=1e-3 python -X faulthandler $PR --code $N --cond $c --keys $k --variant base --tag _xmlarm --out $O/replay --seeds 0,1 --suite $s > $O/logs/pr_$0.log 2>&1'
wait
echo "tracebacks: $(grep -l Traceback $O/logs/*.log | wc -l)"; tail -2 $O/logs/xml.log; tail -1 $O/logs/grip.log
python $P/replay_summ_pf.py $O/replay
echo end $(date)
