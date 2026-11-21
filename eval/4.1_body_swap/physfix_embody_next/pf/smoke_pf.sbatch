#!/bin/bash
#SBATCH --job-name=pf_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# physfix gate before any pf_ collection (exit 0 = all pass; collection jobs depend afterok on this job):
#  (a) CPU checks of pf_cpu (CPUDIR): XML diff, outcome-blind replay (severe 0, QACC 0), armature applied in sweep;
#  (b) regression 7.1 (i)/(ii) with nx_regress.py (render on, OFTi models1 spatial + object dumps, keys (0,0),(2,0)):
#      S, G: embody_toy vs embody_next (switches off) qpos bitwise; S, T0: R85_ARMATURE off vs 1e-3 qpos bitwise;
#  (c) launcher smoke on ALREADY-READ models1 cells (OFTi spatial sham_twin, g_tfc; ep 0 of every task, TRIALS=1) into
#      runs/pf_smoke/: pf_* npz fields, no outcome / video lines in the logs. No pf_ cell is run here.
set -u
export TMPDIR=$HOME/tmp HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
Z=$PROJ_ROOT/vla_zoo; P=$Z/next/pf; N=$PROJ_ROOT/embody_next; T=$PROJ_ROOT/embody_toy
O=$Z/out/pf/smoke_$SLURM_JOB_ID; mkdir -p $O
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date) CPUDIR=$CPUDIR"
CELLS=$Z/tools/zoo_cells.sh
R="python $N/tools/nx_regress.py --render 1 --tasks 0,2 --eps 0-0"
for suite in libero_spatial libero_object; do
  if [ $suite = libero_spatial ]; then RUNS=$Z/runs/oft_sp; X=""; else RUNS=$Z/runs/oft_obj; X="obj_"; fi
  for c in sham_cg grpkp100; do set -- $(bash $CELLS vars $c); b=$1; shift
    env -u R85_ARMATURE "$@" $R --suite $suite --runs $RUNS --code $T --cond $X$c --body $b --out $O/${suite}_${c}_old > $O/${suite}_${c}_old.log 2>&1 &
    env -u R85_ARMATURE "$@" $R --suite $suite --runs $RUNS --code $N --cond $X$c --body $b --out $O/${suite}_${c}_new > $O/${suite}_${c}_new.log 2>&1 &
  done
  for c in sham_cg sham_twin; do set -- $(bash $CELLS vars $c); b=$1; shift
    [ $c = sham_twin ] && env -u R85_ARMATURE "$@" $R --suite $suite --runs $RUNS --code $N --cond $X$c --body $b --out $O/${suite}_${c}_new > $O/${suite}_${c}_new.log 2>&1 &
    env R85_ARMATURE=1e-3 "$@" $R --suite $suite --runs $RUNS --code $N --cond $X$c --body $b --out $O/${suite}_${c}_newarm > $O/${suite}_${c}_newarm.log 2>&1 &
  done
done
wait  # 2026-11-07: regression render processes must finish first (<job> OOM: 2 OFT + 12 render procs on one 40G card)
# launcher smoke (already-read models1 OFTi spatial cells, ep 0 of each task)
export EMBODY_DIR=$N PYTHONPATH=$PROJ_ROOT/openvla-oft M1_FASTSYNC=1 NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $Z/oft
for c in sham_twin g_tfc; do set -- $(bash $Z/next/cells_next.sh vars $c); b=$1; shift
  mkdir -p $Z/runs/pf_smoke/$c
  env BODY=$b "$@" DUMP_DIR=$Z/runs/pf_smoke/$c EP_OFFSET=0 SKIP_DONE=1 python -X faulthandler $P/launch_pf.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --task_suite_name libero_spatial \
    --num_trials_per_task 1 --run_id_note pf_smoke_$c --local_log_dir $O/oft_eval > $O/launch_$c.log 2>&1 &
  sleep 20
done
wait
# combined OFT entry (already-read published spatial sham_cg ep 0 keys; checks the combined checkpoint's unnorm key path)
set -- $(bash $Z/next/cells_next.sh vars sham_cg); b=$1; shift
mkdir -p $Z/runs/pf_smoke/oftc_sham_cg
env HF_HOME=$HOME/hf_cache BODY=$b "$@" DUMP_DIR=$Z/runs/pf_smoke/oftc_sham_cg EP_OFFSET=0 SKIP_DONE=1 python -X faulthandler $P/launch_pf.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 --task_suite_name libero_spatial \
  --num_trials_per_task 1 --run_id_note pf_smoke_oftc --local_log_dir $O/oft_eval > $O/launch_oftc_sham_cg.log 2>&1
echo "regress tracebacks: $(grep -l Traceback $O/*_old.log $O/*_new*.log | wc -l)"
python $P/check_pf.py --cpu $CPUDIR --regr $O --smoke $Z/runs/pf_smoke/sham_twin $Z/runs/pf_smoke/g_tfc $Z/runs/pf_smoke/oftc_sham_cg --logs $O/launch_sham_twin.log $O/launch_g_tfc.log $O/launch_oftc_sham_cg.log $O/oft_eval/*.txt
rc=$?
echo "end=$(date) rc=$rc"
exit $rc
