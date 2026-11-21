#!/bin/bash
#SBATCH --job-name=mic_gate
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-confirm per-block gate (protocol_maskinfo_confirm.md §7 item 7; never reads success). Env: BLOCK (label for
# the copy), T0JOB (t0 manip job id). Runs gate_maskinfo_confirm.py over every block present in out/runs, writes
# out/gate_maskinfo_confirm.json, keeps a copy out/gate_after_${BLOCK}_${SLURM_JOB_ID}.json and appends the summary
# lines (counts only) to out/GATE_LOG.txt.
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/maskinfo_confirm
M=$PROJ_ROOT/maskinfo/out/manip_<job>
python tools/gate_maskinfo_confirm.py --protocol protocol_maskinfo_confirm.md \
  --manip $M/m_0.jsonl $M/m_1.jsonl $M/m_2.jsonl $M/m_3.jsonl $M/m_4.jsonl $M/m_5.jsonl $M/m_6.jsonl $M/m_7.jsonl \
  --t0 out/manip_${T0JOB}/m_*_t0.jsonl --runs out/runs --logs "logs/mic_*_c30_C?_e3[05]_*.log" \
  --json out/gate_maskinfo_confirm.json > out/gate_tmp_$SLURM_JOB_ID.txt 2>&1; rc=$?
cp out/gate_maskinfo_confirm.json out/gate_after_${BLOCK}_${SLURM_JOB_ID}.json 2>/dev/null
{ echo "==== GATE after $BLOCK job $SLURM_JOB_ID $(date) exit $rc gate_json_sha $(sha256sum out/gate_maskinfo_confirm.json | cut -c1-64)"
  grep -E "^(manip|render_det|t0|oft|ofti)" out/gate_tmp_$SLURM_JOB_ID.txt; grep -c Traceback out/gate_tmp_$SLURM_JOB_ID.txt | sed "s/^/tracebacks /"; } >> out/GATE_LOG.txt
tail -12 out/GATE_LOG.txt
