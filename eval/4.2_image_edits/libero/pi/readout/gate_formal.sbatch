#!/bin/bash
#SBATCH --job-name=mipi_gate
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi formal gate (protocol_maskinfo_pi.md v2.4 6fe3d8c4, section 4; never reads success). Main session, 2026-11-12.
# Same arguments as the pre-freeze smoke gates (out/gate_smoke/gate_e40.json argv) except: --runs out/runs, all four
# blocks, default formal keys (tasks 0-9, eps 20-29), default --oft-ref (cross-model scene check against the OFT dumps).
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/maskinfo_pi
echo "gate start $(date) host $(hostname)"
sha256sum -c out/FROZEN --quiet && echo "FROZEN_CHECK OK" || echo "FROZEN_CHECK FAIL"
M=out/manip_pi/run_<job>
python tools/gate_maskinfo_pi.py --protocol protocol_maskinfo_pi.md --runs out/runs --json out/gate_maskinfo_pi.json \
  --manip $M/m_0.jsonl $M/m_1.jsonl $M/m_2.jsonl $M/m_3.jsonl \
  --t0 $M/m_10_t0.jsonl $M/m_11_t0.jsonl $M/m_4_t0.jsonl $M/m_5_t0.jsonl $M/m_6_t0.jsonl $M/m_7_t0.jsonl $M/m_8_t0.jsonl $M/m_9_t0.jsonl \
  --manip-summary $M/summary.json > out/gate_formal_stdout_$SLURM_JOB_ID.txt 2>&1
echo "gate exit $? end $(date)"
grep -c Traceback out/gate_formal_stdout_$SLURM_JOB_ID.txt | sed "s/^/tracebacks /"
sha256sum out/gate_maskinfo_pi.json
