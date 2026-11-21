#!/bin/bash
#SBATCH --job-name=mipi_readout
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi read-out (protocol v2.4 6fe3d8c4; deviation 2 of 2026-11-12). ONE primary run with the amended gate json,
# then ONE sensitivity run with the original gate json (literal key drop), separate outputs and locks. Main session.
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/maskinfo_pi
echo "readout start $(date) host $(hostname)"
sha256sum -c out/FROZEN --quiet && echo "FROZEN_CHECK OK" || { echo "FROZEN_CHECK FAIL"; exit 1; }
sha256sum protocol_maskinfo_pi_deviations.md out/gate_maskinfo_pi.json out/gate_maskinfo_pi_amended.json out/gate_amend_evidence.json readout/gate_amend_v3.py readout/readout_formal.sbatch
SHA=6fe3d8c4eaece3d47da40466daa0ef1b290cdefc90577a862b5f097e722046aa
B=pi05_spatial,pi05_object,pi0_spatial,pi0_object
python analysis/prereg_maskinfo_pi.py --protocol protocol_maskinfo_pi.md --expect-protocol-sha $SHA --runs out/runs \
  --gate out/gate_maskinfo_pi_amended.json --blocks $B --lag-steps 15 --deviations protocol_maskinfo_pi_deviations.md \
  --out out/readout/results_maskinfo_pi.txt --verdict-json out/readout/verdict_maskinfo_pi.json --quiet
echo "primary exit $?"
python analysis/prereg_maskinfo_pi.py --protocol protocol_maskinfo_pi.md --expect-protocol-sha $SHA --runs out/runs \
  --gate out/gate_maskinfo_pi.json --blocks $B --lag-steps 15 --deviations protocol_maskinfo_pi_deviations.md \
  --out out/readout_literal_gate/results_maskinfo_pi_literal_gate.txt --verdict-json out/readout_literal_gate/verdict_maskinfo_pi_literal_gate.json --quiet
echo "literal-gate exit $? end $(date)"
