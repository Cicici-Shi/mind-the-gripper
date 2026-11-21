#!/bin/bash
# maskinfo-pi freeze (protocol v2 9 item 9): chmod a-w the frozen set and write out/FROZEN (sha256sum format).
# Run only after: h200 manipulation checks pass (out/manip_pi/run_<id>/summary.json), v2 smoke + field dry-run with
# analysis/prereg_maskinfo_pi.py --check-fields-only pass, e2e S check pass, blind diff review done.
set -eu
T=$PROJ_ROOT/maskinfo_pi; cd $T
[ -e out/FROZEN ] && { echo "out/FROZEN exists already"; exit 2; }
F="protocol_maskinfo_pi.md $(ls embody/*.py embody/*.npz) lists/mi_pset.json pi/zoo_client_mi.py pi/serve_policy_mi.py
tools/mi_pi_params.sh tools/mi_pi_run.sbatch tools/gl_renderer.py tools/gate_maskinfo_pi.py tools/manip_maskinfo_pi.py
tools/manip_summary_pi.py tools/manip_pi.sbatch tools/submit_mi_pi_formal.sh tools/freeze_mi_pi.sh tools/visibility_pi.py tools/rerun_shard_mi_pi.sh tools/determinism_probe_pi.py tools/determinism_probe.sbatch tools/test_prereg_pi.sbatch tools/mi_pi_prefreeze.sbatch
pi/zoo_client_mi.diff_vs_next.txt analysis/prereg_maskinfo_pi.diff_vs_frozen.txt tools/gate_maskinfo_pi.diff_vs_frozen.txt
tools/manip_maskinfo_pi.diff_vs_frozen.txt tools/manip_summary_pi.diff_vs_frozen.txt
tools/check_fields_pi.py tools/state_e2e_pi.py tools/state_probe_pi.py tools/edit_equiv_pi.py tools/edit_equiv.sbatch
tools/smoke_checks_pi.py analysis/prereg_maskinfo_pi.py analysis/test_prereg_maskinfo_pi.py analysis/prereg_maskinfo_frozen_copy.py
analysis/ref/protocol_maskinfo_pi.md analysis/ref/prereg_maskrepl.py"
echo "openpi $(git -C $PROJ_ROOT/pi05/openpi rev-parse HEAD) $(git -C $PROJ_ROOT/pi05/openpi status --porcelain | wc -l) local changes" > out/FROZEN_ENV
sha256sum $PROJ_ROOT/vla_zoo/openpi_ckpt/pi0_libero/assets/physical-intelligence/libero/norm_stats.json \
  $PROJ_ROOT/pi05/ckpt/pi05_libero/assets/physical-intelligence/libero/norm_stats.json >> out/FROZEN_ENV
sha256sum $F > out/FROZEN
chmod a-w $F out/FROZEN out/FROZEN_ENV; cat out/FROZEN_ENV
sha256sum out/FROZEN; cat out/FROZEN
