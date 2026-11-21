#!/bin/bash
# maskinfo-confirm driver (run as a CPU job afterany the last smoke job). Never reads success.
#  1. checks: 32 smoke npz; prereg --check-fields-only on every smoke npz; gate over out/smoke_runs has no label
#     failures; smoke job logs: GL_RENDERER H200, no episode / dump errors, no missing-weight warnings; t0 gate on
#     ep 30-39 (job T0JOB): 200 keys, 0 bad keys per cell, Object checks PASS, manip / render_det(h200) PASS;
#     prereg unit tests OK (re-run on the final code right before the freeze).  Any failure -> out/AUTODRIVE_STOPPED.txt, nothing frozen, nothing submitted.
#  MODE=check stops here and writes out/SMOKE_CHECKS_OK. MODE=freeze requires out/SMOKE_CHECKS_OK and
#  2. out/BLIND_REVIEW_PASS (written by the executor after the blind review passed), then:
#  3. freeze: chmod a-w + out/FROZEN (sha256 list); 4. submit 8 formal h200 jobs (tools/submit_mi_confirm.sh);
#  5. one CPU gate job per block, afterany that block's second shard; ids -> out/SUBMITTED.txt.
# Env: T0JOB, SMOKE_JOBS ("id id id id").
cd $PROJ_ROOT/maskinfo_confirm
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
L=out/autodrive.log; log() { echo "$(date +%F_%T) $*" | tee -a $L; }
stop() { log "STOP: $*"; echo "$(date) $*" > out/AUTODRIVE_STOPPED.txt; exit 1; }
log "start MODE=$MODE T0JOB=$T0JOB SMOKE_JOBS=$SMOKE_JOBS"
if [ "$MODE" = check ]; then
# ---- 1. checks
N=$(ls out/smoke_runs/*_c30/C?/t*_e*_i*.npz 2>/dev/null | wc -l); log "smoke npz $N"; [ "$N" = 32 ] || stop "smoke npz $N != 32"
python analysis/prereg_maskinfo_confirm.py --check-fields-only out/smoke_runs/*_c30/C?/t*_e*_i*.npz >> $L 2>&1 || stop "check-fields-only failed"
M=$PROJ_ROOT/maskinfo/out/manip_<job>
python tools/gate_maskinfo_confirm.py --protocol protocol_maskinfo_confirm.md --manip $M/m_[0-7].jsonl \
  --t0 out/manip_${T0JOB}/m_*_t0.jsonl --runs out/smoke_runs --json out/gate_smoke_check.json > out/gate_smoke_check.txt 2>&1 || stop "gate on smoke failed"
python - <<PY >> $L 2>&1 || stop "gate smoke / t0 checks failed (see autodrive.log)"
import json, sys
g = json.load(open("out/gate_smoke_check.json"))
bad = []
if g["protocol_sha256"] != "1b24c71d889ae040ceb77a4af57044b671fc8e4fa64fbe9b256a4d2f0553a05d": bad.append("protocol sha")
if (g["manip"] or {}).get("status") != "PASS": bad.append("manip not PASS")
if ((g["manip"] or {}).get("render_det") or {}).get("h200", {}).get("status") != "PASS": bad.append("render_det h200")
t0 = g["t0"]
if t0["n_keys"] != 200: bad.append(f"t0 keys {t0.get('n_keys')}")
if sorted({k[1] for k in t0["keys"]}) != list(range(30, 40)): bad.append("t0 eps not 30-39")
if any(v["bad_keys"] for v in t0["cells"].values()): bad.append("t0 bad keys " + str({c: len(v["bad_keys"]) for c, v in t0["cells"].items()}))
if t0["object_checks"]["status"] != "PASS": bad.append(f"t0 object {t0.get('object_checks')}")
if sorted(g["blocks"]) != sorted(["oft_spatial_c30", "ofti_spatial_c30", "oft_object_c30", "ofti_object_c30"]): bad.append(f"smoke blocks {sorted(g.get('blocks'))}")
for b, r in g["blocks"].items():
    if r["label_fail"]: bad.append(f"{b} label_fail {r.get('label_fail')[:2]}")
    if r["card_type"] != "h200": bad.append(f"{b} card {r.get('card_type')}")
    if any("label" in x for x in r["block_fail_reasons"]): bad.append(f"{b} {r.get('block_fail_reasons')}")
print("GATE_SMOKE_T0", "OK" if not bad else bad, "t0 keys", t0["n_keys"], "t0 bad", {c: len(v["bad_keys"]) for c, v in t0["cells"].items()})
sys.exit(1 if bad else 0)
PY
for j in $SMOKE_JOBS; do f=$(ls logs/mic_smoke_*_$j.log); grep -q "^GL_RENDERER=.*H200" $f || stop "smoke $j GL_RENDERER not H200"
  if grep -q "^DRIVER=" $f; then grep -q "^DRIVER=$(cat out/DRIVER_EXPECT)\$" $f || stop "smoke $j driver != $(cat out/DRIVER_EXPECT)"
  else  # job started before the DRIVER line was added: accept only if node41 has not rebooted since (same driver)
    BT=$(scontrol show node node41 | grep -o "BootTime=[^ ]*" | cut -d= -f2); [ "$BT" = "$(cat out/BOOT_EXPECT)" ] || stop "smoke $j has no DRIVER line and node41 BootTime $BT != BOOT_EXPECT"
    log "smoke $j: no DRIVER line; node41 BootTime $BT unchanged since 10-16, later jobs on it report DRIVER=$(cat out/DRIVER_EXPECT)"; fi; done
grep -q "^DRIVER=$(cat out/DRIVER_EXPECT)\$" logs/mic_t0_${T0JOB}.log || stop "t0 driver != DRIVER_EXPECT"
EV=$(ls logs/eval/*mi_*_c30_C?_e20.txt logs/eval/*mi_*_c30_C?_e0.txt)   # unwrapped local eval logs
NI=$(cat $EV | grep -c "GL_RENDERER_INPROC \[")
NB=$(cat $EV | grep -c "GL_RENDERER_INPROC ASSERT FAILED")
log "in-process GL_RENDERER lines $NI (expect 32: 4 smoke jobs x 8; the first oft_spatial smoke ${OLD_SMOKE:-<job>} predates the check), assert failures $NB"
[ "$NI" = 32 ] && [ "$NB" = 0 ] || stop "in-process GL check"
grep -h "GL_RENDERER_INPROC \[" $EV | sed "s/.*GL_RENDERER_INPROC //" | sort | uniq -c >> $L
E=$(cat $(ls logs/mic_*_c30_C?_e20_*.log logs/mic_*_c30_C?_e0_*.log) | grep -c -E "Episode error|Dump error")
W=$(cat $(ls logs/mic_*_c30_C?_e20_*.log logs/mic_*_c30_C?_e0_*.log) | grep -c -i -E "weights.*(not initialized|newly initialized)|missing keys|unexpected keys")
log "smoke per-process logs: episode/dump errors $E, missing-weight warnings $W"; [ "$E" = 0 ] && [ "$W" = 0 ] || stop "smoke log errors $E / warnings $W"
log "checks OK"; echo "$(date) autodrive check job ${SLURM_JOB_ID}" > out/SMOKE_CHECKS_OK; exit 0
fi
[ "$MODE" = freeze ] || stop "MODE must be check or freeze"
[ -e out/SMOKE_CHECKS_OK ] || stop "no out/SMOKE_CHECKS_OK"
# ---- 2. blind review
[ -e out/BLIND_REVIEW_PASS ] || stop "no out/BLIND_REVIEW_PASS"
log "blind review pass marker: $(cat out/BLIND_REVIEW_PASS)"
[ -e out/FROZEN ] && stop "out/FROZEN already exists"
TESTLOG=logs/tprereg_freeze_${SLURM_JOB_ID}.txt
(cd analysis && TMPDIR=$HOME/tmp python -m unittest test_prereg_maskinfo_confirm > ../$TESTLOG 2>&1)
grep -q "^OK" $TESTLOG || stop "unit tests not OK ($TESTLOG)"; log "unit tests on the code to be frozen: $(grep -E "^Ran" $TESTLOG) $(tail -1 $TESTLOG)"
# ---- 3. freeze
F="protocol_maskinfo_confirm.md analysis/ref/protocol_maskinfo_confirm.md oft/run_libero_eval_mi_c.py tools/gate_maskinfo_confirm.py analysis/prereg_maskinfo_confirm.py analysis/test_prereg_maskinfo_confirm.py lists/first_round_maskinfo.json tools/mi_cells_run_c.sbatch tools/submit_mi_confirm.sh tools/manip_c.sbatch tools/smoke_c.sh tools/extract_first_round.py tools/mic_autodrive.sh tools/mic_autodrive.sbatch tools/gate_block_c.sbatch embody/maskinfo.py embody/maskrepl.py embody/body_env.py tools/mi_cells.sh lists/mi_pset.json tools/gl_renderer.py tools/manip_maskinfo.py"
chmod a-w $F
{ echo "FROZEN $(date) (maskinfo-confirm autodrive job ${SLURM_JOB_ID}; smoke ${SMOKE_JOBS}; t0 ${T0JOB}; blind review: $(cat out/BLIND_REVIEW_PASS))"; sha256sum $F; } > out/FROZEN
chmod a-w out/FROZEN; log "frozen: $(wc -l < out/FROZEN) lines"
sha256sum -c --quiet <(tail -n +2 out/FROZEN) || stop "FROZEN self-check failed"
# ---- 4. submit
IDS=$(bash tools/submit_mi_confirm.sh 2>>$L) || stop "submission failed"
read -a A <<< "$IDS"; [ ${#A[@]} = 8 ] || stop "expected 8 job ids, got: $IDS"
log "submitted: $IDS"
# ---- 5. gate per block
G=""; i=0
for b in oft_spatial_c30 ofti_spatial_c30 oft_object_c30 ofti_object_c30; do
  j=${A[$((2*i+1))]}; g=$(sbatch --parsable --dependency=afterany:$j --export=ALL,BLOCK=$b,T0JOB=$T0JOB tools/gate_block_c.sbatch); G="$G $b:$g"; i=$((i+1))
done
log "gate jobs:$G"
{ echo "maskinfo-confirm formal submission $(date) (autodrive job ${SLURM_JOB_ID})"
  echo "jobs (oft_spatial_c30 e30 e35, ofti_spatial_c30 e30 e35, oft_object_c30 e30 e35, ofti_object_c30 e30 e35; all h200, afterany chain): $IDS"
  echo "gate jobs (afterany each block's 2nd shard; results appended to out/GATE_LOG.txt):$G"
  echo; cat out/FROZEN; echo
  echo "read (NOT by the executor; ONLY when the coordinator says so; per block, after its gate):"
  echo "  cd $PROJ_ROOT/maskinfo_confirm && python analysis/prereg_maskinfo_confirm.py --protocol protocol_maskinfo_confirm.md --expect-protocol-sha 1b24c71d889ae040ceb77a4af57044b671fc8e4fa64fbe9b256a4d2f0553a05d --runs out/runs --gate out/gate_maskinfo_confirm.json --blocks <block> --lag-steps 16 --out out/results_maskinfo_confirm.txt --verdict-json out/verdict_maskinfo_confirm.json"
} > out/SUBMITTED.txt
log "done"
