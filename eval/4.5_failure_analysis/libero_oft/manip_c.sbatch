#!/bin/bash
#SBATCH --job-name=mic_t0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-confirm copy of the frozen tools/manip.sbatch, used only for the protocol §4 item 3 first-frame consistency
# (t0) gate on the confirm keys ep 30-39 (protocol_maskinfo_confirm.md §1: per-key gates still run). Same
# tools/manip_maskinfo.py (byte-identical copy 97829385...), same t0 flags. Changes: dir maskinfo_confirm, h200 card,
# 32 CPUs, and an optional 5th group field = --action-ep-shift (which existing dump's actions are replayed; the
# confirm run uses the SAME action sources as the first-round t0 gate: Spatial S_hook ep 40-49, Object obj_s ep 0-9).
# No policy, no outcomes. Env: GROUPS_T="suite:tasks:offset[:t0[:shift]] ..."; FLAGS; GL_EXPECT.
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
T=$PROJ_ROOT/maskinfo_confirm; cd $T; export PYTHONPATH=$PROJ_ROOT/openvla-oft
O=$T/out/manip_$SLURM_JOB_ID; mkdir -p $O
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader|head -1) GROUPS_T=$GROUPS_T FLAGS=$FLAGS start=$(date)"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); echo "DRIVER=$DRV"
if [ -n "${EXPECT_DRIVER:-}" ] && [ "$DRV" != "$EXPECT_DRIVER" ]; then echo "DRIVER changed: $DRV != $EXPECT_DRIVER (redo the bit-exact check first)"; exit 7; fi
R=$(python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-)
echo "GL_RENDERER=$R"
if [ -n "$GL_EXPECT" ] && ! [[ "$R" =~ $GL_EXPECT ]]; then echo "GL_RENDERER does not match $GL_EXPECT"; exit 5; fi
sha256sum embody/maskinfo.py embody/maskrepl.py tools/manip_maskinfo.py
i=0; for it in $GROUPS_T; do IFS=: read su tk off mode shift <<< "$it"
  X=""; [ "$mode" = t0 ] && X="--t0 --only-first 5 --max-steps 24 --png-keys 0"; [ -n "$shift" ] && X="$X --action-ep-shift $shift"
  python -X faulthandler tools/manip_maskinfo.py --suite $su --tasks $tk --offsets $off --out $O/m_${i}${mode:+_t0}.jsonl --png-dir $O/png $FLAGS $X > $O/p${i}.log 2>&1 &
  i=$((i+1)); sleep 3; done
wait; echo "end=$(date)"; for f in $O/p*.log; do echo "$f: $(grep -c '^{' $f) keys, tracebacks $(grep -c Traceback $f), done $(grep -c 'MANIP_MI done' $f)"; done
