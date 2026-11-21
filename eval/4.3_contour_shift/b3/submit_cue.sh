#!/bin/bash
# protocol_cue_widowx_v3_1_final 闭环十三格的正式采集驱动（§4、§15；§17 第 2 条）。
#   bash submit_cue.sh --stage1   锚点：清单 anchor_keysets（第一阶段正式清单前 10 键）x C0、C3、C4，写 out_cue_anchor/；
#                                 跑完 python cue_cmp.py --stage1，三项全部逐位相同才写 formal_state/STAGE1_OK
#   bash submit_cue.sh            全量：要求 STAGE1_OK；只跑主键集（清单 keysets）x 清单 cells 全部格，写 out_cue_closed/
#   bash submit_cue.sh --post     采后环境不变检查：同样 60 回合，写 out_cue_anchor_post/；跑完 python cue_cmp.py --post
#   加 --dry-run 只生成计划与作业脚本（临时目录），不提交，不要求冻结清单（用 .candidate）。
# 只用整卡 A100-40；数组 %1（CogACT 占另一张；另一张确实空着时才 MAXPAR=2）。可重复运行，只补缺。
# 状态落盘 formal_state/：键集副本、分片、submissions.tsv、submits.json（{块: {格: {键: 提交次数}}}，门槛的补采停止规则用）。
set -euo pipefail
B3=$PROJ_ROOT/bench2_simpler/b3
FROZEN=$B3/FROZEN_cue_widowx
CAND=$B3/FROZEN_cue_widowx.candidate
OUT=$B3/out_cue_closed
PAR=${PAR:-4}
MAXPAR=${MAXPAR:-1}
CHUNK=${CHUNK:-}
DRY=0; STAGE1=0; POST=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --stage1) STAGE1=1 ;;
    --post) POST=1 ;;
    *) echo "未知参数 $arg" >&2; exit 1 ;;
  esac
done
cd "$B3"
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_simpler
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$HOME/tmp
if [ $DRY = 0 ]; then
  [ -f "$FROZEN" ] || { echo "拒绝提交：$FROZEN 不存在（协议未冻结）。" >&2; exit 1; }
  MANIFEST=$FROZEN; ST=$B3/formal_state
  if [ $STAGE1 = 0 ] && [ $POST = 0 ] && [ ! -f "$ST/STAGE1_OK" ]; then
    echo "拒绝提交全量：$ST/STAGE1_OK 不存在（先 --stage1，再 python cue_cmp.py --stage1）。" >&2; exit 1
  fi
  if ! QJOBS=$(squeue -u "$USER" -h -o "%j"); then echo "拒绝提交：squeue 失败。" >&2; exit 1; fi
  if [ -n "$(printf '%s\n' "$QJOBS" | awk '/^b3_cue_f/ {print; exit}')" ]; then
    echo "已有 b3_cue_f 作业在队列里，先等它们结束。" >&2; exit 1
  fi
else
  if [ -f "$FROZEN" ]; then MANIFEST=$FROZEN; else MANIFEST=$CAND; fi
  [ -f "$MANIFEST" ] || { echo "空跑也需要冻结清单或候选（先跑 freeze_cue.py）。" >&2; exit 1; }
  ST=$(mktemp -d "$B3/_dryrun_XXXXXX"); echo "--dry-run：清单 $MANIFEST，临时目录 $ST（不提交）"
fi
python - "$MANIFEST" "$B3" "$DRY" <<'PY'
import hashlib, json, os, sys
man, b3, dry = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3] == "1"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = [f"代码 {f}" for f, h in man["code_sha256"].items() if sha(os.path.join(b3, f)) != h]
bad += [f"键集 {b}" for b, v in man["keysets"].items() if sha(os.path.join(b3, v["path"])) != v["sha256"]]
bad += [f"锚点键集 {b}" for b, v in man["anchor_keysets"].items() if sha(os.path.join(b3, v["path"])) != v["sha256"]]
bad += [] if sha(os.path.join(b3, man["tref_file"])) == man["tref_sha256"] else ["t_ref 清单"]
if bad:
    print(("空跑提示" if dry else "拒绝提交") + "：与冻结清单不一致：" + "，".join(bad), file=sys.stderr); sys.exit(0 if dry else 1)
print("代码、键清单、t_ref 清单的哈希与冻结清单一致", file=sys.stderr)
PY
mkdir -p "$ST/chunks"; [ $DRY = 1 ] || mkdir -p "$B3/logs/formal"
if [ $STAGE1 = 1 ] || [ $POST = 1 ]; then KIND=anchor_keysets; KF=anchor; else KIND=keysets; KF=keys; fi
if [ $STAGE1 = 1 ]; then OUT=$B3/out_cue_anchor; elif [ $POST = 1 ]; then OUT=$B3/out_cue_anchor_post; fi
for b in B1 B2; do
  KS=$B3/$(python -c "import json; print(json.load(open('$MANIFEST'))['$KIND']['$b']['path'])")   # 键集取自清单
  [ -f "$ST/${KF}_$b.json" ] || cp "$KS" "$ST/${KF}_$b.json"
  cmp -s "$ST/${KF}_$b.json" "$KS" || { echo "拒绝提交：$ST/${KF}_$b.json 与冻结键集 $KS 不一致" >&2; exit 1; }
done
TAG=$(date +%Y%m%d_%H%M%S)
MAN="$ST/manifest_$TAG.txt"   # 每行一个数组任务：空格分隔的「块:格:分片文件」，至多 PAR 个
python - "$ST" "$OUT" "$TAG" "$PAR" "${CHUNK:-0}" "$MAN" "$MANIFEST" "$KF" <<'PY'
import json, os, sys
import numpy as np
st, out, tag, par, chunk, man, frozen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
kf = sys.argv[8]
m = json.load(open(frozen))
anchor_run = kf == "anchor"
cells = m["anchor_cells"] if anchor_run else m["cell_order"]  # 锚点键只跑三个锚点格（cue_closed.py 里也拦）
# 每回合秒数（同卡 4 进程，第一阶段实测 B1 22.3、B2 15.4 秒，新格每查询多一次渲染与统计，取 1.35 倍余量）；每进程至多 47 分钟
SEC = {"B1": 30, "B2": 21}  # 冒烟实测 B1 17 到 28、B2 9 到 12 秒；OB 格每查询多一次渲染（见交回吞吐）
tasks = []
for b in ("B1", "B2"):
    keys = json.load(open(os.path.join(st, f"{kf}_{b}.json")))
    n_max = chunk or max(1, int((47 * 60) // SEC[b]))
    for c in cells:
        miss = []
        for k in keys:
            p = os.path.join(out, b, c, k["key"] + ".npz")
            ok = False
            if os.path.exists(p):
                try:
                    with np.load(p) as z:
                        ok = bool(z["complete"])
                except Exception:
                    ok = False
            if not ok:
                miss.append(k)
        n_chunk = -(-len(miss) // n_max) if miss else 0
        n = -(-len(miss) // n_chunk) if n_chunk else 1
        for i in range(0, len(miss), n):
            f = os.path.join(st, "chunks", f"{tag}_{b}_{c}_{i // n:03d}.json")
            json.dump(miss[i:i + n], open(f, "w"))
            tasks.append(f"{b}:{c}:{f}")
        print(f"{b} {c}: 缺 {len(miss)} 键，{n_chunk} 份", file=sys.stderr)
with open(man, "w") as fh:
    for i in range(0, len(tasks), par):
        fh.write(" ".join(tasks[i:i + par]) + "\n")
print(f"共 {len(tasks)} 份", file=sys.stderr)
PY
NT=$(wc -l < "$MAN" | tr -d ' ')
if [ "$NT" = "0" ]; then echo "没有缺的键，全部完成。"; exit 0; fi
echo "本次需要 $NT 个 GPU 作业（每个至多 $PAR 个进程），清单 $MAN"
JOB=$ST/job_$TAG.sbatch
cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH -J b3_cue_f_$TAG
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:58:00
#SBATCH --array=0-$((NT - 1))%$MAXPAR
#SBATCH -o $B3/logs/formal/%x_%A_%a.log
source $B3/jobs/common.sh
nvidia-smi -L | grep -q "A100-PCIE-40GB" || { echo "not A100-PCIE-40GB (whitelist), abort"; exit 1; }
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" $MAN)
for t in \$LINE; do
  IFS=: read b c f <<< "\$t"
  python cue_closed.py --block \$b --cell \$c --keys \$f --out $OUT --skip-done \
    > $B3/logs/formal/run_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${b}_\${c}_\$(basename \$f .json).log 2>&1 &
  sleep 10
done
wait
grep -l "ERROR" $B3/logs/formal/run_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_* || true
EOS
if [ $DRY = 1 ]; then echo "--dry-run：不提交。作业脚本 $JOB；清单 $MAN"; exit 0; fi
JID=$(sbatch --parsable "$JOB")
printf "%s\t%s\t%s\t%s\n" "$(date -Is)" "$JID" "$NT" "$MAN" >> "$ST/submissions.tsv"
python - "$ST" "$MAN" <<'PY'
import json, os, sys
st, man = sys.argv[1], sys.argv[2]
p = os.path.join(st, "submits.json")
d = json.load(open(p)) if os.path.exists(p) else {}
for line in open(man):
    for t in line.split():
        b, c, f = t.split(":", 2)
        for k in json.load(open(f)):
            d.setdefault(b, {}).setdefault(c, {})
            d[b][c][k["key"]] = d[b][c].get(k["key"], 0) + 1
json.dump(d, open(p, "w"), indent=1)
PY
echo "已提交数组作业 $JID（$NT 个任务，同时至多 $MAXPAR 个）。跑完后重跑本脚本只补缺的键；看进度 python cue_status.py。"
