"""第二阶段（CogACT）正式采集进度（盲）。由第一阶段 b2/formal_status.py 改写：路径换成 $PROJ_ROOT/bench2_cogact/b2c，
切片格式换成第二阶段驱动脚本的「块:L:线宽:切片文件」（每个 GPU 池一份清单，一行至多每卡进程数个），
切片内容是 [{cell, key}]；另打印 formal_state/node_fail.tsv 里起不来的节点。其余逻辑不变。

正式采集进度（盲）：只读每条记录的 complete 字段和 _errors/<键>.txt 是否存在，绝不读、不打印任何成败字段。

每块每格输出：完整记录数；「有报错文件、无完整记录」的键；「两者都没有」的键；每个缺键被提交过几次
（= 它出现在多少个真正提交过的切片里：只数 formal_state/submissions.tsv 登记过的清单里列出的切片，空跑和被拒的不算）。
最后一行结论：
  ALL_COMPLETE           每块每格每键都有完整记录；
  NEED_RESUBMIT          还有「两者都没有」的键，或有报错的键只被提交过一次（或从没提交过）：再跑 submit_formal.sh；
  READY_WITH_EXCLUSIONS  剩下的缺键都有报错文件且都被提交过至少两次：可以读数，这些键会被门槛剔除。
用法：python formal_status.py [--root out_formal] [--state formal_state] [--keysets keysets]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

B2 = os.path.dirname(os.path.abspath(__file__))
BLOCKS = ("B1", "B2", "F1", "F2")
CELLS = ("C0", "C1", "C3", "C4", "C5")


def is_complete(p):
    """只取 complete 字段（mmap 式按名取，不碰其他字段）。"""
    if not os.path.exists(p):
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            return bool(z["complete"])
    except Exception:
        return False


def submission_counts(state):
    """(块, 格, 键) -> 被提交过的次数，只数 submissions.tsv 登记的清单。"""
    cnt = collections.Counter()
    tsv = os.path.join(state, "submissions.tsv")
    if not os.path.exists(tsv):
        return cnt, 0
    n_sub = 0
    with open(tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or not os.path.exists(parts[3]):
                continue
            n_sub += 1
            with open(parts[3]) as mf:
                for row in mf:
                    for t in row.split():
                        b, _L, _lw, chunk = t.split(":", 3)
                        with open(chunk) as cf:
                            for it in json.load(cf):
                                cnt[(b, it["cell"], it["key"]["key"])] += 1
    return cnt, n_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(B2, "out_formal"))
    ap.add_argument("--state", default=os.path.join(B2, "formal_state"))
    ap.add_argument("--keysets", default=os.path.join(B2, "keysets"))
    a = ap.parse_args()
    subs, n_sub = submission_counts(a.state)
    print(f"已登记的提交次数：{n_sub}（{os.path.join(a.state, 'submissions.tsv')}）")
    nf = os.path.join(a.state, "node_fail.tsv")
    if os.path.exists(nf):
        rows = [l.rstrip("\n").split("\t") for l in open(nf) if l.strip()]
        print(f"起不来的作业 {len(rows)} 次，节点：{sorted({r[1] for r in rows if len(r) > 1})}")
    neither_any, err_once, err_ok, total, done = False, False, 0, 0, 0
    for b in BLOCKS:
        with open(os.path.join(a.keysets, f"formal_{b}.json")) as f:
            keys = [k["key"] for k in json.load(f)]
        for c in CELLS:
            d = os.path.join(a.root, b, c)
            comp, err, neither = 0, [], []
            for k in keys:
                total += 1
                if is_complete(os.path.join(d, k + ".npz")):
                    comp += 1
                    done += 1
                elif os.path.exists(os.path.join(d, "_errors", k + ".txt")):
                    err.append(k)
                else:
                    neither.append(k)
            if neither:
                neither_any = True
            for k in err:
                if subs[(b, c, k)] < 2:
                    err_once = True
                else:
                    err_ok += 1
            line = f"{b} {c}：完整 {comp}/{len(keys)}；有报错无记录 {len(err)}；两者都没有 {len(neither)}"
            print(line)
            for lab, ks in (("  有报错无记录", err), ("  两者都没有", neither)):
                if ks:
                    print(lab + "：" + "，".join(f"{k}（提交 {subs[(b, c, k)]} 次）" for k in ks))
    print(f"合计：完整 {done}/{total}")
    if done == total:
        print("ALL_COMPLETE")
    elif neither_any or err_once:
        print("NEED_RESUBMIT")
    else:
        print(f"READY_WITH_EXCLUSIONS（{err_ok} 个（块，格，键）有报错文件且已提交至少两次）")


if __name__ == "__main__":
    sys.exit(main())
