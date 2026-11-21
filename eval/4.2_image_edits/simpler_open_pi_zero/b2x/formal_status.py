"""补充协议正式采集进度（盲，b2x 版：块 B2、F1，格按 keys.EXT_CELLS，默认目录 out_formal_ext）：只读每条记录的 complete 字段和 _errors/<键>.txt 是否存在，绝不读、不打印任何成败字段。

每块每格输出：完整记录数；「有报错文件、无完整记录」的键；「两者都没有」的键；每个缺键被提交过几次
（= 它出现在多少个真正提交过的切片里：只数 formal_state/submissions.tsv 登记过的清单里列出的切片，空跑和被拒的不算）。
最后一行结论：
  ALL_COMPLETE           每块每格每键都有完整记录；
  NEED_RESUBMIT          还有「两者都没有」的键，或有报错的键只被提交过一次（或从没提交过）：再跑 submit_formal.sh；
  READY_WITH_EXCLUSIONS  剩下的缺键都有报错文件且都被提交过至少两次：可以读数，这些键会被门槛剔除。
用法：python formal_status.py [--root out_formal_ext] [--state formal_state] [--keysets keysets]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

B2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2)
import keys as K  # noqa: E402

BLOCKS = tuple(K.EXT_CELLS)


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
                        parts = t.split(":")
                        b, c, chunk = parts[0], parts[1], parts[-1]
                        with open(chunk) as cf:
                            for k in json.load(cf):
                                cnt[(b, c, k["key"])] += 1
    return cnt, n_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(B2, "out_formal_ext"))
    ap.add_argument("--state", default=os.path.join(B2, "formal_state"))
    ap.add_argument("--keysets", default=os.path.join(B2, "keysets"))
    a = ap.parse_args()
    subs, n_sub = submission_counts(a.state)
    print(f"已登记的提交次数：{n_sub}（{os.path.join(a.state, 'submissions.tsv')}）")
    neither_any, err_once, err_ok, total, done = False, False, 0, 0, 0
    for b in BLOCKS:
        with open(os.path.join(a.keysets, f"formal_{b}.json")) as f:
            keys = [k["key"] for k in json.load(f)]
        for c in K.EXT_CELLS[b]:
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
