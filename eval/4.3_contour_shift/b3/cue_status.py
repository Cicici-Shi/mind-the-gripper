"""闭环九格正式采集进度（盲）：只读每条记录的 complete 字段与 _errors/<键>.txt 是否存在，不读成败。
提交次数取 formal_state/submits.json（submit_cue.sh 维护）。键集与格只从冻结清单（--manifest）取。末行：ALL_COMPLETE / NEED_RESUBMIT / READY_WITH_EXCLUSIONS
（剩下的缺键都有报错文件且都提交过至少两次，这些键由门槛九格一起剔，见 protocol_bench2.md §4 第 1 项）。
  python cue_status.py [--root out_cue_closed] [--state formal_state]
"""
import argparse
import json
import os
import sys

import numpy as np

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import cue_spec as CS  # noqa: E402


def is_complete(p):
    if not os.path.exists(p):
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            return bool(z["complete"])
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(B3, "out_cue_closed"))
    ap.add_argument("--state", default=os.path.join(B3, "formal_state"))
    ap.add_argument("--manifest", default=os.path.join(B3, "FROZEN_cue_widowx"))
    ap.add_argument("--anchor", action="store_true", help="看锚点键集 x C0、C3、C4（--root 指向 out_cue_anchor 或 _post）")
    a = ap.parse_args()
    kp = CS.keyset_paths(B3, a.manifest, "anchor_keysets" if a.anchor else "keysets")
    cells = CS.ANCHOR_CELLS if a.anchor else json.load(open(a.manifest))["cell_order"]
    sp = os.path.join(a.state, "submits.json")
    subs = json.load(open(sp)) if os.path.exists(sp) else {}
    neither_any, err_once, total, done = False, False, 0, 0
    for b in CS.BLOCKS:
        keys = [k["key"] for k in json.load(open(kp[b]))]
        for c in cells:
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
            neither_any |= bool(neither)
            n_sub = subs.get(b, {}).get(c, {})
            err_once |= any(n_sub.get(k, 0) < 2 for k in err)
            print(f"{b} {c}：完整 {comp}/{len(keys)}；有报错无记录 {len(err)}；两者都没有 {len(neither)}")
            if err:
                print("  有报错无记录：" + "，".join(f"{k}（提交 {n_sub.get(k, 0)} 次）" for k in err))
    print(f"合计：完整 {done}/{total}")
    print("ALL_COMPLETE" if done == total else ("NEED_RESUBMIT" if (neither_any or err_once) else "READY_WITH_EXCLUSIONS"))


if __name__ == "__main__":
    main()
