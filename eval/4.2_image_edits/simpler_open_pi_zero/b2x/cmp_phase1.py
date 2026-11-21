"""补充协议 §1 的重跑比对：b2x 重跑的旧格（C0、C1、C3、C4、C5）与第一阶段正式记录（b2/out_formal，只读）
逐条比 raw_actions、env_actions、frame0_sha256。只读这三个字段，不读任何成败字段。

  python cmp_phase1.py --stage1   第一步：每个旧格至少 EXT_STAGE1_N 个键已比对且全部逐位相同，才写 formal_state/STAGE1_OK
                                  （submit_formal.sh 放全量的前提）；有任何不同或不足，写 STAGE1_FAILED 并退出码 1
  python cmp_phase1.py            全量后再比一次，照报每格相同键数（不写标记）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

B2X = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2X)
import gate as G  # noqa: E402
import keys as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(B2X, "out_formal_ext"))
    ap.add_argument("--ref-root", default=G.REF_ROOT)
    ap.add_argument("--state", default=os.path.join(B2X, "formal_state"))
    ap.add_argument("--stage1", action="store_true")
    a = ap.parse_args()
    res, ok = {}, True
    for b, cells in K.EXT_CELLS.items():
        keys = [k["key"] for k in json.load(open(os.path.join(B2X, "keysets", f"formal_{b}.json")))]
        for c in cells:
            if c not in K.EXT_OLD_CELLS:
                continue
            new = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(a.root, b, c, "*.npz"))}
            cmp_, same, differ, ref_missing = 0, 0, [], []
            for k in keys:
                if k not in new:
                    continue
                rp = os.path.join(a.ref_root, b, c, k + ".npz")
                if not os.path.exists(rp):
                    ref_missing.append(k)
                    continue
                with np.load(rp, allow_pickle=False) as zr, np.load(new[k], allow_pickle=False) as zn:
                    eq = all(f in zr.files and f in zn.files and zr[f].shape == zn[f].shape and np.array_equal(zr[f], zn[f])
                             for f in G.REF_FIELDS)
                cmp_ += 1
                same += eq
                if not eq:
                    differ.append(k)
            res[f"{b}/{c}"] = {"compared": cmp_, "identical": same, "differ": differ, "ref_missing": ref_missing}
            enough = cmp_ >= K.EXT_STAGE1_N
            ok = ok and enough and not differ and not ref_missing
            print(f"{b} {c}：比对 {cmp_} 键，逐位相同 {same}，不同 {len(differ)}，参照缺 {len(ref_missing)}"
                  + ("" if enough or not a.stage1 else f"（不足 {K.EXT_STAGE1_N}）"))
            for k in differ:
                print(f"  不同：{k}")
    if a.stage1:
        os.makedirs(a.state, exist_ok=True)
        name = "STAGE1_OK" if ok else "STAGE1_FAILED"
        for n in ("STAGE1_OK", "STAGE1_FAILED"):
            p = os.path.join(a.state, n)
            if os.path.exists(p):
                os.remove(p)
        with open(os.path.join(a.state, name), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(name)
        return 0 if ok else 1
    print("ALL_IDENTICAL" if all(v["compared"] == v["identical"] and not v["ref_missing"] for v in res.values())
          else "NOT_ALL_IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
