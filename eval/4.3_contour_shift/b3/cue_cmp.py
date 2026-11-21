"""v3.1 的逐位比对（protocol_cue_widowx_v3_1_final.md §4；§17 第 4 条）。参照一律是第一阶段正式记录（b2/out_formal，只读）。
  python cue_cmp.py --stage1   锚点键集（清单 anchor_keysets）x C0、C3、C4，本侧 out_cue_anchor/：三项全部逐位相同才写
                               formal_state/STAGE1_OK，否则 STAGE1_FAILED、退出码 1（submit_cue.sh 放全量的前提）
  python cue_cmp.py --post     同样 60 回合的采后重跑，本侧 out_cue_anchor_post/：写 out_cue_anchor_post/ANCHOR_POST.json（环境不变检查）
  python cue_cmp.py --init     主键集每个键、每个格（out_cue_closed/）对第一阶段同一块同一 episode 试次 0 的原图格记录，
                               比第 0 帧哈希与初始状态（init_robot_qpos、init_actor_poses、init_actor_names、机器人基座位姿），
                               写 out_cue_closed/INIT_CMP.json
  三项比对（--stage1、--post）：raw_actions、env_actions、frame0_sha256 都要比，参照或本侧缺任何一项算不过。
  冒烟模式 --pairs 块:格:参照根,...（--keys-from-root）：只有这里保留「参照没有 frame0_sha256 时只比两个动作字段」的退路。
  --init 也可配 --root out_cue_smoke --keys-from-root 在冒烟的校准键上跑。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import cue_spec as CS  # noqa: E402

REF_FIELDS = ("raw_actions", "env_actions", "frame0_sha256")
INIT_FIELDS = ("init_robot_qpos", "init_actor_poses", "init_actor_names")
REF_ROOT = os.path.expanduser("$PROJ_ROOT/bench2_simpler/b2/out_formal")
FROZEN = os.path.join(B3, "FROZEN_cue_widowx")


def cmp_one(rp, np_, want_fields=None, allow_missing_frame0=False):
    """逐位比 REF_FIELDS。allow_missing_frame0 只给冒烟 --pairs 用：参照没有 frame0_sha256 时只比两个动作字段。"""
    with np.load(rp, allow_pickle=False) as zr, np.load(np_, allow_pickle=False) as zn:
        fields = [f for f in REF_FIELDS if not (allow_missing_frame0 and f == "frame0_sha256" and f not in zr.files)]
        if want_fields is not None:
            want_fields.extend(fields)
        return all(f in zr.files and f in zn.files and zr[f].shape == zn[f].shape and np.array_equal(zr[f], zn[f])
                   for f in fields)


def cmp_init(rp, np_):
    """返回 (第 0 帧相同, 初始状态相同)。基座位姿：参照 init_robot_pose 对本侧 init_robot_pose 与 robot_base_pose。"""
    with np.load(rp, allow_pickle=False) as zr, np.load(np_, allow_pickle=False) as zn:
        f0 = "frame0_sha256" in zr.files and "frame0_sha256" in zn.files and str(zr["frame0_sha256"]) == str(zn["frame0_sha256"])
        ok = all(f in zr.files and f in zn.files and np.array_equal(zr[f], zn[f]) for f in INIT_FIELDS)
        ok = ok and "init_robot_pose" in zr.files and "init_robot_pose" in zn.files and np.array_equal(zr["init_robot_pose"], zn["init_robot_pose"])
        ok = ok and "robot_base_pose" in zn.files and np.array_equal(zr["init_robot_pose"], zn["robot_base_pose"])
        return bool(f0), bool(ok)


def manifest_keys(man_path, kind):
    paths = CS.keyset_paths(B3, man_path, kind)
    return {b: [k for k in json.load(open(p))] for b, p in paths.items()}


def three_way(root, ref_root, keys_by_block, cells, require_all):
    res, ok = {}, True
    for b, keys in keys_by_block.items():
        for c in cells:
            n, same, differ, ref_missing, missing, fields = 0, 0, [], [], [], set()
            for k in keys:
                p = os.path.join(root, b, c, k + ".npz")
                if not os.path.exists(p):
                    missing.append(k)
                    continue
                rp = os.path.join(ref_root, b, c, k + ".npz")
                if not os.path.exists(rp):
                    ref_missing.append(k)
                    continue
                fl = []
                eq = cmp_one(rp, p, fl)
                fields.update(fl)
                n += 1
                same += eq
                if not eq:
                    differ.append(k)
            res.setdefault(b, {})[c] = {"n_ref": n, "equal": same, "differ": differ, "ref_missing": ref_missing,
                                        "missing": missing, "fields": sorted(fields)}
            ok = ok and not differ and not ref_missing and not (require_all and missing) and n > 0
            print(f"{b} {c}：比对 {n}，逐位相同 {same}，不同 {len(differ)}，参照缺 {len(ref_missing)}，本侧缺 {len(missing)}，比的字段 {sorted(fields)}")
            for k in differ:
                print(f"  不同：{k}")
    return res, ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage1", action="store_true")
    g.add_argument("--post", action="store_true")
    g.add_argument("--init", action="store_true")
    g.add_argument("--pairs", default="", help="冒烟：块:格:参照根，逗号分隔")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ref-root", default=REF_ROOT)
    ap.add_argument("--state", default=os.path.join(B3, "formal_state"))
    ap.add_argument("--manifest", default=FROZEN)
    ap.add_argument("--keys-from-root", action="store_true", help="只在冒烟（--pairs、--init）用：键取 root 里已有的记录")
    a = ap.parse_args(argv)
    if a.pairs:
        root = a.root or os.path.join(B3, "out_cue_smoke")
        ok = True
        for b, c, ref in [tuple(x.split(":", 2)) for x in a.pairs.split(",")]:
            new = sorted(glob.glob(os.path.join(root, b, c, "*.npz")))
            n, same, flds = 0, 0, set()
            for p in new:
                rp = os.path.join(ref, b, c, os.path.basename(p))
                if os.path.exists(rp):
                    fl = []
                    e = cmp_one(rp, p, fl, allow_missing_frame0=True)
                    flds.update(fl)
                    n += 1
                    same += e
            ok = ok and n > 0 and same == n
            print(f"[冒烟] {b} {c} 对 {ref}：比对 {n}，逐位相同 {same}，比的字段 {sorted(flds)}")
        print("ALL_IDENTICAL" if ok else "NOT_ALL_IDENTICAL")
        return 0 if ok else 1
    if a.stage1 or a.post:
        root = a.root or os.path.join(B3, "out_cue_anchor" if a.stage1 else "out_cue_anchor_post")
        keys = {b: [k["key"] for k in v] for b, v in manifest_keys(a.manifest, "anchor_keysets").items()}
        res, ok = three_way(root, a.ref_root, keys, CS.ANCHOR_CELLS, require_all=True)
        if a.stage1:
            os.makedirs(a.state, exist_ok=True)
            for nm in ("STAGE1_OK", "STAGE1_FAILED"):
                if os.path.exists(os.path.join(a.state, nm)):
                    os.remove(os.path.join(a.state, nm))
            name = "STAGE1_OK" if ok else "STAGE1_FAILED"
            json.dump(res, open(os.path.join(a.state, name), "w"), ensure_ascii=False, indent=1)
            print(name)
        else:
            json.dump({"ok": ok, "per": res}, open(os.path.join(root, "ANCHOR_POST.json"), "w"), ensure_ascii=False, indent=1)
            print("ANCHOR_POST_OK" if ok else "ANCHOR_POST_FAILED")
        return 0 if ok else 1
    # --init
    root = a.root or os.path.join(B3, "out_cue_closed")
    if a.keys_from_root:
        keys = {b: sorted({os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root, b, "*", "*.npz"))}) for b in CS.BLOCKS}
        cells = sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "*", "*", "*.npz"))})
    else:
        keys = {b: [k["key"] for k in v] for b, v in manifest_keys(a.manifest, "keysets").items()}
        cells = json.load(open(a.manifest))["cell_order"]
    out = {"ok_frame0": {}, "ok_init": {}, "per": {}}
    for b in CS.BLOCKS:
        okf = oki = True
        for c in cells:
            d = {"n": 0, "frame0_equal": 0, "init_equal": 0, "differ_frame0": [], "differ_init": [], "missing": []}
            for k in keys[b]:
                p = os.path.join(root, b, c, k + ".npz")
                if not os.path.exists(p):
                    d["missing"].append(k)
                    continue
                ep = k.split("__")[1]
                rp = os.path.join(a.ref_root, b, "C0", f"{b}__{ep}__t0.npz")
                f0, ini = cmp_init(rp, p) if os.path.exists(rp) else (False, False)
                d["n"] += 1
                d["frame0_equal"] += f0
                d["init_equal"] += ini
                if not f0:
                    d["differ_frame0"].append(k)
                if not ini:
                    d["differ_init"].append(k)
            if d["n"] == 0:
                continue
            okf = okf and not d["differ_frame0"] and (a.keys_from_root or not d["missing"])
            oki = oki and not d["differ_init"] and (a.keys_from_root or not d["missing"])
            out["per"].setdefault(b, {})[c] = d
            print(f"{b} {c}：比对 {d['n']}，第 0 帧相同 {d['frame0_equal']}，初始状态相同 {d['init_equal']}，缺 {len(d['missing'])}")
        out["ok_frame0"][b], out["ok_init"][b] = okf, oki
    json.dump(out, open(os.path.join(root, "INIT_CMP.json"), "w"), ensure_ascii=False, indent=1)
    good = all(out["ok_frame0"].values()) and all(out["ok_init"].values())
    print("INIT_OK" if good else "INIT_DIFFER")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
