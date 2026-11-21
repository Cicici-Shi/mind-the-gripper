"""键核查（协议 v1.1 冻结前；不跑模型、不看成败）：四块全部正式键与校准键，每个键 reset 两遍。
  第一遍按键清单顺序，第二遍倒序（中间夹着别的键），比对两遍 reset 后的状态是否逐位相同；
  每次 reset 后用 ViewMaker.on_reset 解析 G 连杆与保护名单（找不到或个数不对就记失败，不中断）；
  核对实际生效的 episode_id / urdf 与键清单一致（eval_views.check_reset_spec）。
场景签名 = 任务 + urdf + 全部 actor 名与初始位姿（逐位）+ 机器人关节与位姿；统计每块不同场景数、每个场景的键数；
另统计「场景 + 语言指令」的不同数（move near 同一布局下源 / 目标物体不同，只差指令）。
用法：python probe_keys.py --out keyprobe_v11.json
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import traceback

B2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2)
import numpy as np  # noqa: E402

import eval_views as E  # noqa: E402
import keys as K  # noqa: E402
import views as V  # noqa: E402


def state_bytes(env):
    s = E.scene_state(env)
    h = hashlib.sha256()
    for f in ("init_robot_qpos", "init_robot_pose", "init_actor_poses"):
        h.update(np.ascontiguousarray(s[f]).tobytes())
    for f in ("init_actor_names", "init_art_names", "init_art_qpos", "init_urdf_version"):
        h.update(str(s[f]).encode())
    return h.hexdigest(), s


def one(env, blk, k):
    obs, info = E.reset_key(env, k)
    r = {"key": k["key"], "ok": True, "err": None}
    try:
        got_ep, got_urdf = E.check_reset_spec(env, info, k)
        r.update(episode_id=got_ep, urdf=got_urdf)
    except AssertionError as e:
        r.update(ok=False, err=f"reset 规格不符：{e}")
    try:
        vm = V.ViewMaker(env, "C0", K.BLOCKS[blk]["body"], 8 if K.is_bridge(blk) else 4, 8,
                         4 if K.is_bridge(blk) else 2, K.PROTECT[blk], K.PROTECT_EXPECT_N[blk])
        vm.on_reset()
        cam = V.cam_name(env.unwrapped.robot_uid)
        seg = obs["image"][cam]["Segmentation"][..., 1]
        r.update(p_names=vm.p_names, G_px=int(np.isin(seg, vm.gids).sum()), P_px=int(np.isin(seg, vm.pids).sum()))
    except AssertionError as e:
        r.update(ok=False, err=(r["err"] or "") + f" | G/保护解析失败：{e}")
    h, s = state_bytes(env)
    instr = env.get_language_instruction()
    r.update(state=h, task=k["task"], instruction=instr,
             state_instr=hashlib.sha256((h + "|" + instr).encode()).hexdigest())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", nargs="+", default=list(K.BLOCKS))
    a = ap.parse_args()
    import simpler_env
    E.check_env_src()
    res = {}
    for blk in a.blocks:
        keys = [dict(k, which="formal") for k in K.make_keys(blk, "formal")] + \
               [dict(k, which="calib") for k in K.make_keys(blk, "calib")]
        envs = {}
        p1, p2 = {}, {}
        for order, store in ((keys, p1), (keys[::-1], p2)):
            for k in order:
                if k["task"] not in envs:
                    envs[k["task"]] = simpler_env.make(k["task"])
                try:
                    store[k["key"]] = one(envs[k["task"]], blk, k)
                except Exception:
                    store[k["key"]] = {"key": k["key"], "ok": False, "err": traceback.format_exc()[-800:], "state": None}
        for env in envs.values():
            env.close()
        bad = [kk for kk in p1 if not (p1[kk]["ok"] and p2[kk]["ok"])]
        nondet = [kk for kk in p1 if p1[kk]["state"] is None or p1[kk]["state"] != p2[kk]["state"]]
        out = {"n_keys": len(keys), "fail_keys": bad, "fail_msgs": {kk: (p1[kk]["err"], p2[kk]["err"]) for kk in bad},
               "reset_twice_not_bit_equal": nondet}
        for which in ("formal", "calib"):
            ks = [k for k in keys if k["which"] == which]
            sc = collections.Counter(p1[k["key"]]["state"] for k in ks)
            si = collections.Counter(p1[k["key"]]["state_instr"] for k in ks)
            d = {"n": len(ks), "distinct_scenes": len(sc), "keys_per_scene": dict(collections.Counter(sc.values())),
                 "distinct_scene_plus_instruction": len(si),
                 "distinct_instructions": len({p1[k["key"]]["instruction"] for k in ks}),
                 "urdf": dict(collections.Counter(p1[k["key"]].get("urdf") for k in ks))}
            if blk == "F2":
                d["episode_ids"] = len({p1[k["key"]].get("episode_id") for k in ks})
                d["per_episode"] = dict(collections.Counter(collections.Counter(p1[k["key"]].get("episode_id") for k in ks).values()))
                d["distinct_ep_urdf"] = len({(p1[k["key"]].get("episode_id"), p1[k["key"]].get("urdf")) for k in ks})
                d["protect_found_n"] = dict(collections.Counter(len(p1[k["key"]].get("p_names", [])) for k in ks))
            if K.is_bridge(blk):
                d["episode_ids"] = len({p1[k["key"]].get("episode_id") for k in ks})
            out[which] = d
        fs = {p1[k["key"]]["state"] for k in keys if k["which"] == "formal"}
        out["calib_scene_in_formal"] = sum(p1[k["key"]]["state"] in fs for k in keys if k["which"] == "calib")
        fsi = {p1[k["key"]]["state_instr"] for k in keys if k["which"] == "formal"}
        out["calib_scene_plus_instruction_in_formal"] = sum(p1[k["key"]]["state_instr"] in fsi
                                                            for k in keys if k["which"] == "calib")
        out["G_px_min"] = min((r.get("G_px", 0) for r in p1.values()), default=None)
        out["P_px_min"] = min((r.get("P_px", 0) for r in p1.values()), default=None)
        res[blk] = out
        print(blk, json.dumps({kk: v for kk, v in out.items() if kk != "fail_msgs"}, ensure_ascii=False), flush=True)
        for kk, v in list(out["fail_msgs"].items())[:5]:
            print("  FAIL", kk, v, flush=True)
        res[blk + "_rows"] = list(p1.values())
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=0)
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
