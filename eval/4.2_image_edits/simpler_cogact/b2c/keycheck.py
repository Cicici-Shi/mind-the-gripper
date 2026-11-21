"""键核对（补充协议 §3 第 2 条；不跑模型、不读任何成败字段）。

在第二阶段这套仿真（官方 SimplerEnv + CogACT 的步数上限）里，对四块全部正式键与校准键各 reset 一次，和第一阶段记录逐元素比对：
  * 第一阶段的 npz 记录（只取 init_* 与 frame0_sha256 这几个字段，np.load 按名取，不碰成败字段）：
      正式键：$PROJ_ROOT/bench2_simpler/b2/out_formal/<块>/C0/<键>.npz
      校准键：out_calib / out_smoke_v11 / out_calib_aa2 下的 <块>/C0/<键>.npz（有哪个用哪个，取第一个找到的）
    比 init_robot_qpos、init_robot_pose、init_actor_poses（逐元素，np.array_equal），init_actor_names、init_art_names、
    init_art_qpos、init_urdf_version（字符串），frame0_sha256（第 0 帧原图）。不同时记最大绝对差与哪些 actor 不同。
  * 第一阶段键核查 keyprobe_v11_all4.json 里每个键的状态签名（probe_keys.state_bytes 的算法，逐字复制在下面），
    覆盖全部正式键与校准键（含没有 npz 记录的校准键）。
另外核对：实际生效的 episode_id / urdf 与键一致；G 连杆与保护名单能解析；同一 reset 做两遍状态与第 0 帧逐位相同（抽每块前 5 键）。
用法：python keycheck.py --out keycheck.json [--blocks B1 B2 F1 F2]
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import traceback

B2C = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2C)
import numpy as np  # noqa: E402

import eval_views as E  # noqa: E402
import keys as K  # noqa: E402
import views as V  # noqa: E402

P1 = os.path.expanduser("$PROJ_ROOT/bench2_simpler/b2")
P1_CALIB_DIRS = ("out_calib", "out_smoke_v11", "out_calib_aa2")
ARR = ("init_robot_qpos", "init_robot_pose", "init_actor_poses")
STR = ("init_actor_names", "init_art_names", "init_art_qpos", "init_urdf_version")


def state_bytes(s):
    """与第一阶段 probe_keys.state_bytes 相同的签名。"""
    h = hashlib.sha256()
    for f in ARR:
        h.update(np.ascontiguousarray(s[f]).tobytes())
    for f in STR:
        h.update(str(s[f]).encode())
    return h.hexdigest()


def p1_record(block, key, formal):
    dirs = ("out_formal",) if formal else P1_CALIB_DIRS
    for d in dirs:
        p = os.path.join(P1, d, block, "C0", key + ".npz")
        if os.path.exists(p):
            with np.load(p, allow_pickle=False) as z:
                r = {f: z[f] for f in ARR + STR if f in z.files}
                r["frame0_sha256"] = str(z["frame0_sha256"]) if "frame0_sha256" in z.files else None
            return d, r
    return None, None


def compare(mine, ref):
    """逐元素比对；返回 (相同?, 不同之处描述 dict)。"""
    diff = {}
    for f in ARR:
        a, b = np.asarray(mine[f]), np.asarray(ref[f])
        if a.shape != b.shape:
            diff[f] = f"形状 {a.shape} vs {b.shape}"
        elif not np.array_equal(a, b):
            d = np.abs(a - b)
            diff[f] = {"max_abs": float(d.max()), "n_elem_diff": int((a != b).sum())}
            if f == "init_actor_poses":
                names = json.loads(str(mine["init_actor_names"]))
                rows = np.where((a != b).any(axis=1))[0]
                diff[f]["actors"] = [names[i] if i < len(names) else i for i in rows]
    for f in STR:
        if str(mine[f]) != str(ref[f]):
            diff[f] = {"mine": str(mine[f])[:300], "phase1": str(ref[f])[:300]}
    return not diff, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", nargs="+", default=list(K.BLOCKS))
    a = ap.parse_args()
    E.check_env_src()
    kp = json.load(open(os.path.join(P1, "keyprobe_v11_all4.json")))
    res = {}
    for blk in a.blocks:
        b = K.BLOCKS[blk]
        formal = json.load(open(os.path.join(B2C, "keysets", f"formal_{blk}.json")))
        calib = K.make_keys(blk, "calib")
        p1sig = {r["key"]: r.get("state") for r in kp[blk + "_rows"]}
        envs = {}
        rows = []
        for which, ks in (("formal", formal), ("calib", calib)):
            for i, k in enumerate(ks):
                r = {"key": k["key"], "which": which, "ok": True, "err": None}
                try:
                    if k["task"] not in envs:
                        envs[k["task"]] = E.make_env(k["task"], b)
                    env = envs[k["task"]]
                    obs, info = E.reset_key(env, k)
                    try:
                        E.check_reset_spec(env, info, k)
                    except AssertionError as e:
                        r.update(ok=False, err=f"reset 规格不符：{e}")
                    vm = V.ViewMaker(env, "C0", b["body"], b["L"], 8, K.QUERY_EVERY, K.PROTECT[blk], K.PROTECT_EXPECT_N[blk])
                    try:
                        vm.on_reset()
                    except AssertionError as e:
                        r.update(ok=False, err=(r["err"] or "") + f" | G/保护解析失败：{e}")
                    cam = V.cam_name(env.unwrapped.robot_uid)
                    s = E.scene_state(env)
                    f0 = hashlib.sha256(np.ascontiguousarray(obs["image"][cam]["rgb"]).tobytes()).hexdigest()
                    sig = state_bytes(s)
                    r.update(sig=sig, frame0=f0, sig_equal_p1_keyprobe=(p1sig.get(k["key"]) == sig) if k["key"] in p1sig else None)
                    if i < 5:  # 同一 reset 做两遍是否逐位相同
                        obs2, _ = E.reset_key(env, k)
                        s2 = E.scene_state(env)
                        f02 = hashlib.sha256(np.ascontiguousarray(obs2["image"][cam]["rgb"]).tobytes()).hexdigest()
                        r["reset_twice_equal"] = (state_bytes(s2) == sig) and (f02 == f0)
                    src, ref = p1_record(blk, k["key"], which == "formal")
                    r["p1_src"] = src
                    if ref is not None:
                        same, d = compare(s, ref)
                        r["p1_state_equal"] = same
                        if not same:
                            r["p1_state_diff"] = d
                        r["p1_frame0_equal"] = None if ref["frame0_sha256"] is None else (ref["frame0_sha256"] == f0)
                except Exception:
                    r.update(ok=False, err=traceback.format_exc()[-1500:])
                rows.append(r)
        for env in envs.values():
            env.close()
        out = {}
        for which in ("formal", "calib"):
            rs = [r for r in rows if r["which"] == which]
            cnt = lambda f, v: sum(1 for r in rs if r.get(f) is v)  # noqa: E731
            out[which] = {
                "n": len(rs), "fail": [r["key"] for r in rs if not r["ok"]],
                "keyprobe_sig": {"equal": cnt("sig_equal_p1_keyprobe", True), "differ": cnt("sig_equal_p1_keyprobe", False),
                                 "no_ref": sum(1 for r in rs if r.get("sig_equal_p1_keyprobe") is None)},
                "npz_state": {"equal": cnt("p1_state_equal", True), "differ": cnt("p1_state_equal", False),
                              "no_ref": sum(1 for r in rs if "p1_state_equal" not in r)},
                "npz_frame0": {"equal": cnt("p1_frame0_equal", True), "differ": cnt("p1_frame0_equal", False),
                               "no_ref": sum(1 for r in rs if r.get("p1_frame0_equal") is None)},
                "reset_twice_equal": {"checked": sum(1 for r in rs if "reset_twice_equal" in r),
                                      "equal": cnt("reset_twice_equal", True)},
                "distinct_sigs": len({r.get("sig") for r in rs}),
            }
            fields = collections.Counter(f for r in rs for f in r.get("p1_state_diff", {}))
            out[which]["npz_state_diff_fields"] = dict(fields)
            mx = collections.defaultdict(float)
            actors = collections.Counter()
            for r in rs:
                for f, d in r.get("p1_state_diff", {}).items():
                    if isinstance(d, dict) and "max_abs" in d:
                        mx[f] = max(mx[f], d["max_abs"])
                        for an in d.get("actors", []):
                            actors[str(an)] += 1
            out[which]["npz_state_diff_max_abs"] = dict(mx)
            out[which]["npz_state_diff_actors"] = dict(actors)
            out[which]["diff_examples"] = [{"key": r["key"], "diff": r["p1_state_diff"]} for r in rs if "p1_state_diff" in r][:3]
        res[blk] = out
        res[blk + "_rows"] = rows
        print(blk, json.dumps(out, ensure_ascii=False), flush=True)
        for r in [r for r in rows if not r["ok"]][:3]:
            print("  FAIL", r["key"], r["err"], flush=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=0)
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
