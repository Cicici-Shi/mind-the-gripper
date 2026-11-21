"""bench2 门槛与照报项（协议 protocol_bench2.md v1.1 §4；b2x：补充协议 protocol_bench2_ext_fill_lag.md v1 §1、§2），只读非成败字段。

b2x 相对 b2 的改动：
  * 每块的格从 keys.EXT_CELLS 取（B2 四格、F1 八格），C5L 算作要隐藏、要画线的格；
  * 硬门槛 1 的冻结清单核对多两项：滞后步数按格（C5 用 L，C5L 用 cell_lag.C5L）、C1w / C1L 的填充颜色与清单一致；
    硬门槛不过的键从该块全部格（B2 四格、F1 八格）里剔除；
  * 照报项 5 对 C1w / C1L 同 C1（变化像素落在 G 外的个数应为 0）；照报项 6 对 C4、C5、C5L（> 5% 标轮廓操纵不足）；
  * 新照报项 12：每格每个滞后的几何量中位数（按实际显示的轮廓；未满滞后 pre / 已满滞后 full / 全部 all 三段）；
  * 新照报项 14：平涂格（C1、C1w、C1L）的操纵强度与环反差（对本格实际的填充色；F1 另按 urdf 版本分开）；
  * 新照报项 13：重跑的旧格（C0、C1、C3、C4、C5）与参照记录（--ref-root，默认第一阶段 b2/out_formal，只读）
    raw_actions、env_actions、frame0_sha256 逐位相同的键数；只照报，不剔键，不替换数据。

硬门槛（不过的键从该块五格一起剔除）：
  1 记录完整无报错；记录里实际生效的 episode_id / urdf 与键清单一致；记录的采集代码哈希（eval_views / views / keys）、
    L、线宽、G 连杆清单、保护名单与冻结清单（FROZEN_bench2_ext，json）一致；
  2 初始状态逐元素相同；3 噪声种子相同；
  4 整卡 A100-40、非 MIG，且 VK_ICD_FILENAMES 只指向 nvidia 的 ICD（软件渲染器 lavapipe 不可能被加载）。
  每格每项硬门槛不过的键数与占比照报（不另做判定）。
照报项（只报数，不剔键）：5 C1 / C3 掩膜外变化像素（逐位与阈值 30 两种计数）、C3 隐藏后 G 可见像素；
  6 C4 / C5 线在 224 输入里完全消失的帧占比（> 5% 标 OUTLINE_WEAK）；7 G 之外的机器人像素占比；
  8 A/A、9 本体状态探针（calib.py 在校准键上产出，原样并入）；
  10 出过错又重试成功的键数（_errors 下的 .txt.resolved；噪声种子固定，重跑是同一回合，算完成）；
  11 同一键五格 reset 后第 0 帧原图的 sha256 是否相同，按（五格所在节点的组合）分组报不一致的键数。

用法：python gate.py --root <采集根目录> --block B2 --keys keysets/formal_B2.json --out gate_B2.json
      [--manifest FROZEN_bench2_ext 或 FROZEN_bench2_ext.candidate] [--ref-root 参照目录或 none] [--aa aa.json]
  不给 --manifest 时用 $PROJ_ROOT/bench2_simpler/b2x/FROZEN_bench2_ext（存在的话）；两者都没有就不核对清单，并在输出里写明。
  记录位置 <root>/<块>/<格>/<键>.npz，出错回合在 <root>/<块>/<格>/_errors/<键>.txt。
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keys as K  # noqa: E402

CELLS = ("C0", "C1", "C3", "C4", "C5", "C1w", "C1L", "C5L")  # 全部可能的格；每块实际用的格见 K.EXT_CELLS
HIDE_CELLS = ("C3", "C4", "C5", "C5L")
OUTLINE_CELLS = ("C4", "C5", "C5L")
BLACKEN_LIKE_CELLS = ("C1", "C1w", "C1L")  # 区域都是 G，被改像素应全在 G 内
FILL_CELLS = ("C1w", "C1L")
FILL_KEY = {"C1w": "w", "C1L": "L"}
REF_ROOT = os.path.expanduser("$PROJ_ROOT/bench2_simpler/b2/out_formal")  # 第一阶段正式记录（只读）
REF_FIELDS = ("raw_actions", "env_actions", "frame0_sha256")
LINE_VANISH_MAX_FRAC = 0.05   # 照报项 6：轮廓线在 224 输入里完全消失的帧占比上限
OUTLINE_WEAK = "轮廓操纵不足"   # readout.py 按这个字符串读格标记（从这里 import，不另写一份）
GPU_MUST_CONTAIN = ("A100", "40GB")
GPU_MUST_NOT_CONTAIN = ("MIG",)
VK_ICD_NVIDIA = "/usr/share/vulkan/icd.d/nvidia_icd.json"
FROZEN = os.path.expanduser("$PROJ_ROOT/bench2_simpler/b2x/FROZEN_bench2_ext")
CODE_FILES_IN_RECORD = ("eval_views.py", "views.py", "keys.py")  # 记录里存了哈希的采集代码
# 门槛只许读这些字段；success / steps 不在里面
FIELDS = ("complete", "key", "cell", "block", "seeds", "qstats", "gpu_name", "nvidia_smi", "restore_maxdiff",
          "init_robot_qpos", "init_robot_pose", "init_actor_names", "init_actor_poses", "init_art_names",
          "init_art_qpos", "init_urdf_version", "g_links", "p_names", "lag_steps", "line_width", "n_queries",
          "actual_episode_id", "actual_urdf_version", "code_sha", "vk_icd_filenames", "frame0_sha256", "host",
          "fill_rgb", "lag_long", "init_urdf_version")
assert not any(f.startswith("success") for f in FIELDS) and "steps" not in FIELDS


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {f: z[f] for f in FIELDS if f in z.files}


def load_manifest(path):
    if path is None:
        path = FROZEN if os.path.exists(FROZEN) else None
    if path is None:
        return None, None
    with open(path) as f:
        return json.load(f), path


def item5(cell, qs, rec):
    """照报项 5：C1 被改像素全在 G 内；C3（及 C4 / C5 的隐藏部分）隐藏后变化像素全在 G 内、隐藏后 G 可见像素为 0。
    返回不满足的描述（只报，不剔键）。"""
    bad = []
    if cell in BLACKEN_LIKE_CELLS:
        n = sum(q["changed_out_E_px"] for q in qs)
        if n:
            bad.append(f"{cell} 掩膜外被改像素 {n}")
    if cell in HIDE_CELLS:
        n = sum(q["hide_changed_out_G_px"] for q in qs)
        v = sum(q["G_vis_after_hide_px"] for q in qs)
        if n:
            bad.append(f"隐藏后掩膜外变化像素 {n}")
        if v:
            bad.append(f"隐藏后夹爪仍可见像素 {v}")
        if int(rec["restore_maxdiff"]) != 0:
            bad.append(f"恢复可见后与原图差 {int(rec['restore_maxdiff'])}")
    if cell in OUTLINE_CELLS:
        n = sum(q["changed_out_E_px"] for q in qs)
        if n:
            bad.append(f"{cell} 预期区域（G 并画线区）外被改像素 {n}")
    return bad


def gpu_fail(rec):
    """硬门槛 4：返回不满足的描述（空 = 通过）。"""
    g, s = str(rec.get("gpu_name", "")), str(rec.get("nvidia_smi", ""))
    bad = []
    if not all(x in g for x in GPU_MUST_CONTAIN):
        bad.append(f"显卡 {g}")
    if any(x in s or x in g for x in GPU_MUST_NOT_CONTAIN):
        bad.append("MIG")
    icd = str(rec["vk_icd_filenames"]) if "vk_icd_filenames" in rec else None
    if icd != VK_ICD_NVIDIA:
        bad.append(f"VK_ICD_FILENAMES={icd!r}（应只指向 {VK_ICD_NVIDIA}）")
    return bad


def spec_mismatch(r, kd):
    """记录里实际生效的 episode_id / urdf 与键清单的 reset 规格比对；返回不符的描述。"""
    if kd is None:
        return []
    want, bad = kd.get("reset", {}), []
    if "episode_id" in want:
        got = int(r["actual_episode_id"]) if "actual_episode_id" in r else None
        if got != int(want["episode_id"]):
            bad.append(f"episode_id {got} != {want['episode_id']}")
    if "urdf_version" in want:
        got = str(r["actual_urdf_version"]) if "actual_urdf_version" in r else None
        if got != want["urdf_version"]:
            bad.append(f"urdf {got!r} != {want['urdf_version']!r}")
    return bad


def manifest_mismatch(r, man, block, cell=None):
    """记录与冻结清单比对：采集代码哈希、L、线宽、G 连杆清单、保护名单。返回不符的描述。"""
    if man is None:
        return []
    bad = []
    cs = json.loads(str(r["code_sha"])) if "code_sha" in r else {}
    for f in CODE_FILES_IN_RECORD:
        if cs.get(f) != man["code_sha256"][f]:
            bad.append(f"代码 {f} 哈希不符")
    mb = man["blocks"][block]
    if "lag_steps" not in r or int(r["lag_steps"]) != int(mb["L"]):
        bad.append(f"L {int(r['lag_steps']) if 'lag_steps' in r else None} != {mb['L']}")
    if "line_width" not in r or int(r["line_width"]) != int(mb["line_width"]):
        bad.append(f"线宽 {int(r['line_width']) if 'line_width' in r else None} != {mb['line_width']}")
    if "g_links" not in r or json.loads(str(r["g_links"])) != mb["g_links"]:
        bad.append("G 连杆清单不符")
    if "p_names" not in r or not set(json.loads(str(r["p_names"]))) <= set(mb["protect"]):
        bad.append("保护名单不符")
    if cell == "C5L":
        want_l = mb.get("cell_lag", {}).get("C5L")
        if "lag_long" not in r or want_l is None or int(r["lag_long"]) != int(want_l):
            bad.append(f"C5L 滞后 {int(r['lag_long']) if 'lag_long' in r else None} != {want_l}")
    if cell == "C5" and "cell_lag" in mb and int(r["lag_steps"]) != int(mb["cell_lag"]["C5"]):
        bad.append(f"C5 滞后 {int(r['lag_steps'])} != {mb['cell_lag']['C5']}")
    if cell in FILL_CELLS:
        want = mb["fill_rgb"][FILL_KEY[cell]]
        got = [int(x) for x in r["fill_rgb"]] if "fill_rgb" in r else None
        if got != [int(x) for x in want]:
            bad.append(f"{cell} 填充色 {got} != {want}")
    return bad


def geometry_report(cell_recs):
    """照报项 12：每个滞后 l、每段（pre 未满滞后 / full 已满滞后 / all 全部）的 IoU、质心位移（224 像素）、
    可见像素比的中位数；按策略实际看到的那条轮廓算（来自 --geom-lags 记录）。"""
    acc = {}
    for r in cell_recs.values():
        for q in json.loads(str(r["qstats"])):
            for l, d in q.get("geom", {}).items():
                for ph in (d.get("phase", "full"), "all"):
                    g = acc.setdefault(l, {}).setdefault(ph, {"iou": [], "centroid_shift_224": [], "visible_ratio_224": []})
                    for f in g:
                        if d.get(f) is not None:
                            g[f].append(d[f])
    return {l: {ph: {**{f + "_median": (float(np.median(v)) if v else None) for f, v in g.items()}, "frames": len(g["iou"])}
                for ph, g in sorted(phs.items())}
            for l, phs in sorted(acc.items(), key=lambda t: int(t[0]))}


def fill_report(cell_recs):
    """照报项 14：操纵强度 = G 内逐像素 mean_ch |原值 - 填充值| 的像素平均；环反差 = G 外 6 像素一圈的同一量。
    全部与按 urdf 版本分开（来自 --fill-stats 记录）。"""
    acc = {}
    for r in cell_recs.values():
        u = str(r.get("init_urdf_version", ""))
        for q in json.loads(str(r["qstats"])):
            if "fill_strength_sum" not in q:
                continue
            for key in ("all", "urdf:" + (u or "(默认)")):
                a = acc.setdefault(key, [0.0, 0, 0.0, 0])
                a[0] += q["fill_strength_sum"]
                a[1] += q["G_px"]
                a[2] += q["ring_contrast_sum"]
                a[3] += q["ring_px"]
    return {k: {"fill_strength": (v[0] / v[1]) if v[1] else None, "ring_contrast": (v[2] / v[3]) if v[3] else None,
                "G_px": v[1], "ring_px": v[3]} for k, v in sorted(acc.items())}


def rerun_identity(ref_root, block, cell, keys, cell_recs):
    """照报项 13：重跑的旧格与参照记录逐位相同的键数（raw_actions、env_actions、frame0_sha256）。只读这三个字段。"""
    out = {"compared": 0, "identical": 0, "differ": [], "ref_missing": 0}
    for k in keys:
        if k not in cell_recs:
            continue
        rp = os.path.join(ref_root, block, cell, k + ".npz")
        if not os.path.exists(rp):
            out["ref_missing"] += 1
            continue
        with np.load(rp, allow_pickle=False) as z:
            ref = {f: z[f] for f in REF_FIELDS if f in z.files}
        with np.load(cell_recs[k]["_path"], allow_pickle=False) as z:
            new = {f: z[f] for f in REF_FIELDS if f in z.files}
        out["compared"] += 1
        same = all(f in ref and f in new and ref[f].shape == new[f].shape and np.array_equal(ref[f], new[f])
                   for f in REF_FIELDS)
        if same:
            out["identical"] += 1
        else:
            out["differ"].append(k)
    return out


def run(root, block, keys, aa=None, manifest=None, ref_root=REF_ROOT):
    """keys：键清单（dict 列表；只给键名时不核对 reset 规格）。manifest：冻结清单 dict 或 None（不核对）。"""
    kdict = {k["key"]: k for k in keys if isinstance(k, dict)}
    keys = [k["key"] if isinstance(k, dict) else k for k in keys]
    CELLS = tuple(K.EXT_CELLS[block])  # 本块用的格（局部名覆盖全局）
    res = {"block": block, "n_keys": len(keys), "items": {}, "cell_flags": {c: [] for c in CELLS},
           "manifest_checked": manifest is not None}
    recs = {c: {} for c in CELLS}
    fail = {k: [] for k in keys}
    # 1. 每格每键恰好一条完整记录，无报错回合；实际 reset 与键清单一致；与冻结清单一致
    it1 = {"missing": {}, "errors": {}, "extra": {}, "spec_mismatch": {}, "manifest_mismatch": {}}
    retried = {}
    for c in CELLS:
        d = os.path.join(root, block, c)
        present = {f[:-4] for f in os.listdir(d) if f.endswith(".npz") and not f.startswith(".")} if os.path.isdir(d) else set()
        ed = os.path.join(d, "_errors")
        efiles = os.listdir(ed) if os.path.isdir(ed) else []
        errs = {f[:-4] for f in efiles if f.endswith(".txt")}
        resolved = {f[:-len(".txt.resolved")] for f in efiles if f.endswith(".txt.resolved")}
        it1["missing"][c] = sorted(set(keys) - present)
        it1["errors"][c] = sorted(errs & set(keys))
        it1["extra"][c] = sorted(present - set(keys))
        for k in keys:
            if k in present:
                r = load(os.path.join(d, k + ".npz"))
                if not bool(r.get("complete", False)) or str(r["key"]) != k or str(r["cell"]) != c:
                    fail[k].append(f"1:{c} 记录不完整或键/格不符")
                    continue
                sm = spec_mismatch(r, kdict.get(k))
                if sm:
                    fail[k].append(f"1:{c} 实际生效的 reset 与键清单不符：{'; '.join(sm)}")
                    it1["spec_mismatch"].setdefault(c, []).append(k)
                mm = manifest_mismatch(r, manifest, block, c)
                if mm:
                    fail[k].append(f"1:{c} 与冻结清单不符：{'; '.join(mm)}")
                    it1["manifest_mismatch"].setdefault(c, []).append(k)
                r["_path"] = os.path.join(d, k + ".npz")
                recs[c][k] = r
            else:
                fail[k].append(f"1:{c} 缺记录")
            if k in errs:
                fail[k].append(f"1:{c} 有报错回合")
        retried[c] = sorted(k for k in resolved & set(recs[c]) if k not in errs)
    res["items"]["1_records"] = it1
    res["items"]["10_retried_after_error"] = {c: {"n": len(v), "keys": v} for c, v in retried.items()}
    # 2. 同一键五格初始状态逐元素相同；3. 每次查询的噪声种子相同；11. 第 0 帧原图哈希（照报，按节点组合分组）
    n2 = n3 = 0
    f0 = collections.defaultdict(lambda: {"keys": 0, "frame0_inconsistent": 0, "frame0_missing": 0})
    for k in keys:
        have = [c for c in CELLS if k in recs[c]]
        if len(have) >= 2:
            r0 = recs[have[0]][k]
            for c in have[1:]:
                r = recs[c][k]
                same = all(np.array_equal(r0[f], r[f]) for f in ("init_robot_qpos", "init_robot_pose", "init_actor_poses"))
                same = same and all(str(r0[f]) == str(r[f]) for f in ("init_actor_names", "init_art_names",
                                                                        "init_art_qpos", "init_urdf_version"))
                if not same:
                    fail[k].append(f"2:{c} 初始状态与 {have[0]} 不同")
                    n2 += 1
                if not np.array_equal(r0["seeds"], r["seeds"]):
                    fail[k].append(f"3:{c} 噪声种子与 {have[0]} 不同")
                    n3 += 1
        if have:
            combo = "+".join(sorted({str(recs[c][k].get("host", "?")) for c in have}))
            hs = [str(recs[c][k]["frame0_sha256"]) if "frame0_sha256" in recs[c][k] else None for c in have]
            g = f0[combo]
            g["keys"] += 1
            if None in hs:
                g["frame0_missing"] += 1
            elif len(set(hs)) > 1:
                g["frame0_inconsistent"] += 1
    res["items"]["2_init_state_mismatch"] = n2
    res["items"]["3_seed_mismatch"] = n3
    res["items"]["11_frame0_by_host_combo"] = dict(f0)
    # 4（硬）渲染设备；5、6、7（照报）
    it5, it6, it7, gpu_bad = {}, {}, {}, {}
    for c in CELLS:
        n_bad5, frames, vanish, sumstats, arm_frac = 0, 0, 0, {}, []
        for k, r in recs[c].items():
            qs = json.loads(str(r["qstats"]))
            if item5(c, qs, r):
                n_bad5 += 1
            gb = gpu_fail(r)
            if gb:
                fail[k].append(f"4:{c} {'; '.join(gb)}")
                gpu_bad[c] = gpu_bad.get(c, 0) + 1
            for q in qs:
                for f in ("G_px", "changed_px", "changed_out_E_px", "hide_changed_out_G_px", "hide_changed_out_G_t30_px",
                          "G_vis_after_hide_px", "O_px", "O_drawn_px", "O_protected_px", "line_px_224", "arm_px"):
                    if f in q:
                        sumstats.setdefault(f, []).append(q[f])
                if q["R_px"] > 0:
                    arm_frac.append(q["arm_px"] / q["R_px"])
                if c in OUTLINE_CELLS and q.get("Gl_px", 0) > 0:
                    frames += 1
                    vanish += int(q["line_px_224"] == 0)
        it5[c] = {"keys_with_any_violation": n_bad5,
                  "per_frame_sum": {f: int(np.sum(v)) for f, v in sumstats.items()},
                  "per_frame_max": {f: int(np.max(v)) for f, v in sumstats.items()}}
        it7[c] = None if not arm_frac else {"median": float(np.median(arm_frac)), "p90": float(np.percentile(arm_frac, 90)),
                                            "max": float(np.max(arm_frac)), "frames": len(arm_frac)}
        if c in OUTLINE_CELLS:
            frac = None if frames == 0 else vanish / frames
            prot = [a / b for a, b in zip(sumstats.get("O_protected_px", []), sumstats.get("O_px", [])) if b > 0]
            it6[c] = {"frames_with_outline_mask": frames, "frames_line_vanished": vanish, "frac": frac,
                      "protected_frac_median": float(np.median(prot)) if prot else None}
            if frac is None or frac > LINE_VANISH_MAX_FRAC:
                res["cell_flags"][c].append(OUTLINE_WEAK)
    res["items"]["4_gpu_fail_records"] = gpu_bad
    res["items"]["5_pixels_report"] = it5
    res["items"]["6_line_visibility_report"] = it6
    res["items"]["7_arm_outside_G_frac_report"] = it7
    res["items"]["12_lag_geometry"] = {c: geometry_report(recs[c]) for c in CELLS}
    res["items"]["14_fill_strength_ring_contrast"] = {c: fill_report(recs[c]) for c in CELLS if c in BLACKEN_LIKE_CELLS}
    res["items"]["13_rerun_identical_vs_ref"] = {"ref_root": ref_root, **{
        c: rerun_identity(ref_root, block, c, keys, recs[c]) for c in CELLS if c in K.EXT_OLD_CELLS}} \
        if ref_root else "未比对（没有给参照目录）"
    res["items"]["8_9_aa_and_proprio_probe"] = json.load(open(aa)) if aa else "未提供（calib.py 在校准键上产出）"
    # 剔除
    excluded = sorted(k for k, v in fail.items() if v)
    res["excluded_keys"] = excluded
    res["n_excluded"] = len(excluded)
    res["fail_reasons"] = {k: v for k, v in fail.items() if v}
    res["hard_fail_by_cell"] = {}  # 每格每项硬门槛不过的键数与占比，只照报
    for c in CELLS:
        res["hard_fail_by_cell"][c] = {}
        for item in ("1", "2", "3", "4"):
            n = sum(1 for v in fail.values() if any(x.startswith(f"{item}:{c}") for x in v))
            res["hard_fail_by_cell"][c][item] = {"n": n, "frac": (n / len(keys)) if keys else None}
    res["n_retained"] = len(keys) - len(excluded)
    return res


def summary(res):
    L = [f"门槛 {res['block']}：键 {res['n_keys']}，剔除 {res['n_excluded']}，保留 {res['n_retained']}"
         f"（冻结清单{'已' if res['manifest_checked'] else '未'}核对）"]
    it1 = res["items"]["1_records"]
    L.append("1 记录：缺 " + ", ".join(f"{c}={len(v)}" for c, v in it1["missing"].items())
             + "；报错 " + ", ".join(f"{c}={len(v)}" for c, v in it1["errors"].items()))
    for name, lab in (("spec_mismatch", "实际 reset 与键清单不符"), ("manifest_mismatch", "与冻结清单不符")):
        L.append(f"1 {lab}：" + (", ".join(f"{c}={len(v)}" for c, v in it1[name].items()) if it1[name] else "0"))
    L.append("硬门槛不过的键数（格：项 1/2/3/4）：" + "；".join(
        f"{c} " + "/".join(str(d[i]["n"]) for i in ("1", "2", "3", "4")) for c, d in res["hard_fail_by_cell"].items()))
    L.append(f"2 初始状态不同：{res['items']['2_init_state_mismatch']}；3 种子不同：{res['items']['3_seed_mismatch']}；"
             f"4 GPU / ICD 不合格记录：{res['items']['4_gpu_fail_records'] or 0}")
    for c, d in res["items"]["5_pixels_report"].items():
        m = d["per_frame_max"]
        L.append(f"5 {c}：有违例的键 {d['keys_with_any_violation']}；逐帧最大 " + ", ".join(f"{k}={v}" for k, v in m.items()))
    for c, d in res["items"]["6_line_visibility_report"].items():
        L.append(f"6 {c}：有轮廓掩膜的帧 {d['frames_with_outline_mask']}，线完全消失 {d['frames_line_vanished']}，"
                 f"占比 {d['frac']}，被保护截掉的线占比中位数 {d['protected_frac_median']}")
    L.append("7 G 之外机器人像素占比：" + ", ".join(f"{c}={v['median'] if v else None:.4g}" if v else f"{c}=None"
                                             for c, v in res["items"]["7_arm_outside_G_frac_report"].items()))
    L.append("10 出过错又重试成功的键数：" + ", ".join(f"{c}={v['n']}" for c, v in res["items"]["10_retried_after_error"].items()))
    L.append("11 第 0 帧哈希（节点组合：键数 / 不一致 / 缺字段）：" + "；".join(
        f"{h}: {g['keys']}/{g['frame0_inconsistent']}/{g['frame0_missing']}"
        for h, g in res["items"]["11_frame0_by_host_combo"].items()))
    g12 = res["items"].get("12_lag_geometry", {})
    for c, d in g12.items():
        for l, phs in d.items():
            for ph, v in phs.items():
                L.append(f"12 {c} 滞后 {l} 步 [{ph}]：IoU 中位数 {v['iou_median']}，质心位移中位数（224 像素）"
                         f"{v['centroid_shift_224_median']}，可见像素比中位数 {v['visible_ratio_224_median']}（{v['frames']} 帧）")
    for c, d in res["items"].get("14_fill_strength_ring_contrast", {}).items():
        for k, v in d.items():
            L.append(f"14 {c} {k}：操纵强度 {v['fill_strength']}，环反差 {v['ring_contrast']}")
    r13 = res["items"].get("13_rerun_identical_vs_ref")
    if isinstance(r13, dict):
        L.append(f"13 重跑旧格与参照（{r13['ref_root']}）逐位相同的键数：" + "；".join(
            f"{c} {v['identical']}/{v['compared']}（参照缺 {v['ref_missing']}）" for c, v in r13.items() if c != "ref_root"))
        for c, v in r13.items():
            if c != "ref_root" and v["differ"]:
                L.append(f"   {c} 不相同的键：{v['differ']}")
    else:
        L.append(f"13 {r13}")
    L.append(f"格标记：{ {c: v for c, v in res['cell_flags'].items() if v} or '无'}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--manifest", default=None, help="冻结清单 json；默认 FROZEN_bench2_ext（存在的话）")
    ap.add_argument("--aa", default=None)
    ap.add_argument("--ref-root", default=REF_ROOT, help="照报项 13 的参照记录目录（默认第一阶段正式记录，只读）；none 不比")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man, mp = load_manifest(a.manifest)
    res = run(a.root, a.block, json.load(open(a.keys)), a.aa, man, None if a.ref_root == "none" else a.ref_root)
    res["manifest_path"] = mp
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    print(summary(res))


if __name__ == "__main__":
    sys.exit(main())
