"""bench2 第二阶段（CogACT）门槛与照报项。由第一阶段 b2/gate.py 复制改写（第一阶段协议 v1.3 §4，补充协议 §2、§3）。
与第一阶段的差别：
  * 冻结清单换成 $PROJ_ROOT/bench2_cogact/b2c/FROZEN_bench2_cogact；
  * 硬门槛 1 多核四类字段与冻结清单一致：上游文件哈希（CogACT 推理代码、prismatic 骨干、官方 SimplerEnv 入口、分词器文件）、
    策略实际配置（policy_cfg：反归一化键、cfg、DDIM 步数、集成窗口、精度、检查点路径与大小等）、步数上限、控制频率、
    查询间隔（= 1）；第一阶段审查记的局限 L1（权重、上游仓库不进清单）在这里收紧了一部分；
  * 硬门槛 4 另要求 VK_DRIVER_FILES 为空（作业里已 unset，记录里存了值）；卡型从「必须是 A100-40」改为（协调方 2026-11-13）：
    显卡型号在冻结清单 gpu_models 之内、MIG 关闭，且等于冻结清单 gpu_assignment 分给这个键的型号；同一键五格的型号必须相同
    （策略推理在不同型号上不逐位相同，配对比较要求五格同型号）。不过的键五格一起剔除，记在硬门槛 4；
  * 照报项 8 只剩 A/A（本体状态探针对 CogACT 不适用，补充协议 §2 删去第 10 项）；
  * 照报项 6 的「线在策略输入里」= CogACT 两个编码器的 224 输入（views.to_model_all），任一路看得见就算看得见。
第一阶段的原说明：

bench2 门槛与照报项（协议 protocol_bench2.md v1.1 §4），读数前运行，只读非成败字段。

硬门槛（不过的键从该块五格一起剔除）：
  1 记录完整无报错；记录里实际生效的 episode_id / urdf 与键清单一致；记录的采集代码哈希（eval_views / views / keys）、
    L、线宽、G 连杆清单、保护名单与冻结清单（FROZEN_bench2，json）一致；
  2 初始状态逐元素相同；3 噪声种子相同；
  4 整卡 A100-40、非 MIG，且 VK_ICD_FILENAMES 只指向 nvidia 的 ICD（软件渲染器 lavapipe 不可能被加载）。
  每格每项硬门槛不过的键数与占比照报（不另做判定）。
照报项（只报数，不剔键）：5 C1 / C3 掩膜外变化像素（逐位与阈值 30 两种计数）、C3 隐藏后 G 可见像素；
  6 C4 / C5 线在 224 输入里完全消失的帧占比（> 5% 标 OUTLINE_WEAK）；7 G 之外的机器人像素占比；
  8 A/A（calib.py 在校准键上产出，原样并入；第二阶段没有本体状态探针）；
  10 出过错又重试成功的键数（_errors 下的 .txt.resolved；噪声种子固定，重跑是同一回合，算完成）；
  11 同一键五格 reset 后第 0 帧原图的 sha256 是否相同，按（五格所在节点的组合）分组报不一致的键数。

用法：python gate.py --root <采集根目录> --block B1 --keys keys_B1.json --out gate_B1.json
      [--manifest FROZEN_bench2_cogact 或 FROZEN_bench2_cogact.candidate] [--aa aa_B1.json]
  不给 --manifest 时用 $PROJ_ROOT/bench2_cogact/b2c/FROZEN_bench2_cogact（存在的话）；两者都没有就不核对清单，并在输出里写明。
  记录位置 <root>/<块>/<格>/<键>.npz，出错回合在 <root>/<块>/<格>/_errors/<键>.txt。
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

CELLS = ("C0", "C1", "C3", "C4", "C5")
HIDE_CELLS = ("C3", "C4", "C5")
OUTLINE_CELLS = ("C4", "C5")
LINE_VANISH_MAX_FRAC = 0.05   # 照报项 6：轮廓线在 224 输入里完全消失的帧占比上限
OUTLINE_WEAK = "轮廓操纵不足"   # readout.py 按这个字符串读格标记（从这里 import，不另写一份）
GPU_MODELS_DEFAULT = ("NVIDIA A100-PCIE-40GB", "NVIDIA H100 NVL")
GPU_MUST_NOT_CONTAIN = ("MIG",)
VK_ICD_NVIDIA = "/usr/share/vulkan/icd.d/nvidia_icd.json"
FROZEN = os.path.expanduser("$PROJ_ROOT/bench2_cogact/b2c/FROZEN_bench2_cogact")
CODE_FILES_IN_RECORD = ("eval_views.py", "views.py", "keys.py")  # 记录里存了哈希的采集代码
# 门槛只许读这些字段；success / steps 不在里面
FIELDS = ("complete", "key", "cell", "block", "seeds", "qstats", "gpu_name", "nvidia_smi", "restore_maxdiff",
          "init_robot_qpos", "init_robot_pose", "init_actor_names", "init_actor_poses", "init_art_names",
          "init_art_qpos", "init_urdf_version", "g_links", "p_names", "lag_steps", "line_width", "n_queries",
          "actual_episode_id", "actual_urdf_version", "code_sha", "vk_icd_filenames", "frame0_sha256", "host",
          "upstream_sha", "policy_cfg", "max_episode_steps", "control_freq", "query_every", "vk_driver_files")
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
    if cell == "C1":
        n = sum(q["changed_out_E_px"] for q in qs)
        if n:
            bad.append(f"C1 掩膜外被改像素 {n}")
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


def gpu_fail(rec, man=None, block=None, key=None, require_assignment=False):
    """硬门槛 4：返回不满足的描述（空 = 通过）。有冻结清单时型号须在 gpu_models 之内且等于 gpu_assignment 分给该键的型号；
    没有清单时（冻结前看冒烟）退回默认允许的两种型号。require_assignment：正式键必须在分组表里，不在就算不过
    （实现审查 2026-11-13 可选项：不再静默放过）；校准键与冒烟键不在表里，调用方传 False。"""
    g, s = str(rec.get("gpu_name", "")), str(rec.get("nvidia_smi", ""))
    bad = []
    allowed = man["gpu_models"] if man is not None else GPU_MODELS_DEFAULT
    if g not in allowed:
        bad.append(f"显卡 {g} 不在允许的型号 {allowed} 之内")
    if man is not None:
        # 分组表覆盖全部正式键（freeze_manifest 断言）；校准键与冒烟键不在表里，只核型号在允许之列
        # （读数脚本先核对键清单哈希等于冻结的正式键清单，所以真实读数时每个键都在表里）
        amap = man["gpu_assignment"].get(block, {})
        if key in amap:
            if g != amap[key]:
                bad.append(f"显卡 {g} 不是冻结清单分给该键的型号 {amap[key]}")
        elif require_assignment:
            bad.append("正式键不在冻结清单 gpu_assignment 里")
    if any(x in s or x in g for x in GPU_MUST_NOT_CONTAIN):
        bad.append("MIG")
    icd = str(rec["vk_icd_filenames"]) if "vk_icd_filenames" in rec else None
    if icd != VK_ICD_NVIDIA:
        bad.append(f"VK_ICD_FILENAMES={icd!r}（应只指向 {VK_ICD_NVIDIA}）")
    drv = str(rec["vk_driver_files"]) if "vk_driver_files" in rec else None
    if drv != "":
        bad.append(f"VK_DRIVER_FILES={drv!r}（应为空：它优先于 VK_ICD_FILENAMES）")
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


def manifest_mismatch(r, man, block):
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
    # 第二阶段新增：上游文件哈希、策略实际配置、步数上限、控制频率、查询间隔
    up = json.loads(str(r["upstream_sha"])) if "upstream_sha" in r else None
    if up != man["upstream_sha256"]:
        bad.append("上游文件哈希不符" if up is not None else "缺 upstream_sha")
    pc = json.loads(str(r["policy_cfg"])) if "policy_cfg" in r else None
    if pc != man["policy_cfg"][mb["policy_setup"]]:
        bad.append("策略配置不符" if pc is not None else "缺 policy_cfg")
    for f, cast in (("max_episode_steps", int), ("control_freq", float), ("query_every", int)):
        if f not in r or cast(r[f]) != cast(mb[f]):
            bad.append(f"{f} {cast(r[f]) if f in r else None} != {mb[f]}")
    return bad


def run(root, block, keys, aa=None, manifest=None, require_assignment=None):
    """keys：键清单（dict 列表；只给键名时不核对 reset 规格）。manifest：冻结清单 dict 或 None（不核对）。
    require_assignment：每个键都必须在清单 gpu_assignment.<块> 里（正式键）。None = 自动：有清单且键清单里有任何一个键在表里
    就当作正式键清单，要求全部在表里；readout.py 已核对键清单哈希，显式传 True。"""
    kdict = {k["key"]: k for k in keys if isinstance(k, dict)}
    keys = [k["key"] if isinstance(k, dict) else k for k in keys]
    if require_assignment is None:
        amap0 = manifest.get("gpu_assignment", {}).get(block, {}) if manifest is not None else {}
        require_assignment = any(k in amap0 for k in keys)
    require_assignment = bool(require_assignment) and manifest is not None
    res = {"block": block, "n_keys": len(keys), "items": {}, "cell_flags": {c: [] for c in CELLS},
           "manifest_checked": manifest is not None, "require_gpu_assignment": require_assignment}
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
                mm = manifest_mismatch(r, manifest, block)
                if mm:
                    fail[k].append(f"1:{c} 与冻结清单不符：{'; '.join(mm)}")
                    it1["manifest_mismatch"].setdefault(c, []).append(k)
                recs[c][k] = r
            else:
                fail[k].append(f"1:{c} 缺记录")
            if k in errs:
                fail[k].append(f"1:{c} 有报错回合")
        retried[c] = sorted(k for k in resolved & set(recs[c]) if k not in errs)
    res["items"]["1_records"] = it1
    res["items"]["10_retried_after_error"] = {c: {"n": len(v), "keys": v} for c, v in retried.items()}
    # 2. 同一键五格初始状态逐元素相同；3. 每次查询的噪声种子相同；11. 第 0 帧原图哈希（照报，按节点组合分组）
    n2 = n3 = n4m = 0
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
                if str(r0.get("gpu_name", "")) != str(r.get("gpu_name", "")):
                    fail[k].append(f"4:{c} 显卡型号 {r.get('gpu_name')} 与 {have[0]} 的 {r0.get('gpu_name')} 不同")
                    n4m += 1
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
    res["items"]["4_gpu_model_mismatch_within_key"] = n4m
    res["items"]["4_keys_by_gpu_model"] = dict(collections.Counter(
        str(recs[c][k].get("gpu_name", "?")) for k in keys for c in CELLS[:1] if k in recs[c]))
    res["items"]["11_frame0_by_host_combo"] = dict(f0)
    # 4（硬）渲染设备；5、6、7（照报）
    it5, it6, it7, gpu_bad = {}, {}, {}, {}
    for c in CELLS:
        n_bad5, frames, vanish, sumstats, arm_frac = 0, 0, 0, {}, []
        for k, r in recs[c].items():
            qs = json.loads(str(r["qstats"]))
            if item5(c, qs, r):
                n_bad5 += 1
            gb = gpu_fail(r, manifest, block, k, require_assignment)
            if gb:
                fail[k].append(f"4:{c} {'; '.join(gb)}")
                gpu_bad[c] = gpu_bad.get(c, 0) + 1
            for q in qs:
                for f in ("G_px", "changed_px", "changed_out_E_px", "hide_changed_out_G_px", "hide_changed_out_G_t30_px",
                          "G_vis_after_hide_px", "O_px", "O_drawn_px", "O_protected_px", "line_px_224", "line_px_224_geom",
                          "arm_px"):
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
    res["items"]["8_aa"] = json.load(open(aa)) if aa else "未提供（calib.py 在校准键上产出）"
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
             f"4 GPU / ICD 不合格记录：{res['items']['4_gpu_fail_records'] or 0}；同键五格型号不同：{res['items']['4_gpu_model_mismatch_within_key']}；"
             f"按型号（C0）：{res['items']['4_keys_by_gpu_model']}")
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
    L.append(f"格标记：{ {c: v for c, v in res['cell_flags'].items() if v} or '无'}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--manifest", default=None, help="冻结清单 json；默认 FROZEN_bench2_cogact（存在的话）")
    ap.add_argument("--aa", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man, mp = load_manifest(a.manifest)
    res = run(a.root, a.block, json.load(open(a.keys)), a.aa, man)
    res["manifest_path"] = mp
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    print(summary(res))


if __name__ == "__main__":
    sys.exit(main())
