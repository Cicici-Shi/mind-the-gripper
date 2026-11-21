"""WidowX 轮廓位置响应协议 v3.1 定稿（protocol_cue_widowx_v3_1_final.md，sha256 5af593e2…aa0）闭环十三格的门槛与照报（§7 的 15 项）。
§17 第 11 条：重写 _v1_drafts/cue_gate.py。读数（cue_readout.py）现算调用本文件，不读外部 gate.json。

只读门槛要用的字段（FIELDS）：不读成败与轨迹（st_*、success*、env_actions、raw_actions 都不在 FIELDS 里，模块载入时断言）。
first_close_step 只用来定「首次闭合之前」的查询帧（第 9、10 项）。键只按传入的主键集逐个找，不扫目录。

硬门槛（不过的键从该块全部十三格一起剔，并报数）：
  1 每格每键恰好一条完整记录；记录的键、格、块、cfg_id、簇、trial、env_seed 与主键集一致；代码哈希、冻结清单哈希、
    画法与平移量（幅度）、线宽、查询间隔与冻结清单一致。缺记录与报错（protocol_bench2.md §4 第 1 项；补采停止规则照 v1 草稿）：
      - 有完整记录、没有报错文件：通过；
      - 有记录又有报错文件 <键>.txt：剔；
      - 没有记录、有报错文件、提交次数 >= 2（<root>/formal_state/submits.json）：剔；
      - 没有记录、有报错文件、提交次数 < 2（或没有 submits.json）：未采完；
      - 既没有记录也没有报错文件：未采完。
    未采完的条目放在 res["incomplete"]，读数据此拒读。
  2 同一键各格初始状态相同（init_robot_qpos、init_actor_poses、init_actor_names、robot_base_pose 逐元素相同）。
  3 同一键各格每次查询的噪声种子相同。
  4 显卡在白名单、名字不含 MIG；vk_icd_filenames 等于清单值；同一键各格的显卡型号字符串相同。
停下来查的核对（第 5、6 项，check_cmp）：读 ANCHOR_POST.json、INIT_CMP.json，只给出「过 / 不过 + 原因」，处置交给读数（§4）。
照报项（只报数，不剔键）：
  7 各格变化像素落在预期编辑区之外的个数（q_changed_out_E_px）。
  8 线在策略输入里完全消失的帧占比（平移格、副本格、C4）；分母 = 该块该格保留键的、掩膜非空（q_G_px > 0）的查询帧合并；
    > flag_frac 标「轮廓操纵不足」（格级）。
  9 出视野（平移格、副本格）：首次闭合之前的查询帧（q_step <= first_close_step；没闭合取全部）合并，
    q_frac_inview < inview_thr 的帧占比 > flag_frac 标「出视野」（格级）。
 10 任务物平移格按侧：因出画丢掉的比例与「合成不足」标记、被手臂挡掉的比例、贴图落在目标容器上的帧占比、轮廓线被贴图截掉的比例。
 11 副本格：副本与当前轮廓重叠的线像素数（与比例）。
 12 被保护截掉占比、压在手臂上的占比、与任务物外圈相交的线像素数，按格、按正负方向并排；另给分层用的键单：
    前两次查询里 S(+b)、S(−b) 的线都不与任务物外圈相交的键（读数在这些键上单独算一份 κ_2(b)）。
 13 投影核对（按格）：末端投影落在夹爪掩膜包围盒内的帧占比（主），落在外扩 6 像素内的帧占比（原口径并列）；任务物原点投影落在掩膜上。
 14 线在输入里的可见像素数（q_line_px_224；任务物平移格的基准由采集方按 §7 第 14 项定）。
 15 重试成功的键数；同一键各格第 0 帧哈希是否相同，按节点组合分组。

解读（规格没写死、这里选定的做法；交回里逐条列出）：
  G1 第 8、9、10 项的标记只在通过硬门槛的键上算（读数用的就是这些键）；分母为 0 时同样打标记（无法核对按不足处理，同 b2x）。
  G2 「首次闭合之前的查询帧」= q_step <= first_close_step（§7 第 9 项原文「查询步不大于首次闭合步」）；没闭合的回合用全部查询帧。
  G3 种子比对按两格都有的查询逐个比（公共前缀）；查询次数不同的键另照报，不因此剔（闭合早晚不同，回合长度可能不同）。
  G4 硬门槛 1 另核记录里的 manifest_sha256 等于所用冻结清单文件的 sha256。
  G5 有报错文件但提交次数 < 2 算未采完（停止规则要求至少被提交过两次才剔）。
  G6 第 10 项「丢掉两成以上」取 q_ob_lost_offscreen > paste_loss_thr（与视野内比例 < 0.8 同一条边）；一侧超过 flag_frac，
     「那一对」两格都标「合成不足」。
  G7 第 12 项「不与外圈相交」= q_ring_px == 0，前两次查询 = q = 0、1；两格都要满足。
  G8 第 5 项另把 formal_state/STAGE1_OK / STAGE1_FAILED（采前锚点比对）照报，不进处置：处置只看 ANCHOR_POST 与 INIT_CMP（§17 第 12 条）。
  G9 ANCHOR_POST 算过 = ok 为真，且每块每个锚点格 differ、ref_missing、missing 都空、equal == n_ref == 锚点键集键数。
     INIT_CMP 某块算过 = ok_frame0、ok_init 都真，且该块十三格每格 differ_frame0、differ_init、missing 都空、
     frame0_equal == init_equal == n == 主键集键数。文件缺失或字段缺失都算不过（§4「参照缺任何一项算不过」）。
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

CELL_ORDER = ("C0", "C3", "C4", "S-b", "S-a", "S+a", "S+b", "SC-b", "SC+b", "OB-b", "OB+b", "S-h", "S+h")
SHIFT_CELLS = ("S-b", "S-a", "S+a", "S+b", "S-h", "S+h")
COPY_CELLS = ("SC-b", "SC+b")
OB_CELLS = ("OB-b", "OB+b")
LINE_CELLS = ("C4",) + SHIFT_CELLS + COPY_CELLS          # 第 8 项
VIEW_CELLS = SHIFT_CELLS + COPY_CELLS                     # 第 9 项
SIGN_PAIRS = (("S-h", "S+h"), ("S-a", "S+a"), ("S-b", "S+b"), ("SC-b", "SC+b"), ("OB-b", "OB+b"))
ANCHOR_CELLS = ("C0", "C3", "C4")
OUTLINE_WEAK = "轮廓操纵不足"   # 读数按这三个字符串读格标记
OUT_OF_VIEW = "出视野"
SYNTH_WEAK = "合成不足"
MIN_SUBMITS = 2
FIELDS = ("complete", "block", "cell", "key", "cfg_id", "cluster", "task", "trial", "env_seed", "code_sha",
          "manifest_sha256", "cell_kind", "shift_env", "line_width", "query_every", "gpu_name", "host",
          "vk_icd_filenames", "seeds", "frame0_sha256", "init_robot_qpos", "init_actor_poses", "init_actor_names",
          "robot_base_pose", "n_queries", "first_close_step", "q_step", "q_G_px", "q_frac_inview",
          "q_frac_protected", "q_frac_on_arm", "q_ring_px", "q_line_px_224", "q_changed_out_E_px",
          "q_tcp_in_G6", "q_tcp_in_Gbbox", "q_obj_on_mask", "q_edge_touch", "q_copy_overlap", "q_copy_overlap_px",
          "q_ob_paste_rate", "q_ob_lost_offscreen", "q_ob_lost_arm", "q_ob_on_container_px", "q_ob_line_cut_frac",
          "q_ob_T_px")
assert not any(f.startswith("st_") or f.startswith("success") for f in FIELDS)
assert "env_actions" not in FIELDS and "raw_actions" not in FIELDS
INIT_FIELDS = ("init_robot_qpos", "init_actor_poses", "robot_base_pose")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {f: z[f] for f in FIELDS if f in z.files}


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def state_dir(root):
    """formal_state 的位置。根目录有 SYNTHETIC_DATA 标记（合成测试）时：<root>/formal_state 存在就用它，否则 <root>/../formal_state。
    没有标记（真实数据）时固定为本文件所在目录（b3）下的 formal_state，与 cue_readout.REAL_STATE、submit_cue.sh、cue_cmp.py 一致，
    不由 --root 推出（11-14 复核：读数已钉死，门槛的 submits.json 一并钉死）。"""
    if not os.path.exists(os.path.join(root, "SYNTHETIC_DATA")):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "formal_state")
    p = os.path.join(root, "formal_state")
    return p if os.path.isdir(p) else os.path.join(os.path.dirname(os.path.normpath(root)), "formal_state")


def load_submits(root):
    p = os.path.join(state_dir(root), "submits.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def manifest_mismatch(r, man, cell, man_sha):
    bad = []
    try:
        cs = json.loads(str(r["code_sha"])) if "code_sha" in r else {}
    except ValueError:
        cs = {}
    if not cs:
        bad.append("记录没有代码哈希")
    for f, h in cs.items():
        if man["code_sha256"].get(f) != h:
            bad.append(f"代码 {f} 哈希不符")
    if man_sha is not None and str(r.get("manifest_sha256", "")) != man_sha:
        bad.append("manifest_sha256 与所用冻结清单不符")
    want = man["cells"][cell]
    if str(r.get("cell_kind", "")) != want["kind"]:
        bad.append(f"画法 {r.get('cell_kind')} != {want['kind']}")
    se = [int(x) for x in r["shift_env"]] if "shift_env" in r else None
    if se != [int(x) for x in want["shift_env"]]:
        bad.append(f"平移 {se} != {want['shift_env']}")
    if "line_width" not in r or int(r["line_width"]) != int(man["line_width"]):
        bad.append("线宽不符")
    if "query_every" not in r or int(r["query_every"]) != int(man["query_every"]):
        bad.append("查询间隔不符")
    return bad


def key_mismatch(r, kd, block, cell):
    bad = []
    if str(r.get("key", "")) != kd["key"] or str(r.get("cell", "")) != cell or str(r.get("block", "")) != block:
        bad.append("键/格/块不符")
    for f in ("cfg_id", "cluster"):
        if f in kd and str(r.get(f, "")) != str(kd[f]):
            bad.append(f"{f} 不符")
    for f in ("trial", "env_seed"):
        if f in kd and (f not in r or int(r[f]) != int(kd[f])):
            bad.append(f"{f} 不符")
    return bad


def gpu_fail(r, man):
    g = str(r.get("gpu_name", ""))
    bad = []
    if g not in man["gpu_whitelist"]:
        bad.append(f"显卡 {g!r} 不在白名单")
    if "MIG" in g:
        bad.append("MIG")
    icd = str(r["vk_icd_filenames"]) if "vk_icd_filenames" in r else None
    if icd != man["vk_icd_filenames"]:
        bad.append(f"VK_ICD_FILENAMES={icd!r}")
    return bad


def preclose_mask(r):
    qs = np.asarray(r["q_step"])
    fcs = int(r["first_close_step"])
    return np.ones(len(qs), bool) if fcs < 0 else qs <= fcs


def _med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def _frac(m, valid):
    n = int(np.sum(valid))
    return (float(np.sum(m & valid)) / n if n else None), n


def run(root, block, keys, manifest, manifest_sha=None):
    """keys：主键集（dict 列表）；manifest：冻结清单 dict；manifest_sha：清单文件 sha256（None 不核）。"""
    p = manifest["params"]
    flag_frac, inview_thr, loss_thr = float(p["flag_frac"]), float(p["inview_thr"]), float(p["paste_loss_thr"])
    cells = tuple(manifest.get("cell_order", CELL_ORDER))
    kd = {k["key"]: k for k in keys}
    names = [k["key"] for k in keys]
    submits = load_submits(root).get(block, {})
    res = {"block": block, "n_keys": len(names), "items": {}, "cell_flags": {c: [] for c in cells}, "incomplete": []}
    recs = {c: {} for c in cells}
    fail = {k: [] for k in names}
    it1 = {"missing": {}, "errors": {}, "error_excluded": {}, "manifest_mismatch": {}, "key_mismatch": {}}
    retried = {}
    for c in cells:
        d = os.path.join(root, block, c)
        ed = os.path.join(d, "_errors")
        it1["missing"][c], it1["errors"][c], it1["error_excluded"][c], retried[c] = [], [], [], []
        for k in names:
            rp = os.path.join(d, k + ".npz")
            has_err = os.path.exists(os.path.join(ed, k + ".txt"))
            if has_err:
                it1["errors"][c].append(k)
            if not os.path.exists(rp):
                it1["missing"][c].append(k)
                n_sub = int(submits.get(c, {}).get(k, 0))
                if has_err and n_sub >= MIN_SUBMITS:
                    fail[k].append(f"1:{c} 只有报错文件（提交 {n_sub} 次）")
                    it1["error_excluded"][c].append(k)
                elif has_err:
                    res["incomplete"].append([c, k, f"只有报错文件，提交 {n_sub} 次 < {MIN_SUBMITS}"])
                else:
                    res["incomplete"].append([c, k, "既没有记录也没有报错文件"])
                continue
            r = load(rp)
            if not bool(r.get("complete", False)):
                fail[k].append(f"1:{c} 记录不完整")
                continue
            km = key_mismatch(r, kd[k], block, c)
            if km:
                fail[k].append(f"1:{c} 与主键集不符：{'; '.join(km)}")
                it1["key_mismatch"].setdefault(c, []).append(k)
            mm = manifest_mismatch(r, manifest, c, manifest_sha)
            if mm:
                fail[k].append(f"1:{c} 与冻结清单不符：{'; '.join(mm)}")
                it1["manifest_mismatch"].setdefault(c, []).append(k)
            if has_err:
                fail[k].append(f"1:{c} 有报错回合")
            recs[c][k] = r
            if os.path.exists(os.path.join(ed, k + ".txt.resolved")) and not has_err:
                retried[c].append(k)
    res["items"]["1_records"] = it1
    n2 = n3 = 0
    nq_diff = []
    for k in names:
        have = [c for c in cells if k in recs[c]]
        for c in have:
            gb = gpu_fail(recs[c][k], manifest)
            if gb:
                fail[k].append(f"4:{c} {'; '.join(gb)}")
        if len(have) < 2:
            continue
        r0 = recs[have[0]][k]
        gnames = {str(recs[c][k].get("gpu_name", "")) for c in have}
        if len(gnames) > 1:
            fail[k].append(f"4:各格显卡型号不同 {sorted(gnames)}")
        lens = set()
        for c in have[1:]:
            r = recs[c][k]
            same = all(f in r0 and f in r and np.array_equal(np.asarray(r0[f]), np.asarray(r[f])) for f in INIT_FIELDS)
            same = same and "init_actor_names" in r0 and str(r0.get("init_actor_names")) == str(r.get("init_actor_names"))
            if not same:
                fail[k].append(f"2:{c} 初始状态与 {have[0]} 不同")
                n2 += 1
            s0, s1 = np.asarray(r0["seeds"]), np.asarray(r["seeds"])
            m = min(len(s0), len(s1))
            lens.update((len(s0), len(s1)))
            if m == 0 or not np.array_equal(s0[:m], s1[:m]):
                fail[k].append(f"3:{c} 噪声种子与 {have[0]} 不同")
                n3 += 1
        if len(lens) > 1:
            nq_diff.append(k)
    res["items"]["2_init_state_mismatch"] = n2
    res["items"]["3_seed_mismatch"] = n3
    res["items"]["3_n_queries_differ_keys"] = nq_diff
    excluded = sorted(k for k, v in fail.items() if v)
    kept = [k for k in names if k not in set(excluded)]
    res.update(excluded_keys=excluded, n_excluded=len(excluded), kept_keys=kept, n_retained=len(kept),
               fail_reasons={k: v for k, v in fail.items() if v})
    res["hard_fail_by_cell"] = {c: {i: sum(1 for v in fail.values() if any(x.startswith(f"{i}:{c}") for x in v))
                                    for i in ("1", "2", "3", "4")} for c in cells}
    res["hard_fail_gpu_name_differs"] = sum(1 for v in fail.values() if any(x.startswith("4:各格") for x in v))
    # ---------------------------------------------------------------- 照报项（只在保留的键上算，解读 G1）
    it7, it8, it9, it10, it11, it12, it13, it14 = {}, {}, {}, {}, {}, {}, {}, {}
    for c in cells:
        rs = [recs[c][k] for k in kept if k in recs[c]]
        cat = lambda f: np.concatenate([np.asarray(r[f], float).ravel() for r in rs if f in r]) if rs else np.zeros(0)  # noqa: E731
        outE = [float(np.nansum(np.asarray(r["q_changed_out_E_px"], float))) for r in rs]
        it7[c] = {"total_px": float(sum(outE)), "keys_with_any": int(sum(1 for x in outE if x > 0))}
        bb, g6, om = cat("q_tcp_in_Gbbox"), cat("q_tcp_in_G6"), cat("q_obj_on_mask")
        it13[c] = {"tcp_in_Gbbox_frac": _frac(bb == 1, bb >= 0)[0], "tcp_in_Gbbox_frames": int(np.sum(bb >= 0)),
                   "tcp_in_G6_frac": _frac(g6 == 1, g6 >= 0)[0], "obj_on_mask_frac": _frac(om == 1, om >= 0)[0],
                   "obj_on_mask_frames": int(np.sum(om >= 0))}
        lp = cat("q_line_px_224")
        lp = lp[lp >= 0]
        it14[c] = {"median": _med(lp), "min": float(lp.min()) if len(lp) else None, "frames": int(len(lp))}
        if c in LINE_CELLS:
            frames = vanish = 0
            for r in rs:
                gpx, l2 = np.asarray(r["q_G_px"]), np.asarray(r["q_line_px_224"])
                frames += int(np.sum(gpx > 0))
                vanish += int(np.sum((gpx > 0) & (l2 == 0)))
            fv = vanish / frames if frames else None
            it8[c] = {"frames_mask_nonempty": frames, "frames_line_vanished": vanish, "frac": fv}
            if fv is None or fv > flag_frac:
                res["cell_flags"][c].append(OUTLINE_WEAK)
        if c in VIEW_CELLS:
            pre_n = pre_out = 0
            for r in rs:
                iv = np.asarray(r["q_frac_inview"], float)
                pm = preclose_mask(r) & np.isfinite(iv)
                pre_n += int(np.sum(pm))
                pre_out += int(np.sum(iv[pm] < inview_thr))
            fo = pre_out / pre_n if pre_n else None
            it9[c] = {"preclose_frames": pre_n, "frames_inview_below": pre_out, "frac": fo}
            if fo is None or fo > flag_frac:
                res["cell_flags"][c].append(OUT_OF_VIEW)
        if c in OB_CELLS:
            pre_n = pre_bad = cont = 0
            arm, cut, paste = [], [], []
            for r in rs:
                lo = np.asarray(r["q_ob_lost_offscreen"], float)
                pm = preclose_mask(r) & np.isfinite(lo)
                pre_n += int(np.sum(pm))
                pre_bad += int(np.sum(lo[pm] > loss_thr))
                cont += int(np.sum(np.asarray(r["q_ob_on_container_px"], float) > 0))
                arm.extend(np.asarray(r["q_ob_lost_arm"], float).tolist())
                cut.extend(np.asarray(r["q_ob_line_cut_frac"], float).tolist())
                paste.extend(np.asarray(r["q_ob_paste_rate"], float).tolist())
            nfr = sum(len(np.asarray(r["q_step"])) for r in rs)
            it10[c] = {"preclose_frames": pre_n, "frames_lost_offscreen_gt_thr": pre_bad,
                       "frac_lost_offscreen": pre_bad / pre_n if pre_n else None,
                       "lost_arm_median": _med(arm), "lost_arm_mean": float(np.nanmean(arm)) if arm else None,
                       "on_container_frame_frac": cont / nfr if nfr else None,
                       "line_cut_frac_median": _med(cut), "paste_rate_median": _med(paste)}
        if c in COPY_CELLS:
            op, of = cat("q_copy_overlap_px"), cat("q_copy_overlap")
            it11[c] = {"overlap_px_median": _med(op[op >= 0]), "overlap_frac_median": _med(of)}
        if c not in ("C0", "C3"):
            rp = cat("q_ring_px")
            it12[c] = {"protected_frac_median": _med(cat("q_frac_protected")), "on_arm_frac_median": _med(cat("q_frac_on_arm")),
                       "ring_px_median": _med(rp[rp >= 0]), "ring_px_mean": float(np.mean(rp[rp >= 0])) if np.any(rp >= 0) else None,
                       "edge_touch_frames": int(np.sum(cat("q_edge_touch") == 1))}
    # 合成不足：一侧超过就整对标（解读 G6）
    ob_bad = [c for c in OB_CELLS if c in it10 and (it10[c]["frac_lost_offscreen"] is None or it10[c]["frac_lost_offscreen"] > flag_frac)]
    if ob_bad:
        for c in OB_CELLS:
            if c in res["cell_flags"]:
                res["cell_flags"][c].append(SYNTH_WEAK)
    pairs = {}
    for lo, hi in SIGN_PAIRS:
        if lo not in it12 or hi not in it12:
            continue
        pr = {}
        for f in ("protected_frac_median", "on_arm_frac_median", "ring_px_median", "ring_px_mean"):
            a, b = it12[lo][f], it12[hi][f]
            pr[f] = {lo: a, hi: b, "diff_plus_minus": None if a is None or b is None else b - a}
        pairs[f"{lo}/{hi}"] = pr
    ring_free = []
    for k in kept:
        ok = True
        for c in ("S+b", "S-b"):
            rp = np.asarray(recs[c][k]["q_ring_px"])[:2] if k in recs.get(c, {}) else np.array([1])
            ok = ok and len(rp) == 2 and bool(np.all(rp == 0))
        if ok:
            ring_free.append(k)
    f0 = collections.defaultdict(lambda: {"keys": 0, "frame0_inconsistent": 0})
    for k in kept:
        have = [c for c in cells if k in recs[c]]
        combo = "+".join(sorted({str(recs[c][k].get("host", "?")) for c in have}))
        hs = {str(recs[c][k].get("frame0_sha256", "")) for c in have}
        f0[combo]["keys"] += 1
        f0[combo]["frame0_inconsistent"] += int(len(hs) > 1)
    st1 = {n: os.path.exists(os.path.join(state_dir(root), n)) for n in ("STAGE1_OK", "STAGE1_FAILED")}
    res["items"].update({
        "5_stage1_files": st1,
        "7_changed_outside_edit": it7, "8_line_vanish": it8, "9_out_of_view": it9, "10_obj_shift": it10,
        "11_copy_overlap": it11, "12_pixels_by_cell": it12, "12_sign_pairs": pairs,
        "12_ring_free_keys_Sb_q01": ring_free, "13_projection_check": it13, "14_line_px_224": it14,
        "15_retried_after_error": {c: {"n": len(v), "keys": v} for c, v in retried.items()},
        "15_frame0_by_host_combo": dict(f0)})
    return res


# ---------------------------------------------------------------- 第 5、6 项：只判过不过，处置交读数（解读 G9）
def _jload(path):
    if not os.path.exists(path):
        return None, f"缺文件 {path}"
    try:
        with open(path) as f:
            return json.load(f), None
    except ValueError as e:
        return None, f"{path} 不是合法 json：{e}"


def check_anchor_post(path, blocks, anchor_n):
    """anchor_n：{块: 锚点键集键数}。返回 {"ok": bool, "problems": [...]}。"""
    d, err = _jload(path)
    if err:
        return {"ok": False, "problems": [err], "path": path}
    probs = []
    if d.get("ok") is not True:
        probs.append("ok 不为真")
    per = d.get("per", {})
    for b in blocks:
        for c in ANCHOR_CELLS:
            v = per.get(b, {}).get(c)
            if v is None:
                probs.append(f"{b}/{c} 缺比对")
                continue
            for f in ("differ", "ref_missing", "missing"):
                if f not in v or v[f]:
                    probs.append(f"{b}/{c} {f}={v.get(f)}")
            if v.get("equal") != v.get("n_ref") or v.get("n_ref") != anchor_n[b]:
                probs.append(f"{b}/{c} equal={v.get('equal')} n_ref={v.get('n_ref')}（锚点键 {anchor_n[b]}）")
    return {"ok": not probs, "problems": probs, "path": path}


def check_init_cmp(path, blocks, cells, main_n):
    """main_n：{块: 主键集键数}。返回 {块: {"ok": bool, "problems": [...]}}。"""
    d, err = _jload(path)
    if err:
        return {b: {"ok": False, "problems": [err], "path": path} for b in blocks}
    out = {}
    for b in blocks:
        probs = []
        if d.get("ok_frame0", {}).get(b) is not True:
            probs.append("ok_frame0 不为真")
        if d.get("ok_init", {}).get(b) is not True:
            probs.append("ok_init 不为真")
        for c in cells:
            v = d.get("per", {}).get(b, {}).get(c)
            if v is None:
                probs.append(f"{c} 缺比对")
                continue
            for f in ("differ_frame0", "differ_init", "missing"):
                if f not in v or v[f]:
                    probs.append(f"{c} {f}={v.get(f)}")
            if not (v.get("n") == v.get("frame0_equal") == v.get("init_equal") == main_n[b]):
                probs.append(f"{c} n={v.get('n')} frame0_equal={v.get('frame0_equal')} init_equal={v.get('init_equal')}（主键 {main_n[b]}）")
        out[b] = {"ok": not probs, "problems": probs, "path": path}
    return out


def summary(res):
    L = [f"门槛 {res['block']}：键 {res['n_keys']}，剔除 {res['n_excluded']}，保留 {res['n_retained']}；未采完 {len(res['incomplete'])}"]
    it1 = res["items"]["1_records"]
    L.append("1 缺记录 " + ", ".join(f"{c}={len(v)}" for c, v in it1["missing"].items())
             + "；报错 " + ", ".join(f"{c}={len(v)}" for c, v in it1["errors"].items())
             + "；只有报错而剔 " + ", ".join(f"{c}={len(v)}" for c, v in it1["error_excluded"].items()))
    L.append("硬门槛不过的键数（格：项 1/2/3/4）：" + "；".join(
        f"{c} " + "/".join(str(d[i]) for i in ("1", "2", "3", "4")) for c, d in res["hard_fail_by_cell"].items())
        + f"；各格显卡型号不同 {res['hard_fail_gpu_name_differs']}")
    it = res["items"]
    L.append(f"2 初始状态不同 {it['2_init_state_mismatch']}；3 种子不同 {it['3_seed_mismatch']}；查询次数不同的键 {len(it['3_n_queries_differ_keys'])}")
    L.append(f"5 采前锚点比对文件：{it['5_stage1_files']}")
    L.append("7 编辑区外变化像素（总数/有违例键）：" + "；".join(f"{c} {v['total_px']:.0f}/{v['keys_with_any']}" for c, v in it["7_changed_outside_edit"].items()))
    for c, v in it["8_line_vanish"].items():
        L.append(f"8 {c}：线消失 {v['frames_line_vanished']}/{v['frames_mask_nonempty']}（{v['frac']}）")
    for c, v in it["9_out_of_view"].items():
        L.append(f"9 {c}：闭合前出视野帧 {v['frames_inview_below']}/{v['preclose_frames']}（{v['frac']}）")
    for c, v in it["10_obj_shift"].items():
        L.append(f"10 {c}：" + "；".join(f"{f} {x}" for f, x in v.items()))
    for c, v in it["11_copy_overlap"].items():
        L.append(f"11 {c}：重叠像素中位数 {v['overlap_px_median']}，重叠比例中位数 {v['overlap_frac_median']}")
    for p, d in it["12_sign_pairs"].items():
        L.append(f"12 {p}：" + "；".join(f"{f} {v}" for f, v in d.items()))
    L.append(f"12 前两次查询 S(±b) 线都不碰外圈的键：{len(it['12_ring_free_keys_Sb_q01'])}")
    L.append("13 投影核对：" + "；".join(f"{c} 包围盒内 {v['tcp_in_Gbbox_frac']}（外扩 6 像素 {v['tcp_in_G6_frac']}），物原点在掩膜上 {v['obj_on_mask_frac']}"
                                     for c, v in it["13_projection_check"].items()))
    L.append("14 可见线像素中位数：" + "；".join(f"{c} {v['median']}" for c, v in it["14_line_px_224"].items()))
    L.append("15 重试成功键数：" + ", ".join(f"{c}={v['n']}" for c, v in it["15_retried_after_error"].items()))
    L.append("15 第 0 帧哈希（节点组合：键数/不一致）：" + "；".join(f"{h}: {g['keys']}/{g['frame0_inconsistent']}" for h, g in it["15_frame0_by_host_combo"].items()))
    L.append(f"格标记：{ {c: v for c, v in res['cell_flags'].items() if v} or '无'}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--keys", required=True, help="主键集 keysets/cue_<块>.json")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor-post", default=None, help="缺省 <root>/../out_cue_anchor_post/ANCHOR_POST.json")
    ap.add_argument("--init-cmp", default=None, help="缺省 <root>/INIT_CMP.json")
    a = ap.parse_args(argv)
    with open(a.manifest) as f:
        man = json.load(f)
    with open(a.keys) as f:
        keys = json.load(f)
    res = run(a.root, a.block, keys, man, file_sha(a.manifest))
    if res["incomplete"]:
        print(f"拒绝：{a.block} 未采完 {len(res['incomplete'])} 个格键，例如 {res['incomplete'][:3]}", file=sys.stderr)
        return 2
    root = os.path.abspath(a.root)
    res["5_anchor_post"] = check_anchor_post(a.anchor_post or os.path.join(os.path.dirname(root), "out_cue_anchor_post", "ANCHOR_POST.json"),
                                             [a.block], {a.block: int(man["anchor_keysets"][a.block]["n"])})
    res["6_init_cmp"] = check_init_cmp(a.init_cmp or os.path.join(root, "INIT_CMP.json"), [a.block],
                                       tuple(man.get("cell_order", CELL_ORDER)), {a.block: len(keys)})[a.block]
    print(f"5 锚点 --post：{res['5_anchor_post']}\n6 初始比对：{res['6_init_cmp']}")
    res["manifest_path"] = a.manifest
    with open(a.out + ".tmp", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(summary(res))


if __name__ == "__main__":
    sys.exit(main())
