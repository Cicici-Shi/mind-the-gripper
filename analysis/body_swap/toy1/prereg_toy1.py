"""toy1 pre-registered analysis: protocol_toy1.md (sha256 0dc137a2...) as amended by protocol_toy1_astra_decisions.md
(Astra rulings (a)-(g), user-authorised; where they conflict the rulings win). §4 estimands, §5.1 Q0, §5.2 Q1 (route
screening), §5.3 paired-appearance shift probe, §5.4 parts probe (descriptive), descriptive 10-task / failure tables.
Frozen before the first toy shard; run ONCE (§6, ruling (g)). Refuses to start if the protocol or rulings sha256
differs from --expect-protocol-sha / --expect-decisions-sha, and refuses to overwrite an existing --out /
--verdict-json (the out file is created with mode "x" before any data is read).

Usage (cluster, conda env oft, CPU only: srun -p normal -c 4 --mem 8G -t 20 python prereg_toy1.py ...):
  python prereg_toy1.py --expect-protocol-sha SHA --expect-decisions-sha SHA2 --keys first|all --out results_toy1.txt \
      [--runs $PROJ_ROOT/runs_toy] [--gate gate_toy.json] [--probe-shift shift.jsonl] [--probe-parts parts.jsonl] \
      [--protocol $PROJ_ROOT/embody_toy/protocol_toy1.md] [--decisions $PROJ_ROOT/embody_toy/protocol_toy1_astra_decisions.md] \
      [--stages-dir $PROJ_ROOT/embody/analysis] [--B 10000] [--verdict-json verdicts.json]
--keys first = shard ep 40-44 (50 keys); all = both shards ep 40-49 (100 keys), only if both are complete (ruling (g)).
Tests: test_prereg_toy1.py (synthetic data only). All decision lines are the module constants below.

Inference (§4): paired cluster bootstrap over (task, ep) keys, one draw matrix for all closed-loop cells, B = 10000,
seed 0, percentile 95% / 90% CIs; per contrast discordant counts and exact McNemar p. Probes: bootstrap over episodes.

实现口径（协议 + 裁定没写死、我按最字面的读法实现的地方）
 K1  键全集 = 所选片的设计键减去 §8.1 场景不一致被剔除的键。每个估计量用它那两格都有的键；bootstrap 在全集上抽一次。
     ρ = R_hid / R_fc，两项各用自己的共同键（完整片下二者相同；不同时打印提示）。
 K2  §8.1 场景一致：ref_body_pos 按 body 名与 toy_s 同键比较，最大差 < 0.00005 mm；body 集合不同算不一致。不一致的键从所有
     格剔除并列出（不是该格 FAIL）。有键在 toy_s 里无对照 -> 该格此项 UNK；toy_s 自身 NA。
 K3  copy_grip_cmd：取 align_json 顶层，没有再取 align_json["toy"] 子字典。任一集 0 -> FAIL；无 0 但有缺 -> UNK。所有格都查。
 K4  grp_map：G 家族（toy_g、toy_g_tfc、toy_g_hid、toy_g_t、toy_g_cam）应为 "kp100"；不符 -> FAIL，有缺 -> UNK。S 家族只印。
     G_map20 不跑（裁定 (e)）：toy_g_map20 即使存在也不读，M 不算。
 K5  §8.2 只施于 G_T^fc：查询帧 = [::8]；twin_cam_err_mm <= 0.01 且 _deg <= 0.01（全帧最大值另印）；twin_pos_err_mm > 1 mm 的
     查询帧 < 1%；twin_stale 合计 0。旋转残差只印。G_T 的替身量（1 mm 比例、第二批 2 mm/1° 口径、陈旧帧、相机差约 6.4 mm）只描述。
     裁定 (a) 的「G_T 与 G_T^fc 同状态手臂姿态与第三视角一致」只读门槛 json 中 toy_g_tfc 的条目 GATE_ITEM_ARM
     （"arm_thirdview_consistency"）的 status；缺该条目 -> UNK。
 K6  门槛 json 合并：cell status 或任一 item 为 FAIL -> FAIL；否则有非 PASS/NA -> UNK；json 缺该格或无 --gate -> UNK。格状态 =
     自算项与 json 项合并：FAIL -> does not count（估计量照印，判定印「不判定…按线本应…仅供参考」）；UNK -> 照判并标「门槛未核实」。
 K7  配置核对（hide_gripper、twin_cam、wrist_cam_dz_mm）与 GPU 型号缺失只打印，不进门槛。
 K8  完整性（裁定 (g)）：片 = ep 40-44 或 45-49 的 50 键；必需格 = S、G、G_T^fc、G_hid、G_T、S_hid、G_cam。某必需格在某片缺任何
     一集 -> 该片「未完成」。--keys first 要求第一片完成；all 要求两片都完成。未完成时不计算、不打印任何闭环估计量与判定
     （只印门槛与各片缺集数），探针照常（§6 探针可边出边看）。完整性按 §8.1 剔除之前的原始键算；剔除的键之后照协议删去。
 K9  §5.0（裁定 (f) 删除前置）：R_T 照算照印，并印原线（点 >= 15 且 CI 不含 0）过没过，只描述，不阻断 Q0/Q1。
 K10 §5.1 Q0「固定相机重绘效应」：GO = R_fc 点 >= 15 且 95% 下界 > 5；R_fc 95% 上界 < 10 且 R_T 点 >= 15 时，C_P 95% 下界 > 5 ->
     「转向相机」，否则「不确定」；其余「未检出」（不写外观为主、不据此扩量）。所需格 G、G_T^fc、G_T。
 K11 §5.2 Q1 路线筛选：只在 Q0 = GO（含门槛未核实）时判。干扰为主 = ρ 点 >= 0.7 且 H_S 下界 > -5；线索为主 = ρ 点 <= 0.3 或
     H_S 上界 < -15；其余混合。输出注明不能推出夹爪必要或擦除足够。ρ 比值 CI 照算 R_fc* <= 0 的重采样，并印其比例。
     §5.5 分支：干扰 -> A，线索 -> B，混合 -> 人工（脚本只印 Q2/Q3），相机 -> 相机转向；Q0 未检出 / 不确定 -> 人工。
 K12 §5.3 量：P(cond, u) = Σ_k (R_base @ a_k[:3]) · u（世界系投影），a = "actions"（process_action 后）；u 单位化。D_x、D_x10、
     D_y、D_cam、L 定义同协议；副量 k = 1 与夹爪符号翻转率（相对条件 "0"，8 步平均）。"raw" 不用。jsonl 的外观字段读
     "appearance"，没有则读 "side"。
 K13 §5.3 聚合与判定（裁定 (f)）：每外观每集（task, ep）取帧中位数；「检出」= D_x 集中位数 < 0 的集比例 >= 0.70 且
     median|D_x| >= 2 median|L| 且 |D_x10| > |D_x| 的集比例 >= 0.60；否则「未检出」。D_cam 与 D_y 只描述。判定用点估计，
     bootstrap 只给 CI。输出注明过线不能排除读数偏置、不能证明效应来自指尖，未检出不等于不存在。
 K14 0/0′（§8.6 保留）：任一帧 same_0_0p 有 false 即报告，并印噪声底 median|P(0′,u_x) − P(0,u_x)|（集中位数的中位数），只描述
     （原「替代 1.2 倍下限」随相机幅度判据一起删除）。
 K15 操纵与配对检查：每帧 cam_err <= 0.01 mm / 0.01°；每帧 centroid_px["x+5"] 模在 5-12 px；两外观的 (task, ep, step) 集合
     必须相同（裁定 (f) 同帧配对）；帧来源由 ep 推断不混用（全在 40-49 = toy G，全在 0-19 = 第二批 G）。任一 FAIL -> 该外观
     「不判定」并印按线本应结论；配对/来源 FAIL 两外观都不判定。质心方向与 u 像平面投影：jsonl 无投影，UNK。集数、帧数、
     phase 分布不符只警告。手臂不动、同 proprio 由探针工具保证，jsonl 无字段可核。
 K16 解读表（检出 / 未检出）：Panda 外观未检出 -> 行 1；Panda 检出 + Robotiq 未检出 -> 行 2；Panda 检出 + Robotiq 检出且 median|D_x|
     >= 0.5 x Panda -> 行 3；Panda 检出 + Robotiq 检出但不到一半 ->「未定义」不落行；外观不判定或缺 -> 不落行。原行 4（不确定）
     随「不存在」档删除。
 K17 §5.4：d(ci, cj) = mean_k ||a_i[k,:3] − a_j[k,:3]||；逐帧 r_i = d(ci,c1)/d(c0,c1)（d(c0,c1) = 0 跳过）；集内中位数 -> 集间中位数
     （主）、均值、比值均值口径另印。读法全描述：r_2 <= 0.3、r_3 对 r_4、c5 <= 0.3 且 c6 >= 0.7、c5 c6 都在 [0.4, 0.7]。c5、c6 必做
     （裁定 (a)），缺则印「门槛未核实」，仍不进判定。
 K18 描述：10 任务表；conc 相对 S；stages.py 失败类型；contact_* 在线接触；两片 GPU 组成；R10 噪声带标签（|点| <= 3 且 GPU 组成不同）。
 K19 C_P 原预期 |C_P| <= 5 只印数（裁定 (b)：不显著不等于等价），|C_P| >= 10 印 §2.2 触发语；C_R 与 C_P 同号与比值只印。
 K20 读数纪律：打印协议、裁定、脚本三个 sha256；两个 expect sha 必填，不符 -> 退出码 2（不建输出）；输出已存在 -> 退出码 3。

协议 + 裁定的缺口 / 含糊处（未自补规则，交回裁定）
 Q1  ρ 比值 CI 在 R_fc* <= 0 的重采样上发散，剔除 / 截断 / 照算没写；我照算并印比例。
 Q2  ρ 用 R_hid、R_fc 各自共同键还是三格交集没写（完整片下无差别；有 §8.1 剔除时也相同，因为剔除对所有格）。
 Q3  G_T 在 §8 无替身门槛（§8.2 只写 G_T^fc）；§8.2「IK 残差 > 1 mm（同第二批 G_T 门槛）」与 block3 K9 第二批 IK 口径 2 mm/1° 不一致。
 Q4  §8.2 相机差「每个查询帧」与输入说明「每帧」：门槛用查询帧，全帧最大值另印。
 Q5  「方向符号集比例」只对 D_x 还是 D_x 与 D_y 都要：我只用 D_x。
 Q6  Robotiq 外观检出但 median|D_x| 不到 Panda 外观一半：无读法（K16）。
 Q7  质心「方向与 u 的像平面投影一致」脱机核不了；「超出 5-12 px 停下查」我按任一帧超出即 FAIL，若应按比例需给数。
 Q8  §5.4「按集聚合」的集内 / 集间统计量没写；我用中位数。
 Q9  conc 的参照格没写；沿用 block3 K14 相对 S。
 Q10 Q0「未检出」「不确定」与 Q1「混合」之后的全量分支没有映射；脚本只印「需人工」。
 Q11 裁定 (a) 的手臂 / 第三视角一致门槛在 gate json 里的条目名未定；我写死为 "arm_thirdview_consistency"（常量 GATE_ITEM_ARM），
     gate_toy.py 须用同名或改常量。
 Q12 §8.1 剔除的键与「整片 50 键」规则的关系没写：我先按原始键判完整，再按协议剔除（剔除后估计量键数 < 50/100）。
 Q13 基建代码当前把 toy 设置写在 align_json["toy"] 子键，字段名与输入说明不一致；两处都读，冻结前需对一次真实冒烟 dump。
 Q14 §5.3 Panda 外观帧是否「只用成功集」在裁定后的配对设计里已不适用（两外观都用 Robotiq G 帧）；G 帧的成功 / 失败集都可，
     prereg 不核。
 Q15 §5.0「CI 包含 0」取闭区间（只影响描述行）。
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# =============================================================================================== 判定线（草案常量）
# §5.0 前置复现
RT_MIN = 15.0                 # R_T 点估计下限
# §5.1 Q0
GO_POINT = 15.0               # R_fc 点估计 >= 15
GO_LO95 = 5.0                 # 且 95% 下界 > 5
CAM_HI95 = 10.0               # 转向相机：R_fc 95% 上界 < 10
CAM_RT_MIN = 15.0             # 且 R_T 点 >= 15
CAM_CP_LO95 = 5.0             # 且 C_P 95% 下界 > 5（裁定 (f)），否则「不确定」
# §5.2 Q1
RHO_INTERF = 0.7              # ρ 点 >= 0.7
HS_INTERF_LO95 = -5.0         # 且 H_S 95% 下界 > -5
RHO_CUE = 0.3                 # ρ 点 <= 0.3
HS_CUE_HI95 = -15.0           # 或 H_S 95% 上界 < -15
# §2.2 / §5.1 描述性预期（裁定 (b)：差异不显著不能写成 ±5 等价，只印数）
CP_EXPECT, CP_TRIGGER = 5.0, 10.0
R10_BAND = 3.0                # R10 噪声带 |点| <= 3
# §5.3 探针
SIGN_PRESENT = 0.70           # 检出：预测符号集比例 >= 0.70
SPEC_RATIO = 2.0              #       且 median|D_x| >= 2 median|L|（相机幅度比判据已删，裁定 (f)）
DOSE_FRAC = 0.6               #       且 剂量单调集比例 >= 0.60（裁定 (f) 升为必要条件）
ROBOTIQ_HALF = 0.5            # Robotiq 外观 median|D_x| 不低于 Panda 外观的一半（§5.3「读得懂」）
PROBE_EPISODES, PROBE_EP_PER_TASK, PROBE_FRAMES, PROBE_APPROACH, PROBE_PREGRASP = 60, 6, 5, 2, 3
# §8.6 探针操纵检查
PROBE_CAM_MM, PROBE_CAM_DEG = 0.01, 0.01
CENTROID_PX = (5.0, 12.0)     # x+5 质心位移像素范围
# §5.4 部件探针读法（描述）
R2_INTERF = 0.3
PART_FINGER_LO, PART_FINGER_HI = 0.3, 0.7
PART_MIX = (0.4, 0.7)
# §8.1 / §8.2 门槛
FIX_TOL_MM = 0.00005
CAM_TOL_MM, CAM_TOL_DEG = 0.01, 0.01
IK_RES_MM, IK_FRAC = 1.0, 0.01
QUERY_STRIDE = 8
# §3 / §4
B_DEFAULT, SEED = 10000, 0
TASKS = range(10)
EPS = {"all": range(40, 50), "first": range(40, 45)}
SHARDS = [("ep40-44", range(40, 45)), ("ep45-49", range(45, 50))]
H = os.path.expanduser

# code, dir, family, grp_map expected, twin kind, optional, config expectations (printed only, K7)
CELLS = [
    ("S", "toy_s", "S", None, None, False, {"hide_gripper": None}),
    ("G", "toy_g", "G", "kp100", None, False, {"hide_gripper": None}),
    ("G_T^fc", "toy_g_tfc", "G", "kp100", "fc", False, {"twin_cam": "phys"}),
    ("G_hid", "toy_g_hid", "G", "kp100", None, False, {"hide_gripper": "all"}),
    ("G_T", "toy_g_t", "G", "kp100", "t", False, {"twin_cam": "twin"}),
    ("S_hid", "toy_s_hid", "S", None, None, False, {"hide_gripper": "all"}),
    ("G_cam", "toy_g_cam", "G", "kp100", None, False, {"wrist_cam_dz_mm": 6.4}),  # 裁定 (e): 必跑
]  # G_map20 不跑（裁定 (e)）：toy_g_map20 目录即使存在也不读
SPEC = {c[0]: c for c in CELLS}
DIR2CODE = {c[1]: c[0] for c in CELLS}

# §4 estimands: name -> (A, B) meaning SR(A) - SR(B)
EST = {
    "R_T": ("G_T", "G"),
    "R_fc": ("G_T^fc", "G"),
    "C_P": ("G_T", "G_T^fc"),
    "C_R": ("G_cam", "G"),
    "R_hid": ("G_hid", "G"),
    "H_S": ("S_hid", "S"),
    "D": ("S", "G"),
}
Q0_CELLS = ["G", "G_T^fc", "G_T"]
REQUIRED = [c[0] for c in CELLS if not c[5]]
# 裁定 (a)/(6)：§8.2 非夹爪一致门槛改为 gate json 里 toy_g_tfc 的这一条（gate_toy.py 须用此名；改名就改这里）
GATE_ITEM_ARM = "arm_thirdview_consistency"
Q1_CELLS = ["G", "G_T^fc", "G_hid", "S", "S_hid"]
PROBE_CONDS = ["0", "0p", "x-10", "x-5", "x+5", "x+10", "y-5", "y+5", "cam-5", "cam+5"]


# =============================================================================================== output
class Out:
    def __init__(self):
        self.fh = None

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        if self.fh:
            self.fh.write(s + "\n")
            self.fh.flush()


P = Out()


def sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _s(v):
    v = np.asarray(v)
    return str(v.item() if v.shape == () else v)


def f1(x, nd=2):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


# =============================================================================================== loading
_STAGES = None


def stages_mod(d):
    global _STAGES
    if _STAGES is None:
        try:
            sys.dont_write_bytecode = True
            sys.path.insert(0, H(d))
            import stages as st  # noqa
            _STAGES = (st, st.task_objects())
        except Exception as e:  # noqa
            _STAGES = (None, repr(e))
    return _STAGES


def align_get(al, k):
    if not isinstance(al, dict):
        return "<missing>"
    if k in al:
        return al[k]
    sub = al.get("toy")
    if isinstance(sub, dict) and k in sub:
        return sub[k]
    return "<missing>"


def load_cell(path, keyset, stages_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "t*_e*_i*.npz"))):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)\.npz$", f)
        k = (int(m.group(1)), int(m.group(2)))
        if k not in keyset:
            continue
        z = np.load(f, allow_pickle=True)
        F = set(z.files)
        r = {"succ": float(bool(z["success"])), "init": int(z["init"]) if "init" in F else None,
             "host": _s(z["host"]) if "host" in F else None, "gpu": _s(z["gpu"]) if "gpu" in F else None}
        try:
            r["align"] = json.loads(_s(z["align_json"])) if "align_json" in F else None
        except ValueError:
            r["align"] = None
        r["ref"] = ([str(x) for x in z["ref_body_names"]], np.asarray(z["ref_body_pos"], float)) if "ref_body_pos" in F else None
        tw = {}
        for nm in ("twin_pos_err_mm", "twin_rot_err_deg", "twin_cam_err_mm", "twin_cam_err_deg"):
            if nm in F:
                tw[nm] = np.asarray(z[nm], float).ravel()
        if "twin_stale" in F:
            tw["twin_stale"] = int(np.asarray(z["twin_stale"]).sum())
        r["twin"] = tw or None
        c = {}
        for kk in F:
            if kk.startswith("contact_"):
                v = _s(z[kk]) if z[kk].dtype.kind in "US" else z[kk].item()
                if kk.endswith("_pairs") and isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except ValueError:
                        pass
                c[kk[len("contact_"):]] = v
        r["contacts"] = c if "arm_env_contact_steps" in c else None
        st, objs = stages_mod(stages_dir)
        r["stage"] = None
        if st is not None:
            try:
                s = st.episode_stages(z, *objs[k[0]])
                r["stage"] = s["label"] if s else None
            except Exception:  # noqa
                r["stage"] = None
        out[k] = r
    return out


# =============================================================================================== bootstrap
class Boot:
    def __init__(self, n, B, seed=SEED):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(B, n))
        self.W = np.bincount((idx + np.arange(B)[:, None] * n).ravel(), minlength=B * n).reshape(B, n).astype(np.float64)

    def diff(self, Y, a, b):
        if a not in Y or b not in Y:
            return None
        m = ~np.isnan(Y[a]) & ~np.isnan(Y[b])
        if m.sum() == 0:
            return None
        mf = m.astype(float)
        point = 100 * (Y[a][m].mean() - Y[b][m].mean())
        den = self.W @ mf
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = 100 * ((self.W @ np.where(m, Y[a], 0.0)) - (self.W @ np.where(m, Y[b], 0.0))) / den
        ok = bs[~np.isnan(bs)]
        ya, yb = Y[a][m], Y[b][m]
        nb, nc = int(((ya == 1) & (yb == 0)).sum()), int(((ya == 0) & (yb == 1)).sum())
        return {"point": float(point), "ci95": tuple(np.percentile(ok, [2.5, 97.5])), "ci90": tuple(np.percentile(ok, [5, 95])),
                "bs": bs, "n": int(m.sum()), "mask": m, "cells": [a, b], "b": nb, "c": nc, "mcnemar": mcnemar_exact(nb, nc)}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def ep_boot(vals_fn, n, B, seed=SEED):
    """vals_fn(idx_matrix B x n) -> B statistics; returns 95% percentile CI."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    v = np.asarray(vals_fn(idx), float)
    v = v[~np.isnan(v)]
    return tuple(np.percentile(v, [2.5, 97.5])) if len(v) else (float("nan"), float("nan"))


# =============================================================================================== closed loop
class Toy:
    def __init__(self, a):
        self.keys_mode = a.keys
        self.design = [(t, e) for t in TASKS for e in EPS[a.keys]]
        ks = set(self.design)
        allk = {(t, e) for t in TASKS for e in EPS["all"]}
        self.data, self.raw_keys = {}, {}
        for code, d, *_ in CELLS:
            p = os.path.join(H(a.runs), d)
            if os.path.isdir(p):
                rec = load_cell(p, allk, a.stages_dir)
                self.raw_keys[code] = set(rec)
                rec = {k: v for k, v in rec.items() if k in ks}
                if rec:
                    self.data[code] = rec
        self.gates = {c: {} for c in self.data}
        self.excluded = {}
        self.gatejson = None
        if a.gate:
            try:
                self.gatejson = json.load(open(H(a.gate)))
            except (OSError, ValueError) as e:
                self.gatejson = {"_error": repr(e)}

    # ------------------------------------------------------------------ gates
    def fixcheck(self):
        S = self.data.get("S")
        bad_keys = {}
        for c, rec in self.data.items():
            if c == "S":
                self.gates[c]["场景一致(§8.1)"] = ("NA", "参照格自身")
                continue
            if S is None:
                self.gates[c]["场景一致(§8.1)"] = ("UNK", "toy_s 缺，无法对照")
                continue
            worst, n, nos, nbad = 0.0, 0, 0, 0
            for k, r in rec.items():
                if k not in S or S[k]["ref"] is None or r["ref"] is None:
                    nos += 1
                    continue
                sn, sp = S[k]["ref"]
                sidx = {nm: i for i, nm in enumerate(sn)}
                rn, rp = r["ref"]
                ds = [float(np.abs(rp[i] - sp[sidx[nm]]).max()) * 1000 for i, nm in enumerate(rn) if nm in sidx]
                if len(ds) != len(sn) or len(rn) != len(sn):
                    ds.append(float("inf"))  # body set differs -> treated as inconsistent
                d = max(ds) if ds else float("inf")
                n += 1
                if d >= FIX_TOL_MM:
                    nbad += 1
                    bad_keys.setdefault(k, []).append(f"{c} {d:.4f} mm")
                else:
                    worst = max(worst, d)
            st = "UNK" if (nos or n == 0) else "PASS"
            self.gates[c]["场景一致(§8.1)"] = (st, f"对照 toy_s {n} 键，一致键最大差 {worst:.4f} mm；不一致 {nbad} 键（已从所有格剔除）"
                                                + (f"；{nos} 键无 toy_s 对照（门槛未核实）" if nos else ""))
        self.excluded = bad_keys
        for k in bad_keys:
            for rec in self.data.values():
                rec.pop(k, None)

    def align_gates(self):
        for c, rec in self.data.items():
            code, d, fam, gm, tw, opt, cfg = SPEC[c]
            vals = [align_get(r["align"], "copy_grip_cmd") for r in rec.values()]
            n1 = sum(v in (1, "1", True) for v in vals)
            n0 = sum(v in (0, "0", False) for v in vals)
            nm = len(vals) - n1 - n0
            st = "FAIL" if n0 else ("UNK" if nm else "PASS")
            self.gates[c]["copy_grip_cmd=1(§8.1)"] = (st, f"=1: {n1}, =0: {n0}, 缺/其他: {nm}")
            gv = Counter(str(align_get(r["align"], "grp_map")) for r in rec.values())
            if fam == "G":
                nbad = sum(v for k_, v in gv.items() if k_ not in (gm, "<missing>"))
                nmiss = gv.get("<missing>", 0)
                st = "FAIL" if nbad else ("UNK" if nmiss else "PASS")
                self.gates[c]["grp_map(§8.1)"] = (st, f"应为 {gm}；分布 {dict(gv)}")
            else:
                self.gates[c]["grp_map(§8.1)"] = ("NA", f"S 家族只打印：{dict(gv)}")
            # K7 config (printed only)
            notes = []
            for fld, want in cfg.items():
                got = Counter(str(align_get(r["align"], fld)) for r in rec.values())
                ok = set(got) <= {str(want)} or (want is None and set(got) <= {"None", "<missing>", "", "none"})
                notes.append(f"{fld} {dict(got)}" + ("" if ok else f" <<< 设计为 {want}"))
            self.cfg_notes = getattr(self, "cfg_notes", {})
            self.cfg_notes[c] = "；".join(notes)

    def twin_gates(self):
        for c, rec in self.data.items():
            tw = SPEC[c][4]
            if tw is None:
                continue
            recs = [r["twin"] for r in rec.values() if r["twin"]]
            if tw == "fc":
                if len(recs) < len(rec) or not recs:
                    miss = len(rec) - len(recs)
                    for g in ("替身相机位姿差(§8.2)", "IK 残差>1mm<1%(§8.2)", "陈旧帧 0(§8.2)"):
                        self.gates[c][g] = ("UNK", f"{miss} 集无替身字段")
                    if not recs:
                        continue
                ok_fields = all("twin_cam_err_mm" in t and "twin_cam_err_deg" in t for t in recs)
                if ok_fields:
                    qm = np.concatenate([t["twin_cam_err_mm"][::QUERY_STRIDE] for t in recs])
                    qd = np.concatenate([t["twin_cam_err_deg"][::QUERY_STRIDE] for t in recs])
                    am = max(float(t["twin_cam_err_mm"].max()) for t in recs)
                    ad = max(float(t["twin_cam_err_deg"].max()) for t in recs)
                    nb = int(((qm > CAM_TOL_MM) | (qd > CAM_TOL_DEG)).sum())
                    st = "FAIL" if nb else "PASS"
                    self.gates[c]["替身相机位姿差(§8.2)"] = self.gates[c].get("替身相机位姿差(§8.2)") if self.gates[c].get("替身相机位姿差(§8.2)", ("",))[0] == "UNK" and st == "PASS" else \
                        (st, f"查询帧最大 {qm.max():.4f} mm / {qd.max():.4f}°，超 {CAM_TOL_MM} mm/{CAM_TOL_DEG}° 的查询帧 {nb}/{len(qm)}；全帧最大 {am:.4f} mm / {ad:.4f}°")
                else:
                    self.gates[c]["替身相机位姿差(§8.2)"] = ("UNK", "dump 缺 twin_cam_err_* 字段")
                if all("twin_pos_err_mm" in t for t in recs):
                    pe = np.concatenate([t["twin_pos_err_mm"][::QUERY_STRIDE] for t in recs])
                    nb = int((pe > IK_RES_MM).sum())
                    frac = nb / len(pe)
                    st = "PASS" if frac < IK_FRAC else "FAIL"
                    rot = [t["twin_rot_err_deg"][::QUERY_STRIDE] for t in recs if "twin_rot_err_deg" in t]
                    rtxt = f"；旋转残差查询帧最大 {np.concatenate(rot).max():.3f}°（只印）" if rot else ""
                    prev = self.gates[c].get("IK 残差>1mm<1%(§8.2)")
                    self.gates[c]["IK 残差>1mm<1%(§8.2)"] = prev if (prev and prev[0] == "UNK" and st == "PASS") else \
                        (st, f"查询帧 > {IK_RES_MM} mm：{nb}/{len(pe)} = {100 * frac:.2f}%（门槛 < {100 * IK_FRAC:.0f}%）{rtxt}")
                else:
                    self.gates[c]["IK 残差>1mm<1%(§8.2)"] = ("UNK", "dump 缺 twin_pos_err_mm")
                if all("twin_stale" in t for t in recs):
                    s = sum(t["twin_stale"] for t in recs)
                    prev = self.gates[c].get("陈旧帧 0(§8.2)")
                    self.gates[c]["陈旧帧 0(§8.2)"] = prev if (prev and prev[0] == "UNK" and s == 0) else ("PASS" if s == 0 else "FAIL", f"陈旧帧合计 {s}")
                else:
                    self.gates[c]["陈旧帧 0(§8.2)"] = ("UNK", "dump 缺 twin_stale")
                self.gates[c]["吞吐/第0步一致性(§8.2)"] = ("NA", "由门槛 json 承载")
            else:  # G_T: descriptive only (K5)
                if not recs:
                    self.desc_gt = "G_T 无替身字段"
                    continue
                txt = []
                if all("twin_pos_err_mm" in t for t in recs):
                    pe = np.concatenate([t["twin_pos_err_mm"][::QUERY_STRIDE] for t in recs])
                    txt.append(f"查询帧 pos>1mm {100 * (pe > 1).mean():.2f}%")
                    if all("twin_rot_err_deg" in t for t in recs):
                        re_ = np.concatenate([t["twin_rot_err_deg"][::QUERY_STRIDE] for t in recs])
                        txt.append(f"第二批口径 pos>2mm 或 rot>1° {100 * ((pe > 2) | (re_ > 1)).mean():.2f}%")
                if all("twin_stale" in t for t in recs):
                    txt.append(f"陈旧帧 {sum(t['twin_stale'] for t in recs)}")
                if all("twin_cam_err_mm" in t for t in recs):
                    cm = np.concatenate([t["twin_cam_err_mm"] for t in recs])
                    txt.append(f"相机差均值 {cm.mean():.3f} mm（预期约 6.4）")
                self.desc_gt = "；".join(txt)

    def json_gates(self):
        gj = self.gatejson
        for c in self.data:
            d = SPEC[c][1]
            if gj is None:
                self.gates[c]["gate json"] = ("UNK", "未提供 --gate（t0 一致性、物理不变回放、隐藏清单等未核实）")
                continue
            if "_error" in gj:
                self.gates[c]["gate json"] = ("UNK", f"读取失败 {gj['_error']}")
                continue
            e = (gj.get("cells") or {}).get(d)
            if e is None:
                self.gates[c]["gate json"] = ("UNK", f"门槛 json 无 {d}")
                continue
            items = e.get("items") or {}
            sts = [str(e.get("status", "UNK")).upper()] + [str(v.get("status", "UNK")).upper() for v in items.values()]
            st = "FAIL" if "FAIL" in sts else ("UNK" if any(s not in ("PASS", "NA") for s in sts) else "PASS")
            det = f"cell status {e.get('status')}；" + "，".join(f"{k}={v.get('status')}({v.get('value')})" for k, v in items.items())
            if str(e.get("status", "")).upper() == "PASS" and st != "PASS":
                det += "  [提示：cell status 与 items 不一致，按 items 取严]"
            self.gates[c]["gate json"] = (st, det)
            if c == "G_T^fc":
                it = items.get(GATE_ITEM_ARM)
                if it is None:
                    self.gates[c]["G_T/G_T^fc 手臂与第三视角一致"] = ("UNK", f"门槛 json 无条目 {GATE_ITEM_ARM}")
                else:
                    s_ = str(it.get("status", "UNK")).upper()
                    self.gates[c]["G_T/G_T^fc 手臂与第三视角一致"] = (s_ if s_ in ("PASS", "FAIL") else "UNK", f"{GATE_ITEM_ARM}={it.get('status')}({it.get('value')})")

    def status(self, c):
        if c not in self.data:
            return "MISSING"
        sts = [v[0] for v in self.gates[c].values()]
        return "FAIL" if "FAIL" in sts else ("UNK" if "UNK" in sts else "PASS")

    def shard_complete(self):
        """裁定 (g)：每片（50 键）对全部必需格都齐才可用；在 §8.1 剔除之前按原始数据算。"""
        out = {}
        for nm, rng in SHARDS:
            need = [(t, e) for t in TASKS for e in rng]
            miss = {c: sum(1 for k in need if k not in self.raw_keys.get(c, set())) for c in REQUIRED}
            out[nm] = (all(v == 0 for v in miss.values()), miss)
        return out

    def build_Y(self):
        self.keys = [k for k in self.design if k not in self.excluded]
        self.Y = {c: np.array([rec[k]["succ"] if k in rec else np.nan for k in self.keys]) for c, rec in self.data.items()}

    def gpu_comp(self, c, keys=None):
        rec = self.data.get(c, {})
        ks = keys if keys is not None else list(rec)
        return Counter((rec[k].get("gpu") or "?") for k in ks if k in rec)


def gated(toy, cells, raw):
    st = {c: toy.status(c) for c in dict.fromkeys(cells)}
    miss = [c for c, s in st.items() if s == "MISSING"]
    fail = [c for c, s in st.items() if s == "FAIL"]
    unk = [c for c, s in st.items() if s == "UNK"]
    if miss:
        return "missing", f"缺格 {miss}，不判定"
    if fail:
        return "dnc", f"不判定（does not count：{','.join(fail)} 门槛不过）；按判定线本应为「{raw}」，仅供参考"
    if unk:
        return "unverified", f"{raw}（门槛未核实：{','.join(unk)}）"
    return "ok", raw


def r10_tag(toy, r):
    if r is None or abs(r["point"]) > R10_BAND:
        return ""
    keys = [k for k, ok in zip(toy.keys, r["mask"]) if ok]
    comps = [toy.gpu_comp(c, keys) for c in r["cells"]]
    if any("?" in cp for cp in comps):
        return "  [R10: |效应|<=3，GPU 组成有缺，无法判定]"
    if comps[0] != comps[1]:
        return "  [R10: 在渲染节点噪声带内]"
    return ""


def ci(r):
    return f"95% [{r['ci95'][0]:+7.2f}, {r['ci95'][1]:+7.2f}]  90% [{r['ci90'][0]:+7.2f}, {r['ci90'][1]:+7.2f}]"


# =============================================================================================== probes
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v, abs(n - 1)


def frame_metrics(r):
    R = np.asarray(r["R_base"], float)
    ux, ex = _unit(r["u_x"])
    uy, ey = _unit(r["u_y"])
    A = {c: np.asarray(v, float) for c, v in (r.get("actions") or {}).items()}

    def Pc(c, u, k1=False):
        if c not in A:
            return float("nan")
        a = A[c][:, :3]
        if k1:
            a = a[:1]
        return float(((R @ a.T).T @ u).sum())

    m = {"unit_dev": max(ex, ey)}
    for k1, sfx in ((False, ""), (True, "_k1")):
        m["Dx" + sfx] = Pc("x+5", ux, k1) - Pc("x-5", ux, k1)
        m["Dx10" + sfx] = Pc("x+10", ux, k1) - Pc("x-10", ux, k1)
        m["Dy" + sfx] = Pc("y+5", uy, k1) - Pc("y-5", uy, k1)
        m["Dcam" + sfx] = Pc("cam+5", ux, k1) - Pc("cam-5", ux, k1)
        m["L" + sfx] = Pc("x+5", uy, k1) - Pc("x-5", uy, k1)
    m["N"] = Pc("0p", ux) - Pc("0", ux)
    flips = {}
    if "0" in A:
        g0 = np.sign(A["0"][:, 6])
        for c, a in A.items():
            if c != "0":
                flips[c] = float((np.sign(a[:, 6]) != g0).mean())
    m["flips"] = flips
    return m


def load_jsonl(path):
    rows = []
    for ln in open(H(path)):
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def _app(r):
    return r.get("appearance", r.get("side"))


def probe_checks(app, rows):
    """§8.6 manipulation checks for one appearance -> (gates dict, info lines, noise flag)."""
    g, info = {}, []
    cm = [r.get("cam_err_mm") for r in rows]
    cd = [r.get("cam_err_deg") for r in rows]
    if any(v is None for v in cm + cd):
        g["相机位姿差 0"] = ("UNK", f"{sum(v is None for v in cm)} 帧缺 cam_err")
    else:
        nb = sum((a > PROBE_CAM_MM) or (b > PROBE_CAM_DEG) for a, b in zip(cm, cd))
        g["相机位姿差 0"] = ("FAIL" if nb else "PASS", f"最大 {max(cm):.4f} mm / {max(cd):.4f}°，超限帧 {nb}/{len(rows)}")
    cen = [(r.get("centroid_px") or {}) for r in rows]
    if any("x+5" not in c for c in cen):
        g["质心位移 5-12 px"] = ("UNK", f"{sum('x+5' not in c for c in cen)} 帧缺 centroid_px[x+5]")
    else:
        mag = np.array([np.linalg.norm(c["x+5"]) for c in cen])
        nb = int(((mag < CENTROID_PX[0]) | (mag > CENTROID_PX[1])).sum())
        g["质心位移 5-12 px"] = ("FAIL" if nb else "PASS", f"x+5 质心位移 中位 {np.median(mag):.2f} px [{mag.min():.2f}, {mag.max():.2f}]，"
                                                          f"超出 {CENTROID_PX[0]:.0f}-{CENTROID_PX[1]:.0f} 的帧 {nb}/{len(rows)}")
    g["质心方向与 u 投影一致"] = ("UNK", "jsonl 无相机投影，脱机核不了（K15）")
    opp = [float(np.dot(c["x+5"], c["x-5"]) / (np.linalg.norm(c["x+5"]) * np.linalg.norm(c["x-5"]) + 1e-12)) < 0
           for c in cen if "x+5" in c and "x-5" in c]
    info.append(f"代理量：x+5 与 x−5 质心位移反向的帧 {sum(opp)}/{len(opp)}")
    diff = [r for r in rows if not ((r.get("same_0_0p") or {}).get("pixels", False) and (r.get("same_0_0p") or {}).get("actions", False))]
    noise = len(diff) > 0
    info.append(f"0 与 0′ 不同的帧 {len(diff)}/{len(rows)}" + ("（记噪声底，见下；§8.6）" if noise else "（逐像素、逐动作相同）"))
    eps = defaultdict(list)
    for r in rows:
        eps[(int(r["task"]), int(r["ep"]))].append(r)
    per_task = Counter(k[0] for k in eps)
    ph = Counter(r.get("phase") for r in rows)
    bad_n = [k for k, v in eps.items() if len(v) != PROBE_FRAMES]
    bad_ph = [k for k, v in eps.items() if Counter(x.get("phase") for x in v) != Counter({"approach": PROBE_APPROACH, "pregrasp": PROBE_PREGRASP})]
    info.append(f"集 {len(eps)}（设计 {PROBE_EPISODES}）；每任务 {dict(sorted(per_task.items()))}；帧 {len(rows)}；phase {dict(ph)}")
    if len(eps) != PROBE_EPISODES or any(v != PROBE_EP_PER_TASK for v in per_task.values()) or len(per_task) != 10:
        info.append("  警告：集数或每任务集数不是设计值（只警告，不剔除，K15）")
    if bad_n or bad_ph:
        info.append(f"  警告：帧数不是 {PROBE_FRAMES} 的集 {len(bad_n)}，phase 不是 {PROBE_APPROACH}+{PROBE_PREGRASP} 的集 {len(bad_ph)}")
    return g, info, noise


def pair_checks(by):
    """裁定 (f)：两外观在同一批 Robotiq G 帧上配对 -> 共享 (task, ep, step) 集合；帧来源由 ep 推断、不混用。"""
    g = {}
    ks = {a: {(int(r["task"]), int(r["ep"]), int(r["step"])) for r in by.get(a, [])} for a in ("panda", "robotiq")}
    if not ks["panda"] or not ks["robotiq"]:
        g["两外观同帧配对"] = ("UNK", "缺一种外观")
    else:
        only_p, only_r = ks["panda"] - ks["robotiq"], ks["robotiq"] - ks["panda"]
        g["两外观同帧配对"] = ("FAIL" if (only_p or only_r) else "PASS",
                               f"共同帧 {len(ks['panda'] & ks['robotiq'])}，仅 panda {len(only_p)}，仅 robotiq {len(only_r)}")
    eps = {k[1] for a in ks for k in ks[a]}
    if not eps:
        g["帧来源不混用"] = ("UNK", "无帧")
    elif all(40 <= e <= 49 for e in eps):
        g["帧来源不混用"] = ("PASS", "toy G（ep 40-49）")
    elif all(0 <= e <= 19 for e in eps):
        g["帧来源不混用"] = ("PASS", "第二批 G（ep 0-19）")
    else:
        g["帧来源不混用"] = ("FAIL", f"混用：ep {sorted(eps)}")
    return g


def side_stats(rows, B):
    eps = defaultdict(list)
    for r in rows:
        eps[(int(r["task"]), int(r["ep"]))].append(frame_metrics(r))
    keys = sorted(eps)
    names = ["Dx", "Dx10", "Dy", "Dcam", "L", "Dx_k1", "Dx10_k1", "Dy_k1", "Dcam_k1", "L_k1", "N"]
    E = {nm: np.array([np.nanmedian([f[nm] for f in eps[k]]) if not all(math.isnan(f[nm]) for f in eps[k]) else np.nan
                       for k in keys]) for nm in names}
    flips = defaultdict(list)
    for k in keys:
        for f in eps[k]:
            for c, v in f["flips"].items():
                flips[c].append(v)
    unit_dev = max(f["unit_dev"] for k in keys for f in eps[k])
    return keys, E, {c: float(np.mean(v)) for c, v in flips.items()}, unit_dev


def judge_side(E):
    dx = E["Dx"]
    frac = float(np.mean(dx < 0))
    mdx, ml = float(np.nanmedian(np.abs(dx))), float(np.nanmedian(np.abs(E["L"])))
    dose = float(np.mean(np.abs(E["Dx10"]) > np.abs(dx)))
    det = frac >= SIGN_PRESENT and mdx >= SPEC_RATIO * ml and dose >= DOSE_FRAC
    return {"label": "检出" if det else "未检出", "frac": frac, "mdx": mdx, "ml": ml, "dose": dose,
            "mcam": float(np.nanmedian(np.abs(E["Dcam"])))}


PROBE_CAVEAT = "（过线也不能排除读数偏置，更不能证明效应来自指尖；未检出不等于不存在。裁定 (f)）"


def probe_shift(path, B):
    out = {"panda": None, "robotiq": None, "reading": None, "row": None}
    P("\n==== §5.3 Q2 横移探针（同一批 Robotiq G 帧上两种外观配对；按集聚合，bootstrap 按集重采样）")
    if not path:
        P("  缺（未提供 --probe-shift）")
        return out
    rows = load_jsonl(path)
    by = defaultdict(list)
    for r in rows:
        by[_app(r)].append(r)
    pg = pair_checks(by)
    for k, (s, d) in pg.items():
        P(f"  配对检查 {k:14s} {s:5s} {d}")
    res = {}
    for app in ("panda", "robotiq"):
        rs = by.get(app, [])
        P(f"\n  -- 外观 {app}")
        if not rs:
            P("    缺（无该外观帧）")
            res[app] = None
            continue
        g, info, noise = probe_checks(app, rs)
        g.update(pg)
        for k, (s, d) in g.items():
            if k not in pg:
                P(f"    操纵检查 {k:18s} {s:5s} {d}")
        for ln in info:
            P("    " + ln)
        keys, E, flips, udev = side_stats(rs, B)
        j = judge_side(E)
        n = len(keys)
        if udev > 1e-3:
            P(f"    提示：u 偏离单位长度最大 {udev:.4f}（已单位化）")
        dx, ldx, dx10 = E["Dx"], E["L"], E["Dx10"]
        ci_frac = ep_boot(lambda I: (dx[I] < 0).mean(1), n, B)
        ci_mdx = ep_boot(lambda I: np.nanmedian(np.abs(dx[I]), 1), n, B)
        ci_rl = ep_boot(lambda I: np.nanmedian(np.abs(dx[I]), 1) / np.nanmedian(np.abs(ldx[I]), 1), n, B)
        ci_dose = ep_boot(lambda I: (np.abs(dx10[I]) > np.abs(dx[I])).mean(1), n, B)
        P(f"    集 {n}；D_x 预测符号（<0）集比例 {j['frac']:.3f} 95% [{ci_frac[0]:.3f}, {ci_frac[1]:.3f}]（需 >= {SIGN_PRESENT}）")
        P(f"    median|D_x| {j['mdx']:.4f} 95% [{ci_mdx[0]:.4f}, {ci_mdx[1]:.4f}]；median|L| {j['ml']:.4f}；"
          f"|D_x|/|L| {j['mdx'] / j['ml'] if j['ml'] else float('inf'):.3f} 95% [{ci_rl[0]:.3f}, {ci_rl[1]:.3f}]（需 >= {SPEC_RATIO}）")
        P(f"    剂量单调 |D_x10|>|D_x| 集比例 {j['dose']:.3f} 95% [{ci_dose[0]:.3f}, {ci_dose[1]:.3f}]（需 >= {DOSE_FRAC}，必要条件）")
        P(f"    描述（不进判定）：median|D_cam| {j['mcam']:.4f}（相机平移有视差，不是阴性 sham）；"
          f"D_y 预测符号集比例 {float(np.mean(E['Dy'] < 0)):.3f}，median|D_y| {float(np.nanmedian(np.abs(E['Dy']))):.4f}")
        P(f"    描述：噪声底 median|P(0′)−P(0)| {float(np.nanmedian(np.abs(E['N']))):.4f}" + ("" if noise else "（0/0′ 逐位相同）"))
        P(f"    副量 k=1：D_x 符号比例 {float(np.mean(E['Dx_k1'] < 0)):.3f}，median|D_x| {float(np.nanmedian(np.abs(E['Dx_k1']))):.4f}，"
          f"median|L| {float(np.nanmedian(np.abs(E['L_k1']))):.4f}，median|D_cam| {float(np.nanmedian(np.abs(E['Dcam_k1']))):.4f}")
        P("    夹爪符号翻转率（相对条件 0）：" + "，".join(f"{c} {v:.3f}" for c, v in sorted(flips.items())))
        bad = [k for k, (s, _) in g.items() if s == "FAIL"]
        unk = [k for k, (s, _) in g.items() if s == "UNK"]
        if bad:
            j["state"] = "dnc"
            P(f"    判定：不判定（检查不过：{','.join(bad)}）；按判定线本应为「{j['label']}」，仅供参考")
        else:
            j["state"] = "unverified" if unk else "ok"
            P(f"    判定：标尺式补偿{j['label']}" + (f"（检查未核实：{','.join(unk)}）" if unk else "") + PROBE_CAVEAT)
        res[app] = j
    pj, rj = res.get("panda"), res.get("robotiq")
    plab = pj["label"] if (pj and pj["state"] != "dnc") else None
    rlab = rj["label"] if (rj and rj["state"] != "dnc") else None
    out["panda"], out["robotiq"] = plab, rlab
    reading = None
    if plab == "检出":
        if rlab == "未检出":
            reading = "失效"
        elif rlab == "检出":
            reading = "读得懂" if rj["mdx"] >= ROBOTIQ_HALF * pj["mdx"] else "未定义"
        else:
            reading = "缺"
    out["reading"] = reading
    P("\n  -- Robotiq 外观上补偿还在不在（只在 Panda 外观「检出」时有意义）")
    if plab == "检出":
        rtxt = {"失效": "Robotiq 外观上未检出补偿（与 H_ruler「尺子在 Robotiq 上失效」相容）",
                "读得懂": "Robotiq 外观上也检出、幅度不低于 Panda 外观一半（Robotiq 掉分不像读数失效）",
                "缺": "Robotiq 外观缺或不判定",
                "未定义": "Robotiq 外观检出但 median|D_x| 不到 Panda 外观一半：协议未定义，需人工"}[reading]
        extra = f"（Robotiq/Panda median|D_x| = {rj['mdx'] / pj['mdx']:.3f}）" if (rj and pj and pj["mdx"]) else ""
        P(f"    {rtxt}{extra}")
    else:
        P(f"    Panda 外观 = {plab or '缺/不判定'}，此项无意义")
    row = None
    if plab == "未检出":
        row = 1
    elif plab == "检出" and reading == "失效":
        row = 2
    elif plab == "检出" and reading == "读得懂":
        row = 3
    out["row"] = row
    ROWS = {1: "Panda 外观未检出标尺式补偿；Q1 若为「线索」，本探针不支持标尺解读（未检出不等于不存在）",
            2: "Panda 外观检出、Robotiq 外观未检出：与 H_ruler 相容；全量分支 B 以指尖几何为主",
            3: "两外观都检出：模型会按画面补偿，但 Robotiq 掉分不像读数失效，回到 Q1 的干扰 / 模板解释"}
    P(f"  解读表：Panda {plab or '缺'} / Robotiq {rlab or '缺'} -> " + (f"行 {row}：{ROWS[row]}" if row else "不落行（缺、不判定或协议未定义），需人工"))
    P("  " + PROBE_CAVEAT)
    return out


def probe_parts(path, B):
    out = {}
    P("\n==== §5.4 Q3 部件探针（描述性，r_i = d(ci,c1)/d(c0,c1)，集内中位数 -> 集间中位数）")
    if not path:
        P("  缺（未提供 --probe-parts）")
        return out
    rows = load_jsonl(path)
    eps = defaultdict(lambda: defaultdict(list))
    num = defaultdict(float)
    den = 0.0
    skipped = 0

    def d(a, b):
        return float(np.linalg.norm(np.asarray(a, float)[:, :3] - np.asarray(b, float)[:, :3], axis=1).mean())

    for r in rows:
        A = r["actions"]
        if "c0" not in A or "c1" not in A:
            skipped += 1
            continue
        d01 = d(A["c0"], A["c1"])
        if d01 == 0:
            skipped += 1
            continue
        den += d01
        for i in range(2, 7):
            c = f"c{i}"
            if c in A:
                di = d(A[c], A["c1"])
                eps[(int(r["task"]), int(r["ep"]))][c].append(di / d01)
                num[c] += di
    keys = sorted(eps)
    P(f"  帧 {len(rows)}，集 {len(keys)}，跳过帧（缺 c0/c1 或 d(c0,c1)=0）{skipped}；c5/c6 的合成图与覆盖像素计数请看 probe_parts.py 输出（K17）")
    for i in range(2, 7):
        c = f"c{i}"
        v = np.array([np.median(eps[k][c]) for k in keys if eps[k].get(c)])
        if len(v) == 0:
            P(f"  r_{i}: 缺" + ("（c5/c6 已改为必做，裁定 (a)：门槛未核实；仍只作描述）" if i >= 5 else ""))
            out[c] = None
            continue
        med = float(np.median(v))
        lo, hi = ep_boot(lambda I: np.median(v[I], 1), len(v), B)
        out[c] = med
        P(f"  r_{i}: 中位 {med:.3f} 95% [{lo:.3f}, {hi:.3f}]，均值 {v.mean():.3f}，比值均值口径 {num[c] / den:.3f}，集 {len(v)}")
    rd = {}
    r2, r3, r4, r5, r6 = (out.get(f"c{i}") for i in range(2, 7))
    rd["r2"] = None if r2 is None else ("与 Q1 干扰一致" if r2 <= R2_INTERF else "与 Q1 干扰不一致")
    rd["r3_r4"] = None if (r3 is None or r4 is None) else ("c3（指节/两指/指垫）为主要干扰部位" if r3 < r4 else ("c4（基座/联轴器）为主要干扰部位" if r4 < r3 else "c3、c4 相等"))
    if r5 is None or r6 is None:
        rd["c5_c6"] = None
    elif r5 <= PART_FINGER_LO and r6 >= PART_FINGER_HI:
        rd["c5_c6"] = "手指承担效应"
    elif PART_MIX[0] <= r5 <= PART_MIX[1] and PART_MIX[0] <= r6 <= PART_MIX[1]:
        rd["c5_c6"] = "部件不可分，全量不做部件闭环格"
    else:
        rd["c5_c6"] = "无事先读法"
    P("  读法（只用于选全量格，不是确认）：")
    P(f"    r_2 <= {R2_INTERF}：{rd['r2'] or '缺'}")
    P(f"    r_3 对 r_4：{rd['r3_r4'] or '缺'}")
    P(f"    c5/c6：{rd['c5_c6'] or '缺（门槛未核实）'}（c5、c6 不进任何判定；闭环充分性留待全量）")
    out["readings"] = rd
    return out


# =============================================================================================== descriptive
def describe(toy):
    P("\n==== 描述：每格 10 任务表、conc（相对 S）、失败类型（stages.py）、两片 GPU 型号组成")
    S = toy.data.get("S")
    for code, d, *_ in CELLS:
        if code not in toy.data:
            P(f"  {code:8s} ({d}) 缺")
            continue
        rec = toy.data[code]
        pt = {t: (int(sum(r["succ"] for k, r in rec.items() if k[0] == t)), sum(1 for k in rec if k[0] == t)) for t in TASKS}
        k_, n_ = sum(v[0] for v in pt.values()), sum(v[1] for v in pt.values())
        conc = "-"
        if S is not None and code != "S":
            net = {t: 0 for t in TASKS}
            for k in set(rec) & set(S):
                net[k[0]] += (1 - rec[k]["succ"]) - (1 - S[k]["succ"])
            tot = sum(net.values())
            if tot <= 0:
                conc = f"无净增失败（净 {tot:+.0f}）"
            else:
                w = max(net, key=lambda t: net[t])
                conc = f"{net[w] / tot:.2f}（最差任务 {w}，{net[w]:+.0f}/{tot:+.0f}）"
        P(f"  {code:8s} SR {100 * k_ / max(n_, 1):5.1f}% ({k_}/{n_}) | " + " ".join(f"t{t}:{a}/{b}" for t, (a, b) in pt.items()) + f" | conc {conc}")
        labs = Counter(r["stage"] for r in rec.values() if r["stage"] not in ("success",))
        P(f"           失败类型 {dict(labs)}")
        P("           GPU " + "；".join(f"{nm}: " + (", ".join(f"{g} x{m}" for g, m in sorted(toy.gpu_comp(code, [(t, e) for t in TASKS for e in rng]).items())) or "无")
                                       for nm, rng in SHARDS))
        cs = [r["contacts"] for r in rec.values() if r.get("contacts")]
        if cs:
            ce = sum(1 for x in cs if int(x.get("arm_env_contact_steps") or 0) > 0)
            stall = sum(1 for x in cs if int(x.get("stall_steps") or 0) > 0)
            pc = Counter()
            for x in cs:
                prs = x.get("pairs") or {}
                if isinstance(prs, dict):
                    pc.update(prs.keys())
            P(f"           在线接触：臂-环境接触集 {ce}/{len(cs)}，停滞集 {stall}，pairs 前三 {pc.most_common(3)}；缺接触字段 {len(rec) - len(cs)} 集")
        else:
            P(f"           在线接触：无字段")
    if _STAGES and _STAGES[0] is None:
        P(f"  stages.py 导入失败：{_STAGES[1]}")


# =============================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-protocol-sha", required=True)
    ap.add_argument("--keys", choices=["first", "all"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdict-json", default=None)
    ap.add_argument("--runs", default="$PROJ_ROOT/runs_toy")
    ap.add_argument("--gate", default=None)
    ap.add_argument("--probe-shift", default=None)
    ap.add_argument("--probe-parts", default=None)
    ap.add_argument("--protocol", default="$PROJ_ROOT/embody_toy/protocol_toy1.md")
    ap.add_argument("--decisions", default="$PROJ_ROOT/embody_toy/protocol_toy1_astra_decisions.md")
    ap.add_argument("--expect-decisions-sha", required=True)
    ap.add_argument("--stages-dir", default="$PROJ_ROOT/embody/analysis")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    a = ap.parse_args()

    psha = sha256(H(a.protocol))
    ssha = sha256(os.path.abspath(__file__))
    if psha is None:
        sys.stderr.write(f"拒绝运行：协议文件不存在 {a.protocol}\n")
        sys.exit(2)
    if psha != a.expect_protocol_sha.strip().lower():
        sys.stderr.write(f"拒绝运行：协议 sha256 {psha} != --expect-protocol-sha {a.expect_protocol_sha}\n")
        sys.exit(2)
    dsha = sha256(H(a.decisions))
    if dsha is None or dsha != a.expect_decisions_sha.strip().lower():
        sys.stderr.write(f"拒绝运行：裁定文件 sha256 {dsha} != --expect-decisions-sha {a.expect_decisions_sha}\n")
        sys.exit(2)
    for f in (a.out, a.verdict_json):
        if f and os.path.exists(H(f)):
            sys.stderr.write(f"拒绝覆盖：{f} 已存在（§6 只跑一次）\n")
            sys.exit(3)
    P.fh = open(H(a.out), "x")
    P(f"toy1 预注册分析（protocol_toy1.md）  --keys {a.keys}  B={a.B} seed={SEED}")
    P(f"协议 sha256 {psha}（= --expect-protocol-sha）  {a.protocol}")
    P(f"裁定 sha256 {dsha}（= --expect-decisions-sha）  {a.decisions}")
    P(f"脚本 sha256 {ssha}  {os.path.abspath(__file__)}")
    P(f"runs {a.runs}  gate {a.gate}  probe-shift {a.probe_shift}  probe-parts {a.probe_parts}")

    toy = Toy(a)
    toy.fixcheck()
    toy.align_gates()
    toy.twin_gates()
    toy.json_gates()
    toy.build_Y()
    V = {"keys": a.keys, "gates": {}, "n": {}}

    # ---------------- gates
    P(f"\n==== 门槛（§8；本脚本自算项 + 门槛 json 合并，K2-K7）  设计键 {len(toy.design)}，§8.1 剔除 {len(toy.excluded)}")
    for k, v in sorted(toy.excluded.items()):
        P(f"  剔除键 t{k[0]} e{k[1]}: {'; '.join(v)}")
    for code, d, fam, gm, tw, opt, cfg in CELLS:
        if code not in toy.data:
            P(f"  {code:8s} ({d}): 缺" + ("（可选格未跑）" if opt else "  <<< 必需格无数据"))
            V["gates"][code] = "MISSING"
            continue
        rec = toy.data[code]
        missing = [k for k in toy.keys if k not in rec]
        st = toy.status(code)
        V["gates"][code] = st
        tag = {"FAIL": "  <<< does not count", "UNK": "  （门槛未核实）"}.get(st, "")
        mt = "" if not missing else f"；缺键 {len(missing)}：" + " ".join(f"t{t}e{e}" for t, e in missing[:20]) + (" ..." if len(missing) > 20 else "")
        P(f"  {code:8s} ({d}): 键 {len(rec)}/{len(toy.keys)}{mt}  状态 {st}{tag}")
        for gk, (s, det) in toy.gates[code].items():
            P(f"      {gk:24s} {s:5s} {det}")
        P(f"      配置核对（只印，K7）     {toy.cfg_notes.get(code, '')}")
        if code == "G_T":
            P(f"      G_T 替身量（描述，K5）  {getattr(toy, 'desc_gt', '无')}")
    sc = toy.shard_complete()
    for nm, rng in SHARDS:
        ok, miss = sc[nm]
        P(f"  片 {nm}：{'完成' if ok else '未完成'}；必需格缺集数 " + "，".join(f"{c} {n}" for c, n in miss.items()))
    use = ["ep40-44"] if a.keys == "first" else ["ep40-44", "ep45-49"]
    complete = all(sc[nm][0] for nm in use)
    V["shards"] = {nm: sc[nm][0] for nm in sc}
    V["complete"] = complete

    q0 = q1 = None
    if not complete:
        P(f"\n==== 未完成：--keys {a.keys} 需要的片 {use} 有必需格缺集（裁定 (g)：只用全格共同完成的预定 50 键整片，不拼接半片）。"
          "不计算、不打印任何闭环估计量与 Q0/Q1 判定。")
        V["pre"] = V["q0"] = V["q1"] = {"raw": None, "state": "incomplete"}
    else:
        q0, q1 = closed_loop(toy, a, V)

    # ---------------- probes（探索性，§6 可边出边看；与闭环是否完成无关）
    V["q2"] = probe_shift(a.probe_shift, a.B)
    V["q3"] = probe_parts(a.probe_parts, a.B)
    if complete:
        describe(toy)

    # ---------------- summary
    P("\n==== 摘要与 §5.5 全量分支（路线筛选，不是机制确认）")
    P(f"  §5.0（描述）{V['pre']}；Q0 {V['q0']}；Q1 {V['q1']}")
    P(f"  Q2 Panda 外观 {V['q2'].get('panda')} / Robotiq 外观 {V['q2'].get('robotiq')} / 读法 {V['q2'].get('reading')} / 解读表行 {V['q2'].get('row')}")
    if not complete:
        br = "未完成：不出分支"
    elif q0 == "转向相机":
        br = "相机转向：C_P、G_cam 加 200 键，相机位姿剂量 ±3/±6.4/±13 mm 各 100 键"
    elif q0 == "未检出":
        br = "Q0 未检出：不写外观为主、不据此扩量（裁定 (f)），需人工定下一步"
    elif q0 == "不确定":
        br = "Q0 不确定（相机分支 C_P 下界未过）：需人工定下一步"
    elif q0 != "GO":
        br = "Q0 不判定（门槛或缺格），无分支"
    elif q1 == "干扰为主":
        br = "路线筛选：分支 A（擦除路线优先）；不能推出夹爪不必要或擦除足够"
    elif q1 == "线索为主":
        br = "路线筛选：分支 B（重绘路线优先），由 Q2 区分标尺与模板；不能推出夹爪必要"
    elif q1 == "混合/不确定":
        br = "Q1 混合：由 Q2、Q3 的结果按 §5.5 人工定分支；若也不确定走分支 C（K11）"
    else:
        br = "Q1 不判定（门槛或缺格），无分支"
    P(f"  全量分支：{br}")
    V["branch"] = br
    if a.verdict_json:
        with open(H(a.verdict_json), "x") as f:
            json.dump(V, f, ensure_ascii=False, indent=1, default=str)
    P.fh.close()


def closed_loop(toy, a, V):
    boot = Boot(len(toy.keys), a.B)
    res = {nm: boot.diff(toy.Y, A, Bc) for nm, (A, Bc) in EST.items()}
    P(f"\n==== §4 估计量（pp；配对聚类 bootstrap，{len(toy.keys)} 键联合重采样）：点估计、95% / 90% CI、键数、离散键 b(A成B败)/c(A败B成)、McNemar 精确 p")
    for nm, (A, Bc) in EST.items():
        r = res[nm]
        if r is None:
            miss = [c for c in (A, Bc) if c not in toy.data]
            P(f"  {nm:6s} = SR({A}) − SR({Bc})  缺（{'无数据格 ' + ','.join(miss) if miss else '无共同键'}）")
            V["n"][nm] = None
            continue
        st, _ = gated(toy, [A, Bc], "")
        tag = {"dnc": "  <<< does not count", "unverified": "  （门槛未核实）"}.get(st, "")
        P(f"  {nm:6s} = SR({A}) − SR({Bc})  {r['point']:+7.2f}  {ci(r)}  n={r['n']}  b={r['b']} c={r['c']} McNemar p={r['mcnemar']:.4g}{tag}{r10_tag(toy, r)}")
        V["n"][nm] = r["n"]
    V["est"] = {nm: (None if r is None else {"point": r["point"], "ci95": list(r["ci95"])}) for nm, r in res.items()}

    # §5.0 -> 描述（裁定 (f) 删除「R_T 不足禁止后续判定」）
    P("\n==== §5.0 R_T（描述；裁定 (f)：不再作后续判定的前置；本批 G_T 是新配置，不称严格复现旧 36 点）")
    rt = res["R_T"]
    if rt is None:
        P("  R_T 缺")
        V["pre"] = {"raw": None, "state": "missing"}
    else:
        lo, hi = rt["ci95"]
        P(f"  R_T {rt['point']:+.2f} 95% [{lo:+.2f}, {hi:+.2f}]（原 §5.0 线：点 >= {RT_MIN} 且 CI 不含 0 -> "
          f"{'过' if (rt['point'] >= RT_MIN and not (lo <= 0 <= hi)) else '不过'}；只印）")
        V["pre"] = {"raw": rt["point"], "state": gated(toy, EST["R_T"], "")[0]}

    # §5.1
    P("\n==== §5.1 Q0：固定相机重绘效应（R_fc）")
    rf, cp = res["R_fc"], res["C_P"]
    q0 = None
    if rf is None:
        P("  R_fc 缺")
        V["q0"] = {"raw": None, "state": "missing"}
    else:
        lo, hi = rf["ci95"]
        if rf["point"] >= GO_POINT and lo > GO_LO95:
            raw = "GO"
        elif hi < CAM_HI95 and rt is not None and rt["point"] >= CAM_RT_MIN:
            raw = "转向相机" if (cp is not None and cp["ci95"][0] > CAM_CP_LO95) else "不确定"
        else:
            raw = "未检出"
        st, txt = gated(toy, Q0_CELLS, raw)
        P(f"  R_fc {rf['point']:+.2f} 95% [{lo:+.2f}, {hi:+.2f}]；GO: 点 >= {GO_POINT} 且下界 > {GO_LO95}；"
          f"转向相机: 上界 < {CAM_HI95} 且 R_T >= {CAM_RT_MIN} 且 C_P 下界 > {CAM_CP_LO95}（C_P 未过则不确定） -> {txt}")
        if cp is not None:
            P(f"  C_P {cp['point']:+.2f} 95% [{cp['ci95'][0]:+.2f}, {cp['ci95'][1]:+.2f}]")
        V["q0"] = {"raw": raw, "state": st}
        if st in ("ok", "unverified"):
            q0 = raw
            if raw == "GO":
                P("  固定相机重绘效应检出（G_T^fc 还改变了手臂像素，不是纯夹爪外观；裁定 (a)）。")
            elif raw == "转向相机":
                P("  方向一按原题目终止，替换为「腕相机位姿毫米级敏感性」作机制对照；不再做部件实验。")
            elif raw == "未检出":
                P("  固定相机重绘效应未检出（5 点是效应阈值，不是已验证的噪声上界）。")
    cr = res["C_R"]
    if cp:
        P(f"  描述：C_P {cp['point']:+.2f}（原预期 |C_P| <= {CP_EXPECT}，只印数；不显著不等于 ±5 等价）"
          + (f"；|C_P| >= {CP_TRIGGER}：§2.2 触发语——OFT 对毫米级腕相机 / 指长几何敏感" if abs(cp["point"]) >= CP_TRIGGER else ""))
    if cr is not None and cp is not None:
        P(f"  描述：C_R {cr['point']:+.2f} 95% [{cr['ci95'][0]:+.2f}, {cr['ci95'][1]:+.2f}]，与 C_P {'同号' if np.sign(cr['point']) == np.sign(cp['point']) else '不同号'}，"
          f"C_R/C_P = {cr['point'] / cp['point'] if cp['point'] else float('nan'):.2f}（2x2：相机效应是否依赖外观，只印）")
    elif cr is None:
        P("  描述：C_R 缺（G_cam 为必需格，缺即门槛问题）")

    # §5.2
    P("\n==== §5.2 Q1：路线筛选（擦除 vs 重绘；只在 Q0 为 GO 时判；不能推出夹爪必要或擦除足够）")
    hs, rh = res["H_S"], res["R_hid"]
    q1 = None
    if q0 != "GO":
        P(f"  不判（Q0 = {q0 or V['q0'].get('state')}）；§4 规定 ρ 只在 Q0 GO 时算，此处不算 ρ")
        V["q1"] = {"raw": None, "state": "not_entered"}
    elif rh is None or hs is None:
        P("  缺（R_hid 或 H_S 无数据）")
        V["q1"] = {"raw": None, "state": "missing"}
    else:
        rho = rh["point"] / rf["point"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = rh["bs"] / rf["bs"]
        ok = ~np.isnan(rb) & np.isfinite(rb)
        nonpos = float((rf["bs"] <= 0).mean())
        rci95 = tuple(np.percentile(rb[ok], [2.5, 97.5]))
        rci90 = tuple(np.percentile(rb[ok], [5, 95]))
        hlo, hhi = hs["ci95"]
        if rho >= RHO_INTERF and hlo > HS_INTERF_LO95:
            raw = "干扰为主"
        elif rho <= RHO_CUE or hhi < HS_CUE_HI95:
            raw = "线索为主"
        else:
            raw = "混合/不确定"
        st, txt = gated(toy, Q1_CELLS, raw)
        P(f"  ρ = R_hid/R_fc = {rh['point']:+.2f}/{rf['point']:+.2f} = {rho:.3f}  95% [{rci95[0]:.3f}, {rci95[1]:.3f}]  90% [{rci90[0]:.3f}, {rci90[1]:.3f}]"
          f"（R_fc* <= 0 的重采样 {100 * nonpos:.2f}%，照算，K10）" + (f"  [R_hid 键 {rh['n']} 与 R_fc 键 {rf['n']} 不同]" if rh['n'] != rf['n'] else ""))
        P(f"  H_S {hs['point']:+.2f} 95% [{hlo:+.2f}, {hhi:+.2f}]")
        P(f"  线：干扰 = ρ >= {RHO_INTERF} 且 H_S 下界 > {HS_INTERF_LO95}；线索 = ρ <= {RHO_CUE} 或 H_S 上界 < {HS_CUE_HI95} -> 路线筛选：{txt}")
        P("  注：这是路线筛选，不能推出夹爪必要或擦除足够（裁定 (f)）；G_hid 改变遮挡，不是无伪影（裁定 (a)）。")
        V["q1"] = {"raw": raw, "state": st, "rho": rho, "rho_ci95": list(rci95)}
        if st in ("ok", "unverified"):
            q1 = raw
        if RHO_CUE < rho < RHO_INTERF:
            P("  警示（§5.2 事先写定）：ρ 落在 0.3 到 0.7，阴影是未控因素（G_hid 去掉了 Robotiq 阴影，S_hid 去掉了 Panda 阴影）。")
        P("  限制（§5.2）：G_hid 第三人称画面里手臂末端悬空无夹爪、G 的物理臂比 S 后退 48 mm，与 G_T^fc 不同。")
    rd = res["D"]
    if rd:
        P(f"\n  描述：D = SR(S) − SR(G) {rd['point']:+.2f}（toy init 上的 kp100 掉分）")
    return q0, q1


if __name__ == "__main__":
    main()
