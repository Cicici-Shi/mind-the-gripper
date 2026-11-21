"""[prereg_models1_sens.py：物理失效集剔除版（Fable 补裁一 §2.3 第 2 条），由 make_prereg_models1_sens.py 从冻结版断言替换生成，diff 见 prereg_models1_sens.diff]
models1 pre-registered analysis: protocol_models1.md (+ protocol_models1_decisions.md).
§4 estimands, §5 decisions (R1-R3, F1-F2, O1-O2, E1-E5) with the protocol's fixed interpretation sentences, §7.1 slice
rule and gates, §7.2 twin gates, §7.3 d5 branch (rec' rows + warning), Holm appendix, 10-task tables, GPU composition,
online contacts, E1/E2 agreement, Delta_model paired bootstrap against the published OFT / pi0.5 cells.
Frozen before the first full slice; run ONCE at the read-out (§8).

Usage (cluster, conda env oft, CPU only, e.g. sbatch -p normal -c 4 --mem 16G):
  python prereg_models1.py --out results_models1.txt [--verdict-json v.json] [--B 10000]
      [--zoo-runs $PROJ_ROOT/vla_zoo/runs] [--pub-runs $PROJ_ROOT/runs] [--pi05 $PROJ_ROOT/pi05/v2]
      [--zoo-logs $PROJ_ROOT/vla_zoo/logs] [--gate-json zoo_gate.json]
      [--protocol $PROJ_ROOT/vla_zoo/analysis/protocol_models1.md] [--decisions .../protocol_models1_decisions.md]
      [--zoo-cells $PROJ_ROOT/vla_zoo/tools/zoo_cells.sh] [--published-only] [--overwrite]
  --published-only: recompute only the published OFT / pi0.5 numbers (no vla_zoo data is read).
Tests: test_prereg_models1.py (synthetic data only).

Inputs
  OFTi   <zoo-runs>/oft_sp/<cell>/t{task}_e{ep}_i{init}.npz, object <zoo-runs>/oft_obj/obj_<cell>/...
  pi0    <zoo-runs>/pi0/<cell>_e<off>.jsonl (object: obj_<cell>), OVLA <zoo-runs>/openvla/<cell>_e<off>.jsonl (spatial, no g_t)
  cells  src, sham_cg, sham_twin, grpkp100, g_tfc, g_tw_fc, g_ta_fc, g_t  ->  SRC, S, T0, G, GTfc, GTw, GTa, G_T
  published (read only) OFT <pub-runs>/{src,sham_cg,sham_twin,grpkp100,grpkp100_twin,grpkp100_tw,grpkp100_ta},
            object <pub-runs>/{obj_src,obj_s,obj_t0,obj_g,obj_g_t}; pi0.5 <pi05>/<same>_e*.jsonl
            -> SRC, S, T0, G, G_T, G_Tw, G_Ta (prereg_block3_a2.py SPATIAL / OBJECT tables).

Inference (§4): per model and suite one cluster bootstrap over (task, ep) keys, all cells resampled jointly (one draw
matrix), B = 10000, seed 0, percentile 95% / 90%. Bootstrap code copied from prereg_block3_a2.py (Boot), so the
published numbers are recomputed with the identical draw matrix.

实现口径（协议没写死、我按最字面的读法定的地方）
 M1  片 = EP_OFFSET 的 5 集 x 10 任务 = 50 键（片号 = ep // 5）。某格某片 50 键全有正常记录才算「齐」；不齐的片整片不读
     （不拼零散键）。估计量的片数 = 它所有格（含 0 系数格）共同齐的片数。主集合 ep 0-19（4 片，要求 4），R2 用
     ep 0-39（8 片，要求 8）。片数 >= 要求 -> 照判；3 <= 片数 < 要求 -> 只印估计并标「未完成」，不判定；< 3 -> 不印。
     R2 在 400 键下 3-7 片同样只印「未完成」，不退回 200 键判（§3 写死用 400 判 R2）。
 M2  R2 的 |drop_nat(G)| 取 200 键点估计（S 只有 ep 0-19，400 键下 drop_nat 的共同键本来就是那 200 键）。
 M3  rec_w / rec_a / channel 照第三批写法把第三个格以 0 系数并入掩码（三格共同键），已发表数字也这样重算。
 M4  jsonl：error 非空的记录丢弃；同键多条正常记录，新模型取文件序（按 _e<off> 数值、行序）第一条（与 zoo_gate.py、
     SKIP_DONE 语义一致），已发表 pi0.5 沿用 prereg_block3 K18 取最后一条（重算须与第三批逐位一致）。两种计数都印。
 M5  门槛状态：FAIL -> 该格 does not count（估计照印，判定印「不判定…按线本应…仅供参考」）；UNK -> 照判并标「门槛未核实」；
     WARN -> 只印，不影响判定（协议写了要求但没写违反后果的项：T0 对 S 一致率 < 90%、error > 1%、GPU 型号不是指定型号、
     缺在线接触记录、π0 server GPU 不是 H100）。FAIL 项：copy_grip_cmd 任一集 0、场景 fixcheck 超差或 init 不符、
     S/T0 首帧对齐超 0.01 mm / 0.05°、替身查询帧残差 > 1 mm 比例 >= 1%、陈旧帧 > 0、TWIN_CAM=phys 格（GTfc/GTw）相机差 >
     0.01 mm / 0.01°、replan 不等于设计值（π0 5，OVLA 1）、一格内 server GPU 多于一种型号。
 M6  fixcheck：OFTi 用 ref_body_pos 对已发表 OFT SRC 同键（< 0.00005 mm 即 0.0000）；jsonl 用 fix 对同一参照（<= 0.005 mm，
     记录精度）；object 另比非机器人关节首帧 qpos[:3]（OFTi qpos[0]；jsonl objq，SRC 记录的 objq 在等待步之前，不比）。
     init 编号与参照不符计 FAIL。与 prereg_block3 K6 同。
 M7  替身门槛（§7.2）：查询帧 = OFTi [::8]、jsonl 记录里的 *_q 序列（客户端按 replan 抽帧）。IK：指垫残差 > 1 mm 的查询帧
     < 1%（第二批 2 mm / 1° 口径并印）。相机差：TWIN_CAM=phys 的格（GTfc、GTw）查询帧 <= 0.01 mm / 0.01°；T0（TWIN_CAM=twin，相机随 IK 替身手，差 = IK 残差）只印；
     GTa（腕部未换）只印；G_T 印均值（预期约 6.4 mm）。字段缺 -> UNK。第 0 步一致性、吞吐、回归不在 dump 里：
     给 --gate-json 时按 toy1 的 json 格式并入（cells.<dir>.status / items），不给时只印一句说明，不改格状态。
 M8  R1：「95% CI 与已发表 CI 重叠」用闭区间；已发表 CI 为本脚本从已发表数据重算的值（与协议文字不一致时警告）。
 M9  P7（R3 并列的 recovery_GT、F2、O2、object 对 spatial、Δ 旁注）照 prereg_block3_a2.p7：任一点估计恰为 0 记不确定。
 M10 F2：两条都复现 -> 像 π0.5；两条都不复现 -> 像 OFT；其余不确定。O2 同构：两条都复现 -> 复现，两条都不复现 -> 不复现，
     其余不确定（协议 O2 只写「P7 口径」，没写两条怎么合）。
 M11 E3 / E4 的三档：E3 drop_nat(G) 95% 上界 < -7 符合，下界 > -7 不符，其余不确定（第三批 R1 的 L/S/U 分法）；E4 rec_a
     95% 下界 > 0 符合，上界 < 0 不符（反向），其余不确定。E1 一致率 < 95% 或 E1 未完成 -> E3、E4 只印不读。
 M12 E5：|drop_nat(G)| 的 95% 区间 = 符号区间取绝对值（跨 0 时下端为 0）；OVLA 的上端 < OFTi 的下端 -> 「整体小于」，
     否则「相当或更大」。
 M13 d5 门槛（§7.3）：d5 95% 下界 < -7 -> 加印 rec'_fc / rec'_w / rec'_a（= rec - d5；rec'_fc 与 recovery(G) 恒等），
     并在该模型该套件的族结论后加「替身管线代价 d5 = …」警示；判定仍用 §5 定义。
 M14 Holm（透明度）：bootstrap 双侧 p = min(1, 2 min(P*(θ<=0), P*(θ>=0)))，H0 = 0（第三批 K4）。三个族分别校正：
     R = {R1 drop_nat(G), R2 channel(400), R3 recovery(G)}，F = {F1 channel, F2 drop_nat(G), F2 recovery(G)}，
     O = {O1 recovery(G) OFTi, O2 drop_nat(G) π0, O2 recovery(G) π0}。
 M15 R10：OFTi 与已发表 OFT 用第三批规则（|点| <= 3 且所涉格 GPU 组成不同 -> 噪声带）；π0 / OVLA 的估计量与 Δ 按 §7.1
     「一律」：|点| <= 3 即标「在渲染节点噪声带内」。
 M16 Δ_model = est(新) - est(已发表)：新模型格与已发表格放进同一个 Y、同一张 200 键抽样矩阵（配对）。对应关系：
     drop_nat、d5、rec_w、rec_a、channel、recovery(G)、recovery_GT(G)；新的 rec_* / recovery(G) 是固定相机、已发表是移动
     相机，逐行注明；recovery_GT 对已发表 recovery(G) 是配置逐项相同的一对。OVLA 没有已发表对照，不算 Δ。
 M17 frac = recovery(G) / |drop_nat(G)| 只印点估计；|drop_nat(G)| 点 < 7 不印。
 M18 已发表格不过门槛、不套片数规则（第三批已过门槛，键齐 200）；只印键数与 error 计数。
 M19 输出文件已存在 -> 退出码 3（一次读数），--overwrite 才覆盖。sha256 文件缺失印 MISSING，不报错。
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

H = os.path.expanduser
SEED = 0
B_DEFAULT = 10000
TASKS = range(10)
SLICE = 5                      # episodes per slice (EP_OFFSET step)
NEED_MAIN, NEED_R2 = 4, 8      # §7.1: >= 4 slices (R2 three cells >= 8)
MIN_PRINT = 3                  # 3 slices: print "未完成"; < 3: not printed
# §5 decision lines
R1_UPPER = -7.0
HALF, QUARTER = 0.5, 0.25
D5_LINE = -7.0
E3_LINE = -7.0
E1_AGREE_MIN = 95.0
FRAC_MIN = 7.0
R10_BAND = 3.0
T0_S_AGREE_MIN = 90.0
ERR_FRAC_MAX = 0.01
OVLA_SRC_MIN = 75.0
# §7 gates
FIX_TOL_NPZ, FIX_TOL_JSONL = 0.00005, 0.005 + 1e-9
ALIGN_MM, ALIGN_DEG = 0.01, 0.05
CAM_MM, CAM_DEG = 0.01, 0.01
IK_MM, IK_FRAC = 1.0, 0.01
OFT_STRIDE = 8

# code, new-model dir/prefix, copy_grip_cmd design, twin kind
NEW_CELLS = [("SRC", "src", None, None), ("S", "sham_cg", 1, None), ("T0", "sham_twin", 1, "sync"),
             ("G", "grpkp100", 1, None), ("GTfc", "g_tfc", 1, "sync"), ("GTw", "g_tw_fc", 1, "sync"),
             ("GTa", "g_ta_fc", 1, "agent"), ("G_T", "g_t", 1, "move")]
ALIGN_GATED = {"S", "T0"}      # §7.1: Panda bodies gated; PandaR85 cells print only (附录 2 裁定 1)
CAM_GATED = {"GTfc", "GTw"}  # T0: TWIN_CAM=twin (protocol §2 table, = published sham_twin) -> camera follows the IK twin hand, error = IK residual, print only (WORKLOG_vlazoo 8.5)
DENSE = {"G", "GTw", "GTa"}    # §3: OFTi spatial only, ep 20-39
# published (prereg_block3_a2 SPATIAL / OBJECT)
PUB_SP = [("SRC", "src"), ("S", "sham_cg"), ("T0", "sham_twin"), ("G", "grpkp100"), ("G_T", "grpkp100_twin"),
          ("G_Tw", "grpkp100_tw"), ("G_Ta", "grpkp100_ta")]
PUB_OBJ = [("SRC", "obj_src"), ("S", "obj_s"), ("T0", "obj_t0"), ("G", "obj_g"), ("G_T", "obj_g_t")]

# §4 estimands (new models): name -> terms [(coef, cell)]
EST = {
    "drop_nat(G)": [(1, "G"), (-1, "S")],
    "d5": [(1, "T0"), (-1, "S")],
    "rec_fc": [(1, "GTfc"), (-1, "G")],
    "rec_w": [(1, "GTw"), (-1, "G"), (0, "GTa")],
    "rec_a": [(1, "GTa"), (-1, "G"), (0, "GTw")],
    "channel": [(1, "GTw"), (-1, "GTa"), (0, "G")],
    "recovery(G)": [(1, "GTfc"), (-1, "T0"), (-1, "G"), (1, "S")],
    "recovery_GT(G)": [(1, "G_T"), (-1, "T0"), (-1, "G"), (1, "S")],
    "add": [(1, "GTw"), (1, "GTa"), (-1, "GTfc"), (-1, "G")],
}
REC_PRIME = {
    "rec'_fc": [(1, "GTfc"), (-1, "G"), (-1, "T0"), (1, "S")],
    "rec'_w": [(1, "GTw"), (-1, "G"), (-1, "T0"), (1, "S")],
    "rec'_a": [(1, "GTa"), (-1, "G"), (-1, "T0"), (1, "S")],
}
R2_EST = {k: EST[k] for k in ("rec_w", "rec_a", "channel")}
PUB_EST_SP = {
    "drop_nat(G)": [(1, "G"), (-1, "S")],
    "d5": [(1, "T0"), (-1, "S")],
    "rec_w": [(1, "G_Tw"), (-1, "G"), (0, "G_Ta")],
    "rec_a": [(1, "G_Ta"), (-1, "G"), (0, "G_Tw")],
    "channel": [(1, "G_Tw"), (-1, "G_Ta"), (0, "G")],
    "recovery(G)": [(1, "G_T"), (-1, "T0"), (-1, "G"), (1, "S")],
}
PUB_EST_OBJ = {k: PUB_EST_SP[k] for k in ("drop_nat(G)", "d5", "recovery(G)")}
# Δ_model: new estimand -> (published estimand, note)
DELTA_MAP = [("drop_nat(G)", "drop_nat(G)", ""), ("d5", "d5", ""),
             ("rec_w", "rec_w", "新=固定相机，已发表=移动相机"), ("rec_a", "rec_a", "新=固定相机，已发表=移动相机"),
             ("channel", "channel", "新=固定相机，已发表=移动相机"),
             ("recovery(G)", "recovery(G)", "新=GTfc 固定相机 SYNC 开，已发表=G_T 移动相机"),
             ("recovery_GT(G)", "recovery(G)", "配置逐项相同（G_T 对 G_T）")]
# numbers written in the protocol (§0, §5, §7.3): (point, lo, hi); None = not given
PROTO_PUB = {
    ("OFT", "spatial"): {"drop_nat(G)": (-45, -52, -38), "rec_w": (32.5, None, None), "rec_a": (21.5, None, None),
                         "channel": (11, 5, 17.5), "recovery(G)": (36, 28, 44)},
    ("OFT", "object"): {"drop_nat(G)": (-16, -22, -10), "recovery(G)": (-5.5, -13, 2)},
    ("pi05", "spatial"): {"drop_nat(G)": (-17, -22.5, -12), "rec_w": (0.5, None, None), "rec_a": (10, None, None),
                          "channel": (-9.5, -15, -4), "recovery(G)": (11.5, 6, 17.5), "d5": (-3.5, -6.5, -1)},
    ("pi05", "object"): {"drop_nat(G)": (-17, None, None), "recovery(G)": (3.5, -3.5, 10.5)},
}
PROTO_TOL = 0.5 + 1e-9         # protocol numbers are rounded to 0.5

# fixed interpretation sentences (§5, §6, §7.3)
TXT_R_YES = "复现：合训 OFT 的三条结论不是权重版本效应"
TXT_R_NO = "不复现：该条结论对微调数据 / 训练 run 敏感，论文里对应的 OFT 结论降为『该 checkpoint』"
TXT_R_CAM = "R3（GTfc）与 recovery_GT 判定不同：两者都写，差异归到相机钉法，不取其一"
TXT_F_STRONG = "通道排序在 2/2 个家族内一致、跨家族相反（本批最强的句子，仍只是 4 个 checkpoint）"
TXT_F_OFT = ("通道排序不随家族走，π0.5 的第三视角占优降为个案；不得把 π0 与 π0.5 的差归到「读不读 S」"
             "（两者还差训练配方与数据）")
TXT_O1_YES = ("「object 上重绘不救」不是权重版本效应，而是任务 / 场景层面的边界（object 场景无柜子、物体多、腕部构图不同），"
              "论文正文里的任务依赖主张站得住")
TXT_O1_NO = "object 负结果降为合训 checkpoint 个案，toy2 的范围声明（H_fb 仅 spatial）仍不改"
TXT_E5_SMALL = "去掉腕部输入的同骨干模型在同一组身体上掉得更少"
TXT_E5_BIG = "腕部输入不是掉分幅度的必要条件"
TXT_E5_CAVEAT = "OVLA 与 OFTi 还差本体感受、动作头、chunk 与微调配方，不能归因到腕相机"
TXT_E1_NOISE = "实测 < 100% 的部分就是本管线（T4 渲染 + 服务端推理）的噪声底，作为 OVLA 所有 |效应| 的下限参照"
TXT_D5_ON = "触发：加印相对 T0 的 rec'，族结论加警示"
TXT_LIMIT = ("§6 措辞约束：复现只说明「不是权重版本效应」，不说明「是架构」；不复现只写「对微调数据敏感」；"
             "任何「因为 X 架构 / X 输入所以 Y」的句子不许出现")


# =============================================================================================== output
class Out:
    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.lines.append(s)


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


def ci_txt(r, nd=2):
    return f"{r['point']:+.{nd}f} [{r['ci95'][0]:+.{nd}f}, {r['ci95'][1]:+.{nd}f}]"


# =============================================================================================== logs (align fallback)
def parse_align(s):
    out = {}
    for k in ("pos_mm", "rot_deg", "fixture_mm"):
        m = re.search(k + r"=([-\d.]+)", s)
        if m:
            out[k] = float(m.group(1))
    m = re.search(r"copy_grip_cmd=(\d)", s)
    if m:
        out["copy_grip_cmd"] = int(m.group(1))
    return out


def oft_eval_align(log_dir, model_dir, cell_dir):
    """OFTi EVAL logs (unwrapped BODY_ALIGN lines): {(task, ep): align}. Only used when the npz has no align_json."""
    out = {}
    for f in sorted(glob.glob(os.path.join(log_dir, "oft_eval", f"EVAL-*--{model_dir}_{cell_dir}.txt"))):
        cur = None
        for ln in open(f, errors="ignore"):
            mi = re.search(r"INIT_ID: task=(\d+) episode=(\d+)", ln)
            if mi:
                cur = (int(mi.group(1)), int(mi.group(2)))
            elif cur is not None and "BODY_ALIGN:" in ln:
                out.setdefault(cur, parse_align(ln))
    return out


def client_log_align(log_dir, model, cell):
    """jsonl client logs c_<model>_<cell>_<job>_e<off>.log: BODY_ALIGN line followed by the EP line of that episode."""
    out = {}
    for f in sorted(glob.glob(os.path.join(log_dir, f"c_{model}_{cell}_*_e*.log"))):
        if not re.match(re.escape(f"c_{model}_{cell}_") + r"\d+_e\d+\.log$", os.path.basename(f)):
            continue
        pend = None
        for ln in open(f, errors="ignore"):
            if "BODY_ALIGN:" in ln:
                pend = parse_align(ln)
                continue
            m = re.match(r"EP task=(\d+) ep=(\d+)", ln)
            if m is None and ln.startswith("EP {"):
                try:
                    r = json.loads(ln[3:ln.rindex("}") + 1])
                    m = (int(r["task"]), int(r["episode"]))
                except (ValueError, KeyError):
                    m = None
            elif m is not None:
                m = (int(m.group(1)), int(m.group(2)))
            if m is not None:
                if pend is not None:
                    out.setdefault(m, pend)
                pend = None
    return out


# =============================================================================================== loaders
def qpos_layout(names):
    ad, a = {}, 0
    for n in names:
        s = 7 if (n.endswith("_joint0") and not n.startswith(("robot0", "gripper0"))) else 1
        ad[n] = (a, s)
        a += s
    return ad, a


def load_npz_cell(path, neps, want_objq):
    """prereg_block3_a2.load_oft + twin_cam_err fields. Returns ({key: rec}, meta)."""
    out = {}
    nfile = 0
    for f in sorted(glob.glob(os.path.join(path, "t*_e*_i*.npz"))):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)\.npz$", f)
        t, e = int(m.group(1)), int(m.group(2))
        nfile += 1
        if e >= neps:
            continue
        z = np.load(f, allow_pickle=True)
        F = set(z.files)
        r = {"succ": float(bool(z["success"])), "init": int(z["init"]) if "init" in F else int(m.group(3))}
        r["host"] = _s(z["host"]) if "host" in F else None
        r["gpu"] = _s(z["gpu"]) if "gpu" in F else None
        r["align"] = None
        if "align_json" in F:
            try:
                r["align"] = json.loads(_s(z["align_json"]))
            except ValueError:
                pass
        c = {}
        for k in F:
            if k.startswith("contact_"):
                v = _s(z[k]) if z[k].dtype.kind in "US" else z[k].item()
                if k.endswith("_pairs"):
                    try:
                        v = json.loads(v)
                    except (ValueError, TypeError):
                        pass
                c[k[len("contact_"):]] = v
        r["contacts"] = c if "arm_env_contact_steps" in c else None
        tw = {}
        if "twin_pos_err_mm" in F:
            pe = np.asarray(z["twin_pos_err_mm"], float).ravel()
            tw["pos_q"] = pe[::OFT_STRIDE]
            tw["n_all"] = len(pe)
            if "twin_rot_err_deg" in F:
                tw["rot_q"] = np.asarray(z["twin_rot_err_deg"], float).ravel()[::OFT_STRIDE]
            tw["stale"] = int(np.asarray(z["twin_stale"]).sum()) if "twin_stale" in F else None
        if "twin_cam_err_mm" in F:
            cm = np.asarray(z["twin_cam_err_mm"], float).ravel()
            tw["cam_mm_q"], tw["cam_mm_all"] = cm[::OFT_STRIDE], cm
        if "twin_cam_err_deg" in F:
            cd = np.asarray(z["twin_cam_err_deg"], float).ravel()
            tw["cam_deg_q"], tw["cam_deg_all"] = cd[::OFT_STRIDE], cd
        r["twin"] = tw or None
        r["ref"] = ([str(x) for x in z["ref_body_names"]], np.asarray(z["ref_body_pos"], float)) if "ref_body_pos" in F else None
        r["objq"] = None
        if want_objq and "joint_names" in F and "qpos" in F:
            names = [str(x) for x in z["joint_names"]]
            ad, tot = qpos_layout(names)
            q0 = np.asarray(z["qpos"][0], float)
            if tot == len(q0):
                r["objq"] = {n: q0[a:a + s] for n, (a, s) in ad.items() if not n.startswith(("robot0", "gripper0"))}
        out[(t, e)] = r
    return out, {"files": nfile}


def load_jsonl_cell(root, name, neps, dup="first"):
    """prereg_block3_a2.load_pi05 + models1 fields. dup: which clean record of a duplicated key to keep."""
    rx = re.compile("^" + re.escape(name) + r"_e(\d+)\.jsonl$")
    files = sorted((int(rx.match(os.path.basename(f)).group(1)), f) for f in glob.glob(os.path.join(root, name + "_e*.jsonl"))
                   if rx.match(os.path.basename(f)))
    out, n_err, n_dup, n_all = {}, 0, 0, 0
    err_keys = Counter()
    for _, f in files:
        with open(f) as fh:
            lines = fh.readlines()
        for ln in lines:
            if not ln.strip():
                continue
            r = json.loads(ln)
            n_all += 1
            if r.get("error"):
                n_err += 1
                err_keys[(int(r["task"]), int(r["episode"]))] += 1
                continue
            k = (int(r["task"]), int(r["episode"]))
            if k[1] >= neps:
                continue
            if k in out:
                n_dup += 1
                if dup == "first":
                    continue
            tw = r.get("twin") or None
            c = r.get("contacts")
            twin = None
            if isinstance(tw, dict) and tw:
                twin = {"raw": tw}
                if tw.get("pos_err_mm_q") is not None:
                    twin["pos_q"] = np.asarray(tw["pos_err_mm_q"], float)
                twin["stale"] = tw.get("stale")
                if tw.get("cam_err_mm_q") is not None:
                    twin["cam_mm_q"] = np.asarray(tw["cam_err_mm_q"], float)
                if tw.get("cam_err_deg_q") is not None:
                    twin["cam_deg_q"] = np.asarray(tw["cam_err_deg_q"], float)
                if tw.get("cam_err_mm_max") is not None:
                    twin["cam_mm_max"] = float(tw["cam_err_mm_max"])
                if tw.get("cam_err_deg_max") is not None:
                    twin["cam_deg_max"] = float(tw["cam_err_deg_max"])
                if tw.get("cam_err_mm_mean") is not None:
                    twin["cam_mm_mean"] = float(tw["cam_err_mm_mean"])
            out[k] = {
                "succ": float(bool(r["success"])), "init": r.get("init"), "host": r.get("host"), "gpu": r.get("gpu"),
                "align": r.get("align"), "contacts": c if (isinstance(c, dict) and "arm_env_contact_steps" in c) else None,
                "twin": twin, "fix": r.get("fix"), "objq": r.get("objq"), "replan": r.get("replan"),
                "chunk_len": r.get("chunk_len"), "a0": r.get("a0"), "server_host": r.get("server_host"),
                "server_gpu": r.get("server_gpu"), "rtt": r.get("rtt_ms_med") if r.get("rtt_ms_med") is not None else r.get("rtt_ms_mean"),
            }
    return out, {"files": len(files), "lines": n_all, "error_dropped": n_err, "dup_clean": n_dup,
                 "error_keys": len(err_keys), "error_only_keys": len([k for k in err_keys if k not in out])}


# =============================================================================================== bootstrap (prereg_block3_a2.Boot)
class Boot:
    def __init__(self, keys, B, seed=SEED):
        self.keys = keys
        n = len(keys)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(B, n))
        self.W = np.bincount((idx + np.arange(B)[:, None] * n).ravel(), minlength=B * n).reshape(B, n).astype(np.float64)
        del idx

    def est(self, Y, terms):
        cells = [c for _, c in terms]
        if any(c not in Y for c in cells):
            return None
        m = np.ones(len(self.keys), bool)
        for c in cells:
            m &= ~np.isnan(Y[c])
        if m.sum() == 0:
            return None
        mf = m.astype(float)
        point = 100 * sum(a * Y[c][m].mean() for a, c in terms)
        den = self.W @ mf
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = 100 * sum(a * (self.W @ np.where(m, Y[c], 0.0)) / den for a, c in terms if a)
        bs = np.asarray(bs, float)
        bs = bs[~np.isnan(bs)]
        p0 = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())) if len(bs) else float("nan")
        return {"point": float(point), "ci95": tuple(float(x) for x in np.percentile(bs, [2.5, 97.5])),
                "ci90": tuple(float(x) for x in np.percentile(bs, [5, 95])), "p0": float(p0), "n": int(m.sum()),
                "mask": m, "cells": cells}


def holm(ps):
    names = [k for k in ps if ps[k] is not None and not math.isnan(ps[k])]
    order = sorted(names, key=lambda k: ps[k])
    adj, run = {}, 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * ps[k]))
        adj[k] = run
    return adj


# =============================================================================================== pure decision rules
def p7_verdict(ref, new):
    """prereg_block3_a2.p7 (D4): sign agrees and 95% CIs overlap -> 复现; sign differs and new CI excludes 0 -> 不复现."""
    (l1, h1), (l2, h2) = ref["ci95"], new["ci95"]
    if ref["point"] == 0 or new["point"] == 0:
        return "不确定"
    if np.sign(ref["point"]) == np.sign(new["point"]) and not (h1 < l2 or h2 < l1):
        return "复现"
    if np.sign(ref["point"]) != np.sign(new["point"]) and (l2 > 0 or h2 < 0):
        return "不复现"
    return "不确定"


def overlap(a, b):
    (l1, h1), (l2, h2) = a["ci95"], b["ci95"]
    return not (h1 < l2 or h2 < l1)


def r1_verdict(new, pub):
    if overlap(new, pub) and new["ci95"][1] < R1_UPPER:
        return "复现"
    if not overlap(new, pub):
        return "不复现"
    return "不确定"


def r2_verdict(channel, rec_w, drop_point):
    d = abs(drop_point)
    if channel["ci95"][0] > 0 and rec_w["point"] >= HALF * d:
        return "复现"
    if channel["ci95"][1] < 0 or rec_w["ci95"][1] < QUARTER * d:
        return "不复现"
    return "不确定"


def r3_verdict(rec, drop_point):
    d = abs(drop_point)
    if rec["ci95"][0] > 0 and rec["point"] >= HALF * d:
        return "复现"
    if rec["ci95"][1] < QUARTER * d:
        return "不复现"
    return "不确定"


def f1_verdict(channel):
    if channel["ci95"][1] < 0:
        return "像 π0.5"
    if channel["ci95"][0] > 0:
        return "像 OFT"
    return "不确定"


def f2_verdict(v_drop, v_rec):
    if v_drop == "复现" and v_rec == "复现":
        return "像 π0.5"
    if v_drop == "不复现" and v_rec == "不复现":
        return "像 OFT"
    return "不确定"


def o1_verdict(rec, drop_point):
    d = abs(drop_point)
    if rec["ci95"][1] < HALF * d:
        return "复现"
    if rec["ci95"][0] > 0 and rec["point"] >= HALF * d:
        return "不复现"
    return "不确定"


def o2_verdict(v_drop, v_rec):
    if v_drop == "复现" and v_rec == "复现":
        return "复现"
    if v_drop == "不复现" and v_rec == "不复现":
        return "不复现"
    return "不确定"


def e3_verdict(drop):
    lo, hi = drop["ci95"]
    return "符合" if hi < E3_LINE else ("不符" if lo > E3_LINE else "不确定")


def e4_verdict(rec_a):
    lo, hi = rec_a["ci95"]
    return "符合" if lo > 0 else ("不符" if hi < 0 else "不确定")


def abs_interval(r):
    lo, hi = r["ci95"]
    if lo <= 0 <= hi:
        return 0.0, max(-lo, hi)
    return min(abs(lo), abs(hi)), max(abs(lo), abs(hi))


def e5_verdict(ovla_drop, ofti_drop):
    return "整体小于" if abs_interval(ovla_drop)[1] < abs_interval(ofti_drop)[0] else "相当或更大"


def d5_trigger(d5):
    return d5["ci95"][0] < D5_LINE


def slice_status(nsl, need):
    if nsl >= need:
        return "完成"
    if nsl >= MIN_PRINT:
        return "未完成"
    return "不印"


def r_family(vs):
    """vs: {"R1": v, ...}; v in 复现/不复现/不确定 or a non-decision label."""
    if any(v == "不复现" for v in vs.values()):
        return "不复现"
    if all(v == "复现" for v in vs.values()):
        return "复现"
    if any(v not in ("复现", "不复现", "不确定") for v in vs.values()):
        return "未能判定"
    return "不确定"


# =============================================================================================== model container
class ModelData:
    """One model on one suite (new: slice rule + gates; published: read only)."""

    def __init__(self, label, kind, suite, cells, root, a, published=False, model_dir=None, dense=frozenset(), dup="first",
                 expect_replan=None, gpu_expect=None):
        self.label, self.kind, self.suite, self.published = label, kind, suite, published
        self.cells = cells  # [(code, dirname, copy, twin)]
        self.spec = {c[0]: c for c in cells}
        self.dense = set(dense)
        self.expect_replan, self.gpu_expect = expect_replan, gpu_expect
        self.data, self.meta, self.raw = {}, {}, {}
        for code, d, _, _ in cells:
            neps = 40 if code in self.dense else 20
            if kind == "npz":
                p = os.path.join(root, d)
                if not os.path.isdir(p):
                    continue
                rec, meta = load_npz_cell(p, neps, suite == "object")
                if not published:
                    la = None
                    for k, r in rec.items():
                        if (r.get("align") or {}).get("copy_grip_cmd") is None and code != "SRC":
                            la = la if la is not None else oft_eval_align(H(a.zoo_logs), model_dir, d)
                            if k in la:
                                r["align"] = {**la[k], **(r.get("align") or {})}
            else:
                rec, meta = load_jsonl_cell(root, d, neps, dup=dup)
                if not published:
                    la = None
                    for k, r in rec.items():
                        if (r.get("align") or {}).get("copy_grip_cmd") is None and code != "SRC":
                            la = la if la is not None else client_log_align(H(a.zoo_logs), model_dir, d)
                            if k in la:
                                r["align"] = {**la[k], **(r.get("align") or {})}
            self.meta[code] = meta
            if rec:
                self.raw[code] = rec
        # §7.1 slice rule: keep complete slices only (new models)
        self.slices = {}
        for code, rec in self.raw.items():
            if published:
                self.data[code] = rec
                self.slices[code] = set(range(8 if code in self.dense else 4))
                continue
            nsl = 8 if code in self.dense else 4
            full = set()
            for s in range(nsl):
                need = [(t, e) for t in TASKS for e in range(s * SLICE, (s + 1) * SLICE)]
                if all(k in rec for k in need):
                    full.add(s)
            self.slices[code] = full
            kept = {k: r for k, r in rec.items() if k[1] // SLICE in full}
            if kept:
                self.data[code] = kept
        self.gates = {}
        self.keys200 = [(t, e) for t in TASKS for e in range(20)]
        self.keys400 = [(t, e) for t in TASKS for e in range(40)]
        # SENS (i): --exclude-keys of this model x suite ("<model_dir>|libero_<suite>") -> NaN in EVERY cell (paired).
        # Applied after the §7.1 slice rule (dropping records before it would make the whole 50-key slice incomplete)
        # and only to Y: estimands, verdicts, Holm, Δ_model use the reduced keys; gates, task tables, contacts, E1/E2
        # still see all keys (add1 §2.3 item 5: gates are printed as the frozen script judges them, not re-licensed).
        self.sens_model_dir = model_dir
        _ex = set() if published else SENS["exclude"].get(f"{model_dir}|libero_{suite}", set())
        self.sens_excluded = sorted(k for k in _ex if any(k in rec for rec in self.data.values()))
        self.sens_listed = len(_ex)
        self.Y200 = {c: np.array([rec[k]["succ"] if (k in rec and k not in _ex) else np.nan for k in self.keys200]) for c, rec in self.data.items()}
        self.Y400 = {c: np.array([rec[k]["succ"] if (k in rec and k not in _ex) else np.nan for k in self.keys400]) for c, rec in self.data.items()
                     if c in self.dense}

    def nsl(self, cells, universe):
        """Common complete slices of all cells within the universe (4 = ep 0-19, 8 = ep 0-39)."""
        s = set(range(universe))
        for c in dict.fromkeys(cells):
            s &= self.slices.get(c, set())
        return len(s)

    # ------------------------------------------------------------------ gates (new models)
    def fixcheck(self, c, src):
        rec = self.data[c]
        if src is None:
            return ("UNK", "无已发表 OFT SRC 对照")
        worst, n, nos, bad_init = 0.0, 0, 0, 0
        for k, r in rec.items():
            if k not in src:
                nos += 1
                continue
            if r.get("init") is not None and src[k].get("init") is not None and int(r["init"]) != int(src[k]["init"]):
                bad_init += 1
            sn, sp = src[k]["ref"]
            sidx = {nm: i for i, nm in enumerate(sn)}
            ds = []
            if self.kind == "npz":
                if r.get("ref") is not None:
                    rn, rp = r["ref"]
                    ds += [float(np.abs(rp[i] - sp[sidx[nm]]).max()) * 1000 for i, nm in enumerate(rn) if nm in sidx]
                if self.suite == "object" and r.get("objq") and src[k].get("objq"):
                    so = src[k]["objq"]
                    ds += [float(np.abs(np.asarray(v)[:3] - np.asarray(so[j])[:3]).max()) * 1000 for j, v in r["objq"].items() if j in so]
            else:
                for nm, v in (r.get("fix") or {}).items():
                    if nm in sidx:
                        ds.append(float(np.abs(np.asarray(v, float) - sp[sidx[nm]]).max()) * 1000)
                if self.suite == "object" and c != "SRC" and r.get("objq") and src[k].get("objq"):
                    so = src[k]["objq"]
                    ds += [float(np.abs(np.asarray(v, float)[:3] - np.asarray(so[j])[:3]).max()) * 1000 for j, v in r["objq"].items() if j in so]
            if ds:
                worst, n = max(worst, max(ds)), n + 1
        if n == 0:
            return ("UNK", f"无可比键（缺参照键 {nos}）")
        tol = FIX_TOL_NPZ if self.kind == "npz" else FIX_TOL_JSONL
        st = "PASS" if (worst <= tol and bad_init == 0) else "FAIL"
        return (st, f"keys {n}, 最大差 {worst:.4f} mm（门槛 {'0.0000' if self.kind == 'npz' else '<= 0.005 记录精度'}），"
                    f"init 编号不符 {bad_init}" + (f"，{nos} 键无参照" if nos else "")
                    + ("；SRC 的 objq 在等待步之前，只比 fix" if (self.suite == "object" and c == "SRC" and self.kind == "jsonl") else ""))

    def copy_gate(self, c):
        if self.spec[c][2] is None:
            return ("NA", "无迁移")
        vals = [(r.get("align") or {}).get("copy_grip_cmd") for r in self.data[c].values()]
        n1 = sum(v in (1, "1", True) for v in vals)
        n0 = sum(v in (0, "0", False) for v in vals)
        nm = len(vals) - n1 - n0
        det = f"copy_grip_cmd=1: {n1}, =0: {n0}, 缺: {nm}"
        if n0:
            return ("FAIL", det)
        if nm:
            return ("UNK", det)
        return ("PASS", det)

    def align_gate(self, c):
        if self.spec[c][2] is None:
            return ("NA", "无迁移")
        pos = [(r.get("align") or {}).get("pos_mm") for r in self.data[c].values()]
        rot = [(r.get("align") or {}).get("rot_deg") for r in self.data[c].values()]
        pv, rv = [float(x) for x in pos if x is not None], [float(x) for x in rot if x is not None]
        if not rv or not pv:
            return ("UNK", "无对齐记录")
        bad = sum((p is not None and float(p) > ALIGN_MM) or (r is not None and float(r) > ALIGN_DEG) for p, r in zip(pos, rot))
        nmiss = sum(1 for p, r in zip(pos, rot) if p is None or r is None)
        det = f"最大 {max(pv):.4f} mm / {max(rv):.4f}°（门槛 {ALIGN_MM} mm / {ALIGN_DEG}°），超门槛 {bad} 集，缺 {nmiss}"
        if c not in ALIGN_GATED:
            return ("NA", det + "（PandaR85 格只印数，附录 2 裁定 1）")
        if bad:
            return ("FAIL", det)
        return ("UNK" if nmiss else "PASS", det)

    def twin_gates(self, c):
        kind = self.spec[c][3]
        if kind is None:
            return {}
        g = {}
        rec = self.data[c]
        recs = [r["twin"] for r in rec.values() if r.get("twin")]
        nmiss = len(rec) - len(recs)
        if not recs:
            return {"替身记录": ("UNK", "无替身记录")}
        # IK residual on query frames
        pq = [t["pos_q"] for t in recs if t.get("pos_q") is not None]
        if len(pq) == len(recs):
            pe = np.concatenate(pq) if pq else np.zeros(0)
            nb = int((pe > IK_MM).sum())
            frac = nb / len(pe) if len(pe) else float("nan")
            st = ("PASS" if frac < IK_FRAC else "FAIL") if len(pe) else "UNK"
            extra = ""
            if self.kind == "npz" and all(t.get("rot_q") is not None for t in recs):
                re_ = np.concatenate([t["rot_q"] for t in recs])
                extra = f"；第二批口径 pos>2mm 或 rot>1° {100 * ((pe > 2) | (re_ > 1)).mean():.2f}%（只印）"
            elif self.kind == "jsonl":
                miss = sum(int(t["raw"].get("miss") or 0) for t in recs)
                npaint = sum(int(t["raw"].get("n_paint") or 0) for t in recs)
                extra = f"；第二批口径 miss/n_paint {miss}/{npaint}（全部帧，只印）"
            g["IK 残差>1mm<1%"] = (st if not nmiss else ("FAIL" if st == "FAIL" else "UNK"),
                                 f"查询帧 > {IK_MM} mm：{nb}/{len(pe)} = {100 * frac:.2f}%（门槛 < 1%）{extra}" + (f"；{nmiss} 集无替身字段" if nmiss else ""))
        else:
            g["IK 残差>1mm<1%"] = ("UNK", f"{len(recs) - len(pq)} 集缺查询帧残差序列")
        st_ = [t.get("stale") for t in recs]
        if any(s is None for s in st_):
            g["陈旧帧 0"] = ("UNK", "缺 stale 字段")
        else:
            s = int(sum(int(x) for x in st_))
            g["陈旧帧 0"] = ("PASS" if s == 0 else "FAIL", f"陈旧帧合计 {s}")
        # wrist camera pose diff
        have = [t for t in recs if t.get("cam_mm_q") is not None and t.get("cam_deg_q") is not None]
        if kind == "sync" and c in CAM_GATED:
            if len(have) < len(rec):
                g["相机位姿差"] = ("UNK", f"{len(rec) - len(have)} 集缺 cam_err 字段（OFTi dump 需 twin_cam_err_mm/deg；jsonl 需 twin.cam_err_*_q）")
            else:
                qm = np.concatenate([t["cam_mm_q"] for t in have])
                qd = np.concatenate([t["cam_deg_q"] for t in have])
                nb = int(((qm > CAM_MM) | (qd > CAM_DEG)).sum())
                allm = max([float(np.max(t["cam_mm_all"])) for t in have if t.get("cam_mm_all") is not None and len(t["cam_mm_all"])]
                           + [t["cam_mm_max"] for t in have if t.get("cam_mm_max") is not None] or [float("nan")])
                g["相机位姿差"] = ("FAIL" if nb else "PASS",
                                f"查询帧最大 {qm.max() if len(qm) else float('nan'):.4f} mm / {qd.max() if len(qd) else float('nan'):.4f}°，"
                                f"超 {CAM_MM} mm/{CAM_DEG}° 的查询帧 {nb}/{len(qm)}；全帧最大 {allm:.4f} mm")
        elif have:
            qm = np.concatenate([t["cam_mm_q"] for t in have])
            lbl = ("G_T 相机差均值（预期约 6.4 mm，只印）" if kind == "move" else
                   "T0 相机随替身手（TWIN_CAM=twin），差 = IK 残差，只印" if c == "T0" else "相机差（腕部未换，只印）")
            g["相机位姿差"] = ("NA", f"{lbl}：查询帧均值 {qm.mean() if len(qm) else float('nan'):.3f} mm，最大 {qm.max() if len(qm) else float('nan'):.3f} mm，{len(have)} 集")
        else:
            g["相机位姿差"] = ("NA", "无 cam_err 字段（只印项）")
        return g

    def jsonl_gates(self, c):
        g = {}
        rec = self.data[c]
        meta = self.meta[c]
        rp = Counter(r.get("replan") for r in rec.values())
        if self.expect_replan is not None:
            bad = sum(v for k, v in rp.items() if k != self.expect_replan)
            g["replan"] = ("FAIL" if bad else "PASS", f"分布 {dict(rp)}（应全为 {self.expect_replan}）")
        cl = Counter(r.get("chunk_len") for r in rec.values())
        sg = Counter((r.get("server_gpu") or "") for r in rec.values())
        named = {k for k in sg if k}
        if len(named) > 1:
            st = "FAIL"
        elif not named:
            st = "UNK"
        else:
            st = "PASS" if "H100" in next(iter(named)) else "WARN"
        g["server GPU 单一型号（H100）"] = (st, f"server_gpu {dict(sg)}；server_host {dict(Counter(r.get('server_host') for r in rec.values()))}；chunk_len {dict(cl)}")
        n_lines = meta.get("lines", 0)
        e = meta.get("error_dropped", 0)
        g["error 记录"] = ("WARN" if (n_lines and e / n_lines > ERR_FRAC_MAX) else "PASS",
                         f"error 记录 {e}/{n_lines} 行（> 1% 先查 server），只有 error 记录的键 {meta.get('error_only_keys', 0)}，重复正常记录 {meta.get('dup_clean', 0)}")
        rt = [float(r["rtt"]) for r in rec.values() if r.get("rtt") is not None]
        g["RTT"] = ("NA", f"中位 {np.median(rt):.0f} ms，> 500 ms 的集 {sum(1 for x in rt if x > 500)}（只印）" if rt else "无 RTT 字段")
        return g

    def gpu_gate(self, c):
        comp = self.gpu_comp(c)
        if self.gpu_expect is None:
            return None
        ok = all(all(s in g for s in self.gpu_expect) for g in comp if g != "?")
        if "?" in comp:
            return ("WARN", f"有 GPU 缺失：{dict(comp)}")
        return ("PASS" if ok else "WARN", f"{dict(comp)}（应为 {' '.join(self.gpu_expect)}）")

    def contact_gate(self, c):
        nm = sum(1 for r in self.data[c].values() if not r.get("contacts"))
        return ("PASS" if nm == 0 else "WARN", f"缺在线接触记录 {nm} 集")

    def compute_gates(self, src, gate_json=None):
        for c in self.data:
            g = {"fixcheck": self.fixcheck(c, src), "copy_grip_cmd": self.copy_gate(c), "首帧对齐": self.align_gate(c)}
            g.update(self.twin_gates(c))
            if self.kind == "jsonl":
                g.update(self.jsonl_gates(c))
            gg = self.gpu_gate(c)
            if gg:
                g["GPU 型号"] = gg
            g["在线接触"] = self.contact_gate(c)
            if gate_json is not None and self.spec[c][3] is not None:
                d = self.spec[c][1]
                e = (gate_json.get("cells") or {}).get(d) if isinstance(gate_json, dict) else None
                if e is None:
                    g["gate json"] = ("UNK", f"门槛 json 无 {d}")
                else:
                    items = e.get("items") or {}
                    sts = [str(e.get("status", "UNK")).upper()] + [str(v.get("status", "UNK")).upper() for v in items.values()]
                    st = "FAIL" if "FAIL" in sts else ("UNK" if any(s not in ("PASS", "NA") for s in sts) else "PASS")
                    g["gate json"] = (st, f"cell {e.get('status')}；" + "，".join(f"{k}={v.get('status')}({v.get('value')})" for k, v in items.items()))
            self.gates[c] = g
        # T0 vs S outcome agreement (§7.2), WARN only
        if "T0" in self.data and "S" in self.data:
            ks = sorted(set(self.data["T0"]) & set(self.data["S"]))
            if ks:
                ag = 100 * np.mean([self.data["T0"][k]["succ"] == self.data["S"][k]["succ"] for k in ks])
                self.gates["T0"]["T0 对 S 同键一致率"] = ("PASS" if ag >= T0_S_AGREE_MIN else "WARN", f"{ag:.1f}%（{len(ks)} 键，门槛 >= 90%，低于先看图）")

    def status(self, c):
        if c not in self.data:
            return "MISSING"
        sts = [v[0] for v in self.gates.get(c, {}).values()]
        if "FAIL" in sts:
            return "FAIL"
        if "UNK" in sts:
            return "UNVERIFIED"
        return "OK"

    def print_gates(self):
        P(f"\n==== 门槛（{self.label}，{self.suite}）")
        for code, d, _, _ in self.cells:
            sl = sorted(self.slices.get(code, set()))
            nsl_design = 8 if code in self.dense else 4
            m = self.meta.get(code, {})
            if code not in self.data:
                raw = len(self.raw.get(code, {}))
                P(f"  {code:5s} ({d}): 无整片（原始正常记录 {raw} 键）" + (f"；jsonl {m.get('files', 0)} 文件 {m.get('lines', 0)} 行，丢 error {m.get('error_dropped', 0)}" if self.kind == "jsonl" and m else ""))
                continue
            raw = len(self.raw.get(code, {}))
            extra = ""
            if self.kind == "jsonl":
                extra = f"；jsonl {m['files']} 文件 {m['lines']} 行，丢 error {m['error_dropped']}，重复正常记录 {m['dup_clean']}"
            st = self.status(code)
            tag = {"FAIL": "  <<< does not count", "UNVERIFIED": "  （有门槛无法核实）"}.get(st, "")
            P(f"  {code:5s} ({d}): 齐的片 {sl}（设计 {nsl_design} 片），读入 {len(self.data[code])} 键，原始正常记录 {raw} 键{extra}{tag}")
            for gk, (s, det) in self.gates.get(code, {}).items():
                P(f"      {gk:20s} {s:5s} {det}")

    # ------------------------------------------------------------------ descriptive
    def gpu_comp(self, c, keys=None):
        rec = self.data.get(c, {})
        ks = keys if keys is not None else list(rec)
        return Counter((rec[k].get("gpu") or "?") for k in ks if k in rec)

    def per_task(self, c):
        rec = self.data.get(c, {})
        out = {}
        for t in TASKS:
            v = [r["succ"] for k, r in rec.items() if k[0] == t]
            out[t] = (int(sum(v)), len(v))
        return out

    def conc(self, c, ref="S"):
        if c not in self.data or ref not in self.data:
            return None
        a, b = self.data[c], self.data[ref]
        net = {t: 0 for t in TASKS}
        for k in set(a) & set(b):
            net[k[0]] += (1 - a[k]["succ"]) - (1 - b[k]["succ"])
        tot = sum(net.values())
        if tot <= 0:
            return f"无净增失败（净 {tot:+.0f}）"
        w = max(net, key=lambda t: net[t])
        return f"{net[w] / tot:.2f}（最差任务 {w}，{net[w]:+.0f}/{tot:+.0f}）"

    def task_table(self):
        P(f"\n==== 10 任务表 + conc（相对 S）+ GPU 型号组成（{self.label}，{self.suite}；只含整片）")
        for code, d, _, _ in self.cells:
            if code not in self.data:
                P(f"  {code:5s} 缺")
                continue
            pt = self.per_task(code)
            k, n = sum(v[0] for v in pt.values()), sum(v[1] for v in pt.values())
            comp = ", ".join(f"{g} x{m}" for g, m in sorted(self.gpu_comp(code).items()))
            if self.kind == "jsonl":
                sg = Counter((r.get("server_gpu") or "?") for r in self.data[code].values())
                comp = f"client {comp} | server " + ", ".join(f"{g} x{m}" for g, m in sorted(sg.items()))
            cc = self.conc(code) if code not in ("S", "SRC") else "-"
            P(f"  {code:5s} SR {100 * k / max(n, 1):5.1f}% ({k}/{n}) | " + " ".join(f"t{t}:{a}/{b}" for t, (a, b) in pt.items())
              + f" | conc {cc} | GPU {comp}")

    def contacts(self):
        P(f"\n==== 在线接触汇总（{self.label}，{self.suite}）：每任务 接触集数/有记录集数（s=停滞集数），首次接触中位步，pairs 前三")
        for code, d, _, _ in self.cells:
            if code not in self.data:
                continue
            rec = self.data[code]
            nmiss = sum(1 for r in rec.values() if not r.get("contacts"))
            row, detail = [], []
            for t in TASKS:
                rs = [r["contacts"] for k, r in rec.items() if k[0] == t and r.get("contacts")]
                ce = [x for x in rs if int(x.get("arm_env_contact_steps") or 0) > 0]
                stall = sum(1 for x in rs if int(x.get("stall_steps") or 0) > 0)
                row.append(f"t{t}:{len(ce)}/{len(rs)}" + (f"s{stall}" if stall else ""))
                if ce or stall:
                    fc = [int(x["first_contact"]) for x in ce if int(x.get("first_contact", -1)) >= 0]
                    pc = Counter()
                    for x in ce:
                        prs = x.get("pairs") or {}
                        if isinstance(prs, str):
                            try:
                                prs = json.loads(prs)
                            except ValueError:
                                prs = {}
                        pc.update(prs.keys())
                    detail.append(f"      t{t}: 接触 {len(ce)} 集，首次接触中位步 {np.median(fc) if fc else float('nan'):.0f}，"
                                  f"pairs 前三 {pc.most_common(3)}，停滞 {stall} 集")
            P(f"  {code:5s} " + " ".join(row) + (f" | 缺接触字段 {nmiss} 集" if nmiss else ""))
            for ln in detail:
                P(ln)


# =============================================================================================== estimands
def run_est(m, table, universe):
    """Returns {name: result or None}; result carries nsl / need / state."""
    boot = m.boot400 if universe == 8 else m.boot200
    Y = m.Y400 if universe == 8 else m.Y200
    need = NEED_R2 if universe == 8 else NEED_MAIN
    out = {}
    for nm, terms in table.items():
        r = boot.est(Y, terms)
        if r is None:
            out[nm] = None
            continue
        if m.published:
            r["nsl"], r["need"], r["state"] = need, need, "完成"
        else:
            r["nsl"] = m.nsl(r["cells"], universe)
            r["need"] = need
            r["state"] = slice_status(r["nsl"], need)
        out[nm] = r
    return out


def r10_tag(m, r):
    if r is None or abs(r["point"]) > R10_BAND:
        return ""
    if m.kind == "jsonl" and not m.published:
        return "  [在渲染节点噪声带内：π0/OVLA 在 T4 渲染，§7.1 一律标]"
    keys = m.keys400 if len(r["mask"]) == len(m.keys400) else m.keys200
    ks = [k for k, ok in zip(keys, r["mask"]) if ok]
    comps = [m.gpu_comp(c, ks) for c in dict.fromkeys(r["cells"])]
    if any("?" in cp for cp in comps):
        return "  [R10: |效应|<=3，GPU 组成有缺，无法判定]"
    if any(cp != comps[0] for cp in comps[1:]):
        return "  [R10: 在渲染节点噪声带内]"
    return ""


def cell_tags(m, cells):
    if m.published:
        return ""
    f = [c for c in dict.fromkeys(cells) if m.status(c) == "FAIL"]
    u = [c for c in dict.fromkeys(cells) if m.status(c) == "UNVERIFIED"]
    return (f"  <<< does not count（{','.join(f)}）" if f else "") + (f"  （门槛未核实：{','.join(u)}）" if u else "")


# ---------------------------------------------------------------------------------------- SENS (physics breakdown)
SENS = {"exclude": {}, "bounds": None}


def sens_load(path, kind):
    """--exclude-keys: {"<model_dir>|libero_<suite>": [[t, e], ...]}; --bounds: {"<model_dir>|<cell dir>": {"<ep offset>": w}}.
    Top-level fields starting with "_" are ignored. Anything else malformed raises (fail loudly)."""
    with open(H(path)) as fh:
        d = json.load(fh)
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if kind == "exclude":
            out[k] = {(int(t), int(e)) for t, e in v}
        else:
            out[k] = {str(o): int(w) for o, w in v.items() if not str(o).startswith("_")}
    return out


def sens_bound(m, nm, r):
    """SENS (ii): π0 / OVLA (new jsonl models) only. x = 100 * sum_cells |coef| * w_cell / n, w_cell = QACC warnings of that
    cell summed over the slices the estimand actually uses (from its mask); rounded UP to 0.1 (it is an upper bound).
    Two-cell estimands: (w_a + w_b) / n, as add1 §2.3 item 2 writes it."""
    if SENS["bounds"] is None or m.published or m.kind != "jsonl":
        return ""
    terms = {**EST, **REC_PRIME}.get(nm)
    if terms is None:
        return "  失效集上界 缺（未知估计量）"
    keys = m.keys400 if len(r["mask"]) == len(m.keys400) else m.keys200
    offs = sorted({k[1] // SLICE * SLICE for k, ok in zip(keys, r["mask"]) if ok})
    tot, miss = 0, []
    for a, c in terms:
        if not a:
            continue
        ent = SENS["bounds"].get(f"{m.sens_model_dir}|{m.spec[c][1]}")
        if ent is None:
            miss.append(f"{m.sens_model_dir}|{m.spec[c][1]}")
            continue
        for o in offs:
            if str(o) not in ent:
                miss.append(f"{m.sens_model_dir}|{m.spec[c][1]}@{o}")
                continue
            tot += abs(a) * ent[str(o)]
    if miss:
        return f"  失效集上界 缺（{', '.join(miss)}）"
    x = math.ceil(round(1000 * tot / r["n"], 6)) / 10 if r["n"] else float("nan")
    return f"  失效集上界 ±{x:.1f}"


def print_est(m, res, title):
    P(f"\n==== {title}（{m.label}，{m.suite}）：点估计、95% / 90% bootstrap CI、键数、共同整片数")
    if getattr(m, "sens_excluded", None):
        P(f"  [剔除版] 物理失效键 {len(m.sens_excluded)} 个在整片内、已从本模型本套件所有格剔除（json 列出 {m.sens_listed}）：{m.sens_excluded}")
    for nm, r in res.items():
        if r is None:
            P(f"  {nm:16s} 缺（无数据格或无共同键）")
            continue
        if r["state"] == "不印":
            P(f"  {nm:16s} 不印（共同整片 {r['nsl']} < 3）")
            continue
        flag = "" if r["state"] == "完成" else f"  【未完成：共同整片 {r['nsl']}/{r['need']}，只印不判】"
        P(f"  {nm:16s} {r['point']:+7.2f} pp  95% [{r['ci95'][0]:+7.2f}, {r['ci95'][1]:+7.2f}]  90% [{r['ci90'][0]:+7.2f}, {r['ci90'][1]:+7.2f}]"
          f"  n={r['n']} 片={r['nsl']}{flag}{cell_tags(m, r['cells'])}{r10_tag(m, r)}{sens_bound(m, nm, r)}")


def judge(m, rs, raw_fn):
    """rs: list of results needed by one decision line. Returns (verdict, text)."""
    if any(r is None for r in rs):
        return "缺", "缺（无数据）"
    if any(r["state"] == "不印" for r in rs):
        return "缺", "不判定（共同整片 < 3，不印）"
    raw = raw_fn()
    if any(r["state"] == "未完成" for r in rs):
        return "未完成", f"未完成（共同整片 {min(r['nsl'] for r in rs)}/{max(r['need'] for r in rs)}），不判定；按判定线本应为「{raw}」，仅供参考"
    cells = [c for r in rs for c in r["cells"]]
    f = [c for c in dict.fromkeys(cells) if m.status(c) == "FAIL"]
    u = [c for c in dict.fromkeys(cells) if m.status(c) == "UNVERIFIED"]
    if f:
        return "does not count", f"不判定（does not count：{','.join(f)} 门槛不过）；按判定线本应为「{raw}」，仅供参考"
    if u:
        return raw, f"{raw}（门槛未核实：{','.join(u)}）"
    return raw, raw


# =============================================================================================== main analysis
def make_model(label, kind, suite, a, root, model_dir, cells, dense=(), expect_replan=None, gpu_expect=None):
    m = ModelData(label, kind, suite, cells, root, a, published=False, model_dir=model_dir, dense=dense,
                  expect_replan=expect_replan, gpu_expect=gpu_expect)
    m.boot200 = Boot(m.keys200, a.B)
    m.boot400 = Boot(m.keys400, a.B) if dense else None
    return m


def make_pub(label, kind, suite, a):
    cells = [(c, d, 1 if c != "SRC" else None, None) for c, d in (PUB_SP if suite == "spatial" else PUB_OBJ)]
    root = H(a.pub_runs) if kind == "npz" else H(a.pi05)
    m = ModelData(label, kind, suite, cells, root, a, published=True, dup="last")
    m.boot200 = Boot(m.keys200, a.B)
    return m


def pub_section(pubs, V):
    P("\n######## 已发表两模型：从已发表数据重算（只读），与协议文字比对")
    out = {}
    for (lbl, suite), m in pubs.items():
        P(f"\n  -- {m.label} {suite}：每格键数 " + ", ".join(f"{c} {len(m.data.get(c, {}))}" for c, *_ in m.cells)
          + ("；jsonl 丢 error " + ", ".join(f"{c} {m.meta[c].get('error_dropped', 0)}" for c, *_ in m.cells if c in m.meta)
             + "；重复正常记录 " + ", ".join(f"{c} {m.meta[c].get('dup_clean', 0)}" for c, *_ in m.cells if c in m.meta) if m.kind == "jsonl" else ""))
        res = run_est(m, PUB_EST_SP if suite == "spatial" else PUB_EST_OBJ, 4)
        out[(lbl, suite)] = res
        for nm, r in res.items():
            if r is None:
                P(f"     {nm:14s} 缺")
                continue
            P(f"     {nm:14s} {r['point']:+7.2f}  95% [{r['ci95'][0]:+7.2f}, {r['ci95'][1]:+7.2f}]  90% [{r['ci90'][0]:+7.2f}, {r['ci90'][1]:+7.2f}]  n={r['n']}")
        P(f"     协议文字比对（容差 ±0.5，协议数字取到 0.5）：")
        for nm, (pt, lo, hi) in PROTO_PUB.get((lbl, suite), {}).items():
            r = res.get(nm)
            if r is None:
                P(f"       {nm:14s} 协议 {pt:+} -> 重算缺  !!! 警告：无法核对")
                V["proto_mismatch"].append(f"{lbl}/{suite}/{nm}: 重算缺")
                continue
            bad = abs(r["point"] - pt) > PROTO_TOL
            if lo is not None:
                bad |= abs(r["ci95"][0] - lo) > PROTO_TOL or abs(r["ci95"][1] - hi) > PROTO_TOL
            ptxt = f"{pt:+}" + (f" [{lo:+}, {hi:+}]" if lo is not None else "")
            P(f"       {nm:14s} 协议 {ptxt:22s} 重算 {ci_txt(r)} -> " + ("!!! 警告：与协议不一致" if bad else "一致"))
            if bad:
                V["proto_mismatch"].append(f"{lbl}/{suite}/{nm}: 协议 {ptxt} 重算 {ci_txt(r)}")
    return out


def delta_section(new, pub, title, V):
    P(f"\n==== Δ_model = est({new.label}) − est({pub.label})（{new.suite}，按键配对 bootstrap，200 键；只印，判定用 P7 口径）")
    Y = {f"new:{c}": v for c, v in new.Y200.items()}
    Y.update({f"pub:{c}": v for c, v in pub.Y200.items()})
    boot = new.boot200
    pub_tab = PUB_EST_SP if new.suite == "spatial" else PUB_EST_OBJ
    new_tab = EST
    for nn, pn, note in DELTA_MAP:
        if pn not in pub_tab or nn not in new_tab:
            continue
        terms = [(cf, f"new:{c}") for cf, c in new_tab[nn]] + [(-cf, f"pub:{c}") for cf, c in pub_tab[pn]]
        r = boot.est(Y, terms)
        newcells = [c for _, c in new_tab[nn]]
        nsl = new.nsl(newcells, 4)
        stt = slice_status(nsl, NEED_MAIN)
        lab = f"{nn} − {pn}"
        if r is None:
            P(f"  {lab:32s} 缺")
            continue
        if stt == "不印":
            P(f"  {lab:32s} 不印（新模型共同整片 {nsl} < 3）")
            continue
        flag = "" if stt == "完成" else f"  【未完成：{nsl}/4】"
        tag = ""
        if abs(r["point"]) <= R10_BAND:
            tag = "  [在渲染节点噪声带内]" if new.kind == "jsonl" else "  [|Δ|<=3，GPU 组成见任务表]"
        P(f"  {lab:32s} {r['point']:+7.2f}  95% [{r['ci95'][0]:+7.2f}, {r['ci95'][1]:+7.2f}]  90% [{r['ci90'][0]:+7.2f}, {r['ci90'][1]:+7.2f}]"
          f"  n={r['n']}{flag}{'  （' + note + '）' if note else ''}{tag}")
        V["delta"][f"{new.label}/{new.suite}/{nn}"] = round(r["point"], 3)


def p7_line(nm, ref, new, a_lbl, b_lbl):
    if ref is None or new is None:
        P(f"  {nm:16s} 缺")
        return "缺"
    v = p7_verdict(ref, new)
    P(f"  {nm:16s} {a_lbl} {ci_txt(ref)}  {b_lbl} {ci_txt(new)} -> {v}")
    return v


def d5_block(m, res, V, key):
    r = res.get("d5")
    if r is None or r["state"] == "不印":
        P(f"  d5 门槛（§7.3）：d5 缺或不印，无法判定")
        V["d5_trigger"][key] = None
        return None
    trig = d5_trigger(r)
    V["d5_trigger"][key] = trig
    P(f"  d5 门槛（§7.3）：d5 = {ci_txt(r)}，95% 下界 {'<' if trig else '>='} -7 -> {TXT_D5_ON if trig else '不触发'}"
      + ("" if r["state"] == "完成" else f"  【未完成：{r['nsl']}/4】"))
    if trig:
        rp = run_est(m, REC_PRIME, 4)
        for nm, rr in rp.items():
            if rr is None or rr["state"] == "不印":
                P(f"    {nm:10s} 缺或不印")
            else:
                P(f"    {nm:10s} {ci_txt(rr)}  n={rr['n']}" + ("" if rr["state"] == "完成" else f"  【未完成：{rr['nsl']}/4】")
                  + ("（与 recovery(G) 恒等）" if nm == "rec'_fc" else ""))
        return f"替身管线代价 d5 = {ci_txt(r)}"
    return None


def frac_line(res):
    r, d = res.get("recovery(G)"), res.get("drop_nat(G)")
    if r is None or d is None or r["state"] == "不印" or d["state"] == "不印":
        P("  frac = recovery(G)/|drop_nat(G)|：缺")
    elif abs(d["point"]) < FRAC_MIN:
        P(f"  frac：|drop_nat(G)| 点 {abs(d['point']):.2f} < 7，不印")
    else:
        P(f"  frac = recovery(G)/|drop_nat(G)| = {r['point'] / abs(d['point']):.2f}（描述，点估计）")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_models1.txt")
    ap.add_argument("--verdict-json", default=None)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--zoo-runs", default="$PROJ_ROOT/vla_zoo/runs")
    ap.add_argument("--zoo-logs", default="$PROJ_ROOT/vla_zoo/logs")
    ap.add_argument("--pub-runs", default="$PROJ_ROOT/runs")
    ap.add_argument("--pi05", default="$PROJ_ROOT/pi05/v2")
    ap.add_argument("--gate-json", default=None)
    ap.add_argument("--protocol", default="$PROJ_ROOT/vla_zoo/analysis/protocol_models1.md")
    ap.add_argument("--decisions", default="$PROJ_ROOT/vla_zoo/analysis/protocol_models1_decisions.md")
    ap.add_argument("--zoo-cells", default="$PROJ_ROOT/vla_zoo/tools/zoo_cells.sh")
    ap.add_argument("--published-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exclude-keys", default=None, help="SENS: physics_breakdown json {model_dir|libero_suite: [[t,e],...]}")
    ap.add_argument("--bounds", default=None, help="SENS: QACC warning json {model_dir|cell_dir: {ep_offset: w}}")
    a = ap.parse_args(argv)
    P.lines = []
    out = H(a.out)
    if os.path.exists(out) and not a.overwrite:
        print(f"{out} 已存在（一次读数，M19）；要覆盖加 --overwrite", file=sys.stderr)
        return 3
    V = {"proto_mismatch": [], "d5_trigger": {}, "delta": {}, "printed": {}}

    SENS["exclude"], SENS["bounds"] = {}, None
    _me = os.path.abspath(__file__)
    _dir = os.path.dirname(_me)
    P("物理失效集剔除版，diff 见 prereg_models1_sens.diff")
    P("==== 剔除版 sha256（Fable 补裁一 §2.3 第 2 条）")
    for lbl, pth in [("prereg_models1.py（冻结原版，同目录）", os.path.join(_dir, "prereg_models1.py")),
                     ("prereg_models1_sens.py（剔除版）", _me),
                     ("prereg_models1_sens.diff", os.path.join(_dir, "prereg_models1_sens.diff")),
                     ("--exclude-keys", H(a.exclude_keys) if a.exclude_keys else None),
                     ("--bounds", H(a.bounds) if a.bounds else None)]:
        P(f"  {lbl:32s} " + (f"{sha256(pth) or 'MISSING'}  [{pth}]" if pth else "未给"))
    if a.exclude_keys:
        SENS["exclude"] = sens_load(a.exclude_keys, "exclude")
        for k, v in sorted(SENS["exclude"].items()):
            P(f"  剔除表 {k}: {len(v)} 键 {sorted(v)}")
    if a.bounds:
        SENS["bounds"] = sens_load(a.bounds, "bounds")
        for k, v in sorted(SENS["bounds"].items()):
            P(f"  警告表 {k}: " + ", ".join(f"e{o}={w}" for o, w in sorted(v.items(), key=lambda x: int(x[0]))) + f"（合计 {sum(v.values())}）")
    P("  口径：OFTi 按键剔除（某模型某套件任一格被检测器 D 标记的键，从该模型该套件所有格剔除，保持按键配对；在 §7.1 整片规则之后、"
      "只作用于估计量矩阵：估计量、判定、Holm、Δ_model 用剔除后的键；门槛、10 任务表、在线接触、E1/E2 仍按全部键照印，门槛不因剔除追认）。"
      "π0 / OVLA 不剔除，--bounds 给出时每个估计量行末印「失效集上界 ±x」，x = 100 × Σ|系数| × w_格 / n（w_格 = 该格在估计量所用片上的"
      " QACC 警告数之和；两格估计量即 (w_a + w_b) / n），向上取到 0.1。判定读法见补裁一 §2.3 第 3 条（与主读数不同则降为不确定，人工比对）。")
    V["sens_physics"] = {"exclude_keys": a.exclude_keys, "bounds": a.bounds,
                         "exclude_n": {k: len(v) for k, v in SENS["exclude"].items()},
                         "sha256": {"prereg_models1.py": sha256(os.path.join(_dir, "prereg_models1.py")), "prereg_models1_sens.py": sha256(_me),
                                    "exclude": sha256(H(a.exclude_keys)) if a.exclude_keys else None,
                                    "bounds": sha256(H(a.bounds)) if a.bounds else None}}
    P("prereg_models1.py：models1 预注册分析（protocol_models1.md §4、§5、§7、§8）")
    P("==== 冻结文件 sha256")
    me = os.path.abspath(__file__)
    for lbl, pth in [("protocol_models1.md", H(a.protocol)), ("protocol_models1_decisions.md", H(a.decisions)),
                     ("prereg_models1.py（本脚本）", me), ("zoo_cells.sh", H(a.zoo_cells))]:
        P(f"  {lbl:32s} {sha256(pth) or 'MISSING'}  [{pth}]")
    P(f"B={a.B} seed={SEED} zoo-runs={a.zoo_runs} pub-runs={a.pub_runs} pi05={a.pi05} gate-json={a.gate_json}"
      + ("  [--published-only：不读任何 vla_zoo 数据]" if a.published_only else ""))
    P(TXT_LIMIT)

    # ---------------------------------------------------------------- published
    pubs = {}
    for suite in ("spatial", "object"):
        pubs[("OFT", suite)] = make_pub("合训 OFT（已发表）", "npz", suite, a)
        pubs[("pi05", suite)] = make_pub("π0.5（已发表）", "jsonl", suite, a)
    pres = pub_section(pubs, V)
    V["published"] = {f"{k[0]}/{k[1]}/{nm}": (None if r is None else [round(r["point"], 3), round(r["ci95"][0], 3), round(r["ci95"][1], 3)])
                      for k, res in pres.items() for nm, r in res.items()}
    if a.published_only:
        _finish(a, out, V)
        return 0

    gate_json = None
    if a.gate_json:
        try:
            with open(H(a.gate_json)) as fh:
                gate_json = json.load(fh)
        except (OSError, ValueError) as e:
            gate_json = {"cells": {}, "_error": repr(e)}
    else:
        P("\n（未提供 --gate-json：§7.2 第 0 步一致性、吞吐、embody_toy 回归不在 dump 里，本脚本不核，见 M7；不改格状态）")

    zr = H(a.zoo_runs)
    obj = lambda cells: [(c, "obj_" + d, cp, tw) for c, d, cp, tw in cells]
    no_gt = [x for x in NEW_CELLS if x[0] != "G_T"]
    models = {
        ("OFTi", "spatial"): make_model("OFTi", "npz", "spatial", a, os.path.join(zr, "oft_sp"), "oft_sp", NEW_CELLS, dense=DENSE,
                                        gpu_expect=("A100", "40GB")),
        ("OFTi", "object"): make_model("OFTi", "npz", "object", a, os.path.join(zr, "oft_obj"), "oft_obj", obj(NEW_CELLS),
                                       gpu_expect=("A100", "40GB")),
        ("pi0", "spatial"): make_model("π0", "jsonl", "spatial", a, os.path.join(zr, "pi0"), "pi0", NEW_CELLS, expect_replan=5,
                                       gpu_expect=("T4",)),
        ("pi0", "object"): make_model("π0", "jsonl", "object", a, os.path.join(zr, "pi0"), "pi0", obj(NEW_CELLS), expect_replan=5,
                                      gpu_expect=("T4",)),
        ("OVLA", "spatial"): make_model("OVLA", "jsonl", "spatial", a, os.path.join(zr, "openvla"), "openvla", no_gt, expect_replan=1,
                                        gpu_expect=("T4",)),
    }
    RES, RES400 = {}, {}
    warn = {}
    for key, m in models.items():
        src = pubs[("OFT", m.suite)].data.get("SRC")
        m.compute_gates(src, gate_json)
        m.print_gates()
        for code, *_ in m.cells:
            V.setdefault("cells", {})[f"{key[0]}/{key[1]}/{code}"] = {
                "status": m.status(code), "slices": sorted(m.slices.get(code, set())), "n": len(m.data.get(code, {})),
                "gates": {g: v[0] for g, v in m.gates.get(code, {}).items()},
                "meta": {k: v for k, v in m.meta.get(code, {}).items()}}
        res = run_est(m, EST, 4)
        RES[key] = res
        print_est(m, res, "§4 估计量（200 键）")
        for nm, r in res.items():
            V["printed"][f"{key[0]}/{key[1]}/{nm}"] = None if r is None else r["state"]
        if m.dense:
            r4 = run_est(m, R2_EST, 8)
            RES400[key] = r4
            print_est(m, r4, "R2 通道加密估计量（400 键）")
            for nm, r in r4.items():
                V["printed"][f"{key[0]}/{key[1]}/{nm}@400"] = None if r is None else r["state"]
        frac_line(res)
        warn[key] = d5_block(m, res, V, f"{key[0]}/{key[1]}")

    # ---------------------------------------------------------------- §5.1 R family
    P("\n######## §5.1 决策族 R：独立 OFT 复现合训 OFT（spatial）")
    P("  预测（写死）：OFTi 的 G 掉分在 −55 到 −30 之间；rec_w > rec_a，channel > 0；recovery 过半。即复现。")
    m, res, r4 = models[("OFTi", "spatial")], RES[("OFTi", "spatial")], RES400.get(("OFTi", "spatial"), {})
    pr = pres[("OFT", "spatial")]
    dv = {}
    d, dp = res.get("drop_nat(G)"), pr.get("drop_nat(G)")
    if d is not None and dp is not None:
        v, txt = judge(m, [d], lambda: r1_verdict(d, dp))
        P(f"  R1 G 掉分：drop_nat(G) {ci_txt(d)} 对已发表 {ci_txt(dp)}（复现 = CI 重叠且上界 < −7；不复现 = 不重叠）-> {txt}")
    else:
        v = "缺"
        P("  R1 缺")
    dv["R1"] = v
    rc, rw = r4.get("channel"), r4.get("rec_w")
    if rc is not None and rw is not None and d is not None:
        v, txt = judge(m, [rc, rw, d], lambda: r2_verdict(rc, rw, d["point"]))
        P(f"  R2 通道排序（400 键）：channel {ci_txt(rc)}，rec_w {ci_txt(rw)}，|drop_nat(G)| 点 {abs(d['point']):.2f}（200 键，M2）"
          f"；复现 = channel 下界 > 0 且 rec_w 点 >= {HALF * abs(d['point']):.2f}；不复现 = channel 上界 < 0 或 rec_w 上界 < {QUARTER * abs(d['point']):.2f} -> {txt}")
    else:
        v = "缺"
        P("  R2 缺（400 键估计量或 drop_nat(G) 无数据）")
    dv["R2"] = v
    rr = res.get("recovery(G)")
    if rr is not None and d is not None:
        v, txt = judge(m, [rr, d], lambda: r3_verdict(rr, d["point"]))
        P(f"  R3 重绘恢复：recovery(G)（GTfc）{ci_txt(rr)}；复现 = 下界 > 0 且点 >= {HALF * abs(d['point']):.2f}；不复现 = 上界 < {QUARTER * abs(d['point']):.2f} -> {txt}")
    else:
        v = "缺"
        P("  R3 缺")
    dv["R3"] = v
    rg = res.get("recovery_GT(G)")
    vgt = "缺"
    if rg is not None and rg["state"] != "不印":
        vgt_raw = p7_verdict(pr.get("recovery(G)"), rg) if pr.get("recovery(G)") else "缺"
        vgt, txt = judge(m, [rg], lambda: vgt_raw)
        P(f"  并列：recovery_GT(G)（G_T，移动相机，SYNC 关）{ci_txt(rg)} 对已发表 {ci_txt(pr['recovery(G)']) if pr.get('recovery(G)') else '缺'}（P7 口径）-> {txt}")
    else:
        P("  并列：recovery_GT(G) 缺或不印")
    fam = r_family(dv)
    V.update({"R1": dv["R1"], "R2": dv["R2"], "R3": dv["R3"], "recGT_P7": vgt, "R_family": fam})
    P(f"  族结论：{fam}" + (f" —— {TXT_R_YES}" if fam == "复现" else "")
      + (f" —— {TXT_R_NO}；不复现的条：{', '.join(k for k, x in dv.items() if x == '不复现')}" if fam == "不复现" else ""))
    cam_diff = {dv["R3"], vgt} == {"复现", "不复现"}
    V["R3_vs_GT_differ"] = cam_diff
    if cam_diff:
        P(f"  {TXT_R_CAM}")
    if warn.get(("OFTi", "spatial")):
        P(f"  警示（§7.3）：{warn[('OFTi', 'spatial')]}")

    # ---------------------------------------------------------------- §5.2 F family
    P("\n######## §5.2 决策族 F：π0 家族对照（spatial）")
    P("  预测（写死，家族假设）：π0 的通道排序像 π0.5（channel <= 0，第三视角占优）；G 掉分在 −35 到 −10 之间。")
    m, res = models[("pi0", "spatial")], RES[("pi0", "spatial")]
    pp = pres[("pi05", "spatial")]
    rc = res.get("channel")
    if rc is not None:
        f1, txt = judge(m, [rc], lambda: f1_verdict(rc))
        P(f"  F1 通道排序：channel {ci_txt(rc)}（像 π0.5 = 上界 < 0；像 OFT = 下界 > 0）-> {txt}")
    else:
        f1 = "缺"
        P("  F1 缺")
    vd = vr = "缺"
    d, r = res.get("drop_nat(G)"), res.get("recovery(G)")
    if d is not None and r is not None and pp.get("drop_nat(G)") and pp.get("recovery(G)"):
        vd_raw, vr_raw = p7_verdict(pp["drop_nat(G)"], d), p7_verdict(pp["recovery(G)"], r)
        f2, txt = judge(m, [d, r], lambda: f2_verdict(vd_raw, vr_raw))
        P(f"  F2 掉分与恢复：drop_nat(G) {ci_txt(d)} 对 π0.5 {ci_txt(pp['drop_nat(G)'])} -> P7 {vd_raw}；"
          f"recovery(G) {ci_txt(r)} 对 π0.5 {ci_txt(pp['recovery(G)'])} -> P7 {vr_raw}（两条都复现 = 像 π0.5；都不复现 = 像 OFT）-> {txt}")
        vd, vr = vd_raw, vr_raw
    else:
        f2 = "缺"
        P("  F2 缺")
    rg = res.get("recovery_GT(G)")
    if rg is not None and rg["state"] != "不印" and pp.get("recovery(G)"):
        P(f"  并列（透明度）：recovery_GT(G) {ci_txt(rg)} 对 π0.5 recovery(G) {ci_txt(pp['recovery(G)'])} -> P7 {p7_verdict(pp['recovery(G)'], rg)}"
          + ("" if rg["state"] == "完成" else "  【未完成】"))
    V.update({"F1": f1, "F2": f2})
    if f1 == "像 π0.5" and V["R2"] == "复现":
        P(f"  解读：{TXT_F_STRONG}")
        V["F_reading"] = "strong"
    elif f1 == "像 OFT":
        P(f"  解读：{TXT_F_OFT}")
        V["F_reading"] = "oft"
    else:
        P("  解读：无写死句子适用（F1 与 R2 的组合不在 §5.2 解读表内）")
        V["F_reading"] = None
    P("  约束：π0 查询频率每 5 步（OFT 每 8 步、OVLA 每 1 步），通道估计量含反应频率差异；π0 与 π0.5 的差异不归到本体感受、chunk 或训练数据中任何一项（§6 第 3 条）")
    if warn.get(("pi0", "spatial")):
        P(f"  警示（§7.3）：{warn[('pi0', 'spatial')]}")

    # ---------------------------------------------------------------- §5.3 object
    P("\n######## §5.3 object（OFTi 与 π0 各自对本模型 S）")
    P("  预测（写死）：OFTi object 的 G 掉分 −26 到 −6；重绘不恢复（复现合训 OFT）；π0 object 的 recovery 小而含 0（像 π0.5）。")
    m, res = models[("OFTi", "object")], RES[("OFTi", "object")]
    po = pres[("OFT", "object")]
    d, r = res.get("drop_nat(G)"), res.get("recovery(G)")
    if d is not None and r is not None:
        o1, txt = judge(m, [r, d], lambda: o1_verdict(r, d["point"]))
        P(f"  O1 OFTi object 重绘不恢复：recovery(G) {ci_txt(r)}，drop_nat(G) {ci_txt(d)}，对已发表 recovery {ci_txt(po['recovery(G)']) if po.get('recovery(G)') else '缺'}"
          f"（复现 = 上界 < {HALF * abs(d['point']):.2f}；不复现 = 下界 > 0 且点 >= {HALF * abs(d['point']):.2f}）-> {txt}")
    else:
        o1 = "缺"
        P("  O1 缺")
    if o1 == "复现":
        P(f"  解读：{TXT_O1_YES}")
    elif o1 == "不复现":
        P(f"  解读：{TXT_O1_NO}")
    rg = res.get("recovery_GT(G)")
    if rg is not None and rg["state"] != "不印" and po.get("recovery(G)"):
        P(f"  并列（透明度）：recovery_GT(G) {ci_txt(rg)} 对已发表 {ci_txt(po['recovery(G)'])} -> P7 {p7_verdict(po['recovery(G)'], rg)}"
          + ("" if rg["state"] == "完成" else "  【未完成】"))
    if warn.get(("OFTi", "object")):
        P(f"  警示（§7.3）：{warn[('OFTi', 'object')]}")
    m, res = models[("pi0", "object")], RES[("pi0", "object")]
    po5 = pres[("pi05", "object")]
    d, r = res.get("drop_nat(G)"), res.get("recovery(G)")
    if d is not None and r is not None and po5.get("drop_nat(G)") and po5.get("recovery(G)"):
        vd_raw, vr_raw = p7_verdict(po5["drop_nat(G)"], d), p7_verdict(po5["recovery(G)"], r)
        o2, txt = judge(m, [d, r], lambda: o2_verdict(vd_raw, vr_raw))
        P(f"  O2 π0 object：drop_nat(G) {ci_txt(d)} 对 π0.5 {ci_txt(po5['drop_nat(G)'])} -> P7 {vd_raw}；recovery(G) {ci_txt(r)} 对 π0.5 "
          f"{ci_txt(po5['recovery(G)'])} -> P7 {vr_raw}（两条都复现 = 复现，都不复现 = 不复现，M10）-> {txt}")
        P(f"      描述：π0 object recovery(G) 95% CI {'含 0' if r['ci95'][0] <= 0 <= r['ci95'][1] else '不含 0'}（预测：小而含 0）")
    else:
        o2 = "缺"
        P("  O2 缺")
    if warn.get(("pi0", "object")):
        P(f"  警示（§7.3）：{warn[('pi0', 'object')]}")
    V.update({"O1": o1, "O2": o2})
    P("\n  drop_nat(G) 等的 object 对 spatial 符号一致性（第三批 §6 (iii) 口径 = P7，同一模型）：")
    for lbl in ("OFTi", "pi0"):
        rs, ro = RES[(lbl, "spatial")], RES[(lbl, "object")]
        for nm in ("drop_nat(G)", "recovery(G)", "d5"):
            a_, b_ = rs.get(nm), ro.get(nm)
            if a_ is None or b_ is None or a_["state"] == "不印" or b_["state"] == "不印":
                P(f"    {lbl:5s} {nm:12s} 缺或不印")
                continue
            fl = "" if (a_["state"] == "完成" and b_["state"] == "完成") else "  【未完成】"
            P(f"    {lbl:5s} {nm:12s} spatial {ci_txt(a_)}  object {ci_txt(b_)} -> {p7_verdict(a_, b_)}{fl}")

    # ---------------------------------------------------------------- §5.4 E layer
    P("\n######## §5.4 描述层 E：原版 OpenVLA（spatial，不进决策族）")
    m, res = models[("OVLA", "spatial")], RES[("OVLA", "spatial")]
    pt = m.per_task("SRC") if "SRC" in m.data else None
    if pt:
        k, n = sum(v[0] for v in pt.values()), sum(v[1] for v in pt.values())
        sr = 100 * k / max(n, 1)
        P(f"  基线：SRC {k}/{n} = {sr:.1f}%（须 >= 75%）-> {'过' if sr >= OVLA_SRC_MIN else '不过：先查接口'}" + ("" if n >= 200 else f"  【SRC 只有 {n} 键整片】"))
    else:
        P("  基线：SRC 缺")
    e1_ok = False
    for tag, (ca, cb) in (("E1", ("GTw", "G")), ("E2", ("GTa", "GTfc"))):
        if ca not in m.data or cb not in m.data:
            P(f"  {tag} agree({ca}, {cb})：缺")
            V[tag] = None
            continue
        nsl = m.nsl([ca, cb], 4)
        ks = sorted(set(m.data[ca]) & set(m.data[cb]))
        st = slice_status(nsl, NEED_MAIN)
        if st == "不印":
            P(f"  {tag} agree({ca}, {cb})：不印（共同整片 {nsl} < 3）")
            V[tag] = None
            continue
        ag = 100 * np.mean([m.data[ca][k]["succ"] == m.data[cb][k]["succ"] for k in ks])
        a0k = [k for k in ks if m.data[ca][k].get("a0") is not None and m.data[cb][k].get("a0") is not None]
        same = sum(1 for k in a0k if m.data[ca][k]["a0"] == m.data[cb][k]["a0"])
        P(f"  {tag} agree({ca}, {cb}) 逐键结局一致率 {ag:.1f}%（{len(ks)} 键，共同整片 {nsl}）；首块动作 a0 逐位相同 {same}/{len(a0k)} 键"
          + ("" if st == "完成" else "  【未完成】") + f"（期望 100%）")
        V[tag] = round(float(ag), 3)
        if tag == "E1":
            P(f"      {TXT_E1_NOISE}")
            e1_ok = bool(st == "完成" and ag >= E1_AGREE_MIN)
            if not e1_ok:
                P(f"      E1 {'未完成' if st != '完成' else f'一致率 < {E1_AGREE_MIN:.0f}%'} -> E3、E4 只印不读")
    V["E_read"] = e1_ok
    d, ra = res.get("drop_nat(G)"), res.get("rec_a")
    for tag, r_, fn, lbl in (("E3", d, e3_verdict, "drop_nat(G) 95% 上界 < −7（只看第三人称也掉分）"),
                             ("E4", ra, e4_verdict, "rec_a 95% 下界 > 0")):
        if r_ is None or r_["state"] == "不印":
            P(f"  {tag} 缺或不印")
            V[tag] = "缺"
            continue
        v, txt = judge(m, [r_], lambda: fn(r_))
        if not e1_ok and v in ("符合", "不符", "不确定"):
            txt = f"只印不读（E1 条件不满足）；按期望线为「{v}」"
            v = "只印不读:" + v
        P(f"  {tag} {lbl}：{ci_txt(r_)} -> {txt}")
        V[tag] = v
    do, df = res.get("drop_nat(G)"), RES[("OFTi", "spatial")].get("drop_nat(G)")
    if do is not None and df is not None and do["state"] != "不印" and df["state"] != "不印":
        e5 = e5_verdict(do, df)
        ao, af = abs_interval(do), abs_interval(df)
        fl = "" if (do["state"] == "完成" and df["state"] == "完成") else "  【未完成，只印】"
        P(f"  E5 |drop_nat(G)|：OVLA [{ao[0]:.2f}, {ao[1]:.2f}]，OFTi [{af[0]:.2f}, {af[1]:.2f}] -> {e5}{fl}")
        P(f"      「{TXT_E5_SMALL if e5 == '整体小于' else TXT_E5_BIG}」；{TXT_E5_CAVEAT}")
        V["E5"] = e5
    else:
        P("  E5 缺或不印")
        V["E5"] = "缺"
    P("  OVLA 天花板低、单步、无本体感受，所有数字只与自身 S 比；OVLA 只作「输入集合不同」的对照，不是「去掉腕相机的 OFT」（§6 第 4 条）")
    if warn.get(("OVLA", "spatial")):
        P(f"  警示（§7.3）：{warn[('OVLA', 'spatial')]}")

    # ---------------------------------------------------------------- Holm
    P("\n==== Holm 附表（透明度，M14）：bootstrap 双侧 p（H0 = 0）")
    fams = {
        "R": {"R1 drop_nat(G)": RES[("OFTi", "spatial")].get("drop_nat(G)"),
              "R2 channel(400)": RES400.get(("OFTi", "spatial"), {}).get("channel"),
              "R3 recovery(G)": RES[("OFTi", "spatial")].get("recovery(G)")},
        "F": {"F1 channel": RES[("pi0", "spatial")].get("channel"), "F2 drop_nat(G)": RES[("pi0", "spatial")].get("drop_nat(G)"),
              "F2 recovery(G)": RES[("pi0", "spatial")].get("recovery(G)")},
        "O": {"O1 recovery(G) OFTi": RES[("OFTi", "object")].get("recovery(G)"),
              "O2 drop_nat(G) π0": RES[("pi0", "object")].get("drop_nat(G)"),
              "O2 recovery(G) π0": RES[("pi0", "object")].get("recovery(G)")},
    }
    V["holm"] = {}
    for fn, ent in fams.items():
        ok = {k: r for k, r in ent.items() if r is not None and r["state"] != "不印"}
        adj = holm({k: r["p0"] for k, r in ok.items()})
        P(f"  族 {fn}：")
        for k, r in ent.items():
            if k not in ok:
                P(f"    {k:22s} 缺或不印")
            else:
                P(f"    {k:22s} 点 {r['point']:+7.2f}  p = {r['p0']:.4f}  Holm p = {adj.get(k, float('nan')):.4f}"
                  + ("" if r["state"] == "完成" else "  【未完成】"))
                V["holm"][k] = round(adj.get(k, float("nan")), 6)

    # ---------------------------------------------------------------- Δ_model
    P("\n######## 跨模型配对差 Δ_model（透明度，§4；判定仍用 §5 的 P7 口径）")
    for lbl, plbl in (("OFTi", "OFT"), ("pi0", "pi05")):
        for suite in ("spatial", "object"):
            delta_section(models[(lbl, suite)], pubs[(plbl, suite)], "", V)

    # ---------------------------------------------------------------- descriptive
    P("\n######## 描述：10 任务表、GPU 组成、在线接触")
    for m in models.values():
        m.task_table()
    for m in models.values():
        m.contacts()

    _finish(a, out, V)
    return 0


def _finish(a, out, V):
    P("\n==== 冻结文件 sha256（重印）")
    me = os.path.abspath(__file__)
    for lbl, pth in [("protocol_models1.md", H(a.protocol)), ("protocol_models1_decisions.md", H(a.decisions)),
                     ("prereg_models1.py（本脚本）", me), ("zoo_cells.sh", H(a.zoo_cells))]:
        P(f"  {lbl:32s} {sha256(pth) or 'MISSING'}")
    if V["proto_mismatch"]:
        P("\n!!! 警告：已发表数字重算与协议文字不一致：")
        for x in V["proto_mismatch"]:
            P(f"    {x}")
    with open(out, "w") as f:
        f.write("\n".join(P.lines) + "\n")
    if a.verdict_json:
        with open(H(a.verdict_json), "w") as f:
            json.dump(V, f, ensure_ascii=False, indent=1, default=str)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    sys.exit(main())
