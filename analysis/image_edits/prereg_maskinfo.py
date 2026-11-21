"""maskinfo pre-registered read-out: protocol_maskinfo.md v3 (frozen version; sha256 checked at run time via
--expect-protocol-sha = the frozen value, NOT a constant here; deviations live in protocol_maskinfo_deviations.md). Implements §1.2-§1.4, §4 item 6 (physics-failure sensitivity), §5.1-§5.6.
Read unit = block (model x suite); each block is read once, results are APPENDED to one text file and one verdict json.

Usage (cluster, CPU only):
  python prereg_maskinfo.py \
      --protocol $PROJ_ROOT/embody_toy/protocol_maskinfo.md --expect-protocol-sha <sha256> \
      --runs $PROJ_ROOT/runs_maskinfo --gate gate_maskinfo.json --blocks oft_spatial,ofti_spatial \
      --out results_maskinfo.txt --verdict-json verdict_maskinfo.json [--lag-steps 16] [--not-before ISO8601] \
      [--deviations protocol_maskinfo_deviations.md]
Synthetic tests only: add --synthetic (every input/output dir must contain the marker file SYNTHETIC_TEST_DIR) and
optionally --now ISO8601. Tests: test_prereg_maskinfo.py.

Inputs: --runs ROOT/<block>/<cell>/t{task}_e{ep}_i{init}.npz, block in BLOCKS, cell in C0..C7, keys (task 0-9, ep 20-29),
shards ep 20-24 / 25-29. Required npz fields: REQUIRED_FIELDS below (missing -> exit 9, nothing guessed). Per-step
fields have length T+1 (row i = state after i actions, row 0 = initial); `action` is T x 7 (row i = i-th action).

Exit codes: 2 protocol sha mismatch / protocol file missing; 3 block already read (in --out or --verdict-json) or
--out / --verdict-json inconsistent; 4 time gate (--not-before not reached, or --now without --synthetic);
5 block lock exists; 6 synthetic-mode directory check failed (or synthetic / real mixed in one verdict json);
7 wording self-check failed (outputs are still written, flagged); 8 block incomplete (some cell misses some key of
some shard) -> nothing read; 9 input / argument format error (missing npz field, wrong shape, duplicate key,
task/ep/init/cell/block mismatch, unreadable gate json, unknown block name) -> nothing read.
All refusals 2/3/4/5/6/8/9 happen BEFORE any lock is taken and before any outcome is read.

Inference (§5.1, "same method as prereg_maskrepl.py"):
  * cluster = one (task, ep) key; inside a block all eight cells are paired through it and resampled jointly.
  * `Boot` and `mcnemar_exact` are copied verbatim from prereg_maskrepl.py (itself verbatim prereg_toy1.py):
    np.random.default_rng(0).integers(0, n, (B, n)) -> bincount weight matrix W; resampled difference
    = 100 * (W @ y_a - W @ y_b) / (W @ m); percentile CIs 95% = [2.5, 97.5], 90% = [5, 95].
  * McNemar exact two-sided p = 2 * P(Bin(b+c, 1/2) <= min(b, c)), capped at 1.  B = 10000, seed 0.

实现取舍（协议没写死的地方；按协议精神定，理由写在每条后面）
 M1  块内键全集 = 八格都有 npz 的键（块必须完整，故为 100 键）减去门槛 json 里该块任一格 bad_keys 的并集。理由：八格联合
     重采样（§3）要求每个键八格齐全；某格的坏键若只从该格剔除，各对比的键集就不同，联合重采样失效。
 M2  块完整 = 八格每格两片 (ep 20-24、25-29) 的 100 个键都有且只有一个 npz；不完整拒绝读该块（退出码 8）。同键多个 npz、
     npz 内 task / ep / init / mi_cell / mi_block 与路径不符、八格同键 init 不同、缺字段、形状不符、mi_pad_geoms 不是
     JSON 列表：退出码 9。设计键之外的文件（例如预留的 ep 30-39）不打开，只印个数。理由：§5.5「只在八格两片全部完成之后」
     与「禁止读任何单片」；键身份是配对前提。
 M3  结构核对（字段、dtype、形状、元数据）在取锁之前做完，只核对不保留、不打印任何结局相关的值（包括轨迹长度）；所以一旦
     取锁，后面的读数不会因格式问题中断。理由：锁的意义是「只读一次」，格式错误不应该白白烧掉一次读数。
 M4  多个块一起读时先对全部块做预检（已读、锁、完整、格式），任一块不过整次拒绝，不取任何锁。理由：避免半成品状态。
 M5  块级 H（整块 does not count，照印不判）：门槛 json 缺该块；block_ok 非 true；card_type 不是 h200 / h100-96（v3 §4 第 2 项排除 a100-80；h100-96
     按冻结前写入 protocol_maskinfo_deviations.md 的卡型分配，只在其渲染逐位与首帧一致性通过后允许）；门槛 json
     的 protocol_sha256 与本协议不符；本脚本按 dumps 的 gpu 字段重数，块内出现不止一种卡型或与 card_type 不符
     （"H200" -> h200，"H100" 且显存 ≥ 90000 MiB -> h100-96、否则 h100-47，"A100 80GB" -> a100-80，其余 -> other）。理由：最后一条是 §4 第 2 项卡型锁定的复核，取门槛与 dumps
     中更严的一方（同 prereg_maskrepl M3 的做法）。
 M6  格级 H：门槛 json 里该格 status 非 PASS（含 UNK、缺该格）-> 涉及该格的比较标「门槛未过（H 行）」，照算照印，标签写
     「不判（按线本应：X，仅供参考）」。G 行判定涉及 C0 / C3；用词规则涉及 C0（增强判据 SR(C4) − SR(C0)）与 G；任一为 H
     则相应部分写「不判」，其余比较照判。
 M7  擦除损失 F 只作描述（§1.2）。G 行 = lo95(F) <= 0；G 不封锁 P1 / P2 / Δ3-Δ6，照判照写。
 M8  P1 / P2 / Δ4 / Δ5 用 §1.3 规则：「提高」= 点 >= 10 且 lo95 > 0；「未提高」= hi95 < 10；其余「不确定」。Δ3 另加「有害」
     = 点 <= -10 且 hi95 < 0（优先于未提高）；Δ6 只有「有害」（与 Δ3 同法，协议只写了「有害」一词没给线）、「无害」
     （lo95 > -10）、其余「不确定」。
 M9  次级四条（Δ3-Δ6，v3 §5.3）：McNemar 精确 p 做 Holm（m = 4，固定；因门槛未过而不判的比较 p 记 1 参与排序，保守）。
     方向性标签（Δ3 / Δ6 的「有害」、Δ4 / Δ5 的「提高」）要点估计与 CI 条件和 Holm 校正后 p < 0.05 同时成立，CI 条件满足而
     Holm 未过记「不确定（Holm 未过）」；等价型标签（「未提高」、Δ6 的「无害」）只看 CI。P1 / P2 不校正（§5.3 合取已保守），
     P1 的用词（增强 / 部分恢复 / 提高）也不经 Holm（§5.3 的 Holm 只列 Δ3-Δ6）。bootstrap CI 照印不校正。
 M10 用词三档（v3 §1.3，互斥）：P1 为「提高」时，(i) lo95(SR(C4) − SR(C0)) > 0 称「增强」；(ii) 否则 lo95(F) > 0 称「部分恢复」；
     (iii) 否则（G 行生效且不够增强）称「提高」。A 句对应写「带来 Δ1 点的增强」「带来 Δ1 点的部分性能恢复」，第 (iii) 档把
     A 句的「带来 Δ1 点的部分性能恢复（或增强……）」一段换成「后，C4 高于 C3 Δ1 点」，不出现恢复、增强二词，其余逐字。
 M11 物理失效敏感性（§4 第 6 项）：主读数不剔除；另把门槛 json 该块 physics_breakdown 各格键的并集从全部八格剔除，重抽一次
     B x n' 权重矩阵（seed 0）再判一次；G、P1、P2、Δ3-Δ6 任一条最终标签（含 Holm）或 P1 用词不同 -> 该块整体降为「不确定」，
     写明哪条变了，A-G 行不出。J 行（描述性的复现比较）与 §5.4 时机照用主读数印出。并集为空时写「无失效键」。
 M12 解读表：A / B / C 按主标签组合（B 的「Δ3 也提高」用 Holm 后的标签；C = P1 未提高）；D / E / F / G 独立叠加；F 成立时
     不改 E（v2 起删了「在面积效应之外」）。P1 / P2 都已判但不落入 A / B / C 的组合，印「不属于 A / B / C 行」并列标签，不另造
     句子。句中数字为点估计，保留一位小数；C 句的「差 Δ4」填成「差 Δ4（x 点）」。A 句的滞后步数取 --lag-steps（默认 16）。
     B 行要附的 §2.4 第 7 项两路数字与 §2.3 的 C7 改写不在本脚本输入里，只印提醒，须人工补。
 M13 J 行：D(C1)、D(C2)、D(C3) 的三档标签（lo95 >= -10 几乎不掉；hi95 <= -30 大幅掉分；其余中间）与 maskrepl 的
     （几乎不掉 / 中间 / 大幅掉分，即 -4 / -20 / -51 的读数）逐格比较，任一不同即出 J 句；每个块都比。C0-C3 任一为 H 则 J 不判。
 M14 I 行：每次运行按 verdict json 里所有已读块重新生成汇总（不改已读块的判定）；未读块写「未读」。Object 块与同一模型的
     Spatial 块配对：两块都已读且同为 A -> 「跨套件一致」；否则按 Object 块自身分 (i) lo95(F) <= 0 -> 「Object 上擦除本身
     不敏感」，(ii) lo95(F) > 0 且 P1 未提高 -> 「Object 上存在擦除损失但当前轮廓不恢复」；都不属于则只列两块结果；Spatial
     未读另注明。Object 块被降为「不确定」或为 H 时不套 (i)(ii)。
 M15 §5.4 事件（只描述；状态下标 i = 已执行 i 个动作后的状态，0..T；目标物 = mi_obj_pos 第 0 个物体）：
     t_near  首个 i 使 |mi_grip_pos[i] − mi_obj_pos[i, 0]| < 7.5 cm；
     t_close 首个动作下标 i 使 action[i, -1] > 0；「近物区外合拢」= t_close < t_near（未进入近物区记 t_near = 无穷），分母为
             有合拢指令的集；
     t_pad1  首个 i 使 lpad 或 rpad；t_pad2 首个 i 使 lpad 且 rpad；
     t_lift  首个 i 使 mi_obj_pos[i, 0, z] − mi_obj_pos[0, 0, z] > 2 cm；
     t_carry 首个 i 使 j = i..i+15（16 步）每步都满足 lpad[j] ∧ rpad[j] ∧ ¬target_support[j]，且该步位移
             |(obj[j+1] − obj[j]) − (grip[j+1] − grip[j])| < 1 cm（所以 i + 16 <= T）；
     t_drop  （v3）t_carry 之后首个「双侧接触连续丢失 ≥ 4 步」的段（段首 > t_carry，只看 lpad ∧ rpad 是否同时为真，单侧不算），
             且该段前 4 个状态目标物都不在终态，取段首；不满足的段跳过，看下一段；一集只记第一个满足的段；不足 4 步的瞬时丢失
             不算（含被集末截断的不足 4 步的段）。
     终态（执行方口径，近似）：Spatial「在盘上」= 与 mi_obj_pos[:, 1]（盘）水平距离 < 6 cm 且目标物 z 高于盘 z（stages.py
     的 6 cm）；Object「在篓内」= 与篓（第 1 个物体）水平距离 < 8 cm 且目标物 z 低于篓 z + 10 cm。
     失败分型：success / no_reach（无 t_near）/ no_grasp（有 t_near 无 t_lift）/ drop（有 t_drop）/ other。
     量：每格各事件发生比例与中位步、t_close − t_near、t_pad2 − t_close、t_carry − t_pad2 的中位数、近物区外合拢比例、分型
     分布；对 C0 的配对差 = 两格都发生该事件（区间则两端事件都发生）的键上的配对均值差（步），CI 用同一 W 的掩码比值法。
     「抓稳」一词不出现在输出里（持续携带写作「持续携带」）。mi_contact_grip_target / grip_interactive 只核字段不用。
 M16 只读一次：每块一个锁 prereg_maskinfo_<block>.lock（默认在 --out 所在目录），预检全过之后、读任何结局之前以独占方式
     创建；已存在 -> 5。块已出现在 --out（块标题行）或 --verdict-json 的 blocks 里 -> 3。--out 与 --verdict-json 只有
     一个存在 -> 3。verdict json 读-改-写：只新增块，已有块逐字节不动；先写临时文件再 os.replace。
 M17 时间门：--not-before（默认不设，§5.5 不设更早时间门）；当前时刻早于它 -> 4。--now 只许与 --synthetic 同用（否则 4）。
     --synthetic 要求 --runs、门槛 json、--out、--verdict-json、锁所在目录都含 SYNTHETIC_TEST_DIR；已有 verdict json 的
     synthetic 标记与本次不同 -> 6。
 M18 措辞自检：本次追加的全部文本检查 v3 §5.6 全部禁用词（与 v2 逐字相同）与 §1.4 表末点名的词（斜杠写法拆成多个词；「或任何……」类无法
     穷举的，取其可检出的关键片段：充分、面积效应、来自目标损伤、更可能来自位置、而非开合、来自位姿、来自开合、原本依靠
     同一线索、通过它起效、混杂、过早、抓稳、填充 + 遮挡效应）。命中仍照写，末尾印 FAIL，退出码 7。
 M19 协议冻结后一字不改（v3 §5.5），--expect-protocol-sha 给冻结值，不符退出 2；门槛 json 的 protocol_sha256 与之不符整块 H
     （M5）。偏离写在 protocol_maskinfo_deviations.md：给了 --deviations 且文件存在，就在首部与 verdict json 印其 sha256，
     不读内容、不参与判定；给了但不存在照印「不存在」。已读块的脚本 sha 或协议 sha 与本次不同：照读，印警告（不拒绝）。
 M20 GPU 组成表：每格 gpu 字段计数与 host 数，只描述；卡型核对见 M5。§5.1 的「§5 六个输入差别量」不在本脚本输入内，不印。
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

# ============================================================================ constants (pre-registered lines)
BLOCKS = ["oft_spatial", "ofti_spatial", "oft_object", "ofti_object"]
BLOCK_MODEL = {"oft_spatial": "oft", "ofti_spatial": "ofti", "oft_object": "oft", "ofti_object": "ofti"}
BLOCK_SUITE = {"oft_spatial": "spatial", "ofti_spatial": "spatial", "oft_object": "object", "ofti_object": "object"}
MODEL_NAME = {"oft": "合训 OFT", "ofti": "OFTi"}
CELLS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CELL_DESC = {"C0": "原图", "C1": "seg-black", "C2": "box-black", "C3": "alpha0", "C4": "alpha0 + 当前轮廓",
             "C5": "alpha0 + 滞后轮廓", "C6": "box-black 保护物体", "C7": "异位遮罩对照"}
TASKS = list(range(10))
SHARDS = [("ep20-24", list(range(20, 25))), ("ep25-29", list(range(25, 30)))]
DESIGN_KEYS = [(t, e) for t in TASKS for nm, eps in SHARDS for e in eps]
ALLOWED_CARDS = ("h200", "h100-96")  # protocol v3 4 item 2 (a100-80 excluded) + card plan in protocol_maskinfo_deviations.md
B_DEFAULT, SEED = 10000, 0
SYN_MARKER = "SYNTHETIC_TEST_DIR"
STEP_BOOL = ["mi_contact_lpad", "mi_contact_rpad", "mi_contact_grip_target", "mi_contact_target_support",
             "mi_contact_grip_interactive"]
REQUIRED_FIELDS = (["success", "task", "ep", "init", "action", "mi_obj_names", "mi_obj_pos", "mi_grip_pos"] + STEP_BOOL
                   + ["mi_pad_geoms", "gpu", "host", "mi_cell", "mi_block"])

REC_POINT, REC_LO, NOREC_HI = 10.0, 0.0, 10.0  # §1.3
HARM_POINT, HARM_HI = -10.0, 0.0             # §1.3 Δ3 (and Δ6, M8)
HARMLESS_LO = -10.0                          # §1.3 Δ6
G_LO = 0.0                                   # §1.4 G: lo95(F) <= 0
ENH_LO = 0.0                                 # §1.3: lo95(SR(C4) - SR(C0)) > 0 -> 增强
HOLM_M, ALPHA = 4, 0.05                      # §5.3
ALMOST_NO_DROP_LO95, LARGE_DROP_HI95 = -10.0, -30.0   # §1.4 J
NEAR_DIST, LIFT_DZ, CARRY_STEPS, CARRY_TOL, DROP_RUN = 0.075, 0.02, 16, 0.01, 4   # §5.4 (v3)
TERM = {"spatial": (0.06, None), "object": (0.08, 0.10)}             # M15 (executor's approximation)

L_REC, L_NOREC, L_UNC, L_HARM, L_HARMLESS = "提高", "未提高", "不确定", "有害", "无害"
L_UNC_HOLM = "不确定（Holm 未过）"
L_NO, L_MID, L_BIG = "几乎不掉", "中间", "大幅掉分"
W_ENH, W_PART, W_G = "增强", "部分恢复", "提高"
MASKREPL_J = {"C1": L_NO, "C2": L_MID, "C3": L_BIG}   # maskrepl ep 40-49: -4 / -20 / -51

# contrast name -> (a, b, kind); estimate = SR(a) - SR(b)
CONTRASTS = {
    "F": ("C0", "C3", "desc"),
    "P1": ("C4", "C3", "rec"),
    "P2": ("C4", "C5", "rec"),
    "D3": ("C5", "C3", "rec_harm"),
    "D4": ("C1", "C4", "rec"),
    "D5": ("C6", "C2", "rec"),
    "D6": ("C7", "C0", "harm_harmless"),
    "D7": ("C1", "C3", "desc"),
    "EN": ("C4", "C0", "desc"),
}
SECONDARY = ["D3", "D4", "D5", "D6"]
JUDGED = ["P1", "P2"] + SECONDARY
SENS_KEYS = ["G", "P1", "P2"] + SECONDARY + ["P1_word"]
CNAME = {"F": "擦除损失 F = SR(C0) − SR(C3)（描述；lo95 ≤ 0 触发 G 行）", "P1": "P1 Δ1 = SR(C4) − SR(C3)",
         "P2": "P2 Δ2 = SR(C4) − SR(C5)", "D3": "Δ3 = SR(C5) − SR(C3)", "D4": "Δ4 = SR(C1) − SR(C4)",
         "D5": "Δ5 = SR(C6) − SR(C2)", "D6": "Δ6 = SR(C7) − SR(C0)", "D7": "Δ7 = SR(C1) − SR(C3)（描述）",
         "EN": "增强判据 SR(C4) − SR(C0)", "G": "G 行（lo95(F) ≤ 0）", "P1_word": "P1 用词（部分恢复 / 增强）"}

# §1.4 v3 sentences, verbatim from the protocol table (numbers filled in)
ROW = {
    "A": ("在该 checkpoint 与套件上，其他输入保留时，在 alpha0 擦除图上添加当前可见轮廓（保护对象后剩余的部分）{clause}"
          "，且优于滞后 {lag} 步的轮廓（{d2} 点）。因此精确分割遮罩保留了与控制相关的、当前遮挡条件下的身体相关二维线索，"
          "『精确遮罩后不掉分』不能单独作为『不依赖身体视觉』的证据。"),
    "B": "当前与滞后轮廓均带来帮助，本实验尚未分离形状与残留状态信息（滞后轮廓仍携带慢变的位姿、开合与可见性）。",
    "C": ("当前轮廓叠加（保护对象后剩余的部分）不足以复现 seg-black 的表现；seg-black 与之的差 Δ4（{d4} 点）来源尚未定位"
          "（候选包括内部填充、遮挡关系、外侧描边、对象保护截断、背景与阴影差异）。"),
    "D": "滞后轮廓比无轮廓更差，该策略的输出对画面里夹爪轮廓的时序敏感；可能来源包括视觉与本体感受不一致，本实验不区分。",
    "E": "在 box 遮罩上恢复这些原本可见的可交互物体像素带来 {d5} 点收益；box 式遮罩的归因被任务信息损伤混淆。",
    "F": "与 box 同面积同形状、按规则放置到别处且保留夹爪可见的异位遮挡也损伤性能 {d6} 点。",
    "G": "该 checkpoint / 套件上未建立明显的擦除损失，因此对 Δ1 不作『补回损失』的解释",
    "H": "does not count，照印不判",
    "I_both": "在两个 OFT checkpoint 上",
    "I_one": "在 {x} checkpoint 上",
    "I_obj_ok": "跨套件一致",
    "I_obj_i": "Object 上擦除本身不敏感",
    "I_obj_ii": "Object 上存在擦除损失但当前轮廓不恢复",
    "J": "在 ep 20–29 上 seg / box / alpha0 的模式与 ep 40–49 的读数（−4 / −20 / −51）不一致",
}
A_CLAUSE = {W_PART: "带来 {d1} 点的部分性能恢复", W_ENH: "带来 {d1} 点的增强", W_G: "后，C4 高于 C3 {d1} 点"}
NOTE_B = "须附：操纵检查（§2.4 第 7 项）滞后在两路画面里各造成了多大改变（本脚本不读操纵检查输出，须人工补）"
NOTE_F = "须核：§2.3 若 Δocc_P(C7) = occ_P(C7) − occ_P(C0) 跨帧中位数 > 0.05，F 行写法改为「异位遮挡（另遮住 x% 可交互物体像素）」（本脚本不读操纵检查输出）"
NOTE_G = "G 行不封锁 P1 / P2：轮廓添加效果（Δ1，用词按 §1.3 规则）与时序比较（Δ2、Δ3）照判照写。"
LIMITS = ["(a) 本体感受未被直接干预；本实验未分离视觉开合、投影位姿、遮挡关系与跨模态一致性的贡献。",
          "(b) 两路同时干预，P2 不能归因到其中一路；腕部主要改变开合与可见性时序，第三视角还改变投影位姿时序。",
          "(c) 阳性结果反映的是当前遮挡条件下的身体相关二维线索，不一定来自读取夹爪自身位姿。",
          "(d) 阴影影响未单独识别。"]

# v3 §5.6 (Astra list verbatim, rulings additions, this protocol's own) + §1.4 table-end words (M18)
FORBIDDEN = ["夹爪是位置标尺", "模型不使用几何反馈", "重绘收益等于纯视觉损失比例", "架构决定通道偏好", "已经调和 Trace 与 VPro",
             "身体状态信息对控制充分", "充分", "形状的存在本身即足够", "身体信息不在轮廓", "C1 − C4 是填充 + 遮挡效应",
             "填充 + 遮挡效应", "来自目标损伤", "面积效应", "在面积效应之外", "阴影不是因素", "过早合拢", "过早", "抓稳",
             "恢复更可能来自位置", "更可能来自位置", "而非开合", "来自位姿", "来自开合", "seg-black 原本依靠同一线索",
             "原本依靠同一线索", "Robotiq 重绘通过它起效", "通过它起效", "轮廓机制未复现", "混杂",
             "模型读取轮廓", "模型利用轮廓", "视觉伺服", "必要", "遮罩实验无效", "遮罩实验误导", "Trace 的结论错误",
             "所有 VLA", "模板", "先验", "模型用轮廓定位指尖"]

H = os.path.expanduser


class Refuse(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


# ============================================================================ output buffer
class Out:
    def __init__(self):
        self.lines = []
        self.quiet = False

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        self.lines.append(s)
        if not self.quiet:
            print(s, flush=True)

    def text(self):
        return "\n".join(self.lines) + "\n"


P = Out()


def sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def wording_violations(text):
    return [w for w in FORBIDDEN if w in text]


def f1(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


# ============================================================================ inference (copied from prereg_maskrepl.py)
class Boot:
    """Verbatim from prereg_maskrepl.py / prereg_toy1.py Boot: paired cluster bootstrap over keys, one draw matrix."""

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
        return {"point": float(point), "ci95": tuple(float(x) for x in np.percentile(ok, [2.5, 97.5])),
                "ci90": tuple(float(x) for x in np.percentile(ok, [5, 95])), "n": int(m.sum()),
                "sr_a": float(100 * ya.mean()), "sr_b": float(100 * yb.mean()),
                "b": nb, "c": nc, "mcnemar": mcnemar_exact(nb, nc)}

    def mean_diff(self, ya, yb, scale=1.0):
        """Paired mean difference on keys where both are non-NaN (same W, masked-ratio method; for §5.4 timing)."""
        m = ~np.isnan(ya) & ~np.isnan(yb)
        if m.sum() == 0:
            return None
        d = np.where(m, ya - yb, 0.0)
        den = self.W @ m.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = scale * (self.W @ d) / den
        ok = bs[~np.isnan(bs)]
        return {"point": float(scale * (ya[m] - yb[m]).mean()),
                "ci95": tuple(float(x) for x in np.percentile(ok, [2.5, 97.5])), "n": int(m.sum())}


def mcnemar_exact(b, c):
    """Verbatim from prereg_maskrepl.py mcnemar_exact (two-sided exact binomial, capped at 1)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pvals):
    """Holm step-down adjusted p-values for a dict name -> p (m = len(pvals))."""
    names = sorted(pvals, key=lambda k: (pvals[k], k))
    m, run, out = len(names), 0.0, {}
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def label_rec(r):
    pt, (lo, hi) = r["point"], r["ci95"]
    if pt >= REC_POINT and lo > REC_LO:
        return L_REC
    if hi < NOREC_HI:
        return L_NOREC
    return L_UNC


def is_harm(r):
    return r["point"] <= HARM_POINT and r["ci95"][1] < HARM_HI


def ci_label(kind, r):
    """Label by CI rule only (before Holm)."""
    if r is None:
        return None
    if kind == "rec":
        return label_rec(r)
    if kind == "rec_harm":
        return L_HARM if is_harm(r) else label_rec(r)
    if kind == "harm_harmless":
        if is_harm(r):
            return L_HARM
        if r["ci95"][0] > HARMLESS_LO:
            return L_HARMLESS
        return L_UNC
    return None


def g_active(rF):
    return None if rF is None else bool(rF["ci95"][0] <= G_LO)


def p1_word(p1_label, rEN, g):
    """§1.3 wording rule (M10). None when P1 is not 提高 or the inputs are not judged."""
    if p1_label != L_REC or rEN is None or g is None:
        return None
    if rEN["ci95"][0] > ENH_LO:
        return W_ENH
    return W_G if g else W_PART


def j_label(r):
    if r is None:
        return None
    lo, hi = r["ci95"]
    if lo >= ALMOST_NO_DROP_LO95:
        return L_NO
    if hi <= LARGE_DROP_HI95:
        return L_BIG
    return L_MID


def card_of(gpu):
    g = str(gpu)
    if "H200" in g:
        return "h200"
    if "H100" in g:
        m = re.search(r"(\d+)\s*MiB", g)
        return "h100-96" if m and int(m.group(1)) >= 90000 else "h100-47"
    if "A100" in g and "80GB" in g.replace(" ", ""):
        return "a100-80"
    return "other"


# ============================================================================ events (§5.4, M15)
EVENTS = ["t_near", "t_close", "t_pad1", "t_pad2", "t_lift", "t_carry", "t_drop"]
INTERVALS = [("close_minus_near", "t_close", "t_near"), ("pad2_minus_close", "t_pad2", "t_close"),
             ("carry_minus_pad2", "t_carry", "t_pad2")]
FTYPES = ["success", "no_reach", "no_grasp", "drop", "other"]


def _first(mask):
    return int(np.argmax(mask)) if mask.any() else None


def episode_events(z, suite):
    obj_all = np.asarray(z["mi_obj_pos"], float)
    obj, ref = obj_all[:, 0, :], obj_all[:, 1, :]
    grip = np.asarray(z["mi_grip_pos"], float)
    act = np.asarray(z["action"], float)
    lp, rp = np.asarray(z["mi_contact_lpad"], bool), np.asarray(z["mi_contact_rpad"], bool)
    sup = np.asarray(z["mi_contact_target_support"], bool)
    succ = bool(np.asarray(z["success"]).item())
    T = len(act)
    both = lp & rp
    with np.errstate(invalid="ignore"):
        near = np.linalg.norm(grip - obj, axis=1) < NEAR_DIST
        lifted = (obj[:, 2] - obj[0, 2]) > LIFT_DZ
        dstep = np.linalg.norm(np.diff(obj, axis=0) - np.diff(grip, axis=0), axis=1) < CARRY_TOL   # length T
        horiz = np.linalg.norm(obj[:, :2] - ref[:, :2], axis=1)
        r_xy, dz = TERM[suite]
        if suite == "spatial":
            term = (horiz < r_xy) & (obj[:, 2] > ref[:, 2])
        else:
            term = (horiz < r_xy) & (obj[:, 2] < ref[:, 2] + dz)
    ok = both[:T] & ~sup[:T] & dstep                     # step j valid for j in 0..T-1
    t_carry = None
    if T >= CARRY_STEPS:
        win = np.convolve(ok.astype(int), np.ones(CARRY_STEPS, int), "valid") == CARRY_STEPS
        t_carry = _first(win)
    t_drop = None
    if t_carry is not None:
        lost = ~both
        i, n = t_carry + 1, len(lost)
        while i < n:
            if lost[i] and (i == 0 or not lost[i - 1] or i == t_carry + 1):
                j = i
                while j < n and lost[j]:
                    j += 1
                if j - i >= DROP_RUN and not term[i:i + DROP_RUN].any():
                    t_drop = i
                    break
                i = j
            else:
                i += 1
    ev = {"t_near": _first(near), "t_close": _first(act[:, -1] > 0), "t_pad1": _first(lp | rp), "t_pad2": _first(both),
          "t_lift": _first(lifted), "t_carry": t_carry, "t_drop": t_drop}
    if succ:
        ft = "success"
    elif ev["t_near"] is None:
        ft = "no_reach"
    elif ev["t_lift"] is None:
        ft = "no_grasp"
    elif ev["t_drop"] is not None:
        ft = "drop"
    else:
        ft = "other"
    ev.update(succ=float(succ), ftype=ft)
    return ev


# ============================================================================ preflight (no outcomes read)
def _sc(z, k):
    v = np.asarray(z[k])
    if v.shape != ():
        raise ValueError(f"字段 {k} 不是标量")
    return v.item()


def preflight_block(runs, block):
    """Structure-only check of one block. Returns ({cell: {key: path}}, n_ignored). Raises Refuse(8 / 9)."""
    bdir = os.path.join(H(runs), block)
    files, missing, ignored = {}, [], 0
    for c in CELLS:
        cdir = os.path.join(bdir, c)
        found = {}
        for f in sorted(glob.glob(os.path.join(cdir, "*.npz"))):
            m = re.fullmatch(r"t(\d+)_e(\d+)_i(\d+)\.npz", os.path.basename(f))
            if not m:
                raise Refuse(9, f"{block}/{c}: 文件名不合规 {os.path.basename(f)}")
            k = (int(m.group(1)), int(m.group(2)))
            if k not in DESIGN_KEYS:
                ignored += 1
                continue
            if k in found:
                raise Refuse(9, f"{block}/{c}: 键 t{k[0]} e{k[1]} 有多个 npz（不同 init？）")
            found[k] = (f, int(m.group(3)))
        for nm, eps in SHARDS:
            miss = [(t, e) for t in TASKS for e in eps if (t, e) not in found]
            if miss:
                missing.append(f"{c} 片 {nm} 缺 {len(miss)} 键")
        files[c] = found
    if missing:
        raise Refuse(8, f"块 {block} 不完整，拒绝读该块：" + "；".join(missing))
    inits = {}
    for c in CELLS:
        for k, (f, init) in files[c].items():
            bn = f"{block}/{c}/{os.path.basename(f)}"
            try:
                with np.load(f, allow_pickle=True) as z:
                    F = set(z.files)
                    lack = [x for x in REQUIRED_FIELDS if x not in F]
                    if lack:
                        raise Refuse(9, f"{bn}: 缺字段 {lack}")
                    if (int(_sc(z, "task")), int(_sc(z, "ep")), int(_sc(z, "init"))) != (k[0], k[1], init):
                        raise Refuse(9, f"{bn}: npz 内 task / ep / init 与文件名不符")
                    if str(_sc(z, "mi_cell")) != c or str(_sc(z, "mi_block")) != block:
                        raise Refuse(9, f"{bn}: mi_cell / mi_block 与路径不符")
                    s = np.asarray(z["success"])
                    if s.shape != () or s.dtype.kind not in "biu":
                        raise Refuse(9, f"{bn}: success 不是布尔 / 整数标量")
                    for x in ("gpu", "host"):
                        _sc(z, x)
                    pads = json.loads(str(_sc(z, "mi_pad_geoms")))
                    if not isinstance(pads, list) or not pads:
                        raise Refuse(9, f"{bn}: mi_pad_geoms 不是非空 JSON 列表")
                    a, op, gp, on = (np.asarray(z[x]) for x in ("action", "mi_obj_pos", "mi_grip_pos", "mi_obj_names"))
                    T = a.shape[0] if a.ndim == 2 else -1
                    ok = (a.ndim == 2 and a.shape[1] == 7 and T >= 1 and on.ndim == 1 and len(on) >= 2
                          and op.shape == (T + 1, len(on), 3) and gp.shape == (T + 1, 3)
                          and a.dtype.kind == "f" and op.dtype.kind == "f" and gp.dtype.kind == "f")
                    for x in STEP_BOOL:
                        v = np.asarray(z[x])
                        ok = ok and v.shape == (T + 1,) and v.dtype.kind in "bi"
                    if not ok:
                        raise Refuse(9, f"{bn}: action / mi_obj_pos / mi_grip_pos / mi_contact_* / mi_obj_names "
                                        f"形状或 dtype 不符（规格见脚本头；mi_obj_names 至少 2 个：目标物与盘 / 篓）")
            except Refuse:
                raise
            except Exception as ex:  # noqa
                raise Refuse(9, f"{bn}: 读取失败 {type(ex).__name__}")
            if k in inits and inits[k] != init:
                raise Refuse(9, f"{block}: 键 t{k[0]} e{k[1]} 在不同格的 init 不同")
            inits[k] = init
    return {c: {k: v[0] for k, v in files[c].items()} for c in CELLS}, ignored


# ============================================================================ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--expect-protocol-sha", required=True)
    ap.add_argument("--runs", required=True, help="ROOT containing <block>/<cell>/t*_e*_i*.npz")
    ap.add_argument("--gate", required=True)
    ap.add_argument("--blocks", required=True, help="comma list of " + ",".join(BLOCKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdict-json", required=True)
    ap.add_argument("--lock-dir", default=None)
    ap.add_argument("--lag-steps", type=int, choices=[16, 24], default=16)
    ap.add_argument("--deviations", default=None, help="protocol_maskinfo_deviations.md; sha256 printed only")
    ap.add_argument("--not-before", default=None, help="ISO8601; refuse to read before this time")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--now", default=None, help="ISO8601; synthetic only")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.lock_dir is None:
        a.lock_dir = os.path.dirname(os.path.abspath(H(a.out)))
    return a


def lock_path(a, block):
    return os.path.join(H(a.lock_dir), f"prereg_maskinfo_{block}.lock")


def block_header(block):
    return f"######## 块 {block}"


def _aware(s):
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def preflight(a):
    """All refusals before any lock."""
    if a.now is not None and not a.synthetic:
        raise Refuse(4, "拒绝运行：--now 只能与 --synthetic 一起用")
    blocks = [b.strip() for b in a.blocks.split(",") if b.strip()]
    if not blocks or any(b not in BLOCKS for b in blocks) or len(set(blocks)) != len(blocks):
        raise Refuse(9, f"--blocks 不合规：{a.blocks}（可选 {','.join(BLOCKS)}，不重复）")
    psha = sha256(H(a.protocol))
    if psha is None or psha != a.expect_protocol_sha.strip().lower():
        raise Refuse(2, f"拒绝运行：协议 sha256 {psha} != --expect-protocol-sha {a.expect_protocol_sha}")
    real_now = dt.datetime.now(dt.timezone.utc)
    now = _aware(a.now) if a.now else real_now
    if a.not_before:
        nb = _aware(a.not_before)
        if now < nb:
            raise Refuse(4, f"拒绝运行：当前 {now.isoformat()} 早于 --not-before {nb.isoformat()}")
    if a.synthetic:
        dirs = {"--runs": H(a.runs), "--gate": os.path.dirname(os.path.abspath(H(a.gate))),
                "--out": os.path.dirname(os.path.abspath(H(a.out))),
                "--verdict-json": os.path.dirname(os.path.abspath(H(a.verdict_json))), "lock": H(a.lock_dir)}
        bad = [f"{k} {d}" for k, d in dirs.items() if not os.path.exists(os.path.join(d, SYN_MARKER))]
        if bad:
            raise Refuse(6, "拒绝运行：--synthetic 只能读写含 SYNTHETIC_TEST_DIR 标记的目录；缺标记：" + "；".join(bad))
    out_ex, vj_ex = os.path.exists(H(a.out)), os.path.exists(H(a.verdict_json))
    if out_ex != vj_ex:
        raise Refuse(3, f"拒绝运行：--out 与 --verdict-json 只有一个存在（out {out_ex}，verdict {vj_ex}），先人工核对")
    old = None
    if vj_ex:
        try:
            with open(H(a.verdict_json)) as f:
                old = json.load(f)
        except (OSError, ValueError) as e:
            raise Refuse(9, f"已有 verdict json 读不了：{e!r}")
        if bool(old.get("synthetic")) != bool(a.synthetic):
            raise Refuse(6, "拒绝运行：已有 verdict json 的 synthetic 标记与本次不同（合成与正式不得混写）")
        with open(H(a.out)) as f:
            otext = f.read()
        for b in blocks:
            if b in (old.get("blocks") or {}) or (block_header(b) + "（") in otext:
                raise Refuse(3, f"拒绝覆盖：块 {b} 已读过（已在 --out 或 --verdict-json 中；每块只读一次）")
    for b in blocks:
        if os.path.exists(lock_path(a, b)):
            raise Refuse(5, f"拒绝运行：锁文件 {lock_path(a, b)} 已存在（每块只读一次；重读须在附录 A 记偏离后人工删锁）")
    try:
        with open(H(a.gate)) as f:
            gate = json.load(f)
        if not isinstance(gate.get("blocks"), dict):
            raise ValueError("缺 blocks")
    except (OSError, ValueError, AttributeError) as e:
        raise Refuse(9, f"门槛 json 读不了或格式不符：{e!r}")
    gate_sha = sha256(H(a.gate))
    files, ignored = {}, {}
    for b in blocks:
        files[b], ignored[b] = preflight_block(a.runs, b)
    return blocks, files, ignored, gate, gate_sha, psha, now, real_now, old


def main(argv=None):
    a = parse_args(argv)
    P.quiet = a.quiet
    try:
        blocks, files, ignored, gate, gate_sha, psha, now, real_now, old = preflight(a)
    except Refuse as r:
        sys.stderr.write(r.msg + "\n")
        sys.exit(r.code)
    ssha = sha256(os.path.abspath(__file__))
    taken = []
    for b in blocks:
        try:
            with open(lock_path(a, b), "x") as lf:
                json.dump({"time": real_now.isoformat(), "block": b, "argv": sys.argv if argv is None else argv,
                           "script_sha256": ssha, "synthetic": a.synthetic}, lf, ensure_ascii=False)
            taken.append(b)
        except FileExistsError:
            sys.stderr.write(f"拒绝运行：锁文件 {lock_path(a, b)} 已存在（并发？）；本次已取的锁 {taken} 保留\n")
            sys.exit(5)

    V = old or {"synthetic": a.synthetic, "blocks": {}, "runs": []}
    old_blocks_dump = json.dumps(V.get("blocks", {}), sort_keys=True, ensure_ascii=False)
    new_blocks = run_analysis(a, blocks, files, ignored, gate, gate_sha, psha, ssha, now, real_now, V)
    for b, rec in new_blocks.items():
        V["blocks"][b] = rec
    summary_I = section_I(V["blocks"])
    text = P.text()
    viol = wording_violations(text)
    tail = f"\n==== 措辞自检（本次追加文本）：{'FAIL 命中 ' + '、'.join(viol) if viol else 'PASS（无禁用词）'}\n\n"
    for b in blocks:
        V["blocks"][b]["wording_check_of_run"] = {"status": "FAIL" if viol else "PASS", "hits": viol}
    V["summary_I"] = summary_I
    V.setdefault("runs", []).append({"real_time": real_now.isoformat(), "read_time": now.isoformat(), "blocks": blocks,
                                     "protocol_sha256": psha, "script_sha256": ssha, "gate_sha256": gate_sha,
                                     "deviations": ({"path": a.deviations, "sha256": sha256(H(a.deviations))}
                                                    if a.deviations else None),
                                     "wording_check": {"status": "FAIL" if viol else "PASS", "hits": viol}})
    kept = {k: v for k, v in V["blocks"].items() if k not in new_blocks}
    assert json.dumps(kept, sort_keys=True, ensure_ascii=False) == old_blocks_dump, "已读块被改动"
    with open(H(a.out), "a") as fo:
        fo.write(text + tail)
    tmp = H(a.verdict_json) + ".tmp"
    with open(tmp, "w") as fj:
        json.dump(V, fj, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, H(a.verdict_json))
    if not a.quiet:
        print(tail, end="")
    if viol:
        sys.exit(7)


# ============================================================================ analysis
def run_analysis(a, blocks, files, ignored, gate, gate_sha, psha, ssha, now, real_now, V):
    if a.synthetic:
        P("！！合成数据测试运行（--synthetic）：不是正式读数 ！！")
    P("======== maskinfo 预注册读数（protocol_maskinfo.md v3 §1.2–§1.4、§4 第 6 项、§5.1–§5.6）本次追加")
    P(f"协议 sha256 {psha}  {a.protocol}")
    P(f"脚本 sha256 {ssha}  {os.path.abspath(__file__)}")
    P(f"门槛 json sha256 {gate_sha}  {a.gate}")
    if a.deviations:
        dsha = sha256(H(a.deviations))
        P(f"偏离文件 sha256 {dsha or '不存在'}  {a.deviations}（只印不读，不参与判定）")
    P(f"读数真实时刻 {real_now.isoformat()}" + (f"（合成 --now {now.isoformat()}）" if a.now else "")
      + (f"  --not-before {a.not_before}" if a.not_before else "  未设 --not-before"))
    P(f"本次读的块 {blocks}；已读块 {sorted(V.get('blocks', {}))}；B={a.B} seed={SEED}；滞后步数 {a.lag_steps}")
    for b, rec in sorted(V.get("blocks", {}).items()):
        if rec.get("script_sha256") != ssha:
            P(f"  警告：已读块 {b} 用的脚本 sha {rec.get('script_sha256')} 与本次不同（M19）")
        if rec.get("protocol_sha256") != psha:
            P(f"  警告：已读块 {b} 用的协议 sha {rec.get('protocol_sha256')} 与本次不同（M19）")
    gsha_ok = str(gate.get("protocol_sha256", "")).lower() == psha
    P(f"门槛 json protocol_sha256 {gate.get('protocol_sha256')}  {'= 本协议' if gsha_ok else '<<< 不符，全部块 H（M5）'}")
    out = {}
    for b in blocks:
        out[b] = analyze_block(a, b, files[b], ignored[b], gate, gsha_ok, gate_sha, psha, ssha, now, real_now)
    return out


def load_block(files, suite):
    data, meta = {}, {}
    for c in CELLS:
        data[c], meta[c] = {}, {}
        for k, f in files[c].items():
            with np.load(f, allow_pickle=True) as z:
                data[c][k] = episode_events(z, suite)
                meta[c][k] = (str(_sc(z, "gpu")), str(_sc(z, "host")))
    return data, meta


def judge(boot, Y, cell_h, block_h):
    """Estimates and final labels for one universe."""
    est = {k: boot.diff(Y, a_, b_) for k, (a_, b_, _) in CONTRASTS.items()}
    hc = {k: [c for c in CONTRASTS[k][:2] if c in cell_h] for k in CONTRASTS}
    raw = {k: ci_label(CONTRASTS[k][2], est[k]) for k in JUDGED}
    pv = {k: (est[k]["mcnemar"] if (est[k] and not hc[k]) else 1.0) for k in SECONDARY}
    padj = holm(pv)
    raw_holm = dict(raw)
    for k in SECONDARY:
        if raw[k] in (L_REC, L_HARM) and not padj[k] < ALPHA:
            raw_holm[k] = L_UNC_HOLM
    lab = {k: (None if (block_h or hc[k]) else raw_holm[k]) for k in JUDGED}
    g_raw = g_active(est["F"])
    lab["G"] = None if (block_h or hc["F"]) else g_raw
    w_raw = p1_word(raw["P1"], est["EN"], g_raw)
    lab["P1_word"] = None if (block_h or hc["EN"] or hc["F"]) else p1_word(lab["P1"], est["EN"], lab["G"])
    raw_holm["G"], raw_holm["P1_word"] = g_raw, w_raw
    return {"est": est, "h_cells": hc, "raw": raw, "raw_holm": raw_holm, "holm": padj, "p": pv, "labels": lab}


def analyze_block(a, b, files, n_ignored, gate, gsha_ok, gate_sha, psha, ssha, now, real_now):
    suite = BLOCK_SUITE[b]
    P(f"\n{block_header(b)}（{MODEL_NAME[BLOCK_MODEL[b]]} × {suite.capitalize()}）")
    gb = (gate.get("blocks") or {}).get(b)
    block_h = []
    if not gsha_ok:
        block_h.append("门槛 json 的 protocol_sha256 与本协议不符")
    if gb is None:
        block_h.append("门槛 json 缺该块")
        gb = {}
    if gb.get("block_ok") is not True:
        block_h.append(f"block_ok={gb.get('block_ok')}" + (f"（{'；'.join(map(str, gb.get('block_fail_reasons') or []))}）"
                                                          if gb.get("block_fail_reasons") else ""))
    card = gb.get("card_type")
    if card not in ALLOWED_CARDS:
        block_h.append(f"card_type={card}（只允许 {'/'.join(ALLOWED_CARDS)}）")
    gcells = gb.get("cells") or {}

    data, meta = load_block(files, suite)
    gpu_tab = {c: dict(Counter(meta[c][k][0] for k in meta[c])) for c in CELLS}
    hosts = {c: len({meta[c][k][1] for k in meta[c]}) for c in CELLS}
    dump_cards = sorted({card_of(g) for c in CELLS for g in gpu_tab[c]})
    if len(dump_cards) != 1 or (card in ALLOWED_CARDS and dump_cards[0] != card):
        block_h.append(f"dumps 的 gpu 字段卡型 {dump_cards} 与门槛 card_type={card} 不符或不止一种（M5）")
    if n_ignored:
        P(f"  设计键（task 0–9 × ep 20–29）之外的文件 {n_ignored} 个，未打开（M2）")

    def bad(c):
        return {tuple(int(x) for x in k) for k in ((gcells.get(c) or {}).get("bad_keys") or [])}

    bad_union = set().union(*[bad(c) for c in CELLS])
    universe = [k for k in DESIGN_KEYS if k not in bad_union]
    phys = {c: {tuple(int(x) for x in k) for k in ((gb.get("physics_breakdown") or {}).get(c) or [])} for c in CELLS}
    phys_union = set().union(*phys.values())
    cell_h = {c: (gcells.get(c) or {}).get("status") for c in CELLS if (gcells.get(c) or {}).get("status") != "PASS"}

    P("\n==== 门槛与用键（M1、M5、M6）")
    P(f"  门槛：card_type {card}  block_ok {gb.get('block_ok')}  block_fail_reasons {gb.get('block_fail_reasons') or '无'}")
    for c in CELLS:
        gc = gcells.get(c) or {}
        P(f"  {c} {CELL_DESC[c]:16s} status {gc.get('status', '缺')}  bad_keys {len(bad(c))}"
          + (f"  原因 {gc.get('reasons')}" if gc.get("reasons") else "") + f"  物理失效集 {len(phys[c])}")
    P(f"  bad_keys 并集 {len(bad_union)} 键（从全部八格剔除）；键全集 n={len(universe)}")
    if block_h:
        P("  块级：门槛未过（H 行），整块 does not count，照印不判。原因：" + "；".join(block_h))
    elif cell_h:
        P(f"  块级：通过；门槛未过的格 {cell_h}，涉及这些格的比较为 H。")
    else:
        P("  块级：通过，八格门槛均 PASS。")

    P("\n==== GPU 组成（每格 gpu 字段计数；host 数）")
    for c in CELLS:
        P(f"  {c} {gpu_tab[c]}  host 数 {hosts[c]}")
    P(f"  dumps 推得卡型 {dump_cards}")

    def yvecs(keys):
        idx = {k: i for i, k in enumerate(keys)}
        out = {}
        for c in CELLS:
            y = np.full(len(keys), np.nan)
            for k in keys:
                y[idx[k]] = data[c][k]["succ"]
            out[c] = y
        return out

    Y = yvecs(universe)
    boot = Boot(max(len(universe), 1), a.B)
    J = judge(boot, Y, cell_h, bool(block_h))
    sens_univ = [k for k in universe if k not in phys_union]
    S = judge(Boot(max(len(sens_univ), 1), a.B), yvecs(sens_univ), cell_h, bool(block_h)) if sens_univ else None
    changed = []
    if phys_union and S is not None:
        changed = [k for k in SENS_KEYS if J["labels"][k] != S["labels"][k]]
    elif phys_union:
        changed = ["敏感性键全集为空"]
    downgraded = bool(changed) and not block_h

    def dtxt(r, p=True):
        if r is None:
            return "无估计"
        s = (f"{r['point']:+7.2f}  95% [{r['ci95'][0]:+7.2f}, {r['ci95'][1]:+7.2f}]  "
             f"90% [{r['ci90'][0]:+7.2f}, {r['ci90'][1]:+7.2f}]")
        return s + (f"  n={r['n']}  b={r['b']} c={r['c']}  McNemar p={r['mcnemar']:.4g}" if p else "")

    def lab_txt(R, k):
        lb = R["labels"][k]
        if lb is not None or (k == "P1_word" and R["raw_holm"]["P1_word"] is None):
            return {True: "生效", False: "不生效"}.get(lb, lb) if k == "G" else (lb if lb is not None else "不适用（P1 非提高）")
        why = "块门槛未过（H）" if block_h else "格门槛未过（H 行）"
        rl = R["raw_holm"].get(k)
        if k == "G":
            rl = {True: "生效", False: "不生效"}.get(rl, rl)
        return f"不判：{why}（按线本应：{rl}，仅供参考）"

    P(f"\n==== 成功率（键全集 n={len(universe)}；百分比）")
    for c in CELLS:
        P(f"  {c} {CELL_DESC[c]:16s} SR={100 * np.nanmean(Y[c]):5.1f}" + (f"  <<< 格门槛 {cell_h[c]}（H）" if c in cell_h else ""))
    P("  按任务（每任务 10 键中的键全集部分；SR %）")
    P("    格  " + " ".join(f"t{t:<5d}" for t in TASKS))
    for c in CELLS:
        row = []
        for t in TASKS:
            ks = [k for k in universe if k[0] == t]
            row.append(f"{100 * np.mean([data[c][k]['succ'] for k in ks]):5.1f} " if ks else "  nan ")
        P(f"    {c}  " + " ".join(row))

    P("\n==== §5.1 估计量（百分点；配对聚类 bootstrap，八格联合重采样，B=%d，seed 0；b = 前格成而后格败，c 相反）" % a.B)
    for k in ["F", "P1", "P2"] + SECONDARY + ["D7", "EN"]:
        r = J["est"][k]
        extra = f"  Holm 校正 p={J['holm'][k]:.4g}" if k in SECONDARY else ""
        lt = f"  标签：{lab_txt(J, k)}" if k in JUDGED else ""
        if k == "F":
            lt = f"  {CNAME['G']}：{lab_txt(J, 'G')}"
        if k == "EN":
            lt = f"  {CNAME['P1_word']}：{lab_txt(J, 'P1_word')}"
        P(f"  {CNAME[k]}: {dtxt(r)}{extra}{lt}")
    P("  复现量 D_c = SR(c) − SR(C0)（描述；J 行三档标签：lo95 ≥ −10 几乎不掉；hi95 ≤ −30 大幅掉分；其余中间）")
    Dc = {}
    for c in CELLS[1:]:
        Dc[c] = boot.diff(Y, c, "C0")
        P(f"  D({c}) {CELL_DESC[c]:16s} {dtxt(Dc[c])}  三档：{j_label(Dc[c])}" + ("  （H，仅供参考）" if c in cell_h else ""))

    P("\n==== 物理失效敏感性（§4 第 6 项；主读数不剔除）")
    P(f"  各格失效集数 {dict((c, len(phys[c])) for c in CELLS)}；并集 {len(phys_union)} 键")
    if not phys_union:
        P("  无失效键，敏感性读数与主读数相同。")
    elif S is not None:
        P(f"  剔除并集后 n={len(sens_univ)}")
        for k in ["F", "P1", "P2"] + SECONDARY + ["EN"]:
            P(f"    {CNAME[k]}: {dtxt(S['est'][k], p=False)}")
        for k in SENS_KEYS:
            P(f"    {CNAME[k]}：主 {J['labels'][k]} / 敏感性 {S['labels'][k]}")
    if downgraded:
        P("  敏感性判定不同：该块整体降为「不确定」。变化的条目：" + "、".join(CNAME.get(x, x) for x in changed))

    # ---- §1.4
    P("\n==== §1.4 解读表（块内；句子取自协议 v3 表格）")
    rows = []
    L = J["labels"]
    e = J["est"]

    def add(row, text):
        rows.append({"row": row, "text": text})
        P(f"  [{row}] {text}")

    if block_h:
        status = "H"
        add("H", f"整块门槛未过：{ROW['H']}（{'；'.join(block_h)}）")
    elif downgraded:
        status = "不确定"
        P("  该块降为「不确定」（物理失效敏感性），A–G 行不出；变化的条目：" + "、".join(CNAME.get(x, x) for x in changed))
    else:
        status = "judged"
        hs = [k for k in ["F", "P1", "P2"] + SECONDARY + ["EN"] if J["h_cells"][k]]
        if hs:
            add("H", f"门槛未过的格 {sorted(cell_h)}；涉及的比较 {'、'.join(CNAME[k].split(' = ')[0] for k in hs)}：{ROW['H']}")
        if L["G"]:
            add("G", ROW["G"])
            P(f"      {NOTE_G}")
        if L["P1"] == L_REC and L["P2"] == L_REC:
            w = L["P1_word"]
            d1 = f1(e["P1"]["point"])
            clause = A_CLAUSE[w].format(d1=d1) if w else f"后，C4 高于 C3 {d1} 点（用词不判：C0 门槛未过）"
            add("A", ROW["A"].format(clause=clause, lag=a.lag_steps, d2=f1(e["P2"]["point"])))
        elif L["P1"] == L_REC and L["P2"] in (L_NOREC, L_UNC) and L["D3"] == L_REC:
            add("B", ROW["B"])
            P(f"      {NOTE_B}")
        elif L["P1"] == L_NOREC:
            add("C", ROW["C"].format(d4=f1(e["D4"]["point"])))
        elif L["P1"] is not None and L["P2"] is not None:
            P(f"  主比较标签 P1 {L['P1']}，P2 {L['P2']}，Δ3 {L['D3']}：不属于 A / B / C 行，只列标签")
        if L["P1"] == L_REC and L["P1_word"]:
            P(f"  P1 用词（§1.3）：{L['P1_word']}")
        if L["D3"] == L_HARM:
            add("D", ROW["D"])
        if L["D5"] == L_REC:
            add("E", ROW["E"].format(d5=f1(e["D5"]["point"])))
        if L["D6"] == L_HARM:
            add("F", ROW["F"].format(d6=f1(abs(e["D6"]["point"]))))
            P(f"      {NOTE_F}")
    jh = [c for c in ("C0", "C1", "C2", "C3") if c in cell_h]
    jl = {c: j_label(Dc[c]) for c in ("C1", "C2", "C3")}
    if block_h or jh:
        jinfo = {"judged": False, "labels": jl, "reason": "H"}
        P(f"  [J] 不判（{'整块 H' if block_h else '门槛未过的格 ' + str(jh)}）；三档按线本应 {jl}")
    else:
        diff = [c for c in jl if jl[c] != MASKREPL_J[c]]
        jinfo = {"judged": True, "labels": jl, "maskrepl": MASKREPL_J, "differ": diff}
        if diff:
            add("J", ROW["J"] + f"（本块 C1 / C2 / C3：{jl['C1']} / {jl['C2']} / {jl['C3']}；maskrepl：几乎不掉 / 中间 / 大幅掉分）")
        else:
            P(f"  [J] 不触发：D(C1) / D(C2) / D(C3) 三档 {jl['C1']} / {jl['C2']} / {jl['C3']} 与 maskrepl 相同")
    P("  限制（§5.6 必写）：")
    for s in LIMITS:
        P(f"    {s}")

    timing = section_timing(data, universe, boot, cell_h)

    return {"model": BLOCK_MODEL[b], "suite": suite, "status": status,
            "row_letters": [r["row"] for r in rows], "rows": rows,
            "is_A": status == "judged" and any(r["row"] == "A" for r in rows),
            "real_time": real_now.isoformat(), "read_time": now.isoformat(), "protocol_sha256": psha, "script_sha256": ssha,
            "gate_sha256": gate_sha, "synthetic": a.synthetic, "lag_steps": a.lag_steps,
            "gate": {"card_type": card, "block_ok": gb.get("block_ok"), "block_h_reasons": block_h, "cell_h": cell_h,
                     "bad_keys_union": sorted(bad_union), "dump_cards": dump_cards},
            "n_universe": len(universe), "gpu": gpu_tab, "hosts": hosts,
            "sr": {c: float(100 * np.nanmean(Y[c])) for c in CELLS},
            "estimates": {k: _rj(v) for k, v in J["est"].items()}, "D_c": {c: _rj(v) for c, v in Dc.items()},
            "labels": J["labels"], "labels_by_line": J["raw_holm"], "labels_ci_only": J["raw"],
            "holm_p": J["holm"], "mcnemar_p_for_holm": J["p"],
            "sensitivity": {"phys_counts": {c: len(phys[c]) for c in CELLS}, "union": sorted(phys_union),
                            "n": len(sens_univ), "labels": S["labels"] if S else None,
                            "estimates": {k: _rj(v) for k, v in S["est"].items()} if S else None,
                            "changed": changed, "downgraded": downgraded},
            "J": jinfo, "timing": timing}


def _rj(r):
    if r is None:
        return None
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}


def section_timing(data, universe, boot, cell_h):
    P("\n==== §5.4 合拢、接触、携带时机（只描述，不进判定；单位：步；配对差只在两格都发生该事件的键上）")
    uidx = {k: i for i, k in enumerate(universe)}

    def vec(c, fn):
        y = np.full(len(universe), np.nan)
        for k in universe:
            v = fn(data[c][k])
            y[uidx[k]] = np.nan if v is None else float(v)
        return y

    fns = {ev: (lambda d, ev=ev: d[ev]) for ev in EVENTS}
    for nm, x, y in INTERVALS:
        fns[nm] = (lambda d, x=x, y=y: None if (d[x] is None or d[y] is None) else d[x] - d[y])

    def outside(d):
        if d["t_close"] is None:
            return None
        return 1.0 if (d["t_near"] is None or d["t_close"] < d["t_near"]) else 0.0

    V = {c: {m: vec(c, fn) for m, fn in fns.items()} for c in CELLS}
    out = {}

    def med(v):
        v = v[~np.isnan(v)]
        return (float(np.median(v)) if len(v) else float("nan")), int(len(v))

    n = len(universe)
    P("  各事件：发生比例 %（中位步）；t_near 近物区（7.5 cm）、t_pad1 首次指垫接触、t_pad2 双侧接触、t_lift 抬升、"
      "t_carry 持续携带、t_drop 掉落")
    P("  格  " + "  ".join(f"{ev:>16s}" for ev in EVENTS))
    for c in CELLS:
        out[c] = {"events": {}, "intervals": {}}
        parts = []
        for ev in EVENTS:
            m_, k_ = med(V[c][ev])
            out[c]["events"][ev] = {"pct": 100.0 * k_ / n if n else float("nan"), "median": m_, "n": k_}
            parts.append(f"{f1(out[c]['events'][ev]['pct']):>6s}%({f1(m_):>6s})")
        P(f"  {c}  " + "  ".join(f"{s:>16s}" for s in parts) + ("  （H，仅供参考）" if c in cell_h else ""))
    P("  间隔中位数（n）与近物区外合拢比例、失败分型 " + "/".join(FTYPES))
    for c in CELLS:
        parts = []
        for nm, _x, _y in INTERVALS:
            m_, k_ = med(V[c][nm])
            out[c]["intervals"][nm] = {"median": m_, "n": k_}
            parts.append(f"{nm} {f1(m_)} ({k_})")
        ov = np.array([np.nan if outside(data[c][k]) is None else outside(data[c][k]) for k in universe], float)
        ov = ov[~np.isnan(ov)]
        out[c]["outside_near_close_pct"] = float(100 * ov.mean()) if len(ov) else float("nan")
        out[c]["n_close"] = int(len(ov))
        fts = Counter(data[c][k]["ftype"] for k in universe)
        out[c]["ftypes"] = {f: int(fts.get(f, 0)) for f in FTYPES}
        P(f"  {c}  " + "；".join(parts) + f"；近物区外合拢 {f1(out[c]['outside_near_close_pct'])}% ({len(ov)})；"
          + "/".join(str(fts.get(f, 0)) for f in FTYPES))
    P("  对 C0 的配对差（该格 − C0；均值差，95% CI，步）")
    for c in CELLS[1:]:
        out[c]["vs_C0"] = {}
        parts = []
        for m in EVENTS + [i[0] for i in INTERVALS]:
            r = boot.mean_diff(V[c][m], V["C0"][m])
            out[c]["vs_C0"][m] = _rj(r)
            parts.append(f"{m} " + ("无共同键" if r is None else
                                    f"{r['point']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] n={r['n']}"))
        P(f"  {c}  " + "；".join(parts))
    return out


def section_I(blocks):
    """§1.4 I row over every block read so far (M14). Only summarises stored verdicts."""
    P("\n==== §1.4 I 行汇总（跨块；只汇总已读块，未读块写「未读」）")

    def st(b):
        r = blocks.get(b)
        if r is None:
            return "未读"
        return f"{r['status']}，行 {'/'.join(r['row_letters']) or '无'}"

    for b in BLOCKS:
        P(f"  {b}: {st(b)}")
    res = {"blocks": {b: st(b) for b in BLOCKS}}
    sp = ["oft_spatial", "ofti_spatial"]
    isA = {b: bool(blocks.get(b, {}).get("is_A")) for b in BLOCKS}
    read = {b: b in blocks for b in BLOCKS}
    if all(read[b] for b in sp) and all(isA[b] for b in sp):
        t = ROW["I_both"]
    elif sum(isA[b] for b in sp) == 1:
        x = [b for b in sp if isA[b]][0]
        y = [b for b in sp if b != x][0]
        t = ROW["I_one"].format(x=MODEL_NAME[BLOCK_MODEL[x]]) + f"；另一块 {y}（{MODEL_NAME[BLOCK_MODEL[y]]}）：{st(y)}"
    elif not any(read[b] for b in sp):
        t = "两个 Spatial 块均未读"
    else:
        t = "已读的 Spatial 块均非 A，不写 I 行主张；各块结果：" + "；".join(f"{b} {st(b)}" for b in sp)
    P(f"  [I Spatial] {t}")
    res["spatial"] = t
    for ob in ["oft_object", "ofti_object"]:
        sb = ob.replace("object", "spatial")
        r = blocks.get(ob)
        if r is None:
            t = "未读"
        elif read[sb] and isA[sb] and isA[ob]:
            t = ROW["I_obj_ok"]
        else:
            lab = r.get("labels") or {}
            if r["status"] != "judged":
                t = f"Object 块为 {r['status']}，不套 (i)(ii)"
            elif lab.get("G") is True:
                t = ROW["I_obj_i"]
            elif lab.get("G") is False and lab.get("P1") == L_NOREC:
                t = ROW["I_obj_ii"]
            else:
                t = "不属于 (i)(ii)，只列结果"
            t += f"（{sb}：{st(sb)}；{ob}：{st(ob)}）"
        P(f"  [I {ob}] {t}")
        res[ob] = t
    return res


if __name__ == "__main__":
    main()
