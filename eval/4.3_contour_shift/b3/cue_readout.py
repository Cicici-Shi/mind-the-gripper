"""WidowX 轮廓位置响应协议 v3.1 定稿（protocol_cue_widowx_v3_1_final.md，含 §18 冻结时裁定，sha256 a1ce8fdd…7e2d）闭环十三格的一次性读数。
§17 第 12 条：重写 _v1_drafts/cue_readout.py，照 §5（5.1 到 5.6）、§6（只并排照报）、§8（判定表 27 行、落行顺序、修饰语、块内合成）、
§9（措辞）。只在冻结后对真实数据运行；执行方只在合成数据上跑（test_cue_readout.py）。

用法：python3 cue_readout.py --root out_cue_closed [--manifest 合成清单] [--anchor-post …/ANCHOR_POST.json] [--init-cmp …/INIT_CMP.json]
                             [--investigated-unresolved]
  两块（B1、B2）一起读，只读一次；输出固定为 <root>/READOUT_cue_closed.txt 与 .json，任一已存在就拒读。

护栏（照 b2x/readout.py）：
  * 根目录没有 SYNTHETIC_DATA 标记时只认 b3/FROZEN_cue_widowx，不接受 --manifest；有标记时才可用 --manifest。
  * 清单的 protocol_sha256 必须等于本协议哈希；清单 code_sha256 里登记了 cue_readout.py / cue_gate.py 时核自身哈希。
  * 主键集只按清单 keysets.<块> 取：文件 sha256、键数（= n）、试次只能是 5 到 8、块名、无重复；不扫目录。
  * t_ref 清单按 tref_file / tref_sha256 核哈希，且覆盖主键集每个键。
  * 参数（s̃、10 点线、77 键、0.8、0.2、5%、1 cm、判定点、曲线点、bootstrap、localq 第 0 次查询的增益）只从清单取，缺字段拒读。
  * 开头核三样（§17 第 12 条）：ANCHOR_POST.json、INIT_CMP.json、门槛（cue_gate.run 现算）。未采完拒读。
    §4 的处置：锚点（环境不变检查）不过 → 停（拒读）；协调方查过、查不明时加 --investigated-unresolved，新键全部记录只报数、不判行。
    初始比对某块不过 → 停；加 --investigated-unresolved 时该块只报数、不判行。
  * 门槛硬项不过的键从该块全部格一起剔（cue_gate），读数只用保留的键。
  * 真实数据（没有 SYNTHETIC_DATA 标记）下 --anchor-post、--init-cmp 只许取默认路径，否则拒读。
  * 先后核对（§4「全量采完、读数之前」）：formal_state/STAGE1_OK 必须存在（formal_state 的找法同 cue_gate.state_dir）；
    ANCHOR_POST.json 与 out_cue_anchor_post 下每条 npz、INIT_CMP.json 的修改时间都必须晚于主键记录里最晚的一条；任一不满足就拒读
    （即使加了 --investigated-unresolved：那个开关只管「比对做了、结果不同、查不明」）。
  * 真实数据下 --root 钉死：realpath(--root) 必须等于 realpath(b3/out_cue_closed)；STAGE1_OK 与读数锁固定取 b3/formal_state，
    不由 --root 推出（合成数据才用 cue_gate.state_dir(root)）。
  * STAGE1_OK 核内容（cue_cmp.py 写的 {块: {格: {n_ref, equal, differ, ref_missing, missing, fields}}}）：每块 C0、C3、C4 都要
    differ、ref_missing、missing 为空，equal == n_ref == 10，否则拒读。
  * 清单 code_sha256 登记的全部文件（路径相对 b3）逐个核哈希，缺文件或不符就拒读；真实数据与合成数据同一条规则，
    合成测试的清单只登记 cue_readout.py、cue_gate.py 两个真实哈希。
  * 只读一次：开始读成败与轨迹之前，在 formal_state/ 下用 O_EXCL 建 READOUT_cue_closed.lock，写入根目录的 realpath；
    锁已存在就拒读（中途报错、换副本目录都不能重读）。「输出已存在就拒」保留。

估计量（§5；u 向右为正，输入像素；逐键先算，再跨键汇总）：
  成功 = 最后一步的 success（主），success_ever 并列。Rec0、D_SR(x)、Rec(x)、方向差、Rec_SC、D_Fb、D_Fa、D_F 都是逐键配对差的
  按键加权均值（百分点）。t_c = first_close_step（与 env_actions 第 6 维 < 0 的第一个下标核对）。E、H、B 取逐步表第 t 行，用 t // 4
  那次查询的相机参数投影。O(x) = [E(S+x) − E(S−x)] / 2，k_x = 跨键中位数 / s̃_x（均值并列），手项、物项同法。
  bootstrap：按 episode 簇重抽，B、seed、百分位取自清单；每个量各自用 seed 新建随机数发生器；簇全集 = 该量有有效键的簇；
  簇里的有效键全部放进去再取中位数或均值（按键加权）。「为负」= 上界 < 0，「为正」= 下界 > 0，其余「含 0」（上界恰为 0 不算为负）。

解读（规格没写死、这里选定的做法；交回里逐条列出）：
  J1 一档 x 可判（K 组、P 组的点、D_SR 等引用 S±x 的量）= S(+x)、S(−x) 两格都没有「出视野」「轮廓操纵不足」标记；K、A、P、δ_off
     另要求键数 >= min_keys。不可判档在 R 组的档名记「不判」：它既不是「降低」也不是「未达到」，R4 只看可判档，R2 的「小档不是
     降低」在小档不判时成立（句子写「小档被标记，不判」），R3 要两档都可判。
  J2 Δ_dose 要求 a、b 两档都可判且交集 >= min_keys。
  J3 K 行修饰语：剂量修饰语与「两档都有」只接在 K1、K2、K3（K1、K2 才可写「两档都有」）；固定参考时刻修饰语只接 K1、K2（§8 原文）；
     闭合时刻不对称接在 K0 到 K5 的句子上（K6 不写句子，只照报）；「只描述闭合了的键」接在 K1 到 K5。
  J4 F 行通用修饰语（D_Fa「更低」且 δ_off「为负」）在副本格被标记时不加：被标记的格涉及的量不判。
  J5 F2 的「见底」要求平移大档可判；不可判时不加「没有分辨力」那句。
  J6 F2、F3 时 R 行句子换成「平移轮廓削弱恢复」并附 R 行的数字；F2 再接它规定的限定句；F4 的窄句已在 F4 句子里。
  J7 两块「标签不同」= 两块各组的落行（R、H、F、K、A、P）不完全相同。
  J8 A 组第 4、8 步的手项奇部在全部保留键上算，不要求接触前合格；只照报。
  J9 §5.6 的 G^cmd_0 主量取跨键均值（同 localq 的 D_a、D_b 口径），中位数并列；r_0 = 第 0 行末端点加上 env_actions 前 4 步平移
     增量之和（按 robot_base_pose 的四元数 [qw,qx,qy,qz]（位姿 = [x,y,z,qw,qx,qy,qz]）转到世界）后的投影，减末端点的投影。
     噪声底 = 同 episode 4 个试次同格 r_0 到其簇均值的二维距离的平均，跨簇取中位数。
  J10 κ_1(b) 与「执行比乘这个增益」并排：清单里没有执行比，这里只报 κ_1(b) / G^cmd_0(b)（隐含的执行比），不设线。
  J11 §6 的粗略预测数字不在清单里：照协议 §6 原文抄进 SEC6 常量，只并排、不判，不进任何落行。
  J12 按误差分三等份：该块全部平移格（含 S±h）闭合了的回合合在一起，按 np.quantile 的 1/3、2/3 分位分成 <=、中间、> 三份。
  J13 E、H、B 在同一时刻用同一组相机参数投影（取 t 所在查询的），E = H − B 精确成立。
  J14 P 组合格条件 4「第 0 到 4q−1 步没有闭合指令」= first_close_step 为 −1 或 >= 4q。
  J15 厘米版：每条记录在时刻 t 用 t 所在查询的相机，在任务物第 0 行原点的投影处向右 1 个环境像素，射线与任务物初始高度水平面
     求交得桌面方向 d̂；E、H、B 的世界向量投到 d̂。句子里的「约 X cm」= s̃ × 各键 C4 第 0 次查询处每输入像素厘米数的中位数。
  J16 句子里的「12、24、6 输入像素」= round(s̃)（11.9、24.15、5.95 → 12、24、6）。
  J17 R0 时 H、F、K、A、P 不落行，数字照报；C4 被标「轮廓操纵不足」或处置为只报数时全部组不落行。
  J18 「判语」按 §5.1 三档：提高 / 未达到预设 10 点提高 / 不确定；降低 / 未达到预设 10 点下降 / 不确定；更高 / 更低用同一条线。
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cue_gate as G  # noqa: E402

PROTOCOL_SHA256 = "a1ce8fdd34ec855c129591105db694e61bd5150fee48bdb35f8a5930f4ba7e2d"   # 含 §18；前 86655 字节 = 5af593e2…aa0
FROZEN = os.path.join(HERE, "FROZEN_cue_widowx")
BLOCKS = ("B1", "B2")
MAIN_TRIALS = (5, 6, 7, 8)
CELLS = G.CELL_ORDER
W_ENV, H_ENV, W_IN, H_IN = 640, 480, 224, 224     # §2 投影换算
NEAR_M, PUSH_M, MOVE_M = 0.08, 0.03, 0.03         # 阶段阈值：step0/replay_feat.py 第 43 到 59 行（B1/B2 分支）
EPS = 1e-9
OUT_NAME = "READOUT_cue_closed"
ANCHOR_N = 10                                      # §4：锚点键集每块 10 键
REAL_ROOT = os.path.join(HERE, "out_cue_closed")   # 真实数据只认这个根目录
REAL_STATE = os.path.join(HERE, "formal_state")    # 真实数据的 STAGE1_OK 与读数锁

# ================================================================ 判语与句子模板（§8、§9），全部集中在这里
LV_UP = ("提高", "未达到预设 10 点提高", "不确定")
LV_DOWN = ("降低", "未达到预设 10 点下降", "不确定")
HIGHER, LOWER = "更高", "更低"
NEG, POS, ZERO = "为负", "为正", "含 0"
NOT_JUDGED = "不判"
ROWS = {
    # ---- R 组
    "R0": dict(name="恢复没重现", claim="前半句（回填可恢复），也是丙的前提",
               write="新噪声样本上没有重现画回轮廓的恢复：Rec0 = {Rec0}（{Rec0_lv}）",
               nowrite="任何关于错位的句子",
               stop="该块其余各组都不落行，只报数。这条线在该块停，先回头查第一阶段的可重复性"),
    "R1": dict(name="小档已降低", claim="丙；选一",
               write="把轮廓整体平移 {a_px} 输入像素（约 {a_cm} cm），恢复下降，点估计达到预设的 10 点且方向明确：D_SR(a) = {Da}；"
                     "大档 D_SR(b) = {Db}（{Db_lv}）；Rec(a) = {Reca}（{Reca_lv}）",
               nowrite="「至少低 10 点」；一般的容差；「错位比没有线更糟」（擦除格在地板，判不出）；「因为位置没对上」",
               stop="选一在该块成立。Rec(a) 不是「提高」时这半句只成上界。更小的幅度看 H 行"),
    "R2": dict(name="只在大档降低", claim="丙；选一",
               write="平移 {b_px} 输入像素（约 {b_cm} cm）使恢复下降，点估计达到预设的 10 点且方向明确：D_SR(b) = {Db}；"
                     "小档未判为降低：D_SR(a) = {Da}（{Da_lv}）",
               nowrite="「小档以内没有影响」；「有阈值」；「至少低 10 点」；一般的容差；「错位比没有线更糟」；「因为位置没对上」",
               stop="选一在该块成立。不为小档加样"),
    "R3": dict(name="两档都未达到预设下降", claim="丙；选一（按预设效应大小不支持）",
               write="所测两档错位下，恢复的下降都未达到预设的 10 点：D_SR(a) = {Da}，D_SR(b) = {Db}",
               nowrite="「没有削弱恢复」；「恢复不依赖对齐」；「不存在几何依赖」",
               stop="按预设的效应大小，选一在该块不成立。两块都落 R3：位置分支不再作为选择性证据往下做，不做跨策略验证，论文按诊断论文写"),
    "R4": dict(name="反而升高", claim="丙（反向）",
               write="照报，这是没有预期的方向：D_SR(a) = {Da}，D_SR(b) = {Db}",
               nowrite="任何机制解释", stop="停，带数字回去问 Astra"),
    "R5": dict(name="不确定", claim="丙（判不出）",
               write="判不出：D_SR(a) = {Da}（{Da_lv}），D_SR(b) = {Db}（{Db_lv}）",
               nowrite="把判不出写成没有影响，或写成选一不成立",
               stop="不终止这条线，也不算阴性。后续由协调方与 Astra 二选一：不带成功率预测去做跨策略验证，"
                    "或另立补样协议（同样的格，加新的试次）。本协议内不加样"),
    # ---- H 组
    "H1": dict(name="半档已降低", claim="丙",
               write="平移 {h_px} 输入像素（约 {h_cm} cm）已使恢复下降，点估计达到预设的 10 点且方向明确：D_SR(h) = {Dh}；所测最小的错位也削弱恢复",
               nowrite="同 R1；「再小也会」", stop="这半句只成上界（小于 {h_px} 像素）。不再追更小幅度"),
    "H2": dict(name="半档未达到预设下降", claim="丙",
               write="小档的下降达到预设大小，半档未达到：按预设的 10 点，下降出现在 {h_px} 到 {a_px} 输入像素之间；D_SR(h) = {Dh}",
               nowrite="「6 像素以内没有影响」；把这个区间叫作容差", stop="到此为止"),
    "H3": dict(name="半档不确定", claim="丙（判不出）",
               write="半档判不出：D_SR(h) = {Dh}{h_flag}；这半句仍只成上界（小于 {a_px} 像素）",
               nowrite="把判不出写成没有影响", stop="不加样"),
    # ---- F 组
    "F1": dict(name="留着正确轮廓时恢复更高", claim="丙：R 行的下降能不能只归给多了一条线",
               write="画面里同样有一条偏移 {b_px} 像素的线（与正确轮廓交叠的偏移线），留着正确轮廓时恢复更高：D_Fb = {DFb}",
               nowrite="「策略靠正确的线定位」；「证明了位置估计」（正确的线也可能只是帮策略忽略干扰）；「下降只取决于手被带偏多少」",
               stop="丙在该块可以接上这一句。副本到此为止"),
    "F2": dict(name="未判出留着正确轮廓有帮助", claim="同上（不支持）",
               write="多出的线本身也削弱恢复：D_F = {DF}；没有判出留着正确轮廓时恢复更高：D_Fb = {DFb}",
               nowrite="「正确的线没有用」；移走正确线与加入错误线各占多少",
               stop="丙只许写「平移轮廓削弱恢复」，并加「分不开移走正确线与加入错误线各自的贡献，断开的线本身的干扰不能排除」。副本到此为止"),
    "F3": dict(name="副本比平移小档更糟，手部偏移解释不了", claim="同上（不利于把下降读成位置没对上）",
               write="留着正确轮廓、只多画一条与正确轮廓交叠的偏移线，恢复就比平移小档更低：D_Fa = {DFa}；断开的线本身的干扰不能排除。"
                     "手部偏移：{doff_text}",
               nowrite="「恢复依赖对齐」；「位置没对上所以失败」；「两者的手部偏移相等」",
               stop="R 行的句子只许写「平移轮廓削弱恢复」。跨策略验证若做，必须带副本格"),
    "F4": dict(name="其余", claim="同上（判不出）",
               write="判不出：D_Fb = {DFb}，D_Fa = {DFa}，D_F = {DF}{sc_flag}。丙按缺省的窄句写：分不开移走正确线与加入错误线各自的贡献",
               nowrite="「正确的线没有用」；移走正确线与加入错误线各占多少", stop="副本到此为止，不加样"),
    # ---- K 组（{x} 是实际用的档）
    "K0": dict(name="该档不可判", claim="选二（读不出）",
               write="闭合发生率与闭合时刻本身：平移之后有多少键不再闭合：{closure}",
               nowrite="任何位置句", stop="位置读数在该块停。不换幅度重跑"),
    "K1": dict(name="反向偏移", claim="选二",
               write="错位使闭合处的抓取位置朝相反方向偏移（{x_px} 输入像素档）：k_{x} = {k}；手项 k^H_{x} = {kH}，物项 k^B_{x} = {kB}；"
                     "简化递推模型的粗略预测并排：{sec6}",
               nowrite="只靠视觉定位；视觉伺服；增益是视觉权重；对预测的任何判语；「补偿不完全」",
               stop="位置读数到此为止，不加格、幅度、键"),
    "K2": dict(name="手偏了，相对误差未判出", claim="选二（只到手的位置）",
               write="手的位置朝相反方向偏移（{x_px} 输入像素档）：k^H_{x} = {kH}；手与物的相对误差未判出偏移：k_{x} = {k}；物项 k^B_{x} = {kB}",
               nowrite="「抓取位置被偏置」", stop="同 K1：位置读数到此为止，不加格、幅度、键"),
    "K3": dict(name="相对误差偏了，手项未判出", claim="选二（不支持）",
               write="手与物的相对误差朝相反方向偏移（{x_px} 输入像素档）：k_{x} = {k}；手项未判出方向：k^H_{x} = {kH}；物项 k^B_{x} = {kB}",
               nowrite="「手跟着线走」；「偏移不来自手」（没判出不等于没有）", stop="位置读数在该块停"),
    "K4": dict(name="未判出方向", claim="选二（不支持）",
               write="未判出闭合处位置的方向性偏移（{x_px} 输入像素档）：k_{x} = {k}，k^H_{x} = {kH}，k^B_{x} = {kB}",
               nowrite="「不敏感」；「不用轮廓定位」", stop="位置读数在该块停。跨策略验证不带位置预测"),
    "K5": dict(name="同向", claim="选二（反向）",
               write="出现同向偏移（{x_px} 输入像素档）：k^H_{x} = {kH}，k_{x} = {k}，k^B_{x} = {kB}",
               nowrite="「所有几何解释都被否定」", stop="停，带数字回去问 Astra"),
    "K6": dict(name="符号不一致", claim="选二（判不出）",
               write="（不写句子）照报三项增益：手项 k^H_{x} = {kH}，物项 k^B_{x} = {kB}，相对误差 k_{x} = {k}",
               nowrite="任何方向性的句子", stop="位置读数在该块停"),
    # ---- A 组
    "A0": dict(name="没有阳性对照", claim="选二（方向参照，缺）",
               write="没有阳性对照：{why}。K 自身的数照报", nowrite="把 K 的大小与阳性对照相比",
               stop="K 的大小不作相对量级的解读。不换合成办法补做"),
    "A1": dict(name="检出", claim="选二（读数的方向参照）",
               write="同一读数对任务物图像平移（预期方向明确的操纵）读出了朝平移方向的偏移：k^H_obj = {kHo}",
               nowrite="两者之比是线的权重；两者等不等幅的机制含义；「答案已知的操纵」", stop="阳性对照到此为止"),
    "A2": dict(name="未检出", claim="选二（方向参照未检出）",
               write="未检出预期的目标跟随响应：k^H_obj = {kHo}；原因可能是策略反应、视觉冲突或读数灵敏度，本协议分不开。K 自身的数照报",
               nowrite="「读数失效」；把 K 的大小与阳性对照相比；把 K4 的判不出解读成线的作用弱",
               stop="K 的大小不作相对量级的解读"),
    "A3": dict(name="反向", claim="选二（反向）",
               write="照报，这是没有预期的方向：k^H_obj = {kHo}", nowrite="任何机制解释", stop="停，带数字回去问 Astra"),
    # ---- P 组
    "P1": dict(name="接触前已偏", claim="选二里的证一",
               write="在回合起点，同一状态下只改画面，第一段执行已朝相反方向偏移：κ_1({x}) = {k1}；到第 8 步，接触和闭合之前，偏移仍在：κ_2({x}) = {k2}",
               nowrite="「接近途中的同状态执行响应」（同状态的只有回合起点）；「终点偏移全部形成于接触前」；独立验证",
               stop="不再做任何同状态策略查询与回放"),
    "P2": dict(name="只有第一段", claim="证一（只到第一段）",
               write="回合起点的第一段执行有反向响应：κ_1({x}) = {k1}；之后未见累积：κ_2({x}) = {k2}",
               nowrite="「接触前已形成终点偏移」", stop="同 P1：不再做任何同状态策略查询与回放"),
    "P3": dict(name="接触前未见", claim="证一（不支持）",
               write="未见接触前的执行层偏移：κ_1({x}) = {k1}",
               nowrite="「执行响应」", stop="是否补做状态内查询由协调方与 Astra 另议，本协议不自动做"),
}
ROW_COUNT = 27
assert len(ROWS) == ROW_COUNT
R_ORDER = ("R0", "R4", "R1", "R2", "R3", "R5")   # §8 落行顺序
MOD = {
    "R1_B_NOT_DOWN": "大档未判为「降低」（{Db_lv}），未见随幅度加重",
    "R1_UPPER": "Rec(a) 不是「提高」（{Reca_lv}）：这半句只成上界",
    "R2_A_FLAG": "小档被标记（{why}），不判",
    "R_DIR": "方向差 SR(S(+{x})) − SR(S(−{x})) = {v}，区间不含 0",
    "R_TIER_FLAG": "{x} 档被标记（{why}），涉及它的量不判",
    "F1_DSRB": "大档平移的下降不能只归给多了一条线",
    "F1_DF_DOWN": "留着正确轮廓，多画一条线后恢复也下降：D_F = {DF}",
    "F1_DF_NOT": "留着正确轮廓时的下降未达到预设的 10 点：D_F = {DF}",
    "F2_FLOOR": "副本格与平移大档都见底（Rec_SC = {RecSC}，Rec(b) = {Recb}）：这个比较没有分辨力",
    "F3_DOFF_NOTNEG": "没有判出副本格里手被带得更远（δ_off = {doff}，{doff_lv}）",
    "F3_DOFF_NA": "四格都闭合的键不足（{n} < {min_keys}），δ_off 算不出",
    "F3_ALSO_F1": "同时：画面里同样有一条偏移 {b_px} 像素的线，留着正确轮廓时恢复更高：D_Fb = {DFb}；两项发现都保留",
    "F_GENERIC": "副本比平移小档更低，同时手被带得更远：D_Fa = {DFa}，δ_off = {doff}",
    "F_REPORT_ONLY": "R 行是 {R}：副本格只照报三个配对差与 δ_off，不接任何句子",
    "F_SC_FLAG": "副本格被标记（{why}）",
    "K_SUB": "b 档不可判（{why}），各行条件里的 b 换成 a，句子写明幅度",
    "K0_WHY": "{x} 档不可判：{why}",
    "K_DOSE_NEG": "偏移随幅度增大（Δ_dose = {dd}）",
    "K_DOSE_ZERO": "未确认剂量关系（Δ_dose = {dd}）",
    "K_DOSE_POS": "大档的偏移反而小（Δ_dose = {dd}）",
    "K_DOSE_NA": "Δ_dose 不判（{why}）",
    "K_A_TIER": "a 档并列：k_a = {ka}，k^H_a = {kHa}",
    "K_A_BOTH": "两档都有",
    "K_REF": "句子限定为「闭合时刻的位置」；固定参考时刻未见闭合处那样的偏移，终点偏移可能经由闭合时刻形成（k^{{H,ref}}_{x} = {kHref}）",
    "K_TC_ASYM": "闭合时刻不对称：t_c(+{x}) − t_c(−{x}) 中位数 {m:+.1f} 步，区间 [{lo:+.1f}, {hi:+.1f}]",
    "K_CLOSED_ONLY": "位置句只描述闭合发生了的键（成对 {n_pair} 键）；未闭合的键：S(+{x}) {n_up}、S(−{x}) {n_um}；"
                     "全键参考时刻读数并排：k^ref_{x} = {kref}，k^{{H,ref}}_{x} = {kHref}",
    "K2_OBJ": "物体在闭合前朝同一方向被移动（k^B_{x} = {kB}）",
    "K3_OBJ": "这部分偏移来自物体被移动（k^B_{x} = {kB}）",
    "K4_A1": "同一读数读得出任务物平移",
    "K4_NOT_A1": "阳性对照没有做成或未检出，这个读数的灵敏度没有校准",
    "A1_K1": "两种平移引起方向相反的偏移",
    "A0_N": "成对样本 {n} 键 < {min_keys}",
    "A0_FLAG": "任务物平移格被标「合成不足」",
    "P_SUB": "b 档有点不可判（{why}），a 档两点都可判，整组换成 a 档",
    "P_K2_NA": "κ_2 合格键不足或被标记（{why}），视同 κ_2 不为负",
    "P_NONE": "替补之后 κ_1 仍不可判（{why}）：P 组三行都不判",
    "P_KK": "终点偏移可能经由闭合时刻或推动物体形成",
}
NUMBERS_ONLY = {
    "anchor": "锚点比对 / 环境不变检查不过，查不明：新键上的全部记录只报数、不判行",
    "init": "新键对第一阶段的初始比对不过，查不明：该块只报数、不判行",
    "c4": "C4 被标「轮廓操纵不足」：该块不判",
    "r0": "R 行是 R0：该块其余各组都不落行，只报数",
}
R_RESTRICTED = "平移轮廓削弱恢复（{nums}）"
F2_CAVEAT = "分不开移走正确线与加入错误线各自的贡献，断开的线本身的干扰不能排除"
SELECTIVE_BOTH = "选择性证据的前两个分句在该块成立（R1 或 R2，并且 K1，并且 P1）：跨策略验证同时带成功率与位置两条预测"
SELECTIVE_SR = "只有 R1 或 R2 成立：同任务上的跨策略验证只带成功率预测"
SELECTIVE_NONE = "选择性证据的前两个分句在该块不成立"
SELECTIVE_F3 = "F3 成立：同任务上的跨策略验证必须带副本格"
BLOCKS_LEAD = "在一个第三方 π0 复现的 Bridge 权重、两个任务上，"
BLOCKS_DIFF = "标签不同；本实验分不开任务几何、相对剂量（两块的夹爪宽不同）、图像轴相对接近方向的夹角与分辨力"
GLOBAL_STOP = "读数之后，本协议不加幅度、不加格、不加键、不补查询。之后可能做的只有一件，要另立协议：同任务上的跨策略验证"
SEC6_NAME = "简化递推模型的粗略预测"
SEC6 = {  # 协议 §6 原文（粗算，只并排，不判，不进任何落行；解读 J11）
    "B1": {"k_H_b_closure": "约 -0.8 到 -1.0", "kappa_1_b": "-0.49 到 -0.64", "kappa_2_b": "-0.74 到 -0.85"},
    "B2": {"k_H_b_closure": "约 -0.6 到 -0.9", "kappa_1_b": "-0.30 到 -0.50", "kappa_2_b": "-0.58 到 -0.75"},
}
SEC6_SUB_A = "并排的是 §6 的 b 档粗略预测；§6 没有 a 档的预测"
EXEC_RATIO_GAIN = {  # §18.1 第 4 条：执行比 0.85 到 1 乘 localq 第 0 次查询的 b 档增益（-0.607、-0.512），只与 κ_1(b) 并排
    "B1": "-0.52 到 -0.61", "B2": "-0.43 到 -0.51"}
A_CONTAINER_NA = {  # §18.1 第 1 条：茄子块「贴图落在目标容器上」无信息，A 组句子旁必须带上原因
    "B1": "贴图落在目标容器上的帧占比：无信息（环境里没有篮子的 actor，篮子是叠加的照片；能用的 sink 是整个水槽，贴图恒落在它上面）"}
SEC6_NOTE = "只并排，不写「相容」「弱于」「强于」之类的判语；参考时刻的 k^{H,ref}_b 不与它并排"
LOCALQ_NOTE = "localq：探索性，指令层，提出假设的那批回合"
FORBIDDEN_IN_WRITE = ("只靠视觉定位", "视觉伺服", "增益是视觉权重", "画面与本体各占", "至少低 10 点", "没有削弱恢复",
                      "近似线性", "补偿不完全", "不敏感", "没有实际影响", "非单调", "饱和", "有阈值", "相容", "不冲突",
                      "弱于", "强于", "答案已知", "读数失效", "恢复需要正确位置上有线", "因为", "所以", "独立验证", "确认性复现")


class Refuse(Exception):
    pass


def lt(x, t):
    return x < t - EPS


def gt(x, t):
    return x > t + EPS


def ge(x, t):
    return x >= t - EPS


def le(x, t):
    return x <= t + EPS


# ---------------------------------------------------------------- 判语
def sign_of(e):
    """「为负」= 上界 < 0；「为正」= 下界 > 0；其余「含 0」（§5 开头）。e 为 None = 不可判。"""
    if e is None or e.get("point") is None:
        return None
    if lt(e["hi"], 0.0):
        return NEG
    if gt(e["lo"], 0.0):
        return POS
    return ZERO


def lv_up(e, thr):
    if e is None or e.get("point") is None:
        return NOT_JUDGED
    if ge(e["point"], thr) and gt(e["lo"], 0.0):
        return LV_UP[0]
    if lt(e["hi"], thr):
        return LV_UP[1]
    return LV_UP[2]


def lv_down(e, thr):
    if e is None or e.get("point") is None:
        return NOT_JUDGED
    if le(e["point"], -thr) and lt(e["hi"], 0.0):
        return LV_DOWN[0]
    if gt(e["lo"], -thr):
        return LV_DOWN[1]
    return LV_DOWN[2]


def is_higher(e, thr):
    return e is not None and e.get("point") is not None and ge(e["point"], thr) and gt(e["lo"], 0.0)


def is_lower(e, thr):
    return e is not None and e.get("point") is not None and le(e["point"], -thr) and lt(e["hi"], 0.0)


# ---------------------------------------------------------------- bootstrap（按 episode 聚类，按键加权）
class Boot:
    def __init__(self, B, seed, ci):
        self.B, self.seed, self.ci = int(B), int(seed), (float(ci[0]), float(ci[1]))

    def raw(self, vals, clus):
        """返回 (median 点, median 下界, 上界, mean 点, mean 下界, 上界, 键数, 簇数)；簇全集 = 有有效键的簇。"""
        v = np.asarray(vals, np.float64)
        c = np.asarray(clus)
        ok = np.isfinite(v)
        v, c = v[ok], c[ok]
        n = len(v)
        if n == 0:
            return None
        labs, inv = np.unique(c, return_inverse=True)
        nC = len(labs)
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, nC, size=(self.B, nC))
        W = np.zeros((self.B, nC))
        np.add.at(W, (np.repeat(np.arange(self.B), nC), idx.ravel()), 1.0)
        order = np.argsort(v, kind="stable")
        vs = v[order]
        Wk = W[:, inv[order]]
        cum = np.cumsum(Wk, axis=1)
        N = cum[:, -1]
        p1 = np.floor((N - 1) / 2.0)
        p2 = np.floor(N / 2.0)
        j1 = np.argmax(cum > p1[:, None], axis=1)
        j2 = np.argmax(cum > p2[:, None], axis=1)
        med = (vs[j1] + vs[j2]) / 2.0
        mean = (Wk @ vs) / N
        lo, hi = np.percentile(med, self.ci)
        mlo, mhi = np.percentile(mean, self.ci)
        return (float(np.median(v)), float(lo), float(hi), float(np.mean(v)), float(mlo), float(mhi), int(n), int(nC))

    def med(self, vals, clus, scale=1.0):
        """中位数为主、均值并列；scale 乘在全部数上（k = 中位数 / s̃ 时 scale = 1 / s̃）。"""
        r = self.raw(vals, clus)
        if r is None:
            return {"point": None, "lo": None, "hi": None, "mean": None, "mean_lo": None, "mean_hi": None, "n": 0, "n_clusters": 0}
        s = float(scale)
        return {"point": r[0] * s, "lo": r[1] * s, "hi": r[2] * s, "mean": r[3] * s, "mean_lo": r[4] * s, "mean_hi": r[5] * s,
                "n": r[6], "n_clusters": r[7]}

    def mean(self, vals, clus, scale=1.0):
        """按键加权均值（成功率差用，scale = 100 换成百分点）。"""
        r = self.raw(vals, clus)
        if r is None:
            return {"point": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
        s = float(scale)
        return {"point": r[3] * s, "lo": r[4] * s, "hi": r[5] * s, "median": r[0] * s, "n": r[6], "n_clusters": r[7]}


def f_sr(e):
    if e is None or e.get("point") is None:
        return "—"
    return f"{e['point']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}]"


def f_k(e, nd=3):
    if e is None or e.get("point") is None:
        return "—（n=0）" if e is not None else "—"
    s = f"{e['point']:+.{nd}f} [{e['lo']:+.{nd}f}, {e['hi']:+.{nd}f}]"
    if e.get("mean") is not None:
        s += f"（均值 {e['mean']:+.{nd}f}；n={e['n']}）"
    return s


# ---------------------------------------------------------------- 几何
def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def proj_env(P, cam):
    K, Rm, t = cam
    uvw = K @ (Rm @ np.asarray(P, np.float64) + t)
    return uvw[:2] / uvw[2]


def proj_in(P, cam):
    uv = proj_env(P, cam)
    return np.array([uv[0] * W_IN / W_ENV, uv[1] * H_IN / H_ENV])


def ray_plane(uv, cam, z0):
    K, Rm, t = cam
    C = -Rm.T @ t
    d = Rm.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    return C + (z0 - C[2]) / d[2] * d


def table_axis(cam, obj0):
    """图像向右 1 个环境像素在任务物初始高度水平面上对应的桌面单位方向 d̂，与每输入像素厘米数（解读 J15）。"""
    uv0 = proj_env(obj0, cam)
    v = ray_plane(uv0 + np.array([1.0, 0.0]), cam, obj0[2]) - ray_plane(uv0, cam, obj0[2])
    L = float(np.linalg.norm(v))
    return v / L, L * 100.0 * W_ENV / W_IN


def stage_of(tcp, obj, contact, grasp, disp, success):
    """step0/replay_feat.py 第 43 到 59 行的 B1/B2 分支。"""
    d = np.linalg.norm(tcp - obj, axis=1)
    st = 1 if d.min() <= NEAR_M else 0
    if contact.any() or disp.max() >= PUSH_M:
        st = max(st, 2)
    if grasp.any():
        st = 3
        if disp[int(np.argmax(grasp)):].max() >= MOVE_M:
            st = 4
    if success:
        st = 5
    return st


def derive(path, qe, tref, qs, disp_thr):
    """一条闭环记录的逐键量。qs：要取的查询序号（P 组与曲线）。"""
    with np.load(path, allow_pickle=False) as z:
        if not bool(z["complete"]):
            raise Refuse(f"{path} 不完整")
        ea = np.asarray(z["env_actions"], np.float64)
        cl = np.nonzero(ea[:, 6] < 0)[0]
        tc = int(cl[0]) if len(cl) else -1
        if tc != int(z["first_close_step"]):
            raise Refuse(f"{path}：first_close_step {int(z['first_close_step'])} 与 env_actions 推出的 {tc} 不符")
        idx = int(z["task_obj_index"])
        names = json.loads(str(z["st_actor_names"]))
        if names[idx] != str(z["task_obj_name"]):
            raise Refuse(f"{path}：task_obj_index 与 task_obj_name 不符")
        tcp = np.asarray(z["st_tcp_pose"], np.float64)[:, :3]
        obj = np.asarray(z["st_actor_poses"], np.float64)[:, idx, :3]
        contact = np.asarray(z["st_contact"], bool).any(axis=1)
        grasp = np.asarray(z["st_grasp"], bool)
        disp = np.asarray(z["st_obj_disp_xy"], np.float64)
        Ks, Es = np.asarray(z["q_intrinsic"], np.float64), np.asarray(z["q_extrinsic"], np.float64)
        nq, T1 = len(Ks), len(tcp)
        succ, ever = bool(z["success"]), bool(z["success_ever"])
        base = np.asarray(z["robot_base_pose"], np.float64)

    def cam(t):
        q = min(t // qe, nq - 1)
        return Ks[q], Es[q][:3, :3], Es[q][:3, 3]

    def at(t):
        c = cam(t)
        ut, uo, u0 = proj_in(tcp[t], c), proj_in(obj[t], c), proj_in(obj[0], c)
        dh, cpp = table_axis(c, obj[0])
        return {"E": float(ut[0] - uo[0]), "H": float(ut[0] - u0[0]), "B": float(uo[0] - u0[0]),
                "Ev": float(ut[1] - uo[1]), "Hv": float(ut[1] - u0[1]),
                "E_cm": float((tcp[t] - obj[t]) @ dh * 100), "H_cm": float((tcp[t] - obj[0]) @ dh * 100),
                "B_cm": float((obj[t] - obj[0]) @ dh * 100), "cm_per_px": cpp}

    out = {"tc": tc, "closed": tc >= 0, "success": succ, "success_ever": ever, "contact_any": bool(contact.any()),
           "stage": stage_of(tcp, obj, contact, grasp, disp, succ)}
    out["at_tc"] = at(tc) if tc >= 0 else None
    if not (0 <= int(tref) < T1):
        raise Refuse(f"{path}：t_ref {tref} 越界（逐步 {T1} 行）")
    out["at_ref"] = at(int(tref))
    rows = {}
    for q in qs:
        t = qe * q
        if t >= T1:
            rows[q] = None
            continue
        qual = (not contact[:t + 1].any()) and (not grasp[:t + 1].any()) and bool(disp[t] < disp_thr) and (tc < 0 or tc >= t)
        rows[q] = {"H": at(t)["H"], "qual": bool(qual)}
    out["rows"] = rows
    Rb = quat2mat(base[3:7])
    dt = ea[:qe, :3].sum(axis=0)
    c0 = cam(0)
    out["r0"] = (proj_in(tcp[0] + Rb @ dt, c0) - proj_in(tcp[0], c0)).tolist()
    out["cm_per_px_q0"] = table_axis(c0, obj[0])[1]
    return out


# ================================================================ 估计
def compute_block(D, kept, clus, P, gate):
    """D[格][键] = derive 的结果。返回该块全部估计量（§5、§6 并排量）。"""
    bt = Boot(P["B"], P["seed"], P["ci"])
    st = P["s_tilde"]
    mk = P["min_keys"]
    cl = lambda ks: [clus[k] for k in ks]  # noqa: E731
    s = {c: np.array([float(D[c][k]["success"]) for k in kept]) for c in CELLS}
    se = {c: np.array([float(D[c][k]["success_ever"]) for k in kept]) for c in CELLS}
    C = cl(kept)
    est = {"n_kept": len(kept), "n_clusters": len(set(C)), "SR": {c: 100 * float(s[c].mean()) for c in CELLS},
           "SR_ever": {c: 100 * float(se[c].mean()) for c in CELLS}}

    def sr_block(S):
        o = {"Rec0": bt.mean(S["C4"] - S["C3"], C, 100)}
        for x in ("a", "b", "h"):
            avg = (S[f"S+{x}"] + S[f"S-{x}"]) / 2
            o[f"D_SR_{x}"] = bt.mean(avg - S["C4"], C, 100)
            o[f"Rec_{x}"] = bt.mean(avg - S["C3"], C, 100)
            o[f"Dir_{x}"] = bt.mean(S[f"S+{x}"] - S[f"S-{x}"], C, 100)
        sc = (S["SC+b"] + S["SC-b"]) / 2
        o["Rec_SC"] = bt.mean(sc - S["C3"], C, 100)
        o["D_Fb"] = bt.mean(sc - (S["S+b"] + S["S-b"]) / 2, C, 100)
        o["D_Fa"] = bt.mean(sc - (S["S+a"] + S["S-a"]) / 2, C, 100)
        o["D_F"] = bt.mean(sc - S["C4"], C, 100)
        return o

    est["sr"] = sr_block(s)
    est["sr_ever"] = sr_block(se)
    # ---- §5.1 描述量：阶段分布；按误差分三等份
    est["stages"] = {c: [int(sum(1 for k in kept if D[c][k]["stage"] == i)) for i in range(6)] for c in CELLS}
    pool = [(abs(D[c][k]["at_tc"]["E"]), D[c][k]["success"],
             abs(D[c][k]["at_tc"]["E"] - D["C4"][k]["at_tc"]["E"]) if D["C4"][k]["closed"] else None)
            for c in G.SHIFT_CELLS for k in kept if D[c][k]["closed"]]

    def terciles(vals):
        vals = [(e, sc) for e, sc in vals if e is not None]
        if len(vals) < 3:
            return None
        e = np.array([v[0] for v in vals])
        sc = np.array([v[1] for v in vals], float)
        q1, q2 = np.quantile(e, [1 / 3, 2 / 3])
        bins = [e <= q1, (e > q1) & (e <= q2), e > q2]
        return {"cuts_px": [float(q1), float(q2)],
                "bins": [{"n": int(b.sum()), "SR": 100 * float(sc[b].mean()) if b.any() else None} for b in bins]}
    est["err_terciles"] = {"abs_E": terciles([(p[0], p[1]) for p in pool]),
                           "abs_E_minus_C4": terciles([(p[2], p[1]) for p in pool])}
    # ---- §5.3 K 组
    tref_pos = {}
    K = {}
    for x in ("a", "b", "h"):
        cp, cm = f"S+{x}", f"S-{x}"
        Px = [k for k in kept if D[cp][k]["closed"] and D[cm][k]["closed"]]
        o = lambda f, ks, t="at_tc": np.array([(D[cp][k][t][f] - D[cm][k][t][f]) / 2 for k in ks])  # noqa: E731
        kx = {"pairs": Px, "n_pairs": len(Px),
              "k": bt.med(o("E", Px), cl(Px), 1 / st[x]), "kH": bt.med(o("H", Px), cl(Px), 1 / st[x]),
              "kB": bt.med(o("B", Px), cl(Px), 1 / st[x]), "O_Ev_px": bt.med(o("Ev", Px), cl(Px)),
              "O_E_cm": bt.med(o("E_cm", Px), cl(Px)), "O_H_cm": bt.med(o("H_cm", Px), cl(Px)),
              "O_B_cm": bt.med(o("B_cm", Px), cl(Px)),
              "tc_diff": bt.med([D[cp][k]["tc"] - D[cm][k]["tc"] for k in Px], cl(Px)),
              "kref": bt.med(o("E", kept, "at_ref"), C, 1 / st[x]), "kHref": bt.med(o("H", kept, "at_ref"), C, 1 / st[x])}
        flags = [f for c in (cp, cm) for f in gate["cell_flags"].get(c, []) if f in (G.OUTLINE_WEAK, G.OUT_OF_VIEW)]
        why = []
        if flags:
            why.append("被标「" + "」「".join(sorted(set(flags))) + "」")
        kx["flagged"] = bool(flags)
        if len(Px) < mk:
            why.append(f"成对样本 {len(Px)} 键 < {mk}")
        kx["judgeable"] = not why
        kx["why"] = "；".join(why)
        K[x] = kx
    inter = [k for k in K["a"]["pairs"] if k in set(K["b"]["pairs"])]
    dd_ok = K["a"]["judgeable"] and K["b"]["judgeable"] and len(inter) >= mk
    oE = lambda x, k, f: (D[f"S+{x}"][k]["at_tc"][f] - D[f"S-{x}"][k]["at_tc"][f]) / 2  # noqa: E731
    est["dose"] = {"n": len(inter), "judgeable": dd_ok,
                   "why": "" if dd_ok else (f"交集 {len(inter)} 键 < {mk}" if len(inter) < mk else "a 或 b 档不可判"),
                   "d_E_px": bt.med([oE("b", k, "E") - oE("a", k, "E") for k in inter], cl(inter)),
                   "d_H_px": bt.med([oE("b", k, "H") - oE("a", k, "H") for k in inter], cl(inter))}
    for x in K:
        K[x].pop("pairs")
    est["K"] = K
    est["closure"] = {}
    for c in CELLS:
        tcs = np.array([D[c][k]["tc"] for k in kept if D[c][k]["closed"]], float)
        unc = [k for k in kept if not D[c][k]["closed"]]
        est["closure"][c] = {"n_closed": int(len(tcs)), "n_unclosed": len(unc),
                             "tc_median": float(np.median(tcs)) if len(tcs) else None,
                             "tc_q25_q75": [float(np.percentile(tcs, 25)), float(np.percentile(tcs, 75))] if len(tcs) else None,
                             "unclosed_at_ref_E_median": float(np.median([D[c][k]["at_ref"]["E"] for k in unc])) if unc else None,
                             "unclosed_at_ref_H_median": float(np.median([D[c][k]["at_ref"]["H"] for k in unc])) if unc else None,
                             "unclosed_stages": [int(sum(1 for k in unc if D[c][k]["stage"] == i)) for i in range(6)]}
    est["curve5"] = {}
    for c in ("S-b", "S-a", "S+a", "S+b"):
        ks = [k for k in kept if D[c][k]["closed"] and D["C4"][k]["closed"]]
        v = [D[c][k]["at_tc"]["E"] - D["C4"][k]["at_tc"]["E"] for k in ks]
        est["curve5"][c] = {"median_px": float(np.median(v)) if v else None, "n": len(ks)}
    est["curve5"]["C4"] = {"median_px": 0.0, "n": sum(1 for k in kept if D["C4"][k]["closed"])}
    est["cm_per_px_median"] = float(np.median([D["C4"][k]["cm_per_px_q0"] for k in kept]))
    # ---- §5.2 F 组的手部偏移
    four = [k for k in kept if all(D[c][k]["closed"] for c in ("SC+b", "SC-b", "S+a", "S-a"))]
    oH = lambda cp, cm, k, t="at_tc": (D[cp][k][t]["H"] - D[cm][k][t]["H"]) / 2  # noqa: E731
    doff = bt.med([oH("SC+b", "SC-b", k) - oH("S+a", "S-a", k) for k in four], cl(four))
    est["F"] = {"delta_off_n": len(four), "delta_off": doff if len(four) >= mk else None, "delta_off_raw": doff}
    sc_pairs = [k for k in kept if D["SC+b"][k]["closed"] and D["SC-b"][k]["closed"]]
    est["F"]["kH_SC_closure"] = bt.med([oH("SC+b", "SC-b", k) for k in sc_pairs], cl(sc_pairs), 1 / st["b"])
    # ---- §5.5 P 组
    Pq = {}
    allq = sorted(set(P["curve_q"]) | set(P["kappa_q"]) | {1, 2})

    def kappa(cp, cm, x, q, ks=None):
        ks = kept if ks is None else ks
        good = [k for k in ks if D[cp][k]["rows"].get(q) and D[cm][k]["rows"].get(q)
                and D[cp][k]["rows"][q]["qual"] and D[cm][k]["rows"][q]["qual"]]
        e = bt.med([(D[cp][k]["rows"][q]["H"] - D[cm][k]["rows"][q]["H"]) / 2 for k in good], cl(good), 1 / st[x])
        return e, good
    for x in ("a", "b"):
        for q in allq:
            e, good = kappa(f"S+{x}", f"S-{x}", x, q)
            why = []
            if K[x]["flagged"]:
                why.append("S±%s 被标记" % x)
            if len(good) < mk:
                why.append(f"合格键 {len(good)} < {mk}")
            Pq[f"{x}{q}"] = {"q": q, "x": x, "est": e, "n_qual": len(good), "judgeable": not why, "why": "；".join(why)}
    est["P"] = Pq
    rf = [k for k in gate["items"]["12_ring_free_keys_Sb_q01"] if k in set(kept)]
    est["P_ring_free_kappa2_b"] = {"n_keys": len(rf), "est": kappa("S+b", "S-b", "b", 2, rf)[0]}
    e_sc2, g_sc2 = kappa("SC+b", "SC-b", "b", 2)
    _, g_a2 = kappa("S+a", "S-a", "a", 2)
    both = [k for k in g_sc2 if k in set(g_a2)]
    est["F"]["step8_kH_SC"] = e_sc2
    est["F"]["step8_kH_Sa"] = Pq.get("a2", {}).get("est")
    est["F"]["step8_diff_px"] = bt.med([(D["SC+b"][k]["rows"][2]["H"] - D["SC-b"][k]["rows"][2]["H"]) / 2
                                        - (D["S+a"][k]["rows"][2]["H"] - D["S-a"][k]["rows"][2]["H"]) / 2 for k in both], cl(both))
    kHb = K["b"]["kH"]["point"]
    k2b = Pq.get("b%d" % P["kappa_q"][1], {}).get("est", {}).get("point")
    est["P_ratio_kappa2b_over_kHb"] = (k2b / kHb) if (kHb not in (None, 0) and k2b is not None) else None
    # ---- §5.4 A 组
    obp = [k for k in kept if D["OB+b"][k]["closed"] and D["OB-b"][k]["closed"]]
    A = {"n_pairs": len(obp),
         "kH_obj": bt.med([oH("OB+b", "OB-b", k) for k in obp], cl(obp), 1 / st["b"]),
         "k_obj_E": bt.med([(D["OB+b"][k]["at_tc"]["E"] - D["OB-b"][k]["at_tc"]["E"]) / 2 for k in obp], cl(obp), 1 / st["b"]),
         "closure_rate": {c: est["closure"][c]["n_closed"] / len(kept) for c in G.OB_CELLS},
         "SR": {c: est["SR"][c] for c in G.OB_CELLS},
         "contact_rate": {c: float(np.mean([D[c][k]["contact_any"] for k in kept])) for c in G.OB_CELLS}}
    for q in (1, 2):
        ks = [k for k in kept if D["OB+b"][k]["rows"].get(q) and D["OB-b"][k]["rows"].get(q)]
        A[f"kH_row{4 * q}"] = bt.med([(D["OB+b"][k]["rows"][q]["H"] - D["OB-b"][k]["rows"][q]["H"]) / 2 for k in ks], cl(ks), 1 / st["b"])
    A["flagged"] = any(G.SYNTH_WEAK in gate["cell_flags"].get(c, []) for c in G.OB_CELLS)
    est["A"] = A
    # ---- §5.6 第 0 次查询
    r0 = {c: np.array([D[c][k]["r0"] for k in kept]) for c in CELLS}
    G0 = {}
    for x in ("a", "b", "h"):
        v = (r0[f"S+{x}"][:, 0] - r0[f"S-{x}"][:, 0]) / 2 / st[x]
        G0[x] = {"mean": bt.mean(v, C), "median": bt.med(v, C)}
    nrm = lambda a: float(np.median(np.linalg.norm(a, axis=1)))  # noqa: E731
    floor = {}
    for c in CELLS:
        ds = []
        for lab in sorted(set(C)):
            ii = [i for i, cc in enumerate(C) if cc == lab]
            if len(ii) >= 2:
                m = r0[c][ii].mean(axis=0)
                ds.append(float(np.mean(np.linalg.norm(r0[c][ii] - m, axis=1))))
        floor[c] = float(np.median(ds)) if ds else None
    k1b = Pq.get("b%d" % P["kappa_q"][0], {}).get("est", {}).get("point")
    gb = G0["b"]["mean"]["point"]
    est["q0"] = {"G_cmd_0": G0, "localq_q0": P["localq_q0"], "localq_note": LOCALQ_NOTE,
                 "kappa1_b_over_Gcmd0_b": (k1b / gb) if (k1b is not None and gb not in (None, 0)) else None,
                 "norm_C3_minus_C4": nrm(r0["C3"] - r0["C4"]), "norm_C0_minus_C4": nrm(r0["C0"] - r0["C4"]),
                 "norm_C4": nrm(r0["C4"]), "noise_floor_by_cell": floor}
    return est


# ================================================================ 判定（§8）
def _tier_flag_why(est, x):
    return est["K"][x]["why"] if est["K"][x]["flagged"] else ""


def decide_block(block, est, flags, P, numbers_only_reasons):
    thr = P["thr"]
    mk = P["min_keys"]
    st = P["s_tilde"]
    sr = est["sr"]
    K = est["K"]
    px = {x: int(round(st[x])) for x in st}
    cm = {x: f"{st[x] * est['cm_per_px_median']:.1f}" for x in st}
    dec = {"block": block, "rows": {g: None for g in "RHFKAP"}, "sent": {}, "notes": [], "numbers_only": False}
    flagged = {x: K[x]["flagged"] for x in ("a", "b", "h")}
    lv = {"Rec0": lv_up(sr["Rec0"], thr)}
    for x in ("a", "b", "h"):
        lv[f"D_SR_{x}"] = NOT_JUDGED if flagged[x] else lv_down(sr[f"D_SR_{x}"], thr)
        lv[f"Rec_{x}"] = NOT_JUDGED if flagged[x] else lv_up(sr[f"Rec_{x}"], thr)
    sc_flags = sorted({f for c in G.COPY_CELLS for f in flags.get(c, []) if f in (G.OUTLINE_WEAK, G.OUT_OF_VIEW)})
    lv["D_F"] = NOT_JUDGED if sc_flags else lv_down(sr["D_F"], thr)
    lv["Rec_SC"] = NOT_JUDGED if sc_flags else lv_up(sr["Rec_SC"], thr)
    dec["levels"] = lv
    if numbers_only_reasons:
        dec.update(numbers_only=True, notes=list(numbers_only_reasons))
        return dec
    V = dict(Rec0=f_sr(sr["Rec0"]), Rec0_lv=lv["Rec0"], Da=f_sr(sr["D_SR_a"]), Db=f_sr(sr["D_SR_b"]), Dh=f_sr(sr["D_SR_h"]),
             Da_lv=lv["D_SR_a"], Db_lv=lv["D_SR_b"], Reca=f_sr(sr["Rec_a"]), Reca_lv=lv["Rec_a"], Recb=f_sr(sr["Rec_b"]),
             RecSC=f_sr(sr["Rec_SC"]), DFb=f_sr(sr["D_Fb"]), DFa=f_sr(sr["D_Fa"]), DF=f_sr(sr["D_F"]),
             a_px=px["a"], b_px=px["b"], h_px=px["h"], a_cm=cm["a"], b_cm=cm["b"], h_cm=cm["h"], min_keys=mk)

    def put(g, row, mods=(), **kw):
        r = ROWS[row]
        vv = dict(V, **kw)
        dec["rows"][g] = row
        dec["sent"][g] = {"row": row, "name": r["name"], "claim": r["claim"], "write": r["write"].format(**vv),
                          "modifiers": [m for m in mods if m], "nowrite": r["nowrite"], "stop": r["stop"].format(**vv)}
    # ---- R 组：R0、R4、R1、R2、R3、R5
    if lv["Rec0"] != LV_UP[0]:
        put("R", "R0")
        dec["notes"].append(NUMBERS_ONLY["r0"])
        return dec
    r4 = any((not flagged[x]) and is_higher(sr[f"D_SR_{x}"], thr) for x in ("a", "b"))
    if r4:
        R = "R4"
    elif lv["D_SR_a"] == LV_DOWN[0]:
        R = "R1"
    elif lv["D_SR_b"] == LV_DOWN[0]:
        R = "R2"
    elif lv["D_SR_a"] == LV_DOWN[1] and lv["D_SR_b"] == LV_DOWN[1]:
        R = "R3"
    else:
        R = "R5"
    rm = []
    if R == "R1" and lv["D_SR_b"] != LV_DOWN[0]:
        rm.append(MOD["R1_B_NOT_DOWN"].format(**V))
    if R == "R1" and lv["Rec_a"] != LV_UP[0]:
        rm.append(MOD["R1_UPPER"].format(**V))
    if R == "R2" and flagged["a"]:
        rm.append(MOD["R2_A_FLAG"].format(why=_tier_flag_why(est, "a")))
    for x in ("a", "b"):
        if flagged[x]:
            rm.append(MOD["R_TIER_FLAG"].format(x=x, why=_tier_flag_why(est, x)))
        elif sign_of(sr[f"Dir_{x}"]) in (NEG, POS):
            rm.append(MOD["R_DIR"].format(x=x, v=f_sr(sr[f"Dir_{x}"])))
    put("R", R, rm)
    # ---- H 组：只在 R1 时落行
    if R == "R1":
        if flagged["h"]:
            put("H", "H3", h_flag=f"（S±h {K['h']['why']}）")
        elif lv["D_SR_h"] == LV_DOWN[0]:
            put("H", "H1")
        elif lv["D_SR_h"] == LV_DOWN[1]:
            put("H", "H2")
        else:
            put("H", "H3", h_flag="")
    # ---- F 组：只在 R1、R2 时落行
    doff = est["F"]["delta_off"]
    doff_lv = sign_of(doff) if doff is not None else None
    if R in ("R1", "R2"):
        dfa_ok, dfb_ok = not flagged["a"], not flagged["b"]
        fa_lower = dfa_ok and is_lower(sr["D_Fa"], thr)
        f3 = fa_lower and doff_lv != NEG
        f1 = dfb_ok and is_higher(sr["D_Fb"], thr)
        generic = MOD["F_GENERIC"].format(DFa=V["DFa"], doff=f_k(doff, 2)) if (fa_lower and doff_lv == NEG) else None
        if sc_flags:
            put("F", "F4", sc_flag="；" + MOD["F_SC_FLAG"].format(why="「" + "」「".join(sc_flags) + "」"))
        elif f3:
            dt = (MOD["F3_DOFF_NA"].format(n=est["F"]["delta_off_n"], min_keys=mk) if doff is None
                  else MOD["F3_DOFF_NOTNEG"].format(doff=f_k(doff, 2), doff_lv=doff_lv))
            put("F", "F3", [MOD["F3_ALSO_F1"].format(**V) if f1 else None], doff_text=dt)
        elif f1:
            mods = [MOD["F1_DSRB"] if lv["D_SR_b"] == LV_DOWN[0] else None,
                    MOD["F1_DF_DOWN"].format(**V) if lv["D_F"] == LV_DOWN[0] else None,
                    MOD["F1_DF_NOT"].format(**V) if lv["D_F"] == LV_DOWN[1] else None, generic]
            put("F", "F1", mods)
        elif lv["D_F"] == LV_DOWN[0]:
            floor = (not flagged["b"]) and lv["Rec_SC"] != LV_UP[0] and lv["Rec_b"] != LV_UP[0]
            put("F", "F2", [MOD["F2_FLOOR"].format(**V) if floor else None, generic])
        else:
            put("F", "F4", [generic], sc_flag="")
    else:
        dec["notes"].append(MOD["F_REPORT_ONLY"].format(R=R))
    # ---- A 组（先落，K4 要看它）
    A = est["A"]
    if A["n_pairs"] < mk or A["flagged"]:
        why = "；".join(w for w in (MOD["A0_N"].format(n=A["n_pairs"], min_keys=mk) if A["n_pairs"] < mk else None,
                                     MOD["A0_FLAG"] if A["flagged"] else None) if w)
        put("A", "A0", why=why)
    else:
        sA = sign_of(A["kH_obj"])
        put("A", {POS: "A1", ZERO: "A2", NEG: "A3"}[sA], kHo=f_k(A["kH_obj"]))
    if block in A_CONTAINER_NA:
        dec["sent"]["A"]["modifiers"].append(A_CONTAINER_NA[block])
    # ---- K 组
    if K["b"]["judgeable"]:
        x = "b"
    elif K["a"]["judgeable"]:
        x = "a"
    else:
        x = None
    kw = {}
    if x is None:
        cls = "；".join(f"{c} 闭合 {est['closure'][c]['n_closed']}/{est['n_kept']}" for c in ("C4", "S-b", "S-a", "S+a", "S+b"))
        mods = [MOD["K0_WHY"].format(x=t, why=K[t]["why"]) for t in ("b", "a")]
        mods.append(_tc_asym("b", K))
        put("K", "K0", mods, closure=cls)
    else:
        kx = K[x]
        sH, sK = sign_of(kx["kH"]), sign_of(kx["k"])
        if sH == NEG:
            krow = {NEG: "K1", ZERO: "K2", POS: "K6"}[sK]
        elif sH == ZERO:
            krow = {NEG: "K3", ZERO: "K4", POS: "K6"}[sK]
        else:
            krow = "K5"
        kw = dict(x=x, x_px=px[x], k=f_k(kx["k"]), kH=f_k(kx["kH"]), kB=f_k(kx["kB"]),
                  sec6="；".join(f"{n} {v}" for n, v in SEC6[block].items()) + f"（{SEC6_NAME}，粗算"
                  + (f"；{SEC6_SUB_A}" if x == "a" else "") + "）")
        mods = []
        if x == "a":
            mods.append(MOD["K_SUB"].format(why=K["b"]["why"]))
        if krow in ("K1", "K2", "K3"):
            dd = est["dose"]
            if not dd["judgeable"]:
                mods.append(MOD["K_DOSE_NA"].format(why=dd["why"]))
            else:
                sd = sign_of(dd["d_E_px"])
                mods.append(MOD[{NEG: "K_DOSE_NEG", ZERO: "K_DOSE_ZERO", POS: "K_DOSE_POS"}[sd]].format(dd=f_k(dd["d_E_px"], 2)))
            if x == "b":
                mods.append(MOD["K_A_TIER"].format(ka=f_k(K["a"]["k"]), kHa=f_k(K["a"]["kH"])))
                if krow in ("K1", "K2") and K["a"]["judgeable"] and sign_of(K["a"]["k"]) == NEG and sign_of(K["a"]["kH"]) == NEG:
                    mods.append(MOD["K_A_BOTH"])
        if krow in ("K1", "K2") and sign_of(kx["kHref"]) != NEG:
            mods.append(MOD["K_REF"].format(x=x, kHref=f_k(kx["kHref"])))
        if krow == "K2" and sign_of(kx["kB"]) == NEG:
            mods.append(MOD["K2_OBJ"].format(x=x, kB=f_k(kx["kB"])))
        if krow == "K3" and sign_of(kx["kB"]) == POS:
            mods.append(MOD["K3_OBJ"].format(x=x, kB=f_k(kx["kB"])))
        if krow == "K4":
            mods.append(MOD["K4_A1"] if dec["rows"]["A"] == "A1" else MOD["K4_NOT_A1"])
        if krow != "K6":
            mods.append(_tc_asym(x, K))
            mods.append(MOD["K_CLOSED_ONLY"].format(x=x, n_pair=kx["n_pairs"], n_up=est["closure"][f"S+{x}"]["n_unclosed"],
                                                    n_um=est["closure"][f"S-{x}"]["n_unclosed"], kref=f_k(kx["kref"]),
                                                    kHref=f_k(kx["kHref"])))
        put("K", krow, mods, **kw)
        if krow == "K1" and dec["rows"]["A"] == "A1":
            dec["sent"]["A"]["modifiers"].append(MOD["A1_K1"])
    # ---- P 组：先 a 档替补，再判
    Pq = est["P"]
    q1, q2 = P["kappa_q"]
    xp = "b"
    pb1, pb2 = Pq[f"b{q1}"], Pq[f"b{q2}"]
    pmods = []
    if not (pb1["judgeable"] and pb2["judgeable"]) and Pq[f"a{q1}"]["judgeable"] and Pq[f"a{q2}"]["judgeable"]:
        xp = "a"
        pmods.append(MOD["P_SUB"].format(why="；".join(f"κ_{p['q']}(b) {p['why']}" for p in (pb1, pb2) if not p["judgeable"])))
    p1, p2 = Pq[f"{xp}{q1}"], Pq[f"{xp}{q2}"]
    if not p1["judgeable"]:
        dec["notes"].append(MOD["P_NONE"].format(why=p1["why"]))
        dec["P_tier"] = xp
    else:
        k1neg = sign_of(p1["est"]) == NEG
        if p2["judgeable"]:
            k2neg = sign_of(p2["est"]) == NEG
        else:
            k2neg = False
            pmods.append(MOD["P_K2_NA"].format(why=p2["why"]))
        prow = "P1" if (k1neg and k2neg) else "P2" if k1neg else "P3"
        if prow in ("P2", "P3") and dec["rows"]["K"] in ("K1", "K2"):
            pmods.append(MOD["P_KK"])
        put("P", prow, pmods, x=xp, k1=f_k(p1["est"]), k2=f_k(p2["est"]) if p2["judgeable"] else f"不可判（{p2['why']}）")
        dec["P_tier"] = xp
    dec["K_tier"] = x
    # ---- 块内合成（§8）
    dec["synthesis"] = synthesize(dec, est)
    return dec


def _tc_asym(x, K):
    e = K[x]["tc_diff"]
    if K[x]["n_pairs"] and sign_of(e) in (NEG, POS):
        return MOD["K_TC_ASYM"].format(x=x, m=e["point"], lo=e["lo"], hi=e["hi"])
    return None


def synthesize(dec, est):
    rows, S = dec["rows"], dec["sent"]
    out = {}
    R, F = rows["R"], rows["F"]
    parts = []
    if R in ("R1", "R2") and F in ("F2", "F3"):
        nums = f"D_SR(a) = {f_sr(est['sr']['D_SR_a'])}，D_SR(b) = {f_sr(est['sr']['D_SR_b'])}"
        parts.append(R_RESTRICTED.format(nums=nums))
    else:
        parts.append(S["R"]["write"])
    if R in ("R1", "R2") and F:
        parts.append(S["F"]["write"])
        if F == "F2":
            parts.append(F2_CAVEAT)
    if R == "R1" and rows["H"]:
        parts.append(S["H"]["write"])
    out["丙"] = "；".join(parts)
    out["位置（并列，不用因果连接词）"] = S["K"]["write"] if rows["K"] in ("K1", "K2") else None
    out["接触前"] = S["P"]["write"] if rows["P"] else None
    out["方向参照"] = S["A"]["write"] if rows["A"] else None
    sel = []
    if R in ("R1", "R2") and rows["K"] == "K1" and rows["P"] == "P1":
        sel.append(SELECTIVE_BOTH)
    elif R in ("R1", "R2"):
        sel.append(SELECTIVE_SR)
    else:
        sel.append(SELECTIVE_NONE)
    if F == "F3":
        sel.append(SELECTIVE_F3)
    out["选择性证据"] = sel
    return out


# ================================================================ 文本
def fmt_block(block, est, dec, gate, P):
    L = [f"======== {block}：保留键 {est['n_kept']}（{est['n_clusters']} 簇），剔除 {gate['n_excluded']}"]
    L.append("SR（最后一步）：" + "，".join(f"{c} {v:.1f}" for c, v in est["SR"].items()))
    L.append("SR（任一步，并列）：" + "，".join(f"{c} {v:.1f}" for c, v in est["SR_ever"].items()))
    sr = est["sr"]
    L.append("§5.1/§5.2 配对差（百分点）：" + "；".join(f"{n} {f_sr(e)}（{dec['levels'].get(n, '')}）" for n, e in sr.items()))
    L.append("并列（任一步成功）：" + "；".join(f"{n} {f_sr(e)}" for n, e in est["sr_ever"].items()))
    flags = {c: v for c, v in gate["cell_flags"].items() if v}
    L.append(f"格标记：{flags or '无'}")
    if dec["numbers_only"]:
        L.append("【只报数、不判行】" + "；".join(dec["notes"]))
    else:
        for g in "RHFKAP":
            s = dec["sent"].get(g)
            if s is None:
                continue
            L.append(f"【{s['row']} {s['name']}】（{s['claim']}）")
            L.append(f"  可写：{s['write']}")
            for m in s["modifiers"]:
                L.append(f"  修饰：{m}")
            L.append(f"  不能写：{s['nowrite']}")
            L.append(f"  停止条件：{s['stop']}")
        for n in dec["notes"]:
            L.append(f"  注：{n}")
        if "synthesis" in dec:
            for n, v in dec["synthesis"].items():
                L.append(f"  合成·{n}：{v}")
    K = est["K"]
    for x in ("a", "b", "h"):
        k = K[x]
        L.append(f"K {x} 档：成对 {k['n_pairs']}；k {f_k(k['k'])}；k^H {f_k(k['kH'])}；k^B {f_k(k['kB'])}；竖直 O {f_k(k['O_Ev_px'], 2)} px；"
                 f"厘米 O_E {f_k(k['O_E_cm'], 2)}、O_H {f_k(k['O_H_cm'], 2)}、O_B {f_k(k['O_B_cm'], 2)}；t_c(+)−t_c(−) {f_k(k['tc_diff'], 1)}；"
                 f"参考时刻 k^ref {f_k(k['kref'])}、k^{{H,ref}} {f_k(k['kHref'])}；可判 {k['judgeable']} {k['why']}")
    L.append(f"Δ_dose：E {f_k(est['dose']['d_E_px'], 2)} px，手 {f_k(est['dose']['d_H_px'], 2)} px（交集 {est['dose']['n']}，可判 {est['dose']['judgeable']}）")
    L.append("闭合：" + "；".join(f"{c} {v['n_closed']} 闭合，t_c 中位 {v['tc_median']} 四分位 {v['tc_q25_q75']}，未闭合 {v['n_unclosed']}"
                                f"（参考时刻 E {v['unclosed_at_ref_E_median']}，阶段 {v['unclosed_stages']}）" for c, v in est["closure"].items()))
    L.append("五点曲线 E(格) − E(C4)：" + "；".join(f"{c} {v['median_px']}（n={v['n']}）" for c, v in est["curve5"].items()))
    F = est["F"]
    L.append(f"F：δ_off {f_k(F['delta_off_raw'], 2)} px（四格都闭合 {F['delta_off_n']}，算得出 {F['delta_off'] is not None}）；"
             f"副本闭合处手项增益 {f_k(F['kH_SC_closure'])}；第 8 步手项 SC {f_k(F['step8_kH_SC'])}、S±a {f_k(F['step8_kH_Sa'])}、逐键差 {f_k(F['step8_diff_px'], 2)} px")
    A = est["A"]
    L.append(f"A：成对 {A['n_pairs']}；k^H_obj {f_k(A['kH_obj'])}；误差版 {f_k(A['k_obj_E'])}；第 4 步 {f_k(A['kH_row4'])}；第 8 步 {f_k(A['kH_row8'])}；"
             f"闭合率 {A['closure_rate']}；SR {A['SR']}；手碰真实任务物 {A['contact_rate']}；合成不足 {A['flagged']}"
             + (f"；{A_CONTAINER_NA[block]}" if block in A_CONTAINER_NA else ""))
    L.append("P：" + "；".join(f"κ_{p['q']}({p['x']}) {f_k(p['est'])}（合格 {p['n_qual']}，可判 {p['judgeable']}）" for p in est["P"].values()))
    L.append(f"P 分层（前两次查询 S±b 线不碰外圈的 {est['P_ring_free_kappa2_b']['n_keys']} 键）：κ_2(b) {f_k(est['P_ring_free_kappa2_b']['est'])}；"
             f"κ_2(b) / k^H_b = {est['P_ratio_kappa2b_over_kHb']}")
    q0 = est["q0"]
    L.append("§5.6 G^cmd_0：" + "；".join(f"{x} 均值 {f_k(v['mean'])} 中位 {f_k(v['median'])}" for x, v in q0["G_cmd_0"].items())
             + f"；{LOCALQ_NOTE} 第 0 次查询 {q0['localq_q0']}；κ_1(b)/G^cmd_0(b) = {q0['kappa1_b_over_Gcmd0_b']}；"
             f"执行比乘 localq 增益（§18.1 第 4 条，只并排）{EXEC_RATIO_GAIN[block]}，κ_1(b) {f_k(est['P']['b%d' % P['kappa_q'][0]]['est'])}；"
             f"|r_0(C3)−r_0(C4)| {q0['norm_C3_minus_C4']:.2f}，|r_0(C0)−r_0(C4)| {q0['norm_C0_minus_C4']:.2f}，|r_0(C4)| {q0['norm_C4']:.2f}；"
             f"噪声底 {q0['noise_floor_by_cell']}")
    k1e, k2e = est["P"]["b%d" % P["kappa_q"][0]]["est"], est["P"]["b%d" % P["kappa_q"][1]]["est"]
    L.append(f"§6 {SEC6_NAME}（并排）：{SEC6[block]}；实测 k^H_b {f_k(K['b']['kH'])}，κ_1(b) {f_k(k1e)}，"
             f"κ_2(b) {f_k(k2e)}（{SEC6_NOTE}）")
    L.append(f"阶段分布（0 到 5）：{est['stages']}")
    L.append(f"按误差分三等份：{est['err_terciles']}")
    return "\n".join(L)


# ================================================================ 载入与护栏
def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def params(man):
    try:
        p = man["params"]
        P = {"thr": float(p["sr_thr_pts"]), "min_keys": int(p["min_keys"]), "inview_thr": float(p["inview_thr"]),
             "paste_loss_thr": float(p["paste_loss_thr"]), "flag_frac": float(p["flag_frac"]),
             "disp": float(p["precontact_disp_m"]), "ring_px": int(p["ring_px"]),
             "kappa_q": [int(q) for q in p["kappa_q"]], "curve_q": [int(q) for q in p["curve_q"]],
             "B": int(p["bootstrap"]["B"]), "seed": int(p["bootstrap"]["seed"]), "ci": [float(c) for c in p["bootstrap"]["ci"]],
             "localq_q0": p["localq_q0"],
             "s_tilde": {x: float(man["s_tilde"][x]) for x in ("a", "b", "h")}, "qe": int(man["query_every"])}
    except (KeyError, TypeError, ValueError) as e:
        raise Refuse(f"冻结清单缺参数或格式不对：{e!r}")
    if len(P["kappa_q"]) != 2:
        raise Refuse("params.kappa_q 必须是两点")
    return P


def resolve(man_path, rel):
    return rel if os.path.isabs(rel) else os.path.join(os.path.dirname(os.path.abspath(man_path)), rel)


def load_keys(man, mp, block):
    ks = man.get("keysets", {}).get(block)
    if not ks or any(f not in ks for f in ("path", "sha256", "n")):
        raise Refuse(f"冻结清单 keysets.{block} 缺 path / sha256 / n")
    p = resolve(mp, ks["path"])
    if not os.path.exists(p):
        raise Refuse(f"主键集 {p} 不存在")
    raw = open(p, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != ks["sha256"]:
        raise Refuse(f"主键集 {block} 的 sha256 {got} 与冻结清单 {ks['sha256']} 不符")
    keys = json.loads(raw)
    names = [k["key"] for k in keys]
    if len(keys) != int(ks["n"]) or len(set(names)) != len(names):
        raise Refuse(f"主键集 {block} 键数 {len(keys)} 与清单 n={ks['n']} 不符，或有重复键")
    if any(k.get("block") != block or int(k.get("trial", -1)) not in MAIN_TRIALS for k in keys):
        raise Refuse(f"主键集 {block} 含别的块或试次不在 5 到 8")
    return keys


def check_manifest(man):
    if man.get("protocol_sha256") != PROTOCOL_SHA256:
        raise Refuse(f"冻结清单的 protocol_sha256 {man.get('protocol_sha256')} 不是 v3.1 定稿 {PROTOCOL_SHA256}")
    if set(man.get("cell_order", [])) != set(CELLS) or len(man["cell_order"]) != len(CELLS):
        raise Refuse(f"冻结清单 cell_order 不是十三格 {CELLS}")
    if any(c not in man.get("cells", {}) for c in CELLS):
        raise Refuse("冻结清单 cells 缺格")
    for f in ("code_sha256", "gpu_whitelist", "vk_icd_filenames", "line_width", "query_every", "anchor_keysets",
              "tref_file", "tref_sha256"):
        if f not in man:
            raise Refuse(f"冻结清单缺 {f}")
    for f, h in sorted(man["code_sha256"].items()):
        p = f if os.path.isabs(f) else os.path.join(HERE, f)
        if not os.path.exists(p):
            raise Refuse(f"清单 code_sha256 登记的 {f} 不存在（{p}）")
        if sha(p) != h:
            raise Refuse(f"{f} 的哈希与冻结清单不符")
    if any(int(man["anchor_keysets"].get(b, {}).get("n", -1)) != ANCHOR_N for b in BLOCKS):
        raise Refuse(f"冻结清单 anchor_keysets 的 n 不是 {ANCHOR_N}")


def default_anchor_post(root):
    return os.path.join(os.path.dirname(root), "out_cue_anchor_post", "ANCHOR_POST.json")


def default_init_cmp(root):
    return os.path.join(root, "INIT_CMP.json")


def check_stage1(sd):
    """STAGE1_OK 存在，且每块每个锚点格逐位相同（解读同 cue_gate.check_anchor_post）。"""
    p = os.path.join(sd, "STAGE1_OK")
    if not os.path.exists(p):
        raise Refuse(f"{p} 不存在：采前锚点比对没有通过（或没做）")
    try:
        d = json.load(open(p))
    except ValueError as e:
        raise Refuse(f"{p} 不是合法 json：{e}")
    probs = []
    for b in BLOCKS:
        for c in G.ANCHOR_CELLS:
            v = d.get(b, {}).get(c) if isinstance(d, dict) and isinstance(d.get(b), dict) else None
            if not isinstance(v, dict):
                probs.append(f"{b}/{c} 缺比对")
                continue
            for f in ("differ", "ref_missing", "missing"):
                if f not in v or v[f]:
                    probs.append(f"{b}/{c} {f}={v.get(f)}")
            if not (v.get("equal") == v.get("n_ref") == ANCHOR_N):
                probs.append(f"{b}/{c} equal={v.get('equal')} n_ref={v.get('n_ref')}（应为 {ANCHOR_N}）")
    if probs:
        raise Refuse(f"{p} 内容不过：{probs[:5]}")


def check_order(root, keys, ap_path, ic_path, sd):
    """§4 的先后：STAGE1_OK 存在且内容过；采后锚点与初始比对都晚于全部主键记录。不满足就拒读。"""
    check_stage1(sd)
    latest, latest_p = None, None
    for b in BLOCKS:
        for c in CELLS:
            for k in keys[b]:
                p = os.path.join(root, b, c, k["key"] + ".npz")
                if os.path.exists(p):
                    m = os.stat(p).st_mtime_ns
                    if latest is None or m > latest:
                        latest, latest_p = m, p
    if latest is None:
        raise Refuse("没有任何主键记录")
    later = [ap_path, ic_path]
    ad = os.path.dirname(ap_path)
    for dp, _, fs in os.walk(ad):
        later += [os.path.join(dp, f) for f in fs if f.endswith(".npz")]
    for p in later:
        if not os.path.exists(p):
            raise Refuse(f"{p} 不存在：无法核它晚于全量采集")
        if os.stat(p).st_mtime_ns <= latest:
            raise Refuse(f"{p} 的修改时间不晚于主键记录里最晚的一条（{latest_p}）：§4 要求全量采完之后再做")


def take_lock(root, sd):
    os.makedirs(sd, exist_ok=True)
    lp = os.path.join(sd, OUT_NAME + ".lock")
    try:
        fd = os.open(lp, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise Refuse(f"读数锁 {lp} 已存在（内容：{open(lp).read().strip()!r}），拒绝第二次读数")
    with os.fdopen(fd, "w") as fh:
        fh.write(os.path.realpath(root) + "\n")
    return lp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="out_cue_closed")
    ap.add_argument("--manifest", default=None, help="只在根目录有 SYNTHETIC_DATA 标记时可用")
    ap.add_argument("--anchor-post", default=None, help="缺省 <root>/../out_cue_anchor_post/ANCHOR_POST.json")
    ap.add_argument("--init-cmp", default=None, help="缺省 <root>/INIT_CMP.json")
    ap.add_argument("--investigated-unresolved", action="store_true",
                    help="协调方已查过锚点或初始比对的不同、查不明：不过的部分只报数、不判行（§4）")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    txt, js = os.path.join(root, OUT_NAME + ".txt"), os.path.join(root, OUT_NAME + ".json")
    if os.path.exists(txt) or os.path.exists(js):
        raise Refuse(f"{txt} 或 .json 已存在，拒绝第二次读数")
    synthetic = os.path.exists(os.path.join(root, "SYNTHETIC_DATA"))
    if synthetic:
        mp = a.manifest
        if not mp or not os.path.exists(mp):
            raise Refuse("合成数据需要 --manifest 指向存在的清单")
    else:
        for nm, v, dflt in (("--anchor-post", a.anchor_post, default_anchor_post(root)), ("--init-cmp", a.init_cmp, default_init_cmp(root))):
            if v is not None and os.path.abspath(v) != dflt:
                raise Refuse(f"真实数据下 {nm} 只许取默认路径 {dflt}")
        if a.manifest is not None and os.path.abspath(a.manifest) != FROZEN:
            raise Refuse("真实数据不接受别的 --manifest，只认 b3/FROZEN_cue_widowx")
        if os.path.realpath(root) != os.path.realpath(REAL_ROOT):
            raise Refuse(f"真实数据的 --root 只能是 {REAL_ROOT}（给的是 {os.path.realpath(root)}）")
        if not os.path.exists(FROZEN):
            raise Refuse("真实数据只能在冻结（b3/FROZEN_cue_widowx 存在）后读")
        mp = FROZEN
    sd = G.state_dir(root) if synthetic else REAL_STATE
    with open(mp) as f:
        man = json.load(f)
    check_manifest(man)
    P = params(man)
    man_sha = sha(mp)
    keys = {b: load_keys(man, mp, b) for b in BLOCKS}
    tp = resolve(mp, man["tref_file"])
    if not os.path.exists(tp) or sha(tp) != man["tref_sha256"]:
        raise Refuse("t_ref 清单缺失，或 sha256 与冻结清单 tref_sha256 不符")
    tref = json.load(open(tp))
    for b in BLOCKS:
        if any(k["key"] not in tref.get(b, {}) for k in keys[b]):
            raise Refuse(f"t_ref 清单缺 {b} 的主键")
    # ---- 开头核三样（§17 第 12 条、§4）
    anchor_n = {b: int(man["anchor_keysets"][b]["n"]) for b in BLOCKS}
    ap_path = os.path.abspath(a.anchor_post) if a.anchor_post else default_anchor_post(root)
    ic_path = os.path.abspath(a.init_cmp) if a.init_cmp else default_init_cmp(root)
    anc = G.check_anchor_post(ap_path, BLOCKS, anchor_n)
    ini = G.check_init_cmp(ic_path, BLOCKS, CELLS, {b: len(keys[b]) for b in BLOCKS})
    if not anc["ok"] and not a.investigated_unresolved:
        raise Refuse(f"锚点 --post（环境不变检查）不过或缺失，停在读数之前查原因：{anc['problems'][:5]}")
    for b in BLOCKS:
        if not ini[b]["ok"] and not a.investigated_unresolved:
            raise Refuse(f"{b} 新键初始比对不过或缺失，停在读数之前查原因：{ini[b]['problems'][:5]}")
    gates = {}
    for b in BLOCKS:
        g = G.run(root, b, keys[b], man, man_sha)
        if g["incomplete"]:
            raise Refuse(f"{b}：未采完 {len(g['incomplete'])} 个格键，例如 {g['incomplete'][:3]}")
        if not g["kept_keys"]:
            raise Refuse(f"{b}：门槛后没有保留的键")
        gates[b] = g
    check_order(root, keys, ap_path, ic_path, sd)
    lock = take_lock(root, sd)
    # ---- 读数
    res = {"protocol_sha256": PROTOCOL_SHA256, "lock": lock, "manifest_path": mp, "manifest_sha256": man_sha, "synthetic": synthetic,
           "params": P, "anchor_post": anc, "init_cmp": ini, "blocks": {}}
    texts = []
    qs = sorted(set(P["curve_q"]) | set(P["kappa_q"]) | {1, 2})
    for b in BLOCKS:
        g = gates[b]
        kept = g["kept_keys"]
        clus = {k["key"]: str(k["cluster"]) for k in keys[b]}
        D = {c: {k: derive(os.path.join(root, b, c, k + ".npz"), P["qe"], tref[b][k], qs, P["disp"]) for k in kept} for c in CELLS}
        est = compute_block(D, kept, clus, P, g)
        reasons = []
        if not anc["ok"]:
            reasons.append(NUMBERS_ONLY["anchor"])
        if not ini[b]["ok"]:
            reasons.append(NUMBERS_ONLY["init"])
        if G.OUTLINE_WEAK in g["cell_flags"].get("C4", []):
            reasons.append(NUMBERS_ONLY["c4"])
        dec = decide_block(b, est, g["cell_flags"], P, reasons)
        res["blocks"][b] = {"decision": dec, "estimate": est, "gate": g, "gate_summary": G.summary(g)}
        texts.append(fmt_block(b, est, dec, g, P))
    lab = {b: tuple(res["blocks"][b]["decision"]["rows"][x] for x in "RHFKAP") for b in BLOCKS}
    both = {"lead": BLOCKS_LEAD + "；".join(f"{b}：{'、'.join(r for r in lab[b] if r) or '只报数'}" for b in BLOCKS),
            "labels_differ": lab["B1"] != lab["B2"], "global_stop": GLOBAL_STOP}
    if both["labels_differ"]:
        both["diff_note"] = BLOCKS_DIFF
    res["both_blocks"] = both
    head = [f"WidowX 轮廓位置响应 v3.1 一次性读数（{'合成数据' if synthetic else '真实数据'}）；清单 {mp}（sha256 {man_sha}）",
            f"锚点 --post：{'过' if anc['ok'] else '不过 ' + str(anc['problems'][:3])}；初始比对：" +
            "；".join(f"{b} {'过' if ini[b]['ok'] else '不过'}" for b in BLOCKS)]
    tail = [f"两块并排：{both['lead']}" + (f"（{both['diff_note']}）" if both["labels_differ"] else ""), GLOBAL_STOP]
    text = "\n".join(head + texts + tail)
    res["text"] = text
    fd = os.open(txt, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as fh:
        fh.write(text + "\n\n" + "\n\n".join(res["blocks"][b]["gate_summary"] for b in BLOCKS) + "\n")
    fd = os.open(js, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print(text)
    return res


if __name__ == "__main__":
    try:
        main()
    except Refuse as e:
        print("拒读：", e, file=sys.stderr)
        sys.exit(2)
