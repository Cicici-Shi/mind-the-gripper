"""WidowX 轮廓位置响应协议 v3.1（protocol_cue_widowx_v3_1_final.md，sha256 5af593e2…aa0）的写死参数，采集与清单共用。
冻结时这些值原样写进 FROZEN_cue_widowx（freeze_cue.py）；门槛与读数只从冻结清单取，不 import 本文件的数值。
"""
BLOCKS = ("B1", "B2")
TARGET = {"B1": "eggplant", "B2": "bridge_spoon_generated_modified"}  # §2 目标参考点 = 任务物 actor 原点
# §3 目标容器：茄子块的篮子 = actor "sink"（环境 put_on_in_scene.py 376 行起；episode 的目标 dummy_sink_target_plane 不可见），
# 勺子块的毛巾 = "table_cloth_generated_shorter"（也在保护名单里）
CONTAINER = {"B1": "sink", "B2": "table_cloth_generated_shorter"}
TCP_LINK = "ee_gripper_link"
W_ENV, H_ENV = 640, 480
AMP_IN = {"h": 6, "a": 12, "b": 24}           # §3 输入像素
AMP_ENV = {"h": 17, "a": 34, "b": 69}          # §3 环境像素（输入像素 x 640/224 取整）
S_TILDE = {k: round(v * 224.0 / W_ENV, 4) for k, v in AMP_ENV.items()}  # 5.95、11.9、24.15
QUERY_EVERY = 4
LINE_WIDTH = 8
RING_PX = 6
PROJ_DIL_PX = 6

# §3 闭环十三格：格名 -> (画法, 环境像素平移 [dx, dy])；OB 的平移作用在任务物图像上，轮廓不动
CELLS = {
    "C0": ("C0", (0, 0)), "C3": ("C3", (0, 0)), "C4": ("C4", (0, 0)),
    "S-b": ("S", (-69, 0)), "S-a": ("S", (-34, 0)), "S+a": ("S", (34, 0)), "S+b": ("S", (69, 0)),
    "SC-b": ("SC", (-69, 0)), "SC+b": ("SC", (69, 0)),
    "OB-b": ("OB", (-69, 0)), "OB+b": ("OB", (69, 0)),
    "S-h": ("S", (-17, 0)), "S+h": ("S", (17, 0)),
}
CELL_ORDER = ["C0", "C3", "C4", "S-b", "S-a", "S+a", "S+b", "SC-b", "SC+b", "OB-b", "OB+b", "S-h", "S+h"]
ANCHOR_CELLS = ["C0", "C3", "C4"]
ANCHOR_DIRS = ("out_cue_anchor", "out_cue_anchor_post")  # §4：锚点记录单独的目录
STAGE1_N = 10

# §13 的数值线（冻结清单 params；门槛与读数只从清单取）
PARAMS = {
    "sr_thr_pts": 10,            # 第 4 条：三档 10 点
    "min_keys": 77,              # 第 6 条：成对样本与合格键
    "inview_thr": 0.8,           # 第 7 条：视野内比例
    "paste_loss_thr": 0.2,       # 第 7 条：贴图因出画丢掉两成以上
    "flag_frac": 0.05,           # 第 8 条：标记线 5% 的帧
    "precontact_disp_m": 0.01,   # 第 9 条：水平 1 cm
    "ring_px": 6,                # 第 10 条
    "kappa_q": [1, 2],           # 第 11 条：P 组判定点（第 4、8 步）
    "curve_q": [1, 2, 3, 4, 5],
    "bootstrap": {"B": 10000, "seed": 0, "ci": [2.5, 97.5]},  # 第 5 条
}


def keyset_paths(b3_dir, manifest, kind="keysets"):
    """键集只从冻结清单（或候选）登记的路径取，没登记就报错（§17 第 6 条：不回退到第一阶段正式清单）。
    kind = "keysets"（主键集）或 "anchor_keysets"。返回 {块: 绝对路径}。"""
    import json as _j
    import os as _o
    assert manifest and _o.path.exists(manifest), f"需要冻结清单或候选：{manifest}"
    ks = _j.load(open(manifest)).get(kind)
    assert ks and all(b in ks for b in BLOCKS), f"清单里没有登记 {kind}"
    return {b: _o.path.join(b3_dir, ks[b]["path"]) for b in BLOCKS}
