"""bench2 键清单与 reset 规格（协议 protocol_bench2.md v1 §1、§3）。

键 = 一次显式指定的 reset。
  * Bridge（B1 茄子、B2 勺子）：键 =（episode_id 0..23，试次 0..4），120 键；按 episode_id 聚类。
    fork 里 episode_id 取模 24（茄子 8 个位置 x 3 个朝向；勺子 12 个位置对 x 2 个朝向），经 obj_init_options 指定。
    reset seed 只由（块，episode_id）定，所以同一 episode_id 的试次之间只差模型噪声。
  * F1（可乐罐）：键 = 显式 reset seed，每个键一个独立场景；按键重采样。
    作者 eval_simpler_fractal.sh 里的横放 / 竖放 / 立放三个环境各 40 个种子（三段互不相交）。
    prepackaged 模式下物体位置、urdf 版本都由 reset seed 的随机数决定。
  * F2（move near，协议 v1.1）：键 =（episode_id 0..59，试次 0、1），120 键，按 episode_id 聚成 60 簇。
    episode_id 经 obj_init_options 指定（move_near_in_scene.py 539-542 行，取模 60）。
    urdf 版本不能经 options 指定：prepackaged 模式每次 reset 由 _episode_rng.choice 抽（同文件 186-193 行），
    实测 urdf = URDF_ORDER[RandomState(seed).choice(4)]（旧键核查 400/400 相符）。所以对每个（episode_id, urdf）
    预先确定地搜出种子写进键清单；eval_views 在 reset 后断言实际 urdf 与 episode_id 与键一致。
    试次 t 的 urdf 序号 = (episode_id + 2t) mod 4；四个版本各 30 次。
  * 校准键：Bridge 用试次 9；F1 用种子 900000 起，与正式种子（0..119）不相交；
    F2 用（episode_id, urdf 序号 (episode_id + 1) mod 4），试次记为 9，与正式键在（episode_id, urdf）上不相交。
"""
import hashlib
import json
import sys

import numpy as np

CKPT_BRIDGE = "$PROJ_ROOT/bench2_simpler/ckpt/bridge_beta_step19296_2024-12-26_22-30_42.pt"
CKPT_FRACTAL = "$PROJ_ROOT/bench2_simpler/b2/ckpt/fractal_beta_step29576_2024-12-29_13-10_42.pt"

# 环境 id 与作者评测脚本（slurm/eval_simpler_bridge.sh、eval_simpler_fractal.sh）一致
BLOCKS = {
    "B1": dict(tasks=["widowx_put_eggplant_in_basket"], ckpt=CKPT_BRIDGE, cfg="config/eval/bridge.yaml", body="widowx"),
    "B2": dict(tasks=["widowx_spoon_on_towel"], ckpt=CKPT_BRIDGE, cfg="config/eval/bridge.yaml", body="widowx"),
    "F1": dict(tasks=["google_robot_pick_horizontal_coke_can", "google_robot_pick_vertical_coke_can",
                      "google_robot_pick_standing_coke_can"], ckpt=CKPT_FRACTAL, cfg="config/eval/fractal_coke.yaml",
               body="google"),
    "F2": dict(tasks=["google_robot_move_near_v0"], ckpt=CKPT_FRACTAL, cfg="config/eval/fractal_move.yaml", body="google"),
}
BRIDGE_EPISODES = 24
BRIDGE_FORMAL_TRIALS = [0, 1, 2, 3, 4]
BRIDGE_CALIB_TRIAL = 9
FRACTAL_FORMAL_PER_BLOCK = 120
FRACTAL_CALIB_SEED0 = 900000

# ---------------------------------------------------------------- 补充协议 v1（protocol_bench2_ext_fill_lag.md）
# 块与格：B2 做填充对照（§3：F0 = C0，Fk = C1，Fw = C1w，FL = C1L）；F1 另做更长的滞后（§4：C3、C4、C5、C5L）。
# 键沿用第一阶段冻结的正式键清单。
EXT_CELLS = {
    "B2": ["C0", "C1", "C1w", "C1L"],
    "F1": ["C0", "C1", "C1w", "C1L", "C3", "C4", "C5", "C5L"],
}
# 第一阶段已有、要求重跑后与第一阶段正式记录逐位相同的格（§1）；新格是 C1w、C1L、C5L
EXT_OLD_CELLS = ("C0", "C1", "C3", "C4", "C5")
EXT_NEW_CELLS = ("C1w", "C1L", "C5L")
EXT_STAGE1_N = 10  # §1：先每个旧格跑这么多键做逐位比对，全部相同才放全量

# 保护名单（协议 v1 §2）：逐块写死，只含可移动的任务物体（actor 名），不含篮子、水槽、柜子、桌面
PROTECT = {
    "B1": ["eggplant"],
    "B2": ["bridge_spoon_generated_modified", "table_cloth_generated_shorter"],
    "F1": ["opened_coke_can"],
    # move near：场景里的三个物体（源、目标、干扰）都是可移动的任务物体；8 个候选名里每回合出现 3 个
    "F2": ["blue_plastic_bottle", "baked_opened_pepsi_can", "orange", "baked_opened_7up_can", "baked_apple",
           "baked_sponge", "baked_opened_coke_can", "baked_opened_redbull_can"],
}
PROTECT_EXPECT_N = {"B1": 1, "B2": 2, "F1": 1, "F2": 3}  # 每回合场景里应找到的保护物体个数

# urdf 版本序号：与作者环境代码（move_near_in_scene.py 187-191 行）和 SimplerEnv
# scripts/rt1_move_near_visual_matching.sh 第 15 行列出的顺序相同（脚本里的 None 即 ""）
URDF_ORDER = ["", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2",
              "recolor_cabinet_visual_matching_1"]
F2_EPISODES = 60
F2_FORMAL_TRIALS = [0, 1]
F2_CALIB_TRIAL = 9
F2_SEED_BASE = {"formal": 3_000_000, "calib": 4_000_000}


def urdf_of_seed(seed):
    """prepackaged move near 在 reset(seed) 时抽到的 urdf 序号（_episode_rng 的第一次抽样）。"""
    return int(np.random.RandomState(int(seed)).choice(len(URDF_ORDER)))


def f2_seed(which, ep, trial, u):
    """确定地搜种子：从 base + (ep*10 + 试次)*100 起往上找第一个抽到 urdf 序号 u 的种子。"""
    base = F2_SEED_BASE[which] + (ep * 10 + trial) * 100
    for s in range(base, base + 100):
        if urdf_of_seed(s) == u:
            return s
    raise RuntimeError((which, ep, trial, u))


def is_bridge(block):
    return block in ("B1", "B2")


def noise_seed(block, cfg_id, trial, q):
    """第 q 次查询前的 torch 种子：sha256(块|场景配置|试次|查询序号) 前 8 字节，与格无关。"""
    return int.from_bytes(hashlib.sha256(f"{block}|{cfg_id}|{trial}|{q}".encode()).digest()[:8], "big")


def bridge_env_seed(block, ep):
    return int.from_bytes(hashlib.sha256(f"{block}|ep{ep:02d}|env".encode()).digest()[:4], "big")


def _key(block, task, cfg_id, trial, env_seed, reset, cluster):
    return dict(key=f"{block}__{cfg_id}__t{trial}", block=block, task=task, cfg_id=cfg_id, trial=int(trial),
                env_seed=int(env_seed), reset=reset, cluster=cluster)


def make_keys(block, which):
    """which = 'formal' 或 'calib'。Bridge 先铺满 24 个 episode 再加试次。"""
    b = BLOCKS[block]
    out = []
    if is_bridge(block):
        trials = BRIDGE_FORMAL_TRIALS if which == "formal" else [BRIDGE_CALIB_TRIAL]
        for t in trials:
            for e in range(BRIDGE_EPISODES):
                out.append(_key(block, b["tasks"][0], f"ep{e:02d}", t, bridge_env_seed(block, e),
                                {"episode_id": e}, f"ep{e:02d}"))
    elif block == "F2":
        for e in range(F2_EPISODES):
            trials = [(t, (e + 2 * t) % 4) for t in F2_FORMAL_TRIALS] if which == "formal" \
                else [(F2_CALIB_TRIAL, (e + 1) % 4)]
            for t, u in trials:
                cid = f"ep{e:02d}_u{u}"
                out.append(_key(block, b["tasks"][0], cid, t, f2_seed(which, e, t, u),
                                {"episode_id": e, "urdf_version": URDF_ORDER[u]}, f"ep{e:02d}"))
        if which == "formal":  # 先铺满 60 个 episode 再加试次
            out.sort(key=lambda k: (k["trial"], k["reset"]["episode_id"]))
    else:
        n_task = len(b["tasks"])
        per = FRACTAL_FORMAL_PER_BLOCK // n_task
        base = 0 if which == "formal" else FRACTAL_CALIB_SEED0
        n = per if which == "formal" else 40
        for ti, task in enumerate(b["tasks"]):
            short = task.replace("google_robot_", "")
            for i in range(n):
                # F1 三个朝向环境用互不相交的种子段（同一种子在三个环境里抽到几乎相同的位置与 urdf，冒烟时核实）
                s = base + ti * n + i
                cid = f"{short}_s{s}"
                out.append(_key(block, task, cid, 0, s, {}, cid))
    return out


def spread(keys, n, by_task=True):
    """校准用：取 n 个。多环境的块按环境轮流取（每个环境的前几个）；单环境取前 n 个。"""
    tasks = sorted({k["task"] for k in keys}, key=[k["task"] for k in keys].index)
    if len(tasks) == 1 or not by_task:
        return keys[:n]
    out, i = [], 0
    pools = {t: [k for k in keys if k["task"] == t] for t in tasks}
    while len(out) < n:
        t = tasks[i % len(tasks)]
        out.append(pools[t][i // len(tasks)])
        i += 1
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--set", choices=["formal", "calib"], required=True)
    ap.add_argument("--n", type=int, default=None, help="calib：取前 n 个（多环境块按环境轮流）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ks = make_keys(a.block, a.set)
    if a.set == "calib" and a.n:
        ks = spread(ks, a.n)
    json.dump(ks, open(a.out, "w"), indent=1)
    print(a.block, a.set, len(ks), "keys ->", a.out, file=sys.stderr)
