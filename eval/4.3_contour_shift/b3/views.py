"""bench2 五格画面（协议 protocol_bench2.md §2），只改喂给策略的图。

  C0 原图
  C1 涂黑：当步（可见时）分割图里属于 G 的像素置 0
  C3 擦掉：G 的可视体设为不可见后重新取观测（visual matching 的真实背景叠加照常进行），取 RGB
  C4 擦掉后画当前轮廓：C3 的图上，用当步可见时的 G 掩膜画轮廓，黑色；落在受保护物体像素上的线不画
  C5 擦掉后画滞后轮廓：同 C4，掩膜取 L 步之前那一帧（t < L 时取第 0 步）

轮廓与保护规则照 LIBERO 的 $PROJ_ROOT/maskinfo/embody/maskinfo.py（sha 3d301bad...）移植，不另起一套：
  O = (G 减去其 k_in 次 4 邻域腐蚀，帧外按 G 处理) 并 (G 的 k_out 次 3x3 膨胀减去 G)，k_in = w // 2，k_out = w - k_in；
  画线区 = O 减去 P_full，P_full = 隐藏夹爪后那次渲染的分割图里属于受保护集合 P 的像素；线色 (0, 0, 0)。
  滞后帧按步号查（maskinfo_pi 的做法），并断言取到的确实是 t - L（t < L 时为第 0 步）。
受保护集合 P（协议 v1 §2）：keys.PROTECT 逐块写死的 actor 名（只含可移动的任务物体），每回合断言找到的个数。
线宽（v1）：round(3 x 图宽 / 256)。滞后 L（v1）：两个查询间隔与 0.8 秒中较长者，IoU 按轮廓线像素算。
"""
import cv2
import numpy as np

CELLS = ("C0", "C1", "C3", "C4", "C5")
HIDE_CELLS = ("C3", "C4", "C5")
OUTLINE_CELLS = ("C4", "C5")

# 夹爪连杆集合 G：连杆名匹配 finger|gripper（协议 §2），按身体写死；运行时断言与匹配结果一致
G_LINKS = {
    "widowx": ["gripper_link", "gripper_prop_link", "gripper_bar_link", "fingers_link", "left_finger_link",
               "right_finger_link", "ee_gripper_link"],
    "google": ["link_gripper", "link_gripper_tcp", "link_finger_right", "link_finger_tip_right",
               "link_finger_nail_right", "link_finger_left", "link_finger_tip_left", "link_finger_nail_left"],
}
MODEL_SIZE = (224, 224)  # open-pi-zero SimplerAdapter: cv2.resize(image, (224, 224), INTER_LANCZOS4)
LINE_VIS_THR = 16  # 缩放到 224 后，C4/C5 图与 C3 底图在某像素任一通道差 > 16（满量程 255）才算「线还看得见」


def line_width_for(width_px):
    """协议 v1 §2：round(3 x 图宽 / 256)（Python 的 round，640 宽 -> round(7.5) = 8）。"""
    return int(round(3 * width_px / 256))


def lag_for(query_every, control_hz):
    """协议 v1 §2：两个查询间隔与 0.8 秒中较长者，向上取到查询间隔的整数倍（单位：控制步）。"""
    import math
    need = max(2 * query_every, math.ceil(0.8 * control_hz - 1e-9))
    return int(math.ceil(need / query_every) * query_every)


def cam_name(robot_uid):
    return "3rd_view_camera" if "widowx" in robot_uid.lower() else "overhead_camera"


# ---------------------------------------------------------------- 掩膜工具（逐字移植 maskinfo / maskrepl）
def _dilate_sq(mask, r):
    m = np.asarray(mask, bool)
    if r <= 0:
        return m.copy()
    a = m.copy()
    for s in range(1, r + 1):
        a[s:] |= m[:-s]
        a[:-s] |= m[s:]
    b = a.copy()
    for s in range(1, r + 1):
        b[:, s:] |= a[:, :-s]
        b[:, :-s] |= a[:, s:]
    return b


def erode4(mask, k=1, outside=False):
    m = np.asarray(mask, bool)
    for _ in range(k):
        p = np.pad(m, 1, constant_values=outside)
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def outline(G, width):
    G = np.asarray(G, bool)
    k_in = width // 2
    k_out = width - k_in
    inner = G & ~erode4(G, k_in, outside=True) if k_in > 0 else np.zeros_like(G)
    outer = _dilate_sq(G, k_out) & ~G if k_out > 0 else np.zeros_like(G)
    return inner | outer


def iou(a, b):
    u = int((a | b).sum())
    return None if u == 0 else float((a & b).sum()) / u


def to_model(img):
    return cv2.resize(img, MODEL_SIZE, interpolation=cv2.INTER_LANCZOS4)


def mask_to_model(m):
    """几何口径：掩膜按面积缩放到 224，覆盖率 >= 0.5 的像素算在内。"""
    return cv2.resize(np.asarray(m, np.float32), MODEL_SIZE, interpolation=cv2.INTER_AREA) >= 0.5


def rgb_of(raw_cam):
    """与 RGBDObservationWrapper 相同的转换：Color[..., :3] 0..1 浮点 -> uint8。"""
    return np.clip(raw_cam["Color"][..., :3] * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 每回合的画面器
class ViewMaker:
    def __init__(self, env, cell, body, lag_steps, line_width, query_every, protect_names, protect_n):
        assert cell in CELLS, cell
        assert lag_steps > 0 and lag_steps % query_every == 0, (lag_steps, query_every)
        self.env, self.u = env, env.unwrapped
        self.cell, self.body = cell, body
        self.L, self.w, self.qe = int(lag_steps), int(line_width), int(query_every)
        self.cam = cam_name(self.u.robot_uid)
        self.protect_names, self.protect_n = list(protect_names), int(protect_n)

    # reset 之后必须调用：reconfigure 会重建机器人和物体，连杆 id 和可视体都换了
    def on_reset(self):
        u = self.u
        links = u.agent.robot.get_links()
        import re
        pat = re.compile(r"finger|gripper", re.I)
        matched = [l.name for l in links if pat.search(l.name)]
        want = G_LINKS[self.body]
        assert sorted(matched) == sorted(want), f"G 连杆清单与常量不符：{matched} vs {want}"
        self.g_links = [l for l in links if l.name in want]
        self.gids = np.array(sorted(l.id for l in self.g_links), np.int32)
        self.robot_ids = np.array(sorted(l.id for l in links), np.int32)
        acts = [x for x in u.get_actors() if x.name in self.protect_names]
        assert len(acts) == self.protect_n, f"保护物体个数 {len(acts)} != {self.protect_n}：{[x.name for x in acts]}"
        self.p_names = [x.name for x in acts]
        p_ids = [x.id for x in acts]
        self.pids = np.array(sorted(p_ids), np.int32)
        self.n_vb = sum(len(l.get_visual_bodies()) for l in self.g_links)
        self.hist = {}  # 查询步号 -> 可见时的 G 掩膜
        self.restore_maxdiff = None

    def _set_vis(self, v):
        for l in self.g_links:
            for vb in l.get_visual_bodies():
                vb.set_visibility(v)

    def render_hidden(self):
        """隐藏 G 渲染一次，取 RGB 与分割，随即恢复可见。物理状态不受影响（可视性只影响渲染）。"""
        self._set_vis(0.0)
        try:
            raw = self.u.get_obs()["image"][self.cam]
            rgb, seg = rgb_of(raw), raw["Segmentation"][..., 1].copy()
        finally:
            self._set_vis(1.0)
        return rgb, seg

    def check_restore(self, obs):
        """reset 后核一次：隐藏再恢复后重新渲染，必须与 reset 的观测逐像素相同。"""
        self.render_hidden()
        raw = self.u.get_obs()["image"][self.cam]
        self.restore_maxdiff = int(np.abs(rgb_of(raw).astype(int) - obs["image"][self.cam]["rgb"].astype(int)).max())
        return self.restore_maxdiff

    def make(self, obs, step):
        """obs：env.reset / env.step 的观测（夹爪可见）。返回 (策略输入图, 统计量 dict)。"""
        assert step % self.qe == 0, (step, self.qe)
        rgb = obs["image"][self.cam]["rgb"]
        seg = obs["image"][self.cam]["Segmentation"][..., 1]
        G = np.isin(seg, self.gids)
        self.hist[int(step)] = G.copy()
        R = np.isin(seg, self.robot_ids)
        st = {"step": int(step), "G_px": int(G.sum()), "R_px": int(R.sum()), "arm_px": int((R & ~G).sum()),
              # 涂黑操纵强度的原料：G 内像素三通道平均值之和（涂黑后为 0，所以差 = 原值）
              "G_rgb_sum": float(rgb[G].astype(np.float64).mean(axis=1).sum()) if G.any() else 0.0}
        # 当前与滞后掩膜的 IoU（任何格都算，供滞后校准）：掩膜 IoU 与 LIBERO 口径的轮廓 IoU 各一份
        if step >= self.L:
            Gp = self.hist[int(step) - self.L]
            st["iou_lag_outline"] = iou(outline(G, self.w), outline(Gp, self.w))  # v1 判据：按线像素
            st["iou_lag_mask"] = iou(G, Gp)  # 仅参考
        if step >= 2 * self.L:  # 供「L 加倍一次」的判断（v1 §2），不必重跑
            Gp2 = self.hist[int(step) - 2 * self.L]
            st["iou_lag2_outline"] = iou(outline(G, self.w), outline(Gp2, self.w))
        E = np.zeros_like(G)
        if self.cell == "C0":
            img = rgb
        elif self.cell == "C1":
            img = rgb.copy()
            img[G] = 0
            E = G
        else:
            img_h, seg_h = self.render_hidden()
            st["G_vis_after_hide_px"] = int(np.isin(seg_h, self.gids).sum())
            d = np.abs(img_h.astype(np.int16) - rgb.astype(np.int16))
            ch = d.max(axis=2) > 0
            st["hide_changed_px"] = int(ch.sum())
            st["hide_changed_out_G_px"] = int((ch & ~G).sum())
            st["hide_changed_out_G_t30_px"] = int(((d.sum(axis=2) > 30) & ~G).sum())
            img = img_h
            E = G.copy()
            if self.cell in OUTLINE_CELLS:
                if self.cell == "C4":
                    Gl, lag_step = G, int(step)
                else:
                    lag_step = max(int(step) - self.L, 0)
                    Gl = self.hist[lag_step]
                    assert lag_step == (int(step) - self.L if step >= self.L else 0)
                O = outline(Gl, self.w)
                P_full = np.isin(seg_h, self.pids)
                draw = O & ~P_full
                img = img_h.copy()
                img[draw] = 0
                E = G | draw
                lm = np.abs(to_model(img).astype(np.int16) - to_model(img_h).astype(np.int16)).max(axis=2)
                st.update(lag_step=lag_step, O_px=int(O.sum()), O_drawn_px=int(draw.sum()),
                          O_protected_px=int((O & P_full).sum()), P_full_px=int(P_full.sum()),
                          line_px_224=int((lm > LINE_VIS_THR).sum()),
                          line_px_224_geom=int(mask_to_model(draw).sum()),
                          Gl_px=int(Gl.sum()))
        diff = np.any(img != rgb, axis=2)
        st["changed_px"] = int(diff.sum())
        st["changed_out_E_px"] = int((diff & ~E).sum())
        st["E_px"] = int(E.sum())
        return img, st


# ================================================================ b3 原型（2026-11-13）：信息内容格的画法，参数化，未冻结
# 以上 C0 到 C5 的代码与 b2/views.py 逐字相同；以下只新增，不改上面任何一行。
# 新格都以 C3 的擦掉底图为底，线色、线宽、保护截断与 C4 相同：
#   T 固定开度模板：机械臂保持当前关节，只把夹爪手指关节设为固定值，渲染一次取可见 G 掩膜画轮廓，随即恢复关节。
#   S 整体偏移：当前轮廓 O（与 C4 同一个 O）在图像坐标里平移 (dx, dy)，出画部分丢弃，再按当步保护物体截断。
#   R 碎片重排：C4 实际画出的线切成若干段，在其包围盒内重排，再按当步保护物体截断。
import hashlib
import math

NEW_CELLS = ("T", "S", "R", "X", "SC")
# 草案 protocol_cue_draft_v0 §2.0：所有画线格 = 擦除底图上把 D = Q 减 P_full(t) 置黑，各格只在 Q 上不同。
#   T：Q = outline(G_tpl)，var {"tpl": [q_open 向量]}（草案 §2.2；"open"/"f0"/小数 是早先的可选定义，保留）
#   S：Q = outline(G) 平移，var {"mode": "axis", "axis": [1,0] 水平（草案默认）或 [0,1] 竖直（备用）, "mag", "sign"}
#   X：Q = scramble_tiles(outline(G))（草案 §2.3，主口径）；R 是早先的「对 C4 实画线按线段重排」，只作可选
#   SC：Q = outline(G) 并 平移后的 outline(G)（反方建议改 2：当前轮廓保留，另画一条偏移副本），var 同 S
#   T 的闭合版：var {"tpl": [参考轨迹最小开度的手指关节向量]}，与张开版同一函数
# 手指关节（agent 配置里的 gripper_joint_names；运行时断言存在）与「全开」的定义：
# 两种身体的 gripper_closedness（googlerobot.py 60 行、widowx.py 85 行）为 0 时的关节值。
FINGER_JOINTS = {"widowx": ["left_finger", "right_finger"], "google": ["joint_finger_right", "joint_finger_left"]}
TCP_LINK = {"widowx": "ee_gripper_link", "google": "link_gripper_tcp"}
# 指尖间距用的两个连杆（草案 §2.2 的 q_open 取法）：连杆原点之间的欧氏距离
FINGERTIP_LINKS = {"widowx": ["left_finger_link", "right_finger_link"], "google": ["link_finger_tip_left", "link_finger_tip_right"]}


def scramble_seed(block, key, q, cam):
    """草案 §2.3 第 4 条：sha256("scramble|块|键|查询序号|相机") 前 8 字节。"""
    return int.from_bytes(hashlib.sha256(f"scramble|{block}|{key}|{q}|{cam}".encode()).digest()[:8], "big")


def _dihedral(t, k):
    """正方形 8 种对称：k 0..3 旋转 90 度的次数，k 4..7 再加左右镜像。"""
    t = np.rot90(t, k % 4)
    return t[:, ::-1] if k >= 4 else t


def _tile_grid(O, s):
    """草案 §2.3 第 2 条：紧包围盒向右下扩到 s 的整数倍；越界整体平移回画面；仍放不下裁到最大整数倍。返回 (y0, x0, ny, nx)。"""
    H, W = O.shape
    ys, xs = np.nonzero(O)
    x0, y0 = int(xs.min()), int(ys.min())
    nx, ny = int(math.ceil((int(xs.max()) + 1 - x0) / s)), int(math.ceil((int(ys.max()) + 1 - y0) / s))
    nx, ny = min(nx, W // s), min(ny, H // s)
    x0 = min(x0, W - nx * s)
    y0 = min(y0, H - ny * s)
    return y0, x0, ny, nx


def _h64(sstr):
    return int.from_bytes(hashlib.sha256(sstr.encode()).digest()[:8], "big")


def episode_perm(seed_str, ny, nx):
    """每回合固定的排列（专家意见第 4 条的备选）：种子不含查询序号。源瓦片按 h("src"|行|列) 排序、目标瓦片按
    h("dst"|行|列) 排序，第 k 个源放到第 k 个目标；行列号相对网格左上角。包围盒增减一行一列时只有部分配对改变。
    对称变换 = h("tr"|源行|源列) mod 8。返回 (perm, tr)：目标 i（行优先）放源 perm[i]，变换 tr[i]。"""
    n = ny * nx
    rc = [(r, c) for r in range(ny) for c in range(nx)]
    src = sorted(range(n), key=lambda i: _h64(f"{seed_str}|src|{rc[i][0]}|{rc[i][1]}"))
    dst = sorted(range(n), key=lambda i: _h64(f"{seed_str}|dst|{rc[i][0]}|{rc[i][1]}"))
    perm = np.empty(n, np.int64)
    perm[np.array(dst)] = np.array(src)
    tr = np.array([_h64(f"{seed_str}|tr|{rc[perm[i]][0]}|{rc[perm[i]][1]}") % 8 for i in range(n)], np.int64)
    return perm, tr


def scramble_tiles(O, s0, rng, min_nonempty=4, episode_seed=None):
    """草案 §2.3：瓦片边长 s0（不足 min_nonempty 个非空瓦片时减半一次，仍不足则原样返回并标退化）。
    全部瓦片位置做均匀随机排列，每片独立均匀抽 8 种正方形对称之一；网格外的线原样保留。返回 (Q, info)。"""
    O = np.asarray(O, bool)
    info = {"s0": int(s0)}
    if not O.any():
        info.update(empty=True, degenerate=False)
        return O.copy(), info
    for s in (int(s0), int(s0) // 2):
        y0, x0, ny, nx = _tile_grid(O, s)
        grid = O[y0:y0 + ny * s, x0:x0 + nx * s]
        tiles = grid.reshape(ny, s, nx, s).swapaxes(1, 2).reshape(ny * nx, s, s)
        nonempty = int(tiles.reshape(ny * nx, -1).any(1).sum())
        if nonempty >= min_nonempty:
            break
    inside = np.zeros_like(O)
    inside[y0:y0 + ny * s, x0:x0 + nx * s] = True
    outside_px = int((O & ~inside).sum())
    info.update(s=int(s), ny=int(ny), nx=int(nx), n_tiles=int(ny * nx), n_nonempty=nonempty, outside_grid_px=outside_px,
                empty=False)
    if nonempty < min_nonempty:
        info["degenerate"] = True
        return O.copy(), info
    if episode_seed is None:  # 草案做法：每次查询重新排
        perm = rng.permutation(ny * nx)
        tr = rng.integers(0, 8, ny * nx)
    else:  # 每回合固定
        perm, tr = episode_perm(episode_seed, ny, nx)
    new = np.stack([_dihedral(tiles[perm[i]], int(tr[i])) for i in range(ny * nx)])
    out = O & ~inside
    out[y0:y0 + ny * s, x0:x0 + nx * s] = new.reshape(ny, nx, s, s).swapaxes(1, 2).reshape(ny * s, nx * s)
    info.update(degenerate=False, n_fixed_nonempty=int(sum(1 for i in range(ny * nx) if perm[i] == i and tiles[i].any())))
    return out, info


def reg_shift(A, B, maxs):
    """草案 §2.4 配准：在 |dx|、|dy| <= maxs 的整数平移里，找使 IoU(A, 平移后的 B) 最大者（B 平移出画的部分丢弃）。
    并列取模长最小者。A = 滞后掩膜，B = 当前掩膜。用 FFT 互相关一次算出全部平移的交与平移后面积（取整为整数计数）。
    返回 (dx, dy, 最大 IoU)。dx 向右为正：平移后的 B 是 B 向右移 dx。"""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    H, W = A.shape
    sh = (2 * H, 2 * W)
    fa, fb = np.fft.rfft2(A, sh), np.fft.rfft2(B, sh)
    fo = np.fft.rfft2(np.ones((H, W)), sh)
    inter = np.rint(np.fft.irfft2(fa * np.conj(fb), sh))
    nb = np.rint(np.fft.irfft2(fo * np.conj(fb), sh))
    na = A.sum()
    ds = np.arange(-maxs, maxs + 1)
    I = inter[np.ix_(ds % sh[0], ds % sh[1])]
    N = nb[np.ix_(ds % sh[0], ds % sh[1])]
    den = na + N - I
    iou_ = np.where(den > 0, I / np.maximum(den, 1), 0.0)
    best = iou_.max()
    cand = np.argwhere(iou_ >= best - 1e-12)
    norms = (ds[cand[:, 0]] ** 2 + ds[cand[:, 1]] ** 2)
    j = cand[np.argmin(norms)]
    return int(ds[j[1]]), int(ds[j[0]]), float(best)


def edge_touch(G):
    """G 是否触及画面四条边。"""
    G = np.asarray(G, bool)
    return {"left": bool(G[:, 0].any()), "right": bool(G[:, -1].any()), "top": bool(G[0].any()), "bottom": bool(G[-1].any())}


def finger_open_closed(body, qlim):
    """返回 (全开, 全闭) 的手指关节值。WidowX：closedness = (上限 - q) / (上限 - 下限)，全开 = 上限；
    Google：closedness = q / 上限，全开 = 0、全闭 = 上限。"""
    qlim = np.asarray(qlim, np.float64)
    if body == "widowx":
        return qlim[:, 1].copy(), qlim[:, 0].copy()
    return np.zeros(len(qlim)), qlim[:, 1].copy()


def mask_geom(G, W=None, H=None):
    """掩膜的几何量（原始像素坐标）：像素数、质心、包围盒、主轴角（度，-90..90）。空掩膜返回 None。"""
    ys, xs = np.nonzero(G)
    n = int(len(xs))
    if n == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = xs - cx, ys - cy
    cxx, cyy, cxy = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    ang = 0.5 * math.degrees(math.atan2(2 * cxy, cxx - cyy))
    lam1 = 0.5 * (cxx + cyy) + math.sqrt(max(0.25 * (cxx - cyy) ** 2 + cxy ** 2, 0.0))
    lam2 = 0.5 * (cxx + cyy) - math.sqrt(max(0.25 * (cxx - cyy) ** 2 + cxy ** 2, 0.0))
    return dict(n=n, cx=cx, cy=cy, x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()) + 1, y1=int(ys.max()) + 1,
                ang=ang, elong=float(lam1 / lam2) if lam2 > 1e-9 else None)


def shift_mask(M, dx, dy):
    """整数平移，出画部分丢弃，空出来的部分补 False。dx 向右为正，dy 向下为正。"""
    M = np.asarray(M, bool)
    H, W = M.shape
    out = np.zeros_like(M)
    dx, dy = int(dx), int(dy)
    if abs(dx) >= W or abs(dy) >= H:
        return out
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = M[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def shuffle_seed(block, cfg_id, trial, q):
    """碎片重排的随机种子：sha256(块|场景配置|试次|查询序号|shuffle) 前 8 字节；与格、与参数无关。"""
    return int.from_bytes(hashlib.sha256(f"{block}|{cfg_id}|{trial}|{q}|shuffle".encode()).digest()[:8], "big")


def _kmeans_pts(pts, k, rng, iters=12):
    k = max(1, min(int(k), len(pts)))
    c = pts[rng.choice(len(pts), k, replace=False)].astype(np.float64)
    lab = np.zeros(len(pts), np.int64)
    for _ in range(iters):
        d = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = pts[lab == j].mean(0)
    return lab, k


def shuffle_lines(D, mode, n_seg, rng, flip=True, tries=20):
    """把线 D（bool 掩膜）切段后在 D 的包围盒内重排。
    mode = "perm"：包围盒切成 g x g 格（g = round(sqrt(n_seg))），格子做随机置换（尽量让非空格离开原位），flip 时每格随机转 180 度；
           像素总量精确守恒（包围盒补齐到整格后多出的边上若有线会被裁掉，照记）。
    mode = "rigid"：线像素按坐标 k-means 成 n_seg 段，每段绕自身质心随机旋转 0..360 度、随机平移到包围盒内（最近邻插值），
           每段试 tries 个落点取重叠最少者；段间仍重叠会少算像素，照记。返回 (新线, 信息 dict)。"""
    D = np.asarray(D, bool)
    H, W = D.shape
    ys, xs = np.nonzero(D)
    info = {"mode": mode, "n_seg": int(n_seg)}
    if len(xs) == 0:
        return D.copy(), info
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    out = np.zeros_like(D)
    if mode == "perm":
        g = max(1, int(round(math.sqrt(n_seg))))
        th, tw = int(math.ceil(h / g)), int(math.ceil(w / g))
        cv = np.zeros((g * th, g * tw), bool)
        cv[:h, :w] = D[y0:y1, x0:x1]
        tiles = cv.reshape(g, th, g, tw).swapaxes(1, 2).reshape(g * g, th, tw)
        nonempty = tiles.reshape(g * g, -1).any(1)
        best, best_fix = None, None
        for _ in range(200):
            p = rng.permutation(g * g)
            fix = int((nonempty & (p == np.arange(g * g))).sum())
            if best is None or fix < best_fix:
                best, best_fix = p, fix
            if fix == 0:
                break
        # 新位置 i 放原来的第 best[i] 格
        new = tiles[best].copy()
        if flip:
            rot = rng.random(g * g) < 0.5
            new[rot] = new[rot][:, ::-1, ::-1]
        cv2_ = new.reshape(g, g, th, tw).swapaxes(1, 2).reshape(g * th, g * tw)
        out[y0:y1, x0:x1] = cv2_[:h, :w]
        info.update(grid=g, fixed_nonempty_tiles=int(best_fix), n_nonempty_tiles=int(nonempty.sum()))
    elif mode == "rigid":
        pts = np.stack([xs, ys], 1).astype(np.float64)
        lab, k = _kmeans_pts(pts, n_seg, rng)
        for j in range(k):
            sel = lab == j
            if not sel.any():
                continue
            m = np.zeros((H, W), np.uint8)
            m[ys[sel], xs[sel]] = 1
            cx, cy = float(xs[sel].mean()), float(ys[sel].mean())
            th_ = float(rng.uniform(0.0, 360.0))
            M = cv2.getRotationMatrix2D((cx, cy), th_, 1.0)
            # 旋转后段的半宽高（按段内像素旋转后的极值）
            px = np.stack([xs[sel] - cx, ys[sel] - cy], 1)
            R2 = M[:, :2]
            rp = px @ R2.T
            rx0, rx1, ry0, ry1 = rp[:, 0].min(), rp[:, 0].max(), rp[:, 1].min(), rp[:, 1].max()
            lo_x, hi_x = x0 - rx0, x1 - 1 - rx1
            lo_y, hi_y = y0 - ry0, y1 - 1 - ry1
            # 试 tries 个随机落点，取与已放好的段重叠最少的一个（减少重叠造成的像素损失；仍由同一 rng 决定，可复现）
            best_r, best_ov = None, None
            for _ in range(int(tries)):
                ncx = float(rng.uniform(lo_x, hi_x)) if hi_x > lo_x else 0.5 * (lo_x + hi_x)
                ncy = float(rng.uniform(lo_y, hi_y)) if hi_y > lo_y else 0.5 * (lo_y + hi_y)
                M2 = M.copy()
                M2[0, 2] += ncx - cx
                M2[1, 2] += ncy - cy
                r = cv2.warpAffine(m, M2, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 0
                ov = int((r & out).sum())
                if best_ov is None or ov < best_ov:
                    best_r, best_ov = r, ov
                if ov == 0:
                    break
            out |= best_r
        clip = np.zeros_like(D)
        clip[y0:y1, x0:x1] = True
        out &= clip
        info.update(k=int(k))
    else:
        raise ValueError(mode)
    return out, info


def _phys_snapshot(u):
    """仿真状态快照：机器人 qpos/qvel/qacc/qf/驱动目标/根位姿、全部连杆位姿与速度、场景内全部 actor 与其他关节体。"""
    r = u.agent.robot
    s = {"qpos": r.get_qpos(), "qvel": r.get_qvel(), "qacc": r.get_qacc(), "qf": r.get_qf(),
         "drive_target": r.get_drive_target(), "root": np.concatenate([r.pose.p, r.pose.q])}
    s["links"] = np.array([np.concatenate([l.pose.p, l.pose.q, l.get_velocity(), l.get_angular_velocity()])
                           for l in r.get_links()])
    s["actors"] = np.array([np.concatenate([a.pose.p, a.pose.q, a.get_velocity(), a.get_angular_velocity()])
                            if hasattr(a, "get_velocity") else np.concatenate([a.pose.p, a.pose.q, np.zeros(6)])
                            for a in u.get_actors()]).reshape(-1, 13)
    arts = [a for a in u._scene.get_all_articulations() if a is not r]
    s["arts"] = np.concatenate([np.concatenate([a.get_qpos(), a.get_qvel()]) for a in arts]) if arts else np.zeros(0)
    return {k: np.array(v, np.float64, copy=True) for k, v in s.items()}


def _snap_diff(a, b):
    """逐元素比较两个快照。返回 (是否全等, 每项最大绝对差)。NaN 位置一致也算相等。"""
    eq, md = True, {}
    for k in a:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            eq, md[k] = False, float("inf")
            continue
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        md[k] = float(np.nanmax(np.abs(x - y))) if x.size else 0.0
        if not bool(same.all()):
            eq = False
    return eq, md


class ViewMaker3(ViewMaker):
    """在 ViewMaker 上加新格与校准几何量。C0 到 C5 仍走父类 make()，一行不改。
    var（dict，新格的参数）：
      T：tpl = "open" | "f0" | 0..1 的小数（开度比例，1 = 全开、0 = 全闭，按 gripper_closedness 的线性定义）
      S：mode = "axis" | "lag_frame" | "lag_block"；mag = 偏移像素（原始图像尺度）；sign = +1 | -1；
         axis = [ux, uy]（mode=axis 时的方向，会归一化）；dir_block = [ux, uy]（lag_block 的方向，lag_frame 位移太小时的回退）；
         min_norm = lag_frame 判定方向所需的最小质心位移（原始像素，默认 1.0）
      R：mode = "perm" | "rigid"；n_seg（perm 时取 round(sqrt) 作每边格数）；flip（perm 时是否随机转 180 度，默认 True）；
         tries（rigid 时每段试几个落点，默认 20）
    """

    def setup3(self, block, cfg_id, trial, var=None, geo_lags=(), shadow=None, key=None):
        self.block, self.cfg_id, self.trial = block, cfg_id, int(trial)
        self.key = key
        self.R_cur = None  # 当步可见的机器人掩膜（偏移格记「落在非夹爪机器人像素上」用）
        self.var = dict(var or {})
        self.geo_lags = tuple(int(x) for x in geo_lags)
        self.shadow = shadow  # None 或 dict：C0 回合里额外渲染、只记统计量、不改策略输入
        self.n_tpl_render = 0
        self.tpl_restore_all_equal = True
        self.tpl_restore_maxdiff = {}

    def on_reset3(self):
        self.on_reset()
        u = self.u
        r = u.agent.robot
        aj = r.get_active_joints()
        want = FINGER_JOINTS[self.body]
        names = [j.name for j in aj]
        assert all(w in names for w in want), (want, names)
        self.fj_idx = np.array([names.index(w) for w in want], np.int64)
        qlim = np.asarray(r.get_qlimits(), np.float64)[self.fj_idx]
        self.f_open, self.f_closed = finger_open_closed(self.body, qlim)
        self.f_q0 = np.asarray(r.get_qpos(), np.float64)[self.fj_idx].copy()
        self.tcp = [l for l in r.get_links() if l.name == TCP_LINK[self.body]][0]
        self.kin = {}  # 查询步号 -> 运动学量
        self.lag_dir_fallback = 0

    # ---------------------------------------------------------------- 运动学与几何
    def _kin_now(self):
        u = self.u
        q = np.asarray(u.agent.robot.get_qpos(), np.float64)
        p = self.tcp.pose
        return dict(closedness=float(u.agent.gripper_closedness), finger_q=q[self.fj_idx].tolist(),
                    tcp_p=np.asarray(p.p, np.float64).tolist(), tcp_q=np.asarray(p.q, np.float64).tolist())

    def _tip_now(self):
        lk = {l.name: l for l in self.u.agent.robot.get_links()}
        fa, fb = (lk[n] for n in FINGERTIP_LINKS[self.body])
        return float(np.linalg.norm(np.asarray(fa.pose.p) - np.asarray(fb.pose.p)))

    def tpl_q(self, spec):
        if isinstance(spec, (list, tuple)):  # 草案 §2.2：冻结清单里的 q_open 向量
            v = np.asarray(spec, np.float64)
            assert v.shape == self.f_open.shape, (v, self.f_open)
            return v
        if spec == "open":
            return self.f_open.copy()
        if spec == "f0":
            return self.f_q0.copy()
        frac = float(spec)
        assert 0.0 <= frac <= 1.0, spec
        return self.f_closed + frac * (self.f_open - self.f_closed)

    def render_template(self, qf):
        """手指关节设为 qf、其余关节不动，渲染一次取 RGB 与分割，然后恢复 qpos、qvel，核对仿真状态逐元素相同。
        机制与 render_hidden 相同（改、渲染、在 finally 里复原）；不同处是这里改的是关节状态，所以前后各拍一次快照比较。"""
        u, r = self.u, self.u.agent.robot
        before = _phys_snapshot(u)
        q, qv = r.get_qpos(), r.get_qvel()
        q2 = np.array(q, copy=True)
        q2[self.fj_idx] = qf
        try:
            r.set_qpos(q2.astype(q.dtype))
            raw = u.get_obs()["image"][self.cam]
            rgb, seg = rgb_of(raw), raw["Segmentation"][..., 1].copy()
            q_applied = np.asarray(r.get_qpos(), np.float64)[self.fj_idx]
            lk = {l.name: l for l in r.get_links()}
            fa, fb = (lk[n] for n in FINGERTIP_LINKS[self.body])
            self.last_tpl_tip = float(np.linalg.norm(np.asarray(fa.pose.p) - np.asarray(fb.pose.p)))
        finally:
            r.set_qpos(q)
            r.set_qvel(qv)
        after = _phys_snapshot(u)
        eq, md = _snap_diff(before, after)
        self.n_tpl_render += 1
        if not eq:
            self.tpl_restore_all_equal = False
        for k, v in md.items():
            self.tpl_restore_maxdiff[k] = max(self.tpl_restore_maxdiff.get(k, 0.0), v)
        if getattr(self, "strict_restore", True):
            assert eq, f"模板渲染后仿真状态未复原：{md}"
        return rgb, seg, q_applied

    def render_template_rp(self, qf):
        """不碰物理的模板渲染（草案 §4 第 5 项的后备路径，b3 默认）。
        背景：改 qpos 再复原的做法实测不逐位（B2 回放：复原后 qpos/qvel 全等，但连杆位姿差约 1.8e-6，下一步起轨迹分叉）。
        做法：用 pinocchio 运动学模型分别算当前关节与模板关节下各连杆相对机器人根的位姿，rel = FK_cur^-1 * FK_tpl；
        对位姿变化了的连杆（手指及其子连杆），把其可视体的局部位姿临时设为 rel * 原局部位姿，渲染一次，finally 里复原。
        物理场景里的关节、连杆、碰撞体一概不动。前后仍拍快照核对。"""
        import sapien.core as sapien
        u, r = self.u, self.u.agent.robot
        before = _phys_snapshot(u)
        if getattr(self, "_pin", None) is None:
            self._pin = r.create_pinocchio_model()
        q = np.asarray(r.get_qpos(), np.float64)
        q2 = q.copy()
        q2[self.fj_idx] = qf
        links = r.get_links()
        pin = self._pin
        pin.compute_forward_kinematics(q)
        cur = [pin.get_link_pose(i) for i in range(len(links))]
        pin.compute_forward_kinematics(q2)
        tpl = [pin.get_link_pose(i) for i in range(len(links))]
        saved, moved = [], []
        try:
            for l in links:
                i = l.get_index()
                rel = cur[i].inv() * tpl[i]
                if np.abs(rel.p).max() < 1e-9 and abs(abs(float(rel.q[0])) - 1.0) < 1e-12:
                    continue
                moved.append(l.name)
                for vb in l.get_visual_bodies():
                    lp = vb.local_pose
                    saved.append((vb, lp))
                    vb.set_pose(rel * lp)
            raw = u.get_obs()["image"][self.cam]
            rgb, seg = rgb_of(raw), raw["Segmentation"][..., 1].copy()
        finally:
            for vb, lp in saved:
                vb.set_pose(lp)
        lk = {l.name: l for l in links}
        root = r.pose
        fa, fb = (root * tpl[lk[n].get_index()] for n in FINGERTIP_LINKS[self.body])
        self.last_tpl_tip = float(np.linalg.norm(np.asarray(fa.p) - np.asarray(fb.p)))
        self.last_tpl_moved = moved
        after = _phys_snapshot(u)
        eq, md = _snap_diff(before, after)
        self.n_tpl_render += 1
        if not eq:
            self.tpl_restore_all_equal = False
        for k, v in md.items():
            self.tpl_restore_maxdiff[k] = max(self.tpl_restore_maxdiff.get(k, 0.0), v)
        if getattr(self, "strict_restore", True):
            assert eq, f"模板渲染后仿真状态未复原：{md}"
        return rgb, seg, np.asarray(qf, np.float64)

    def _lag_mask(self, step, L):
        return self.hist[max(int(step) - int(L), 0)]

    def _shift_vec(self, G, step, v):
        mode = v.get("mode", "axis")
        src = mode
        if mode == "dxdy":  # 直接给整数平移 [dx, dy]（回放统计用）
            dx, dy = (int(x) for x in v["dxdy"])
            n = max(np.hypot(dx, dy), 1e-12)
            return dx, dy, src, [dx / n, dy / n]
        if mode == "axis":
            u_ = np.asarray(v.get("axis", [1.0, 0.0]), np.float64)
        elif mode in ("lag_frame", "lag_block"):
            u_ = None
            if mode == "lag_frame":
                g0, g1 = mask_geom(G), mask_geom(self._lag_mask(step, self.L))
                if g0 is not None and g1 is not None:
                    d = np.array([g1["cx"] - g0["cx"], g1["cy"] - g0["cy"]])  # 当前 -> 滞后
                    if np.hypot(*d) >= float(v.get("min_norm", 1.0)):
                        u_ = d
            if u_ is None:
                db = v.get("dir_block")
                assert db is not None, "lag_block 或 lag_frame 回退需要 dir_block"
                u_ = np.asarray(db, np.float64)
                if mode == "lag_frame":
                    src = "lag_frame_fallback"
                    self.lag_dir_fallback += 1
        else:
            raise ValueError(mode)
        u_ = u_ / max(np.hypot(*u_), 1e-12)
        s = float(v.get("sign", 1)) * float(v["mag"])
        return int(round(s * u_[0])), int(round(s * u_[1])), src, u_.tolist()

    def _vis224(self, img, img_h):
        lm = np.abs(to_model(img).astype(np.int16) - to_model(img_h).astype(np.int16)).max(axis=2)
        return int((lm > LINE_VIS_THR).sum())

    def _draw_new(self, cell, v, G, step, img_h, seg_h, st, prefix=""):
        """返回新格的策略输入图与画线区；统计量写进 st（键名带 prefix）。"""
        P_full = np.isin(seg_h, self.pids)
        O_cur = outline(G, self.w)
        D_cur = O_cur & ~P_full  # C4 实际画出的线
        q = int(step) // self.qe
        if cell == "T":
            qf = self.tpl_q(v.get("tpl", "open"))
            # tpl_method："renderpose"（默认，不碰物理）| "qpos"（改关节再复原，实测不逐位，只作可选）
            if v.get("tpl_method", "renderpose") == "qpos":
                _, seg_t, q_app = self.render_template(qf)
            else:
                _, seg_t, q_app = self.render_template_rp(qf)
            Gt = np.isin(seg_t, self.gids)
            O = outline(Gt, self.w)
            draw = O & ~P_full
            gt, gc = mask_geom(Gt), mask_geom(G)
            st.update({prefix + "tpl_q": qf.tolist(), prefix + "tpl_q_applied": q_app.tolist(),
                       prefix + "Gt_px": int(Gt.sum()), prefix + "iou_tpl_mask": iou(Gt, G),
                       prefix + "iou_tpl_outline": iou(O, O_cur), prefix + "iou_tpl_drawn": iou(draw, D_cur),
                       prefix + "tpl_tip_dist": self.last_tpl_tip, prefix + "cur_tip_dist": self._tip_now(),
                       prefix + "tpl_moved_links": getattr(self, "last_tpl_moved", None),
                       prefix + "tpl_centroid_shift": None if (gt is None or gc is None) else
                       [gt["cx"] - gc["cx"], gt["cy"] - gc["cy"]]})
        elif cell == "S":
            dx, dy, src, u_ = self._shift_vec(G, step, v)
            O = shift_mask(O_cur, dx, dy)
            draw = O & ~P_full
            n0 = int(O_cur.sum())
            st.update({prefix + "shift_dxdy": [dx, dy], prefix + "shift_dir_src": src, prefix + "shift_dir": u_,
                       prefix + "O_cur_px": n0, prefix + "O_shift_inview_px": int(O.sum()),
                       prefix + "frac_inview": None if n0 == 0 else float(O.sum()) / n0,
                       prefix + "frac_protected": None if O.sum() == 0 else float((O & P_full).sum()) / float(O.sum()),
                       prefix + "iou_shift_drawn_vs_cur": iou(draw, D_cur),
                       prefix + "frac_on_arm": None if (self.R_cur is None or draw.sum() == 0) else
                       float((draw & self.R_cur & ~G).sum()) / float(draw.sum()),
                       prefix + "iou224_shiftQ_vs_Q": iou(mask_to_model(O), mask_to_model(O_cur))})
        elif cell == "SC":
            dx, dy, src, u_ = self._shift_vec(G, step, v)
            Oc = shift_mask(O_cur, dx, dy)
            O = O_cur | Oc
            draw = O & ~P_full
            n0 = int(O_cur.sum())
            Dc = Oc & ~P_full
            st.update({prefix + "shift_dxdy": [dx, dy], prefix + "copy_inview_px": int(Oc.sum()),
                       prefix + "frac_inview": None if n0 == 0 else float(Oc.sum()) / n0,
                       prefix + "frac_protected": None if Oc.sum() == 0 else float((Oc & P_full).sum()) / float(Oc.sum()),
                       prefix + "frac_on_arm": None if (self.R_cur is None or Dc.sum() == 0) else
                       float((Dc & self.R_cur & ~G).sum()) / float(Dc.sum()),
                       prefix + "copy_overlap_with_cur": None if Oc.sum() == 0 else float((Oc & O_cur).sum()) / float(Oc.sum())})
        elif cell == "X":
            # var {"seed_mode": "query"（草案，默认）| "episode"（每回合固定排列，种子不含查询序号）, "s", "min_nonempty"}
            rng = np.random.default_rng(scramble_seed(self.block, self.key, q, self.cam))
            eseed = f"scramble|{self.block}|{self.key}|{self.cam}" if v.get("seed_mode", "query") == "episode" else None
            O, info = scramble_tiles(O_cur, int(v.get("s", 4 * self.w)), rng, int(v.get("min_nonempty", 4)), eseed)
            draw = O & ~P_full
            gc, gx = mask_geom(D_cur), mask_geom(draw)
            n0 = int(D_cur.sum())
            bb = None
            if gc is not None and gx is not None:
                bb = ((gx["x1"] - gx["x0"]) * (gx["y1"] - gx["y0"])) / float((gc["x1"] - gc["x0"]) * (gc["y1"] - gc["y0"]))
            st.update({prefix + "scr_info": info, prefix + "scr_px_ratio": None if n0 == 0 else float(draw.sum()) / n0,
                       prefix + "scr_bbox_area_ratio": bb, prefix + "iou_scr_vs_cur": iou(draw, D_cur),
                       prefix + "scr_outside_grid_frac": None if not O_cur.any() else info.get("outside_grid_px", 0) / float(O_cur.sum())})
        elif cell == "R":
            rng = np.random.default_rng(shuffle_seed(self.block, self.cfg_id, self.trial, q))
            Rm, info = shuffle_lines(D_cur, v.get("mode", "perm"), int(v.get("n_seg", 16)), rng, bool(v.get("flip", True)),
                                     int(v.get("tries", 20)))
            O = Rm
            draw = Rm & ~P_full
            gc, gr = mask_geom(D_cur), mask_geom(draw)
            n0 = int(D_cur.sum())
            bb = None
            if gc is not None and gr is not None:
                bb = ((gr["x1"] - gr["x0"]) * (gr["y1"] - gr["y0"])) / float((gc["x1"] - gc["x0"]) * (gc["y1"] - gc["y0"]))
            st.update({prefix + "shuf_info": info, prefix + "shuf_px_ratio": None if n0 == 0 else float(draw.sum()) / n0,
                       prefix + "shuf_bbox_area_ratio": bb,
                       prefix + "shuf_bbox": None if gr is None else [gr["x0"], gr["y0"], gr["x1"], gr["y1"]],
                       prefix + "cur_bbox": None if gc is None else [gc["x0"], gc["y0"], gc["x1"], gc["y1"]],
                       prefix + "iou_shuf_vs_cur": iou(draw, D_cur),
                       prefix + "shuf_protected_px": int((Rm & P_full).sum())})
        else:
            raise ValueError(cell)
        img = img_h.copy()
        img[draw] = 0
        st.update({prefix + "O_px": int(O.sum()), prefix + "O_drawn_px": int(draw.sum()),
                   prefix + "O_protected_px": int((O & P_full).sum()), prefix + "line_px_224": self._vis224(img, img_h),
                   prefix + "line_px_224_geom": int(mask_to_model(draw).sum())})
        return img, draw

    def _geo_stats(self, G, step, st):
        """校准几何量：当前 G 的几何、可见像素（原始与 224）、运动学；对每个 geo_lags 算与滞后掩膜的 IoU。"""
        H, W = G.shape
        st["geo"] = mask_geom(G)
        st["G_px_224"] = int(mask_to_model(G).sum())
        st["kin"] = self.kin[int(step)]
        O = outline(G, self.w)
        lagd = {}
        for L in self.geo_lags:
            if step >= L:
                Gp = self.hist[int(step) - L]
                lagd[str(L)] = {"iou_mask": iou(G, Gp), "iou_outline": iou(O, outline(Gp, self.w))}
        st["geo_lag"] = lagd

    def make3(self, obs, step):
        """新入口。C0 到 C5 直接交给父类 make()（不改）；新格走 _draw_new。geo_lags 与 shadow 只追加统计量。"""
        assert step % self.qe == 0, (step, self.qe)
        self.kin[int(step)] = self._kin_now()
        self.R_cur = np.isin(obs["image"][self.cam]["Segmentation"][..., 1], self.robot_ids)
        if self.cell in CELLS:
            img, st = self.make(obs, step)
            G = self.hist[int(step)]
        else:
            rgb = obs["image"][self.cam]["rgb"]
            seg = obs["image"][self.cam]["Segmentation"][..., 1]
            G = np.isin(seg, self.gids)
            self.hist[int(step)] = G.copy()
            R = np.isin(seg, self.robot_ids)
            st = {"step": int(step), "G_px": int(G.sum()), "R_px": int(R.sum()), "arm_px": int((R & ~G).sum())}
            if step >= self.L:
                Gp = self.hist[int(step) - self.L]
                st["iou_lag_outline"] = iou(outline(G, self.w), outline(Gp, self.w))
                st["iou_lag_mask"] = iou(G, Gp)
            img_h, seg_h = self.render_hidden()
            st["G_vis_after_hide_px"] = int(np.isin(seg_h, self.gids).sum())
            img, draw = self._draw_new(self.cell, self.var, G, step, img_h, seg_h, st)
            E = G | draw
            diff = np.any(img != rgb, axis=2)
            st["changed_px"] = int(diff.sum())
            st["changed_out_E_px"] = int((diff & ~E).sum())
            st["E_px"] = int(E.sum())
        if self.geo_lags:
            self._geo_stats(G, step, st)
        if self.shadow:
            # 只在原图格用：额外渲染，统计量进 st["shadow"]，策略输入仍是 img
            img_h, seg_h = self.render_hidden()
            sh = {}
            for name, (cell, v) in self.shadow.items():
                d = {}
                self._draw_new(cell, v, G, step, img_h, seg_h, d)
                sh[name] = d
            st["shadow"] = sh
        return img, st

    def panels(self, obs, step, specs):
        """对照图：同一状态下按 specs（[(名字, 格, var)]）出各格画面；C0/C3/C4 用父类同样的定义现算。"""
        rgb = obs["image"][self.cam]["rgb"]
        G = self.hist[int(step)]
        img_h, seg_h = self.render_hidden()
        P_full = np.isin(seg_h, self.pids)
        out = []
        for name, cell, v in specs:
            if cell == "C0":
                img = rgb
            elif cell == "C3":
                img = img_h
            elif cell == "C4":
                img = img_h.copy()
                img[outline(G, self.w) & ~P_full] = 0
            else:
                img, _ = self._draw_new(cell, v, G, step, img_h, seg_h, {})
            out.append((name, img))
        return out
