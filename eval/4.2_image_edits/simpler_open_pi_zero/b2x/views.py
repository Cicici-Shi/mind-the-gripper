"""bench2 画面（协议 protocol_bench2.md §2；b2x 加补充协议 protocol_bench2_ext_fill_lag.md v1 的新格），只改喂给策略的图。

b2x 新增（都是新分支或只在参数打开时才算的统计；原有五格的取图路径一字未改，重跑的旧格要与第一阶段记录逐位相同）：
  C1w 白色填充、C1L 浅色填充：区域与 C1 完全相同（当步可见时分割图里属于 G 的像素），填 fill_rgb（参数，来自冻结清单）
  C5L 擦掉后画更长滞后的轮廓：同 C5，滞后用 lag_long（参数，冻结清单按规则定）；t < lag_long 时取第 0 步
  可选统计（只进 qstats 或回合级字段，不改图）：
    color_stats：整回合 G 像素、G 外一圈（RING_PX 像素宽）像素的颜色计数（求浅色 FL、操纵强度、环反差用）
    fill_stats ：C1 / C1w / C1L 每帧的操纵强度与环反差的分子分母（对本格实际的填充色）
    geom_lags  ：对每个滞后 l，按策略实际看到的那条轮廓（t < l 时为第 0 步的轮廓）算与当前轮廓的
                 IoU、质心位移（224 尺度像素）、可见像素比，并标「未满滞后」（t < l）或「已满滞后」

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
# 补充协议的新格（与上面三个元组分开，旧格的判断不受影响）
FILL_CELLS = ("C1w", "C1L")
LONG_LAG_CELLS = ("C5L",)
ALL_CELLS = CELLS + FILL_CELLS + LONG_LAG_CELLS
ALL_HIDE_CELLS = HIDE_CELLS + LONG_LAG_CELLS
ALL_OUTLINE_CELLS = OUTLINE_CELLS + LONG_LAG_CELLS
FILLED_CELLS = ("C1",) + FILL_CELLS  # 区域都是 G 的平涂格
RING_PX = 6  # 环反差：G 掩膜向外 6 像素的一圈（方形膨胀，不含 G；膨胀不出画面，所以不含画面外）

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
def pack_rgb(px):
    px = np.asarray(px, np.int64).reshape(-1, 3)
    return (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]


def ring_of(G, r=RING_PX):
    return _dilate_sq(G, r) & ~np.asarray(G, bool)


def absdiff_sum(px, rgb):
    """sum over 像素 of mean_ch |px - rgb|（操纵强度与环反差共用的逐像素量）。"""
    if len(px) == 0:
        return 0.0
    return float(np.abs(px.astype(np.int32) - np.asarray(rgb, np.int32)).mean(axis=1).sum())


class ViewMaker:
    def __init__(self, env, cell, body, lag_steps, line_width, query_every, protect_names, protect_n,
                 fill_rgb=None, lag_long=None, color_stats=False, fill_stats=False, geom_lags=()):
        assert cell in ALL_CELLS, cell
        assert lag_steps > 0 and lag_steps % query_every == 0, (lag_steps, query_every)
        # 补充协议的参数（旧格的取图不用它们）
        self.fill_rgb = None if fill_rgb is None else tuple(int(x) for x in fill_rgb)
        if cell in FILL_CELLS:
            assert self.fill_rgb is not None and len(self.fill_rgb) == 3 and all(0 <= x <= 255 for x in self.fill_rgb)
        self.L_long = None if lag_long is None else int(lag_long)
        if cell in LONG_LAG_CELLS:
            assert self.L_long is not None and self.L_long > 0 and self.L_long % query_every == 0, (lag_long, query_every)
        self.color_stats, self.fill_stats = bool(color_stats), bool(fill_stats)
        self.geom_lags = tuple(int(x) for x in geom_lags)
        assert all(l > 0 and l % query_every == 0 for l in self.geom_lags), self.geom_lags
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
        self.g_colors, self.ring_colors = {}, {}  # 整回合颜色计数（color_stats）：packed rgb -> 像素数

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

    def color_counts(self):
        """回合级：(G 颜色计数, 环颜色计数)，各为 N x 4 的 int64 数组（r, g, b, 像素数）。"""
        def arr(d):
            if not d:
                return np.zeros((0, 4), np.int64)
            k = np.array(sorted(d), np.int64)
            return np.stack([k >> 16, (k >> 8) & 255, k & 255, np.array([d[x] for x in k.tolist()], np.int64)], 1)
        return arr(self.g_colors), arr(self.ring_colors)

    def _geom(self, G, seg, step):
        """对每个滞后 l：显示的滞后帧 = max(t - l, 0)（与 C5 / C5L 的画法相同），算
        iou（全分辨率线像素）、centroid_shift_224（两条轮廓线像素质心的距离，按 224x224 输入换算）、
        visible_ratio_224（滞后轮廓可见像素 / 当前轮廓可见像素；可见 = 去掉当步可见分割里受保护物体像素后，
        按面积缩放到 224 覆盖率 >= 0.5），phase = pre（t < l，未满滞后）或 full（已满滞后）。"""
        H, W = G.shape
        sy, sx = MODEL_SIZE[1] / H, MODEL_SIZE[0] / W
        P = np.isin(seg, self.pids)
        Oc = outline(G, self.w)
        vis_c = int(mask_to_model(Oc & ~P).sum())
        out = {}
        for l in self.geom_lags:
            ls = max(int(step) - l, 0)
            Ol = outline(self.hist[ls], self.w)
            d = {"phase": "pre" if step < l else "full", "lag_step": ls, "iou": iou(Oc, Ol), "vis_cur_224": vis_c,
                 "vis_lag_224": int(mask_to_model(Ol & ~P).sum())}
            if Oc.any() and Ol.any():
                yc, xc = np.nonzero(Oc)
                yl, xl = np.nonzero(Ol)
                d["centroid_shift_224"] = float(np.hypot((xc.mean() - xl.mean()) * sx, (yc.mean() - yl.mean()) * sy))
            else:
                d["centroid_shift_224"] = None
            d["visible_ratio_224"] = None if vis_c == 0 else d["vis_lag_224"] / vis_c
            out[str(l)] = d
        return out

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
        if self.color_stats:  # 补充协议 §2、§3：整回合 G 与环的颜色计数（原图像素）
            ring = ring_of(G)
            for store, px in ((self.g_colors, rgb[G]), (self.ring_colors, rgb[ring])):
                u, c = np.unique(pack_rgb(px), return_counts=True)
                for a, b in zip(u.tolist(), c.tolist()):
                    store[a] = store.get(a, 0) + b
        if self.geom_lags:  # 补充协议 §2：按策略实际看到的那条轮廓算的滞后几何量
            st["geom"] = self._geom(G, seg, step)
        E = np.zeros_like(G)
        if self.cell == "C0":
            img = rgb
        elif self.cell == "C1":
            img = rgb.copy()
            img[G] = 0
            E = G
        elif self.cell in FILL_CELLS:  # 补充协议：区域与 C1 完全相同，只换填充颜色
            img = rgb.copy()
            img[G] = np.array(self.fill_rgb, np.uint8)
            E = G
            st["fill_rgb"] = list(self.fill_rgb)
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
            if self.cell in ALL_OUTLINE_CELLS:
                if self.cell == "C4":
                    Gl, lag_step = G, int(step)
                elif self.cell == "C5":
                    lag_step = max(int(step) - self.L, 0)
                    Gl = self.hist[lag_step]
                    assert lag_step == (int(step) - self.L if step >= self.L else 0)
                else:  # C5L：同 C5，滞后用 L_long
                    lag_step = max(int(step) - self.L_long, 0)
                    Gl = self.hist[lag_step]
                    assert lag_step == (int(step) - self.L_long if step >= self.L_long else 0)
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
        if self.fill_stats and self.cell in FILLED_CELLS:  # 补充协议 §2：对本格实际填充色的强度与环反差（分子、分母）
            frgb = (0, 0, 0) if self.cell == "C1" else self.fill_rgb
            ring = ring_of(G)
            st.update(fill_strength_sum=absdiff_sum(rgb[G], frgb), ring_contrast_sum=absdiff_sum(rgb[ring], frgb),
                      ring_px=int(ring.sum()))
        diff = np.any(img != rgb, axis=2)
        st["changed_px"] = int(diff.sum())
        st["changed_out_E_px"] = int((diff & ~E).sum())
        st["E_px"] = int(E.sum())
        return img, st
