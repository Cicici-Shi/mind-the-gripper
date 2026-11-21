"""bench2 第二阶段（CogACT）五格画面。由第一阶段 b2/views.py 复制，画面逻辑（G、涂黑、隐藏、轮廓、保护、滞后）一字未改；
只改了「策略实际输入」的缩放（to_model）：CogACT 不用 open-pi-zero 的 cv2 LANCZOS 224，而是把整张原图交给
DINOv2 与 SigLIP 两个编码器各自的 torchvision Resize((224, 224))（prismatic resize-naive，conf/models.py 第 494 行；
dinosiglip_vit.py 第 91-111 行），插值取自 timm 的数据配置。MODEL_INTERP 写死为两者都是 BICUBIC，
eval_views 每个进程启动时用模型真实的变换核对 to_model 逐位相同，不同就拒跑。
「线还看得见」= 至少一个编码器的 224 输入里看得见（两个编码器的插值相同时两者一样）。
第一阶段的原说明：

五格画面（协议 protocol_bench2.md §2），只改喂给策略的图。

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
from PIL import Image

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
MODEL_SIZE = (224, 224)  # CogACT（prismatic dinosiglip-224px，resize-naive）：两个编码器都缩到 224 x 224，不保宽高比
MODEL_INTERP = {"dino": Image.BICUBIC, "siglip": Image.BICUBIC}  # timm 数据配置的插值；eval_views 启动时与模型真实变换核对
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


def to_model_all(img):
    """CogACT 两个编码器各自看到的 224 图（归一化之前的 uint8）：与 torchvision Resize((224,224)) 作用在 PIL 图上相同。"""
    pil = Image.fromarray(np.ascontiguousarray(img))
    return {n: np.asarray(pil.resize(MODEL_SIZE, resample=r)) for n, r in MODEL_INTERP.items()}


def to_model(img):
    """存图与目视用：DINOv2 那一路的 224 输入（两路插值相同时与 SigLIP 一路逐位相同）。"""
    return to_model_all(img)["dino"]


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
                # 线在策略实际输入里是否看得见：每个编码器的 224 输入各算一次，取看得见像素多的那一路
                ti, th = to_model_all(img), to_model_all(img_h)
                lm_n = {n: int((np.abs(ti[n].astype(np.int16) - th[n].astype(np.int16)).max(axis=2) > LINE_VIS_THR).sum())
                        for n in ti}
                st.update(lag_step=lag_step, O_px=int(O.sum()), O_drawn_px=int(draw.sum()),
                          O_protected_px=int((O & P_full).sum()), P_full_px=int(P_full.sum()),
                          line_px_224=max(lm_n.values()), line_px_224_by_encoder=lm_n,
                          line_px_224_geom=int(mask_to_model(draw).sum()),
                          Gl_px=int(Gl.sum()))
        diff = np.any(img != rgb, axis=2)
        st["changed_px"] = int(diff.sum())
        st["changed_out_E_px"] = int((diff & ~E).sum())
        st["E_px"] = int(E.sum())
        return img, st
