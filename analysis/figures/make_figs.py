"""Figures for the paper: Figure 1 (overview teaser), body-swap results (fig2_body), image-edit strip (fig_edits), contour shift (fig2_shift).
fig1_overview and fig_edits are collages of simulator screenshots that are NOT in this repository; set FIG1_SRC to a folder that
holds figs_maskinfo/..., code/analysis_sync/figs_toy1/... and code/bench2_simpler/b3_smoke/localq/... to rebuild them
(they are skipped with a note if FIG1_SRC is unset). fig2_body and fig2_shift use only numbers in this repo: fig2_shift parses the
readout text under results/. Output goes to $FIG_OUT (default ./figs), as .pdf and .png.
生成论文的四张图：图 1 总览和画面编辑示例用已有的仿真截图拼成，换身体结果图和轮廓平移图直接用数字或读数文件画。"""
import io, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.environ.get("FIG1_SRC", "")
OUT = os.environ.get("FIG_OUT", "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "axes.spines.top": False, "axes.spines.right": False})

if P:  # Figure 1 and the image-edit strip need the screenshots (see module docstring)
    # ---------- 图 1 ----------
    manip = Image.open(f"{P}/figs_maskinfo/manip_spatial_t0_e40_h200_w3_C0C1C3C4C5.png").convert("RGB")
    toy = Image.open(f"{P}/code/analysis_sync/figs_toy1/sheet_t0e0_2_pre33.png").convert("RGB")
    cue = Image.open(f"{P}/code/bench2_simpler/b3_smoke/localq/B1/B1__ep00__t9_q005_full.jpg").convert("RGB")

    def mcell(row, col=2):          # manip 拼图：左侧标签列约 60 像素，顶部约 20 像素，每格 256
        x0, y0 = 60 + 256 * col, 20 + 256 * row
        return manip.crop((x0, y0, x0 + 256, y0 + 256))

    def tcell(row):                 # toy1 拼图：每行 16 像素标签 + 256 像素图，第一列是第三人称视角
        y0 = row * 272 + 16
        return toy.crop((0, y0, 256, y0 + 256))

    def ccell(idx):                 # 平移拼图：3x3，每格上方有标签条
        W, H = cue.size
        cw, ch = W / 3, H / 3
        r, c = divmod(idx, 3)
        lab = ch - cw * 480 / 640
        box = (int(c * cw), int(r * ch + lab), int((c + 1) * cw), int((r + 1) * ch))
        im = cue.crop(box)
        w, h = im.size                # 裁成正方形，取夹爪所在的右侧区域
        return im.crop((int(w * 0.25), 0, int(w * 0.25) + h, h))

    def zoom(im, box):
        return im.crop(box).resize((400, 400), Image.LANCZOS)

    def ccell_raw(idx):
        W, H = cue.size
        cw, ch = W / 3, H / 3
        r, c = divmod(idx, 3)
        lab = ch - cw * 480 / 640
        return cue.crop((int(c * cw), int(r * ch + lab), int((c + 1) * cw), int((r + 1) * ch)))

    from matplotlib.patches import FancyBboxPatch
    LZ = (0, 42, 148, 190)
    MZ = (18, 42, 166, 190)
    def wz(idx):
        im = ccell_raw(idx); w_, h_ = im.size
        return im.crop((int(w_ * 0.22), int(h_ * 0.22), int(w_ * 0.22) + int(h_ * 0.76), int(h_ * 0.98))).resize((400, 400), Image.LANCZOS)

    # ---- 图 1：研究总览（首页题图）----
    # 上方一条是研究设定（通用模型改提示词换身体、审查冻结的 VLA、VLA 驱动新身体），下方 (a) 到 (d) 四栏各对应一个结论。
    # 整张图画在一个以英寸为单位的坐标系里；只有这张图用 Arial，其余图的 rcParams 不受影响。
    # 样式从简：黑字、细线、直角截图、(a)(b)(c)(d) 栏标，不用圆角卡片、编号圆点和图标。
    def _fig1():
        import numpy as np
        from PIL import ImageFilter
        from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
        from matplotlib.lines import Line2D
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties

        FONT = "Arial"
        W, H = 5.5, 2.40
        M = 0.012                                           # 四周留一点边，外框描边不被裁掉
        INK, MUTE, RULE, GRID = "#111111", "#555555", "#9A9A9A", "#D9D9D9"
        ARM, GRIP = "#9A84DC", "#E8B004"                    # 只用来标手臂和夹爪，避开四个模型的 Okabe-Ito 色相
        PC = {"OFT": "#0072B2", "π0.5": "#D55E00", "OFTi": "#009E73", "π0": "#CC79A7"}
        FS, FB = 6.2, 7.0                                   # 全图只用两个字号：正文和小标题
        S = 2                                               # 截图先放大 2 倍再合成，边缘更平滑
        DPI_IMG = 400                                       # 嵌入 PDF 时每张截图的分辨率

        # ---------- 版式 ----------
        top = H - M; NH = 0.44; nb = top - NH               # 设定条三个框的上边、高度、下边
        G = 0.12; CW = (W - 2 * M - 3 * G) / 4              # 四栏之间的间隔、栏宽
        cxs = [M + i * (CW + G) for i in range(4)]; ct = nb - 0.15            # 各栏左边、栏的上边
        TW_ = (CW - 0.13) / 2; TH = 0.50                    # 缩略图宽高；两张图之间留 0.13 画箭头
        ASP = TW_ / TH

        # ---------- 截图 ----------
        def up(im):
            return im.resize((im.width * S, im.height * S), Image.LANCZOS)
        seg = np.asarray(mcell(5, 0)).astype(int)
        def mask(m):                                        # 放大并磨平锯齿，返回 0 到 1 的软掩码
            a = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((256 * S, 256 * S), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.3))).astype(float) / 255
            return np.clip((a - 0.5) * 5 + 0.5, 0, 1)[..., None]
        arm = (seg[..., 2] > seg[..., 0] + 50) & (seg[..., 2] > seg[..., 1] + 50)
        arm[108:, 125:] = False                             # 灶台左沿和下沿各有一条被标成手臂的细线，去掉
        m_arm = mask(arm)
        m_grip = mask((seg[..., 0] > seg[..., 1] + 60) & (seg[..., 0] > seg[..., 2] + 60))
        rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
        def tint(a, m, color, k):
            return a * (1 - k * m) + rgb(color) * (k * m)
        panda = np.asarray(up(mcell(0, 0))).astype(float)
        BOX = (0, 112 - 83 / ASP, 166, 112 + 83 / ASP)      # 夹爪周围的裁剪框（原图像素）
        def crop(a, box=BOX):
            x0, y0, x1, y1 = [round(v * S) for v in box]
            return np.clip(a[y0:y1, x0:x1], 0, 255).astype(np.uint8)
        gray = panda.mean(axis=2, keepdims=True)
        flat = 0.22 * gray + 0.78 * 232                     # 压平明暗，只留轮廓感
        grow = lambda m, r: np.asarray(Image.fromarray((m[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(r)).filter(ImageFilter.GaussianBlur(0.8))).astype(float)[..., None] / 255
        edge = np.clip(grow(m_arm, 2 * S + 1) - m_arm, 0, 1)                    # 手臂轮廓外沿的一圈描边，灰度打印时也看得出形状
        newarm = tint(panda * (1 - m_arm) + flat * m_arm, m_arm, ARM, 0.50)
        im_newarm = crop(newarm * (1 - edge) + rgb("#4A3399") * edge, (0, 0, 212, 212 / ASP))
        im_arm0 = crop(panda, (0, 0, 212, 212 / ASP))
        im_hero = crop(tint(tint(panda, m_arm, ARM, 0.55), m_grip, GRIP, 0.70), (0, 0, 200, 200))
        im_panda, im_erased, im_fill = crop(panda), crop(np.asarray(up(mcell(2, 0))).astype(float)), crop(np.asarray(up(mcell(1, 0))).astype(float))
        im_robotiq, im_repaint = crop(np.asarray(up(tcell(0))).astype(float)), crop(np.asarray(up(tcell(2))).astype(float))

        with plt.rc_context({"font.family": FONT}):
            fig = plt.figure(figsize=(W, H))
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

            def T(x, y, s, size=FS, color=INK, weight="normal", ha="left", va="baseline", z=5, **kw):
                return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, **kw)
            def tw(s, size=FS, weight="normal"):
                return TextPath((0, 0), s, size=size, prop=FontProperties(family=FONT, weight=weight)).get_extents().x1 / 72
            def line(xs, ys, color=RULE, lw=0.5, z=2, **kw):
                ax.add_line(Line2D(xs, ys, lw=lw, color=color, zorder=z, **kw))
            def img(a, x, y, w, h, z=2):                       # 直角、细边框的截图
                a = np.asarray(Image.fromarray(a).resize((round(w * DPI_IMG), round(h * DPI_IMG)), Image.LANCZOS))
                ax.imshow(a, extent=(x, x + w, y, y + h), zorder=z, interpolation="none", aspect="auto")
                ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#6E6E6E", lw=0.4, zorder=z + 0.2))
            def arrow(x0, y0, x1, y1, color=INK, lw=0.6, z=4):
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.36,head_width=0.15", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=8))
            def pol(x, y, name, color=INK):                    # 带下标的模型名，返回右端 x
                base, sub = (name[0], name[1:]) if name.startswith("π") else (name, "")
                T(x, y, base, color=color)
                x += tw(base) + 0.004
                if sub:
                    T(x, y - FS * 0.24 / 72, sub, size=FS * 0.78, color=color); x += tw(sub, FS * 0.78)
                return x

            # ================= 上方：设定 =================
            ty = [top - 0.130, top - 0.252, top - 0.364]        # 三行文字的基线
            cy = nb + NH / 2
            def node(x, w, dashed=False):
                ax.add_patch(FancyBboxPatch((x + 0.03, nb + 0.03), w - 0.06, NH - 0.06, boxstyle="round,pad=0.03,rounding_size=0.03", fc="none", ec=INK,
                                            lw=0.6, ls=((0, (3.5, 2.5)) if dashed else "-"), zorder=1))
            wa, wc, we = 1.42, 1.38, 1.40
            gap = (W - 2 * M - wa - wc - we) / 2
            xa = M; xc = xa + wa + gap; xe = xc + wc + gap
            # 通用模型：虚线框，本文没有运行它
            node(xa, wa, dashed=True)
            T(xa + 0.08, ty[0], "General-purpose model", size=FB, weight="bold")
            T(xa + 0.08, ty[1], "e.g. GPT-6 Astra (not run here)", color=MUTE)
            T(xa + 0.08, ty[2], "re-prompted for the new body")
            # 冻结的 VLA，第三行是四个模型的颜色图例
            node(xc, wc)
            T(xc + 0.08, ty[0], "Frozen VLA policy", size=FB, weight="bold")
            T(xc + 0.08, ty[1], "fast, low-level, not retrained")
            x = xc + 0.08
            for n in ("OFT", "π0.5", "OFTi", "π0"):
                ax.add_patch(Rectangle((x, ty[2] + 0.004), 0.05, 0.05, fc=PC[n], ec="none", zorder=4))
                x = pol(x + 0.072, ty[2], n) + 0.075
            # 机器人身体：分割掩码把手臂和夹爪染上色
            node(xe, we)
            ih = NH - 0.09
            img(im_hero, xe + 0.045, nb + 0.045, ih, ih)
            tx = xe + 0.045 + ih + 0.07
            T(tx, ty[0], "New robot body", size=FB, weight="bold")
            T(tx, ty[1], "we change one part:")
            x = tx
            for word, c in (("arm", ARM), ("gripper", GRIP)):
                ax.add_patch(Rectangle((x, ty[2] + 0.004), 0.05, 0.05, fc=c, ec="none", zorder=4))
                T(x + 0.072, ty[2], word); x += 0.072 + tw(word) + 0.05
                if word == "arm":
                    T(x, ty[2], "or"); x += tw("or") + 0.05
            # 连接箭头
            arrow(xa + wa + 0.04, cy, xc - 0.04, cy)
            T((xa + wa + xc) / 2, cy + 0.04, "supervises", ha="center")
            arrow(xc + wc + 0.04, cy + 0.05, xe - 0.04, cy + 0.05)
            arrow(xe - 0.04, cy - 0.05, xc + wc + 0.04, cy - 0.05)
            T((xc + wc + xe) / 2, cy + 0.09, "actions", ha="center")
            T((xc + wc + xe) / 2, cy - 0.17, "images", ha="center")

            # ================= 下方：四栏 =================
            # 分组标题：一条细线，中间断开写字
            gb = ct + 0.045
            for (i0, i1, lab) in ((0, 1, "Replace one body part"), (2, 3, "Edit the gripper in the image")):
                xl, xr = cxs[i0], cxs[i1] + CW; xm = (xl + xr) / 2; hw = tw(lab, FS) / 2 + 0.06
                line([xl, xm - hw], [gb + 0.024] * 2); line([xm + hw, xr], [gb + 0.024] * 2)
                T(xm, gb, lab, ha="center")
            for i in range(3):                                    # 栏间竖线
                xv = cxs[i] + CW + G / 2
                line([xv, xv], [M + 0.02, ct - 0.03], color=GRID)

            def panel(i, title, cond, thumbs, link=True):
                x = cxs[i]
                T(x, ct - 0.105, "(%s) %s" % ("abcd"[i], title), size=FB, weight="bold")
                T(x, ct - 0.215, cond, color=MUTE)
                y_t = ct - 0.255 - TH
                for k, (a, lab) in enumerate(thumbs):
                    tx_ = x + k * (TW_ + 0.13)
                    img(a, tx_, y_t, TW_, TH)
                    T(tx_ + TW_ / 2, y_t - 0.092, lab, ha="center")
                if link:
                    arrow(x + TW_ + 0.025, y_t + TH / 2, x + TW_ + 0.105, y_t + TH / 2)
                return x, x + CW, y_t - 0.150                     # 图表区域的左、右、上边
            def verdict(i, head, lines):
                x = cxs[i]
                T(x, M + 0.268, head, size=FB, weight="bold")
                for k, s in enumerate(lines):
                    T(x, M + 0.158 - 0.102 * k, s)
            def header(xl, yt, title, ticks, fx):                 # 图表表头：左侧轴名，右侧刻度；返回第一行的上边
                T(xl, yt - 0.075, title, color=MUTE)
                for v, s in ticks:
                    T(fx(v), yt - 0.075, s, color=MUTE, ha="center")
                return yt - 0.105
            RP = 0.094                                            # 行距

            # ---- (a)(b)：换手臂、换夹爪。横轴是成功率变化（百分点），0 在右，两栏同一把尺 ----
            # 数值与下面图 2 的 R 表一致（OFT、π0.5 在 LIBERO-Spatial 上的成功率变化）
            sc = 0.0128
            def loss_axis(xl, xr, yt):
                fx = lambda v: xr - 0.012 + v * sc
                y0 = header(xl, yt, "Δ success (points)", [(-20, "−20"), (0, "0")], fx)
                line([fx(-20)] * 2, [y0 + 0.005, y0 - RP * 4], color=GRID, lw=0.45)
                line([fx(0)] * 2, [y0 + 0.005, y0 - RP * 4], color=INK, lw=0.5, z=3)
                return fx, y0
            def bar(fx, v, y, c, h, v0=0.0, hollow=False):
                ax.add_patch(Rectangle((fx(v), y - h / 2), fx(v0) - fx(v), h, fc=("white" if hollow else c), ec=(c if hollow else "none"), lw=0.55, zorder=4))

            xl, xr, yt = panel(0, "Replace the arm", "same end-effector interface", [(im_arm0, "Panda"), (im_newarm, "new arm")])
            fx, y0 = loss_axis(xl, xr, yt)
            for r, (lab, a, b) in enumerate([("Kinova Gen3", -2.0, 0.0), ("KUKA IIWA", -5.0, -1.5), ("Panda, one joint locked", 0.0, -2.0), ("UR5e", -13.0, -15.0)]):
                yc = y0 - RP * (r + 0.5)
                T(xl, yc - 0.022, lab)
                for v, c, dy in ((a, PC["OFT"], 0.0195), (b, PC["π0.5"], -0.0195)):
                    if lab == "UR5e":                             # 实心：关掉前臂碰撞后仍有的损失；空心：前臂碰撞带来的那部分
                        nc = -4.0 if c == PC["OFT"] else -6.0
                        bar(fx, v, yc + dy, c, 0.030, v0=nc, hollow=True); bar(fx, nc, yc + dy, c, 0.034)
                    else:
                        bar(fx, min(v, -0.5), yc + dy, c, 0.034)  # 0.0 也留一条细线，表示有数据
                if lab == "UR5e":
                    xe_ = fx(-15) - 0.105
                    T(xe_, yc - 0.022, "forearm collision", color=MUTE, ha="right")
                    arrow(xe_ + 0.02, yc, fx(-15) - 0.018, yc, color=MUTE, lw=0.5)
            verdict(0, "Little loss", ["at most 5 points on three arms;", "UR5e: mostly forearm collision"])

            xl, xr, yt = panel(1, "Replace the gripper", "same Panda arm", [(im_panda, "Panda"), (im_robotiq, "Robotiq")])
            fx, y0 = loss_axis(xl, xr, yt)
            for r, (lab, a, b) in enumerate([("Rethink", -4.5, -32.5), ("Robotiq", -45.0, -17.0)]):
                yc = y0 - RP * 2 * (r + 0.5)
                T(xl, yc - 0.022, lab)
                for v, c, dy in ((a, PC["OFT"], 0.036), (b, PC["π0.5"], -0.036)):
                    bar(fx, v, yc + dy, c, 0.052)
                    T(fx(v) - 0.025, yc + dy - 0.022, f"−{abs(v):.1f}", ha="right", bbox=dict(fc="white", ec="none", pad=0.5))
            verdict(1, "Larger loss", ["which gripper works better", "depends on the policy"])

            # ---- (c)：抹掉 / 填实。每条线是一个模型-任务组合（表 2：未编辑、抹掉、黑色填实）----
            xl, xr, yt = panel(2, "Erase the gripper", "native body, image edit only", [(im_erased, "erased"), (im_fill, "solid fill")], link=False)
            x0p = xl + 0.55; s3 = (xr - 0.07 - x0p) / 100; fx = lambda v: x0p + v * s3
            RP3 = RP * 4 / 3
            y0 = header(xl, yt, "success (%)", [(0, "0"), (50, "50"), (100, "100")], fx)
            for v in (0, 50, 100):
                line([fx(v)] * 2, [y0 + 0.005, y0 - RP3 * 3], color=GRID, lw=0.45)
            t2 = [(96, 41, 97), (100, 62, 98), (98, 17, 99), (97, 70, 98), (99, 59, 99), (99, 0, 99), (97, 8, 97), (97, 67, 98),
                  (83.3, 5.0, 80.0), (87.5, 0.8, 90.0), (98.3, 6.7, 60.0), (75.8, 5.0, 45.0), (69.2, 0.0, 65.8), (69.2, 1.7, 64.2), (96.7, 1.7, 75.8), (90.0, 15.8, 59.2)]
            ys = [y0 - RP3 * (k + 0.5) for k in range(3)]
            for u, e, f in t2:
                line([fx(u), fx(e), fx(f)], ys, color="#333333", lw=0.5, z=3, alpha=0.36, solid_joinstyle="round")
            for k, lab in enumerate(("unedited", "erased", "solid fill")):
                T(xl, ys[k] - 0.022, lab)
                for row in t2:
                    ax.add_patch(Circle((fx(row[k]), ys[k]), 0.0135, fc="#333333", ec="none", alpha=0.55, zorder=4))
            verdict(2, "Erasing lowers success", ["all 16 policy-task combinations;", "a solid fill restores most"])

            # ---- (d)：画回原夹爪。空心：换上硬指 Robotiq；实心：画回 Panda 夹爪；竖线和右列：原身体（4.1 节）----
            xl, xr, yt = panel(3, "Repaint the gripper", "original drawn over the new one", [(im_robotiq, "○ Robotiq"), (im_repaint, "● repainted")])
            x0p = xl + 0.34; s4 = (xr - 0.27 - x0p) / 50; fx = lambda v: x0p + (v - 50) * s4
            y0 = header(xl, yt, "success (%)", [(70, "70"), (90, "90")], fx)
            T(xr, yt - 0.075, "native", color=MUTE, ha="right")
            for v in (50, 70, 90):
                line([fx(v)] * 2, [y0 + 0.005, y0 - RP * 4], color=GRID, lw=0.45)
            for r, (n, a, b, c) in enumerate([("OFT", 53.5, 89.5, 98.5), ("π0.5", 82.0, 90.0, 99.0), ("OFTi", 69.2, 90.0, 98.0), ("π0", 67.0, 90.5, 96.5)]):
                yc = y0 - RP * (r + 0.5)
                pol(xl, yc - 0.022, n)
                line([fx(a) + 0.02, fx(b) - 0.02], [yc, yc], color=PC[n], lw=1.1, z=3, solid_capstyle="butt")
                ax.add_patch(Circle((fx(a), yc), 0.021, fc="white", ec=PC[n], lw=1.0, zorder=4))
                ax.add_patch(Circle((fx(b), yc), 0.025, fc=PC[n], ec="none", zorder=4))
                line([fx(c)] * 2, [yc - 0.03, yc + 0.03], color=PC[n], lw=1.3, z=4)
                T(xr, yc - 0.022, f"{c:.1f}", ha="right")
            verdict(3, "Rises to about 90%", ["from 53.5–82.0%, if the new", "gripper can hold the object"])

            ax.set_xlim(0, W); ax.set_ylim(0, H)
            fig.savefig(f"{OUT}/fig1_overview.pdf", dpi=300); fig.savefig(f"{OUT}/fig1_overview.png", dpi=300); plt.close(fig)

    _fig1()

    # ---- 画面编辑示例（4.2 节用）----
    cells = [(zoom(mcell(0), MZ), "Unedited"), (zoom(mcell(1), MZ), "Solid fill"), (zoom(mcell(2), MZ), "Erased"),
             (zoom(mcell(3), MZ), "Contour"), (zoom(mcell(4), MZ), "Lagged contour"), (wz(0), "WidowX contour"), (wz(4), "Shifted contour")]
    fig, axs = plt.subplots(1, 7, figsize=(5.5, 0.98))
    for a2_, (im, t) in zip(axs, cells):
        a2_.imshow(im); a2_.axis("off"); a2_.set_title(t, fontsize=5.8, pad=2)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.85, bottom=0.01, wspace=0.04)
    fig.savefig(f"{OUT}/fig_edits.pdf", dpi=300); fig.savefig(f"{OUT}/fig_edits.png", dpi=200); plt.close(fig)
else:
    print("FIG1_SRC not set: skipping fig1_overview and fig_edits (screenshots are not in the repo)")

# ---------- 图 2：从读数文件解析 ----------
txt = io.open(f"{REPO}/results/4.3_contour_shift/READOUT_cue_widowx_v3_1_2026-11-15.txt", encoding="utf-8").read()
blocks = re.split(r"^======== (B[12])", txt, flags=re.M)
data = {}
for k in range(1, len(blocks), 2):
    name, body = blocks[k], blocks[k + 1]
    sr_line = re.search(r"SR（最后一步）：(.*)", body).group(1)
    sr = {m.group(1): float(m.group(2)) for m in re.finditer(r"(C0|C3|C4|S[-+][hab]|SC[-+]b|OB[-+]b) ([\d.]+)", sr_line)}
    oh = {}
    for d in "hab":
        line = re.search(r"^K %s 档：.*$" % d, body, flags=re.M).group(0)
        m = re.search(r"O_H (-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", line)
        oh[d] = tuple(float(x) for x in m.groups())
    data[name] = (sr, oh)
    print(name, sr, oh)

px = {"h": 6, "a": 12, "b": 24}
names = {"B1": "Eggplant", "B2": "Spoon"}
col = {"B1": "#0072B2", "B2": "#D55E00"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 1.95))
for b in ("B1", "B2"):
    sr, oh = data[b]
    xs = [-24, -12, -6, 0, 6, 12, 24]
    ys = [sr["S-b"], sr["S-a"], sr["S-h"], sr["C4"], sr["S+h"], sr["S+a"], sr["S+b"]]
    a1.plot(xs, ys, "-o", ms=3, lw=1, color=col[b], label=names[b])
    a1.axhline(sr["C3"], color=col[b], lw=0.6, ls=":")
    a1.axhline(sr["C0"], color=col[b], lw=0.6, ls="--")
    d = ["h", "a", "b"]
    y = [-oh[k][0] for k in d]
    lo = [(-oh[k][0]) - (-oh[k][2]) for k in d]; hi = [(-oh[k][1]) - (-oh[k][0]) for k in d]
    a2.errorbar([px[k] for k in d], y, yerr=[lo, hi], fmt="-o", ms=3, lw=1, capsize=2, color=col[b], label=names[b])
a1.set_xlabel("Horizontal shift of the drawn contour (input pixels)"); a1.set_ylabel("Success rate (%)")
a1.set_xticks([-24, -12, -6, 0, 6, 12, 24]); a1.set_ylim(0, 92)
a1.plot([], [], ls="--", lw=0.6, color="0.4", label="unedited")
a1.plot([], [], ls=":", lw=0.6, color="0.4", label="erased, no contour")
a1.legend(frameon=False, fontsize=6.3, loc="upper right", bbox_to_anchor=(1.0, 0.88), ncol=2, columnspacing=0.8, handlelength=1.6)
a2.set_xlabel("Shift magnitude (input pixels)"); a2.set_ylabel("Hand displacement at closure,\nopposite to the shift (cm)")
a2.set_xticks([6, 12, 24]); a2.set_ylim(0, 4.2); a2.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(f"{OUT}/fig2_shift.pdf"); fig.savefig(f"{OUT}/fig2_shift.png", dpi=200); plt.close(fig)
print("done")

# ---------- 图 2（新）：换手臂、换夹爪的成功率变化，替换原表 2 ----------
# 数值取自报告原表 2，该表已于 2026-11-21 由独立 agent 逐格对照
# results_block2_cg / results_block3_final / results_models1 / results_rescue_pilot / results_physfix 核过。
R = [  # (标签, 组, {模型: (点, 下, 上)})
 ("Arm replaced, Panda gripper kept", "hdr", {}),
 ("Kinova Gen3", "arm", {"OFT": (-2.0, -5.0, 0.5), "π0.5": (0.0, -2.0, 2.0)}),
 ("KUKA IIWA", "arm", {"OFT": (-5.0, -9.0, -1.5), "π0.5": (-1.5, -4.0, 1.0)}),
 ("Panda, one joint locked", "arm", {"OFT": (0.0, -2.0, 2.0), "π0.5": (-2.0, -4.5, 0.0)}),
 ("UR5e", "arm", {"OFT": (-13.0, -18.0, -8.5), "π0.5": (-15.0, -20.0, -10.0)}),
 ("UR5e, forearm collision off", "arm", {"OFT": (-4.0, -7.0, -1.0), "π0.5": (-6.0, -9.5, -3.0)}),
 ("Arm appearance only", "arm", {"OFT": (-1.0, -3.5, 1.5), "π0.5": (2.5, -1.0, 6.0)}),
 ("Gripper replaced, Panda arm kept", "hdr", {}),
 ("Rethink, extended travel", "grip", {"OFT": (-4.5, -8.5, -1.0), "π0.5": (-32.5, -39.0, -26.0)}),
 ("Robotiq, soft fingers", "grip", {"OFT": (-87.0, -91.5, -82.0), "OFTi": (-83.0, -88.0, -77.5)}),
 ("→ then repaint Panda gripper", "rep", {"OFT": (6.0, -0.5, 13.0), "OFTi": (-5.0, -12.0, 1.5)}),
 ("Robotiq, stiff fingers*", "grip", {"OFT": (-45.0, -52.0, -38.0), "π0.5": (-17.0, -22.5, -12.0), "OFTi": (-30.0, -36.5, -23.5), "π0": (-29.5, -36.5, -23.0)}),
 ("→ then repaint Panda gripper*", "rep", {"OFT": (36.0, 28.0, 44.0), "π0.5": (11.5, 6.0, 17.5), "OFTi": (22.5, 15.5, 30.0), "π0": (24.5, 16.5, 33.0)}),
 ("Gripper appearance only", "grip", {"OFT": (-20.0, -26.0, -14.0), "π0.5": (1.0, -2.0, 4.0)}),
]
pol = ["OFT", "π0.5", "OFTi", "π0"]
pc = {"OFT": "#0072B2", "π0.5": "#D55E00", "OFTi": "#009E73", "π0": "#CC79A7"}
mk = {"OFT": "o", "π0.5": "s", "OFTi": "^", "π0": "D"}
off = {"OFT": -0.27, "π0.5": -0.09, "OFTi": 0.09, "π0": 0.27}
hi_ = [r[1] for r in R].index("hdr", 1)
parts = [(R[1:hi_], "Arm replaced, Panda gripper kept", (-22, 8)), (R[hi_ + 1:], "Gripper replaced, Panda arm kept", (-95, 50))]
fig, axs = plt.subplots(2, 1, figsize=(5.5, 3.0), gridspec_kw={"height_ratios": [len(parts[0][0]), len(parts[1][0])], "hspace": 0.45})
for ax, (rows, title, xl) in zip(axs, parts):
    n = len(rows); ys = list(range(n))[::-1]
    for y, (lab, g, d) in zip(ys, rows):
        if g == "rep":
            ax.axhspan(y - 0.5, y + 0.5, color="0.94", lw=0, zorder=0)
        for p, (v, lo, hi) in d.items():
            ax.errorbar(v, y + off[p], xerr=[[v - lo], [hi - v]], fmt=mk[p], ms=3.2, lw=0.9, capsize=1.4, color=pc[p], zorder=3,
                        mfc=("white" if g == "rep" else pc[p]), mec=pc[p])
    ax.axvline(0, color="0.3", lw=0.6, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(*xl); ax.set_ylim(-0.6, n - 0.4); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title(title, fontsize=7, weight="bold", loc="left", pad=3)
axs[1].set_xlabel("Change in success rate (percentage points, 95% CI)", fontsize=7)
lab = {"OFT": "OFT", "π0.5": r"$\pi_{0.5}$", "OFTi": "OFTi", "π0": r"$\pi_0$"}
for p in pol:
    axs[0].plot([], [], mk[p], ms=3.5, color=pc[p], label=lab[p])
axs[0].plot([], [], "o", ms=3.5, mfc="white", mec="0.3", color="0.3", label="recovery (open)")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=5, handletextpad=0.2, columnspacing=0.9)
fig.subplots_adjust(left=0.30, right=0.99, top=0.86, bottom=0.12)
fig.savefig(f"{OUT}/fig2_body.pdf"); fig.savefig(f"{OUT}/fig2_body.png", dpi=200); plt.close(fig)
print("fig2_body done")
