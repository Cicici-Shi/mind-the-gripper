"""生成论文的图（Figure 1 overview、Figure 3 contour shift、Figure 2 body changes）。
Figure 1 is a collage of simulator screenshots that are NOT in this repository; set FIG1_SRC to a folder that holds
figs_maskinfo/..., code/analysis_sync/figs_toy1/... and code/bench2_simpler/b3_smoke/localq/... to rebuild it.
The other figures parse readout text files under results/. Output goes to $FIG_OUT (default ./figs).
图 1 用已有的仿真截图拼成，图 2 直接从平移实验的一次性读数文件里解析数字来画。"""
import io, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.environ.get("FIG1_SRC", "")
OUT = os.environ.get("FIG_OUT", "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

# ---------- 图 1 ----------
if P:  # Figure 1 needs the screenshots (see module docstring)
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

    top = [(mcell(0), "Panda (native)"), (tcell(0), "Robotiq gripper"), (tcell(2), "Robotiq, Panda repainted"),
           (ccell(0), "WidowX, contour"), (ccell(4), "WidowX, shifted contour")]
    bot = [(mcell(0), "Unedited"), (mcell(1), "Solid fill"), (mcell(2), "Erased"), (mcell(3), "Contour"), (mcell(4), "Lagged contour")]
    fig, axes = plt.subplots(2, 5, figsize=(5.5, 2.45))
    for ax, (im, t) in zip(axes[0], top):
        ax.imshow(im); ax.set_title(t, fontsize=6, pad=2); ax.axis("off")
    for ax, (im, t) in zip(axes[1], bot):
        ax.imshow(im); ax.set_title(t, fontsize=6, pad=2); ax.axis("off")
    fig.text(0.005, 0.74, "(a)", fontsize=7, weight="bold"); fig.text(0.005, 0.27, "(b)", fontsize=7, weight="bold")
    fig.subplots_adjust(left=0.035, right=0.999, top=0.94, bottom=0.005, wspace=0.03, hspace=0.16)
    fig.savefig(f"{OUT}/fig1_overview.pdf", dpi=300); fig.savefig(f"{OUT}/fig1_overview.png", dpi=200); plt.close(fig)

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
a1.text(0.02, 0.62, "dashed: unedited image", transform=a1.transAxes, fontsize=5.5, color="0.35")
a1.text(0.02, 0.10, "dotted: gripper erased, no contour", transform=a1.transAxes, fontsize=5.5, color="0.35")
a1.legend(frameon=False, fontsize=6, loc="center right")
a2.set_xlabel("Shift magnitude (input pixels)"); a2.set_ylabel("Hand displacement at closure,\nopposite to the shift (cm)")
a2.set_xticks([6, 12, 24]); a2.set_ylim(0, 4.2); a2.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(f"{OUT}/fig2_shift.pdf"); fig.savefig(f"{OUT}/fig2_shift.png", dpi=200); plt.close(fig)
print("done")

# ---------- 图 2（新）：换手臂、换夹爪的成功率变化，替换原表 2 ----------
# 数值取自报告原表 2，该表已于 2026-11-21 由独立 agent 逐格对照
# results_block2_cg / results_block3_final / results_models1 / results_rescue_pilot / results_physfix 核过。
R = [  # (标签, 组, {模型: (点, 下, 上)})
 ("Arm replaced, Panda gripper kept", "hdr", {}),
 ("Kinova Gen3 (Spatial)", "arm", {"OFT": (-2.0, -5.0, 0.5), "π0.5": (0.0, -2.0, 2.0)}),
 ("Kinova Gen3 (Object)", "arm", {"OFT": (-3.0, -6.5, 0.5), "π0.5": (-1.0, -5.0, 3.0)}),
 ("KUKA IIWA", "arm", {"OFT": (-5.0, -9.0, -1.5), "π0.5": (-1.5, -4.0, 1.0)}),
 ("Panda, one joint locked", "arm", {"OFT": (0.0, -2.0, 2.0), "π0.5": (-2.0, -4.5, 0.0)}),
 ("UR5e", "arm", {"OFT": (-13.0, -18.0, -8.5), "π0.5": (-15.0, -20.0, -10.0)}),
 ("UR5e, forearm collision off", "arm", {"OFT": (-4.0, -7.0, -1.0), "π0.5": (-6.0, -9.5, -3.0)}),
 ("UR5e (Object)", "arm", {"OFT": (0.0, -3.0, 3.0), "π0.5": (0.0, -3.5, 3.5)}),
 ("Arm appearance only", "arm", {"OFT": (-1.0, -3.5, 1.5), "π0.5": (2.5, -1.0, 6.0)}),
 ("Gripper replaced, Panda arm kept", "hdr", {}),
 ("Rethink, extended travel", "grip", {"OFT": (-4.5, -8.5, -1.0), "π0.5": (-32.5, -39.0, -26.0)}),
 ("Robotiq, soft fingers", "grip", {"OFT": (-87.0, -91.5, -82.0), "OFTi": (-83.0, -88.0, -77.5)}),
 ("→ then repaint Panda gripper", "rep", {"OFT": (6.0, -0.5, 13.0), "OFTi": (-5.0, -12.0, 1.5)}),
 ("Robotiq, soft fingers (Object)", "grip", {"OFT": (-7.0, -11.5, -3.0), "OFTi": (-0.5, -4.5, 3.5)}),
 ("→ then repaint Panda gripper", "rep", {"OFT": (5.5, 1.0, 10.5), "OFTi": (-1.5, -5.5, 2.5)}),
 ("Robotiq, stiff fingers*", "grip", {"OFT": (-45.0, -52.0, -38.0), "π0.5": (-17.0, -22.5, -12.0), "OFTi": (-30.0, -36.5, -23.5), "π0": (-29.5, -36.5, -23.0)}),
 ("→ then repaint Panda gripper*", "rep", {"OFT": (36.0, 28.0, 44.0), "π0.5": (11.5, 6.0, 17.5), "OFTi": (22.5, 15.5, 30.0), "π0": (24.5, 16.5, 33.0)}),
 ("→ then repaint wrist view only*", "rep", {"OFT": (32.5, 25.5, 39.5), "π0.5": (0.5, -5.5, 6.0), "OFTi": (15.75, 11.5, 20.0), "π0": (12.5, 5.0, 19.5)}),
 ("→ then repaint third-person only*", "rep", {"OFT": (21.5, 15.0, 28.0), "π0.5": (10.0, 5.0, 15.5), "OFTi": (5.75, 2.0, 9.5), "π0": (15.0, 7.0, 22.5)}),
 ("→ then hide gripper (100 ep.)*", "rep", {"OFT": (-52.0, -62.0, -42.0)}),
 ("Gripper appearance only", "grip", {"OFT": (-20.0, -26.0, -14.0), "π0.5": (1.0, -2.0, 4.0)}),
]
pol = ["OFT", "π0.5", "OFTi", "π0"]
pc = {"OFT": "#0072B2", "π0.5": "#D55E00", "OFTi": "#009E73", "π0": "#CC79A7"}
off = {"OFT": -0.24, "π0.5": -0.08, "OFTi": 0.08, "π0": 0.24}
n = len(R)
fig, ax = plt.subplots(figsize=(5.5, 3.15))
ys = list(range(n))[::-1]
for y, (lab, g, d) in zip(ys, R):
    if g == "rep":
        ax.axhspan(y - 0.5, y + 0.5, color="0.94", lw=0, zorder=0)
    for p, (v, lo, hi) in d.items():
        ax.errorbar(v, y + off[p], xerr=[[v - lo], [hi - v]], fmt="o", ms=2.6, lw=0.8, capsize=1.2, color=pc[p], zorder=3,
                    mfc=("white" if g == "rep" else pc[p]), mec=pc[p])
ax.axvline(0, color="0.3", lw=0.6, zorder=1)
ax.axhline(ys[9] + 0.5, color="0.6", lw=0.5, ls="--")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in R], fontsize=6)
for t, r in zip(ax.get_yticklabels(), R):
    if r[1] == "hdr": t.set_fontweight("bold")
ax.set_xlabel("Change in success rate (percentage points, 95% CI)")
ax.set_xlim(-95, 50); ax.set_ylim(-0.7, n - 0.3)
lab = {"OFT": "OFT", "π0.5": r"$\pi_{0.5}$", "OFTi": "OFTi", "π0": r"$\pi_0$"}
for p in pol:
    ax.plot([], [], "o", ms=3, color=pc[p], label=lab[p])
ax.plot([], [], "o", ms=3, mfc="white", mec="0.3", color="0.3", label="recovery")
ax.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=5, handletextpad=0.2, columnspacing=0.8)
ax.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig2_body.pdf"); fig.savefig(f"{OUT}/fig2_body.png", dpi=200); plt.close(fig)
print("fig2_body done")
