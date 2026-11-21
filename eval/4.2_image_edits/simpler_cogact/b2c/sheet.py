"""把冒烟帧拼成对照图（第二阶段，由第一阶段 b2/sheet.py 改写）：每行一个（键，查询帧），每列一格，用 CogACT 224 输入尺度的图。
帧文件名 <块>_<格>_<键>_q<步>_224.png（eval_views --frames-dir）。用法：python sheet.py <帧目录> <块> <输出 png> [224|full]"""
import glob
import os
import re
import sys

from PIL import Image, ImageDraw

fdir, block, out = sys.argv[1], sys.argv[2], sys.argv[3]
suffix = sys.argv[4] if len(sys.argv) > 4 else "224"
cells = ["C0", "C1", "C3", "C4", "C5"]
pat = re.compile(rf"^{block}_(C\d)_(.+)_(q\d+)_{suffix}\.png$")
ims, rows = {}, set()
for p in glob.glob(os.path.join(fdir, f"{block}_*_{suffix}.png")):
    m = pat.match(os.path.basename(p))
    if not m:
        continue
    c, k, q = m.groups()
    im = Image.open(p).convert("RGB")
    if suffix == "full":
        im = im.resize((im.width // 3, im.height // 3))
    ims[(k + " " + q, c)] = im
    rows.add(k + " " + q)
rows = sorted(rows)
w, h = next(iter(ims.values())).size
sheet = Image.new("RGB", (w * len(cells), (h + 14) * len(rows)), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(rows):
    for j, c in enumerate(cells):
        if (r, c) in ims:
            sheet.paste(ims[(r, c)], (j * w, i * (h + 14) + 14))
        d.text((j * w + 2, i * (h + 14) + 1), f"{c} {r.replace(block + '__', '')}", fill="black")
sheet.save(out, optimize=True)
print(out, sheet.size)
