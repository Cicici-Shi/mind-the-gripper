"""把冒烟帧拼成对照图：每行一个（键，查询帧），每列一格（原尺寸缩到一半）。本地运行。"""
import glob, os, sys
from PIL import Image, ImageDraw
root, block, out = sys.argv[1], sys.argv[2], sys.argv[3]
suffix = sys.argv[4] if len(sys.argv) > 4 else "full"
ALL = ["C0", "C1", "C1w", "C1L", "C3", "C4", "C5", "C5L"]  # b2x：只排有帧的格
cells = [c for c in ALL if glob.glob(f"{root}/{block}/{c}/frames/*_{suffix}.png")]
rows = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in glob.glob(f"{root}/{block}/*/frames/*_{suffix}.png")})
ims = {}
for r in rows:
    for c in cells:
        p = f"{root}/{block}/{c}/frames/{r}_{suffix}.png"
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            ims[(r, c)] = im.resize((im.width // 2, im.height // 2)) if suffix == "full" else im
w, h = next(iter(ims.values())).size
sheet = Image.new("RGB", (w * len(cells), (h + 18) * len(rows)), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(rows):
    for j, c in enumerate(cells):
        if (r, c) in ims:
            sheet.paste(ims[(r, c)], (j * w, i * (h + 18) + 18))
        d.text((j * w + 4, i * (h + 18) + 3), f"{c} {r}", fill="black")
sheet.save(out)
print(out, sheet.size)
