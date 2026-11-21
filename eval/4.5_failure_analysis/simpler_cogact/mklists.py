"""生成校验清单（每块每格第 0 和第 60 个键）与全量清单（2400 条分 16 份）。只读 out_formal 的文件名。"""
import os, sys
REC = os.path.expanduser("$PROJ_ROOT/bench2_cogact/b2c/out_formal")
allrows = []
for b in ("B1", "B2", "F1", "F2"):
    val = []
    for c in ("C0", "C1", "C3", "C4", "C5"):
        keys = sorted(f[:-4] for f in os.listdir(os.path.join(REC, b, c)) if f.endswith(".npz") and not f.startswith("."))
        assert len(keys) == 120, (b, c, len(keys))
        val += [f"{b} {c} {keys[0]}", f"{b} {c} {keys[60]}"]
        allrows += [f"{b} {c} {k}" for k in keys]
    open(f"val_{b}.txt", "w").write("\n".join(val) + "\n")
assert len(allrows) == 2400
for i in range(16):
    open(f"full_{i:02d}.txt", "w").write("\n".join(allrows[i::16]) + "\n")
print("val 4 files, full 16 files,", len(allrows), "rows")
