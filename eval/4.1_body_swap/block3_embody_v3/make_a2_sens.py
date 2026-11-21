"""Generate prereg_block3_a2_sens.py = cp prereg_block3_a2.py + asserted one-occurrence replacements (Fable add1 §2.3 item 4, §4 item 3). Group name = key of out/physics_breakdown_published.json (physics_breakdown_scan.py --preset published).
Only adds --exclude-keys JSON: OFT object records lose the listed (task, episode) keys in EVERY cell (paired); pi0.5 untouched.
Usage: python make_a2_sens.py A2_PATH OUT_PATH"""
import hashlib, sys
A2, OUT = sys.argv[1], sys.argv[2]
src = open(A2, encoding="utf-8").read()
assert hashlib.sha256(src.encode("utf-8")).hexdigest() == "003032bace8edc99bb032cc32f504318e4d7640bc23e71b974735d2a9bd6eb22", "a2 sha256 mismatch"
R = [
 # S1 constants + loader
 ('H = os.path.expanduser\n',
  'H = os.path.expanduser\n'
  '# sens（Fable 补裁一 §2.3 第 4 条）：a2 原版 sha256；--exclude-keys 读取的组名\n'
  'A2_SHA = "003032bace8edc99bb032cc32f504318e4d7640bc23e71b974735d2a9bd6eb22"\n'
  'EXCL_GROUP = "oft_pub|libero_object"\n'
  '\n'
  '\n'
  'def load_exclude(path):\n'
  '    """sens: {group: [[t, e], ...]} or {group: {cell: [[t, e], ...]}} -> set of (t, e) keys (union over cells)."""\n'
  '    d = json.load(open(H(path)))\n'
  '    if EXCL_GROUP not in d:\n'
  '        sys.exit(f"--exclude-keys: 文件里没有组 {EXCL_GROUP}（有 {sorted(d)}）")\n'
  '    v = d[EXCL_GROUP]\n'
  '    if isinstance(v, dict):\n'
  '        v = [k for ks in v.values() for k in ks]\n'
  '    return {(int(k[0]), int(k[1])) for k in v}\n'),
 # S2 exclusion at load time (OFT, object only), after src_oft is bound to the unfiltered SRC
 ('        self.src_oft = src_oft if src_oft is not None else (self.data.get("SRC") if kind == "oft" else None)\n',
  '        self.src_oft = src_oft if src_oft is not None else (self.data.get("SRC") if kind == "oft" else None)\n'
  '        if kind == "oft" and suite == "object" and getattr(a, "exclude_keys", None):  # sens: 物理失效集剔除，所有格同剔（保配对）\n'
  '            ex = load_exclude(a.exclude_keys)\n'
  '            gone = {c: sum(k in ex for k in rec) for c, rec in self.data.items()}\n'
  '            self.data = {c: {k: r for k, r in rec.items() if k not in ex} for c, rec in self.data.items()}\n'
  '            P(f"\\n==== 物理失效集剔除（{self.name}，object）：{a.exclude_keys}（sha256 {sha256(H(a.exclude_keys))}，组 {EXCL_GROUP}）")\n'
  '            P(f"  {len(ex)} 键：{sorted(ex)}")\n'
  '            P("  每格剔除键数：" + "  ".join(f"{c} -{n}" for c, n in gone.items()))\n'),
 # S3 pi0.5 object fixcheck keeps the unfiltered OFT SRC (same object as before when nothing is excluded)
 ('        models[1].src_oft = models[0].data.get("SRC")\n        nogr =',
  '        models[1].src_oft = models[0].src_oft  # sens: 未剔除的 OFT SRC；不剔除时与 data["SRC"] 是同一对象\n        nogr ='),
 # S4 CLI
 ('    ap.add_argument("--amendment", default="$PROJ_ROOT/embody_v3/protocol_block3_amendment1.md")\n',
  '    ap.add_argument("--amendment", default="$PROJ_ROOT/embody_v3/protocol_block3_amendment1.md")\n'
  '    ap.add_argument("--exclude-keys", default=None)  # sens\n'),
 # S5 first line
 ('    P("prereg_block3_a2.py：按附录 2 改动 D1 到 D4，其余与 f1da2c63 逐行相同")\n',
  '    P("事后稳健性检查：物理失效集剔除版；检测器结局盲、定义于重算之前")\n'
  '    P("prereg_block3_a2.py：按附录 2 改动 D1 到 D4，其余与 f1da2c63 逐行相同")\n'),
 # S6 sha256 section
 ('                          ("prereg_block3_a2.py（本脚本）", me, None)]:\n',
  '                          ("prereg_block3_a2.py（a2 原版）", os.path.join(os.path.dirname(me), "prereg_block3_a2.py"), A2_SHA),\n'
  '                          ("prereg_block3_a2_sens.py（本脚本）", me, None)]:\n'),
]
for i, (old, new) in enumerate(R, 1):
    n = src.count(old)
    assert n == 1, f"S{i}: old text occurs {n} times"
    src = src.replace(old, new)
open(OUT, "w", encoding="utf-8").write(src)
print(f"written {OUT}: {len(R)} replacements, sha256 {hashlib.sha256(src.encode('utf-8')).hexdigest()}")
