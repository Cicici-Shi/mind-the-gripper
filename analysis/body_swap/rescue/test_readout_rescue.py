"""Synthetic-data tests for readout_rescue.py (never touches real runs). python test_readout_rescue.py"""
import io, json, os, shutil, sys, tempfile, contextlib
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout_rescue as R  # noqa: E402

ENV = R.cells_env()
res = []


def ok(c, msg):
    res.append(bool(c)); print(("OK   " if c else "FAIL ") + msg)


def make(root, spec, drop_key=None, bad_env=None):
    """spec: {model: {cellname: number of successes n (keys[:n] succeed)}}"""
    for lab, m in R.CKPTS:
        for nm, c in R.CELLS.items():
            d = os.path.join(root, m, c); os.makedirs(d)
            n = spec[m][nm]
            for i, (t, e) in enumerate(R.KEYS):
                if drop_key == (m, nm, (t, e)):
                    continue
                env = dict(ENV[c])
                if bad_env == (m, nm, (t, e)):
                    env["GRP_MAP"] = "kp100"
                np.savez_compressed(os.path.join(d, f"t{t}_e{e}_i{(t * 7 + e) % 50}.npz"), success=i < n,
                                    gpu="NVIDIA A100-PCIE-40GB", pf_blind=1, pf_qacc_count=0,
                                    pf_cell_env=json.dumps(env, sort_keys=True))


def run(spec, B=10000, **kw):
    root = tempfile.mkdtemp(prefix="rs_test_")
    try:
        make(root, spec, **kw)
        out = os.path.join(root, "res.txt"); js = os.path.join(root, "v.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(["--runs", root, "--B", str(B), "--out", out, "--json", js])
        v = json.load(open(js)) if os.path.exists(js) else None
        txt = buf.getvalue()
        if rc == 0:  # second call must refuse (one readout only)
            with contextlib.redirect_stdout(io.StringIO()):
                ok(R.main(["--runs", root, "--out", out]) == 2, "  second readout refused (output exists)")
        return rc, v, txt
    finally:
        shutil.rmtree(root)


both = lambda s, g, t: {"rs_oftc_sp": {"S": s, "G2": g, "GTfc2": t}, "rs_ofti_sp": {"S": s, "G2": g, "GTfc2": t}}
# 1 large drop
rc, v, _ = run(both(160, 100, 150))
ok(rc == 0 and all(v["checkpoints"][k]["tier"] == "值得做" for k in ("合训 OFT", "OFTi")) and len(v["enters"]) == 2,
   f"drop -30 -> 值得做 both ({v and v['checkpoints']['合训 OFT']['drop_nat']})")
ok(abs(v["checkpoints"]["OFTi"]["recovery"]["point"] - 25) < 1e-9, "recovery point +25")
# 2 edge with recovery
rc, v, _ = run(both(160, 140, 160))
c = v["checkpoints"]["合训 OFT"]
ok(c["tier"] == "边缘" and c["enters"] and c["note"] == "可救幅度小" and c["drop_nat"]["ci95"][1] < 0,
   f"drop -10, rec +10 -> 边缘 enters 可救幅度小 (drop {c['drop_nat']}, rec {c['recovery']})")
# 3 edge without recovery
rc, v, _ = run(both(160, 140, 140))
c = v["checkpoints"]["OFTi"]
ok(c["tier"] == "边缘" and not c["enters"] and v["enters"] == [], "drop -10, rec 0 -> 边缘 not entering")
# 4 no drop
rc, v, txt = run(both(150, 150, 150))
ok(v["enters"] == [] and all(x["tier"] == "不值得做" for x in v["checkpoints"].values()) and R.NOT_WORTH_TXT in txt,
   "no drop -> 不值得做 + fixed sentence")
# 5 mixed checkpoints
rc, v, txt = run({"rs_oftc_sp": {"S": 160, "G2": 100, "GTfc2": 100}, "rs_ofti_sp": {"S": 150, "G2": 150, "GTfc2": 150}})
ok(v["enters"] == ["合训 OFT"] and R.NOT_WORTH_TXT not in txt, "OFT worth, OFTi not -> only 合训 OFT enters")
# 6 missing key refuses without SR
rc, v, txt = run(both(160, 100, 150), drop_key=("rs_ofti_sp", "GTfc2", (9, 19)))
ok(rc == 3 and v is None and "SR(" not in txt and "missing 1" in txt, "missing key -> refuse, no SR printed")
# 7 env mismatch refuses
rc, v, txt = run(both(160, 100, 150), bad_env=("rs_oftc_sp", "G2", (3, 4)))
ok(rc == 3 and "cell_env mismatch 1" in txt and "SR(" not in txt, "cell_env mismatch -> refuse")
# 8 verdict boundaries (pure function)
V = lambda p, hi, rlo=-1.0: R.verdict({"point": p, "ci95": (p - 5, hi)}, {"point": 0, "ci95": (rlo, 9)})
ok(V(-15, -1)[:2] == ("值得做", True), "p=-15 hi<0 -> 值得做")
ok(V(-15, 0.0)[0] == "不值得做", "p=-15 hi=0 -> 不值得做 (hi must be < 0)")
ok(V(-30, 2.0)[0] == "不值得做", "p=-30 hi>0 -> 不值得做")
ok(V(-14.99, -1, 0.5)[:2] == ("边缘", True), "p=-14.99 hi<0 rec_lo>0 -> 边缘 enters")
ok(V(-7, -0.1, 0.01)[:2] == ("边缘", True), "p=-7 boundary -> 边缘")
ok(V(-7, -0.1, 0.0)[:2] == ("边缘", False), "p=-7 rec_lo=0 -> 边缘 not entering")
ok(V(-6.99, -0.1, 5)[0] == "不值得做", "p=-6.99 -> 不值得做")
ok(V(+5, 9)[0] == "不值得做", "positive drop -> 不值得做")
ok(V(-15 + 1e-12, -1)[0] == "值得做" and V(-15 - 1e-12, -1)[0] == "值得做", "float noise at -15 -> 值得做")
ok(V(-7 + 1e-12, -1, 1)[0] == "边缘", "float noise at -7 -> 边缘")
ok(V(-20, 1e-12)[0] == "不值得做" and V(-20, -1e-12)[0] == "不值得做" and V(-20, -0.5)[0] == "值得做", "hi = 0 +- float noise counts as 0 (not < 0)")
# 9 frozen import guard
old = R.FROZEN_SHA; R.FROZEN_SHA = "0" * 64
try:
    R.boot_cls(); ok(False, "sha guard")
except SystemExit:
    ok(True, "frozen prereg_models1 sha mismatch -> refuse")
R.FROZEN_SHA = old
print(f"{sum(res)}/{len(res)} OK"); sys.exit(0 if all(res) else 1)
