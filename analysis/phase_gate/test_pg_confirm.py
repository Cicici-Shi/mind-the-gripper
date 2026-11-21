"""Synthetic tests of analysis/pg_confirm.py (protocol_phasegate_confirm.md section 9). Run: python analysis/test_pg_confirm.py

Episodes are written with per-task uniform outcomes, so every bootstrap interval is exact (zero width) and each
hypothesis lands on a known side of its line. One synthetic episode: 80 actions, own close at step tc (no lift)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_confirm as C

N_ACT = 80
EXP = C.expected_keys(C.TASKS, C.EPS)


def ep_arrays(window, tc):
    ph = np.asarray([0 if (tc < 0 or k <= tc) else 1 for k in range(N_ACT + 1)], np.int8)
    return ph, C.gate_rule(window, ph)


def write_cell(root, block, mode, window, fail, tc=40, tc_of=None, success_obj=False, label=None, skip=(),
               err=(), logs=None):
    """fail(task, ep) -> bool; tc_of(task, ep) -> own close step (default tc)."""
    d = os.path.join(root, block, f"{mode}_{window}")
    os.makedirs(d, exist_ok=True)
    for (t, e), init in EXP.items():
        if (t, e) in skip:
            continue
        c = tc_of(t, e) if tc_of else tc
        ph, g = ep_arrays(window, c)
        y = np.array(None, dtype=object) if success_obj else (not fail(t, e))
        np.savez_compressed(os.path.join(d, f"t{t}_e{e}_i{init}.npz"), success=y, task=t, ep=e, init=init,
                            action=np.zeros((N_ACT, 7), np.float32), pg_mode=mode, pg_window=label or window,
                            pg_lift_m=0.03, pg_phase=ph, pg_gate=g, pg_t_close=c, pg_t_lift=-1)
    logs = logs or os.path.join(root, "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, f"EVAL-libero_spatial-openvla-2026_11_11-00_00_00--pg_{block}_{mode}_{window}_e5.txt"), "w") as f:
        f.write(f"PHASEGATE: mode={mode} window={window} lift_m=0.03\n")
        for (t, e), init in EXP.items():
            f.write(f"\nTask: x\nINIT_ID: task={t} episode={e} init={init}\nStarting episode 1...\n")
            if (t, e) in err:
                f.write("Episode error: synthetic\nTraceback (most recent call last):\n")


def frac(p):
    """fail the first round(p * 20) episodes of every task: per-task uniform drop of 100 p points"""
    k = int(round(p * len(C.EPS)))
    return lambda t, e: (e - C.EPS[0]) < k


def make_model(root, model, spec, **kw):
    """spec[mode] = dict(window -> fail fraction), optional "_tc" (window -> tc or callable)."""
    blk = f"{model}_spatial"
    for mode in C.MODES:
        s = spec[mode]
        for w in C.WINDOWS:
            tcs = s.get("_tc", {}).get(w, 40)
            write_cell(root, blk, mode, w, s[w] if callable(s[w]) else frac(s[w]),
                       tc=tcs if not callable(tcs) else 40, tc_of=tcs if callable(tcs) else None,
                       err=s.get("_err", ()), **kw)
    return os.path.join(root, blk)


def run(root, dirs, **kw):
    txt, ok = C.report(dirs, os.path.join(root, "logs"), **kw)
    for f in C.FORBIDDEN:
        assert f not in txt
    # "集中" only inside the two allowed sentences
    for ln in txt.splitlines():
        if "集中" in ln:
            assert "掉分集中在闭合之前：只改闭合之前掉" in ln or "块达到『掉分集中在闭合之前』" in ln, ln
    return txt, ok


def labels(txt):
    return [ln.split("label: ", 1)[1] for ln in txt.splitlines() if ln.startswith("label: ")]


def hline(txt):
    return [ln for ln in txt.splitlines() if ln.startswith("H1 ")]


BASE = {"none": 0.0, "approach": 0.30, "post": 0.05, "all": 0.35}  # H1-H4 hold, additive
NOT_EXCL = {"none": 0.0, "approach": 0.30, "post": 0.15, "all": 0.45}  # H3 fails (D_post = -15)
NO_H1 = {"none": 0.0, "approach": 0.05, "post": 0.0, "all": 0.05}  # D_all = -5
H2_FAIL = {"none": 0.0, "approach": 0.05, "post": 0.0, "all": 0.30}  # D_approach = -5, H2 fails
H4_FAIL = {"none": 0.0, "approach": 0.20, "post": 0.20, "all": 0.40}  # C = 0, H4 fails (H3 fails too)
SUPER = {"none": 0.0, "approach": 0.20, "post": 0.05, "all": 0.50}  # A = -25: worse than the sum
SUB = {"none": 0.0, "approach": 0.30, "post": 0.05, "all": 0.20}  # A = +15: milder than the sum


def tmpdir():
    return tempfile.TemporaryDirectory()


def test_all_hold_and_total_sentence():
    with tmpdir() as r:
        a = make_model(r, "oft", {"look": BASE, "erase": NOT_EXCL})
        b = make_model(r, "ofti", {"look": BASE, "erase": NO_H1})
        txt, ok = run(r, [a, b])
        assert ok
        lb = labels(txt)
        assert lb[0] == C.L_CONC.format(x=30.0), lb[0]
        assert lb[1] == C.L_NOT_EXCL and lb[3] == C.L_NO_H1, lb
        assert hline(txt)[0].startswith("H1 成立  H2 成立  H3 成立  H4 成立"), hline(txt)[0]
        assert hline(txt)[1].startswith("H1 成立  H2 成立  H3 不成立  H4 成立")
        assert hline(txt)[3].startswith("H1 不成立")
        assert C.L_TOTAL.format(x=2) in txt, txt
        assert "C = D_approach - D_post:  -25.0 [ -25.0, -25.0]" in txt
        assert "tasks with D_approach < D_post: 10 of 10" in txt
        assert "A/A (look:none - erase:none, n=200):   +0.0" in txt


def test_cannot_branches():
    with tmpdir() as r:
        a = make_model(r, "oft", {"look": H2_FAIL, "erase": H4_FAIL})
        txt, _ = run(r, [a])
        lb = labels(txt)
        assert lb[0].startswith(C.L_CANNOT + "：只改闭合之前   -5.0") and hline(txt)[0].startswith("H1 成立  H2 不成立"), lb[0]
        assert lb[1].startswith(C.L_CANNOT) and "H4 不成立" in hline(txt)[1], (lb[1], hline(txt)[1])
        assert "（总句需要四块，本次输入 2 块）" in txt


def test_additivity_sentence():
    with tmpdir() as r:
        a = make_model(r, "oft", {"look": SUPER, "erase": SUB})
        lb = labels(run(r, [a])[0])
        assert lb[0].endswith(C.L_ADD.format(w="更糟", y=25.0) + "。"), lb[0]
        assert lb[1].endswith(C.L_ADD.format(w="更轻", y=15.0) + "。"), lb[1]
        # additive block: no additivity sentence
        b = make_model(r, "ofti", {"look": BASE, "erase": BASE})
        assert "全程改比两半之和" not in labels(run(r, [b])[0])[0]


def test_aa_trigger():
    with tmpdir() as r:
        hi = dict(BASE, none=0.10)  # look:none 10 points below erase:none (fail first 2 eps per task)
        lo = dict(BASE, none=0.0)
        a = make_model(r, "oft", {"look": {w: (v + 0.10 if w != "none" else 0.10) for w, v in BASE.items()}, "erase": lo})
        txt, _ = run(r, [a])
        assert "|D_AA| >= 10" in txt and labels(txt) == [C.L_AA, C.L_AA], labels(txt)
        assert "（A/A 触发，不出成立句）" in txt and "达到" not in txt.split("（总句")[1].replace("未达到", "")
        # 5 points: no trigger
        b = make_model(r, "ofti", {"look": {w: v + 0.05 for w, v in BASE.items()}, "erase": BASE})
        txt = run(r, [b])[0]
        assert "below the noise-floor line" in txt and C.L_AA not in txt
        del hi


def test_rule_self_close():
    with tmpdir() as r:
        s = dict(BASE, _tc={"approach": 5})  # approach cell closes at 5: 6 gated steps < 0.5 x 40
        a = make_model(r, "oft", {"look": s, "erase": BASE})
        txt = run(r, [a])[0]
        assert "approach window closed early YES" in txt and "窗口被干预提前关闭，不可解读" in txt
        assert hline(txt)[0].startswith("H1 成立  H2 不判  H3 成立  H4 不判"), hline(txt)[0]
        assert labels(txt)[0].startswith("（协议 §4 未定义这一组合：H1 成立，H2、H4 不判（窗口规则）"), labels(txt)[0]


def test_rule_no_close():
    with tmpdir() as r:
        s = dict(BASE, _tc={"approach": lambda t, e: -1 if e < 8 else 40})  # 15% never close (3 of 20)
        a = make_model(r, "oft", {"look": s, "erase": BASE})
        txt = run(r, [a])[0]
        assert "approach never-closed > 10% YES" in txt and "H2 不判" in hline(txt)[0]
        s = dict(BASE, _tc={"approach": lambda t, e: -1 if e < 7 else 40})  # exactly 10%: not over
        b = make_model(r, "ofti", {"look": s, "erase": BASE})
        assert "approach never-closed > 10% no" in run(r, [b])[0]


def test_rule_dose():
    with tmpdir() as r:
        s = dict(BASE, _tc={"post": 60})  # post opens after 60: queries 64, 72 -> 2 < 4
        a = make_model(r, "oft", {"look": s, "erase": BASE})
        txt = run(r, [a])[0]
        assert "post query frames median < 4 YES" in txt and "H3 不判" in hline(txt)[0], hline(txt)[0]
        assert "H3 不判（剂量规则）" in labels(txt)[0]
        # tc 40: queries 48 56 64 72 = 4, not below the line
        assert "post query frames median < 4 no" in txt.split("block oft / erase")[1]


def test_key_exclusion():
    with tmpdir() as r:
        err = {(0, 5), (3, 9)}  # 2 of 200 = 1%: excluded in every cell of the block, block still read
        a = make_model(r, "oft", {"look": dict(BASE, _err=err), "erase": BASE})
        txt = run(r, [a])[0]
        assert "block oft / look: n=198 keys (excluded 2 of 200)" in txt and "episode-error keys 2" in txt
        assert "A/A (look:none - erase:none, n=198)" in txt
        err = {(t, e) for t in range(10) for e in (5, 6)}  # 20 of 200 = 10%
        b = make_model(r, "ofti", {"look": dict(BASE, _err=err), "erase": BASE})
        assert "按协议 §5 该块重跑一次" in labels(run(r, [b])[0])[0]
        assert "该块不合格" in labels(run(r, [b], rerun=("ofti:look",))[0])[0]


def test_integrity_and_label_mismatch():
    with tmpdir() as r:
        a = make_model(r, "oft", {"look": BASE, "erase": BASE})
        # label in the dump differs from the path -> integrity FAIL, no readout
        write_cell(r, "oft_spatial", "erase", "post", frac(0.05), label="approach")
        txt, ok = run(r, [a])
        assert not ok and "directory says erase/post" in txt and "no readout" in txt and "label:" not in txt
    with tmpdir() as r:
        a = make_model(r, "oft", {"look": BASE, "erase": BASE})
        os.remove(sorted(__import__("glob").glob(os.path.join(a, "look_all", "*.npz")))[0])
        txt, ok = run(r, [a])
        assert not ok and "199 npz, expected 200" in txt
    with tmpdir() as r:
        a = make_model(r, "oft", {"look": BASE, "erase": BASE})
        os.makedirs(os.path.join(a, "look_grasp"))
        assert "cell not in the protocol" in run(r, [a])[0]
    with tmpdir() as r:  # gate inconsistent with phase
        a = make_model(r, "oft", {"look": BASE, "erase": BASE})
        f = sorted(__import__("glob").glob(os.path.join(a, "erase_post", "*.npz")))[0]
        z = dict(np.load(f))
        z["pg_gate"] = ~z["pg_gate"]
        np.savez_compressed(f, **z)
        txt, ok = run(r, [a])
        assert not ok and "pg_gate inconsistent with pg_phase for window post" in txt
    with tmpdir() as r:  # query steps not every 8 actions
        a = make_model(r, "oft", {"look": BASE, "erase": BASE})
        fs = sorted(__import__("glob").glob(os.path.join(a, "look_post", "*.npz")))
        for f, q in ((fs[0], np.arange(0, N_ACT, 8)), (fs[1], np.arange(0, N_ACT, 5))):
            z = dict(np.load(f))
            z["query_step"] = q
            np.savez_compressed(f, **z)
        txt, ok = run(r, [a])
        assert not ok and txt.count("query_step is not") == 1 and os.path.basename(fs[1]) in txt
    with tmpdir() as r:  # wrong init for the key, missing eval log
        a = make_model(r, "oft", {"look": BASE, "erase": BASE})
        d = os.path.join(a, "look_none")
        f = sorted(os.listdir(d))[0]
        z = dict(np.load(os.path.join(d, f)))
        os.remove(os.path.join(d, f))
        z["init"] = (int(z["init"]) + 1) % 50
        np.savez_compressed(os.path.join(d, f"t{int(z['task'])}_e{int(z['ep'])}_i{int(z['init'])}.npz"), **z)
        for lf in os.listdir(os.path.join(r, "logs")):
            if "erase_all" in lf:
                os.remove(os.path.join(r, "logs", lf))
        txt, ok = run(r, [a])
        assert not ok and "the permutation gives" in txt and "erase_all: no eval log" in txt


def test_check_fields_only_never_reads_success():
    with tmpdir() as r:
        # success stored as an object array: reading it with allow_pickle=False raises
        blk = "oft_spatial"
        for mode in C.MODES:
            for w in C.WINDOWS:
                write_cell(r, blk, mode, w, frac(0), success_obj=True)
        txt, ok = run(r, [os.path.join(r, blk)], check_only=True)
        assert ok and "CHECK_FIELDS PASS" in txt and "label:" not in txt
        try:
            run(r, [os.path.join(r, blk)])
        except ValueError:
            pass
        else:
            raise SystemExit("full readout did not touch success")
        # smoke-sized key set (task 0, ep 0)
    with tmpdir() as r:
        global EXP
        old, EXP = EXP, C.expected_keys((0,), (0,))
        try:
            for mode in C.MODES:
                for w in C.WINDOWS:
                    write_cell(r, blk, mode, w, frac(0))
            txt, ok = run(r, [os.path.join(r, blk)], tasks=(0,), eps=(0,), check_only=True)
            assert ok and "1 per cell" in txt, txt
        finally:
            EXP = old


def test_out_written_once():
    with tmpdir() as r:
        out = os.path.join(r, "results_phasegate_confirm.txt")
        open(out, "w").write("x")
        sys.argv = ["pg_confirm.py", r, "--eval-logs", r, "--out", out]
        try:
            C.main()
        except SystemExit as e:
            assert "exists" in str(e)
        else:
            raise SystemExit("existing results file overwritten")


def test_constants():
    assert (C.DROP, C.AA_MAX, C.SELF_CLOSE_FRAC, C.NO_CLOSE_MAX, C.DOSE_MIN, C.EXCL_MAX) == (-10.0, 10.0, 0.5, 0.10, 4, 0.05)
    assert (C.B, C.SEED, C.QUERY_EVERY) == (5000, 0, 8) and C.EPS == tuple(range(5, 25)) and C.TASKS == tuple(range(10))
    assert C.WINDOWS == ("none", "approach", "post", "all") and C.MODES == ("look", "erase")
    assert C.parse_range("5-24") == C.EPS and C.parse_range("0") == (0,)
    # query frames: pg_gate[0:n:8]
    ph, g = ep_arrays("post", 40)
    assert int(g[0:N_ACT:8].sum()) == 4


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
    print(f"OK {n} pg_confirm tests")
