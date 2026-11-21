"""phasegate-confirm readout (protocol_phasegate_confirm.md v1, sections 2 to 5 and 9).

Input: one directory per model, <OUTROOT>/<model>_spatial/, holding the eight cells <mode>_<window> for mode in
look / erase and window in none / approach / post / all (one npz per key t<task>_e<ep>_i<init>.npz). A block is one
(model, mode). Output: every estimate of section 2, the A/A of section 3, H1 to H4 and the block label of section 4,
the exclusion and invalidity rules of section 5, and the per-block conclusion for the overall sentence (printed when
the four blocks are given, i.e. both models).

Episode errors (section 5) come from the OFT eval logs (--eval-logs, the --local_log_dir of tools/pg_run.sbatch):
the npz has no error flag, and the eval log carries "INIT_ID: task=.. episode=.." before every episode and
"Episode error" / "Dump error" after a failed one. These logs never contain success.

Usage:
  python analysis/pg_confirm.py out/confirm/oft_spatial out/confirm/ofti_spatial --eval-logs logs/eval \
      --out out/results_phasegate_confirm.txt
  python analysis/pg_confirm.py out/smoke_confirm/oft_spatial --eval-logs logs/eval --check-fields-only \
      --tasks 0 --eps 0-0
--check-fields-only checks fields, labels, key sets and gate consistency and never reads success.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pg_readout import boot  # task-clustered percentile bootstrap (B = 5000, seed 0)

# ---- protocol constants (section 1 to 5; do not edit after the freeze) ----
MODES = ("look", "erase")
WINDOWS = ("none", "approach", "post", "all")
TASKS = tuple(range(10))
EPS = tuple(range(5, 25))
N_INIT = 50  # RandomState(5478 + task).permutation(50)[ep]
QUERY_EVERY = 8  # OFT reads one frame every 8 steps: query frames = pg_gate[0:n:8], n = number of actions
B, SEED = 5000, 0
DROP = -10.0  # H1 / H2: point estimate <= -10 and hi95 < 0; H3: lo95 > -10
AA_MAX = 10.0  # |D_AA| >= 10 -> both blocks of the model "uncertain (noise floor)"
SELF_CLOSE_FRAC = 0.5  # approach gated-steps median < 0.5 x none first-close median -> H2, H4 not judged
NO_CLOSE_MAX = 0.10  # approach cell never-closed share > 10% -> H2, H4 not judged
DOSE_MIN = 4  # post gated query frames median < 4 -> H3 not judged
EXCL_MAX = 0.05  # excluded keys > 5% -> rerun the block once; still > 5% -> block fails

REQ = ("success", "task", "ep", "init", "action", "pg_mode", "pg_window", "pg_lift_m", "pg_phase", "pg_gate",
       "pg_t_close", "pg_t_lift")

# label texts of section 4 (the only allowed wordings)
L_NO_H1 = "该块未重现全程掉分，时段问题在该块不成立。"
L_CONC = "掉分集中在闭合之前：只改闭合之前掉 {x:.1f} 点，只改闭合之后掉分小于 10 点。"
L_NOT_EXCL = "闭合之前更伤，但闭合之后的掉分没有排除 10 点。"
L_CANNOT = "不能说闭合之前更伤"
L_AA = "不确定（噪声底过大）"
L_ADD = "全程改比两半之和{w} {y:.1f} 点"
L_TOTAL = "四块中 {x} 块达到『掉分集中在闭合之前』。"
FORBIDDEN = ("模型只在接近段看手", "抓住之后不需要视觉")


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return tuple(out)


def expected_keys(tasks, eps):
    return {(t, e): int(np.random.RandomState(5478 + t).permutation(N_INIT)[e]) for t in tasks for e in eps}


def gate_rule(window, phase):
    p = np.asarray(phase)
    return {"none": np.zeros(len(p), bool), "all": np.ones(len(p), bool), "approach": p == 0, "post": p >= 1}[window]


def check_cell(cell_dir, mode, window, exp):
    """Fields and labels only (success is never read). Returns (problems, per-key non-outcome records)."""
    bad, recs = [], {}
    if not os.path.isdir(cell_dir):
        return [f"{cell_dir}: missing cell"], recs
    files = sorted(glob.glob(os.path.join(cell_dir, "t*_e*_i*.npz")))
    if len(files) != len(exp):
        bad.append(f"{cell_dir}: {len(files)} npz, expected {len(exp)}")
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
            miss = [k for k in REQ if k not in z.files]
            if miss:
                bad.append(f"{f}: missing {miss}")
                continue
            key, init = (int(z["task"]), int(z["ep"])), int(z["init"])
            if os.path.basename(f) != f"t{key[0]}_e{key[1]}_i{init}.npz":
                bad.append(f"{f}: file name does not match task/ep/init {key + (init,)}")
            if (str(z["pg_mode"]), str(z["pg_window"])) != (mode, window):
                bad.append(f"{f}: dump says {z['pg_mode']}/{z['pg_window']}, directory says {mode}/{window}")
            if key not in exp:
                bad.append(f"{f}: key {key} outside the protocol key set")
            elif exp[key] != init:
                bad.append(f"{f}: init {init}, the permutation gives {exp[key]}")
            n = len(z["action"])
            g, ph = np.asarray(z["pg_gate"], bool), np.asarray(z["pg_phase"])
            if len(g) != n + 1 or len(ph) != n + 1:
                bad.append(f"{f}: pg_gate {len(g)} / pg_phase {len(ph)} entries for {n} actions")
                continue
            if not np.array_equal(g, gate_rule(window, ph)):
                bad.append(f"{f}: pg_gate inconsistent with pg_phase for window {window}")
            if "query_step" in z.files and not np.array_equal(np.asarray(z["query_step"]), np.arange(0, n, QUERY_EVERY)):
                bad.append(f"{f}: query_step is not 0, {QUERY_EVERY}, 16, ... (gated query frames assume it)")
            recs[key] = {"gated": int(g.sum()), "gated_q": int(g[0:n:QUERY_EVERY].sum()),
                         "t_close": int(z["pg_t_close"]), "file": f}
        except Exception as e:  # truncated or unreadable npz
            bad.append(f"{f}: unreadable ({e!r})")
    missing = sorted(set(exp) - set(recs))
    if missing and len(files) == len(exp):
        bad.append(f"{cell_dir}: keys missing {missing[:5]}")
    return bad, recs


def log_errors(log_dir, block, cell, exp):
    """Keys with "Episode error" / "Dump error" in the eval logs of this cell; None if no log was found."""
    pat = os.path.join(log_dir, f"EVAL-*--pg_{block}_{cell}_e*.txt")
    files = sorted(glob.glob(pat))
    if not files:
        return None, []
    err, seen = set(), set()
    for f in files:
        cur = None
        for ln in open(f, errors="replace"):
            m = re.match(r"INIT_ID: task=(\d+) episode=(\d+) init=(\d+)", ln)
            if m:
                cur = (int(m.group(1)), int(m.group(2)))
                seen.add(cur)
            elif (ln.startswith("Episode error") or ln.startswith("Dump error")) and cur in exp:
                err.add(cur)
    return err, files


def check_model(model_dir, log_dir, exp):
    """Integrity of all eight cells (section 5 completeness) plus the error keys per block, without success."""
    block = os.path.basename(os.path.normpath(model_dir))
    bad, recs, errs, used_logs = [], {}, {}, []
    for mode in MODES:
        errs[mode] = set()
        for w in WINDOWS:
            c = f"{mode}_{w}"
            b, r = check_cell(os.path.join(model_dir, c), mode, w, exp)
            bad += b
            recs[(mode, w)] = r
            e, files = log_errors(log_dir, block, c, exp)
            if e is None:
                bad.append(f"{block}/{c}: no eval log in {log_dir} (needed for the episode-error rule)")
            else:
                errs[mode] |= e
                used_logs += files
    for d in sorted(glob.glob(os.path.join(model_dir, "*_*"))):
        if os.path.isdir(d) and tuple(os.path.basename(d).split("_", 1)) not in {(m, w) for m in MODES for w in WINDOWS}:
            bad.append(f"{d}: cell not in the protocol")
    return block, bad, recs, errs, used_logs


def load_success(recs):
    for r in recs.values():
        with np.load(r["file"], allow_pickle=False) as z:
            r["y"] = float(bool(z["success"]))


def est(keys, d):
    lo, hi = boot(keys, d, B=B, seed=SEED)
    return {"pt": float(np.mean([d[k] for k in keys])), "lo": lo, "hi": hi}


def fmt(e):
    return f"{e['pt']:+6.1f} [{e['lo']:+6.1f},{e['hi']:+6.1f}]"


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def block_result(model, mode, recs, excl, rerun):
    """Section 2, 4 and 5 for one block; recs[(mode, w)] carry y."""
    cells = {w: recs[(mode, w)] for w in WINDOWS}
    keys = sorted(set.intersection(*(set(c) for c in cells.values())) - excl)
    n_all = len(set.union(*(set(c) for c in cells.values())))
    res = {"model": model, "mode": mode, "n": len(keys), "excluded": len(excl), "n_keys": n_all, "lines": []}
    y = {w: {k: 100.0 * cells[w][k]["y"] for k in keys} for w in WINDOWS}
    D = {w: est(keys, {k: y[w][k] - y["none"][k] for k in keys}) for w in ("approach", "post", "all")}
    C = est(keys, {k: y["approach"][k] - y["post"][k] for k in keys})
    A = est(keys, {k: y["all"][k] - y["approach"][k] - y["post"][k] + y["none"][k] for k in keys})
    res.update(D=D, C=C, A=A)
    # descriptives
    desc = {}
    for w in WINDOWS:
        c = [cells[w][k] for k in keys]
        tc = [r["t_close"] for r in c if r["t_close"] >= 0]
        fail = [r for r in c if r["y"] == 0]
        desc[w] = {"sr": 100 * np.mean([r["y"] for r in c]), "gated": med([r["gated"] for r in c]),
                   "gated_q": med([r["gated_q"] for r in c]), "t_close": med(tc),
                   "no_close": 100 * (1 - len(tc) / len(c)),
                   "fail_closed": sum(r["t_close"] >= 0 for r in fail), "fail_never": sum(r["t_close"] < 0 for r in fail)}
    res["desc"] = desc
    tasks = sorted({k[0] for k in keys})
    per_t = {t: [k for k in keys if k[0] == t] for t in tasks}
    da = {t: np.mean([y["approach"][k] - y["none"][k] for k in per_t[t]]) for t in tasks}
    dp = {t: np.mean([y["post"][k] - y["none"][k] for k in per_t[t]]) for t in tasks}
    res["dir"] = (sum(da[t] < dp[t] for t in tasks), sum(da[t] == dp[t] for t in tasks), len(tasks))
    # section 5 invalidity rules
    ap = cells["approach"]
    self_close = desc["approach"]["gated"] < SELF_CLOSE_FRAC * desc["none"]["t_close"]
    no_close = np.mean([ap[k]["t_close"] < 0 for k in keys]) > NO_CLOSE_MAX
    low_dose = desc["post"]["gated_q"] < DOSE_MIN
    res["rules"] = {"self_close": bool(self_close), "no_close": bool(no_close), "low_dose": bool(low_dose)}
    excl_frac = len(excl) / max(n_all, 1)
    res["excl_over"] = excl_frac > EXCL_MAX
    # section 4 hypotheses (None = not judged)
    h1 = D["all"]["pt"] <= DROP and D["all"]["hi"] < 0
    h24_judged = not (self_close or no_close)
    h2 = (D["approach"]["pt"] <= DROP and D["approach"]["hi"] < 0) if h24_judged else None
    h4 = (C["hi"] < 0) if h24_judged else None
    h3 = (D["post"]["lo"] > DROP) if not low_dose else None
    res["H"] = {"H1": bool(h1), "H2": h2, "H3": h3, "H4": h4}
    res["conc"] = False
    if res["excl_over"]:
        res["label"] = (f"剔除键 {len(excl)}/{n_all} 超过 5%：该块不合格。" if rerun else
                        f"剔除键 {len(excl)}/{n_all} 超过 5%：按协议 §5 该块重跑一次，本次不出标签。")
        return res
    if not h1:
        res["label"] = L_NO_H1
        return res
    if h2 and h3 and h4:
        res["label"], res["conc"] = L_CONC.format(x=-D["approach"]["pt"]), True
    elif h2 and h4 and h3 is False:
        res["label"] = L_NOT_EXCL
    elif h2 is False or h4 is False:
        res["label"] = (f"{L_CANNOT}：只改闭合之前 {fmt(D['approach'])}，只改闭合之后 {fmt(D['post'])}。")
    else:
        why = "H2、H4 不判（窗口规则）" if not h24_judged else "H3 不判（剂量规则）"
        res["label"] = f"（协议 §4 未定义这一组合：H1 成立，{why}；不出标签）"
    if A["lo"] > 0 or A["hi"] < 0:
        res["label"] += " " + L_ADD.format(w="更糟" if A["pt"] < 0 else "更轻", y=abs(A["pt"])) + "。"
    return res


def yn(h):
    return "不判" if h is None else ("成立" if h else "不成立")


def report(model_dirs, log_dir, tasks=TASKS, eps=EPS, rerun=(), check_only=False):
    exp = expected_keys(tasks, eps)
    L = ["phasegate-confirm readout (protocol_phasegate_confirm.md v1)", f"keys: tasks {tasks[0]}-{tasks[-1]} x ep "
         f"{eps[0]}-{eps[-1]} = {len(exp)} per cell; eval logs: {log_dir}", ""]
    ok_all, models = True, []
    for md in model_dirs:
        block, bad, recs, errs, logs = check_model(md, log_dir, exp)
        L.append(f"== {block}: integrity {'PASS' if not bad else 'FAIL'} (8 cells, {len(logs)} eval logs) ==")
        L += [f"  {b}" for b in bad[:30]] + ([f"  ... {len(bad) - 30} more"] if len(bad) > 30 else [])
        for mode in MODES:
            L.append(f"  {mode}: episode-error keys {len(errs[mode])}" + (f" {sorted(errs[mode])[:10]}" if errs[mode] else ""))
        ok_all &= not bad
        models.append((block, bad, recs, errs))
    if check_only or not ok_all:
        L += ["", f"CHECK_FIELDS {'PASS' if ok_all else 'FAIL'}" + ("" if ok_all or check_only else
                                                                    " -> no readout (section 5 completeness)")]
        return "\n".join(L), ok_all
    blocks = []
    for block, _, recs, errs in models:
        for r in recs.values():
            load_success(r)
        model = block.rsplit("_", 1)[0]
        res = {m: block_result(model, m, recs, errs[m], f"{model}:{m}" in rerun) for m in MODES}
        # section 3: A/A = look:none vs erase:none on keys valid in both blocks
        ka = sorted((set(recs[("look", "none")]) & set(recs[("erase", "none")])) - errs["look"] - errs["erase"])
        aa = est(ka, {k: 100.0 * (recs[("look", "none")][k]["y"] - recs[("erase", "none")][k]["y"]) for k in ka})
        aa_hit = abs(aa["pt"]) >= AA_MAX
        L += ["", f"######## model {model} ########",
              f"A/A (look:none - erase:none, n={len(ka)}): {fmt(aa)}  -> {'|D_AA| >= 10: both blocks ' + L_AA if aa_hit else 'below the noise-floor line'}"]
        for m in MODES:
            r = res[m]
            if aa_hit:
                r["label"], r["conc"] = L_AA, False
            L += ["", f"== block {model} / {m}: n={r['n']} keys (excluded {r['excluded']} of {r['n_keys']}) =="]
            L.append(f"{'cell':9s} {'SR':>6s} {'D vs none [95%]':>24s} {'gated steps':>12s} {'gated queries':>14s} "
                     f"{'t_close':>8s} {'no close':>9s} {'fail closed/never':>18s}")
            for w in WINDOWS:
                d = r["desc"][w]
                dd = fmt(r["D"][w]) if w != "none" else ""
                L.append(f"{w:9s} {d['sr']:6.1f} {dd:>24s} {d['gated']:12.0f} {d['gated_q']:14.0f} {d['t_close']:8.0f} "
                         f"{d['no_close']:8.0f}% {d['fail_closed']:>8d}/{d['fail_never']:<9d}")
            L.append(f"C = D_approach - D_post: {fmt(r['C'])}    A = D_all - (D_approach + D_post): {fmt(r['A'])}")
            L.append(f"tasks with D_approach < D_post: {r['dir'][0]} of {r['dir'][2]} (ties {r['dir'][1]})")
            ru = r["rules"]
            L.append(f"rules: approach window closed early {'YES' if ru['self_close'] else 'no'} | approach never-closed "
                     f"> 10% {'YES' if ru['no_close'] else 'no'} | post query frames median < 4 {'YES' if ru['low_dose'] else 'no'}")
            if ru["self_close"]:
                L.append("  approach 格：窗口被干预提前关闭，不可解读（H2、H4 不判）")
            L.append("H1 " + yn(r["H"]["H1"]) + "  H2 " + yn(r["H"]["H2"]) + "  H3 " + yn(r["H"]["H3"]) + "  H4 " + yn(r["H"]["H4"])
                     + ("  （A/A 触发，不出成立句）" if aa_hit else ""))
            L.append(f"label: {r['label']}")
            blocks.append(r)
    L.append("")
    if len(blocks) == 4:
        L.append(L_TOTAL.format(x=sum(r["conc"] for r in blocks)))
    else:
        L.append(f"（总句需要四块，本次输入 {len(blocks)} 块）")
    for r in blocks:
        L.append(f"  {r['model']} / {r['mode']}: {'达到' if r['conc'] else '未达到'}  {r['label']}")
    txt = "\n".join(L)
    for f in FORBIDDEN:
        assert f not in txt, f
    return txt, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dirs", nargs="+")
    ap.add_argument("--eval-logs", required=True, help="the --local_log_dir of the runs (tools/pg_run.sbatch: $T/logs/eval)")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--eps", default="5-24")
    ap.add_argument("--rerun", default="", help="comma list of model:mode blocks that are already the section-5 rerun")
    ap.add_argument("--check-fields-only", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.out and os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; the readout is written once")
    txt, ok = report(a.model_dirs, a.eval_logs, parse_range(a.tasks), parse_range(a.eps),
                     tuple(x for x in a.rerun.split(",") if x), a.check_fields_only)
    print(txt)
    if a.out and ok and not a.check_fields_only:
        open(a.out, "w").write(txt + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
