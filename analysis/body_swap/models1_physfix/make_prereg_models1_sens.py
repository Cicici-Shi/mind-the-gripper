"""Generate prereg_models1_sens.py from the frozen prereg_models1.py (Fable add1 §2.3 item 2, §4 item 2).

cp + asserted replacements: every old string must occur exactly once in the current text before it is replaced
(same method as prereg_block3_a2.py, WORKLOG 9.2). Then writes prereg_models1_sens.diff (diff -u frozen -> sens).
Reproducible: python make_prereg_models1_sens.py  (run in $PROJ_ROOT/vla_zoo/analysis; refuses to overwrite read-only output)
Edits:
  S0 docstring head: one line naming the file as the physics-breakdown sensitivity copy
  S1 ModelData.__init__ Y matrices: --exclude-keys keys of this model x suite -> NaN in every cell (paired), after the
     §7.1 slice rule; gates / task tables / contacts / E1-E2 keep all keys
  S2 helpers (SENS state, sens_bound) before print_est; print_est header line lists the excluded keys of that model
  S3 print_est estimand row: + "失效集上界 ±x.x" for new jsonl models (π0 / OVLA) when --bounds is given
  S4 argparse: --exclude-keys, --bounds
  S5 main: first output line + sha256 block (two scripts, diff, two json) + loading of the two json files
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "prereg_models1.py")
DST = os.path.join(HERE, "prereg_models1_sens.py")
DIFF = os.path.join(HERE, "prereg_models1_sens.diff")
FROZEN_SHA = "29faca6ae9d63233b21378b62b61f40655a052cab924bd0c20e451f1e1c346e5"

R = []

# S0 -------------------------------------------------------------------------------------------------------------
R.append(("S0 docstring",
'''"""models1 pre-registered analysis: protocol_models1.md (+ protocol_models1_decisions.md).
''',
'''"""[prereg_models1_sens.py：物理失效集剔除版（Fable 补裁一 §2.3 第 2 条），由 make_prereg_models1_sens.py 从冻结版断言替换生成，diff 见 prereg_models1_sens.diff]
models1 pre-registered analysis: protocol_models1.md (+ protocol_models1_decisions.md).
'''))

# S1 -------------------------------------------------------------------------------------------------------------
R.append(("S1 Y matrices",
'''        self.Y200 = {c: np.array([rec[k]["succ"] if k in rec else np.nan for k in self.keys200]) for c, rec in self.data.items()}
        self.Y400 = {c: np.array([rec[k]["succ"] if k in rec else np.nan for k in self.keys400]) for c, rec in self.data.items()
                     if c in self.dense}
''',
'''        # SENS (i): --exclude-keys of this model x suite ("<model_dir>|libero_<suite>") -> NaN in EVERY cell (paired).
        # Applied after the §7.1 slice rule (dropping records before it would make the whole 50-key slice incomplete)
        # and only to Y: estimands, verdicts, Holm, Δ_model use the reduced keys; gates, task tables, contacts, E1/E2
        # still see all keys (add1 §2.3 item 5: gates are printed as the frozen script judges them, not re-licensed).
        self.sens_model_dir = model_dir
        _ex = set() if published else SENS["exclude"].get(f"{model_dir}|libero_{suite}", set())
        self.sens_excluded = sorted(k for k in _ex if any(k in rec for rec in self.data.values()))
        self.sens_listed = len(_ex)
        self.Y200 = {c: np.array([rec[k]["succ"] if (k in rec and k not in _ex) else np.nan for k in self.keys200]) for c, rec in self.data.items()}
        self.Y400 = {c: np.array([rec[k]["succ"] if (k in rec and k not in _ex) else np.nan for k in self.keys400]) for c, rec in self.data.items()
                     if c in self.dense}
'''))

# S2 -------------------------------------------------------------------------------------------------------------
R.append(("S2 helpers + print_est header",
'''def print_est(m, res, title):
    P(f"\\n==== {title}（{m.label}，{m.suite}）：点估计、95% / 90% bootstrap CI、键数、共同整片数")
''',
'''# ---------------------------------------------------------------------------------------- SENS (physics breakdown)
SENS = {"exclude": {}, "bounds": None}


def sens_load(path, kind):
    """--exclude-keys: {"<model_dir>|libero_<suite>": [[t, e], ...]}; --bounds: {"<model_dir>|<cell dir>": {"<ep offset>": w}}.
    Top-level fields starting with "_" are ignored. Anything else malformed raises (fail loudly)."""
    with open(H(path)) as fh:
        d = json.load(fh)
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if kind == "exclude":
            out[k] = {(int(t), int(e)) for t, e in v}
        else:
            out[k] = {str(o): int(w) for o, w in v.items() if not str(o).startswith("_")}
    return out


def sens_bound(m, nm, r):
    """SENS (ii): π0 / OVLA (new jsonl models) only. x = 100 * sum_cells |coef| * w_cell / n, w_cell = QACC warnings of that
    cell summed over the slices the estimand actually uses (from its mask); rounded UP to 0.1 (it is an upper bound).
    Two-cell estimands: (w_a + w_b) / n, as add1 §2.3 item 2 writes it."""
    if SENS["bounds"] is None or m.published or m.kind != "jsonl":
        return ""
    terms = {**EST, **REC_PRIME}.get(nm)
    if terms is None:
        return "  失效集上界 缺（未知估计量）"
    keys = m.keys400 if len(r["mask"]) == len(m.keys400) else m.keys200
    offs = sorted({k[1] // SLICE * SLICE for k, ok in zip(keys, r["mask"]) if ok})
    tot, miss = 0, []
    for a, c in terms:
        if not a:
            continue
        ent = SENS["bounds"].get(f"{m.sens_model_dir}|{m.spec[c][1]}")
        if ent is None:
            miss.append(f"{m.sens_model_dir}|{m.spec[c][1]}")
            continue
        for o in offs:
            if str(o) not in ent:
                miss.append(f"{m.sens_model_dir}|{m.spec[c][1]}@{o}")
                continue
            tot += abs(a) * ent[str(o)]
    if miss:
        return f"  失效集上界 缺（{', '.join(miss)}）"
    x = math.ceil(round(1000 * tot / r["n"], 6)) / 10 if r["n"] else float("nan")
    return f"  失效集上界 ±{x:.1f}"


def print_est(m, res, title):
    P(f"\\n==== {title}（{m.label}，{m.suite}）：点估计、95% / 90% bootstrap CI、键数、共同整片数")
    if getattr(m, "sens_excluded", None):
        P(f"  [剔除版] 物理失效键 {len(m.sens_excluded)} 个在整片内、已从本模型本套件所有格剔除（json 列出 {m.sens_listed}）：{m.sens_excluded}")
'''))

# S3 -------------------------------------------------------------------------------------------------------------
R.append(("S3 estimand row bound",
'''          f"  n={r['n']} 片={r['nsl']}{flag}{cell_tags(m, r['cells'])}{r10_tag(m, r)}")
''',
'''          f"  n={r['n']} 片={r['nsl']}{flag}{cell_tags(m, r['cells'])}{r10_tag(m, r)}{sens_bound(m, nm, r)}")
'''))

# S4 -------------------------------------------------------------------------------------------------------------
R.append(("S4 argparse",
'''    ap.add_argument("--overwrite", action="store_true")
''',
'''    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exclude-keys", default=None, help="SENS: physics_breakdown json {model_dir|libero_suite: [[t,e],...]}")
    ap.add_argument("--bounds", default=None, help="SENS: QACC warning json {model_dir|cell_dir: {ep_offset: w}}")
'''))

# S5 -------------------------------------------------------------------------------------------------------------
R.append(("S5 header + load",
'''    P("prereg_models1.py：models1 预注册分析（protocol_models1.md §4、§5、§7、§8）")
''',
'''    SENS["exclude"], SENS["bounds"] = {}, None
    _me = os.path.abspath(__file__)
    _dir = os.path.dirname(_me)
    P("物理失效集剔除版，diff 见 prereg_models1_sens.diff")
    P("==== 剔除版 sha256（Fable 补裁一 §2.3 第 2 条）")
    for lbl, pth in [("prereg_models1.py（冻结原版，同目录）", os.path.join(_dir, "prereg_models1.py")),
                     ("prereg_models1_sens.py（剔除版）", _me),
                     ("prereg_models1_sens.diff", os.path.join(_dir, "prereg_models1_sens.diff")),
                     ("--exclude-keys", H(a.exclude_keys) if a.exclude_keys else None),
                     ("--bounds", H(a.bounds) if a.bounds else None)]:
        P(f"  {lbl:32s} " + (f"{sha256(pth) or 'MISSING'}  [{pth}]" if pth else "未给"))
    if a.exclude_keys:
        SENS["exclude"] = sens_load(a.exclude_keys, "exclude")
        for k, v in sorted(SENS["exclude"].items()):
            P(f"  剔除表 {k}: {len(v)} 键 {sorted(v)}")
    if a.bounds:
        SENS["bounds"] = sens_load(a.bounds, "bounds")
        for k, v in sorted(SENS["bounds"].items()):
            P(f"  警告表 {k}: " + ", ".join(f"e{o}={w}" for o, w in sorted(v.items(), key=lambda x: int(x[0]))) + f"（合计 {sum(v.values())}）")
    P("  口径：OFTi 按键剔除（某模型某套件任一格被检测器 D 标记的键，从该模型该套件所有格剔除，保持按键配对；在 §7.1 整片规则之后、"
      "只作用于估计量矩阵：估计量、判定、Holm、Δ_model 用剔除后的键；门槛、10 任务表、在线接触、E1/E2 仍按全部键照印，门槛不因剔除追认）。"
      "π0 / OVLA 不剔除，--bounds 给出时每个估计量行末印「失效集上界 ±x」，x = 100 × Σ|系数| × w_格 / n（w_格 = 该格在估计量所用片上的"
      " QACC 警告数之和；两格估计量即 (w_a + w_b) / n），向上取到 0.1。判定读法见补裁一 §2.3 第 3 条（与主读数不同则降为不确定，人工比对）。")
    V["sens_physics"] = {"exclude_keys": a.exclude_keys, "bounds": a.bounds,
                         "exclude_n": {k: len(v) for k, v in SENS["exclude"].items()},
                         "sha256": {"prereg_models1.py": sha256(os.path.join(_dir, "prereg_models1.py")), "prereg_models1_sens.py": sha256(_me),
                                    "exclude": sha256(H(a.exclude_keys)) if a.exclude_keys else None,
                                    "bounds": sha256(H(a.bounds)) if a.bounds else None}}
    P("prereg_models1.py：models1 预注册分析（protocol_models1.md §4、§5、§7、§8）")
'''))


def main():
    with open(SRC, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == FROZEN_SHA, f"frozen source sha256 {sha} != {FROZEN_SHA}"
    txt = raw.decode("utf-8")
    for name, old, new in R:
        n = txt.count(old)
        assert n == 1, f"{name}: old text occurs {n} times (must be exactly 1)"
        txt = txt.replace(old, new, 1)
        print(f"{name}: replaced (1 occurrence)")
    for p in (DST, DIFF):
        if os.path.exists(p) and not os.access(p, os.W_OK):
            sys.exit(f"{p} is read-only (frozen); refusing to overwrite")
    with open(DST, "w", encoding="utf-8") as f:
        f.write(txt)
    d = subprocess.run(["diff", "-u", "prereg_models1.py", "prereg_models1_sens.py"], cwd=HERE, capture_output=True, text=True)
    assert d.returncode == 1, f"diff rc {d.returncode}: {d.stderr}"
    with open(DIFF, "w", encoding="utf-8") as f:
        f.write(d.stdout)
    for p in (SRC, DST, DIFF):
        with open(p, "rb") as f:
            print(f"{hashlib.sha256(f.read()).hexdigest()}  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
