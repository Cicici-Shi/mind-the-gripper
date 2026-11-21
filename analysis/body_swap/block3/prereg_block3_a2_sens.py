"""Block-3 pre-registered analysis: protocol_block3.md section 8 (items 1-8) + protocol_block3_amendment1.md
(rulings 1, 2, 3.1, 3.3, 4 and section 7 items 1, 3, 5). Frozen before the full runs; run once
(and a second time only if section 5 densification is triggered).

Usage (cluster, conda env oft, CPU job: sbatch -p normal --cpus-per-task=4 --mem=16G):
  python prereg_block3.py                         # spatial, n = 200 -> results_block3_final.txt
  python prereg_block3.py --n 400                 # section 5 second look -> results_block3_final_n400.txt
  python prereg_block3.py --suite object          # section 6 -> results_block3_object.txt
Paths: --runs $PROJ_ROOT/runs  --pi05 $PROJ_ROOT/pi05/v2  --slurm-logs ~/logs  --eval-logs $PROJ_ROOT/openvla-oft/experiments/logs
       --pi05-logs $PROJ_ROOT/pi05/logs  --gr-t0 $PROJ_ROOT/embody_v3/out/gr_t0  --r7 $PROJ_ROOT/embody_v3/out/r7_predictions.tsv
       --stages-dir $PROJ_ROOT/embody/analysis  --protocol/--amendment (for sha256 only)  --out FILE  --B 10000

Inference (protocol section 2): per model, paired cluster bootstrap over (task, episode) keys, all cells resampled
jointly (one draw matrix per model), B = 10000, seed 0, percentile 95% and 90% CIs. OFT is primary; pi0.5 is P7.

Implementation readings (实现口径) of places the protocol leaves open. Each is the most literal reading I could find.
 K1  Key universe. The bootstrap resamples the design key set (tasks 0-9 x episodes 0..n/10-1: 200, or 400 with
     --n 400), one draw matrix per model shared by every estimand ("all cells resampled jointly"). Each estimand
     uses the keys present in all of ITS cells (protocol 4.1 "按共同键分析"); in a resample its SRs are ratio means
     over the drawn copies of those keys. Missing keys per cell are printed. Block 2 instead intersected all cells;
     with 20+ cells and conditional cells that would let one missing key in an unrelated cell change every estimate.
 K2  Episode range: --n 200 keeps episodes 0-19 of every task, --n 400 keeps 0-39 (EP_OFFSET 20..35 slices).
     pi0.5 is never densified; in --n 400 mode pi0.5 is analysed at 200 exactly as in the first look.
 K3  --n 400: section A re-prints the whole n = 200 analysis; section B recomputes P9 and P10 (and the gripper
     family estimands, gates, root-offset statistics, failure split) on 400 keys; the final decision table takes
     P9/P10 from 400 and P8'/P11 from 200 (section 5 "决策以 400 为准"; only P9, P10 are recomputed).
 K4  Bootstrap two-sided p (decision family P8', P9, P10, P11, OFT only): p = min(1, 2 min(P*(theta <= 0),
     P*(theta >= 0))), null value 0, the block-1 prereg.py convention (the protocol names no null value);
     Holm step-down over the four. For P11 the bootstrapped statistic is the count Delta3 over the drawn task-3 keys.
 K5  P11 premise "ur5e_cg task 3 <= 4/20" and manipulation ">= 15 of 20" are applied as absolute counts on the
     available task-3 keys; a warning is printed if that is not 20. McNemar exact p: two-sided binomial on the
     discordant task-3 pairs, p = min(1, 2 P(Bin(b+c, 1/2) <= min(b, c))). The P11 lines are applied to pi0.5 too
     (section 3 P7: "P11 用 Delta3 的同一判定线").
 K6  fixcheck (4.1, 0.0000 mm). OFT: ref_body_pos vs the OFT SRC dump of the same key, gate = max diff < 0.00005 mm
     (prints as 0.0000). pi0.5: "fix" vs OFT SRC ref_body_pos; the jsonl stores 5 decimals (0.01 mm), so the
     gate is max diff <= 0.005 mm (half the last stored digit), i.e. 0.0000 at the recorded precision (WORKLOG:
     "0.0045 mm 为记录精度"). Object suite: additionally every non-robot joint's first-step qpos[:3] (OFT qpos[0];
     pi0.5 "objq") vs OFT obj_src qpos[0]. The pi0.5 SRC record's objq is taken before the wait steps (native env)
     while every BodyEnv record's objq is after them, so for pi0.5 obj_src only "fix" is compared (objq not
     comparable, printed as such).
 K7  copy_grip_cmd gate (4.1): per-episode flag from the npz align_json / jsonl "align", else the BODY_ALIGN log
     line. A cell fails ("does not count") only if some episode shows 0. Cells designed not to copy (S0 = sham,
     iiwa, ur5e; block 1) and SRC (no transfer) are "not applicable". Old block-2 logs carry no copy field: such a
     cell is "unverifiable (field absent)", not failed; the protocol records (0.4, A.1) that these cells copied.
 K8  t0 alignment gate (4.3 + ruling 1): every episode pos_mm <= 0.01 and rot_deg <= 0.05 from the same align
     source as K7, applied to every transferred cell (SRC has none); the per-cell maximum angle is printed.
     BODY_ALIGN fixture_mm is NOT a residual (align.apply_static_body_pose returns how far fixtures were moved,
     ~14.7 mm in normal runs), so the "fixture 0" part of 4.3 is checked by fixcheck (K6), not by fixture_mm.
 K9  Twin gates. IK twins (T0, G_T, G_Tw, G_Ta, RT2_T, K3_T and the other block-2 twins): OFT query frames are
     twin_*[::8] (index 0 = initial frame); miss = pos > 2 mm or rot > 1 deg; gate < 1%. pi0.5 has no per-query
     rotation, so it uses twin.miss / twin.n_paint over all painted frames (block-2 prereg_block2.py). Sunk twins
     (G_R and the conditional G_Rw, which I treat as a sunk twin with wrist-only paint): pad residual > 1 mm on
     query frames < 1% (OFT twin_pos_err_mm[::8]; pi0.5 twin.pos_err_mm_q, fallback over_1mm / n_paint).
     Stale frames: sum over episodes must be 0. "Wrist camera pose diff 0" is not in the dumps: printed as
     unverifiable here (it was checked in the smoke runs).
 K10 G_R step-0 gate (ruling 2): read $PROJ_ROOT/embody_v3/out/gr_t0/*.jsonl. (i) every record must have agentview
     and wrist mean diff outside the gripper mask < 0.11 (if the record's method is not a segmentation mask, or the
     outside values are absent, the hidden-gripper fallback gr_vs_t0_hidden is used); inside-mask values printed.
     (ii) every record gr_vs_g wrist mean < 8; agentview gr_vs_g, g_vs_ref, gr_vs_ref printed only, with host/GPU.
     (iii) visual check is manual. The gate applies to the G_R cell of both models (it is a rendering check).
     A missing gr_t0 file makes the gate "unverified": the cell is not marked "does not count" but every
     decision using it carries "门槛未核实" (the protocol does not say what a missing check means).
 K11 Root-body offset (ruling 3.1): twin_root_off_mm[::8] (OFT) / root_off_mm_q (pi0.5) = query frames; lateral =
     hypot(dx, dy). Frame k is the observation after k policy steps; contact_first_grip_obj = s is the 0-based step
     during which the gripper first touched a free object, so frames k <= s are "before contact", k > s "after";
     s = -1 means never (all frames before). p95 for the P9 downgrade is over all query frames of all episodes.
     Missing offsets -> downgrade condition (c) "unverifiable".
 K12 P9 one-sided reading (ruling 2): P9 confirmed -> "confirmed (conservative direction)"; P9 falsified and P10
     confirmed -> "falsified"; P9 falsified/indeterminate and P10 falsified/indeterminate -> "未能判定（工具限制）".
     If P10 does not count or is missing the rule cannot be applied and is printed as such. G_Rw trigger: P9 final
     reading is not "confirmed" and P10 is not "confirmed" (a descriptive/does-not-count P9 is "not confirmed").
 K13 Contact-split of G_R's net extra failures vs T0 (ruling 3.1), OFT only: stages.py labels (no_reach = before
     contact; every other failure label = after), and a second version with contact_first_grip_obj (-1 = before).
     "Net" = per-category failures(G_R) - failures(T0) on common keys; share_before = net_before / net_total.
     Also printed: the discordant keys (T0 success, G_R failure) by category. ">= half before" -> appearance
     reading stands; share 0 -> "confirmed (with tool caveat)"; 0 < share < 0.5 is not defined by the ruling and is
     printed as needing a human call.
 K14 conc(X): per task net = failures(X) - failures(S) on common keys; conc = max_t net_t / sum_t net_t (sum over
     all tasks, negatives included); undefined if the sum <= 0.
 K15 R1 is printed for every drop_nat X; only P6j3, K3, RT2, UR5e_nc are "new bodies" and only K3, RT2 trigger X_T.
     Triggers are read from OFT (all estimands are defined on OFT, section 2). R2 uses the point |drop_nat(X)|.
 K16 R7: SR per task from OFT (primary) and pi0.5 (printed, not scored). "SR > 25%" rows are scored only where
     S (sham_cg, same model) has SR >= 75% on that task; "SR <= 50%" rows (c_t >= 0.5) always. Only PandaL3 (P6j3)
     and Kinova3 (K3) rows enter the score (ruling 3.3); others printed as reference. A table whose sha256 differs
     from the frozen value is printed but not scored.
 K17 R10: GPU model = the nvidia-smi name (the a100-40 slurm slice reports "NVIDIA A100 80GB PCIe" like a100-80;
     rendering differences come from hardware). Composition = counts of names over the estimand's keys per cell.
     A difference with |point| <= 3 whose cells' compositions differ is tagged "在渲染节点噪声带内"; if any GPU is
     unknown the tag reads "无法判定". Host/GPU come from the npz/jsonl fields, else the slurm job log of the job
     that ran the key (highest job id that logged it).
 K18 pi0.5: records with a non-empty error are dropped; for duplicate clean records of a key the last one in
     (sorted file, line) order is kept (prereg_block2.py overwrote likewise); both counts are printed.
 K19 "does not count": any FAILED gate. Estimates are still printed; a decision that needs such a cell prints
     "不判定 (does not count)" followed by what the lines would give, marked 仅供参考.
 K20 Object suite expectation (iii) compares each object estimand with the same estimand on spatial for the same
     model (spatial drop_nat(UR5e) = UR5e_cg - S), using the P7 rule (sign agrees and 95% CIs overlap ->
     consistent; sign differs and the object CI excludes 0 -> inconsistent; else indeterminate).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

PROTO_SHA = "3875470456102d60668a02f6a7ff0610faf2b32a15f8112821fff52781690616"
AMEND_SHA = "eee344bef35f20115ef98179fed415c11df464af1493445cdfae28d5d349cd5d"
R7_SHA = "9a353acb7b877aea12bd68aa42939f2cc9db13a64e8b73fb0bafadd515d2ee6b"
# D5 (附录 2 §5): 附录 2 sha256 与本副本的来源
AMEND2_SHA = "3ef9ff455ce9316440f6081213339c1a8c1e7bf3bc4e3c7070414af5e4ec62bf"
FROZEN_SHA = "f1da2c63cbc16ab346e569b59d9dd50cae8504766860640f445a17c5613976cf"
# D1 (附录 2 裁定 1): §4.3 首帧门槛只施于第三批新身体
T0_GATE_CELLS = {"spatial": {"P6j3", "K3", "K3_T", "RT2", "RT2_T", "UR5e_nc"}, "object": {"K3"}}
SEED = 0
H = os.path.expanduser
# sens（Fable 补裁一 §2.3 第 4 条）：a2 原版 sha256；--exclude-keys 读取的组名
A2_SHA = "003032bace8edc99bb032cc32f504318e4d7640bc23e71b974735d2a9bd6eb22"
EXCL_GROUP = "oft_pub|libero_object"


def load_exclude(path):
    """sens: {group: [[t, e], ...]} or {group: {cell: [[t, e], ...]}} -> set of (t, e) keys (union over cells)."""
    d = json.load(open(H(path)))
    if EXCL_GROUP not in d:
        sys.exit(f"--exclude-keys: 文件里没有组 {EXCL_GROUP}（有 {sorted(d)}）")
    v = d[EXCL_GROUP]
    if isinstance(v, dict):
        v = [k for ks in v.values() for k in ks]
    return {(int(k[0]), int(k[1])) for k in v}

# code, OFT dir / pi0.5 prefix, copy_grip_cmd by design (None = no transfer), twin kind
SPATIAL = [
    ("SRC", "src", None, None), ("S", "sham_cg", 1, None), ("S0", "sham", 0, None), ("T0", "sham_twin", 1, "ik"),
    ("IIWA", "iiwa", 0, None), ("IIWA_cg", "iiwa_cg", 1, None), ("IIWA_T", "iiwa_twin", 1, "ik"),
    ("UR5e", "ur5e", 0, None), ("UR5e_cg", "ur5e_cg", 1, None), ("UR5e_T", "ur5e_twin", 1, "ik"),
    ("UR5e_R", "ur5e_rev", 1, "ik"),
    ("G", "grpkp100", 1, None), ("G_T", "grpkp100_twin", 1, "ik"), ("G_Tw", "grpkp100_tw", 1, "ik"),
    ("G_Ta", "grpkp100_ta", 1, "ik"), ("G_R", "g_r", 1, "sunk"), ("G_Rw", "g_rw", 1, "sunk"),
    ("P6j3", "p6j3", 1, None), ("K3", "k3", 1, None), ("UR5e_nc", "ur5e_nc", 1, None), ("RT2", "rt2", 1, None),
    ("RT2_T", "rt2_twin", 1, "ik"), ("K3_T", "k3_twin", 1, "ik"),
]
OBJECT = [("SRC", "obj_src", None, None), ("S", "obj_s", 1, None), ("T0", "obj_t0", 1, "ik"),
          ("UR5e", "obj_ur5e", 1, None), ("K3", "obj_k3", 1, None), ("G", "obj_g", 1, None),
          ("G_T", "obj_g_t", 1, "ik")]
GRIP_FAMILY = ["S", "T0", "G", "G_T", "G_R", "G_Tw", "G_Ta"]  # section 5 (plus SRC for densification)
NEW_BODIES = ["P6j3", "K3", "RT2", "UR5e_nc"]

# estimands: name -> (terms [(coef, cell)], key filter or None, kind)
T3 = lambda k: k[0] == 3
NOT_T3 = lambda k: k[0] != 3
SPATIAL_EST = {}
for _x in ["P6j3", "K3", "RT2", "UR5e_nc", "UR5e_cg", "IIWA_cg", "G"]:
    SPATIAL_EST[f"drop_nat({_x})"] = ([(1, _x), (-1, "S")], None, "sr")
SPATIAL_EST.update({
    "d6": ([(1, "P6j3"), (-1, "S")], None, "sr"),
    "d_Grev": ([(1, "G_R"), (-1, "T0")], None, "sr"),
    "rec_w": ([(1, "G_Tw"), (-1, "G"), (0, "G_Ta")], None, "sr"),
    "rec_a": ([(1, "G_Ta"), (-1, "G"), (0, "G_Tw")], None, "sr"),
    "channel": ([(1, "G_Tw"), (-1, "G_Ta"), (0, "G")], None, "sr"),
    "recovery(G)": ([(1, "G_T"), (-1, "T0"), (-1, "G"), (1, "S")], None, "sr"),
    "recovery(K3)": ([(1, "K3_T"), (-1, "T0"), (-1, "K3"), (1, "S")], None, "sr"),
    "recovery(RT2)": ([(1, "RT2_T"), (-1, "T0"), (-1, "RT2"), (1, "S")], None, "sr"),
    "inter_G": ([(1, "G"), (-1, "G_T"), (-1, "G_R"), (1, "T0")], None, "sr"),
    "Delta3": ([(1, "UR5e_nc"), (-1, "UR5e_cg")], T3, "count"),
    "spec": ([(1, "UR5e_nc"), (-1, "UR5e_cg")], NOT_T3, "sr"),
    "d_cg(sham)": ([(1, "S"), (-1, "S0")], None, "sr"),
    "d_cg(iiwa)": ([(1, "IIWA_cg"), (-1, "IIWA")], None, "sr"),
    "d_cg(ur5e)": ([(1, "UR5e_cg"), (-1, "UR5e")], None, "sr"),
    "d_Grw = SR(G_Rw)-SR(T0)": ([(1, "G_Rw"), (-1, "T0")], None, "sr"),
})
GRIP_EST = ["drop_nat(G)", "d_Grev", "rec_w", "rec_a", "channel", "recovery(G)", "inter_G"]
OBJECT_EST = {
    "drop_nat(UR5e)": ([(1, "UR5e"), (-1, "S")], None, "sr"),
    "drop_nat(K3)": ([(1, "K3"), (-1, "S")], None, "sr"),
    "drop_nat(G)": ([(1, "G"), (-1, "S")], None, "sr"),
    "recovery(G)": ([(1, "G_T"), (-1, "T0"), (-1, "G"), (1, "S")], None, "sr"),
    "d5": ([(1, "T0"), (-1, "S")], None, "sr"),
}
# spatial counterpart of each object estimand (K20)
OBJ_TO_SPATIAL = {"drop_nat(UR5e)": ([(1, "UR5e_cg"), (-1, "S")], None, "sr"),
                  "drop_nat(K3)": SPATIAL_EST["drop_nat(K3)"], "drop_nat(G)": SPATIAL_EST["drop_nat(G)"],
                  "recovery(G)": SPATIAL_EST["recovery(G)"], "d5": ([(1, "T0"), (-1, "S")], None, "sr")}
DECISION_CELLS = {"P8'": ["P6j3", "S"], "P9": ["G_R", "T0"], "P10": ["G_Tw", "G_Ta", "G"], "P11": ["UR5e_nc", "UR5e_cg"]}


# ----------------------------------------------------------------------------------------------- output
class Out:
    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.lines.append(s)


P = Out()


def sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _s(v):
    v = np.asarray(v)
    return str(v.item() if v.shape == () else v)


def fmt(x, nd=1, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------------------------- logs
_RICH_HEAD = re.compile(r"^(?:\d\d/\d\d \[\d\d:\d\d:\d\d\]\s+|\s+)(?:INFO|WARNING|ERROR|DEBUG)\s+\|\s?>>\s?(.*)$")
_RICH_TAIL = re.compile(r"\s+[\w./-]+\.py:\d+\s*$")


def unwrap_rich(lines):
    """Join rich-logging messages that were wrapped over several lines."""
    msgs = []
    for ln in lines:
        ln = ln.rstrip("\n")
        m = _RICH_HEAD.match(ln)
        if m:
            msgs.append(_RICH_TAIL.sub("", m.group(1)).strip())
        elif ln.startswith("          ") and msgs:
            msgs[-1] += " " + _RICH_TAIL.sub("", ln).strip()
        else:
            msgs.append(ln.strip())
    return msgs


def parse_align(s):
    out = {}
    for k in ("pos_mm", "rot_deg", "fixture_mm"):
        m = re.search(k + r"=([-\d.]+)", s)
        if m:
            out[k] = float(m.group(1))
    m = re.search(r"copy_grip_cmd=(\d)", s)
    if m:
        out["copy_grip_cmd"] = int(m.group(1))
    return out


def oft_log_info(slurm_dir, eval_dir, wanted):
    """{dir: {(task, ep): {"job", "host", "gpu", "align"}}} from slurm logs (+ EVAL logs for align)."""
    info = defaultdict(dict)
    for f in glob.glob(os.path.join(slurm_dir, "cond_*.log")):
        mj = re.search(r"_(\d+)\.log$", f)
        if not mj:
            continue
        job = int(mj.group(1))
        try:
            lines = open(f, errors="ignore").readlines()
        except OSError:
            continue
        if not lines:
            continue
        m = re.match(r"node=(\S+) gpu=(.*?) NOTE=(\S+)", lines[0])
        if not m:
            continue
        host, gpu, note = m.group(1), m.group(2).strip(), m.group(3)
        obj = "cond_obj_" in os.path.basename(f) or any("libero_object" in ln for ln in lines[:80])
        d = ("obj_" if obj else "") + note
        if d not in wanted:
            continue
        cur, ran = None, {}
        for msg in unwrap_rich(lines[1:]):
            mi = re.search(r"INIT_ID: task=(\d+) episode=(\d+)", msg)
            if mi:
                cur = (int(mi.group(1)), int(mi.group(2)))
                continue
            if cur is None:
                continue
            if "BODY_ALIGN:" in msg:
                ran.setdefault(cur, {})["align"] = parse_align(msg)
            elif re.search(r"Success: (True|False)", msg):
                ran.setdefault(cur, {})
        for k, v in ran.items():
            old = info[d].get(k)
            if old is None or job > old["job"]:
                info[d][k] = {"job": job, "host": host, "gpu": gpu, "align": v.get("align")}
    # EVAL logs: unwrapped BODY_ALIGN lines; fill align where the slurm log gave none
    for f in sorted(glob.glob(os.path.join(eval_dir, "EVAL-libero_*--*.txt"))):
        m = re.search(r"EVAL-libero_(spatial|object)-.*?--(.+)\.txt$", os.path.basename(f))
        if not m:
            continue
        d = ("obj_" if m.group(1) == "object" else "") + m.group(2)
        if d not in wanted:
            continue
        cur = None
        for ln in open(f, errors="ignore"):
            mi = re.search(r"INIT_ID: task=(\d+) episode=(\d+)", ln)
            if mi:
                cur = (int(mi.group(1)), int(mi.group(2)))
            elif cur is not None and "BODY_ALIGN:" in ln:
                e = info[d].setdefault(cur, {"job": -1, "host": None, "gpu": None, "align": None})
                if not e.get("align"):
                    e["align_eval"] = parse_align(ln)
    return info


def pi05_log_info(log_dir, wanted):
    """{prefix: {(task, ep): {"job", "host", "gpu", "align"}}} from client logs v2_<name>_<job>_e<off>.log."""
    jobs = {}
    for f in glob.glob(os.path.join(log_dir, "*.log")):
        b = os.path.basename(f)
        if b.startswith(("v2_", "server_")):
            continue
        mj = re.search(r"_(\d+)\.log$", b)
        if not mj:
            continue
        try:
            first = open(f, errors="ignore").readline()
        except OSError:
            continue
        m = re.match(r"node=(\S+) gpu=(.*?) (?:NAME|LIST)=", first)
        if m:
            jobs[int(mj.group(1))] = (m.group(1), m.group(2).strip())
    info = defaultdict(dict)
    for f in glob.glob(os.path.join(log_dir, "v2_*_e*.log")):
        m = re.match(r"v2_(.+)_(\d+)_e(\d+)\.log$", os.path.basename(f))
        if not m or m.group(1) not in wanted:
            continue
        name, job = m.group(1), int(m.group(2))
        host, gpu = jobs.get(job, (None, None))
        pend = None
        for ln in open(f, errors="ignore"):
            if ln.startswith("BODY_ALIGN:"):
                pend = parse_align(ln)
            elif ln.startswith("EP {"):
                try:
                    r = json.loads(ln[3:])
                except ValueError:
                    continue
                if not r.get("error"):
                    k = (r["task"], r["episode"])
                    old = info[name].get(k)
                    if old is None or job > old["job"]:
                        info[name][k] = {"job": job, "host": host, "gpu": gpu, "align": pend}
                pend = None
    return info


# ----------------------------------------------------------------------------------------------- loaders
_STAGES = None


def stages_mod(stages_dir):
    global _STAGES
    if _STAGES is None:
        try:
            sys.dont_write_bytecode = True  # never write __pycache__ into $PROJ_ROOT/embody
            sys.path.insert(0, H(stages_dir))
            import stages as st  # noqa
            _STAGES = (st, st.task_objects())
        except Exception as e:  # noqa
            _STAGES = (None, repr(e))
    return _STAGES


def qpos_layout(names):
    ad, a = {}, 0
    for n in names:
        s = 7 if (n.endswith("_joint0") and not n.startswith(("robot0", "gripper0"))) else 1
        ad[n] = (a, s)
        a += s
    return ad, a


def load_oft(path, neps, want_stage, want_objq, stages_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "t*_e*_i*.npz"))):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)\.npz$", f)
        t, e = int(m.group(1)), int(m.group(2))
        if e >= neps:
            continue
        z = np.load(f, allow_pickle=True)
        F = set(z.files)
        r = {"succ": float(bool(z["success"])), "init": int(z["init"])}
        r["host"] = _s(z["host"]) if "host" in F else None
        r["gpu"] = _s(z["gpu"]) if "gpu" in F else None
        r["align"] = None
        if "align_json" in F:
            try:
                r["align"] = json.loads(_s(z["align_json"]))
            except ValueError:
                pass
        c = {}
        for k in F:
            if k.startswith("contact_"):
                v = _s(z[k]) if z[k].dtype.kind in "US" else z[k].item()
                if k.endswith("_pairs") or k == "contact_error":
                    try:
                        v = json.loads(v) if k.endswith("_pairs") else v
                    except ValueError:
                        pass
                c[k[len("contact_"):]] = v
        r["contacts"] = c if ("arm_env_contact_steps" in c) else None
        r["first_grip"] = int(c["first_grip_obj"]) if "first_grip_obj" in c else None
        if "twin_pos_err_mm" in F:
            pe = np.asarray(z["twin_pos_err_mm"], float)
            re_ = np.asarray(z["twin_rot_err_deg"], float)
            r["twin"] = {"pos_q": pe[::8], "rot_q": re_[::8], "stale": int(z["twin_stale"]), "n": len(pe)}
        else:
            r["twin"] = None
        if "twin_root_off_mm" in F:
            ro = np.asarray(z["twin_root_off_mm"], float).reshape(-1, 3)
            r["root_q"] = (np.arange(len(ro))[::8], ro[::8])
        r["ref"] = ([str(x) for x in z["ref_body_names"]], np.asarray(z["ref_body_pos"], float))
        if want_objq:
            names = [str(x) for x in z["joint_names"]]
            ad, tot = qpos_layout(names)
            q0 = np.asarray(z["qpos"][0], float)
            r["objq"] = ({n: q0[a:a + s] for n, (a, s) in ad.items() if not n.startswith(("robot0", "gripper0"))}
                         if tot == len(q0) else None)
        if want_stage:
            st, objs = stages_mod(stages_dir)
            r["stage"] = None
            if st is not None:
                try:
                    s = st.episode_stages(z, *objs[t])
                    r["stage"] = s["label"] if s else None
                except Exception:  # noqa
                    r["stage"] = None
        out[(t, e)] = r
    return out


def load_pi05(v2, name, neps):
    rx = re.compile("^" + re.escape(name) + r"_e(\d+)\.jsonl$")
    files = sorted((int(rx.match(os.path.basename(f)).group(1)), f) for f in glob.glob(os.path.join(v2, name + "_e*.jsonl"))
                   if rx.match(os.path.basename(f)))
    out, n_err, n_dup, n_all = {}, 0, 0, 0
    for _, f in files:
        for ln in open(f):
            if not ln.strip():
                continue
            r = json.loads(ln)
            n_all += 1
            if r.get("error"):
                n_err += 1
                continue
            k = (int(r["task"]), int(r["episode"]))
            if k[1] >= neps:
                continue
            if k in out:
                n_dup += 1
            tw = r.get("twin")
            c = r.get("contacts")
            out[k] = {
                "succ": float(bool(r["success"])), "init": r.get("init"), "host": r.get("host"), "gpu": r.get("gpu"),
                "align": r.get("align"), "contacts": c if (isinstance(c, dict) and "arm_env_contact_steps" in c) else None,
                "first_grip": (int(c["first_grip_obj"]) if isinstance(c, dict) and "first_grip_obj" in c else None),
                "twin": tw, "fix": r.get("fix"), "objq": r.get("objq"),
            }
            if tw and tw.get("root_off_mm_q"):
                ro = np.asarray(tw["root_off_mm_q"], float).reshape(-1, 3)
                out[k]["root_q"] = (np.arange(len(ro)) * 5, ro)
    return out, {"lines": n_all, "error_dropped": n_err, "dup_clean": n_dup, "files": len(files)}


# ----------------------------------------------------------------------------------------------- bootstrap
class Boot:
    def __init__(self, keys, B, seed=SEED):
        self.keys = keys
        n = len(keys)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(B, n))
        self.W = np.bincount((idx + np.arange(B)[:, None] * n).ravel(), minlength=B * n).reshape(B, n).astype(np.float64)
        del idx
        self.task = np.array([k[0] for k in keys])

    def est(self, Y, terms, filt=None, kind="sr"):
        cells = [c for _, c in terms]
        if any(c not in Y for c in cells):
            return None
        m = np.ones(len(self.keys), bool)
        for c in cells:
            m &= ~np.isnan(Y[c])
        if filt is not None:
            m &= np.array([filt(k) for k in self.keys])
        if m.sum() == 0:
            return None
        mf = m.astype(float)
        if kind == "sr":
            point = 100 * sum(a * Y[c][m].mean() for a, c in terms)
            den = self.W @ mf
            with np.errstate(invalid="ignore", divide="ignore"):
                bs = 100 * sum(a * (self.W @ np.where(m, Y[c], 0.0)) / den for a, c in terms if a)
        else:
            point = float(sum(a * Y[c][m].sum() for a, c in terms))
            bs = sum(a * (self.W @ np.where(m, Y[c], 0.0)) for a, c in terms if a)
        bs = np.asarray(bs, float)
        bs = bs[~np.isnan(bs)]
        p0 = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())) if len(bs) else float("nan")
        return {"point": float(point), "ci95": tuple(np.percentile(bs, [2.5, 97.5])),
                "ci90": tuple(np.percentile(bs, [5, 95])), "p0": p0, "n": int(m.sum()), "mask": m, "cells": cells}


def holm(ps):
    names = [k for k in ps if ps[k] is not None and not math.isnan(ps[k])]
    order = sorted(names, key=lambda k: ps[k])
    adj, run = {}, 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * ps[k]))
        adj[k] = run
    return adj


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ----------------------------------------------------------------------------------------------- model analysis
class Model:
    """All data and gates for one model on one suite."""

    def __init__(self, name, kind, cells, neps, a, suite, src_oft=None):
        self.name, self.kind, self.neps, self.suite = name, kind, neps, suite
        self.spec = {c: (d, cp, tw) for c, d, cp, tw in cells}
        self.data, self.meta = {}, {}
        self.keys = [(t, e) for t in range(10) for e in range(neps)]
        wanted = {d for _, d, _, _ in cells}
        if kind == "oft":
            self.loginfo = oft_log_info(H(a.slurm_logs), H(a.eval_logs), wanted)
            for c, (d, _, _) in self.spec.items():
                p = os.path.join(H(a.runs), d)
                if os.path.isdir(p):
                    rec = load_oft(p, neps, want_stage=(suite == "spatial" and c in ("G_R", "T0")),
                                   want_objq=(suite == "object"), stages_dir=a.stages_dir)
                    if rec:
                        self.data[c] = rec
        else:
            self.loginfo = pi05_log_info(H(a.pi05_logs), wanted)
            for c, (d, _, _) in self.spec.items():
                rec, meta = load_pi05(H(a.pi05), d, neps)
                self.meta[c] = meta
                if rec:
                    self.data[c] = rec
        # fill host/gpu/align from logs
        for c, rec in self.data.items():
            li = self.loginfo.get(self.spec[c][0], {})
            for k, r in rec.items():
                e = li.get(k, {})
                r["src_hg"] = "dump"
                if not r.get("gpu") and e.get("gpu"):
                    r["host"], r["gpu"], r["src_hg"] = e.get("host"), e.get("gpu"), "log"
                al = r.get("align") or {}
                if "copy_grip_cmd" not in al or "rot_deg" not in al:
                    la = e.get("align") or e.get("align_eval") or {}
                    al = {**la, **al}
                r["align2"] = al
        self.src_oft = src_oft if src_oft is not None else (self.data.get("SRC") if kind == "oft" else None)
        if kind == "oft" and suite == "object" and getattr(a, "exclude_keys", None):  # sens: 物理失效集剔除，所有格同剔（保配对）
            ex = load_exclude(a.exclude_keys)
            gone = {c: sum(k in ex for k in rec) for c, rec in self.data.items()}
            self.data = {c: {k: r for k, r in rec.items() if k not in ex} for c, rec in self.data.items()}
            P(f"\n==== 物理失效集剔除（{self.name}，object）：{a.exclude_keys}（sha256 {sha256(H(a.exclude_keys))}，组 {EXCL_GROUP}）")
            P(f"  {len(ex)} 键：{sorted(ex)}")
            P("  每格剔除键数：" + "  ".join(f"{c} -{n}" for c, n in gone.items()))
        self.Y = {c: np.array([rec[k]["succ"] if k in rec else np.nan for k in self.keys]) for c, rec in self.data.items()}
        self.gates = {}

    # ------------------------------------------------------------------ gates
    def fixcheck(self, c):
        rec = self.data[c]
        src = self.src_oft
        if src is None:
            return ("UNK", "无 OFT SRC 对照")
        if self.kind == "oft" and c == "SRC":
            return ("NA", "SRC 自身")
        worst, n, nos = 0.0, 0, 0
        objq_note = ""
        for k, r in rec.items():
            if k not in src:
                nos += 1
                continue
            sn, sp = src[k]["ref"]
            sidx = {nm: i for i, nm in enumerate(sn)}
            ds = []
            if self.kind == "oft":
                rn, rp = r["ref"]
                ds += [float(np.abs(rp[i] - sp[sidx[nm]]).max()) * 1000 for i, nm in enumerate(rn) if nm in sidx]
                if self.suite == "object" and r.get("objq") and src[k].get("objq"):
                    so = src[k]["objq"]
                    ds += [float(np.abs(v[:3] - so[j][:3]).max()) * 1000 for j, v in r["objq"].items() if j in so]
            else:
                for nm, v in (r.get("fix") or {}).items():
                    if nm in sidx:
                        ds.append(float(np.abs(np.asarray(v) - sp[sidx[nm]]).max()) * 1000)
                if self.suite == "object":
                    if c == "SRC":
                        objq_note = "；pi0.5 SRC 的 objq 记录在等待步之前，不可比，只比 fix"
                    elif r.get("objq") and src[k].get("objq"):
                        so = src[k]["objq"]
                        ds += [float(np.abs(np.asarray(v[:3]) - so[j][:3]).max()) * 1000 for j, v in r["objq"].items() if j in so]
            if ds:
                worst, n = max(worst, max(ds)), n + 1
        if n == 0:
            return ("UNK", f"无可比键（缺 SRC 键 {nos}）")
        tol = 0.00005 if self.kind == "oft" else 0.005 + 1e-9
        st = "PASS" if worst <= tol else "FAIL"
        return (st, f"keys {n}, 最大差 {worst:.4f} mm（门槛 {'0.0000' if self.kind == 'oft' else '<= 0.005 记录精度'}）"
                    + (f"，{nos} 键无 SRC 对照" if nos else "") + objq_note)

    def copy_gate(self, c):
        design = self.spec[c][1]
        if design is None:
            return ("NA", "无迁移")
        vals = [r["align2"].get("copy_grip_cmd") for r in self.data[c].values()]
        n1, n0, nm = sum(v == 1 for v in vals), sum(v == 0 for v in vals), sum(v is None for v in vals)
        det = f"copy_grip_cmd=1: {n1}, =0: {n0}, 缺: {nm}"
        if design == 0:
            return ("NA", det + "（第一批设计上不复制，只入 d_cg 描述）")
        if n0:
            return ("FAIL", det)
        if nm == len(vals):
            return ("UNK", det + "（日志无此字段，无法核对；协议 0.4/A.1 记载此格复制）")
        return ("PASS", det)

    def align_gate(self, c):
        if self.spec[c][1] is None:
            return ("NA", "无迁移")
        pos = [r["align2"].get("pos_mm") for r in self.data[c].values()]
        rot = [r["align2"].get("rot_deg") for r in self.data[c].values()]
        pv, rv = [x for x in pos if x is not None], [x for x in rot if x is not None]
        if not rv:
            return ("UNK", "无对齐记录")
        bad = sum((p is not None and p > 0.01) or (r is not None and r > 0.05) for p, r in zip(pos, rot))
        det = f"最大 {max(pv) if pv else float('nan'):.4f} mm / {max(rv):.4f}°（门槛 0.01 mm / 0.05°），超门槛 {bad} 集，缺 {len(pos) - len(rv)}"
        if c not in T0_GATE_CELLS[self.suite]:  # D1
            return ("NA", det + "（身体于第一/二批接受，§4.3 新身体门槛不适用；附录 2 裁定 1）")
        return ("FAIL" if bad else "PASS", det)

    def twin_gate(self, c):
        kind = self.spec[c][2]
        if kind is None:
            return None
        recs = [r for r in self.data[c].values() if r.get("twin")]
        if not recs:
            return ("UNK", "无替身记录"), ("UNK", "无替身记录")
        stale = 0
        if self.kind == "oft":
            pe = np.concatenate([r["twin"]["pos_q"] for r in recs])
            re_ = np.concatenate([r["twin"]["rot_q"] for r in recs])
            stale = sum(r["twin"]["stale"] for r in recs)
            if kind == "ik":
                bad, n, what = int(((pe > 2) | (re_ > 1)).sum()), len(pe), "查询帧 IK 未命中（pos>2 mm 或 rot>1°）"
            else:
                bad, n, what = int((pe > 1).sum()), len(pe), "查询帧指垫残差 > 1 mm"
        else:
            stale = sum(int(r["twin"].get("stale") or 0) for r in recs)
            if kind == "ik":
                bad = sum(int(r["twin"].get("miss") or 0) for r in recs)
                n, what = sum(int(r["twin"].get("n_paint") or 0) for r in recs), "全部帧 IK 未命中（miss/n_paint）"
            else:
                q = [r["twin"].get("pos_err_mm_q") for r in recs]
                if all(x is not None for x in q):
                    pe = np.concatenate([np.asarray(x, float) for x in q])
                    bad, n, what = int((pe > 1).sum()), len(pe), "查询帧指垫残差 > 1 mm"
                else:
                    bad = sum(int(r["twin"].get("over_1mm") or 0) for r in recs)
                    n, what = sum(int(r["twin"].get("n_paint") or 0) for r in recs), "全部帧指垫残差 > 1 mm（缺查询帧序列）"
        frac = 100.0 * bad / n if n else float("nan")
        g1 = ("PASS" if n and frac < 1 else ("FAIL" if n else "UNK"), f"{what} {bad}/{n} = {frac:.2f}%（门槛 < 1%），{len(recs)} 集")
        g2 = ("PASS" if stale == 0 else "FAIL", f"陈旧帧 {stale}")
        return g1, g2

    def compute_gates(self, gr_t0):
        for c in self.data:
            g = {}
            g["fixcheck"] = self.fixcheck(c)
            g["copy_grip_cmd"] = self.copy_gate(c)
            g["t0 对齐"] = self.align_gate(c)
            tg = self.twin_gate(c)
            if tg:
                g["替身残差"], g["陈旧帧"] = tg
                if self.spec[c][2] == "sunk":
                    g["腕相机位姿差 0"] = ("UNK", "dump 无此字段，以冒烟记录为准（不作 does-not-count 依据）")
            if c == "G_R" and self.suite == "spatial":
                g["G_R 第 0 步 (i)"], g["G_R 第 0 步 (ii)"] = gr_t0["i"], gr_t0["ii"]
            self.gates[c] = g

    def status(self, c):
        if c not in self.data:
            return "MISSING"
        g = self.gates.get(c, {})
        hard = {k: v for k, v in g.items() if k != "腕相机位姿差 0"}
        if any(v[0] == "FAIL" for v in hard.values()):
            return "FAIL"
        if any(v[0] == "UNK" for k, v in hard.items() if k != "copy_grip_cmd" or self.spec[c][1] == 1):
            return "UNVERIFIED"
        return "OK"

    def print_gates(self):
        P(f"\n==== 门槛（{self.name}，{self.suite}，每格 {self.neps * 10} 键设计）")
        for c, (d, _, _) in self.spec.items():
            if c not in self.data:
                P(f"  {c:8s} ({d}): 缺（无数据）")
                continue
            n = len(self.data[c])
            extra = ""
            if self.kind == "pi05":
                m = self.meta[c]
                extra = f"；jsonl {m['files']} 个文件 {m['lines']} 行，丢 error {m['error_dropped']}，重复正常记录 {m['dup_clean']}"
            st = self.status(c)
            tag = {"FAIL": "  <<< does not count", "UNVERIFIED": "  （有门槛无法核实）"}.get(st, "")
            P(f"  {c:8s} ({d}): 键 {n}/{self.neps * 10}（缺 {self.neps * 10 - n}）{extra}{tag}")
            for gk, (s, det) in self.gates[c].items():
                P(f"      {gk:16s} {s:5s} {det}")

    # ------------------------------------------------------------------ descriptive tables
    def gpu_comp(self, c, keys=None):
        rec = self.data.get(c, {})
        ks = keys if keys is not None else list(rec)
        return Counter((rec[k].get("gpu") or "?") for k in ks if k in rec)

    def per_task(self, c):
        rec = self.data.get(c, {})
        out = {}
        for t in range(10):
            v = [r["succ"] for k, r in rec.items() if k[0] == t]
            out[t] = (int(sum(v)), len(v))
        return out

    def conc(self, c, ref="S"):
        if c not in self.data or ref not in self.data:
            return None
        a, b = self.data[c], self.data[ref]
        net = {t: 0 for t in range(10)}
        for k in set(a) & set(b):
            net[k[0]] += (1 - a[k]["succ"]) - (1 - b[k]["succ"])
        tot = sum(net.values())
        if tot <= 0:
            return f"无净增失败（净 {tot:+.0f}）"
        w = max(net, key=lambda t: net[t])
        return f"{net[w] / tot:.2f}（最差任务 {w}，{net[w]:+.0f}/{tot:+.0f}）"

    def r8(self):
        P(f"\n==== R8 按任务表 + conc(X)（相对 S）+ R10 GPU 型号组成（{self.name}，{self.suite}）")
        for c, (d, _, _) in self.spec.items():
            if c not in self.data:
                P(f"  {c:8s} 缺")
                continue
            pt = self.per_task(c)
            k, n = sum(v[0] for v in pt.values()), sum(v[1] for v in pt.values())
            comp = ", ".join(f"{g} x{m}" for g, m in sorted(self.gpu_comp(c).items()))
            cc = self.conc(c) if c not in ("S", "SRC") else "-"
            P(f"  {c:8s} SR {100 * k / max(n, 1):5.1f}% ({k}/{n}) | " + " ".join(f"t{t}:{a}/{b}" for t, (a, b) in pt.items())
              + f" | conc {cc} | GPU {comp}")

    def contacts(self):
        P(f"\n==== 在线接触汇总（§8 第 6 条；{self.name}，{self.suite}）：每任务 接触集数/有记录集数，首次接触中位步，pairs 前三（按出现集数），停滞集数")
        for c, (d, _, _) in self.spec.items():
            if c not in self.data:
                continue
            rec = self.data[c]
            nmiss = sum(1 for r in rec.values() if not r.get("contacts"))
            row = []
            detail = []
            for t in range(10):
                rs = [r["contacts"] for k, r in rec.items() if k[0] == t and r.get("contacts")]
                ce = [x for x in rs if int(x.get("arm_env_contact_steps") or 0) > 0]
                stall = sum(1 for x in rs if int(x.get("stall_steps") or 0) > 0)
                row.append(f"t{t}:{len(ce)}/{len(rs)}" + (f"s{stall}" if stall else ""))
                if ce or stall:
                    fc = [int(x["first_contact"]) for x in ce if int(x.get("first_contact", -1)) >= 0]
                    pc = Counter()
                    for x in ce:
                        prs = x.get("pairs") or {}
                        if isinstance(prs, str):
                            try:
                                prs = json.loads(prs)
                            except ValueError:
                                prs = {}
                        pc.update(prs.keys())
                    detail.append(f"      t{t}: 接触 {len(ce)} 集，首次接触中位步 {np.median(fc) if fc else float('nan'):.0f}，"
                                  f"pairs 前三 {pc.most_common(3)}，停滞 {stall} 集")
            P(f"  {c:8s} " + " ".join(row) + (f" | 缺接触字段 {nmiss} 集" if nmiss else ""))
            for ln in detail:
                P(ln)
        if "UR5e_nc" in self.data:
            rec = self.data["UR5e_nc"]
            off = Counter(str((r.get("contacts") or {}).get("forearm_geoms_off")) for r in rec.values())
            pen3 = sum(1 for k, r in rec.items() if k[0] == 3 and int((r.get("contacts") or {}).get("forearm_pen_steps") or 0) > 0)
            pen_o = sum(1 for k, r in rec.items() if k[0] != 3 and int((r.get("contacts") or {}).get("forearm_pen_steps") or 0) > 0)
            mx = max([float((r.get("contacts") or {}).get("forearm_pen_max_mm") or 0) for r in rec.values()] or [0])
            P(f"  UR5e_nc 小臂：forearm_geoms_off 分布 {dict(off)}；穿透集 任务 3 {pen3}，任务 3 以外 {pen_o}；最深 {mx:.1f} mm")


# ----------------------------------------------------------------------------------------------- G_R specific
def _r4(v):
    return "缺" if v is None else f"{float(v):.4f}"


def gr_t0_gate(path):
    files = sorted(glob.glob(os.path.join(H(path), "*.jsonl")))
    recs = []
    for f in files:
        for ln in open(f):
            if ln.strip():
                recs.append(json.loads(ln))
    if not recs:
        return {"i": ("UNK", f"缺 {path}/*.jsonl"), "ii": ("UNK", "缺"), "lines": []}
    lines = []
    ok_i, ok_ii, used = True, True, Counter()
    mx = defaultdict(float)
    # D2 (附录 2 裁定 2): (i) = 隐藏夹爪口径，agentview 与腕部各自按格平均（全部记录合并、且按 host/gpu 分作业）< 0.11
    hid_all = defaultdict(list)
    hid_job = defaultdict(lambda: defaultdict(list))
    seg_all, inside_all, t0ref_all = defaultdict(list), defaultdict(list), defaultdict(list)
    hid_missing = 0
    for r in recs:
        hid = r.get("gr_vs_t0_hidden") or {}
        a = r.get("gr_vs_t0") or {}
        job = f"{r.get('host')}/{r.get('gpu')}"
        if hid.get("agent_mean") is None or hid.get("wrist_mean") is None:
            hid_missing += 1
        else:
            for cam in ("agent", "wrist"):
                hid_all[cam].append(float(hid[cam + "_mean"]))
                hid_job[job][cam].append(float(hid[cam + "_mean"]))
        for cam in ("agent", "wrist"):
            if a.get(cam + "_mean_outside_mask") is not None:
                seg_all[cam].append(float(a[cam + "_mean_outside_mask"]))
            if a.get(cam + "_mean_inside_mask") is not None:
                inside_all[cam].append(float(a[cam + "_mean_inside_mask"]))
            t0r = r.get("t0_vs_ref") or {}
            if t0r.get(cam + "_mean") is not None:
                t0ref_all[cam].append(float(t0r[cam + "_mean"]))
    for r in recs:
        a = r.get("gr_vs_t0") or {}
        hid = r.get("gr_vs_t0_hidden") or {}
        meth = str(a.get("method", ""))
        if a.get("agent_mean_outside_mask") is not None and a.get("wrist_mean_outside_mask") is not None and "hid" not in meth.lower():
            ag, wr, src = a["agent_mean_outside_mask"], a["wrist_mean_outside_mask"], "mask:" + meth
        elif hid.get("agent_mean") is not None:
            ag, wr, src = hid["agent_mean"], hid.get("wrist_mean"), "hidden"
        else:
            ag, wr, src = None, None, "none"
        used[src] += 1
        if ag is None or wr is None or not (ag < 0.11 and wr < 0.11):
            ok_i = False
        g = r.get("gr_vs_g") or {}
        if g.get("wrist_mean") is None or not g["wrist_mean"] < 8:
            ok_ii = False
        for nm, v in [("i_agent", ag), ("i_wrist", wr), ("inside_agent", a.get("agent_mean_inside_mask")),
                      ("inside_wrist", a.get("wrist_mean_inside_mask")), ("gr_vs_g_wrist", g.get("wrist_mean")),
                      ("gr_vs_g_agent", g.get("agent_mean"))]:
            if v is not None:
                mx[nm] = max(mx[nm], float(v))
        lines.append(f"      t{r.get('task')} e{r.get('episode')} host={r.get('host')} gpu={r.get('gpu')} "
                     f"(i) agent {_r4(ag)} wrist {_r4(wr)} [{src}] inside agent {a.get('agent_mean_inside_mask')} wrist {a.get('wrist_mean_inside_mask')} "
                     f"mask_px {a.get('mask_px_agent')}/{a.get('mask_px_wrist')} | (ii) G_R-G wrist {g.get('wrist_mean')} agent {g.get('agent_mean')} "
                     f"| G-ref agent {(r.get('g_vs_ref') or {}).get('agent_mean')} | G_R-ref agent {(r.get('gr_vs_ref') or {}).get('agent_mean')}")
    hosts = Counter(f"{r.get('host')}/{r.get('gpu')}" for r in recs)

    def _d(v):
        v = np.asarray(v, float)
        return f"均值 {v.mean():.4f} 最大 {v.max():.4f} ≥0.11 键 {int((v >= 0.11).sum())}/{len(v)}" if len(v) else "缺"
    means = {cam: float(np.mean(hid_all[cam])) for cam in ("agent", "wrist") if hid_all[cam]}
    job_means = {j: {cam: float(np.mean(v[cam])) for cam in ("agent", "wrist") if v[cam]} for j, v in hid_job.items()}
    if hid_missing:
        st_i = "UNK"
    else:
        allm = list(means.values()) + [x for jm in job_means.values() for x in jm.values()]
        st_i = "PASS" if allm and all(x < 0.11 for x in allm) else "FAIL"
    jm_txt = "；".join(f"{j}（{len(hid_job[j]['agent'])} 键）agent {m_.get('agent', float('nan')):.4f} wrist {m_.get('wrist', float('nan')):.4f}"
                      for j, m_ in sorted(job_means.items()))
    i = (st_i, f"隐藏夹爪口径按格平均（附录 2 裁定 2，门槛 < 0.11）：全部 {len(recs)} 个初始状态 agent {means.get('agent', float('nan')):.4f} "
               f"wrist {means.get('wrist', float('nan')):.4f}；分作业 {jm_txt}；缺隐藏字段 {hid_missing}\n"
               f"        描述（不设门槛）：隐藏口径逐键 agent {_d(hid_all['agent'])}，wrist {_d(hid_all['wrist'])}\n"
               f"        描述：分割口径 M 外 agent {_d(seg_all['agent'])}，wrist {_d(seg_all['wrist'])}；M 内均值 agent "
               f"{np.mean(inside_all['agent']) if inside_all['agent'] else float('nan'):.3f} wrist {np.mean(inside_all['wrist']) if inside_all['wrist'] else float('nan'):.3f}\n"
               f"        噪声参照：同作业 T0 对参考 Panda agent {_d(t0ref_all['agent'])}，wrist {_d(t0ref_all['wrist'])}\n"
               f"        （冻结版逐键口径：{'过' if ok_i else '不过'}，M 外最大 agent {mx['i_agent']:.4f} wrist {mx['i_wrist']:.4f}，口径 {dict(used)}）；节点/GPU {dict(hosts)}")
    ii = ("PASS" if ok_ii else "FAIL", f"G_R 对 G 腕部最大 {mx['gr_vs_g_wrist']:.3f}（门槛 < 8）；agentview 只报数，最大 {mx['gr_vs_g_agent']:.3f}")
    return {"i": i, "ii": ii, "lines": lines}


def root_offset(m, c="G_R"):
    """Ruling 3.1 statistics. Returns (lines, p95_or_None, alert)."""
    if c not in m.data:
        return [f"  {c} 缺"], None, None
    rec = m.data[c]
    have = [r for r in rec.values() if "root_q" in r]
    out = []
    if not have:
        out.append(f"  {c}: 无根 body 偏移记录（twin_root_off_mm / root_off_mm_q 缺），降级条件 (c) 无法核实")
        return out, None, None
    buckets = defaultdict(list)
    axial = []
    nfg = 0
    for r in have:
        fr, ro = r["root_q"]
        lat = np.hypot(ro[:, 0], ro[:, 1])
        axial.append(ro[:, 2])
        s = r.get("first_grip")
        for f_, l_ in zip(fr, lat):
            buckets["全部"].append(l_)
            if s is None:
                buckets["未知（缺 first_grip_obj）"].append(l_)
            elif s < 0 or f_ <= s:
                buckets["接触前"].append(l_)
            else:
                buckets["接触后"].append(l_)
        nfg += s is not None
    ax = np.concatenate(axial)
    out.append(f"  {c}: {len(have)}/{len(rec)} 集有偏移记录，{nfg} 集有 first_grip_obj；轴向 dz 中位 {np.median(ax):.2f} mm，范围 [{ax.min():.2f}, {ax.max():.2f}]")
    for b in ["全部", "接触前", "接触后", "未知（缺 first_grip_obj）"]:
        v = np.asarray(buckets.get(b, []), float)
        if len(v):
            out.append(f"      横向 {b:6s} 帧 {len(v):5d}  中位 {np.median(v):.3f}  p95 {np.percentile(v, 95):.3f}  最大 {v.max():.3f} mm  "
                       f"> 2 mm 比例 {100 * (v > 2).mean():.2f}%")
    p95 = float(np.percentile(buckets["全部"], 95))
    pre = np.asarray(buckets.get("接触前", []), float)
    # D3 (附录 2 裁定 3): 接触前（自由物体口径）> 1 mm 的帧只印数与任务分布；警报改为逐帧恒等式
    #   横向偏移 = |q1 + q2| / 2（Panda 两指关节值，mm）。OFT 字段：npz "gripper_qpos"（每步 2 维，m），
    #   第 k 个查询帧（策略走过 k 步后的观测）对 gripper_qpos[k]；最后一帧（k = 步数）无对应行，跳过。π0.5 jsonl 无手指关节值。
    pre_tasks = Counter()
    for (t_, e_), r in rec.items():
        if "root_q" not in r:
            continue
        fr, ro = r["root_q"]
        lat = np.hypot(ro[:, 0], ro[:, 1])
        s = r.get("first_grip")
        if s is not None:
            pre_tasks[t_] += int(sum(1 for f_, l_ in zip(fr, lat) if (s < 0 or f_ <= s) and l_ > 1))
    npre = int((pre > 1).sum()) if len(pre) else 0
    out.append(f"  接触前（自由物体口径）横向 > 1 mm 的查询帧 {npre}" + (f"（最大 {pre.max():.2f} mm），任务分布 {dict(sorted((k, v) for k, v in pre_tasks.items() if v))}；"
               "附录 2 裁定 3：固定体推指，不作警报" if npre else ""))
    alert = None
    if m.kind == "oft":
        _ap = argparse.ArgumentParser(add_help=False)
        _ap.add_argument("--runs", default="$PROJ_ROOT/runs")
        runs = H(_ap.parse_known_args()[0].runs)
        dev, nfr, nep = 0.0, 0, 0
        for (t_, e_), r in rec.items():
            if "root_q" not in r:
                continue
            fs = glob.glob(os.path.join(runs, m.spec[c][0], f"t{t_}_e{e_}_i*.npz"))
            if not fs:
                continue
            z = np.load(fs[0], allow_pickle=True)
            if "gripper_qpos" not in z.files:
                continue
            gq = np.asarray(z["gripper_qpos"], float)
            fr, ro = r["root_q"]
            lat = np.hypot(ro[:, 0], ro[:, 1])
            ok = fr < len(gq)
            asym = np.abs(gq[fr[ok], 0] + gq[fr[ok], 1]) * 1000 / 2
            if ok.any():
                dev = max(dev, float(np.abs(lat[ok] - asym).max()))
            nfr += int(ok.sum())
            nep += 1
        if nfr:
            alert = dev > 0.5
            out.append(f"  恒等式核对（字段 gripper_qpos）：{nep} 集 {nfr} 个查询帧，max |横向 − |q1+q2|/2| = {dev:.3f} mm（门槛 ≤ 0.5）"
                       + ("  !!!!!!!! 停下查：实现有别的问题 !!!!!!!!" if alert else "，实现与设计一致"))
        else:
            out.append("  恒等式核对：dump 无 gripper_qpos，无法核对")
    else:
        out.append("  恒等式核对：π0.5 jsonl 无手指关节值；同一代码路径，以 OFT 核对为准")
    out.append(f"  降级线：横向 p95 = {p95:.3f} mm {'> 6.4 -> P9 降为 descriptive' if p95 > 6.4 else '<= 6.4，不降级'}")
    return out, p95, alert


def failure_split(m):
    """Ruling 3.1 contact split of G_R's net extra failures vs T0 (OFT spatial)."""
    out = []
    if "G_R" not in m.data or "T0" not in m.data:
        return ["  缺 G_R 或 T0"], None
    a, b = m.data["G_R"], m.data["T0"]
    keys = sorted(set(a) & set(b))
    res = {}
    for ver, fn in [("stages.py（no_reach=接触前，其余失败=接触后）",
                     lambda r: None if r.get("stage") is None else ("pre" if r["stage"] == "no_reach" else "post")),
                    ("contact_first_grip_obj（-1=接触前）",
                     lambda r: None if r.get("first_grip") is None else ("pre" if r["first_grip"] < 0 else "post"))]:
        cnt = {"G_R": Counter(), "T0": Counter()}
        disc = Counter()
        for k in keys:
            for nm, rr in (("G_R", a[k]), ("T0", b[k])):
                if rr["succ"] < 1:
                    cnt[nm][fn(rr) or "unknown"] += 1
            if b[k]["succ"] == 1 and a[k]["succ"] < 1:
                disc[fn(a[k]) or "unknown"] += 1
        net = {c: cnt["G_R"][c] - cnt["T0"][c] for c in ("pre", "post", "unknown")}
        tot = net["pre"] + net["post"]
        n_unk = cnt["G_R"]["unknown"] + cnt["T0"]["unknown"]
        share = net["pre"] / tot if (tot > 0 and n_unk == 0) else None
        out.append(f"  [{ver}] 共同键 {len(keys)}：G_R 失败 {dict(cnt['G_R'])}，T0 失败 {dict(cnt['T0'])}；净增 接触前 {net['pre']:+d} 接触后 {net['post']:+d}"
                   f"（未知 {net['unknown']:+d}）；配对（T0 成、G_R 败）{dict(disc)}")
        if n_unk:
            out.append(f"      有 {n_unk} 个失败集无法按此口径分类（如 T0 来自第二批、无 contact_first_grip_obj）：份额不计算")
        elif share is None:
            out.append("      净增失败 <= 0：份额无定义")
        else:
            if share >= 0.5:
                v = "至少一半在接触前 -> 伪影不能解释丢分，P9 的「外观」解释成立"
            elif share <= 0:
                v = "全在接触后 -> P9 写成「确认（含工具警示）」，接触帧本体滑动列为并列候选解释"
            else:
                v = "接触前份额在 0 与 1/2 之间：裁定未定量「几乎全在接触后」，需人工判断是否加工具警示"
            out.append(f"      接触前份额 {share:.2f}：{v}")
        res[ver] = share
    return out, res


# ----------------------------------------------------------------------------------------------- printing estimands
def run_estimands(m, boot, table):
    res = {}
    for nm, (terms, filt, kind) in table.items():
        r = boot.est(m.Y, terms, filt, kind)
        res[nm] = r
    return res


def r10_tag(m, r):
    if r is None or abs(r["point"]) > 3:
        return ""
    keys = [k for k, ok in zip(m.keys, r["mask"]) if ok]
    comps = [m.gpu_comp(c, keys) for c in dict.fromkeys(r["cells"])]
    if any("?" in cp for cp in comps):
        return "  [R10: |效应|<=3，GPU 组成有缺，无法判定]"
    if any(cp != comps[0] for cp in comps[1:]):
        return "  [R10: 在渲染节点噪声带内]"
    return ""


def bad_cells(m, cells):
    st = {c: m.status(c) for c in dict.fromkeys(cells)}
    return [c for c, s in st.items() if s == "FAIL"], [c for c, s in st.items() if s == "UNVERIFIED"], [c for c, s in st.items() if s == "MISSING"]


def print_estimands(m, res, title):
    P(f"\n==== {title}（{m.name}）：点估计、95% / 90% bootstrap CI、键数")
    for nm, r in res.items():
        if r is None:
            terms = (SPATIAL_EST.get(nm) or OBJECT_EST.get(nm) or ([], None, None))[0]
            miss = [c for _, c in terms if c not in m.data]
            P(f"  {nm:26s} 缺（无数据格：{', '.join(miss) or '无共同键'}）")
            continue
        f, u, _ = bad_cells(m, r["cells"])
        unit = "集" if nm == "Delta3" else "pp"
        tag = (f"  <<< does not count（{','.join(f)}）" if f else "") + (f"  （门槛未核实：{','.join(u)}）" if u else "")
        P(f"  {nm:26s} {r['point']:+7.2f} {unit}  95% [{r['ci95'][0]:+7.2f}, {r['ci95'][1]:+7.2f}]  90% [{r['ci90'][0]:+7.2f}, {r['ci90'][1]:+7.2f}]"
          f"  n={r['n']}{tag}{r10_tag(m, r)}")


def verdict_ci(r, line, direction):
    """direction 'below': confirmed if hi < line, falsified if lo > line."""
    lo, hi = r["ci95"]
    if direction == "below":
        return "确认" if hi < line else ("证伪" if lo > line else "不确定")
    return "确认" if lo > line else ("证伪" if hi < line else "不确定")


def gated(m, cells, v):
    f, u, miss = bad_cells(m, cells)
    if miss:
        return "缺", f"缺格 {miss}"
    if f:
        return "does not count", f"不判定（does not count：{','.join(f)} 门槛不过）；按判定线本应为「{v}」，仅供参考"
    if u:
        return v, f"{v}（门槛未核实：{','.join(u)}）"
    return v, v


def decide(m, res, p95=None, tag=""):
    """Section 3 decisions for one model; returns {Pk: verdict}."""
    out = {}
    P(f"\n==== 决策 §3{tag}（{m.name}）")
    r = res.get("d6")
    if r:
        v, txt = gated(m, DECISION_CELLS["P8'"], verdict_ci(r, -7.0, "below"))
        out["P8'"] = v
        P(f"  P8'（自由度）d6 {r['point']:+.2f} 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] 线 -7 -> {txt}")
        P("        辅助（描述）：P6j3 闭环回放 76.8%、跟踪误差 0.92 mm（协议 §3 原文数字）")
    else:
        out["P8'"] = "缺"
        P("  P8' 缺")
    r = res.get("d_Grev")
    if r:
        raw = verdict_ci(r, -21.75, "below")
        v, txt = gated(m, DECISION_CELLS["P9"], raw)
        if v not in ("does not count", "缺"):
            if p95 is None:
                txt += "（降级条件 (c) 横向偏移 p95 无法核实）"
            elif p95 > 6.4:
                v, txt = "descriptive", f"descriptive（横向偏移 p95 {p95:.2f} > 6.4 mm，P9 降为描述性）；按判定线为「{raw}」，仅供参考"
        out["P9_raw"], out["P9"] = raw, v
        P(f"  P9（夹爪外观）d_Grev {r['point']:+.2f} 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] 线 -21.75 -> {txt}")
        ri = res.get("inter_G")
        if ri:
            P(f"        inter_G（只报 CI）{ri['point']:+.2f} 95% [{ri['ci95'][0]:+.2f}, {ri['ci95'][1]:+.2f}]")
    else:
        out["P9"] = "缺"
        P("  P9 缺")
    rc, rw = res.get("channel"), res.get("rec_w")
    if rc and rw:
        lo, hi = rc["ci95"]
        raw = "确认" if (lo > 0 and rw["point"] >= 17.25) else ("证伪" if hi < 0 else "不确定")
        v, txt = gated(m, DECISION_CELLS["P10"], raw)
        out["P10"] = v
        P(f"  P10（相机通道）channel {rc['point']:+.2f} 95% [{lo:+.2f}, {hi:+.2f}]，rec_w 点 {rw['point']:+.2f}（需 >= 17.25）-> {txt}")
    else:
        out["P10"] = "缺"
        P("  P10 缺")
    # P11
    out["P11"] = p11(m, res)
    return out


def p11(m, res):
    r = res.get("Delta3")
    if r is None or "UR5e_nc" not in m.data or "UR5e_cg" not in m.data:
        P("  P11 缺（UR5e_nc 或 UR5e_cg 无数据）")
        return "缺"
    a, b = m.data["UR5e_nc"], m.data["UR5e_cg"]
    k3 = sorted(k for k in set(a) & set(b) if k[0] == 3)
    n3 = len(k3)
    cg3 = int(sum(b[k]["succ"] for k in k3))
    nc3 = int(sum(a[k]["succ"] for k in k3))
    pen = sum(1 for k, rr in a.items() if k[0] == 3 and int((rr.get("contacts") or {}).get("forearm_pen_steps") or 0) > 0)
    pen_known = sum(1 for k, rr in a.items() if k[0] == 3 and rr.get("contacts") and "forearm_pen_steps" in rr["contacts"])
    bb = sum(1 for k in k3 if a[k]["succ"] == 1 and b[k]["succ"] == 0)
    cc = sum(1 for k in k3 if a[k]["succ"] == 0 and b[k]["succ"] == 1)
    pm = mcnemar_exact(bb, cc)
    warn = "" if n3 == 20 else f"（警告：任务 3 配对键 {n3} != 20）"
    P(f"  P11（碰撞反事实）任务 3 配对键 {n3}{warn}：UR5e_cg {cg3}/{n3}，UR5e_nc {nc3}/{n3}，Delta3 = {r['point']:+.0f}"
      f"  bootstrap 95% [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]；McNemar 精确 p = {pm:.4g}（不一致对 {bb} 正 / {cc} 负）")
    prem = cg3 <= 4
    manip = pen >= 15
    P(f"        前提检查 UR5e_cg 任务 3 成功 <= 4：{cg3} -> {'复现' if prem else '前提未复现'}")
    P(f"        操纵检查 UR5e_nc 任务 3 小臂穿透 >= 15 集：{pen}（有穿透字段 {pen_known} 集）-> {'生效' if manip else '操纵未生效'}")
    raw = "确认" if r["point"] >= 10 else ("证伪" if r["point"] <= 4 else "不确定")
    if not manip:
        v = "操纵未生效，不判定"
        P(f"        -> {v}（按判定线为「{raw}」，仅供参考）")
        return v
    if not prem:
        v = "前提未复现，只作描述"
        P(f"        -> {v}（按判定线为「{raw}」，仅供参考）")
        return v
    v, txt = gated(m, DECISION_CELLS["P11"], raw)
    P(f"        判定（+10 确认 / +4 证伪）-> {txt}")
    return v


def holm_table(entries, title):
    """entries: {Pk: (estimand name, result)}"""
    ps = {k: (r["p0"] if r else None) for k, (_, r) in entries.items()}
    adj = holm(ps)
    P(f"\n==== {title}：bootstrap 双侧 p（H0 = 0，见口径 K4）与 Holm 校正（族 P8'、P9、P10、P11，OFT）")
    for k, (nm, r) in entries.items():
        if r is None:
            P(f"  {k:4s} {nm:10s} 缺")
        else:
            P(f"  {k:4s} {nm:10s} 点 {r['point']:+7.2f}  p = {r['p0']:.4f}  Holm p = {adj.get(k, float('nan')):.4f}")


def p7(oft_res, pi_res, names, title="P7 复现（π0.5 对 OFT）", a_lbl="OFT", b_lbl="π0.5"):
    P(f"\n==== {title}：符号一致且 95% CI 重叠 = 复现；符号相反且 {b_lbl} CI 不含 0 = 不复现；其余 不确定")
    out = {}
    for nm in names:
        a, b = oft_res.get(nm), pi_res.get(nm)
        if not a or not b:
            P(f"  {nm:26s} 缺")
            continue
        (l1, h1), (l2, h2) = a["ci95"], b["ci95"]
        if a["point"] == 0 or b["point"] == 0:  # D4 (附录 2 §4 第 3 条): 点估计恰为 0 没有符号
            v = "不确定"
        elif np.sign(a["point"]) == np.sign(b["point"]) and not (h1 < l2 or h2 < l1):
            v = "复现"
        elif np.sign(a["point"]) != np.sign(b["point"]) and (l2 > 0 or h2 < 0):
            v = "不复现"
        else:
            v = "不确定"
        out[nm] = v
        P(f"  {nm:26s} {a_lbl} {a['point']:+7.2f} [{l1:+.2f}, {h1:+.2f}]  {b_lbl} {b['point']:+7.2f} [{l2:+.2f}, {h2:+.2f}] -> {v}")
    return out


def r1_r2(m, res):
    P(f"\n==== R1 新身体分类 / R2 L 类归因（{m.name}）")
    cls = {}
    for x in ["P6j3", "K3", "RT2", "UR5e_nc", "UR5e_cg", "IIWA_cg", "G"]:
        r = res.get(f"drop_nat({x})")
        if r is None:
            cls[x] = "缺"
            P(f"  R1 {x:8s} 缺")
            continue
        lo, hi = r["ci95"]
        c = "S" if lo > -7 else ("L" if hi < -7 else "U")
        cls[x] = c
        f, u, _ = bad_cells(m, r["cells"])
        P(f"  R1 {x:8s} drop_nat {r['point']:+6.2f} [{lo:+.2f}, {hi:+.2f}] -> {c} 类" + ("" if x in NEW_BODIES else "（非新身体，仅参考）")
          + (f"  <<< does not count（{','.join(f)}）" if f else "") + (f"（门槛未核实 {','.join(u)}）" if u else ""))
    for x in ["K3", "RT2"]:
        r, d = res.get(f"recovery({x})"), res.get(f"drop_nat({x})")
        if cls.get(x) != "L":
            P(f"  R2 {x}: 非 L 类（{cls.get(x)}），不适用")
        elif r is None:
            P(f"  R2 {x}: L 类但 {x}_T 无数据（条件格未跑或缺）")
        else:
            half = 0.5 * abs(d["point"])
            lo, hi = r["ci95"]
            v = "外观为主" if lo > half else ("执行为主" if hi < half else "混合")
            f, u, _ = bad_cells(m, r["cells"])
            P(f"  R2 {x}: recovery {r['point']:+.2f} [{lo:+.2f}, {hi:+.2f}] 对 0.5|drop_nat| = {half:.2f} -> {v}"
              + (f"  <<< does not count（{','.join(f)}）" if f else ""))
    r = res.get("recovery(G)")
    if r:
        P(f"  recovery(G)（§2，描述）{r['point']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    return cls


def r7(models, path):
    P("\n==== R7 回放接触预测（只对 P6j3、K3 计分；IIWA、UR5e、UR5e_nc、RT2、Panda 行为校准/参考，不计分）")
    sh = sha256(path)
    if sh is None:
        P(f"  缺 {path}")
        return
    ok = sh == R7_SHA
    P(f"  预测表 sha256 {sh} {'= 冻结值' if ok else '!= 冻结值 ' + R7_SHA + '，只打印不计分'}")
    rows = [ln.rstrip("\n").split("\t") for ln in open(path)][1:]
    bodymap = {"PandaL3": "P6j3", "Kinova3": "K3", "IIWA": "IIWA_cg", "UR5e": "UR5e_cg", "UR5e_nc": "UR5e_nc",
               "Panda": "S", "PandaRT2": "RT2"}
    for m in models:
        P(f"  -- {m.name}{'（主）' if m.kind == 'oft' else '（描述，不计分）'}")
        sP = m.per_task("S") if "S" in m.data else None
        score = defaultdict(lambda: [0, 0])
        for row in rows:
            if len(row) < 6:
                continue
            body, t, ct, pred = row[0], int(row[1]), float(row[4]), row[5]
            cell = bodymap.get(body)
            pt = m.per_task(cell) if cell in m.data else None
            sr = 100 * pt[t][0] / pt[t][1] if pt and pt[t][1] else None
            ssr = 100 * sP[t][0] / sP[t][1] if sP and sP[t][1] else None
            scored_body = body in ("PandaL3", "Kinova3")
            txt = f"    {body:8s} t{t} c_t {ct:.2f} 预测「{pred}」 SR({cell}) {'缺' if sr is None else f'{sr:.0f}%'} SR(S) {'缺' if ssr is None else f'{ssr:.0f}%'}"
            if not scored_body:
                if ct > 0 or body == "UR5e":
                    P(txt + " [参考，不计分]")
                continue
            if pred.startswith("SR > 25%"):
                if ssr is None or sr is None:
                    P(txt + " -> 缺数据")
                elif ssr >= 75:
                    c = sr > 25
                    score[body][0] += c
                    score[body][1] += 1
                    P(txt + f" -> 计分：{'正确' if c else '错误'}")
                else:
                    P(txt + " -> S < 75%，不计分")
            elif "50" in pred and ("<=" in pred or "≤" in pred):
                if sr is None:
                    P(txt + " -> 缺数据")
                else:
                    c = sr <= 50
                    score[body][0] += c
                    score[body][1] += 1
                    P(txt + f" -> 计分：{'正确' if c else '错误'}")
            else:
                P(txt + " -> 无预测")
        if ok:
            for b, (k, n) in score.items():
                P(f"    {b}: 预测正确 {k} / 预测数 {n}")
            k = sum(v[0] for v in score.values())
            n = sum(v[1] for v in score.values())
            P(f"    合计（P6j3 + K3）：{k} / {n}")


def r9(res):
    r = res.get("d_cg(sham)")
    P("\n==== R9 复制开关效应 d_cg(X)（描述）")
    for x in ["sham", "iiwa", "ur5e"]:
        rr = res.get(f"d_cg({x})")
        P(f"  d_cg({x}) " + ("缺" if rr is None else f"{rr['point']:+.2f} [{rr['ci95'][0]:+.2f}, {rr['ci95'][1]:+.2f}]"))
    if r:
        P(f"  预测 |d_cg(sham)| <= 3（点）：{'成立' if abs(r['point']) <= 3 else '不成立'}；附录 A.3 |d_cg(sham)| > 5：{'触发' if abs(r['point']) > 5 else '不触发'}")


def spec_line(res):
    r = res.get("spec")
    if r:
        lo, hi = r["ci95"]
        P(f"  spec（任务 3 以外 UR5e_nc − UR5e_cg）{r['point']:+.2f} [{lo:+.2f}, {hi:+.2f}] -> 95% CI {'在' if (lo >= -5 and hi <= 5) else '不在'} ±5 内")
    for x in ["UR5e_nc", "IIWA_cg"]:
        rr = res.get(f"drop_nat({x})")
        if rr:
            P(f"  drop_nat({x}) {rr['point']:+.2f} [{rr['ci95'][0]:+.2f}, {rr['ci95'][1]:+.2f}]（并列，描述）")


# ----------------------------------------------------------------------------------------------- triggers
def triggers(m, cls, dec, n200_full):
    P("\n==== 条件触发（按 OFT）")
    for x, cell in [("RT2", "RT2_T"), ("K3", "K3_T")]:
        c = cls.get(x)
        if c in (None, "缺"):
            P(f"  {cell}：无法判定（缺数据）")
        else:
            st = m.status(x)
            P(f"  {cell}：{'触发' if c == 'L' else '不触发'}（{x} R1 = {c} 类）" + ("  注意：该格 does not count" if st == "FAIL" else ""))
    p10 = dec.get("P10")
    if not n200_full:
        P("  加密到 400：无法判定（夹爪家族 7 格未全部齐 200 集）" + f"；P10 当前 = {p10}")
    elif p10 == "不确定":
        P("  加密到 400：触发（P10 于 200 集判不确定）")
    elif p10 in ("确认", "证伪"):
        P(f"  加密到 400：不触发（P10 = {p10}）")
    elif p10 == "does not count":
        P("  加密到 400：不触发（P10 不计入判定，不是『不确定』；协议未覆盖此情形，需人工决定）")
    else:
        P(f"  加密到 400：无法判定（P10 = {p10}）")


def p9_final(dec, title):
    P(f"\n==== P9 单侧解释（裁定 2）与 G_Rw 触发{title}")
    p9, p10, raw = dec.get("P9"), dec.get("P10"), dec.get("P9_raw")
    if p9 in (None, "缺"):
        P("  P9 缺 -> 单侧解释无法应用；G_Rw：无法判定（缺数据）")
        return None
    if p9 == "确认":
        final = "确认（保守方向：第三人称线索更弱仍丢分）"
    elif p9 in ("证伪", "不确定"):
        if p10 == "确认" and p9 == "证伪":
            final = "证伪（P10 确认，腕部通道为主）"
        elif p10 in ("证伪", "不确定"):
            final = "未能判定（工具限制：G_R 的第三人称线索弱于 G）——不写成证伪"
        elif p10 == "确认":
            final = "不确定"
        else:
            final = f"{p9}（P10 = {p10}，单侧规则无法应用）"
    else:
        final = f"{p9}（P9 不进入单侧规则；按判定线 {raw}）"
    P(f"  P9 最终读法：{final}")
    if p10 in (None, "缺"):
        P("  G_Rw：无法判定（P10 缺数据）")
    else:
        trig = p9 != "确认" and p10 != "确认"
        P(f"  G_Rw：{'触发' if trig else '不触发'}（P9 = {p9}，P10 = {p10}；条件：P9 未确认且 P10 未确认）")
    return final


# ----------------------------------------------------------------------------------------------- main flows
def analyse_spatial(a, neps, label, gr):
    models = []
    for kind, nm in [("oft", "OpenVLA-OFT"), ("pi05", "pi0.5-LIBERO")]:
        ne = neps if kind == "oft" else min(neps, 20)
        models.append(Model(nm, kind, SPATIAL, ne, a, "spatial"))
    models[1].src_oft = models[0].data.get("SRC")
    if neps > 20:  # pi0.5 SRC comparison needs only ep < 20, already a subset
        pass
    allres = {}
    for m in models:
        m.compute_gates(gr)
        m.print_gates()
        if gr["lines"] and m.kind == "oft":
            P("    G_R 第 0 步逐状态（gr_t0）：")
            for ln in gr["lines"]:
                P(ln)
        boot = Boot(m.keys, a.B)
        res = run_estimands(m, boot, SPATIAL_EST)
        allres[m.kind] = (m, res, boot)
        print_estimands(m, res, f"§2 估计量 {label}")
    return models, allres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["spatial", "object"], default="spatial")
    ap.add_argument("--n", type=int, choices=[200, 400], default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--runs", default="$PROJ_ROOT/runs")
    ap.add_argument("--pi05", default="$PROJ_ROOT/pi05/v2")
    ap.add_argument("--slurm-logs", default="~/logs")
    ap.add_argument("--eval-logs", default="$PROJ_ROOT/openvla-oft/experiments/logs")
    ap.add_argument("--pi05-logs", default="$PROJ_ROOT/pi05/logs")
    ap.add_argument("--gr-t0", default="$PROJ_ROOT/embody_v3/out/gr_t0")
    ap.add_argument("--r7", default="$PROJ_ROOT/embody_v3/out/r7_predictions.tsv")
    ap.add_argument("--stages-dir", default="$PROJ_ROOT/embody/analysis")
    ap.add_argument("--protocol", default="$PROJ_ROOT/embody_v3/protocol_block3.md")
    ap.add_argument("--amendment", default="$PROJ_ROOT/embody_v3/protocol_block3_amendment1.md")
    ap.add_argument("--exclude-keys", default=None)  # sens
    a = ap.parse_args()
    if a.suite == "object" and a.n == 400:
        sys.exit("--n 400 只用于 spatial（协议 §5）")
    out = a.out or {("spatial", 200): "results_block3_final.txt", ("spatial", 400): "results_block3_final_n400.txt",
                    ("object", 200): "results_block3_object.txt"}[(a.suite, a.n)]
    P("事后稳健性检查：物理失效集剔除版；检测器结局盲、定义于重算之前")
    P("prereg_block3_a2.py：按附录 2 改动 D1 到 D4，其余与 f1da2c63 逐行相同")
    P(f"prereg_block3.py  suite={a.suite}  n={a.n}  B={a.B}  seed={SEED}")
    P(f"runs={a.runs} pi05={a.pi05} slurm-logs={a.slurm_logs} eval-logs={a.eval_logs} pi05-logs={a.pi05_logs} gr-t0={a.gr_t0}")

    if a.suite == "spatial":
        gr = gr_t0_gate(a.gr_t0)
        P(f"\n######## 第一部分：n = 200（{'第二次看数里的第一阶段复算' if a.n == 400 else '第一次看数'}）")
        models, allres = analyse_spatial(a, 20, "n=200", gr)
        m, res, _ = allres["oft"]
        mp, resp, _ = allres["pi05"]
        P("\n==== 裁定 3.1 G_R 根 body 偏移（查询帧，横向 = hypot(dx, dy)）")
        p95s = {}
        for mm in models:
            P(f"  -- {mm.name}")
            lines, p95, _ = root_offset(mm)
            for ln in lines:
                P(ln)
            p95s[mm.kind] = p95
        dec = decide(m, res, p95s["oft"])
        decp = decide(mp, resp, p95s["pi05"], tag="（π0.5，按 P7 读，不进决策族）")
        holm_table({"P8'": ("d6", res.get("d6")), "P9": ("d_Grev", res.get("d_Grev")),
                    "P10": ("channel", res.get("channel")), "P11": ("Delta3", res.get("Delta3"))}, "Holm 附表（n=200）")
        p7(res, resp, list(SPATIAL_EST))
        cls = r1_r2(m, res)
        r1_r2(mp, resp)
        P("\n==== P11 辅助（描述）")
        spec_line(res)
        r9(res)
        r7(models, H(a.r7))
        for mm in models:
            mm.r8()
        for mm in models:
            mm.contacts()
        P("\n==== 裁定 3.1 接触前 / 后失败分解（G_R 相对 T0 的净增失败，OFT；P9 确认时用于判别，此处照印）")
        for ln in failure_split(m)[0]:
            P(ln)
        full = all(c in m.data and len(m.data[c]) == 200 for c in GRIP_FAMILY)
        triggers(m, cls, dec, full)
        p9_final(dec, "（n=200）")
        if a.n == 400:
            P("\n######## 第二部分：n = 400（§5 第二次看数：只重算 P9、P10；决策以 400 为准；两阶段看数，第一阶段只在 P10 不确定时继续，不做 alpha 分配）")
            m4 = Model("OpenVLA-OFT", "oft", [x for x in SPATIAL if x[0] in set(GRIP_FAMILY) | {"SRC"}], 40, a, "spatial")
            m4.compute_gates(gr)
            m4.print_gates()
            boot4 = Boot(m4.keys, a.B)
            res4 = run_estimands(m4, boot4, {k: SPATIAL_EST[k] for k in GRIP_EST})
            print_estimands(m4, res4, "夹爪家族估计量 n=400")
            P("\n==== 裁定 3.1 G_R 根 body 偏移（n=400）")
            lines, p95_4, _ = root_offset(m4)
            for ln in lines:
                P(ln)
            dec4 = {}
            P("\n==== 决策 §3（n=400，只 P9、P10）")
            r = res4.get("d_Grev")
            if r:
                raw = verdict_ci(r, -21.75, "below")
                v, txt = gated(m4, DECISION_CELLS["P9"], raw)
                if v not in ("does not count", "缺") and p95_4 is not None and p95_4 > 6.4:
                    v, txt = "descriptive", f"descriptive（p95 {p95_4:.2f} > 6.4）；按判定线为「{raw}」，仅供参考"
                elif v not in ("does not count", "缺") and p95_4 is None:
                    txt += "（降级条件 (c) 无法核实）"
                dec4["P9"], dec4["P9_raw"] = v, raw
                P(f"  P9 d_Grev {r['point']:+.2f} 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] -> {txt}")
            rc, rw = res4.get("channel"), res4.get("rec_w")
            if rc and rw:
                lo, hi = rc["ci95"]
                raw = "确认" if (lo > 0 and rw["point"] >= 17.25) else ("证伪" if hi < 0 else "不确定")
                dec4["P10"], txt = gated(m4, DECISION_CELLS["P10"], raw)
                P(f"  P10 channel {rc['point']:+.2f} 95% [{lo:+.2f}, {hi:+.2f}]，rec_w {rw['point']:+.2f} -> {txt}")
            P("\n==== 200 与 400 并列")
            for k in GRIP_EST:
                r2, r4 = res.get(k), res4.get(k)
                P(f"  {k:14s} n=200 " + ("缺" if not r2 else f"{r2['point']:+.2f} [{r2['ci95'][0]:+.2f}, {r2['ci95'][1]:+.2f}]")
                  + "   n=400 " + ("缺" if not r4 else f"{r4['point']:+.2f} [{r4['ci95'][0]:+.2f}, {r4['ci95'][1]:+.2f}]"))
            P("\n==== 裁定 3.1 接触前 / 后失败分解（n=400）")
            for ln in failure_split(m4)[0]:
                P(ln)
            m4.r8()
            final = {"P8'": dec.get("P8'"), "P9": dec4.get("P9", "缺"), "P10": dec4.get("P10", "缺"), "P11": dec.get("P11")}
            f8 = final["P8'"]
            P(f"\n==== 最终决策表：P8' {f8}（n=200），P9 {final['P9']}（n=400），P10 {final['P10']}（n=400），P11 {final['P11']}（n=200）")
            holm_table({"P8'": ("d6", res.get("d6")), "P9": ("d_Grev", res4.get("d_Grev")),
                        "P10": ("channel", res4.get("channel")), "P11": ("Delta3", res.get("Delta3"))}, "Holm 附表（最终：P9/P10 取 400）")
            p9_final({**dec4}, "（n=400，最终）")
    else:
        # §6 object
        models = [Model("OpenVLA-OFT", "oft", OBJECT, 20, a, "object"), Model("pi0.5-LIBERO", "pi05", OBJECT, 20, a, "object")]
        models[1].src_oft = models[0].src_oft  # sens: 未剔除的 OFT SRC；不剔除时与 data["SRC"] 是同一对象
        nogr = {"i": ("NA", ""), "ii": ("NA", ""), "lines": []}
        sp = {}
        for m in models:
            m.compute_gates(nogr)
            m.print_gates()
            boot = Boot(m.keys, a.B)
            res = run_estimands(m, boot, OBJECT_EST)
            print_estimands(m, res, "§6 object 估计量")
            sp[m.kind] = (m, res)
        P("\n==== spatial 同名估计量（为期望 (iii) 现算，n=200）")
        spres = {}
        for kind, nm in [("oft", "OpenVLA-OFT"), ("pi05", "pi0.5-LIBERO")]:
            need = {c for t in OBJ_TO_SPATIAL.values() for _, c in t[0]}
            ms = Model(nm + " spatial", kind, [x for x in SPATIAL if x[0] in need or x[0] == "SRC"], 20, a, "spatial")
            if kind == "pi05":
                ms.src_oft = spres["oft"][0].data.get("SRC")
            ms.compute_gates({"i": ("NA", ""), "ii": ("NA", ""), "lines": []})
            bs = Boot(ms.keys, a.B)
            r = run_estimands(ms, bs, OBJ_TO_SPATIAL)
            print_estimands(ms, r, "spatial 对照")
            spres[kind] = (ms, r)
        P("\n==== §6 事先写定的期望（描述，不判定）")
        for kind in ("oft", "pi05"):
            m, res = sp[kind]
            P(f"  -- {m.name}")
            r = res.get("drop_nat(UR5e)")
            if r:
                lo, hi = r["ci95"]
                c = "S" if lo > -7 else ("L" if hi < -7 else "U")
                P(f"  (i) UR5e R1 类 = {c}（期望 S 或 U：{'符合' if c in 'SU' else '不符'}）")
            else:
                P("  (i) UR5e 缺")
            if "UR5e" in m.data and "S" in m.data:
                pu, ps = m.per_task("UR5e"), m.per_task("S")
                bad = [t for t in range(10) if pu[t][1] and ps[t][1] and 100 * pu[t][0] / pu[t][1] <= 25 and 100 * ps[t][0] / ps[t][1] >= 75]
                P(f"      「SR <= 25% 而 S >= 75%」的任务：{bad if bad else '无'}（期望无）")
            r = res.get("recovery(G)")
            P("  (ii) recovery(G) " + ("缺" if not r else f"{r['point']:+.2f}：符号{'为正（符合）' if r['point'] > 0 else '非正（不符）'}"))
            p7(spres[kind][1], res, list(OBJECT_EST), title=f"(iii) object 对 spatial 符号一致性（{m.name}，P7 口径）",
               a_lbl="spatial", b_lbl="object")
        p7(sp["oft"][1], sp["pi05"][1], list(OBJECT_EST), title="P7 复现（object，π0.5 对 OFT）")
        cls = None
        for kind in ("oft", "pi05"):
            m, res = sp[kind]
            P(f"\n==== R1（object，{m.name}）")
            for x in ["UR5e", "K3", "G"]:
                r = res.get(f"drop_nat({x})")
                if r:
                    lo, hi = r["ci95"]
                    P(f"  {x:5s} {r['point']:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {'S' if lo > -7 else ('L' if hi < -7 else 'U')} 类")
        for kind in ("oft", "pi05"):
            sp[kind][0].r8()
        for kind in ("oft", "pi05"):
            sp[kind][0].contacts()
        _ = cls

    # sha256
    P("\n==== 冻结文件 sha256")
    me = os.path.abspath(__file__)
    for lbl, pth, exp in [("protocol_block3.md", H(a.protocol), PROTO_SHA), ("protocol_block3_amendment1.md", H(a.amendment), AMEND_SHA),
                          ("r7_predictions.tsv", H(a.r7), R7_SHA), ("stages.py（失败阶段口径）", os.path.join(H(a.stages_dir), "stages.py"), None),
                          ("protocol_block3_amendment2.md", os.path.join(os.path.dirname(H(a.amendment)), "protocol_block3_amendment2.md"), AMEND2_SHA),
                          ("prereg_block3.py（冻结原版）", os.path.join(os.path.dirname(me), "prereg_block3.py"), FROZEN_SHA),
                          ("prereg_block3_a2.py（a2 原版）", os.path.join(os.path.dirname(me), "prereg_block3_a2.py"), A2_SHA),
                          ("prereg_block3_a2_sens.py（本脚本）", me, None)]:
        s = sha256(pth)
        chk = "" if exp is None else (" = 冻结值" if s == exp else f" != 冻结值 {exp}" if s else f"（文件不在此处；冻结值 {exp}）")
        P(f"  {lbl:34s} {s or '缺'}{chk}  [{pth}]")
    with open(out, "w") as f:
        f.write("\n".join(P.lines) + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
