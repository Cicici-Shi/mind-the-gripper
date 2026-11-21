"""models1 driver v2 (2026-11-05): one tick = look at what is on disk and in squeue, gate shard 0, (re)submit what is
missing. Run by tools/m1_driver.sbatch, which resubmits itself every 20 min until 2026-11-09 12:00 SGT. State in
$PROJ_ROOT/vla_zoo/out/m1_driver_state.json, log $PROJ_ROOT/vla_zoo/out/m1_driver.log. Never prints a success rate, except the
OVLA SRC baseline (protocol 5.4, the one allowed read).

Changes vs v1 (tools/m1_driver_v1_20261105.py), per protocol_models1_gate_rulings_fable.md section 4 items 1, 2, 9:
 1. Release is per CELL, not per model x suite (protocol 7.1 "each cell's shard 0 first"). zoo_gate_m1.py still runs once
    all cells of a model-suite have shard 0; each cell is then judged on its own gate items (scene, copy_grip_cmd,
    contacts, error rate, IK, camera, align, stale, replan, records, GPU model, and the wall ratio twin/native attributed
    to the twin cell). Passing cells are released ("auto"); failing cells are listed as PENDING-HUMAN (a person decides
    whether to collect them) and do not hold back the other cells.
    State `released` keys: "model|suite" (v1 format, still honoured: every cell of that model-suite is released) or
    "model|suite|cell" (cell = on-disk name, e.g. obj_g_ta_fc; the bare name g_ta_fc is accepted too).
    The non-gated src cell (offsets > 0) follows its model-suite: allowed once any cell of it is released.
 2. T0-vs-S outcome agreement < 0.90 is a WARN line in the log, not a block (protocol 7.2 says "look at the pictures";
    7.3 d5 handles it at readout).
 3. Side checks run for every release, automatic or by hand: OFTi checkpoint-load log check (oft_load_check) and the
    OVLA SRC 200 >= 75% baseline (common/zoo_gate.py --cell src --show-success). If one fails, no shard > 0 of that
    model-suite is submitted (src included) and the reason is logged. A passing SRC check is cached in state
    `side_ok` so the SRC success line is read once, not every tick.
--dry-run: compute what this tick would do and print it; submits nothing, cancels nothing, writes no state, no log, no
list files (gate json, if a gate would run, goes to $HOME/tmp).
Jobs: unchanged from v1 (OFTi one job per a100-40 card, QOS <= 2, <= 8 shards; pi0 / OVLA one server + <= 2 client jobs
of <= 11 shards).
"""
import datetime as dt, glob, json, os, re, subprocess, sys

Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
STATE = f"{Z}/out/m1_driver_state.json"
LOG = f"{Z}/out/m1_driver.log"
DEADLINE = dt.datetime(2026, 11, 9, 12, 0)
CELLS = ["sham_cg", "sham_twin", "grpkp100", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"]
MAIN = [0, 5, 10, 15]
DENSE = [20, 25, 30, 35]
TOY = f"{os.path.expanduser('~')}/proj/embody_toy"
GROUPS = (("oft_sp", "libero_spatial"), ("oft_obj", "libero_object"), ("openvla", "libero_spatial"))
DRY = False
PLAN = []          # dry-run: what would have been done
_dry_n = [0]


def log(m):
    s = f"{dt.datetime.now():%m-%d %H:%M} {m}"
    if DRY:
        PLAN.append(("log", m))
        print("[dry] " + s, flush=True)
        return
    print(s, flush=True)
    open(LOG, "a").write(s + "\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def act(kind, cmd, **kw):
    """Side-effecting command (sbatch / scancel). Dry-run: record and return a fake job id."""
    if DRY:
        _dry_n[0] += 1
        PLAN.append((kind, cmd, kw))
        print(f"[dry] would {kind}: {cmd}" + (f"  {kw}" if kw else ""), flush=True)
        return f"DRY{_dry_n[0]}"
    return sh(cmd)


def required():
    """(model, suite, cell_dir_or_prefix_name, offset) in priority order."""
    R = []
    sp = [("oft_sp", "libero_spatial", c, o) for o in MAIN for c in CELLS] + [("oft_sp", "libero_spatial", "src", o) for o in (10, 15)]
    sp += [("oft_sp", "libero_spatial", c, o) for o in DENSE for c in ("grpkp100", "g_tw_fc", "g_ta_fc")]
    ob = [("oft_obj", "libero_object", "obj_" + c, o) for o in MAIN for c in CELLS] + [("oft_obj", "libero_object", "obj_src", o) for o in (10, 15)]
    R += sp + ob
    for su, pre in (("libero_spatial", ""), ("libero_object", "obj_")):
        R += [("pi0", su, pre + c, o) for o in MAIN for c in CELLS]
    R += [("openvla", "libero_spatial", c, o) for o in MAIN for c in CELLS if c != "g_t"]
    R += [("openvla", "libero_spatial", "src", o) for o in MAIN]
    return R


def group_cells(m, su):
    """On-disk names of the gated cells of a model-suite."""
    return [("obj_" if su == "libero_object" else "") + c for c in CELLS if not (m == "openvla" and c == "g_t")]


def base(c):
    return c[4:] if c.startswith("obj_") else c


def n_done(model, cell, off):
    if model.startswith("oft"):
        ks = set()
        for f in glob.glob(f"{Z}/runs/{model}/{cell}/t*_e*_i*.npz"):
            m = re.search(r"t(\d+)_e(\d+)_", os.path.basename(f))
            if off <= int(m.group(2)) < off + 5:
                ks.add((int(m.group(1)), int(m.group(2))))
        return len(ks)
    f = f"{Z}/runs/{model}/{cell}_e{off}.jsonl"
    if not os.path.exists(f):
        return 0
    ks = set()
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            if r.get("error") is None and off <= r["episode"] < off + 5:
                ks.add((r["task"], r["episode"]))
    return len(ks)


def squeue():
    out = sh("squeue -u $USER -h -o '%i|%j|%T'")
    return {l.split("|")[0]: (l.split("|")[1], l.split("|")[2]) for l in out.splitlines() if l}


def run_gate(model, suite, cells):
    """Run the frozen zoo_gate_m1.py on shard 0; returns its json (dict) or raises."""
    d = os.path.expanduser("~/tmp") if DRY else f"{Z}/out"
    j = f"{d}/gate_{model}_{suite}_s0{'_dry' if DRY else ''}.json"
    gm = "oft_sp" if model == "oft_sp" else model
    sh(f"cd {Z} && python3 tools/zoo_gate_m1.py --model {gm} --suite {suite} --offsets 0 --cells {','.join(base(c) for c in cells)} --json {j} > {j}.txt 2>&1")
    return json.load(open(j))


def cell_verdicts(g, model):
    """Per-cell gate failures and group-level warnings from a zoo_gate_m1 json.
    Returns ({cell(bare name as in the gate json): [failure strings]}, [warning strings])."""
    lim = 1.5 if model.startswith("oft") else 2.0
    exp_replan = {"pi0": 5, "openvla": 1}.get(model)
    bad = {}
    for c, R in g["cells"].items():
        b = bad.setdefault(c, [])
        for k in ("scene_gate", "copy_gate", "contacts_gate", "error_rate_gate", "ik_gate", "cam_gate"):
            if k in R and R[k] is not True:
                b.append(f"{c}.{k}={R[k]}")
        if R.get("align_gate") is False:
            b.append(f"{c}.align")
        if R.get("stale_total"):
            b.append(f"{c}.stale={R['stale_total']}")
        if exp_replan is not None and R.get("replan") != [exp_replan]:
            b.append(f"{c}.replan={R.get('replan')}")
        if R.get("records") != 50:
            b.append(f"{c}.records={R.get('records')}")
        if model.startswith("oft"):
            if R.get("gpu_models") and any("A100" not in x for x in R["gpu_models"]):
                b.append(f"{c}.gpu={R['gpu_models']}")
        elif R.get("gpu_models") and any("T4" not in x for x in R["gpu_models"]):
            b.append(f"{c}.gpu={R['gpu_models']}")
    for k, v in (g.get("wall_ratio") or {}).items():   # key "twin/native": the twin cell carries the overhead
        if v is not None and v > lim:
            tw = k.split("/")[0]
            bad.setdefault(tw, []).append(f"wall {k}={v}")
    warns = []
    a = (g.get("agree") or {}).get("T0_vs_S") or {}
    if a.get("outcome_agree") is not None and a["outcome_agree"] < 0.90:
        warns.append(f"T0_vs_S={a['outcome_agree']} < 0.90 (WARN only: protocol 7.2 'look at the pictures', 7.3 d5 at readout)")
    return bad, warns


def is_released(st, m, su, cell):
    """Gated cell released? v1 key 'model|suite' releases every cell; v2 key 'model|suite|cell' one cell."""
    rel = st["released"]
    return f"{m}|{su}" in rel or f"{m}|{su}|{cell}" in rel or f"{m}|{su}|{base(cell)}" in rel


def oft_load_check(model):
    """7.4: OFTi checkpoint load (no missing-weights warning) and num_images/use_proprio from the job logs."""
    fs = glob.glob(f"{Z}/logs/m1_{model}_*_e0_*.log")
    txt = "".join(open(f, errors="replace").read()[:400000] for f in fs)
    warn = len(re.findall(r"(?i)weights.*(not initialized|newly initialized)|missing keys|unexpected keys", txt))
    return warn == 0, f"{len(fs)} logs, missing-weight warnings {warn}, 'num_images_in_input = 2' {txt.count('num_images_in_input = 2') or txt.count('num_images_in_input=2')}, 'use_proprio = True' {txt.count('use_proprio = True') or txt.count('use_proprio=True')}"


def ovla_src_check():
    """5.4: OVLA SRC 200 >= 75%. The only success rate this driver may read."""
    if not all(n_done("openvla", "src", o) >= 50 for o in MAIN):
        return False, "SRC 200 not complete yet"
    out = sh(f"cd {Z} && python3 common/zoo_gate.py --model openvla --cell src --show-success | tail -1")
    mm = re.search(r"SRC success (\d+)/(\d+)", out)
    if not mm or int(mm.group(1)) < 0.75 * int(mm.group(2)) or int(mm.group(2)) < 200:
        return False, f"OVLA SRC baseline check not passed: {out[:200]}"
    return True, f"OVLA SRC baseline {out[:120]}"


def main(dry=False):
    global DRY
    DRY = dry
    PLAN.clear()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    st.setdefault("released", {"pi0|libero_spatial": "hand 01:25", "pi0|libero_object": "hand 01:40"})
    st.setdefault("claims", {})   # jobid -> list of "model|cell|off"
    st.setdefault("gate_msgs", {})
    st.setdefault("side_ok", {})  # "model|suite" -> message of a passed side check (cached, SRC read once)
    st.setdefault("side_msgs", {})
    now = dt.datetime.now()
    q = squeue()
    st["claims"] = {j: v for j, v in st["claims"].items() if j in q}
    claimed = {x for v in st["claims"].values() for x in v}
    req = required()
    todo = [(m, su, c, o) for m, su, c, o in req if n_done(m, c, o) < 50]
    log(f"tick(v2): {len(req) - len(todo)}/{len(req)} shards complete; queue " + ",".join(f"{j}:{n}:{s[0]}" for j, (n, s) in q.items() if n.startswith("m1_")))

    # gates, per cell
    for m, su in GROUPS:
        key = f"{m}|{su}"
        cells = group_cells(m, su)
        if all(is_released(st, m, su, c) for c in cells):
            continue
        if not all(n_done(m, c, 0) >= 50 for c in cells):
            continue
        try:
            g = run_gate(m, su, cells)
        except Exception as e:
            msg = f"gate json unreadable {e!r}"
            if st["gate_msgs"].get(key) != msg:
                log(f"GATE {key}: WAIT {msg}")
                st["gate_msgs"][key] = msg
            continue
        bad, warns = cell_verdicts(g, m)
        passed = [c for c in cells if not bad.get(base(c)) and base(c) in g["cells"]]
        failed = {c: bad.get(base(c)) or [f"{base(c)} missing from gate json"] for c in cells if c not in passed}
        msg = ("; ".join(x for c in cells for x in failed.get(c, [])) or "all interface gates pass") + \
              ("" if not warns else " | WARN " + "; ".join(warns))
        if st["gate_msgs"].get(key) != msg:
            log(f"GATE {key}: pass {','.join(passed) or '-'}; FAIL {','.join(failed) or '-'} :: {msg}")
            for w in warns:
                log(f"WARN {key}: {w}")
            pend = [c for c in failed if not is_released(st, m, su, c)]
            if pend:
                log(f"PENDING-HUMAN {key}: cells {','.join(pend)} failed shard-0 gates; to collect, add "
                    f"'{key}|<cell>' to released in {STATE}; other cells are not held back")
            st["gate_msgs"][key] = msg
        for c in passed:
            if not is_released(st, m, su, c):
                st["released"][f"{key}|{c}"] = f"auto {now:%m-%d %H:%M}"
                log(f"RELEASED {key}|{c}")

    if now >= DEADLINE:
        log("deadline reached: no new submissions")
        if not DRY:
            json.dump(st, open(STATE, "w"), indent=1)
        return PLAN

    side_cache = {}

    def side(m, su):
        """Side checks forced for every release (auto or hand). Cached per tick; OVLA SRC pass cached in state."""
        key = f"{m}|{su}"
        if key in side_cache:
            return side_cache[key]
        if m.startswith("oft"):
            ok, msg = oft_load_check(m)
            msg = "load: " + msg
        elif m == "openvla":
            if key in st["side_ok"]:
                ok, msg = True, st["side_ok"][key]
            else:
                ok, msg = ovla_src_check()
                if ok:
                    st["side_ok"][key] = f"{msg} (checked {now:%m-%d %H:%M})"
        else:
            ok, msg = True, "no side check"
        if st["side_msgs"].get(key) != [ok, msg]:
            log(f"SIDE {key}: {'PASS' if ok else 'FAIL, not submitting shards > 0 of ' + key} {msg}")
            st["side_msgs"][key] = [ok, msg]
        side_cache[key] = ok
        return ok

    def allowed(m, su, c, o):
        if o == 0:
            return True
        if base(c) == "src":
            rel = any(is_released(st, m, su, x) for x in group_cells(m, su))
        else:
            rel = is_released(st, m, su, c)
        return rel and side(m, su)

    # OFTi: up to 2 card jobs (QOS), each up to 8 shards
    oft_jobs = [j for j, (n, s) in q.items() if n.startswith("m1_oft")]
    free = [x for x in todo if x[0].startswith("oft") and f"{x[0]}|{x[2]}|{x[3]}" not in claimed and allowed(x[0], x[1], x[2], x[3])]
    while len(oft_jobs) < 2 and free:
        m = free[0][0]
        runs = [x for x in free if x[0] == m][:8]
        ck = "moojink/openvla-7b-oft-finetuned-libero-spatial" if m == "oft_sp" else "moojink/openvla-7b-oft-finetuned-libero-object"
        R = " ".join(f"{c}:{o}" for _, _, c, o in runs)
        jid = act("sbatch", f"cd {Z} && sbatch --parsable --time=05:59:00 --job-name=m1_{m} --export=ALL,MODEL={m},CKPT={ck},SUITE={runs[0][1]},RUNS='{R}' oft/run_oft_m1.sbatch")
        log(f"submit OFT {m} job {jid}: {R}")
        st["claims"][jid] = [f"{m}|{c}|{o}" for _, _, c, o in runs]
        oft_jobs.append(jid)
        free = [x for x in free if x not in runs]
    # pi0 / OVLA
    for m in ("pi0", "openvla"):
        free = [x for x in todo if x[0] == m and f"{x[0]}|{x[2]}|{x[3]}" not in claimed and allowed(x[0], x[1], x[2], x[3])]
        if not free:
            continue
        srv = [j for j, (n, s) in q.items() if n == f"m1_srvd_{m}" or (m == "openvla" and n == "m1_srvN_ovla")]
        if srv:
            srvj = srv[0]
        else:
            if m == "pi0":
                srvj = act("sbatch", f"cd {Z} && sbatch --parsable --job-name=m1_srvd_pi0 --gres=gpu:h100-47:1 --time=04:00:00 --export=ALL,MODEL=pi0,SERVER=openpi:pi0_libero:{Z}/openpi_ckpt/pi0_libero common/run_server.sbatch")
            else:
                srvj = act("sbatch", f"cd {Z} && sbatch --parsable --job-name=m1_srvd_openvla --gres=gpu:h100-47:1 --time=06:00:00 --export=ALL,MODEL=openvla,NSRV=3,SERVER=zoo:zoo_openvla:{Z}/openvla/adapter.py:unnorm_key=libero_spatial common/run_server_multi.sbatch")
            log(f"submit server {m} {srvj}")
        ncl = len([j for j, (n, s) in q.items() if n.startswith(f"m1_d_{m}")])
        while ncl < 2 and free:
            runs = free[:11]
            lst = f"{Z}/lists/m1_d_{m}_{now:%m%d%H%M}_{ncl}.txt"
            lines = []
            for _, su, c, o in runs:
                v = sh(f"{Z}/tools/zoo_cells.sh vars {base(c)}").split()
                lines.append(f"{base(c)} {v[0]} {su} {o} {' '.join(v[1:])}\n")
            if not DRY:
                with open(lst, "w") as fh:
                    fh.writelines(lines)
            extra = ",SRVN=3" if m == "openvla" else ""
            proto = "openpi" if m == "pi0" else "raw"
            jid = act("sbatch", f"cd {Z} && sbatch --parsable --dependency=after:{srvj} --time=03:00:00 --job-name=m1_d_{m} --gres=gpu:nv:1 --cpus-per-task=44 "
                      f"--export=ALL,M1_FASTSYNC=1{extra},MODEL={m},SERVER=remote:{m},SRVJOB={srvj},PROTO={proto},EMBODY_DIR={TOY},LIST={lst} common/run_zoo.sbatch",
                      list_lines="".join(lines))
            log(f"submit clients {m} job {jid} (server {srvj}): " + " ".join(f"{c}:{o}" for _, _, c, o in runs))
            st["claims"][jid] = [f"{m}|{c}|{o}" for _, _, c, o in runs]
            free = free[11:]
            ncl += 1
    # idle servers: no client job of that model queued/running and nothing left -> cancel
    for j, (n, s) in q.items():
        if n.startswith("m1_srvd_") or n == "m1_srvN_ovla":
            m = "openvla" if n == "m1_srvN_ovla" else n[len("m1_srvd_"):]
            if not any(x[0] == m for x in todo) or (s == "RUNNING" and not any(nn.startswith(f"m1_d_{m}") for nn, _ in q.values())
                                                     and not [x for x in todo if x[0] == m and allowed(x[0], x[1], x[2], x[3])]):
                act("scancel", f"scancel {j}")
                log(f"cancel idle server {j} ({m})")
    if not DRY:
        json.dump(st, open(STATE, "w"), indent=1)
    return PLAN


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv[1:]
    plan = main(dry=dry)
    if dry:
        acts = [p for p in plan if p[0] != "log"]
        print(f"[dry] summary: {len(acts)} action(s)" + ("" if acts else " -- nothing to submit or cancel"))
