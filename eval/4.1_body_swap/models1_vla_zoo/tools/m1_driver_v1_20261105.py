"""models1 driver (2026-11-04): one tick = look at what is on disk and in squeue, gate shard 0, (re)submit what is
missing. Run by tools/m1_driver.sbatch, which resubmits itself every 20 min until 2026-11-09 12:00 SGT. State in
$PROJ_ROOT/vla_zoo/out/m1_driver_state.json, log $PROJ_ROOT/vla_zoo/out/m1_driver.log. Never prints a success rate.

Release rule (protocol_models1.md 7.1 / 7.4: "each cell's shard 0 first, full only after the gates"): a model-suite is
released when every cell's shard 0 is complete (50 keys without error) and the zoo_gate_m1.py interface gates pass:
scene, copy_grip_cmd, Panda first-frame alignment, contact log, error rate <= 1%, IK, stale, camera (TWIN_CAM=phys
cells), replan (pi0 5 / OVLA 1), wall ratio twin/native <= 2 (OFTi <= 1.5), T0-vs-S agreement >= 90%. pi0 spatial /
object were released by hand (WORKLOG_vlazoo 8.3, 8.5) and are pre-marked. A failed gate is logged and NOT released
(needs a person). Step-0 consistency (t0check, model independent) is not part of this rule (spatial passed; object
pending a decision, does not stop data collection).
Jobs: OFTi one job per a100-40 card (QOS <= 2), RUNS taken from the protocol 9 priority list (spatial main, spatial
densify ep 20-39, object); pi0 / OVLA: one server job (OVLA: 3 server processes, h100-47 = 47 GB slice) + client jobs of <= 11 shards each.
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


def log(m):
    s = f"{dt.datetime.now():%m-%d %H:%M} {m}"
    print(s, flush=True)
    open(LOG, "a").write(s + "\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


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


def gate_ok(model, suite):
    j = f"{Z}/out/gate_{model}_{suite}_s0.json"
    cells = CELLS if model != "openvla" else [c for c in CELLS if c != "g_t"]
    gm = "oft_sp" if model == "oft_sp" else model
    sh(f"cd {Z} && python3 tools/zoo_gate_m1.py --model {gm} --suite {suite} --offsets 0 --cells {','.join(cells)} --json {j} > {j}.txt 2>&1")
    try:
        g = json.load(open(j))
    except Exception as e:
        return False, f"gate json unreadable {e!r}"
    bad = []
    lim = 1.5 if model.startswith("oft") else 2.0
    exp_replan = {"pi0": 5, "openvla": 1}.get(model)
    for c, R in g["cells"].items():
        for k in ("scene_gate", "copy_gate", "contacts_gate", "error_rate_gate", "ik_gate", "cam_gate"):
            if k in R and R[k] is not True:
                bad.append(f"{c}.{k}={R[k]}")
        if R.get("align_gate") is False:
            bad.append(f"{c}.align")
        if R.get("stale_total"):
            bad.append(f"{c}.stale={R['stale_total']}")
        if exp_replan is not None and R.get("replan") != [exp_replan]:
            bad.append(f"{c}.replan={R.get('replan')}")
        if R.get("records") != 50:
            bad.append(f"{c}.records={R.get('records')}")
        if model.startswith("oft"):
            if R.get("gpu_models") and any("A100" not in x for x in R["gpu_models"]):
                bad.append(f"{c}.gpu={R['gpu_models']}")
        elif R.get("gpu_models") and any("T4" not in x for x in R["gpu_models"]):
            bad.append(f"{c}.gpu={R['gpu_models']}")
    for k, v in g["wall_ratio"].items():
        if v is not None and v > lim:
            bad.append(f"wall {k}={v}")
    a = (g.get("agree") or {}).get("T0_vs_S") or {}
    if a.get("outcome_agree") is not None and a["outcome_agree"] < 0.90:
        bad.append(f"T0_vs_S={a['outcome_agree']}")
    return (not bad), "; ".join(bad) or "all interface gates pass"


def oft_load_check(model):
    """7.4: OFTi checkpoint load (no missing-weights warning) and num_images/use_proprio from the job logs."""
    fs = glob.glob(f"{Z}/logs/m1_{model}_*_e0_*.log")
    txt = "".join(open(f, errors="replace").read()[:400000] for f in fs)
    warn = len(re.findall(r"(?i)weights.*(not initialized|newly initialized)|missing keys|unexpected keys", txt))
    return warn == 0, f"{len(fs)} logs, missing-weight warnings {warn}, 'num_images_in_input = 2' {txt.count('num_images_in_input = 2') or txt.count('num_images_in_input=2')}, 'use_proprio = True' {txt.count('use_proprio = True') or txt.count('use_proprio=True')}"


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    st.setdefault("released", {"pi0|libero_spatial": "hand 01:25", "pi0|libero_object": "hand 01:40"})
    st.setdefault("claims", {})   # jobid -> list of "model|cell|off"
    st.setdefault("gate_msgs", {})
    now = dt.datetime.now()
    q = squeue()
    # drop claims of finished jobs
    st["claims"] = {j: v for j, v in st["claims"].items() if j in q}
    claimed = {x for v in st["claims"].values() for x in v}
    req = required()
    todo = [(m, su, c, o) for m, su, c, o in req if n_done(m, c, o) < 50]
    log(f"tick: {len(req) - len(todo)}/{len(req)} shards complete; queue " + ",".join(f"{j}:{n}:{s[0]}" for j, (n, s) in q.items() if n.startswith("m1_")))
    # gates
    for m, su in (("oft_sp", "libero_spatial"), ("oft_obj", "libero_object"), ("openvla", "libero_spatial")):
        key = f"{m}|{su}"
        if key in st["released"]:
            continue
        cells = [("obj_" if su == "libero_object" else "") + c for c in CELLS if not (m == "openvla" and c == "g_t")]
        if all(n_done(m, c, 0) >= 50 for c in cells):
            ok, msg = gate_ok(m, su)
            if m.startswith("oft"):
                ok2, msg2 = oft_load_check(m)
                ok, msg = ok and ok2, msg + " | load: " + msg2
            if m == "openvla" and not all(n_done(m, "src", o) >= 50 for o in MAIN):
                ok, msg = False, msg + " | SRC 200 not complete yet"
            if st["gate_msgs"].get(key) != msg:
                log(f"GATE {key}: {'PASS' if ok else 'FAIL/WAIT'} {msg}")
                st["gate_msgs"][key] = msg
            if ok:
                if m == "openvla":   # 5.4: SRC 200 must be >= 75% before the other cells are released
                    out = sh(f"cd {Z} && python3 common/zoo_gate.py --model openvla --cell src --show-success | tail -1")
                    mm = re.search(r"SRC success (\d+)/(\d+)", out)
                    if not mm or int(mm.group(1)) < 0.75 * int(mm.group(2)) or int(mm.group(2)) < 200:
                        log(f"GATE {key}: OVLA SRC baseline check not passed: {out[:200]}")
                        json.dump(st, open(STATE, "w"), indent=1)
                        continue
                    log(f"GATE {key}: OVLA SRC baseline {out[:120]}")
                st["released"][key] = f"auto {now:%m-%d %H:%M}"
                log(f"RELEASED {key}")
    if now >= DEADLINE:
        log("deadline reached: no new submissions")
        json.dump(st, open(STATE, "w"), indent=1)
        return

    def allowed(m, su, o):
        return o == 0 or f"{m}|{su}" in st["released"]

    # OFTi: up to 2 card jobs (QOS), each up to 8 shards
    oft_jobs = [j for j, (n, s) in q.items() if n.startswith("m1_oft")]
    free = [x for x in todo if x[0].startswith("oft") and f"{x[0]}|{x[2]}|{x[3]}" not in claimed and allowed(x[0], x[1], x[3])]
    while len(oft_jobs) < 2 and free:
        m = free[0][0]
        runs = [x for x in free if x[0] == m][:8]
        ck = "moojink/openvla-7b-oft-finetuned-libero-spatial" if m == "oft_sp" else "moojink/openvla-7b-oft-finetuned-libero-object"
        R = " ".join(f"{c}:{o}" for _, _, c, o in runs)
        jid = sh(f"cd {Z} && sbatch --parsable --time=05:59:00 --job-name=m1_{m} --export=ALL,MODEL={m},CKPT={ck},SUITE={runs[0][1]},RUNS='{R}' oft/run_oft_m1.sbatch")
        log(f"submit OFT {m} job {jid}: {R}")
        st["claims"][jid] = [f"{m}|{c}|{o}" for _, _, c, o in runs]
        oft_jobs.append(jid)
        free = [x for x in free if x not in runs]
    # pi0 / OVLA
    for m in ("pi0", "openvla"):
        free = [x for x in todo if x[0] == m and f"{x[0]}|{x[2]}|{x[3]}" not in claimed and allowed(x[0], x[1], x[3])]
        if not free:
            continue
        srv = [j for j, (n, s) in q.items() if n == f"m1_srvd_{m}" or (m == "openvla" and n == "m1_srvN_ovla")]
        if srv:
            srvj = srv[0]
        else:
            if m == "pi0":
                srvj = sh(f"cd {Z} && sbatch --parsable --job-name=m1_srvd_pi0 --gres=gpu:h100-47:1 --time=04:00:00 --export=ALL,MODEL=pi0,SERVER=openpi:pi0_libero:{Z}/openpi_ckpt/pi0_libero common/run_server.sbatch")
            else:
                srvj = sh(f"cd {Z} && sbatch --parsable --job-name=m1_srvd_openvla --gres=gpu:h100-47:1 --time=06:00:00 --export=ALL,MODEL=openvla,NSRV=3,SERVER=zoo:zoo_openvla:{Z}/openvla/adapter.py:unnorm_key=libero_spatial common/run_server_multi.sbatch")
            log(f"submit server {m} {srvj}")
        ncl = len([j for j, (n, s) in q.items() if n.startswith(f"m1_d_{m}")])
        while ncl < 2 and free:
            runs = free[:11]
            lst = f"{Z}/lists/m1_d_{m}_{now:%m%d%H%M}_{ncl}.txt"
            with open(lst, "w") as fh:
                for _, su, c, o in runs:
                    base = c[4:] if c.startswith("obj_") else c
                    v = sh(f"{Z}/tools/zoo_cells.sh vars {base}").split()
                    fh.write(f"{base} {v[0]} {su} {o} {' '.join(v[1:])}\n")
            extra = ",SRVN=3" if m == "openvla" else ""
            proto = "openpi" if m == "pi0" else "raw"
            jid = sh(f"cd {Z} && sbatch --parsable --dependency=after:{srvj} --time=03:00:00 --job-name=m1_d_{m} --gres=gpu:nv:1 --cpus-per-task=44 "
                     f"--export=ALL,M1_FASTSYNC=1{extra},MODEL={m},SERVER=remote:{m},SRVJOB={srvj},PROTO={proto},EMBODY_DIR={TOY},LIST={lst} common/run_zoo.sbatch")
            log(f"submit clients {m} job {jid} (server {srvj}): " + " ".join(f"{c}:{o}" for _, _, c, o in runs))
            st["claims"][jid] = [f"{m}|{c}|{o}" for _, _, c, o in runs]
            free = free[11:]
            ncl += 1
    # idle servers: no client job of that model queued/running and nothing left -> cancel
    for j, (n, s) in q.items():
        if n.startswith("m1_srvd_") or n == "m1_srvN_ovla":
            m = "openvla" if n == "m1_srvN_ovla" else n[len("m1_srvd_"):]
            if not any(x[0] == m for x in todo) or (s == "RUNNING" and not any(nn.startswith(f"m1_d_{m}") for nn, _ in q.values())
                                                     and not [x for x in todo if x[0] == m and allowed(x[0], x[1], x[3])]):
                sh(f"scancel {j}")
                log(f"cancel idle server {j} ({m})")
    json.dump(st, open(STATE, "w"), indent=1)


if __name__ == "__main__":
    main()
