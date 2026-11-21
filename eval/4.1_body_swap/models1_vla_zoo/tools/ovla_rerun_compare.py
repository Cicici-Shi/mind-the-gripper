"""Compare the OVLA re-run with the recorded episodes, then summarise twin reachability (Fable add1 4.4).

Reading fixed in advance by the ruling (add1 section 4 item 4) and written here before any re-run output exists:
  a re-run key REPRESENTS its recorded episode iff
    (1) its twin residual series has a frame > 1 mm and |first over-1-mm step (re-run) - (recorded)| <= 10 steps, and
    (2) max residual "same order": max(a, b) / min(a, b) <= 10 ** 0.5 (within half a decade; executor's number,
        the ruling only says 同量级).
  Reachability is summarised ONLY from representative keys (non-representative keys are listed separately, labelled),
  and only as: position + orientation reachable / unreachable fraction over 50 random seeds, position-only, which twin
  joints sit at a limit, distance from the base and tilt of the target.
Outcome firewall: never reads `success` (the re-run records do not contain it; recorded records are popped at load),
never prints episode length, step count or query count of a protocol cell. The SRC smoke mode prints interface
quantities of the SRC cell only (the one cell whose outcome the protocol allows to be read).
Usage:
  python ovla_rerun_compare.py smoke  --rerun-dir out/ovla_rerun/smoke
  python ovla_rerun_compare.py main   --rerun-dir out/ovla_rerun/main [--diag J] [--verify J]
"""
import argparse, collections, glob, json, os, re

import numpy as np

Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
SAME_ORDER = 10 ** 0.5
MAX_FIRST_DIFF = 10
LIM = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-2.8973, 2.8973],
                [-0.0175, 3.7525], [-2.8973, 2.8973]])


def orig_records(cell):
    out = {}
    for f in sorted(glob.glob(os.path.join(Z, "runs", "openvla", f"{cell}_e*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.pop("success", None)
            if r.get("error") is not None:
                continue
            k = (int(r["task"]), int(r["episode"]))
            if k not in out:  # same rule as ik_diag_m1.load_jsonl
                out[k] = r
    return out


def rerun_records(rd, cell):
    out = {}
    for f in sorted(glob.glob(os.path.join(rd, f"{cell}_e*_p*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            r.pop("success", None)
            if r.get("stub") or r.get("error") is not None:
                continue
            out[(int(r["task"]), int(r["episode"]))] = r
    return out


def first_over(pe):
    i = np.nonzero(np.asarray(pe, float) > 1)[0]
    return int(i[0]) if len(i) else None


def detector_d(z):
    """physics_breakdown_scan.py detector D (arm |dq| > pi/2 per step or > 0.1 rad beyond a Panda limit; Robotiq
    finger outside [-0.5, 1.2] rad). Returns a flag string only."""
    q = np.asarray(z["qpos"], float)
    names = [str(n) for n in z["joint_names"]]
    arm = [i for i, n in enumerate(names) if re.match(r"robot0_joint\d$", n)]
    flags = []
    if len(q) > 1 and len(arm) == 7:
        qa = q[:, arm]
        if (np.abs(np.diff(qa, axis=0)).max(axis=1) > np.pi / 2).any():
            flags.append("A-jump")
        if ((qa < LIM[:, 0] - 0.1) | (qa > LIM[:, 1] + 0.1)).any():
            flags.append("A-limit")
    fi = [i for i, n in enumerate(names) if n == "gripper0_finger_joint"]
    if fi and ((q[:, fi[0]] < -0.5) | (q[:, fi[0]] > 1.2)).any():
        flags.append("B-finger")
    return ",".join(flags) or "-"


def cmd_smoke(a):
    kj = json.load(open(os.path.join(Z, "out", "ovla_rerun", "keys.json")))
    k = tuple(kj["smoke_src_key"])
    O, R = orig_records("src"), rerun_records(a.rerun_dir, "src")
    o, r = O[k], R[k]
    z = np.load(os.path.join(Z, r["rerun_npz"]))
    oc, rc = o.get("contacts") or {}, r.get("contacts") or {}
    print(f"SRC smoke key {k} (SRC = native Panda; the protocol allows reading this cell)")
    print(f"  scene fix identical {o['fix'] == r['fix']}; objq identical {o['objq'] == r['objq']}; init {o['init']} / {r['init']}")
    print(f"  a0 bitwise identical {o['a0'] == r['a0']}")
    for f in ("queries", "steps"):
        print(f"  {f}: recorded {o[f]} re-run {r[f]}")
    for f in ("first_contact", "first_grip_obj", "stall_steps", "arm_env_contact_steps"):
        print(f"  contacts.{f}: recorded {oc.get(f)} re-run {rc.get(f)}")
    print(f"  server recorded {o.get('server_host')} {o.get('server_gpu')}; re-run {r.get('server_host')} {r.get('server_gpu')}; "
          f"embody_dir {r.get('embody_dir')} fastsync {r.get('fastsync')} cell_env {r.get('cell_env')}")
    print(f"  npz members: { {n: z[n].shape for n in z.files if n != 'joint_names'} }")
    print(f"  executed actions == queries (replan 1): {len(z['qpos']) == r['queries']}; action rows {len(z['action'])}; QACC warnings {int(z['qacc_warn'][-1]) if len(z['qacc_warn']) else None}; "
          f"detector D {detector_d(z)}; 'success' in npz: {'success' in z.files}; 'success' in re-run record: {'success' in r}")
    # recorded first action == re-run first recorded action (a0 is the first executed block, replan 1)
    print(f"  first executed action re-run == recorded a0[0]: {np.allclose(z['action'][0], np.asarray(o['a0'][0], np.float32))}")


def cmd_main(a):
    kj = json.load(open(os.path.join(Z, "out", "ovla_rerun", "keys.json")))
    cell = kj["cell"]
    keys = [tuple(k) for k in kj["keys"]]
    O, R = orig_records(cell), rerun_records(a.rerun_dir, cell)
    print(f"== part 1: does the re-run represent the recorded trajectory? cell {cell}, {len(keys)} keys")
    print(f"   rule: re-run has > 1 mm, |first over-1-mm step diff| <= {MAX_FIRST_DIFF}, max residual ratio <= {SAME_ORDER:.2f}")
    print("   key | scene= | a0= | first>1mm rec/re | max mm rec/re | >1mm frames rec/re | rot max deg rec/re | QACC re | D re | REPRESENTS")
    rep, nonrep, missing = [], [], []
    for k in keys:
        if k not in R:
            missing.append(k)
            print(f"   {k} | re-run missing")
            continue
        o, r = O[k], R[k]
        pe_o = np.asarray(o["twin"]["pos_err_mm_q"], float)
        z = np.load(os.path.join(Z, r["rerun_npz"]))
        pe_r = np.asarray(z["twin_pos_err_mm"], float)
        fo, fr = first_over(pe_o), first_over(pe_r)
        mo, mr = float(pe_o.max()), float(pe_r.max())
        ok1 = fo is not None and fr is not None and abs(fo - fr) <= MAX_FIRST_DIFF
        ok2 = min(mo, mr) > 0 and max(mo, mr) / min(mo, mr) <= SAME_ORDER
        ok = ok1 and ok2
        (rep if ok else nonrep).append(k)
        qa = int(z["qacc_warn"][-1] - z["qacc_warn0"]) if len(z["qacc_warn"]) else 0
        print(f"   {k} | {o['fix'] == r['fix'] and o['objq'] == r['objq']} | {o['a0'] == r['a0']} | {fo}/{fr} | {mo:.1f}/{mr:.1f} | "
              f"{int((pe_o > 1).sum())}/{int((pe_r > 1).sum())} | {o['twin']['rot_err_deg_max']:.2f}/{float(z['twin_rot_err_deg'].max()):.2f} | "
              f"{qa} | {detector_d(z)} | {'YES' if ok else 'no'}")
    print(f"   representative {len(rep)}/{len(keys)}: {rep}; not representative {nonrep}; missing {missing}")

    if a.verify and os.path.exists(a.verify):
        V = json.load(open(a.verify))["rows"]
        print("== BodyEnv IK-chain replay on the re-run qpos (reach_diag_m1 verify): re-run residuals vs replay")
        for v in V:
            print(f"   {tuple(v['key'])}: |pos diff| max {v['pos_absdiff_max_mm']:.4f} mm, >1 mm frames rec {v['over1_rec']} replay {v['over1_replay']} same {v['over1_same']}")

    if not (a.diag and os.path.exists(a.diag)):
        print("no diag json; part 2 skipped")
        return
    rows = json.load(open(a.diag))["rows"]
    print("== part 2: twin reachability of the > 1 mm frames (50 random seeds, full pose and position only)")

    def summ(R_, label):
        if not R_:
            print(f"  [{label}] no frames")
            return
        fb = np.array([r["full_best_pos_mm"] for r in R_]); fr_ = np.array([r["full_best_rot_deg"] for r in R_])
        reach = np.array([r["full_seeds_le1mm"] > 0 for r in R_]); po = np.array([r["posonly_mm"] for r in R_])
        lim = collections.Counter(j for r in R_ for j in r["twin_joints_at_limit"])
        plim = collections.Counter(j for r in R_ for j in r["phys_joints_lt5deg"])
        dist = np.array([r["target_dist_from_base_m"] for r in R_]); tilt = np.array([r["target_tilt_deg"] for r in R_])
        n = len(R_)
        print(f"  [{label}] frames {n} keys {len(set(tuple(r['key']) for r in R_))}")
        print(f"    position + orientation, any of 50 seeds <= 1 mm: reachable {reach.sum()}/{n} = {reach.mean():.1%}, unreachable {n - reach.sum()}/{n}")
        print(f"    best of 50 seeds: pos mm p50/p90/max {np.percentile(fb, [50, 90, 100]).round(2).tolist()}, rot deg p50/max {np.percentile(fr_, [50, 100]).round(2).tolist()}")
        print(f"    position only: <= 0.01 mm {int((po <= 0.01).sum())}/{n}, <= 1 mm {int((po <= 1).sum())}/{n}, max {po.max():.2f} mm")
        print(f"    twin joints within 1 deg of a limit (best full-pose solution): {dict(sorted(lim.items()))}; frames with none {sum(1 for r in R_ if not r['twin_joints_at_limit'])}")
        print(f"    physical arm joints within 5 deg of a limit: {dict(sorted(plim.items()))}")
        print(f"    target distance from twin base m p10/p50/p90/max {np.percentile(dist, [10, 50, 90, 100]).round(3).tolist()}; tilt deg p10/p50/p90/max {np.percentile(tilt, [10, 50, 90, 100]).round(0).tolist()}")

    over = [r for r in rows if r["kind"] == "over1"]
    summ([r for r in over if tuple(r["key"]) in rep], "representative keys, > 1 mm frames (THE reading)")
    print("  per representative key: frames, reachable, twin joints at limit, dist p50 m, tilt p50 deg")
    for k in rep:
        R_ = [r for r in over if tuple(r["key"]) == k]
        if R_:
            lim = collections.Counter(j for r in R_ for j in r["twin_joints_at_limit"])
            print(f"    {k}: {len(R_)}, {sum(r['full_seeds_le1mm'] > 0 for r in R_)}, {dict(sorted(lim.items()))}, "
                  f"{np.median([r['target_dist_from_base_m'] for r in R_]):.3f}, {np.median([r['target_tilt_deg'] for r in R_]):.0f}")
    summ([r for r in over if tuple(r["key"]) not in rep], "NOT representative keys (listed only, not a reading)")
    summ([r for r in rows if r["kind"] == "control"], "controls (< 0.05 mm frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "main"])
    ap.add_argument("--rerun-dir", required=True)
    ap.add_argument("--diag", default="")
    ap.add_argument("--verify", default="")
    a = ap.parse_args()
    a.rerun_dir = os.path.abspath(a.rerun_dir)
    {"smoke": cmd_smoke, "main": cmd_main}[a.mode](a)


if __name__ == "__main__":
    main()
