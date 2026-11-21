"""maskinfo-pi determinism probe (coordinator request 11-08, after job <job>). Against a running
pi/serve_policy_mi.py. Sends ONE fixed request (frame 0 images + state of a smoke C0 dump, fixed noise_seed; success is
never read) in a labelled sequence and saves the returned action chunks; no outcome anywhere.
  --mode cold   : q0 (first q=0 of the process: log transform ON, also the JIT compile), q0 (OFF), q0 (OFF)
  --mode warm   : warm (mi_q=1, same seed: compile, no transform), q0 (ON), q0 (OFF)
  --mode debug  : q0, q0 (server --debug-state-transform: transform ON on every q=0)
  --mode load   : q0, q0 (transform already consumed by the running clients; concurrent clients on the same card)
  --compare DIR : bit comparison of all saved chunks, within process (a) and across processes (b)
"""
import argparse, glob, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["cold", "warm", "debug", "load"])
ap.add_argument("--compare", default="")
ap.add_argument("--port", type=int)
ap.add_argument("--npz")
ap.add_argument("--out")
ap.add_argument("--seed", type=int, default=424242)
ap.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
a = ap.parse_args()

if a.compare:
    def cmp(x, y):
        return {"bit_identical": bool(x.shape == y.shape and np.array_equal(x, y)),
                "n_diff": int((x != y).sum()), "max_abs": float(np.abs(x - y).max())}
    R = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(os.path.join(a.compare, "*.npz")))}
    bad_a = 0
    for model in sorted({k.split("_")[0] for k in R}):
        procs = {k.split("_", 1)[1]: R[k] for k in R if k.split("_")[0] == model}
        print(f"== {model}")
        for p, d in procs.items():  # (a) within one server process
            keys = list(d)
            for k in keys[1:]:
                c = cmp(d[keys[0]], d[k])
                bad_a += not c["bit_identical"]
                print(f"(a) within {p}: {keys[0]} vs {k}: {json.dumps(c)}")
        ref = sorted(procs)[0]
        k0 = lambda d: [k for k in d if k.startswith("q0")][-1]
        for p in sorted(procs):  # (b) across server processes, last q0 of each (transform OFF unless debug)
            if p != ref:
                print(f"(b) {ref} vs {p}: {json.dumps(cmp(procs[ref][k0(procs[ref])], procs[p][k0(procs[p])]))}")
    print("WITHIN_PROCESS_ALL_IDENTICAL", bad_a == 0)
    sys.exit(0 if bad_a == 0 else 1)

import math
from openpi_client import image_tools, websocket_client_policy


def q2aa(q):
    q = q.copy(); q[3] = np.clip(q[3], -1, 1); den = np.sqrt(1 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


z = np.load(a.npz, allow_pickle=True)
img, wr = np.asarray(z["mr_img_agent"][0]), np.asarray(z["mr_img_wrist"][0])
S = np.asarray(z["mi_S_server_received"], np.float64)
f = lambda x: image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(x[::-1, ::-1]), 224, 224))
req = lambda q: {"observation/image": f(img), "observation/wrist_image": f(wr), "observation/state": S.copy(),
                 "prompt": a.prompt, "noise_seed": a.seed, "mi_q": q}
cl = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", a.port)
seq = {"cold": [("q0_on_compile", 0), ("q0_off1", 0), ("q0_off2", 0)],
       "warm": [("warm_mi_q1", 1), ("q0_on", 0), ("q0_off", 0)],
       "debug": [("q0_on1", 0), ("q0_on2", 0)],
       "load": [("q0_off1_load", 0), ("q0_off2_load", 0)]}[a.mode]
out = {}
for name, q in seq:
    out[name] = np.asarray(cl.infer(req(q))["actions"])
np.savez(a.out, **out)
print("DETPROBE saved", a.mode, list(out))
