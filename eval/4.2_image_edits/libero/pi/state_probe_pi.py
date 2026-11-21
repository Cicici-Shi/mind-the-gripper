"""maskinfo-pi end-to-end state-path probe (Astra ruling 11), against the running pi/serve_policy_mi.py server.
One real observation (frame 0 of a smoke C0 dump: query images + 8-d state; success never read) is sent as:
  A  : state s,             noise key q=0          A2 : identical request again
  B  : state s perturbed,   noise key q=0          C  : state s, noise key q=1
Expected: A == A2 bit-exactly (fixed noise, no hidden rng); C != A (noise is used). pi0 reads the state: B != A.
pi0.5 has no state path (discrete_state_input False, no state token): B == A bit-exactly.
Usage: python -I tools/state_probe_pi.py --port P --npz <smoke C0 npz> --expect-state used|unused
"""
import argparse
import math
import sys

import numpy as np
from openpi_client import image_tools, websocket_client_policy

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--npz", required=True)
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--prompt", default="pick up the black bowl")
ap.add_argument("--expect-state", choices=["used", "unused"], required=True)
a = ap.parse_args()


def q2aa(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (quat[:3] * 2.0 * math.acos(quat[3])) / den


z = np.load(a.npz, allow_pickle=True)
img = np.asarray(z["mr_img_agent"][0])
wr = np.asarray(z["mr_img_wrist"][0])
s = np.concatenate((np.asarray(z["eef_pos"][0], np.float64), q2aa(np.asarray(z["eef_quat"][0], np.float64)),
                    np.asarray(z["gripper_qpos"][0], np.float64)))


def el(state, q, seed=None):
    f = lambda x: image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(x[::-1, ::-1]), 224, 224))
    return {"observation/image": f(img), "observation/wrist_image": f(wr), "observation/state": state,
            "prompt": a.prompt, "noise_seed": (990000 + q) if seed is None else seed, "mi_q": 1 + q}


cl = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
sp = s.copy()
sp[:3] += 0.05
sp[6:] = [0.0, 0.0]
rA, rA2, rB, rC = cl.infer(el(s, 0)), cl.infer(el(s, 0)), cl.infer(el(sp, 0)), cl.infer(el(s, 1))
A, A2, B, C = (np.asarray(r["actions"]) for r in (rA, rA2, rB, rC))
noise_differs = rA["mi_noise"]["md5"] != rC["mi_noise"]["md5"] and rA["mi_noise"]["md5"] == rA2["mi_noise"]["md5"]
rD = cl.infer(el(s, 0, seed=990000 + 2 ** 40))  # same low 32 bits as A: do the high seed bits reach the noise?
print(f"STATE_PROBE seed_high_bits_change_noise={rD['mi_noise']['md5'] != rA['mi_noise']['md5']} (reported only)")
rep, noise_used, state_eq = np.array_equal(A, A2), not np.array_equal(A, C), np.array_equal(A, B)
# protocol 2.4 / ruling: different seeds must give different NOISE (checked on the noise itself)
print(f"STATE_PROBE repeat_bit_identical={rep} different_seed_different_noise={noise_differs} noise_changes_actions={noise_used} "
      f"perturbed_state_bit_identical={state_eq} max_abs_diff_state={float(np.abs(A - B).max()):.3g}")
ok = rep and noise_differs and (state_eq if a.expect_state == "unused" else not state_eq)
print("STATE_PROBE_OK", ok)
sys.exit(0 if ok else 1)
