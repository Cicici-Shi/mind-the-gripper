"""生成局部响应检验的候选冻结清单（只写 FROZEN_localq.candidate；改名为 FROZEN_localq 由协调方做）。
  python freeze_localq.py --out FROZEN_localq.candidate
内容：采集与读数代码的 sha256；正式键清单（b2/keysets/formal_B1、B2，与 FROZEN_bench2 里的哈希比对）；
画面定义（名字、类型、轴、输入像素、换算后的环境像素）；读数参数；目标物与外扩圈；参考夹爪宽度；状态来源目录。"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import localq as LQ  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert os.path.basename(a.out) != "FROZEN_localq", "本脚本只生成候选，不生成 FROZEN_localq"
B2 = os.path.join(HERE, "..", "b2")
fz = json.load(open(os.path.join(B2, "FROZEN_bench2")))
man = {"note": "局部响应检验冻结清单候选（protocol_cue_local.md）；协调方核对后改名为 FROZEN_localq",
       "code_sha256": {f: sha(os.path.join(HERE, f)) for f in ("localq.py", "localq_readout.py", "test_localq_readout.py", "views.py",
                                                               "eval_views.py", "keys.py", "jobs/common.sh")},
       "keylist": {}, "views": [], "readout": {"a_in": 12.0, "b_in": 24.0, "inview_min": 0.8, "min_keys": 60, "near": 0.05,
                                               "B": 10000, "seed": 0, "queries_main": "preclose", "cluster": "cfg_id (episode, 24)"},
       "target": LQ.TARGET, "ring_px": LQ.RING_PX, "img_wh": [640, 480],
       "reference_gripper_width_input_px": {"B2": 48.125, "B1": "未量（协调方按 B2 的值定 a、b）"},
       "src": {"B1": "b2/out_formal/B1/C4", "B2": "b2/out_formal/B2/C4"}, "gpu_required": "A100-40 或 A100-80（零平移逐位复现）"}
for b in ("B1", "B2"):
    p = os.path.join(B2, "keysets", f"formal_{b}.json")
    h = sha(p)
    assert h == fz["keylist_sha256"][b], (b, h)
    man["keylist"][b] = {"path": "b2/keysets/" + os.path.basename(p), "sha256": h, "n": len(json.load(open(p)))}
for name, typ, axis, s_in in LQ.DEFAULT_VIEWS:
    e = {"name": name, "type": typ, "axis": axis, "input_px": s_in}
    if typ != "zero":
        e["env_px"] = LQ.env_px(axis, s_in, 640, 480)
    man["views"].append(e)
json.dump(man, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(man, indent=1, ensure_ascii=False))
