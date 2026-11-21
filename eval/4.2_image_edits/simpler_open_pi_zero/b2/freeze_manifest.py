"""生成冻结清单候选 FROZEN_bench2.candidate（json）。只生成 candidate，不生成 FROZEN_bench2（协调方核对后改名）。

内容：eval_views.py、views.py、keys.py、gate.py、readout.py、submit_formal.sh、jobs/common.sh 的 sha256；
四份正式键清单 keysets/formal_<块>.json 的 sha256；各块的 L（calib.json 的 L_final，按协议规则由校准 C0 回合的轮廓 IoU 定）、
线宽、G 连杆清单、保护名单、键数，以及块级标记 blacken_weak / lag_weak（calib_flags.json，读数脚本直接从这里取）；
渲染用的 Vulkan ICD。gate.py 只用其中采集代码（eval_views / views / keys）的哈希与 L、线宽、G、保护名单核对每条记录；
submit_formal.sh 提交前核对 code_sha256 里全部文件与键清单的哈希；readout.py 核对键清单哈希、键数并取块级标记。
用法：python freeze_manifest.py [--calib calib.json] [--flags calib_flags.json] [--out FROZEN_bench2.candidate]
"""
import argparse
import hashlib
import json
import os
import sys

B2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2)
import gate as G  # noqa: E402
import keys as K  # noqa: E402
import views as V  # noqa: E402

CODE_FILES = ("eval_views.py", "views.py", "keys.py", "gate.py", "readout.py", "submit_formal.sh", "jobs/common.sh")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=os.path.join(B2, "calib.json"))
    ap.add_argument("--flags", default=os.path.join(B2, "calib_flags.json"))
    ap.add_argument("--out", default=os.path.join(B2, "FROZEN_bench2.candidate"))
    a = ap.parse_args()
    if os.path.basename(a.out) == "FROZEN_bench2":
        sys.exit("拒绝：本脚本只生成 candidate，FROZEN_bench2 由协调方改名")
    calib = json.load(open(a.calib))
    flags = json.load(open(a.flags))
    man = {"note": "bench2 冻结清单候选；协调方核对后改名为 FROZEN_bench2",
           "code_sha256": {f: sha(os.path.join(B2, f)) for f in CODE_FILES},
           "keylist_sha256": {}, "blocks": {}, "vk_icd_filenames": G.VK_ICD_NVIDIA}
    for b in K.BLOCKS:
        kp = os.path.join(B2, "keysets", f"formal_{b}.json")
        ks = json.load(open(kp))
        assert [k["key"] for k in ks] == [k["key"] for k in K.make_keys(b, "formal")], f"{kp} 与 keys.py 不一致"
        man["keylist_sha256"][b] = sha(kp)
        c = calib[b]
        fl = flags[b]
        assert int(fl["L"]) == int(c["L_final"]), (b, fl, c["L_final"])
        assert bool(fl["blacken_weak"]) == bool(c["blacken_weak"]) and bool(fl["lag_weak"]) == bool(c["lag_weak"]), b
        body = K.BLOCKS[b]["body"]
        assert c["g_links"] == V.G_LINKS[body], b
        man["blocks"][b] = {"tasks": K.BLOCKS[b]["tasks"], "n_keys": len(ks), "L": int(c["L_final"]),
                            "L_rule_base": int(c["L"]), "iou_lag_outline_median": c["iou_lag_outline_median"],
                            "line_width": int(c["line_width"]), "query_every": int(c["query_every"]),
                            "control_freq": c["control_freq"], "max_episode_steps": int(c["max_episode_steps"]),
                            "g_links": V.G_LINKS[body], "protect": K.PROTECT[b],
                            "protect_expect_n": K.PROTECT_EXPECT_N[b],
                            "blacken_weak": bool(fl["blacken_weak"]), "blacken_strength": c["blacken_strength"],
                            "lag_weak": bool(fl["lag_weak"])}
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
