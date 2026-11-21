"""生成第二阶段冻结清单候选 FROZEN_bench2_cogact.candidate（json）。由第一阶段 b2/freeze_manifest.py 改写。
只生成 candidate，不生成 FROZEN_bench2_cogact（协调方核对后改名）。

内容：
  * code_sha256：eval_views.py、views.py、keys.py、gate.py、readout.py、submit_formal.sh、jobs/common.sh；
  * keylist_sha256：四份正式键清单（必须与第一阶段 FROZEN_bench2 里的逐一相同，补充协议 §2）；
    phase1_frozen_sha256：第一阶段冻结清单本身的哈希（只读引用）；
  * upstream_sha256：上游文件哈希（CogACT 推理代码、prismatic 骨干、官方 SimplerEnv 入口、分词器文件），
    取自冒烟与校准记录，所有记录必须一致；
  * policy_cfg：两种 policy_setup 下 CogACT 实际生效的配置（反归一化键、cfg、DDIM 步数、集成窗口、精度、检查点路径与大小等），
    取自记录，所有记录必须一致；
  * blocks：各块的任务、键数、L（calib.json 的 L_final：第一阶段的步数，按 IoU 规则核查后）、线宽、查询间隔 1、
    步数上限、控制频率、G 连杆、保护名单、块级标记 blacken_weak / lag_weak（读数脚本从这里取）、policy_setup；
  * 渲染用的 Vulkan ICD；
  * 卡型（协调方 2026-11-13）：gpu_models（允许的型号）与 gpu_assignment（每块每键分到的型号，keys.gpu_assignment：
    GPU_PATTERN[键在冻结键清单里的序号 % 长度]，只依赖序号）。门槛逐键核对记录的型号等于分到的型号、同键五格同型号。
gate.py 用采集代码哈希、上游哈希、策略配置、L、线宽、G、保护名单、步数上限、控制频率、查询间隔核对每条记录；
submit_formal.sh 提交前核对 code_sha256 与键清单；readout.py 核对键清单哈希、键数并取块级标记。
用法：python freeze_manifest.py [--calib calib.json] [--flags calib_flags.json] [--records out_calib out_smoke]
      [--out FROZEN_bench2_cogact.candidate]
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

B2C = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2C)
import gate as G  # noqa: E402
import gpu_groups as GG  # noqa: E402
import keys as K  # noqa: E402
import views as V  # noqa: E402

CODE_FILES = ("eval_views.py", "views.py", "keys.py", "gate.py", "readout.py", "submit_formal.sh", "jobs/common.sh",
              "gpu_groups.py", "icd_probe.py")
PHASE1_FROZEN = os.path.expanduser("$PROJ_ROOT/bench2_simpler/b2/FROZEN_bench2")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=os.path.join(B2C, "calib.json"))
    ap.add_argument("--flags", default=os.path.join(B2C, "calib_flags.json"))
    ap.add_argument("--records", nargs="+", default=[os.path.join(B2C, d) for d in (
        "out_calib", "out_calib_aa", "out_smoke", "out_calib_h100", "out_calib_h100_aa", "out_smoke_h100_v2")])
    ap.add_argument("--phase1-frozen", default=PHASE1_FROZEN)
    ap.add_argument("--out", default=os.path.join(B2C, "FROZEN_bench2_cogact.candidate"))
    a = ap.parse_args()
    if os.path.basename(a.out) == "FROZEN_bench2_cogact":
        sys.exit("拒绝：本脚本只生成 candidate，FROZEN_bench2_cogact 由协调方改名")
    calib = json.load(open(a.calib))
    flags = json.load(open(a.flags))
    p1 = json.load(open(a.phase1_frozen))
    # 上游哈希与策略配置：从记录里取，全部记录必须一致（按 policy_setup 分）
    ups, pcfg, n_rec = set(), {}, 0
    for root in a.records:
        for p in glob.glob(os.path.join(root, "*", "*", "*.npz")):
            with np.load(p, allow_pickle=False) as z:
                ups.add(str(z["upstream_sha"]))
                pc = json.loads(str(z["policy_cfg"]))
                pcfg.setdefault(pc["policy_setup"], set()).add(json.dumps(pc, sort_keys=True))
                n_rec += 1
    assert n_rec > 0, "没有记录"
    assert len(ups) == 1, f"记录里的上游哈希不止一种：{len(ups)}"
    for s_, v in pcfg.items():
        assert len(v) == 1, f"{s_} 的策略配置不止一种：{v}"
    man = {"note": "bench2 第二阶段（CogACT）冻结清单候选；协调方核对后改名为 FROZEN_bench2_cogact",
           "code_sha256": {f: sha(os.path.join(B2C, f)) for f in CODE_FILES},
           "keylist_sha256": {}, "phase1_frozen_sha256": sha(a.phase1_frozen),
           "upstream_sha256": json.loads(next(iter(ups))),
           "policy_cfg": {s_: json.loads(next(iter(v))) for s_, v in sorted(pcfg.items())},
           "records_used_for_upstream_and_policy": n_rec,
           "blocks": {}, "vk_icd_filenames": G.VK_ICD_NVIDIA,
           "gpu_models": sorted(GG.GPU_MODELS.values()),
           "gpu_assignment_rule": f"型号 = GPU_MODELS[GPU_PATTERN[i % {len(GG.GPU_PATTERN)}]]，i = 键在 keysets/formal_<块>.json 里的序号，"
                                  f"GPU_PATTERN = {GG.GPU_PATTERN!r}，GPU_MODELS = {GG.GPU_MODELS}",
           "gpu_assignment": {}, "gpu_assignment_counts": {}}
    for b in K.BLOCKS:
        kp = os.path.join(B2C, "keysets", f"formal_{b}.json")
        ks = json.load(open(kp))
        assert [k["key"] for k in ks] == [k["key"] for k in K.make_keys(b, "formal")], f"{kp} 与 keys.py 不一致"
        h = sha(kp)
        assert h == p1["keylist_sha256"][b], f"{b} 键清单与第一阶段冻结的不同"
        man["keylist_sha256"][b] = h
        man["gpu_assignment"][b] = GG.gpu_assignment(ks)
        assert set(man["gpu_assignment"][b]) == {k["key"] for k in ks}, b
        man["gpu_assignment_counts"][b] = {m: sum(1 for v in man["gpu_assignment"][b].values() if v == m)
                                           for m in man["gpu_models"]}
        c = calib[b]
        fl = flags[b]
        blk = K.BLOCKS[b]
        assert int(fl["L"]) == int(c["L_final"]), (b, fl, c["L_final"])
        assert bool(fl["blacken_weak"]) == bool(c["blacken_weak"]) and bool(fl["lag_weak"]) == bool(c["lag_weak"]), b
        assert int(c["L"]) == int(blk["L"]), (b, c["L"], blk["L"])
        assert int(c["query_every"]) == K.QUERY_EVERY and int(c["max_episode_steps"]) == int(blk["max_episode_steps"]), b
        assert float(c["control_freq"]) == float(blk["control_freq"]), b
        body = blk["body"]
        assert c["g_links"] == V.G_LINKS[body], b
        assert c["policy_cfg"] == man["policy_cfg"][blk["policy_setup"]], b
        man["blocks"][b] = {"tasks": blk["tasks"], "n_keys": len(ks), "L": int(c["L_final"]),
                            "L_phase1_steps": int(blk["L"]), "iou_lag_outline_median": c["iou_lag_outline_median"],
                            "iou_lag2_outline_median": c["iou_lag2_outline_median"],
                            "line_width": int(c["line_width"]), "query_every": int(c["query_every"]),
                            "control_freq": c["control_freq"], "sim_freq": c["sim_freq"],
                            "max_episode_steps": int(c["max_episode_steps"]), "policy_setup": blk["policy_setup"],
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
