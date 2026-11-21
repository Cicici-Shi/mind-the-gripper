"""生成冻结清单候选 FROZEN_cue_widowx.candidate（protocol_cue_widowx_v3_1_final.md §14、§15；§17 第 1 条）。
只写 .candidate；正式清单 FROZEN_cue_widowx 由协调方在实现审查通过后核对改名，本脚本永不创建它。
  python freeze_cue.py --protocol <协议文件> --protocol-sha256 <协调方给的值> \
      --keyset B1=keysets/cue_B1.json --keyset B2=keysets/cue_B2.json \
      --anchor-keyset B1=keysets/anchor_B1.json --anchor-keyset B2=keysets/anchor_B2.json \
      --tref tref_cue_widowx_v31.json --localq0 cue_localq0.json --measured cue_measured_v31.json
协议文件不存在、或算出的 sha256 与 --protocol-sha256 不等，拒绝写清单。
主键集的每个键试次只能是 5..8；锚点键集 = 第一阶段正式清单前 10 个键（逐字段相同）；t_ref 清单必须覆盖主键集每个键。
"""
import argparse
import hashlib
import json
import os
import sys

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import cue_spec as CS  # noqa: E402

CODE = ["cue_closed.py", "cue_obcheck.py", "views.py", "eval_views.py", "keys.py", "cue_spec.py", "cue_cmp.py", "cue_status.py",
        "cue_gate.py", "cue_readout.py", "test_cue_readout.py", "cue_tref_resol.py", "submit_cue.sh", "jobs/common.sh"]
# 实现审查建议改第 4 条：test_cue_readout.py 作为冻结条件第 6 项的证据登记；权重与 open-pi-zero 提交号见 --model-info


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def parse_ks(items, name):
    d = dict(x.split("=", 1) for x in items)
    assert set(d) == set(CS.BLOCKS), f"{name} 必须给 B1、B2 两块"
    return {b: os.path.abspath(p) for b, p in d.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--keyset", action="append", required=True)
    ap.add_argument("--anchor-keyset", action="append", required=True)
    ap.add_argument("--tref", required=True)
    ap.add_argument("--localq0", required=True)
    ap.add_argument("--measured", default=None, help="§14 的实测值汇总（照报，不进任何判定）")
    ap.add_argument("--reachability", default=None, help="§14 冻结条件第 3、4 项的标注（json）")
    ap.add_argument("--model-info", required=True, help="权重文件 sha256、open-pi-zero 提交号等（cue_model_info.json）")
    ap.add_argument("--phase1-keyset-dir", default=os.path.join(B3, "..", "b2", "keysets"))
    ap.add_argument("--out", default=os.path.join(B3, "FROZEN_cue_widowx.candidate"))
    a = ap.parse_args(argv)
    assert not os.path.normpath(a.out).endswith("FROZEN_cue_widowx"), "本脚本只写候选"
    if not os.path.exists(a.protocol):
        raise SystemExit(f"拒绝：协议文件不存在 {a.protocol}")
    got = sha(a.protocol)
    if got != a.protocol_sha256:
        raise SystemExit(f"拒绝：协议文件哈希 {got} 与给定 {a.protocol_sha256} 不符")
    missing = [f for f in CODE if not os.path.exists(os.path.join(B3, f))]
    if missing:
        raise SystemExit(f"拒绝：缺代码文件 {missing}")
    kp, ap_ = parse_ks(a.keyset, "--keyset"), parse_ks(a.anchor_keyset, "--anchor-keyset")
    tk = json.load(open(a.tref))
    for b in CS.BLOCKS:
        ks = json.load(open(kp[b]))
        assert all(k["block"] == b and int(k["trial"]) in (5, 6, 7, 8) for k in ks), f"主键集 {b} 的试次只能是 5..8"
        miss = [k["key"] for k in ks if k["key"] not in tk.get(b, {})]
        assert not miss, f"t_ref 清单缺 {b} 的 {len(miss)} 个键"
        an = json.load(open(ap_[b]))
        ph = json.load(open(os.path.join(a.phase1_keyset_dir, f"formal_{b}.json")))
        assert an == ph[:CS.STAGE1_N], f"锚点键集 {b} 须与第一阶段正式清单前 {CS.STAGE1_N} 个键逐字段相同"
    rel = lambda p: os.path.relpath(p, B3)
    man = {
        "note": "protocol_cue_widowx v3.1 冻结清单候选；协调方在实现审查通过后核对并改名为 FROZEN_cue_widowx",
        "protocol_file": os.path.basename(a.protocol), "protocol_sha256": got,
        "code_sha256": {f: sha(os.path.join(B3, f)) for f in CODE},
        "keysets": {b: {"path": rel(p), "sha256": sha(p), "n": len(json.load(open(p)))} for b, p in kp.items()},
        "anchor_keysets": {b: {"path": rel(p), "sha256": sha(p), "n": len(json.load(open(p)))} for b, p in ap_.items()},
        "tref_file": rel(os.path.abspath(a.tref)), "tref_sha256": sha(a.tref),
        "cells": {c: {"kind": k, "shift_env": list(s)} for c, (k, s) in CS.CELLS.items()},
        "cell_order": CS.CELL_ORDER, "anchor_cells": CS.ANCHOR_CELLS, "stage1_n": CS.STAGE1_N,
        "target": CS.TARGET, "container": CS.CONTAINER, "tcp_link": CS.TCP_LINK,
        "amp_in": CS.AMP_IN, "amp_env": CS.AMP_ENV, "s_tilde": CS.S_TILDE,
        "params": dict(CS.PARAMS, localq_q0=json.load(open(a.localq0))),
        "reachability": json.load(open(a.reachability)) if a.reachability else None,
        "line_width": CS.LINE_WIDTH, "query_every": CS.QUERY_EVERY, "ring_px": CS.RING_PX,
        "vk_icd_filenames": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        # 实现审查建议改第 3 条：删掉 SXM4（锚点与第一阶段记录都只在 PCIE 上，SXM4 没有过锚点比对；作业也排除了 node23 节点）
        "gpu_whitelist": ["NVIDIA A100-PCIE-40GB"],
        "model": json.load(open(a.model_info)), "model_info_sha256": sha(a.model_info),
        "ref_root_phase1": "$PROJ_ROOT/bench2_simpler/b2/out_formal",  # 不展开，清单与生成它的机器无关
        "measured_14": json.load(open(a.measured)) if a.measured else None,
    }
    s = json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True)
    open(a.out, "w").write(s)
    print(a.out, sha(a.out))


if __name__ == "__main__":
    main()
