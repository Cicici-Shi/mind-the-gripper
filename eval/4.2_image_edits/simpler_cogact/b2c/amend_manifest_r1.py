"""一次性脚本：按补充协议「修订 R1」（2026-11-14）裁定五第 3 步，由原冻结清单生成修订清单候选。
读 FROZEN_bench2_cogact.orig_5ef81795（原清单，只读），只改五处，写 FROZEN_bench2_cogact.r1.candidate：
  gpu_assignment（四块全部键 -> NVIDIA A100-PCIE-40GB）、gpu_assignment_counts（A100 120、H100 0）、
  gpu_assignment_rule、note（末尾加「；修订 R1」）、新增 amendment_r1。其余字段（含 code_sha256、gpu_models）一个不动。
不替换 FROZEN_bench2_cogact：由协调方核对后替换。不用 freeze_manifest.py（它从 gpu_groups.py 取分配）。
本文件不在清单哈希里。用法：python amend_manifest_r1.py
"""
import hashlib
import json
import os
import sys

B2C = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(B2C, "FROZEN_bench2_cogact.orig_5ef81795")
OUT = os.path.join(B2C, "FROZEN_bench2_cogact.r1.candidate")
ORIG_SHA = "5ef81795e84280ce2caced84831dffa7c0ba29450ae59c96fb6996b41112826d"
PROTOCOL_R1_SHA = "462d8530c709c350494a5fdceaf272f7538013538820a0314b64d0bf36385f23"  # 协调方给的修订后协议哈希
A100 = "NVIDIA A100-PCIE-40GB"
H100 = "NVIDIA H100 NVL"
CHANGED = {"gpu_assignment", "gpu_assignment_counts", "gpu_assignment_rule", "note", "amendment_r1"}


def main():
    raw = open(ORIG, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == ORIG_SHA, "原清单哈希不符"
    if os.path.exists(OUT):
        sys.exit(f"拒绝：{OUT} 已存在")
    orig = json.loads(raw)
    assert "amendment_r1" not in orig
    assert set(orig["gpu_models"]) == {A100, H100}
    man = json.loads(raw)  # 深拷贝，保持原字段顺序
    man["gpu_assignment"] = {b: {k: A100 for k in amap} for b, amap in orig["gpu_assignment"].items()}
    man["gpu_assignment_counts"] = {b: {A100: len(amap), H100: 0} for b, amap in orig["gpu_assignment"].items()}
    man["gpu_assignment_rule"] = "修订 R1（2026-11-14）：全部键分给 NVIDIA A100-PCIE-40GB；原规则见 FROZEN_bench2_cogact.orig_5ef81795"
    man["note"] = orig["note"] + "；修订 R1"
    man["amendment_r1"] = {"date": "2026-11-14",
                           "protocol": "protocol_bench2_phase2_cogact.md 末尾「修订 R1」",
                           "protocol_sha256_after_r1": PROTOCOL_R1_SHA,
                           "orig_manifest": "FROZEN_bench2_cogact.orig_5ef81795",
                           "orig_manifest_sha256": ORIG_SHA,
                           "void_dir": "out_formal_void_h100",
                           "void_list": "formal_state/void_h100_r1.tsv"}
    # 验收：除五处外逐字段相同；分配覆盖的键集合不变
    for k in set(orig) | set(man):
        if k not in CHANGED:
            assert orig.get(k) == man.get(k), k
    for b in orig["gpu_assignment"]:
        assert set(orig["gpu_assignment"][b]) == set(man["gpu_assignment"][b]), b
        assert len(man["gpu_assignment"][b]) == orig["blocks"][b]["n_keys"] == 120, b
    with open(OUT + ".tmp", "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(hashlib.sha256(open(OUT, "rb").read()).hexdigest(), OUT)


if __name__ == "__main__":
    main()
