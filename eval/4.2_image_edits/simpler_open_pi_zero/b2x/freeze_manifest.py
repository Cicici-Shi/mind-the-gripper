"""生成补充协议 v1 的冻结清单候选 FROZEN_bench2_ext.candidate（json）。只生成 candidate，不生成 FROZEN_bench2_ext。

内容：
  * b2x 的 eval_views.py、views.py、keys.py、gate.py、readout.py、submit_formal.sh、jobs/common.sh 的 sha256；
  * 两份正式键清单 keysets/formal_B2.json、formal_F1.json 的 sha256（核对与第一阶段冻结的同名文件逐字节相同）；
  * 各块：第一阶段冻结清单的 L、线宽、G、保护名单、键数等原样继承；本扩展的格；按格的滞后 cell_lag（C5 = L，
    C5L = 更长滞后）；三种填充的颜色（k 黑、w 白、L 浅色；补充协议 v1.1；两块颜色相同）；参考轨迹上的操纵强度（F1 另按 urdf）、
    环反差与「低反差」标记；F1 的滞后几何量（4、8、12 步，pre / full / all）、更长滞后的选取依据、
    开头轮廓不动的那一段；校准键重跑与第一阶段逐位比对的结果（全部来自 calib_ext.json）；
  * FL 的取法核对数据（逐通道中位数、最近真实颜色的距离、峰数）；渲染 ICD；第一阶段冻结清单的 sha256（溯源）。
参数：--lag-long 不给时取 calib_ext.json 按规则选出的值（协调方可另定）。填充颜色全部取自 calib_ext.json（白色在 ext_calib.py 里定）。
用法：python freeze_manifest.py [--calib-ext calib_ext.json] [--lag-long N] [--out FROZEN_bench2_ext.candidate]
"""
import argparse
import hashlib
import json
import os
import sys

B2X = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.join(os.path.dirname(B2X), "b2")
sys.path.insert(0, B2X)
import gate as G  # noqa: E402
import keys as K  # noqa: E402

CODE_FILES = ("eval_views.py", "views.py", "keys.py", "gate.py", "readout.py", "submit_formal.sh", "jobs/common.sh")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-ext", default=os.path.join(B2X, "calib_ext.json"))
    ap.add_argument("--phase1", default=os.path.join(B2, "FROZEN_bench2"))
    ap.add_argument("--lag-long", type=int, default=None, help="C5L 的滞后；不给就用 calib_ext.json 的规则选择")
    ap.add_argument("--out", default=os.path.join(B2X, "FROZEN_bench2_ext.candidate"))
    a = ap.parse_args()
    if os.path.basename(a.out) == "FROZEN_bench2_ext":
        sys.exit("拒绝：本脚本只生成 candidate，FROZEN_bench2_ext 由协调方改名")
    p1 = json.load(open(a.phase1))
    ce = json.load(open(a.calib_ext))
    assert set(ce["fills"]) == {"k", "w", "L"}, ("calib_ext.json 的填充不是 k / w / L，重跑 ext_calib.py", ce["fills"])
    rule = ce["blocks"]["F1"]["lag_long_rule"]
    lag_long = a.lag_long if a.lag_long is not None else int(rule["chosen"])
    man = {"note": "bench2 补充协议 v1.1（填充颜色与更长滞后）冻结清单候选；协调方核对后改名为 FROZEN_bench2_ext",
           "phase1_frozen_sha256": sha(a.phase1),
           "code_sha256": {f: sha(os.path.join(B2X, f)) for f in CODE_FILES},
           "keylist_sha256": {}, "blocks": {},
           "FL": {k: v for k, v in ce["FL"].items() if k != "channel_hist"},
           "fills": ce["fills"], "lag_long": lag_long, "lag_long_rule": rule,
           "lag_long_source": "规则选择" if a.lag_long is None else "参数指定",
           "vk_icd_filenames": G.VK_ICD_NVIDIA, "ref_root_phase1": G.REF_ROOT}
    for b in K.EXT_CELLS:
        kp = os.path.join(B2X, "keysets", f"formal_{b}.json")
        assert sha(kp) == p1["keylist_sha256"][b], f"{kp} 与第一阶段冻结的键清单不同"
        man["keylist_sha256"][b] = sha(kp)
        mb = {k: v for k, v in p1["blocks"][b].items()
              if k in ("tasks", "n_keys", "L", "line_width", "query_every", "control_freq", "max_episode_steps",
                       "g_links", "protect", "protect_expect_n")}
        c = ce["blocks"][b]
        cell_lag = {"C5": int(mb["L"])}
        if "C5L" in K.EXT_CELLS[b]:
            cell_lag["C5L"] = lag_long
        mb.update(cells=K.EXT_CELLS[b], cell_lag=cell_lag, fill_rgb=ce["fills"],
                  fill_strength=c["fill_strength"], ring_contrast=c["ring_contrast"], low_contrast=c["low_contrast"],
                  calib_rerun_vs_phase1=c["rerun_vs_phase1_calib"])
        if "fill_strength_by_urdf" in c:
            mb["fill_strength_by_urdf"] = c["fill_strength_by_urdf"]
        if "geometry" in c:
            for l in (mb["L"], lag_long):
                assert str(l) in c["geometry"], f"calib_ext.json 缺滞后 {l} 的几何量"
            mb.update(geom_ref=c["geometry"], frozen_segment=c["frozen_segment"],
                      first_success_steps_calib=c["first_success_steps"])
        man["blocks"][b] = mb
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
