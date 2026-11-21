"""第二阶段卡型分组（协调方 2026-11-13：可以多种卡型一起用，但同一个键的五格必须在同一型号上跑，配对比较才成立）。
单独成文件，不放进 keys.py：keys.py 的哈希记在每条记录里，冒烟与校准记录是用不含分组的 keys.py 跑的。
本文件的哈希记进冻结清单 code_sha256，驱动脚本提交前核对；分组结果（每键的型号）写进冻结清单 gpu_assignment，门槛逐键核对。

型号按 torch.cuda.get_device_name 的字符串认；gres h100-47 与 h100-96 是同一型号 H100 NVL。
键 -> 型号只由键在冻结键清单（keysets/formal_<块>.json）里的序号 i 定：GPU_MODELS[GPU_PATTERN[i % len(GPU_PATTERN)]]，
与成败、场景、格无关。比例按实测吞吐定（见交回报告 REPORT_cogact_phase2_impl_2026-11-13.md 第 7 节）：
  * 份额：A100 份额 = A100 池每小时键数 / 全部池每小时键数。按协调方 2026-11-13 的占卡规则：A100-40 至多 1 张
    （WidowX 主实验同时在用），H100 只用 node32（h100-47，按两张一进程计）。实测一个进程每键五格：A100-40 约 148 秒
    （约 24 键/时），h100-47 约 123 到 141 秒（作业 <job> 推算 / 作业 <job> 定稿冒烟实测，约 25 到 29 键/时，
    两张约 51 到 59 键/时），A100 份额约 0.29 到 0.32；
  * 周期须与键清单的结构互素：B1、B2 的序号 i = 试次 x 24 + 配置，F1 是 3 个任务各 40 键，F2 是 60 个聚类各 2 键。
    周期 3、4、6 整除 24，会把 B 块的 A100 键压在少数几个配置上（周期 6 时全在 4 个配置上），所以不用；
    周期 7 与 24、40、60 都互素。"AHHAHHH"（2/7 = 0.286）每块 35 个 A100 键，B 块落在 21/24 个配置上、
    每配置至多 2 个，F1 三个任务 12/11/12，F2 每聚类至多 2 个。
  * 2026-11-13 中间曾按「h100-96 也可用」改成 "AHHHHHH"，协调方随后定 H100 只用 node32，改回本值。
"""
GPU_MODELS = {"A": "NVIDIA A100-PCIE-40GB", "H": "NVIDIA H100 NVL"}
GPU_PATTERN = "AHHAHHH"


def gpu_model_of(index):
    return GPU_MODELS[GPU_PATTERN[index % len(GPU_PATTERN)]]


def gpu_assignment(keylist):
    """冻结键清单（按文件里的顺序）-> {键名: 型号}。"""
    return {k["key"]: gpu_model_of(i) for i, k in enumerate(keylist)}
