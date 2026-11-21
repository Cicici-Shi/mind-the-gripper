# 第三批提交命令清单（按 protocol_block3.md 冻结版，2026-10-27；未提交，_cg 除外）

协议：`protocols/protocol_block3.md`（sha256 3875470456102d60668a02f6a7ff0610faf2b32a15f8112821fff52781690616）。
代码：`$PROJ_ROOT/embody_v3`。所有命令都带 `EMBODY_DIR=$PROJ_ROOT/embody_v3` 和 `COPY_GRIP_CMD=1`（协议 §0.4：新格一律复制，对照 S = sham_cg）。
评测脚本已加在线接触记录（`run_libero_eval.py`、`pi05_body_eval.py`，备份 `.bak_20261027`），新的 dump / jsonl 自动带 `contact_*` 字段。

**GPU 一律写型号**：`--gres=gpu:a100-80:1`、`gpu:a100-40:1` 或 `gpu:h200-141:1`，再加 `--exclude='node39,node38'`。
裸 `gpu:1` 会落到 Titan V / T4（12 到 16 GB）：OFT 内存不足、π0.5 无 GPU，而作业状态仍是 COMPLETED（_cg 首轮就是这样零数据）。
每人按卡型限并发：a100-40 ×2、a100-80 ×1、h200 ×1。下面的函数默认 a100-40，第三个参数可换型号。

## 状态总表

| 格 | 状态 | 依据 |
|---|---|---|
| sham_cg、iiwa_cg、ur5e_cg（OFT 12 + π0.5 3） | **已提交**（协调者按型号重提，15 个作业） | 附录 A |
| P6j3、K3、G_Tw、G_Ta | **可立即提交** | 上一轮冒烟已过；本轮 CPU 回放与接触预测已冻结 |
| G_R（沉装） | **先别提交，等决定** | 指垫残差门槛已过（查询帧 > 1 mm 0/410）；但「第 0 步对 G 的 agentview 平均差 < 1.0」不过（1.68 到 1.82），原因见 WORKLOG「第三批：冻结协议后的实现（10/27）」 |
| UR5e_nc | **可立即提交** | 操纵在回放上生效（任务 3 20/20 穿过柜子），任务 3 以外回放与 UR5e 相同 |
| RT2 | **可立即提交** | CPU 回放 73.2% ≥ 60% 门槛 |
| RT2_T、K3_T | **条件触发**：对应身体 R1 判为 L 类后才提交 | 协议 §1.1 |
| libero_object 7 格 | **值得层第 2 位**，先各跑 SRC 20 集核基线（协议 §6） | 入口两个模型各 1 集已通 |
| 加密到 400 | **条件触发**：P10 判「不确定」时 | 协议 §5 |

## 函数

```bash
V=$PROJ_ROOT/embody_v3
XC='node39,node38'
sub_oft() {  # sub_oft NOTE BODY "EXTRA=..,.." [GPU型号，默认 a100-40] [片，默认 "0 5 10 15"]
  local gpu=${4:-a100-40} offs=${5:-"0 5 10 15"}
  for off in $offs; do
    sbatch --job-name=cond_$1_e$off --gres=gpu:$gpu:1 --exclude="$XC" \
      --export=ALL,NOTE=$1,BODY=$2,TRIALS=5,EP_OFFSET=$off,SKIP_DONE=1,EMBODY_DIR=$V,COPY_GRIP_CMD=1${3:+,$3} \
      ~/run_cond.sbatch
  done
}
sub_pi() {  # sub_pi NAME BODY "EXTRA=.." [GPU型号，默认 a100-40]
  sbatch --job-name=pi05_$1 --gres=gpu:${4:-a100-40}:1 --exclude="$XC" \
    --export=ALL,NAME=$1,BODY=$2,HIDE=0,EMBODY_DIR=$V,COPY_GRIP_CMD=1${3:+,$3} \
    $PROJ_ROOT/pi05/run_pi05_v2.sbatch
}
```

注意：`~/run_cond.sbatch` 固定 `--task_suite_name libero_spatial`，object 用下面单独的 sbatch（见 libero_object 一节）；SRC 不设 BODY（`run_cond.sbatch` 里 BODY 为空即原生）。

## 必须层（spatial，可立即提交）

```bash
# OFT（结果 $PROJ_ROOT/runs/<NOTE>/）
sub_oft p6j3        PandaL3
sub_oft k3          Kinova3
sub_oft g_r         Panda    "TWIN_RENDER=1,TWIN_BODY=PandaR85,TWIN_MOUNT=sunk"
sub_oft grpkp100_tw PandaR85 "GRP_KP=100,TWIN_RENDER=1,TWIN_CAMS=wrist"
sub_oft grpkp100_ta PandaR85 "GRP_KP=100,TWIN_RENDER=1,TWIN_CAMS=agent"
sub_oft ur5e_nc     UR5e     "NO_FOREARM_CONTACT=1"
sub_oft rt2         PandaRT2
# π0.5（结果 $PROJ_ROOT/pi05/v2/<NAME>_e*.jsonl）
sub_pi p6j3        PandaL3
sub_pi k3          Kinova3
sub_pi g_r         Panda    "TWIN_RENDER=1,TWIN_BODY=PandaR85,TWIN_MOUNT=sunk"
sub_pi grpkp100_tw PandaR85 "GRP_KP=100,TWIN_RENDER=1,TWIN_CAMS=wrist"
sub_pi grpkp100_ta PandaR85 "GRP_KP=100,TWIN_RENDER=1,TWIN_CAMS=agent"
sub_pi ur5e_nc     UR5e     "NO_FOREARM_CONTACT=1"
sub_pi rt2         PandaRT2
```
OFT 28 个作业、π0.5 7 个。第 0 片先停看门槛（协议 §4.2：替身格同键一致率、G_R 指垫残差 > 1 mm 帧 < 1%、UR5e_nc 任务 3 ≥ 15/20 集 `contact_forearm_pen_steps > 0`），再放 5/10/15 片：可以先只提交 `sub_oft X BODY "..." a100-40 "0"`。

## 条件触发（spatial）

```bash
# 只在 R1 把 RT2 判为 L 类后
sub_oft rt2_twin PandaRT2 "TWIN_RENDER=1";  sub_pi rt2_twin PandaRT2 "TWIN_RENDER=1"
# 只在 R1 把 K3 判为 L 类后（受 1% 查询帧 IK 门槛约束）
sub_oft k3_twin  Kinova3  "TWIN_RENDER=1";  sub_pi k3_twin  Kinova3  "TWIN_RENDER=1"
# 只在 P10 判「不确定」后（协议 §5，仅 OFT，加 SRC）
for c in "sham_cg Panda" "sham_twin Panda TWIN_RENDER=1" "grpkp100 PandaR85 GRP_KP=100" \
         "grpkp100_twin PandaR85 GRP_KP=100,TWIN_RENDER=1" "g_r Panda TWIN_RENDER=1,TWIN_BODY=PandaR85,TWIN_MOUNT=sunk" \
         "grpkp100_tw PandaR85 GRP_KP=100,TWIN_RENDER=1,TWIN_CAMS=wrist" "grpkp100_ta PandaR85 GRP_KP=100,TWIN_RENDER=1,TWIN_CAMS=agent"; do
  set -- $c; sub_oft $1 $2 "$3" a100-40 "20 25 30 35"
done
# SRC 加密：BODY 为空
for off in 20 25 30 35; do sbatch --job-name=cond_src_e$off --gres=gpu:a100-40:1 --exclude="$XC" \
  --export=ALL,NOTE=src,BODY=,TRIALS=5,EP_OFFSET=$off,SKIP_DONE=1,EMBODY_DIR=$V ~/run_cond.sbatch; done
```
注意加密时 sham_twin、grpkp100、grpkp100_twin 的新片用合并版跑，与它们原来的 dev 版前 200 集在 COPY_GRIP_CMD 上一致（替身与 Robotiq 格本来就复制），合并版对这三格逐步相同（回归 0 差）。

## libero_object（值得层第 2 位；协议 §6）

`~/run_cond.sbatch` 写死了 spatial，用 `$PROJ_ROOT/embody_v3/tools/run_cond_object.sbatch`（与 run_cond.sbatch 只差 `--task_suite_name libero_object`，结果写 `$PROJ_ROOT/runs/obj_<NOTE>/`）；π0.5 用 `$PROJ_ROOT/embody_v3/tools/run_pi05_object.sbatch`（`--suite libero_object`，结果 `$PROJ_ROOT/pi05/v2/obj_<NAME>_e*.jsonl`）。

```bash
sub_obj() {  # sub_obj NOTE BODY "EXTRA" [片]
  for off in ${4:-"0 5 10 15"}; do
    sbatch --job-name=cond_obj_$1_e$off --gres=gpu:a100-40:1 --exclude="$XC" \
      --export=ALL,NOTE=$1,BODY=$2,TRIALS=5,EP_OFFSET=$off,SKIP_DONE=1,EMBODY_DIR=$V,COPY_GRIP_CMD=1${3:+,$3} \
      $V/tools/run_cond_object.sbatch
  done
}
sub_piobj() { sbatch --job-name=pi05_obj_$1 --gres=gpu:a100-40:1 --exclude="$XC" \
  --export=ALL,NAME=$1,BODY=$2,HIDE=0,EMBODY_DIR=$V,COPY_GRIP_CMD=1${3:+,$3} $V/tools/run_pi05_object.sbatch; }
# 第一步：SRC 基线 20 集（每任务 2 集，EP_OFFSET 0，TRIALS 2）——低于 90% 先查接口，不放后面的格
sbatch --job-name=cond_obj_src_base --gres=gpu:a100-40:1 --exclude="$XC" \
  --export=ALL,NOTE=src,BODY=,TRIALS=2,EP_OFFSET=0,SKIP_DONE=1,EMBODY_DIR=$V $V/tools/run_cond_object.sbatch
sbatch --job-name=pi05_obj_src_base --gres=gpu:a100-40:1 --exclude="$XC" \
  --export=ALL,NAME=src_base,BODY=SRC,HIDE=0,EMBODY_DIR=$V,TRIALS=2,OFFSETS=0 $V/tools/run_pi05_object.sbatch
# 第二步（基线过了）：7 格 × 200 集
sub_obj src ""          ;  sub_piobj src SRC
sub_obj s     Panda     ;  sub_piobj s     Panda
sub_obj t0    Panda    "TWIN_RENDER=1"                        ;  sub_piobj t0    Panda    "TWIN_RENDER=1"
sub_obj ur5e  UR5e      ;  sub_piobj ur5e  UR5e
sub_obj k3    Kinova3   ;  sub_piobj k3    Kinova3
sub_obj g     PandaR85 "GRP_KP=100"                           ;  sub_piobj g     PandaR85 "GRP_KP=100"
sub_obj g_t   PandaR85 "GRP_KP=100,TWIN_RENDER=1"             ;  sub_piobj g_t   PandaR85 "GRP_KP=100,TWIN_RENDER=1"
```
场景一致性：object 的家具只有地面，物体由 init 给出；`tools/fixcheck_object.py` 比较每集首步全部非机器人关节 qpos 与 obj_src 同一集（0.0000 mm）。

## 第二批补跑（已提交，协调者按型号重提）

```bash
sub_oft sham_cg Panda;  sub_oft iiwa_cg IIWA;  sub_oft ur5e_cg UR5e
sub_pi  sham_cg Panda;  sub_pi  iiwa_cg IIWA;  sub_pi  ur5e_cg UR5e
```
在线接触记录在 10/27 约 17:45 装进评测脚本；那之前已经启动的 _cg 作业（sham_cg e0、e5，ur5e_cg 的前两片）没有 `contact_*` 字段，之后启动的有。
