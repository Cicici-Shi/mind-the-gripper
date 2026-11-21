# phasegate（探索档，2026-11-10）

问题：冻结 VLA 在任务的哪个阶段需要看见手（erase），哪个阶段需要是自己的手（look）。
物理全程是原装 Panda，只在接近、抓取、搬运三个窗口之一改画面。窗口由回合自己的事件在线判定，定义见 `embody/phasegate.py`。

来源：整包复制 `$PROJ_ROOT/maskinfo_confirm`（embody、oft、tools、lists、analysis）加 `$PROJ_ROOT/maskinfo_pi/pi`，旧目录没动。
改动只有三处，开关不设时走原路径：
- `embody/phasegate.py`（新）：窗口状态机。
- `embody/body_env.py`：加 `_pg_init / _pg_apply / _pg_erase / pg_fields`，`set_init_state` 与 `step` 在 `PG_MODE` 非空时走新分支（相对源文件只增 72 行）。
- `oft/run_libero_eval_pg.py`：dump 多写 `pg_*` 字段；进程内渲染器断言的允许名单可配（`PG_GL_ALLOW`）。

格子：`PG_MODE` ∈ {look, erase} × `PG_WINDOW` ∈ {none, approach, grasp, transport, all}。
- look = 已发表 G_R 的沉装替身（`TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk`），替身每步都画，只在开门时用它的图。
- erase = 两路相机里夹爪全部 geom 置 alpha 0，与 `HIDE_GRIPPER=all` 同一开关，但逐步重渲染后恢复。

用法：
- 测试：`python analysis/test_phasegate.py`，`python analysis/test_pg_readout.py`
- 冒烟（无策略、无结局）：`sbatch tools/pg_smoke.sbatch`
- 探索格（每格 50 集 = 10 任务 × 5 集，共 10 格）：
  `sbatch --export=ALL,MODEL=oft,SUITE=spatial,JOBS="erase:none erase:approach erase:grasp erase:transport erase:all look:none look:approach look:grasp" tools/pg_run.sbatch`
  `sbatch --export=ALL,MODEL=oft,SUITE=spatial,JOBS="look:transport look:all" tools/pg_run.sbatch`
- 读数：`python analysis/pg_readout.py out/runs/oft_spatial --out out/results_pg_oft_spatial.txt`
- 卡型（2026-11-10 补）：`CARD` ∈ h200（最多 8 个 OFT 进程）| a100-80（最多 4）| a100-40（最多 2），A100 时命令行覆盖
  `--gres=gpu:<卡型>:1 --exclude=node39,node38`（node39 是 ARM64，node29 是 H100，都不能用）。
- π 家族：`tools/pg_pi_run.sbatch`（`MODEL=pi05|pi0`，参数同 `pg_run.sbatch`），客户端 `pi/zoo_client_pg.py`，服务端
  `pi/serve_policy_mi.py`（固定噪声，噪声键同 maskinfo-pi，来自 `tools/mi_pi_params.sh`）；落盘格式与 OFT 相同，读数脚本通用。

探索档规矩：不预注册，不进报告，只用来挑；挑中的格换没用过的集号按正式流程确认。
已知限制：窗口是内生的（干预会改变合拢和抬起的时刻，从而改变自己的窗口长度），读数里逐格印出窗口长度。

## phasegate_confirm（2026-11-10 晚，本目录）
正式确认的代码，协议见 `protocol_phasegate_confirm.md`（v1）。从 `$PROJ_ROOT/phasegate` 复制（embody、oft、pi、tools、analysis、README；不含 out、logs）；那边和 `$PROJ_ROOT/phasegate_yoke` 都只读。
- 唯一的行为改动：`embody/phasegate.py` 增加窗口 `post`（阶段为 grasp 或 transport 时开门，即 approach 的补集）。原有五个窗口不变。
- `tools/pg_run.sbatch`：目录改成本目录，`OUTROOT` 默认 `$T/out/confirm`，卡型核对改成「同系列且显存不小于申请值」。
- `tools/pg_smoke.py`：post 用例；`analysis/pg_readout.py`：认识 post 格。
- 读数：`python analysis/pg_confirm.py out/confirm/oft_spatial out/confirm/ofti_spatial --eval-logs logs/eval --out out/results_phasegate_confirm.txt`；
  只核字段（不读 success）：加 `--check-fields-only`。出错回合从 `logs/eval` 的评测日志里读（`Episode error` / `Dump error`）。
- 测试：`python analysis/test_phasegate.py`，`python analysis/test_pg_readout.py`，`python analysis/test_pg_confirm.py`（登录节点上约 2 分钟）。
