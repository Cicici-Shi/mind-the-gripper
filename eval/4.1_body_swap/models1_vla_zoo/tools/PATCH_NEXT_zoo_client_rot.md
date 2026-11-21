# 补丁说明（不部署）：旋转残差进记录、进门槛输出

来源：`protocol_models1_gate_rulings_fable.md` §4 第 3 条。本文只是下一批模型（第二批 models2）用的说明和 diff 草稿，**本批不改** `common/zoo_client.py`（sha256 e0182776b93dab86c9237bfa4bb48d89a6b85399d7016051488bd493b34059d9）与 `tools/zoo_gate_m1.py`（冻结，sha256 389733733e651c4a7ab712ae4d26fb2a3be10e57148fceebafcdc6a7e2c94b75）。写于 2026-11-05。

## 问题

- `zoo_client.py` 的 `twin` 字段只存了位置残差的逐查询帧序列 `pos_err_mm_q`，旋转残差只存了整集最大值 `rot_err_deg_max` 和与位置合并的 `miss` 计数，没有逐查询帧的 `rot_err_deg_q`。
- `zoo_gate_m1.py` 的 `load_jsonl` 因此把 `rot_q` 写死为 `[]`，π0 / OVLA 的旋转残差没有进门槛，也没印出来。
- OFT 路径 `load_npz` 已经从 npz 读出 `twin_rot_err_deg[::8]` 放进 `rot_q`，但后面的 IK 段只用 `pos_q`，`rot_q` 没被用到。
- IK 段只印 `> 1 mm` 比例和 `> 2 mm` 帧数，没有最大值与分位数，门槛失败时判不了是一两集的长尾还是普遍偏移。
- 协议 §7.2 写的是「第二批口径 pos > 2 mm 或 rot > 1° 并印」，rot 这一半目前对所有模型都没有输出。

## 改法 1：`common/zoo_client.py` 写 `rot_err_deg_q`

与 `pos_err_mm_q` 同一取样（每 `replan` 步一帧，即查询帧），同样保留 4 位小数。同时补一个按查询帧算的 `miss_q`，口径与 `miss` 相同，分母与门槛一致。

```diff
--- a/common/zoo_client.py
+++ b/common/zoo_client.py
@@ -269,7 +269,11 @@
                         "cam_err_mm_q": [round(float(a), 5) for a in env.twin_stats["cam_err_mm"][::args.replan]],
                         "cam_err_deg_q": [round(float(a), 5) for a in env.twin_stats["cam_err_deg"][::args.replan]]}),
                     "pos_err_mm_q": [round(float(a), 4) for a in env.twin_stats["pos_err_mm"][::args.replan]],
+                    # models2: rotation residual per query frame (same sampling as pos_err_mm_q)
+                    "rot_err_deg_q": [round(float(a), 4) for a in env.twin_stats["rot_err_deg"][::args.replan]],
+                    "miss_q": int(sum(1 for a, b in zip(env.twin_stats["pos_err_mm"][::args.replan],
+                                                        env.twin_stats["rot_err_deg"][::args.replan]) if a > 2 or b > 1)),
                     **({} if not env.twin_stats.get("root_off_mm") else {  # NUS-PATCH 2026-10-31: G_R root offset
```

注意：
- `twin_stats["rot_err_deg"]` 由 embody_toy `body_env.py` 的各个 `_twin_paint*` 填写，长度与 `pos_err_mm` 一致（`_twin_paint_inner` 第 645 / 646 行、`_twin_paint_sunk` 第 808 / 809 行成对 append；`_twin_paint_copy` 填 0.0）。不需要改 embody_toy。
- 每集 jsonl 记录会多约 `步数 / replan` 个浮点。OVLA replan=1、约 220 步，每条多约 2 KB，可接受。
- 不新增任何结局相关字段。

## 改法 2：`zoo_gate_m1` 的后继版本（另起文件名，例如 `tools/zoo_gate_m2.py`，不覆盖冻结的 m1 版）

1. `load_jsonl` 读新字段，旧记录没有该字段时为 `None`（区分「无字段」与「全 0」）：

```diff
-                tw = {"pos_q": tw.get("pos_err_mm_q", []), "rot_q": [], "stale": tw.get("stale", 0),
+                tw = {"pos_q": tw.get("pos_err_mm_q", []), "rot_q": tw.get("rot_err_deg_q"), "stale": tw.get("stale", 0),
```

2. `load_npz` 已有 `rot_q = twin_rot_err_deg[::8]`，不用改；另加全部帧序列以便量取样偏差（§4 第 4 条）：

```diff
             tw = {"pos_q": pe[::8].tolist(), "rot_q": re_[::8].tolist(), "stale": int(z["twin_stale"]),
+                  "pos_all": pe.tolist(), "rot_all": re_.tolist(),
```

3. IK 段印最大值、分位数和旋转一半，门槛判定按 models2 协议事先写定（下面的数字只是草稿，须在 models2 协议冻结前定）：

```diff
         pq = [x for t in tws for x in t["pos_q"]]
         R["ik_query_frames"] = len(pq)
         R["ik_over_1mm_frac"] = round(sum(1 for x in pq if x > 1) / max(len(pq), 1), 5)
         R["ik_over_2mm"] = sum(1 for x in pq if x > 2)
+        if pq:
+            R["ik_pos_mm_q_max"] = round(max(pq), 4)
+            R["ik_pos_mm_q_pct"] = {p: round(float(np.percentile(pq, p)), 4) for p in (50, 90, 99, 99.9)}
+        rq_have = [t["rot_q"] for t in tws if t["rot_q"] is not None]
+        if rq_have:
+            rq = [x for v in rq_have for x in v]
+            R["ik_rot_records"] = len(rq_have)              # records that carry the rotation field
+            R["ik_rot_deg_q_max"] = round(max(rq), 4) if rq else None
+            R["ik_rot_deg_q_pct"] = {p: round(float(np.percentile(rq, p)), 4) for p in (50, 90, 99, 99.9)} if rq else None
+            R["ik_over_1deg"] = sum(1 for x in rq if x > 1)
+            pr = [(a, b) for t in tws if t["rot_q"] is not None for a, b in zip(t["pos_q"], t["rot_q"])]
+            R["ik_miss_2mm_or_1deg_frac"] = round(sum(1 for a, b in pr if a > 2 or b > 1) / max(len(pr), 1), 5)
+        else:
+            R["ik_rot"] = "no field (records predate rot_err_deg_q)"
+        pa = [x for t in tws for x in (t.get("pos_all") or [])]
+        if pa:                                               # OFT only: all frames vs query frames (sampling bias, 4.4)
+            R["ik_over_1mm_frac_allframes"] = round(sum(1 for x in pa if x > 1) / len(pa), 5)
         R["ik_gate"] = R["ik_over_1mm_frac"] < 0.01
```

4. 门槛本身：models2 协议里要逐条写明 (a) 位置门槛用查询帧还是全部帧（看 OFT 上 `ik_over_1mm_frac_allframes` 与查询帧比例是否有差）；(b) 是否加旋转门槛（例如 `ik_miss_2mm_or_1deg_frac < 1%`），以及残差门槛是否按模型消费的相机分别定（裁定 §4 第 6、7 条）；(c) 每条门槛不过的后果。草稿里 `ik_gate` 保持 m1 口径不动，新字段只印。

## 验证（部署时做）

- 用一个 replan=5 的 π0 冒烟作业跑 2 集，检查 jsonl 的 `rot_err_deg_q` 长度等于 `pos_err_mm_q`，且与 `rot_err_deg_max` 相容（`max(rot_err_deg_q) <= rot_err_deg_max`）。
- 后继门槛脚本在 m1 的旧 jsonl 上运行，应印「no field」，`ik_gate` 与冻结 m1 版逐位相同；在 OFT npz 上印出的 `ik_rot_deg_q_max` 等于 `max(twin_rot_err_deg[::8])`。
- 整个过程不读、不印任何 `success` 字段。
