"""相机参数与末端参考点投影的核查（回放参考记录，不跑策略）。
每块 1 个校准键，在若干查询帧：取 obs["camera_param"] 的内外参，把 TCP（WidowX ee_gripper_link，Google link_gripper_tcp）
与保护物体中心投影到 640 宽原图与 224 策略输入（横 224/W、竖 224/H），画点存小图，并报投影点相对 G 掩膜包围盒的位置。
  python proj_check.py --block B2 --keys keysets/dev_B2.json --ref ../b2/out_calib --out out_proj"""
import argparse
import json
import os
import sys

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import keys as K  # noqa: E402
import views as V  # noqa: E402


def project(p_world, intr, extr):
    pc = extr[:3, :3] @ np.asarray(p_world, float) + extr[:3, 3]
    uvw = intr @ pc
    return uvw[:2] / uvw[2], float(pc[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries", default="0,3,6,9,12")
    a = ap.parse_args()
    import simpler_env
    from eval_views import reset_key
    blk = K.BLOCKS[a.block]
    k = json.load(open(a.keys))[0]
    z = np.load(os.path.join(a.ref, a.block, "C0", k["key"] + ".npz"))
    acts, qe = z["env_actions"], int(z["query_every"])
    env = simpler_env.make(k["task"])
    u = env.unwrapped
    obs, _ = reset_key(env, k)
    vm = V.ViewMaker3(env, "C0", blk["body"], int(z["lag_steps"]), int(z["line_width"]), qe, K.PROTECT[a.block],
                      K.PROTECT_EXPECT_N[a.block])
    vm.setup3(a.block, k["cfg_id"], k["trial"], {}, (), None, key=k["key"])
    vm.on_reset3()
    prot = [x for x in u.get_actors() if x.name in vm.p_names]
    qs = {int(x) for x in a.queries.split(",")}
    os.makedirs(os.path.join(a.out, a.block), exist_ok=True)
    rows, tiles = [], []
    for i in range(len(acts)):
        if i % qe == 0 and i // qe in qs:
            cp = obs["camera_param"][vm.cam]
            intr, extr = np.asarray(cp["intrinsic_cv"], float), np.asarray(cp["extrinsic_cv"], float)
            rgb = obs["image"][vm.cam]["rgb"].copy()
            H, W = rgb.shape[:2]
            G = np.isin(obs["image"][vm.cam]["Segmentation"][..., 1], vm.gids)
            g = V.mask_geom(G)
            uv, depth = project(vm.tcp.pose.p, intr, extr)
            r = {"q": i // qe, "tcp_uv": uv.tolist(), "tcp_depth_m": depth,
                 "tcp_uv_224": [uv[0] * 224 / W, uv[1] * 224 / H], "G_bbox": None if g is None else [g["x0"], g["y0"], g["x1"], g["y1"]],
                 "tcp_in_G_bbox": None if g is None else bool(g["x0"] <= uv[0] < g["x1"] and g["y0"] <= uv[1] < g["y1"]),
                 "tcp_on_G_pixel": bool(0 <= int(uv[1]) < H and 0 <= int(uv[0]) < W and G[int(uv[1]), int(uv[0])]),
                 "objects": {}}
            for o in prot:
                ouv, od = project(o.pose.p, intr, extr)
                r["objects"][o.name] = {"uv": ouv.tolist(), "uv_224": [ouv[0] * 224 / W, ouv[1] * 224 / H], "depth_m": od}
                cv2.drawMarker(rgb, (int(round(ouv[0])), int(round(ouv[1]))), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 3)
            cv2.drawMarker(rgb, (int(round(uv[0])), int(round(uv[1]))), (255, 0, 255), cv2.MARKER_CROSS, 28, 3)
            if i == 0:
                r["intrinsic_cv"], r["extrinsic_cv"] = intr.tolist(), extr.tolist()
            rows.append(r)
            tiles.append(V.to_model(rgb))
        obs, *_ = env.step(acts[i])
    sheet = np.concatenate(tiles, axis=1)
    cv2.imwrite(os.path.join(a.out, a.block, f"proj_{k['key']}_224.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    json.dump({"key": k["key"], "cam": vm.cam, "camera_param_keys": sorted(obs["camera_param"][vm.cam].keys()), "rows": rows},
              open(os.path.join(a.out, a.block, "proj.json"), "w"), indent=1)
    print("[proj]", a.block, k["key"], [(r["q"], [round(x) for x in r["tcp_uv"]], r["tcp_in_G_bbox"], r["tcp_on_G_pixel"]) for r in rows], flush=True)


if __name__ == "__main__":
    main()
