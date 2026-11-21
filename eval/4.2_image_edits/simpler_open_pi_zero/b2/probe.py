"""冒烟前的环境核查（不跑模型）：每个块的环境
  1. reset 会不会重建场景：reconfigure=False / True 两种 reset 前后，连杆 id、可视体对象是否换了，隐藏状态是否保留；
  2. 同一键 reset 两次（中间夹另一个键）初始状态是否逐元素相同；
  3. 同一状态渲染两次 RGB 是否逐像素相同（关抽屉用光线追踪着色器，重点看它）；
  4. 打印控制频率、步数上限、图尺寸、线宽、G 连杆、受保护集合 P、渲染器。
"""
import json
import os
import sys

B2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2)
import numpy as np  # noqa: E402

import eval_views as E  # noqa: E402
import keys as K  # noqa: E402
import views as V  # noqa: E402


def state_vec(env):
    s = E.scene_state(env)
    return np.concatenate([s["init_robot_qpos"], s["init_robot_pose"], s["init_actor_poses"].ravel()]), s


def main():
    import simpler_env
    E.check_env_src()
    todo = [(blk, t) for blk in (sys.argv[1:] or list(K.BLOCKS)) for t in K.BLOCKS[blk]["tasks"]]
    for blk, task in todo:
        b = K.BLOCKS[blk]
        print(f"===== {blk} {task}", flush=True)
        env = simpler_env.make(task)
        u = env.unwrapped
        ks = [k for k in K.make_keys(blk, "calib") if k["task"] == task]
        ka, kb = ks[0], ks[len(ks) // 2]
        obs, info = E.reset_key(env, ka)
        cam = V.cam_name(u.robot_uid)
        W = obs["image"][cam]["rgb"].shape[1]
        vm = V.ViewMaker(env, "C3", b["body"], 8, V.line_width_for(W), 2, K.PROTECT[blk], K.PROTECT_EXPECT_N[blk])
        vm.on_reset()
        print(blk, "robot_uid", u.robot_uid, "cam", cam, "img", obs["image"][cam]["rgb"].shape,
              "ctrl_hz", u._control_freq, "sim_hz", u._sim_freq, "max_steps", env.spec.max_episode_steps,
              "line_width", V.line_width_for(W), flush=True)
        print(blk, "G links", [(l.name, l.id) for l in vm.g_links], "n_visual_bodies", vm.n_vb, flush=True)
        print(blk, "P", vm.p_names, "| all actors", [x.name for x in u.get_actors()], flush=True)
        if not K.is_bridge(blk):
            us = []
            for sd in range(8):
                _, inf = env.reset(seed=sd, options={"reconfigure": True})
                us.append((sd, u.urdf_version, inf.get("episode_id"), [round(float(x), 3) for x in
                           (inf.get("obj_init_pose_wrt_robot_base") or inf.get("episode_source_obj_init_pose_wrt_robot_base")).p]))
            print(blk, "seed -> (urdf, episode_id, src obj xyz):", us, flush=True)
            obs, info = E.reset_key(env, ka)
            vm.on_reset()
        print(blk, "renderer", repr(getattr(u, "_renderer", None)), "shader", getattr(u, "shader_dir", None),
              flush=True)
        try:
            r = u._renderer
            print(blk, "renderer attrs", [x for x in dir(r) if not x.startswith("__")][:40], flush=True)
        except Exception as e:
            print(blk, "renderer attr err", e)
        print(blk, "reset info A", json.dumps(E.jsonable(info))[:600], flush=True)
        va, sa = state_vec(env)
        seg = obs["image"][cam]["Segmentation"][..., 1]
        print(blk, "G px", int(np.isin(seg, vm.gids).sum()), "robot px", int(np.isin(seg, vm.robot_ids).sum()),
              "P px", int(np.isin(seg, vm.pids).sum()), flush=True)
        # 3. 同一状态渲染两次
        r1 = V.rgb_of(u.get_obs()["image"][cam])
        r2 = V.rgb_of(u.get_obs()["image"][cam])
        d = np.abs(r1.astype(int) - r2.astype(int))
        print(blk, "render twice: diff px", int((d.max(2) > 0).sum()), "max", int(d.max()),
              "vs reset obs max", int(np.abs(r1.astype(int) - obs["image"][cam]["rgb"].astype(int)).max()), flush=True)
        h1, _ = vm.render_hidden()
        h2, _ = vm.render_hidden()
        dh = np.abs(h1.astype(int) - h2.astype(int))
        print(blk, "hidden render twice: diff px", int((dh.max(2) > 0).sum()), "max", int(dh.max()), flush=True)
        # 1. reset 是否重建场景
        ids_before = sorted(l.id for l in vm.g_links)
        vb_before = [id(vb) for l in vm.g_links for vb in l.get_visual_bodies()]
        vm._set_vis(0.0)
        for recon in (False, True):
            opts = {"obj_init_options": dict(ka["reset"]), "reconfigure": recon}
            o2, _ = env.reset(seed=ka["env_seed"], options=opts)
            links = [l for l in u.agent.robot.get_links() if l.name in V.G_LINKS[b["body"]]]
            ids_after = sorted(l.id for l in links)
            vb_after = [id(vb) for l in links for vb in l.get_visual_bodies()]
            seg2 = o2["image"][cam]["Segmentation"][..., 1]
            print(blk, f"reset(reconfigure={recon}): link ids same {ids_before == ids_after} "
                       f"same robot object {links[0] is vm.g_links[0]} visual-body objects same {vb_before == vb_after} "
                       f"G px visible after reset {int(np.isin(seg2, np.array(ids_after)).sum())}", flush=True)
            vm.on_reset()
            vm._set_vis(0.0)
            ids_before = ids_after
            vb_before = vb_after
        vm._set_vis(1.0)
        # 2. 同一键 reset 两次（中间夹 kb）
        E.reset_key(env, kb)
        vb_, _ = state_vec(env)
        E.reset_key(env, ka)
        va2, sa2 = state_vec(env)
        same = va.shape == va2.shape and bool(np.array_equal(va, va2))
        print(blk, "same key twice -> state bit-equal", same,
              "max abs diff", None if va.shape != va2.shape else float(np.abs(va - va2).max()),
              "| other key differs", not (va.shape == vb_.shape and np.array_equal(va, vb_)), flush=True)
        env.close()


if __name__ == "__main__":
    main()
