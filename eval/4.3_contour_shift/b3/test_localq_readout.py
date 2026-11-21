"""localq_readout.py 的合成测试：python test_localq_readout.py（临时目录在本目录 _t_localq 下，跑完删除）。"""
import json
import os
import shutil
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import localq_readout as RD  # noqa: E402

TMP = os.path.join(HERE, "_t_localq")
VIEWS = {"h-b": ("h", -24), "h-a": ("h", -12), "h+a": ("h", 12), "h+b": ("h", 24), "v-b": ("v", -24), "v+b": ("v", 24),
         "copy-b": ("c", -24), "copy+b": ("c", 24)}


def keylist(n_ep=24, trials=5):
    return [{"key": f"B1__ep{e:02d}__t{t}", "block": "B1", "cfg_id": f"ep{e:02d}", "cluster": f"ep{e:02d}", "trial": t}
            for t in range(trials) for e in range(n_ep)]


def make(keys, resp, inview=lambda k, n, q: 1.0, n_q=8, close_q=5, check2=lambda k: True, check1=lambda k: True,
         skip=(), noise=0.0, seed=1):
    """resp(name, axis, s, key, q) -> 该画面相对零平移的响应分量 R（输入像素）。"""
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(os.path.join(TMP, "B1", "_errors"))
    rng = np.random.default_rng(seed)
    for k in keys:
        if k["key"] in skip:
            continue
        meta = []
        for q in range(n_q):
            r0 = rng.normal(0, 3, 2)
            V = {"zero": {"r_in": r0.tolist(), "frac_inview": 1.0, "gripper": [1, 1, 1, 1]}}
            for n, (ax, s) in VIEWS.items():
                Rh, Rv = resp(n, ax, s, k["key"], q)
                Rh += rng.normal(0, noise)
                Rv += rng.normal(0, noise)
                V[n] = {"r_in": [r0[0] + Rh, r0[1] + Rv], "frac_inview": inview(k["key"], n, q), "gripper": [1, 1, 1, 1]}
            meta.append({"q": q, "step": 4 * q, "pre_close": q <= close_q, "first_close_step": 4 * close_q, "G_px": 1000, "views": V})
        np.savez(os.path.join(TMP, "B1", k["key"] + ".npz"), cfg_id=k["cfg_id"], check1_available=True,
                 check1_success_steps_equal=check1(k["key"]), check2_zero_bit_equal_all=check2(k["key"]))
        json.dump(meta, open(os.path.join(TMP, "B1", k["key"] + ".json"), "w"))
    json.dump(keys, open(os.path.join(TMP, "keys.json"), "w"))


def read(**kw):
    out = os.path.join(TMP, "r.json")
    if os.path.exists(out):
        os.remove(out)
    argv = ["--root", TMP, "--block", "B1", "--keys", os.path.join(TMP, "keys.json"), "--out", out, "--B", "400"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return RD.main(argv)


def linear(kh, kv=0.0, copy=0.0, dose_extra=0.0):
    def f(n, ax, s, key, q):
        if ax == "h":
            g = kh * s * (1 + dose_extra * (abs(s) > 12))
            return g, 0.0
        if ax == "v":
            return 0.0, kv * s
        return copy * s, 0.0
    return f


class T(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_reverse_with_dose(self):
        make(keylist(), linear(-0.3, kv=-0.3, dose_extra=0.5), noise=0.3)
        r = read()
        self.assertEqual(r["outcome"], "反向且有剂量")
        self.assertAlmostEqual(r["estimates"]["D_a"]["est"], -0.3, delta=0.02)
        self.assertAlmostEqual(r["estimates"]["D_b"]["est"], -0.45, delta=0.02)
        self.assertEqual(r["direction_specificity"], "响应跟着平移的方向走")

    def test_reverse_no_dose(self):
        # 饱和：两档幅度的响应相同（R = -4 * sign(s)）-> D 为负但 |R(±b)| 不大于 |R(±a)|
        def f(n, ax, s, key, q):
            return (-4.0 * np.sign(s), 0.0) if ax == "h" else (0.0, 0.0)
        make(keylist(), f, noise=0.3)
        self.assertEqual(read()["outcome"], "反向但无剂量")

    def test_near_zero(self):
        make(keylist(), linear(0.0), noise=0.05)
        r = read()
        self.assertEqual(r["outcome"], "近零")

    def test_same_direction(self):
        make(keylist(), linear(0.3), noise=0.3)
        self.assertEqual(read()["outcome"], "同向或非单调")

    def test_inview_exclusion_level_not_judged(self):
        # h+b 在 70 个键的全部状态出视野 -> 有效键 50 < 60 -> 涉及 b 的估计量不判
        bad = {k["key"] for k in keylist()[:70]}
        make(keylist(), linear(-0.3), inview=lambda k, n, q: 0.5 if (k in bad and n == "h+b") else 1.0, noise=0.3)
        r = read()
        self.assertIsNone(r["estimates"]["D_b"])
        self.assertEqual(r["outcome"], "判不出（档不足）")
        self.assertEqual(r["level_keys"]["h+b"], 50)
        self.assertEqual(r["n_state_level_excluded_inview"]["h+b"], 70 * 6)  # 只数闭合前的 6 个查询

    def test_check_failures_excluded(self):
        make(keylist(), linear(-0.3), check2=lambda k: not k.endswith("t0"), check1=lambda k: k != "B1__ep01__t1", noise=0.3)
        r = read()
        self.assertEqual(r["excluded"]["check2"], 24)
        self.assertEqual(r["excluded"]["check1"], 1)
        self.assertEqual(r["n_keys_used"], 95)

    def test_missing_refused(self):
        make(keylist(), linear(-0.3), skip={"B1__ep03__t2"})
        with self.assertRaises(SystemExit):
            read()

    def test_error_only_excluded(self):
        make(keylist(), linear(-0.3), skip={"B1__ep03__t2"}, noise=0.3)
        open(os.path.join(TMP, "B1", "_errors", "B1__ep03__t2.txt"), "w").close()
        r = read()
        self.assertEqual(r["excluded"]["error_only"], 1)

    def test_preclose_only(self):
        # 闭合后的查询给巨大同向响应；只用闭合前 -> 仍判反向
        def f(n, ax, s, key, q):
            k = -0.3 if q <= 5 else 5.0
            return (k * s, 0.0) if ax == "h" else (0.0, 0.0)
        make(keylist(), f, noise=0.3)
        self.assertEqual(read()["signs"]["D_b"], "负")
        self.assertEqual(read(queries="all")["signs"]["D_b"], "正")

    def test_copy_readings(self):
        make(keylist(), linear(-0.3, copy=0.0), noise=0.05)
        self.assertEqual(read()["copy_reading"], "响应来自正确位置上的线被移走，而不是旁边多了一条线")
        make(keylist(), linear(-0.3, copy=-0.2), noise=0.05)
        self.assertTrue(read()["copy_reading"].startswith("旁边多一条线"))

    def test_cross_component(self):
        def f(n, ax, s, key, q):
            if ax == "h":
                return -0.3 * s, -0.3 * s
            if ax == "v":
                return -0.3 * s, 0.01 * s
            return 0.0, 0.0
        make(keylist(), f, noise=0.1)
        self.assertTrue(read()["direction_specificity"].startswith("存在交叉"))

    def test_key_weighted_cluster_mean(self):
        # 每键常数响应：键均值的平均 = 点估计；bootstrap 确定
        rng = np.random.default_rng(3)
        kd = {k["key"]: rng.normal(-0.2, 0.1) for k in keylist()}
        make(keylist(), lambda n, ax, s, key, q: ((kd[key] * s, 0.0) if ax == "h" else (0.0, 0.0)))
        r1, r2 = read(), read()
        self.assertAlmostEqual(r1["estimates"]["D_b"]["est"], float(np.mean(list(kd.values()))), places=9)
        self.assertEqual(r1["estimates"]["D_b"], r2["estimates"]["D_b"])
        self.assertEqual(r1["estimates"]["D_b"]["n_clusters"], 24)

    def test_beta_requires_all_four(self):
        make(keylist(), linear(-0.3), inview=lambda k, n, q: 0.5 if (n == "h-a" and q % 2 == 0) else 1.0, noise=0.0)
        r = read()
        self.assertAlmostEqual(r["estimates"]["beta"]["est"], -0.3, places=9)

    def test_refuse_second_readout(self):
        make(keylist(), linear(-0.3), noise=0.3)
        out = os.path.join(TMP, "r.json")
        argv = ["--root", TMP, "--block", "B1", "--keys", os.path.join(TMP, "keys.json"), "--out", out, "--B", "100"]
        RD.main(argv)
        with self.assertRaises(SystemExit):
            RD.main(argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
