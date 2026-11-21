"""Minimal tests for tools/m1_driver_v2.py on synthetic gate results + state (no cluster calls, no success rates).
Run: cd $PROJ_ROOT/vla_zoo && python3 tools/test_m1_driver_v2.py"""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_driver_v2 as d

GOOD = {"records": 50, "scene_gate": True, "copy_gate": True, "contacts_gate": True, "error_rate_gate": True,
        "align_gate": True, "stale_total": 0, "gpu_models": ["NVIDIA A100-PCIE-40GB"]}
TWIN = ("sham_twin", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t")


def gate(model, cells, **over):
    g = {"cells": {}, "wall_ratio": {"sham_twin/sham_cg": 1.1, "g_tfc/grpkp100": 1.2}, "agree": {"T0_vs_S": {"outcome_agree": 0.95}}}
    for c in cells:
        R = dict(GOOD)
        if model == "openvla":
            R.update(gpu_models=["Tesla T4"], replan=[1])
        if c in TWIN:
            R.update(ik_gate=True, cam_gate=True)
        g["cells"][c] = R
    for path, v in over.items():          # "cell__field" or "T0"
        if path == "T0":
            g["agree"]["T0_vs_S"]["outcome_agree"] = v
        else:
            c, f = path.split("__")
            g["cells"][c][f] = v
    return g


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir=os.path.expanduser("~/tmp"))
        os.makedirs(f"{self.tmp}/lists")
        d.Z, d.STATE, d.LOG = self.tmp, f"{self.tmp}/state.json", f"{self.tmp}/log.txt"
        self.cmds, self.gate_calls, self.src_calls, self.load_calls = [], [], [], []
        self.gates, self.load_ok, self.src_ok = {}, True, True
        self.done = {}            # (model, cell, off) -> n ; default: shard 0 complete everywhere, others 0; pi0 / oft_sp complete

        def n_done(m, c, o):
            if (m, c, o) in self.done:
                return self.done[(m, c, o)]
            if m in ("pi0", "oft_sp"):
                return 50
            if m == "openvla" and c == "src":
                return 50
            return 50 if o == 0 else 0

        def sh(cmd):
            self.cmds.append(cmd)
            if "zoo_cells.sh" in cmd:
                return "X 1 2"
            if "sbatch" in cmd or "scancel" in cmd:
                return str(9000 + len(self.cmds))
            raise AssertionError(f"unexpected sh: {cmd}")

        def run_gate(m, su, cells):
            self.gate_calls.append(m)
            return self.gates[m]

        def load(m):
            self.load_calls.append(m)
            return self.load_ok, f"7 logs, missing-weight warnings {0 if self.load_ok else 3}"

        def src():
            self.src_calls.append(1)
            return (True, "OVLA SRC baseline SRC success 169/200") if self.src_ok else (False, "OVLA SRC baseline check not passed: SRC success 140/200")

        d.n_done, d.sh, d.run_gate, d.oft_load_check, d.ovla_src_check = n_done, sh, run_gate, load, src
        d.squeue = lambda: {}

    def state(self, released):
        json.dump({"released": released, "claims": {}, "gate_msgs": {}}, open(d.STATE, "w"))

    def logs(self):
        return open(d.LOG).read() if os.path.exists(d.LOG) else ""

    def submitted(self):
        st = json.load(open(d.STATE))
        return {x for v in st["claims"].values() for x in v}

    BASE = {"pi0|libero_spatial": "h", "pi0|libero_object": "h", "oft_sp|libero_spatial": "auto"}

    def test_one_cell_fails_others_released(self):
        self.state({**self.BASE, "openvla|libero_spatial": "h"})
        self.gates["oft_obj"] = gate("oft_obj", [c for c in d.CELLS], g_ta_fc__ik_gate=False)
        for c in d.CELLS:                       # offsets 10/15 already on disk -> one OFT job fits all that is free
            for o in (10, 15):
                self.done[("oft_obj", "obj_" + c, o)] = 50
        d.main(dry=False)
        st = json.load(open(d.STATE))
        rel = {k for k in st["released"] if k.startswith("oft_obj|")}
        self.assertEqual(rel, {f"oft_obj|libero_object|obj_{c}" for c in d.CELLS if c != "g_ta_fc"})
        self.assertIn("PENDING-HUMAN oft_obj|libero_object: cells obj_g_ta_fc", self.logs())
        sub = self.submitted()
        self.assertIn("oft_obj|obj_g_tfc|5", sub)
        self.assertIn("oft_obj|obj_src|10", sub)          # src follows the released cells
        self.assertIn("oft_obj|obj_src|15", sub)
        self.assertFalse([x for x in sub if "obj_g_ta_fc" in x])
        # next tick: FAIL message unchanged -> not re-logged, failed cell still held
        n = self.logs().count("PENDING-HUMAN")
        d.main(dry=False)
        self.assertEqual(self.logs().count("PENDING-HUMAN"), n)
        # human releases the failed cell in the new cell format (bare name accepted)
        st = json.load(open(d.STATE)); st["released"]["oft_obj|libero_object|g_ta_fc"] = "hand"; st["claims"] = {}
        json.dump(st, open(d.STATE, "w"))
        d.main(dry=False)
        self.assertIn("oft_obj|obj_g_ta_fc|5", self.submitted())

    def test_t0_below_090_is_warn_only(self):
        self.state({**self.BASE, "oft_obj|libero_object": "h"})
        self.gates["openvla"] = gate("openvla", [c for c in d.CELLS if c != "g_t"], T0=0.82)
        d.main(dry=False)
        st = json.load(open(d.STATE))
        self.assertEqual(len([k for k in st["released"] if k.startswith("openvla|")]), 6)
        self.assertIn("WARN openvla|libero_spatial: T0_vs_S=0.82", self.logs())
        self.assertIn("openvla|sham_twin|5", self.submitted())

    def test_hand_release_load_check_fails_no_submit(self):
        self.state({**self.BASE, "oft_obj|libero_object": "hand", "openvla|libero_spatial": "hand"})
        self.load_ok = False
        d.main(dry=False)
        sub = self.submitted()
        self.assertFalse([x for x in sub if x.startswith("oft_obj|")])
        self.assertIn("SIDE oft_obj|libero_object: FAIL", self.logs())
        self.assertTrue([x for x in sub if x.startswith("openvla|")])   # OVLA side check passes -> submitted
        self.assertEqual(self.gate_calls, [])

    def test_hand_release_src_check_fails_no_submit(self):
        self.state({**self.BASE, "oft_obj|libero_object": "hand", "openvla|libero_spatial": "hand"})
        self.src_ok = False
        d.main(dry=False)
        sub = self.submitted()
        self.assertFalse([x for x in sub if x.startswith("openvla|")])
        self.assertFalse([c for c in self.cmds if "m1_srvd_openvla" in c or "m1_d_openvla" in c])
        self.assertIn("SIDE openvla|libero_spatial: FAIL", self.logs())
        self.assertTrue([x for x in sub if x.startswith("oft_obj|")])
        # SRC passes later: read once, then cached in state
        self.src_ok = True; n = len(self.src_calls)
        st = json.load(open(d.STATE)); st["claims"] = {}; json.dump(st, open(d.STATE, "w"))
        d.main(dry=False); d.main(dry=False)
        self.assertEqual(len(self.src_calls), n + 1)
        self.assertIn("openvla|libero_spatial", json.load(open(d.STATE))["side_ok"])

    def test_old_format_released_and_dry_run(self):
        self.state({**self.BASE, "oft_obj|libero_object": "hand", "openvla|libero_spatial": "hand"})
        before = open(d.STATE).read()
        plan = d.main(dry=True)
        self.assertEqual(self.gate_calls, [])                      # whole group released, no gate run
        self.assertEqual(open(d.STATE).read(), before)             # dry-run writes no state
        self.assertFalse(os.path.exists(d.LOG))                    # ... no log
        self.assertEqual(os.listdir(f"{self.tmp}/lists"), [])     # ... no list files
        self.assertFalse([c for c in self.cmds if "sbatch" in c or "scancel" in c])   # ... nothing submitted
        kinds = [p for p in plan if p[0] == "sbatch"]
        self.assertTrue(any("m1_oft_obj" in p[1] for p in kinds))
        self.assertTrue(any("m1_d_openvla" in p[1] for p in kinds))
        self.assertTrue(all(d.is_released(json.loads(before), "oft_obj", "libero_object", "obj_" + c) for c in d.CELLS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
