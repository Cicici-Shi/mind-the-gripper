"""Synthetic test of analysis/pg_readout.py. Run: python analysis/test_pg_readout.py"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_readout as R


def write(d, mode, window, task, init, y, gate):
    os.makedirs(d, exist_ok=True)
    g = np.asarray(gate, bool)
    np.savez_compressed(os.path.join(d, f"t{task}_e{init}_i{init}.npz"), success=y, task=task, ep=init, init=init,
                        pg_mode=mode, pg_window=window, pg_gate=g, pg_phase=np.zeros(len(g), np.int8),
                        pg_t_close=3, pg_t_lift=-1 if not y else 7)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        blk = os.path.join(tmp, "oft_spatial")
        # none: all succeed; grasp window: tasks 0-4 fail; all: every key fails; approach / transport: no effect
        for task in range(10):
            for init in range(5):
                for w in R.WINDOWS[:5]:  # the five exploration windows (post has its own test)
                    y = {"none": True, "approach": True, "transport": True, "grasp": task >= 5, "all": False}[w]
                    gate = [w == "all" or (w == "grasp" and 3 < i <= 7) or (w == "approach" and i <= 3)
                            or (w == "transport" and i > 7) for i in range(11)]
                    write(os.path.join(blk, f"erase_{w}"), "erase", w, task, init, y, gate)
        txt = R.report(blk)
        assert "\npost " not in txt  # no post cell, no post row (old blocks read exactly as before)
        rows = {ln.split()[0]: ln for ln in txt.splitlines() if ln.split() and ln.split()[0] in R.WINDOWS}
        assert rows["none"].split()[1:3] == ["50", "100.0"], rows["none"]
        assert "-50.0" in rows["grasp"] and "0.50" in rows["grasp"], rows["grasp"]
        assert "-100.0" in rows["all"], rows["all"]
        assert "+0.0" in rows["approach"] and "+0.0" in rows["transport"]
        assert "== mode look ==" not in txt
        # the grasp interval is task-clustered: it must be wide (tasks are all-or-nothing), not the key-level +-14
        lo, hi = [float(x) for x in rows["grasp"].split("[")[1].split("]")[0].split(",")]
        assert lo <= -70 and hi >= -30, (lo, hi)
        # label mismatch between directory and dump is refused
        write(os.path.join(blk, "erase_none"), "erase", "grasp", 0, 99, True, [False] * 5)
        try:
            R.report(blk)
        except SystemExit:
            print("OK readout tests")
            return
        raise SystemExit("mismatched cell label accepted")


def test_post_row():
    with tempfile.TemporaryDirectory() as tmp:
        blk = os.path.join(tmp, "oft_spatial")
        for task in range(4):
            for init in range(3):
                for w in ("none", "post"):
                    write(os.path.join(blk, f"look_{w}"), "look", w, task, init, w == "none" or task >= 2,
                          [w == "post" and i > 3 for i in range(11)])
        txt = R.report(blk)
        row = [ln for ln in txt.splitlines() if ln.startswith("post ")]
        assert len(row) == 1 and "-50.0" in row[0], txt
    print("OK readout post tests")


if __name__ == "__main__":
    main()
    test_post_row()
