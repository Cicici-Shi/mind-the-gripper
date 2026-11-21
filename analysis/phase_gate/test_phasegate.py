"""Unit tests for embody/phasegate.py (pure logic, no simulator). Run: python analysis/test_phasegate.py"""
import importlib.util
import os
import sys

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embody", "phasegate.py")
_s = importlib.util.spec_from_file_location("phasegate", _p)
pg = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pg)

OPEN, CLOSE = [0, 0, 0, 0, 0, 0, -1.0], [0, 0, 0, 0, 0, 0, 1.0]


def run(window, script, lift=0.03):
    """script: list of (action, z after the action). Returns the gate object after the episode."""
    g = pg.PhaseGate(window, lift)
    g.reset(1.0)
    for a, z in script:
        g.update(a, z)
    return g


# descend open for 3 steps, close at action 3, dip 1 cm, then rise 1 cm per step
SCRIPT = [(OPEN, 0.95), (OPEN, 0.90), (OPEN, 0.85), (CLOSE, 0.85), (CLOSE, 0.84), (CLOSE, 0.85), (CLOSE, 0.86),
          (CLOSE, 0.87), (CLOSE, 0.90), (OPEN, 0.90)]


def test_phases():
    g = run("none", SCRIPT)
    # observation index: 0 initial, k after action k-1
    assert g.t_close == 3 and g.t_lift == 7, (g.t_close, g.t_lift)
    assert g.phases == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], g.phases
    assert len(g.phases) == len(SCRIPT) + 1 == len(g.gates)


def test_windows():
    ph = run("none", SCRIPT).phases
    assert run("none", SCRIPT).gates == [False] * len(ph)
    assert run("all", SCRIPT).gates == [True] * len(ph)
    for i, w in enumerate(pg.PHASES):
        assert run(w, SCRIPT).gates == [p == i for p in ph], w
    # the three windows partition the episode
    tot = [sum(x) for x in zip(*(run(w, SCRIPT).gates for w in pg.PHASES))]
    assert tot == [1] * len(ph)


def test_lift_uses_lowest_point_since_close():
    # close high above the object, keep descending 10 cm, then rise 2 cm: not a lift yet; 3 cm above the dip: lift
    s = [(CLOSE, 1.00), (CLOSE, 0.95), (CLOSE, 0.90), (CLOSE, 0.92), (CLOSE, 0.93)]
    g = run("none", s)
    assert g.t_close == 0 and g.t_lift == 4 and g.phases == [0, 1, 1, 1, 1, 2], (g.t_lift, g.phases)


def test_latched_and_never_closing():
    # reopening after the lift does not return to approach
    g = run("none", SCRIPT + [(OPEN, 0.5), (CLOSE, 0.5)])
    assert g.phases[-1] == 2
    # a policy that never closes stays in approach; one that closes and never lifts stays in grasp
    assert set(run("none", [(OPEN, 0.9)] * 20).phases) == {0}
    assert run("none", [(CLOSE, 0.9)] * 20).phases == [0] + [1] * 20
    g = run("none", [(OPEN, 0.9)] * 5)
    f = g.fields("erase")
    assert f["pg_t_close"] == -1 and f["pg_t_lift"] == -1 and f["pg_phase"].shape == (6,) and f["pg_gate"].dtype == bool and "pg_changed_px" not in f


def test_reset_reuses_object():
    g = pg.PhaseGate("grasp")
    for _ in range(2):
        g.reset(1.0)
        for a, z in SCRIPT:
            g.update(a, z)
        assert g.t_close == 3 and g.t_lift == 7 and len(g.gates) == len(SCRIPT) + 1


def test_post_window():
    # phasegate-confirm: post = grasp + transport = complement of approach
    ap, po = run("approach", SCRIPT), run("post", SCRIPT)
    assert po.gates == [not g for g in ap.gates], po.gates
    assert [a + b for a, b in zip(ap.gates, po.gates)] == [1] * (len(SCRIPT) + 1)
    assert po.gates == [False] * 4 + [True] * 7 and po.phases == ap.phases
    ph = run("none", SCRIPT).phases
    assert po.gates == [p >= 1 for p in ph]
    # same as grasp OR transport, step by step
    assert po.gates == [a or b for a, b in zip(run("grasp", SCRIPT).gates, run("transport", SCRIPT).gates)]
    # an episode that never closes: post never opens; one that closes and never lifts: open from the close on
    assert run("post", [(OPEN, 0.9)] * 20).gates == [False] * 21
    assert run("post", [(CLOSE, 0.9)] * 20).gates == [False] + [True] * 20
    f = po.fields("erase")
    assert f["pg_window"] == "post" and f["pg_gate"].tolist() == po.gates and f["pg_t_close"] == 3
    assert set(f) == {"pg_mode", "pg_window", "pg_lift_m", "pg_phase", "pg_gate", "pg_t_close", "pg_t_lift"}
    # reopening after the lift (latched transport) keeps post open
    assert run("post", SCRIPT + [(OPEN, 0.5), (CLOSE, 0.5)]).gates[-3:] == [True] * 3
    assert pg.WINDOWS[:5] == ("none", "approach", "grasp", "transport", "all") and pg.WINDOWS[5] == "post"


def test_bad_args():
    for bad in ("", "Grasp", "pregrasp", "Post", "postgrasp"):
        try:
            pg.PhaseGate(bad)
        except AssertionError:
            continue
        raise SystemExit(f"window {bad!r} accepted")


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
    print(f"OK {n} tests")
    sys.exit(0)
