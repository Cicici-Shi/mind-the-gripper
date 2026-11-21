"""phasegate (exploration tier, 2026-11-10): show an image intervention only inside one task window.

The physical body is the stock Panda throughout. Per step the policy either sees the plain physical image or the
intervened one, decided by the window the running episode is in:

  approach   from the first policy step until the policy first commands "close" (env gripper action > 0)
  grasp      from that step until the hand has risen PG_LIFT_M above its lowest point since the close command
  transport  after that (latched: the episode never goes back)

Windows are defined by the episode's own events, with no object ground truth. They are therefore endogenous: an
intervention that delays the close command also lengthens its own approach window. Per-step phase and gate state are
dumped so the readout can report window lengths per cell.

Interventions (PG_MODE):
  look   the gripper pixels are replaced by a Robotiq 2F-85 drawn at the Panda pad position (the G_R sunk twin);
         needs BODY=Panda TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk
  erase  the gripper is removed from both views (alpha 0 on every gripper geom, same switch as HIDE_GRIPPER=all)

PG_WINDOW: none | approach | grasp | transport | all. "none" and "all" are the two anchors (never / always).

Yoke mode (phasegate_yoke, 2026-11-10): the window is not decided by the running episode's events but by fixed step
counts taken from the same key (task, ep, init) in an unintervened source cell (PG_YOKE_DIR, window "none"): with the
source's t_close / t_lift, observation k is approach if k <= t_close, grasp if t_close < k <= t_lift, transport if
k > t_lift (t_close = -1: approach throughout; t_lift = -1: grasp after the close; an episode longer than the source
stays in the last phase). The episode's own events are still tracked (pg_t_close, pg_t_lift, pg_phase_own); pg_phase
and pg_gate are the fixed-window ones. Reason: under erase, the policy may close at once, so the event window closes
itself (erase:approach was open a few steps only).

These policies read a single frame per query, so switching the image between steps adds no temporal inconsistency
of its own.
"""
import numpy as np

WINDOWS = ("none", "approach", "grasp", "transport", "all")
MODES = ("look", "erase")
PHASES = ("approach", "grasp", "transport")


def yoke_phase(k, t_close, t_lift):
    """Fixed-window phase index of observation k (k = 0 initial, k = after action k - 1) from source event times."""
    if t_close < 0 or k <= t_close:
        return 0
    if t_lift < 0 or k <= t_lift:
        return 1
    return 2


class PhaseGate:
    def __init__(self, window, lift_m=0.03, yoke=False):
        assert window in WINDOWS, window
        assert lift_m > 0, lift_m
        self.window, self.lift_m = window, float(lift_m)
        self.yoke = bool(yoke)
        self._yoke_next = None  # (t_close, t_lift, src) set by set_yoke, consumed by the next reset
        self.yoke_t = None
        self.reset(0.0, _init=True)

    def set_yoke(self, t_close, t_lift, src=""):
        """Yoke mode only: source event times for the NEXT episode (call before reset / set_init_state)."""
        assert self.yoke, "set_yoke on a PhaseGate built without yoke=True"
        tc, tl = int(t_close), int(t_lift)
        assert tc >= -1 and tl >= -1, (tc, tl)
        assert tc >= 0 or tl == -1, f"source lifted without closing: t_close={tc} t_lift={tl}"
        assert tl == -1 or tl > tc, f"source lift not after the close: t_close={tc} t_lift={tl}"
        self._yoke_next = (tc, tl, str(src))

    def reset(self, eef_z, _init=False):
        """Call once per episode with the hand height of the initial observation. Yoke mode: needs a set_yoke for
        this episode (each source time pair is used by exactly one episode)."""
        if self.yoke and not _init:
            assert self._yoke_next is not None, "yoke mode: set_yoke was not called for this episode"
            self.yoke_t, self._yoke_next = self._yoke_next, None
        self.phase = "approach"
        self.t = 0  # number of policy actions seen
        self.t_close = None  # index of the first action that commanded close
        self.t_lift = None  # index of the action after which the lift threshold was passed
        self._zmin = None
        self.phases = [PHASES.index(self.phase)]  # phase of observation k (k = 0 initial, k = after action k - 1)
        if self.yoke:
            self.phases_own = list(self.phases)  # own-event phases; self.phases holds the fixed-window ones
            self.phases[0] = self._yoke_phase(0)
        self.gates = [self.active()]
        self._z0 = float(eef_z)

    def _yoke_phase(self, k):
        if self.yoke_t is None:  # only before the first episode (constructor reset)
            return 0
        return yoke_phase(k, self.yoke_t[0], self.yoke_t[1])

    def update(self, action, eef_z):
        """Call BEFORE executing the action. action: the env action about to be executed (last entry = gripper,
        > 0 closes); eef_z: hand height of the current (pre-action) observation, so the lift is detected one step
        late. Returns whether the observation this action will produce must be intervened on."""
        z = float(eef_z)
        if self.phase == "approach" and float(action[-1]) > 0:
            self.phase, self.t_close, self._zmin = "grasp", self.t, z
        elif self.phase == "grasp":
            self._zmin = min(self._zmin, z)
            if z - self._zmin >= self.lift_m:
                self.phase, self.t_lift = "transport", self.t
        self.t += 1
        if self.yoke:
            self.phases_own.append(PHASES.index(self.phase))
            self.phases.append(self._yoke_phase(self.t))
        else:
            self.phases.append(PHASES.index(self.phase))
        self.gates.append(self.active())
        return self.gates[-1]

    def active(self):
        """Gate of the latest observation: event phase, or in yoke mode the fixed-window phase."""
        if self.yoke:
            return self.window == "all" or self.window == PHASES[self.phases[-1]]
        return self.window == "all" or self.window == self.phase

    def fields(self, mode):
        """npz fields for the episode dump."""
        return {"pg_mode": mode, "pg_window": self.window, "pg_lift_m": self.lift_m,
                "pg_phase": np.asarray(self.phases, np.int8), "pg_gate": np.asarray(self.gates, bool),
                "pg_t_close": -1 if self.t_close is None else int(self.t_close),
                "pg_t_lift": -1 if self.t_lift is None else int(self.t_lift),
                **(self._yoke_fields() if self.yoke else {})}

    def _yoke_fields(self):
        """Yoke-only fields (absent without yoke, so non-yoke dumps keep exactly their old keys)."""
        tc, tl, src = self.yoke_t if self.yoke_t is not None else (-1, -1, "")  # None only before any episode
        return {"pg_yoke": np.int8(1), "pg_yoke_t_close": int(tc), "pg_yoke_t_lift": int(tl),
                "pg_yoke_src": src, "pg_phase_own": np.asarray(self.phases_own, np.int8)}
