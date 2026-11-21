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

PG_WINDOW: none | approach | grasp | transport | all | post | av | ar. "none" and "all" are the two anchors (never / always).
"post" (phasegate-confirm, 2026-11-10) is open in grasp and transport, i.e. the complement of approach: from the
observation after the first close command to the end of the episode.
"av" and "ar" (phasegate-recovery, 2026-11-11, protocol_recovery.md v1 section 2.2) add a sub-state that runs beside
the phase and looks only at the gripper command: sub = pre (no close command yet), verify (from the observation after
the first close command up to the observation before the first later open command), reapproach (from the
observation produced by that open command on, latched). av is open in approach or verify, ar in approach or reapproach. Identities per observation:
av or ar = all, av and ar = approach. The six earlier windows return exactly what they returned before.

These policies read a single frame per query, so switching the image between steps adds no temporal inconsistency
of its own.
"""
import numpy as np

WINDOWS = ("none", "approach", "grasp", "transport", "all", "post", "av", "ar")
MODES = ("look", "erase")
PHASES = ("approach", "grasp", "transport")
SUBS = ("pre", "verify", "reapproach")  # phasegate-recovery: gripper-command sub-state, dumped as pg_sub


class PhaseGate:
    def __init__(self, window, lift_m=0.03):
        assert window in WINDOWS, window
        assert lift_m > 0, lift_m
        self.window, self.lift_m = window, float(lift_m)
        self.reset(0.0)

    def reset(self, eef_z):
        """Call once per episode with the hand height of the initial observation."""
        self.phase = "approach"
        self.t = 0  # number of policy actions seen
        self.t_close = None  # index of the first action that commanded close
        self.t_lift = None  # index of the action after which the lift threshold was passed
        self._zmin = None
        self.sub = "pre"  # phasegate-recovery
        self.t_reopen = None  # index of the first action after t_close that commanded open
        self.subs = [SUBS.index(self.sub)]
        self.phases = [PHASES.index(self.phase)]  # phase of observation k (k = 0 initial, k = after action k - 1)
        self.gates = [self.active()]
        self._z0 = float(eef_z)

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
        # phasegate-recovery: sub-state from the gripper command only (the phase logic above is unchanged)
        if self.t_close is not None and self.t_reopen is None and self.t > self.t_close and float(action[-1]) <= 0:
            self.t_reopen = self.t
        self.sub = "pre" if self.t_close is None else ("verify" if self.t_reopen is None else "reapproach")
        self.t += 1
        self.phases.append(PHASES.index(self.phase))
        self.subs.append(SUBS.index(self.sub))
        self.gates.append(self.active())
        return self.gates[-1]

    def active(self):
        if self.window == "post":  # phasegate-confirm: complement of approach
            return self.phase != "approach"
        if self.window == "av":  # phasegate-recovery: approach + verify
            return self.phase == "approach" or self.sub == "verify"
        if self.window == "ar":  # phasegate-recovery: approach + reapproach
            return self.phase == "approach" or self.sub == "reapproach"
        return self.window == "all" or self.window == self.phase

    def fields(self, mode):
        """npz fields for the episode dump."""
        return {"pg_mode": mode, "pg_window": self.window, "pg_lift_m": self.lift_m,
                "pg_phase": np.asarray(self.phases, np.int8), "pg_gate": np.asarray(self.gates, bool),
                "pg_t_close": -1 if self.t_close is None else int(self.t_close),
                "pg_t_lift": -1 if self.t_lift is None else int(self.t_lift),
                "pg_sub": np.asarray(self.subs, np.int8),  # phasegate-recovery (written for every window)
                "pg_t_reopen": -1 if self.t_reopen is None else int(self.t_reopen)}
