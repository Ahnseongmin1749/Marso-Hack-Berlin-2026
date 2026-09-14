"""Axis-locked pick/place sequencer for the WarehouseSort RGB track.

The sequencer deliberately has no access to simulator object state. A caller must
provide the currently locked parcel/bin XY coordinates from an RGB perception
module (or a calibration-based detector). It then emits normalized
``pd_ee_delta_pos`` actions with only the phase-appropriate degrees of freedom.
"""
from dataclasses import dataclass
from enum import Enum, auto

import torch


SEQUENCED_TASK = (
    "Sort one parcel at a time into its matching color bin. "
    "Lock the current target until release. Follow this exact sequence: "
    "move horizontally at a safe height, descend, close, lift, move to the matching bin, "
    "descend, open, then lift. Use the shortest direct collision-free motion; never idle, "
    "hesitate, backtrack, or change targets mid-sequence."
)


class Phase(Enum):
    MOVE_X_PICK = auto()
    MOVE_Y_PICK = auto()
    DOWN_PICK = auto()
    CLOSE = auto()
    LIFT_PICK = auto()
    MOVE_X_BIN = auto()
    MOVE_Y_BIN = auto()
    DOWN_BIN = auto()
    OPEN = auto()
    ABORT_OPEN = auto()
    RETREAT = auto()


@dataclass(frozen=True)
class SequencerTarget:
    parcel_xy: tuple[float, float]
    bin_xy: tuple[float, float]


class AxisLockedSequencer:
    """Deterministic one-parcel state machine.

    ``state`` is the raw 26D WarehouseSort state. Its TCP position is indices
    18:21. Actions are normalized for Panda's +/-0.1m Cartesian delta limits.
    Call ``set_target`` only after the previous target has been released.
    """

    def __init__(self, safe_z=0.20, pick_z=0.061, drop_z=0.070,
                 tolerance=0.006, tolerance_z=0.002, hold_steps=3, phase_budget=15):
        self.safe_z = float(safe_z)
        self.pick_z = float(pick_z)
        self.drop_z = float(drop_z)
        self.tolerance = float(tolerance)
        self.tolerance_z = float(tolerance_z)
        self.hold_steps = int(hold_steps)
        self.phase_budget = int(phase_budget)
        self.target = None
        self.phase = None
        self.phase_steps = 0
        self.completed = False
        self.failed = False

    def set_target(self, target: SequencerTarget):
        if self.target is not None and not self.completed:
            raise RuntimeError("current target is still locked")
        self.target = target
        self.phase = Phase.MOVE_X_PICK
        self.phase_steps = 0
        self.completed = False
        self.failed = False

    def reset(self):
        self.target = None
        self.phase = None
        self.phase_steps = 0
        self.completed = False
        self.failed = False

    def _advance(self):
        if self.phase in (Phase.OPEN, Phase.ABORT_OPEN):
            self.phase = Phase.RETREAT
            self.phase_steps = 0
            return
        order = list(Phase)
        i = order.index(self.phase)
        if i + 1 == len(order):
            self.completed = True
            self.phase = None
        else:
            self.phase = order[i + 1]
            self.phase_steps = 0

    def _command(self, state, axis, target, gripper, device, tolerance=None):
        pos = state[18:21]
        error = float(target) - float(pos[axis])
        action = torch.zeros(4, dtype=torch.float32, device=device)
        action[3] = float(gripper)
        if abs(error) <= (self.tolerance if tolerance is None else tolerance) or self.phase_steps >= self.phase_budget:
            self._advance()
            return action, True
        action[axis] = max(-1.0, min(1.0, error / 0.1))
        return action, False

    def _command_locked_xy_z(self, state, x, y, z, gripper, device):
        """Move Z while compensating the small XY drift caused by IK.

        The high-level route remains X -> Y -> Z.  This low-level hold is only
        active during descent and prevents the Panda wrist from walking off the
        parcel/bin center as the arm changes configuration.
        """
        pos = state[18:21]
        action = torch.zeros(4, dtype=torch.float32, device=device)
        action[3] = float(gripper)
        ex, ey, ez = float(x) - float(pos[0]), float(y) - float(pos[1]), float(z) - float(pos[2])
        action[0] = max(-1.0, min(1.0, ex / 0.1))
        action[1] = max(-1.0, min(1.0, ey / 0.1))
        action[2] = max(-1.0, min(1.0, ez / 0.1))
        done = (
            abs(ex) <= self.tolerance
            and abs(ey) <= self.tolerance
            and abs(ez) <= self.tolerance_z
        ) or self.phase_steps >= self.phase_budget
        if done:
            self._advance()
            return torch.zeros(4, dtype=torch.float32, device=device), True
        return action, False

    def act(self, state):
        """Return one normalized action for a single raw 26D state tensor."""
        if self.target is None or self.completed:
            return torch.zeros(4, dtype=torch.float32, device=state.device)
        self.phase_steps += 1
        px, py = self.target.parcel_xy
        bx, by = self.target.bin_xy
        if self.phase is Phase.MOVE_X_PICK:
            action, done = self._command(state, 0, px, 1.0, state.device)
        elif self.phase is Phase.MOVE_Y_PICK:
            action, done = self._command(state, 1, py, 1.0, state.device)
        elif self.phase is Phase.DOWN_PICK:
            action, done = self._command_locked_xy_z(state, px, py, self.pick_z, 1.0, state.device)
        elif self.phase is Phase.CLOSE:
            action = torch.tensor([0, 0, 0, -1], dtype=torch.float32, device=state.device)
            done = self.phase_steps >= self.hold_steps
            if done:
                # The final state element is the RGB-track-safe proprioceptive
                # ``is_grasped`` bit.  Do not waste the remaining horizon
                # carrying an object that was never captured.
                grasped = state.numel() >= 26 and float(state[25]) > 0.5
                if grasped:
                    self._advance()
                else:
                    self.failed = True
                    self.phase = Phase.ABORT_OPEN
                    self.phase_steps = 0
        elif self.phase is Phase.LIFT_PICK:
            action, done = self._command(state, 2, self.safe_z, -1.0, state.device)
        elif self.phase is Phase.MOVE_X_BIN:
            action, done = self._command(state, 0, bx, -1.0, state.device)
        elif self.phase is Phase.MOVE_Y_BIN:
            action, done = self._command(state, 1, by, -1.0, state.device)
        elif self.phase is Phase.DOWN_BIN:
            action, done = self._command_locked_xy_z(state, bx, by, self.drop_z, -1.0, state.device)
        elif self.phase is Phase.OPEN:
            action = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=state.device)
            done = self.phase_steps >= self.hold_steps
            if done: self._advance()
        elif self.phase is Phase.ABORT_OPEN:
            action = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=state.device)
            done = self.phase_steps >= self.hold_steps
            if done: self._advance()
        elif self.phase is Phase.RETREAT:
            action, done = self._command(state, 2, self.safe_z, 1.0, state.device)
        else:  # pragma: no cover
            raise AssertionError(self.phase)
        return action
