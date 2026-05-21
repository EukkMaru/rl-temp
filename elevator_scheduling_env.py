"""Single-elevator pickup-call scheduling wrapper."""

from __future__ import annotations

from scheduling_common import LinearRequestSchedulingEnv, bit_is_set


class ElevatorSchedulingEnv(LinearRequestSchedulingEnv):
    """
    Simplified single-elevator pickup-call scheduling.

    State:
        (floor, direction, call_mask)

    Actions:
        move_down, move_up, open, wait

    This simplified wrapper models pickup calls only. It is intentionally kept
    small so that exact DP and tabular RL are both feasible.
    """

    def __init__(self, n_floors=5, request_prob=0.10, **kwargs):
        super().__init__(n_positions=n_floors, request_prob=request_prob, **kwargs)
        self.actions = ["move_down", "move_up", "open", "wait"]
        self.left_action = "move_down"
        self.right_action = "move_up"
        self.serve_action = "open"
        self.wait_action = "wait"

    def _state_label(self, state) -> str:
        pos, direction, mask = state
        d = "U" if direction == 1 else "D"
        req = "".join("1" if bit_is_set(mask, i) else "0" for i in range(self.n_positions))
        return f"floor={pos}, dir={d}, calls={req}"

    def _action_symbol(self, action) -> str:
        return {"move_down": "Down", "move_up": "Up", "open": "Open", "wait": "W"}.get(action, action)
