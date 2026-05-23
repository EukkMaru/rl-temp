"""Disk-head scheduling wrapper."""

from __future__ import annotations

from scheduling_common import LinearRequestSchedulingEnv, bit_is_set


class DiskSchedulingEnv(LinearRequestSchedulingEnv):
    """
    Disk head scheduling over a linear cylinder/track abstraction.

    State:
        (head_track, direction, request_mask)

    Actions:
        seek_left, seek_right, serve, wait
    """

    def __init__(self, n_tracks=6, request_prob=0.08, **kwargs):
        super().__init__(n_positions=n_tracks, request_prob=request_prob, **kwargs)
        self.actions = ["seek_left", "seek_right", "serve", "wait"]
        self.left_action = "seek_left"
        self.right_action = "seek_right"
        self.serve_action = "serve"
        self.wait_action = "wait"

    def _state_label(self, state) -> str:
        pos, direction, mask = state
        d = "R" if direction == 1 else "L"
        req = "".join("1" if bit_is_set(mask, i) else "0" for i in range(self.n_positions))
        return f"track={pos}, dir={d}, io={req}"

    def _action_symbol(self, action) -> str:
        return {"seek_left": "<", "seek_right": ">", "serve": "S", "wait": "W"}.get(action, action)
