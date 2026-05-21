"""CPU job-class scheduling wrapper."""

from __future__ import annotations

import itertools
from typing import Sequence

from scheduling_common import BaseSchedulingEnv


class CPUSchedulingEnv(BaseSchedulingEnv):
    """
    CPU job-class scheduling MDP.

    State:
        (short_queue, medium_queue, long_queue, current_mode)

    current_mode:
        0 = none/idle, 1 = short, 2 = medium, 3 = long

    Actions:
        run_short, run_medium, run_long, idle
    """

    MODE_NONE = 0
    MODE_SHORT = 1
    MODE_MEDIUM = 2
    MODE_LONG = 3

    def __init__(
        self,
        max_queue: int = 3,
        arrival_probs: Sequence[float] = (0.40, 0.25, 0.15),
        waiting_costs: Sequence[float] = (1.0, 1.0, 1.0),
        service_rewards: Sequence[float] = (2.0, 2.5, 3.0),
        switch_cost: float = 0.30,
        invalid_penalty: float = 2.00,
        idle_penalty: float = 0.10,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        if max_queue < 1:
            raise ValueError("max_queue must be at least 1")
        if len(arrival_probs) != 3:
            raise ValueError("arrival_probs must have three probabilities")
        if any(p < 0 or p > 1 for p in arrival_probs):
            raise ValueError("arrival probabilities must be in [0, 1]")

        self.max_queue = max_queue
        self.arrival_probs = tuple(float(p) for p in arrival_probs)
        self.waiting_costs = tuple(float(c) for c in waiting_costs)
        self.service_rewards = tuple(float(r) for r in service_rewards)
        self.switch_cost = switch_cost
        self.invalid_penalty = invalid_penalty
        self.idle_penalty = idle_penalty

        self.actions = ["run_short", "run_medium", "run_long", "idle"]
        self.action_to_class = {
            "run_short": 0,
            "run_medium": 1,
            "run_long": 2,
        }
        self.class_to_mode = [self.MODE_SHORT, self.MODE_MEDIUM, self.MODE_LONG]
        self.mode_names = {0: "none", 1: "short", 2: "medium", 3: "long"}

        self.states = [
            (s, m, l, mode)
            for s in range(max_queue + 1)
            for m in range(max_queue + 1)
            for l in range(max_queue + 1)
            for mode in (self.MODE_NONE, self.MODE_SHORT, self.MODE_MEDIUM, self.MODE_LONG)
        ]
        self.start_state = (0, 0, 0, self.MODE_NONE)

    def _queue_wait_cost(self, queues: Sequence[int]) -> float:
        return -sum(q * c for q, c in zip(queues, self.waiting_costs))

    def _apply_action_no_arrival(self, state, action):
        queues = list(state[:3])
        current_mode = state[3]
        invalid = False
        completed = False
        switched = False
        idled = False
        reward = 0.0

        if action in self.action_to_class:
            cls = self.action_to_class[action]
            target_mode = self.class_to_mode[cls]
            if queues[cls] <= 0:
                invalid = True
            else:
                queues[cls] -= 1
                completed = True
                reward += self.service_rewards[cls]
                if current_mode not in (self.MODE_NONE, target_mode):
                    switched = True
                    reward -= self.switch_cost
                current_mode = target_mode

        elif action == "idle":
            idled = True
            current_mode = self.MODE_NONE
            reward -= self.idle_penalty

        else:
            raise ValueError(f"Unknown action: {action}")

        reward += self._queue_wait_cost(queues)
        if invalid:
            reward -= self.invalid_penalty
        post_state = (queues[0], queues[1], queues[2], current_mode)
        info = {"completed": completed, "invalid": invalid, "switched": switched, "idled": idled}
        return post_state, reward, info

    def _arrival_outcomes(self, post_action_state):
        queues = list(post_action_state[:3])
        mode = post_action_state[3]

        for bits in itertools.product([0, 1], repeat=3):
            next_queues = list(queues)
            p = 1.0
            for cls, arrives in enumerate(bits):
                prob = self.arrival_probs[cls]
                if arrives:
                    p *= prob
                    if next_queues[cls] < self.max_queue:
                        next_queues[cls] += 1
                else:
                    p *= 1.0 - prob
            yield (next_queues[0], next_queues[1], next_queues[2], mode), p

    def _sample_arrival(self, post_action_state):
        queues = list(post_action_state[:3])
        mode = post_action_state[3]
        for cls, prob in enumerate(self.arrival_probs):
            if self.rng.random() < prob and queues[cls] < self.max_queue:
                queues[cls] += 1
        return (queues[0], queues[1], queues[2], mode)

    def _state_label(self, state) -> str:
        s, m, l, mode = state
        return f"S={s}, M={m}, L={l}, mode={self.mode_names[mode]}"

    def _action_symbol(self, action) -> str:
        return {
            "run_short": "Run-S",
            "run_medium": "Run-M",
            "run_long": "Run-L",
            "idle": "Idle",
        }.get(action, action)
