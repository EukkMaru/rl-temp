"""Shared pieces for the scheduling environments."""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple


Direction = int  # -1 or +1


def bit_is_set(mask: int, idx: int) -> bool:
    return (mask & (1 << idx)) != 0


def set_bit(mask: int, idx: int) -> int:
    return mask | (1 << idx)


def clear_bit(mask: int, idx: int) -> int:
    return mask & ~(1 << idx)


def mask_indices(mask: int, n: int) -> List[int]:
    return [i for i in range(n) if bit_is_set(mask, i)]


def mask_count(mask: int) -> int:
    return int(mask.bit_count())


class BaseSchedulingEnv(ABC):
    """Base class for the small finite MDPs used in this project."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._transition_cache: Dict[Tuple[tuple, str], Dict[tuple, float]] = {}
        self._reward_cache: Dict[Tuple[tuple, str], float] = {}

    def is_terminal(self, state):
        # Episodes stop by horizon, not by terminal states.
        return False

    def get_random_policy(self):
        policy = {}
        for s in self.states:
            policy[s] = {a: 1.0 / len(self.actions) for a in self.actions}
        return policy

    def next_state(self, state, action):
        # Deterministic action effect before stochastic arrivals.
        post_state, _, _ = self._apply_action_no_arrival(state, action)
        return post_state

    def step(self, state, action):
        post_state, reward, _info = self._apply_action_no_arrival(state, action)
        next_state = self._sample_arrival(post_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def transition_prob(self, s_next, state, action):
        key = (state, action)
        if key not in self._transition_cache:
            post_state, _reward, _info = self._apply_action_no_arrival(state, action)
            probs = defaultdict(float)
            for arrival_state, p in self._arrival_outcomes(post_state):
                probs[arrival_state] += p
            self._transition_cache[key] = dict(probs)
        return self._transition_cache[key].get(s_next, 0.0)

    def reward(self, state, action, s_next):
        # Reward is based on the action effect; arrivals only affect the next state.
        key = (state, action)
        if key not in self._reward_cache:
            _post_state, reward, _info = self._apply_action_no_arrival(state, action)
            self._reward_cache[key] = reward
        return self._reward_cache[key]

    def print_values(self, V, decimals=2, max_rows=40):
        print(f"=== Values for {self.__class__.__name__} ===")
        for s in self.states[:max_rows]:
            print(f"{self._state_label(s):<35s} {V.get(s, 0.0):8.{decimals}f}")
        if len(self.states) > max_rows:
            print(f"... ({len(self.states) - max_rows} more states omitted)")
        print()

    def print_policy(self, policy, max_rows=40):
        print(f"=== Policy for {self.__class__.__name__} ===")
        for s in self.states[:max_rows]:
            if isinstance(policy[s], dict):
                best_action = max(policy[s], key=policy[s].get)
            else:
                best_action = policy[s]
            print(f"{self._state_label(s):<35s} -> {self._action_symbol(best_action)}")
        if len(self.states) > max_rows:
            print(f"... ({len(self.states) - max_rows} more states omitted)")
        print()

    @abstractmethod
    def _apply_action_no_arrival(self, state, action):
        """Return (post_action_state, reward, info)."""
        raise NotImplementedError

    @abstractmethod
    def _arrival_outcomes(self, post_action_state):
        """Yield (next_state, probability) for all possible arrival outcomes."""
        raise NotImplementedError

    @abstractmethod
    def _sample_arrival(self, post_action_state):
        """Sample and return next_state after stochastic arrivals."""
        raise NotImplementedError

    @abstractmethod
    def _state_label(self, state) -> str:
        raise NotImplementedError

    @abstractmethod
    def _action_symbol(self, action) -> str:
        raise NotImplementedError


class LinearRequestSchedulingEnv(BaseSchedulingEnv):
    """Common line-based request model used by disk and elevator examples."""

    def __init__(
        self,
        n_positions: int = 5,
        request_prob: float = 0.10,
        start_position: int | None = None,
        start_direction: Direction = 1,
        start_mask: int = 0,
        move_cost: float = 0.20,
        waiting_cost: float = 1.00,
        service_reward: float = 3.00,
        invalid_penalty: float = 2.00,
        wait_penalty: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        if n_positions < 2:
            raise ValueError("n_positions must be at least 2")
        if not (0.0 <= request_prob <= 1.0):
            raise ValueError("request_prob must be in [0, 1]")

        self.n_positions = n_positions
        self.request_prob = request_prob
        self.move_cost = move_cost
        self.waiting_cost = waiting_cost
        self.service_reward = service_reward
        self.invalid_penalty = invalid_penalty
        self.wait_penalty = wait_penalty

        self.left_action = "left"
        self.right_action = "right"
        self.serve_action = "serve"
        self.wait_action = "wait"
        self.actions = [self.left_action, self.right_action, self.serve_action, self.wait_action]

        self.states = [
            (pos, direction, mask)
            for pos in range(n_positions)
            for direction in (-1, 1)
            for mask in range(1 << n_positions)
        ]

        if start_position is None:
            start_position = n_positions // 2
        self.start_state = (start_position, start_direction, start_mask)

    def _base_cost(self, mask_after_action: int) -> float:
        return -self.waiting_cost * mask_count(mask_after_action)

    def _apply_action_no_arrival(self, state, action):
        pos, direction, mask = state
        invalid = False
        moved = False
        served = False
        waited = False

        if action == self.left_action:
            if pos == 0:
                invalid = True
            else:
                pos -= 1
                direction = -1
                moved = True

        elif action == self.right_action:
            if pos == self.n_positions - 1:
                invalid = True
            else:
                pos += 1
                direction = 1
                moved = True

        elif action == self.serve_action:
            if bit_is_set(mask, pos):
                mask = clear_bit(mask, pos)
                served = True
            else:
                invalid = True

        elif action == self.wait_action:
            waited = True

        else:
            raise ValueError(f"Unknown action: {action}")

        reward = self._base_cost(mask)
        if moved:
            reward -= self.move_cost
        if waited:
            reward -= self.wait_penalty
        if served:
            reward += self.service_reward
        if invalid:
            reward -= self.invalid_penalty

        post_state = (pos, direction, mask)
        info = {"moved": moved, "served": served, "invalid": invalid, "waited": waited}
        return post_state, reward, info

    def _arrival_outcomes(self, post_action_state):
        pos, direction, mask = post_action_state
        empty_positions = [i for i in range(self.n_positions) if not bit_is_set(mask, i)]

        for bits in itertools.product([0, 1], repeat=len(empty_positions)):
            next_mask = mask
            p = 1.0
            for idx, arrives in zip(empty_positions, bits):
                if arrives:
                    next_mask = set_bit(next_mask, idx)
                    p *= self.request_prob
                else:
                    p *= 1.0 - self.request_prob
            yield (pos, direction, next_mask), p

    def _sample_arrival(self, post_action_state):
        pos, direction, mask = post_action_state
        next_mask = mask
        for i in range(self.n_positions):
            if not bit_is_set(next_mask, i) and self.rng.random() < self.request_prob:
                next_mask = set_bit(next_mask, i)
        return (pos, direction, next_mask)

    def _state_label(self, state) -> str:
        pos, direction, mask = state
        d = "R" if direction == 1 else "L"
        req = "".join("1" if bit_is_set(mask, i) else "0" for i in range(self.n_positions))
        return f"pos={pos}, dir={d}, req={req}"

    def _action_symbol(self, action) -> str:
        return str(action)
