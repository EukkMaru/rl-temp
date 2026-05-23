"""Shared base class for the scheduling MDP."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple


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
