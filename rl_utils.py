"""Small tabular RL helper functions used by the scheduling experiments."""

from __future__ import annotations

import random
from collections import defaultdict


def make_Q(env, initial_value=0.0):
    return defaultdict(lambda: {action: float(initial_value) for action in env.actions})


def sample_from_probs(action_probs):
    r = random.random()
    cumulative = 0.0
    last_action = None
    for action, prob in action_probs.items():
        cumulative += prob
        last_action = action
        if r <= cumulative:
            return action
    return last_action


def greedy_action_from_Q(env, Q, state):
    values = Q[state]
    return max(env.actions, key=lambda action: (values[action], -env.actions.index(action)))


def epsilon_greedy_action(env, Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(env.actions)
    return greedy_action_from_Q(env, Q, state)


def greedy_policy_from_Q(env, Q):
    policy = {}
    for state in env.states:
        best_action = greedy_action_from_Q(env, Q, state)
        policy[state] = {action: 0.0 for action in env.actions}
        policy[state][best_action] = 1.0
    return policy


def epsilon_soft_policy_from_Q(env, Q, epsilon):
    policy = {}
    action_count = len(env.actions)
    for state in env.states:
        best_action = greedy_action_from_Q(env, Q, state)
        policy[state] = {action: epsilon / action_count for action in env.actions}
        policy[state][best_action] += 1.0 - epsilon
    return policy
