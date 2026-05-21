"""Evaluation helpers for scheduling environments."""

from __future__ import annotations

import math
import random

from rl_utils import sample_from_probs


def evaluate_scheduling_policy(env, policy, num_episodes=200, max_steps=200, start_state=None, seed=12345):
    if start_state is None:
        start_state = env.start_state

    returns = []
    action_counts = {a: 0 for a in env.actions}

    old_random_state = random.getstate()
    old_env_rng_state = env.rng.getstate()

    for episode_idx in range(num_episodes):
        random.seed(seed + episode_idx)
        env.rng.seed(seed + episode_idx)
        state = start_state
        ep_return = 0.0
        for _t in range(max_steps):
            action = sample_from_probs(policy[state])
            action_counts[action] += 1
            next_state, reward, done = env.step(state, action)
            ep_return += reward
            state = next_state
            if done:
                break
        returns.append(ep_return)

    random.setstate(old_random_state)
    env.rng.setstate(old_env_rng_state)

    total_actions = sum(action_counts.values()) or 1
    avg_return = sum(returns) / len(returns)
    if len(returns) > 1:
        variance = sum((value - avg_return) ** 2 for value in returns) / (len(returns) - 1)
        stderr = math.sqrt(variance / len(returns))
    else:
        stderr = 0.0
    return {
        "avg_return": avg_return,
        "stderr": stderr,
        "action_rate": {a: action_counts[a] / total_actions for a in env.actions},
    }


def print_policy_score_table(rows):
    print("=== Scheduling Policy Comparison ===")
    print("Method                  AvgReturn    StdErr")
    print("-------------------------------------------")
    for row in rows:
        print(f"{row['name']:<23s}{row['avg_return']:10.2f}{row.get('stderr', 0.0):10.2f}")
    print()
