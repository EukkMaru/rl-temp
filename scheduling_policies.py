"""
Heuristic policies for the scheduling MDP wrappers.

Policies are returned in the same stochastic-policy dictionary format used by
rl_utils.evaluate_policy and dp_solver:
    policy[state][action] = probability
"""

from __future__ import annotations

from scheduling_common import bit_is_set, mask_indices


def deterministic_policy(env, action_fn):
    policy = {}
    for s in env.states:
        a = action_fn(s)
        policy[s] = {action: 0.0 for action in env.actions}
        policy[s][a] = 1.0
    return policy


def random_policy(env):
    return env.get_random_policy()


def linear_nearest_request_policy(env):
    """Move toward the closest pending request; serve if already there."""
    def choose(state):
        pos, _direction, mask = state
        if bit_is_set(mask, pos):
            return env.serve_action
        pending = mask_indices(mask, env.n_positions)
        if not pending:
            return env.wait_action
        target = min(pending, key=lambda i: abs(i - pos))
        if target < pos:
            return env.left_action
        if target > pos:
            return env.right_action
        return env.serve_action
    return deterministic_policy(env, choose)


def linear_scan_policy(env):
    """SCAN/elevator-style policy: continue current direction until boundary."""
    def choose(state):
        pos, direction, mask = state
        if bit_is_set(mask, pos):
            return env.serve_action
        if mask == 0:
            return env.wait_action
        if direction == 1:
            return env.right_action if pos < env.n_positions - 1 else env.left_action
        return env.left_action if pos > 0 else env.right_action
    return deterministic_policy(env, choose)


def linear_look_policy(env):
    """LOOK policy: continue direction only while requests remain that way."""
    def choose(state):
        pos, direction, mask = state
        if bit_is_set(mask, pos):
            return env.serve_action
        if mask == 0:
            return env.wait_action
        pending = mask_indices(mask, env.n_positions)
        if direction == 1:
            if any(i > pos for i in pending):
                return env.right_action
            return env.left_action
        if any(i < pos for i in pending):
            return env.left_action
        return env.right_action
    return deterministic_policy(env, choose)


def cpu_shortest_job_first_policy(env):
    def choose(state):
        s, m, l, _mode = state
        if s > 0:
            return "run_short"
        if m > 0:
            return "run_medium"
        if l > 0:
            return "run_long"
        return "idle"
    return deterministic_policy(env, choose)


def cpu_longest_job_first_policy(env):
    def choose(state):
        s, m, l, _mode = state
        if l > 0:
            return "run_long"
        if m > 0:
            return "run_medium"
        if s > 0:
            return "run_short"
        return "idle"
    return deterministic_policy(env, choose)


def cpu_max_queue_policy(env):
    """Run the class with the largest queue; ties favor shorter jobs."""
    def choose(state):
        queues = state[:3]
        if sum(queues) == 0:
            return "idle"
        cls = max(range(3), key=lambda i: (queues[i], -i))
        return ["run_short", "run_medium", "run_long"][cls]
    return deterministic_policy(env, choose)


def cpu_sticky_policy(env):
    """Prefer the current mode if its queue is nonempty; otherwise use max-queue."""
    def choose(state):
        s, m, l, mode = state
        queues = [s, m, l]
        if sum(queues) == 0:
            return "idle"
        mode_to_action = {1: "run_short", 2: "run_medium", 3: "run_long"}
        if mode in (1, 2, 3) and queues[mode - 1] > 0:
            return mode_to_action[mode]
        cls = max(range(3), key=lambda i: (queues[i], -i))
        return ["run_short", "run_medium", "run_long"][cls]
    return deterministic_policy(env, choose)
