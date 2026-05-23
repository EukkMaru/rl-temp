"""CPU scheduling heuristic policies."""

from __future__ import annotations

def deterministic_policy(env, action_fn):
    policy = {}
    for s in env.states:
        a = action_fn(s)
        policy[s] = {action: 0.0 for action in env.actions}
        policy[s][a] = 1.0
    return policy


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
