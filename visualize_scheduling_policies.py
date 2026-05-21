"""Text view of a few rollout traces."""

from __future__ import annotations

import random

from control_agents import MonteCarloControlAgent, QLearningAgent, SARSAAgent
from cpu_scheduling_env import CPUSchedulingEnv
from disk_scheduling_env import DiskSchedulingEnv
from dp_solver import DynamicProgrammingSolver
from elevator_scheduling_env import ElevatorSchedulingEnv
from rl_utils import greedy_policy_from_Q
from scheduling_common import bit_is_set, set_bit
from scheduling_policies import (
    cpu_longest_job_first_policy,
    cpu_max_queue_policy,
    cpu_shortest_job_first_policy,
    cpu_sticky_policy,
    linear_look_policy,
    linear_nearest_request_policy,
    linear_scan_policy,
)


STEPS = 12
VIS_TRAIN_EPISODES = 2000
CPU_VIS_TRAIN_EPISODES = 2500


def mask_from_indices(indices):
    mask = 0
    for idx in indices:
        mask = set_bit(mask, idx)
    return mask


def choose_action(policy, state):
    action_probs = policy[state]
    return max(action_probs, key=action_probs.get)


def simulate(env, policy, start_state, steps=STEPS):
    frames = []
    state = start_state
    total_reward = 0.0

    for t in range(steps):
        action = choose_action(policy, state)
        next_state, reward, done = env.step(state, action)
        total_reward += reward
        frames.append(
            {
                "t": t,
                "state": state,
                "action": action,
                "reward": reward,
                "total_reward": total_reward,
                "next_state": next_state,
            }
        )
        state = next_state
        if done:
            break

    return frames


def render_linear_state(env, state, marker):
    pos, _direction, mask = state
    cells = []
    for idx in range(env.n_positions):
        if idx == pos and bit_is_set(mask, idx):
            cells.append(f"[{marker}*]")
        elif idx == pos:
            cells.append(f"[{marker} ]")
        elif bit_is_set(mask, idx):
            cells.append("[ *]")
        else:
            cells.append(f"[{idx} ]")
    return " ".join(cells)


def render_disk_frame(env, frame):
    pos, direction, _mask = frame["state"]
    direction_label = "R" if direction == 1 else "L"
    return (
        f"t={frame['t']:02d} | {render_linear_state(env, frame['state'], 'H')} "
        f"| track={pos} dir={direction_label} action={frame['action']:<10s} "
        f"reward={frame['reward']:6.2f}"
    )


def render_elevator_frame(env, frame):
    pos, direction, _mask = frame["state"]
    direction_label = "U" if direction == 1 else "D"
    return (
        f"t={frame['t']:02d} | {render_linear_state(env, frame['state'], 'E')} "
        f"| floor={pos} dir={direction_label} action={frame['action']:<10s} "
        f"reward={frame['reward']:6.2f}"
    )


def render_cpu_frame(_env, frame):
    s, m, l, mode = frame["state"]
    mode_names = {0: "idle", 1: "short", 2: "medium", 3: "long"}
    return (
        f"t={frame['t']:02d} | short:{'#' * s or '-':<3s} "
        f"medium:{'#' * m or '-':<3s} long:{'#' * l or '-':<3s} "
        f"| mode={mode_names[mode]:<6s} action={frame['action']:<10s} "
        f"reward={frame['reward']:6.2f}"
    )


def print_policy_run(name, frames, render_frame, selected=False):
    label = f"{name} [SELECTED]" if selected else name
    total = frames[-1]["total_reward"] if frames else 0.0
    print(f"\n  {label}")
    print(f"  {'-' * len(label)}")
    for frame in frames:
        print("  " + render_frame(frame))
    print(f"  Total return over {len(frames)} shown steps: {total:.2f}")


def train_learning_policies(env_factory, train_episodes, max_steps, seed_base):
    policies = []

    print("  preparing Value Iteration...", flush=True)
    dp_env = env_factory(seed_base)
    dp_policy, _dp_V = DynamicProgrammingSolver(dp_env, gamma=0.95, theta=1e-4).value_iteration()
    policies.append(("Value Iteration", dp_env, dp_policy, False))

    baseline_episodes = min(1000, train_episodes)

    print(f"  training MC Control ({baseline_episodes} episodes)...", flush=True)
    mc_env = env_factory(seed_base + 1)
    random.seed(seed_base + 1)
    mc_env.rng.seed(seed_base + 1)
    mc = MonteCarloControlAgent(mc_env, gamma=0.95, epsilon=0.08, max_steps=max_steps)
    mc.train(num_episodes=baseline_episodes, start_state=mc_env.start_state)
    policies.append(("MC Control", mc_env, greedy_policy_from_Q(mc_env, mc.Q), False))

    print(f"  training SARSA ({baseline_episodes} episodes)...", flush=True)
    sarsa_env = env_factory(seed_base + 2)
    random.seed(seed_base + 2)
    sarsa_env.rng.seed(seed_base + 2)
    sarsa = SARSAAgent(sarsa_env, gamma=0.95, alpha=0.12, epsilon=0.08, max_steps=max_steps)
    sarsa.train(num_episodes=baseline_episodes, start_state=sarsa_env.start_state)
    policies.append(("SARSA", sarsa_env, greedy_policy_from_Q(sarsa_env, sarsa.Q), False))

    print(f"  training selected Q-learning ({train_episodes} episodes)...", flush=True)
    q_env = env_factory(seed_base + 3)
    random.seed(seed_base + 3)
    q_env.rng.seed(seed_base + 3)
    q_agent = QLearningAgent(q_env, gamma=0.95, alpha=0.12, epsilon=0.08, max_steps=max_steps)
    q_agent.train(num_episodes=train_episodes, start_state=q_env.start_state)
    policies.append(("Q-learning", q_env, greedy_policy_from_Q(q_env, q_agent.Q), True))

    print("  done.\n", flush=True)
    return policies


def print_domain(title, subtitle):
    print("\n" + "=" * 88)
    print(title)
    print(subtitle)
    print("=" * 88)


def visualize_disk():
    print_domain(
        "Disk Head Scheduling",
        "H is the disk head. * means a pending I/O request. [H*] means request at current head.",
    )
    start_state = (2, 1, mask_from_indices([0, 3]))
    specs = [
        ("Nearest", lambda env: linear_nearest_request_policy(env), False),
        ("SCAN", lambda env: linear_scan_policy(env), False),
        ("LOOK", lambda env: linear_look_policy(env), False),
    ]
    for idx, (name, policy_factory, selected) in enumerate(specs):
        env = DiskSchedulingEnv(n_tracks=4, request_prob=0.08, seed=100 + idx)
        frames = simulate(env, policy_factory(env), start_state)
        print_policy_run(name, frames, lambda frame: render_disk_frame(env, frame), selected)

    for name, env, policy, selected in train_learning_policies(
        lambda seed: DiskSchedulingEnv(n_tracks=4, request_prob=0.08, seed=seed),
        train_episodes=VIS_TRAIN_EPISODES,
        max_steps=100,
        seed_base=110,
    ):
        frames = simulate(env, policy, start_state)
        print_policy_run(name, frames, lambda frame: render_disk_frame(env, frame), selected)


def visualize_elevator():
    print_domain(
        "Elevator Pickup Scheduling",
        "E is the elevator. * means a pending pickup call. [E*] means call at current floor.",
    )
    start_state = (2, 1, mask_from_indices([0, 3]))
    specs = [
        ("Nearest", lambda env: linear_nearest_request_policy(env), False),
        ("Collective / SCAN", lambda env: linear_scan_policy(env), False),
        ("LOOK", lambda env: linear_look_policy(env), False),
    ]
    for idx, (name, policy_factory, selected) in enumerate(specs):
        env = ElevatorSchedulingEnv(n_floors=4, request_prob=0.10, seed=200 + idx)
        frames = simulate(env, policy_factory(env), start_state)
        print_policy_run(name, frames, lambda frame: render_elevator_frame(env, frame), selected)

    for name, env, policy, selected in train_learning_policies(
        lambda seed: ElevatorSchedulingEnv(n_floors=4, request_prob=0.10, seed=seed),
        train_episodes=VIS_TRAIN_EPISODES,
        max_steps=100,
        seed_base=210,
    ):
        frames = simulate(env, policy, start_state)
        print_policy_run(name, frames, lambda frame: render_elevator_frame(env, frame), selected)


def visualize_cpu():
    print_domain(
        "CPU Job Scheduling",
        "# is one waiting job in a queue. The mode is the class that ran most recently.",
    )
    start_state = (2, 1, 2, CPUSchedulingEnv.MODE_NONE)
    specs = [
        ("Shortest Job First", lambda env: cpu_shortest_job_first_policy(env), False),
        ("Longest Job First", lambda env: cpu_longest_job_first_policy(env), False),
        ("Max Queue", lambda env: cpu_max_queue_policy(env), False),
        ("Sticky", lambda env: cpu_sticky_policy(env), False),
    ]
    for idx, (name, policy_factory, selected) in enumerate(specs):
        env = CPUSchedulingEnv(max_queue=2, arrival_probs=(0.35, 0.25, 0.15), seed=300 + idx)
        frames = simulate(env, policy_factory(env), start_state)
        print_policy_run(name, frames, lambda frame: render_cpu_frame(env, frame), selected)

    for name, env, policy, selected in train_learning_policies(
        lambda seed: CPUSchedulingEnv(max_queue=2, arrival_probs=(0.35, 0.25, 0.15), seed=seed),
        train_episodes=CPU_VIS_TRAIN_EPISODES,
        max_steps=100,
        seed_base=310,
    ):
        frames = simulate(env, policy, start_state)
        print_policy_run(name, frames, lambda frame: render_cpu_frame(env, frame), selected)


def main():
    print("Scheduling Policy CLI Visualization", flush=True)
    print(f"Showing {STEPS} sampled steps per policy from the same domain-specific start state.", flush=True)
    print("Q-learning is the selected method.", flush=True)
    visualize_disk()
    visualize_elevator()
    visualize_cpu()


if __name__ == "__main__":
    main()
