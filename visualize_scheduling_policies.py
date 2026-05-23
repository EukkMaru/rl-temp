"""Text view of CPU scheduling rollout traces."""

from __future__ import annotations

import random

from control_agents import MonteCarloControlAgent, QLearningAgent, SARSAAgent
from cpu_scheduling_env import CPUSchedulingEnv
from dp_solver import DynamicProgrammingSolver
from rl_utils import greedy_policy_from_Q
from scheduling_policies import (
    cpu_longest_job_first_policy,
    cpu_max_queue_policy,
    cpu_shortest_job_first_policy,
    cpu_sticky_policy,
)


STEPS = 12
VIS_TRAIN_EPISODES = 2500


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
            }
        )
        state = next_state
        if done:
            break

    return frames


def render_cpu_frame(frame):
    s, m, l, mode = frame["state"]
    mode_names = {0: "idle", 1: "short", 2: "medium", 3: "long"}
    return (
        f"t={frame['t']:02d} | short:{'#' * s or '-':<3s} "
        f"medium:{'#' * m or '-':<3s} long:{'#' * l or '-':<3s} "
        f"| mode={mode_names[mode]:<6s} action={frame['action']:<10s} "
        f"reward={frame['reward']:6.2f}"
    )


def print_policy_run(name, frames, selected=False):
    label = f"{name} [SELECTED]" if selected else name
    total = frames[-1]["total_reward"] if frames else 0.0
    print(f"\n  {label}")
    print(f"  {'-' * len(label)}")
    for frame in frames:
        print("  " + render_cpu_frame(frame))
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


def main():
    print("CPU Scheduling CLI Visualization")
    print("# is one waiting job. Q-learning is the selected method.")

    start_state = (2, 1, 2, CPUSchedulingEnv.MODE_NONE)
    env_kwargs = {"max_queue": 2, "arrival_probs": (0.35, 0.25, 0.15)}

    heuristic_specs = [
        ("Shortest Job First", cpu_shortest_job_first_policy),
        ("Longest Job First", cpu_longest_job_first_policy),
        ("Max Queue", cpu_max_queue_policy),
        ("Sticky", cpu_sticky_policy),
    ]
    for idx, (name, policy_factory) in enumerate(heuristic_specs):
        env = CPUSchedulingEnv(**env_kwargs, seed=300 + idx)
        frames = simulate(env, policy_factory(env), start_state)
        print_policy_run(name, frames)

    for name, env, policy, selected in train_learning_policies(
        lambda seed: CPUSchedulingEnv(**env_kwargs, seed=seed),
        train_episodes=VIS_TRAIN_EPISODES,
        max_steps=100,
        seed_base=310,
    ):
        frames = simulate(env, policy, start_state)
        print_policy_run(name, frames, selected)


if __name__ == "__main__":
    main()
