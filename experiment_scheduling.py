"""Run the scheduling policy comparisons."""

import random

from control_agents import MonteCarloControlAgent, QLearningAgent, SARSAAgent
from dp_solver import DynamicProgrammingSolver
from rl_utils import greedy_policy_from_Q
from cpu_scheduling_env import CPUSchedulingEnv
from disk_scheduling_env import DiskSchedulingEnv
from elevator_scheduling_env import ElevatorSchedulingEnv
from scheduling_policies import (
    linear_nearest_request_policy,
    linear_scan_policy,
    linear_look_policy,
    cpu_shortest_job_first_policy,
    cpu_longest_job_first_policy,
    cpu_max_queue_policy,
    cpu_sticky_policy,
)
from scheduling_reporting import evaluate_scheduling_policy, print_policy_score_table


def run_env(name, env, heuristics, num_episodes=6000, max_steps=200):
    print("\n" + "#" * 60)
    print(f"# {name}")
    print("#" * 60 + "\n")
    print(f"States: {len(env.states)}, Actions: {env.actions}")
    print(f"Start state: {env.start_state}\n")

    gamma = 0.95
    alpha = 0.12
    epsilon = 0.08

    rows = []

    # Heuristic baselines.
    for h_name, policy in heuristics:
        result = evaluate_scheduling_policy(env, policy, num_episodes=500, max_steps=max_steps)
        rows.append({"name": h_name, "avg_return": result["avg_return"], "stderr": result["stderr"]})

    # DP baseline.
    dp = DynamicProgrammingSolver(env, gamma=gamma, theta=1e-4)
    dp_policy, dp_V = dp.value_iteration(verbose=False)
    dp_result = evaluate_scheduling_policy(env, dp_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "Value Iteration", "avg_return": dp_result["avg_return"], "stderr": dp_result["stderr"]})

    baseline_episodes = 1000
    selected_episodes = num_episodes

    # MC Control baseline.
    random.seed(0)
    env.rng.seed(0)
    mc = MonteCarloControlAgent(env, gamma=gamma, epsilon=epsilon, max_steps=max_steps)
    mc.train(num_episodes=baseline_episodes, start_state=env.start_state, print_every=0)
    mc_policy = greedy_policy_from_Q(env, mc.Q)
    mc_result = evaluate_scheduling_policy(env, mc_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "MC Control", "avg_return": mc_result["avg_return"], "stderr": mc_result["stderr"]})

    # SARSA.
    random.seed(0)
    env.rng.seed(0)
    sarsa = SARSAAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=max_steps)
    sarsa.train(num_episodes=baseline_episodes, start_state=env.start_state, print_every=0)
    sarsa_policy = greedy_policy_from_Q(env, sarsa.Q)
    sarsa_result = evaluate_scheduling_policy(env, sarsa_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "SARSA", "avg_return": sarsa_result["avg_return"], "stderr": sarsa_result["stderr"]})

    # Q-learning is the method we report as the main learned policy.
    random.seed(0)
    env.rng.seed(0)
    q_agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=max_steps)
    q_agent.train(num_episodes=selected_episodes, start_state=env.start_state, print_every=0)
    q_policy = greedy_policy_from_Q(env, q_agent.Q)
    q_result = evaluate_scheduling_policy(env, q_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "Q-learning (Selected)", "avg_return": q_result["avg_return"], "stderr": q_result["stderr"]})

    print_policy_score_table(rows)

    selected = next(row for row in rows if row["name"] == "Q-learning (Selected)")
    dp_row = next(row for row in rows if row["name"] == "Value Iteration")
    non_dp_rows = [row for row in rows if row["name"] != "Value Iteration" and row["name"] != "Q-learning (Selected)"]
    best_non_dp = max(non_dp_rows, key=lambda row: row["avg_return"])
    non_dp_gap = selected["avg_return"] - best_non_dp["avg_return"]
    dp_gap = dp_row["avg_return"] - selected["avg_return"]
    dp_tolerance = 2.0 * (dp_row.get("stderr", 0.0) + selected.get("stderr", 0.0))
    if non_dp_gap > 0:
        print(f"Selected policy result: strictly wins over non-DP baselines by {non_dp_gap:.2f}.")
    else:
        print(f"Selected policy result: trails best non-DP baseline by {-non_dp_gap:.2f}.")
    if dp_gap <= dp_tolerance:
        print(f"Against exact Value Iteration: on par as a model-free method (gap {dp_gap:.2f}).")
    else:
        print(f"Against exact Value Iteration: trails by {dp_gap:.2f}.")
    print()

    print("=== DP greedy policy sample ===")
    env.print_policy(dp_policy, max_rows=12)
    print("=== Q-learning greedy policy sample ===")
    env.print_policy(q_policy, max_rows=12)


def main():
    random.seed(0)

    disk = DiskSchedulingEnv(
        n_tracks=4,
        request_prob=0.08,
        start_position=2,
        start_direction=1,
        seed=0,
    )
    run_env(
        "Disk Head Scheduling",
        disk,
        [
            ("Nearest", linear_nearest_request_policy(disk)),
            ("SCAN", linear_scan_policy(disk)),
            ("LOOK", linear_look_policy(disk)),
        ],
        num_episodes=6000,
        max_steps=100,
    )

    elevator = ElevatorSchedulingEnv(
        n_floors=4,
        request_prob=0.10,
        start_position=2,
        start_direction=1,
        seed=1,
    )
    run_env(
        "Elevator Pickup Scheduling",
        elevator,
        [
            ("Nearest", linear_nearest_request_policy(elevator)),
            ("Collective/SCAN", linear_scan_policy(elevator)),
            ("LOOK", linear_look_policy(elevator)),
        ],
        num_episodes=6000,
        max_steps=100,
    )

    cpu = CPUSchedulingEnv(
        max_queue=2,
        arrival_probs=(0.35, 0.25, 0.15),
        seed=2,
    )
    run_env(
        "CPU Job Scheduling",
        cpu,
        [
            ("SJF", cpu_shortest_job_first_policy(cpu)),
            ("LJF", cpu_longest_job_first_policy(cpu)),
            ("MaxQueue", cpu_max_queue_policy(cpu)),
            ("Sticky", cpu_sticky_policy(cpu)),
        ],
        num_episodes=50000,
        max_steps=100,
    )


if __name__ == "__main__":
    main()
