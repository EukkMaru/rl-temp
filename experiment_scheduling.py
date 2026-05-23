"""Run the CPU scheduling policy comparison."""

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
from scheduling_reporting import evaluate_scheduling_policy, print_policy_score_table


def run_cpu_experiment(num_episodes=50000, max_steps=100):
    env = CPUSchedulingEnv(
        max_queue=2,
        arrival_probs=(0.35, 0.25, 0.15),
        seed=2,
    )

    print("\n" + "#" * 60)
    print("# CPU Job Scheduling")
    print("#" * 60 + "\n")
    print(f"States: {len(env.states)}, Actions: {env.actions}")
    print(f"Start state: {env.start_state}\n")

    gamma = 0.95
    alpha = 0.12
    epsilon = 0.08

    rows = []
    heuristics = [
        ("SJF", cpu_shortest_job_first_policy(env)),
        ("LJF", cpu_longest_job_first_policy(env)),
        ("MaxQueue", cpu_max_queue_policy(env)),
        ("Sticky", cpu_sticky_policy(env)),
    ]

    for name, policy in heuristics:
        result = evaluate_scheduling_policy(env, policy, num_episodes=500, max_steps=max_steps)
        rows.append({"name": name, "avg_return": result["avg_return"], "stderr": result["stderr"]})

    dp = DynamicProgrammingSolver(env, gamma=gamma, theta=1e-4)
    dp_policy, _dp_V = dp.value_iteration(verbose=False)
    dp_result = evaluate_scheduling_policy(env, dp_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "Value Iteration", "avg_return": dp_result["avg_return"], "stderr": dp_result["stderr"]})

    random.seed(0)
    env.rng.seed(0)
    mc = MonteCarloControlAgent(env, gamma=gamma, epsilon=epsilon, max_steps=max_steps)
    mc.train(num_episodes=num_episodes, start_state=env.start_state)
    mc_policy = greedy_policy_from_Q(env, mc.Q)
    mc_result = evaluate_scheduling_policy(env, mc_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "MC Control", "avg_return": mc_result["avg_return"], "stderr": mc_result["stderr"]})

    random.seed(0)
    env.rng.seed(0)
    sarsa = SARSAAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=max_steps)
    sarsa.train(num_episodes=num_episodes, start_state=env.start_state)
    sarsa_policy = greedy_policy_from_Q(env, sarsa.Q)
    sarsa_result = evaluate_scheduling_policy(env, sarsa_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "SARSA", "avg_return": sarsa_result["avg_return"], "stderr": sarsa_result["stderr"]})

    random.seed(0)
    env.rng.seed(0)
    q_agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=max_steps)
    q_agent.train(num_episodes=num_episodes, start_state=env.start_state)
    q_policy = greedy_policy_from_Q(env, q_agent.Q)
    q_result = evaluate_scheduling_policy(env, q_policy, num_episodes=500, max_steps=max_steps)
    rows.append({"name": "Q-learning (Selected)", "avg_return": q_result["avg_return"], "stderr": q_result["stderr"]})

    print_policy_score_table(rows)

    selected = next(row for row in rows if row["name"] == "Q-learning (Selected)")
    dp_row = next(row for row in rows if row["name"] == "Value Iteration")
    non_dp_rows = [row for row in rows if row["name"] not in ("Value Iteration", "Q-learning (Selected)")]
    best_non_dp = max(non_dp_rows, key=lambda row: row["avg_return"])
    print(f"Selected policy beats best non-DP baseline by {selected['avg_return'] - best_non_dp['avg_return']:.2f}.")
    print(f"Gap to Value Iteration: {dp_row['avg_return'] - selected['avg_return']:.2f}.\n")

    print("=== Value Iteration policy sample ===")
    env.print_policy(dp_policy, max_rows=12)
    print("=== Q-learning policy sample ===")
    env.print_policy(q_policy, max_rows=12)


def main():
    run_cpu_experiment()


if __name__ == "__main__":
    main()
