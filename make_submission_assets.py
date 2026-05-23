"""Build result tables and charts for the CPU scheduling experiment."""

from __future__ import annotations

import csv
import random
from pathlib import Path

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
from scheduling_reporting import evaluate_scheduling_policy


OUT_DIR = Path("submission_assets")
MAX_STEPS = 100
EVAL_EPISODES = 500
TRAIN_EPISODES = 50_000


def compute_results():
    env = CPUSchedulingEnv(max_queue=2, arrival_probs=(0.35, 0.25, 0.15), seed=2)
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
        rows.append(evaluate_policy(env, name, policy))

    dp_policy, _dp_v = DynamicProgrammingSolver(env, gamma=gamma, theta=1e-4).value_iteration()
    rows.append(evaluate_policy(env, "Value Iteration", dp_policy))

    random.seed(0)
    env.rng.seed(0)
    mc = MonteCarloControlAgent(env, gamma=gamma, epsilon=epsilon, max_steps=MAX_STEPS)
    mc.train(num_episodes=TRAIN_EPISODES, start_state=env.start_state)
    rows.append(evaluate_policy(env, "MC Control", greedy_policy_from_Q(env, mc.Q)))

    random.seed(0)
    env.rng.seed(0)
    sarsa = SARSAAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=MAX_STEPS)
    sarsa.train(num_episodes=TRAIN_EPISODES, start_state=env.start_state)
    rows.append(evaluate_policy(env, "SARSA", greedy_policy_from_Q(env, sarsa.Q)))

    random.seed(0)
    env.rng.seed(0)
    q_agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=MAX_STEPS)
    q_agent.train(num_episodes=TRAIN_EPISODES, start_state=env.start_state)
    rows.append(evaluate_policy(env, "Q-learning (Selected)", greedy_policy_from_Q(env, q_agent.Q)))
    return rows


def evaluate_policy(env, method, policy):
    result = evaluate_scheduling_policy(env, policy, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
    return {"method": method, **result}


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "avg_return", "stderr", "run_short", "run_medium", "run_long", "idle"])
        for row in rows:
            action_rate = row["action_rate"]
            writer.writerow(
                [
                    row["method"],
                    f"{row['avg_return']:.4f}",
                    f"{row['stderr']:.4f}",
                    f"{action_rate['run_short']:.4f}",
                    f"{action_rate['run_medium']:.4f}",
                    f"{action_rate['run_long']:.4f}",
                    f"{action_rate['idle']:.4f}",
                ]
            )


def write_svg_bar_chart(rows, path):
    width = 920
    height = 520
    margin_left = 170
    margin_right = 40
    margin_top = 50
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(row["avg_return"] for row in rows) * 1.08
    bar_gap = 14
    bar_height = (plot_height - bar_gap * (len(rows) - 1)) / len(rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="460" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">CPU Scheduling Policy Comparison</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333"/>',
    ]

    for tick in range(0, 121, 20):
        x = margin_left + (tick / max_value) * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{tick}</text>')

    for idx, row in enumerate(rows):
        y = margin_top + idx * (bar_height + bar_gap)
        bar_width = (row["avg_return"] / max_value) * plot_width
        fill = "#2f6f9f" if row["method"] != "Q-learning (Selected)" else "#c7532c"
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + bar_height * 0.66:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="13">{row["method"]}</text>'
        )
        parts.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'fill="{fill}" rx="3"/>'
        )
        parts.append(
            f'<text x="{margin_left + bar_width + 7:.1f}" y="{y + bar_height * 0.66:.1f}" '
            f'font-family="Arial" font-size="13">{row["avg_return"]:.2f}</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle" '
        'font-family="Arial" font-size="13">Average undiscounted return over 500 fixed-seed episodes</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = compute_results()
    write_csv(rows, OUT_DIR / "policy_results.csv")
    write_svg_bar_chart(rows, OUT_DIR / "policy_results.svg")
    print(f"Wrote {OUT_DIR / 'policy_results.csv'}")
    print(f"Wrote {OUT_DIR / 'policy_results.svg'}")


if __name__ == "__main__":
    main()
