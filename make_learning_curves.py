"""Train MC / SARSA / Q-learning under identical 50k-episode budget,
record per-episode returns, and plot moving-average learning curves
plus a 'greedy evaluation along training' curve.

Outputs:
    submission_assets/learning_curves.png  (matplotlib)
    submission_assets/learning_curves.svg  (matplotlib)
    submission_assets/learning_curves.csv  (raw moving-average data)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from control_agents import MonteCarloControlAgent, QLearningAgent, SARSAAgent
from cpu_scheduling_env import CPUSchedulingEnv
from dp_solver import DynamicProgrammingSolver
from rl_utils import greedy_policy_from_Q
from scheduling_reporting import evaluate_scheduling_policy

OUT_DIR = Path("submission_assets")
TRAIN_EPISODES = 50_000
MAX_STEPS = 100
EVAL_EPISODES = 200          # used for periodic greedy eval
EVAL_EVERY = 2_500           # eval every N training episodes
WINDOW = 1_000               # moving average window for episode_returns


def moving_average(x, window):
    if len(x) < window:
        return np.array(x, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (csum[window:] - csum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, ma])


def train_with_eval_checkpoints(agent, env, num_episodes, eval_every, eval_episodes):
    """Train agent in small chunks; periodically evaluate the greedy policy."""
    checkpoints = []
    done = 0
    while done < num_episodes:
        chunk = min(eval_every, num_episodes - done)
        agent.train(num_episodes=chunk, start_state=env.start_state)
        done += chunk
        greedy = greedy_policy_from_Q(env, agent.Q)
        result = evaluate_scheduling_policy(
            env, greedy, num_episodes=eval_episodes, max_steps=MAX_STEPS, seed=99999
        )
        checkpoints.append((done, result["avg_return"]))
    return checkpoints


def run():
    env = CPUSchedulingEnv(max_queue=2, arrival_probs=(0.35, 0.25, 0.15), seed=2)
    gamma, alpha, epsilon = 0.95, 0.12, 0.08

    dp_policy, _ = DynamicProgrammingSolver(env, gamma=gamma, theta=1e-4).value_iteration()
    dp_eval = evaluate_scheduling_policy(env, dp_policy, num_episodes=EVAL_EPISODES,
                                         max_steps=MAX_STEPS, seed=99999)
    vi_baseline = dp_eval["avg_return"]
    print(f"Value Iteration baseline (greedy eval): {vi_baseline:.3f}")

    agents = []

    print("Training MC Control...")
    random.seed(0); env.rng.seed(0)
    mc = MonteCarloControlAgent(env, gamma=gamma, epsilon=epsilon, max_steps=MAX_STEPS)
    mc_ckpts = train_with_eval_checkpoints(mc, env, TRAIN_EPISODES, EVAL_EVERY, EVAL_EPISODES)
    agents.append(("MC Control", mc.episode_returns, mc_ckpts, "#888888"))

    print("Training SARSA...")
    random.seed(0); env.rng.seed(0)
    sarsa = SARSAAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=MAX_STEPS)
    sarsa_ckpts = train_with_eval_checkpoints(sarsa, env, TRAIN_EPISODES, EVAL_EVERY, EVAL_EPISODES)
    agents.append(("SARSA", sarsa.episode_returns, sarsa_ckpts, "#2f6f9f"))

    print("Training Q-learning...")
    random.seed(0); env.rng.seed(0)
    qa = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=MAX_STEPS)
    q_ckpts = train_with_eval_checkpoints(qa, env, TRAIN_EPISODES, EVAL_EVERY, EVAL_EPISODES)
    agents.append(("Q-learning", qa.episode_returns, q_ckpts, "#c7532c"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    for name, ep_returns, _ckpts, color in agents:
        ma = moving_average(np.asarray(ep_returns, dtype=float), WINDOW)
        ax1.plot(np.arange(1, len(ma) + 1), ma, label=name, color=color, linewidth=1.5)
    ax1.axhline(vi_baseline, linestyle="--", color="#202020",
                label=f"Value Iteration ({vi_baseline:.2f})", linewidth=1.2)
    ax1.set_xlabel("training episode")
    ax1.set_ylabel(f"avg return (moving avg, window={WINDOW})")
    ax1.set_title("(a) Behaviour-policy return during training")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    for name, _ep_returns, ckpts, color in agents:
        xs = [c[0] for c in ckpts]
        ys = [c[1] for c in ckpts]
        ax2.plot(xs, ys, marker="o", markersize=4, label=name, color=color, linewidth=1.5)
    ax2.axhline(vi_baseline, linestyle="--", color="#202020",
                label=f"Value Iteration ({vi_baseline:.2f})", linewidth=1.2)
    ax2.set_xlabel("training episode")
    ax2.set_ylabel("greedy policy avg return (200 eps)")
    ax2.set_title("(b) Greedy-policy return at checkpoints")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Learning curves under identical 50,000-episode budget", fontsize=13)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "learning_curves.png", dpi=140)
    fig.savefig(OUT_DIR / "learning_curves.svg")
    plt.close(fig)

    with (OUT_DIR / "learning_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "checkpoint_episode", "greedy_eval_return"])
        for name, _ep, ckpts, _c in agents:
            for ep, ret in ckpts:
                w.writerow([name, ep, f"{ret:.4f}"])

    print(f"Wrote {OUT_DIR / 'learning_curves.png'} / .svg / .csv")


if __name__ == "__main__":
    run()
