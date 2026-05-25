"""Policy heatmap: VI vs Q-learning over all 108 states.

Layout: 4 rows (mode = none/short/medium/long), 27 columns
(one per (q_S, q_M, q_L) combination, lex-ordered).

Each cell colored by the greedy action:
    run_short  -> blue
    run_medium -> green
    run_long   -> orange
    idle       -> light grey

Also computes the action-agreement rate between Q-learning and VI,
and a disagreement-overlay panel that marks the cells where they
choose different actions.

Outputs:
    submission_assets/policy_heatmap.png
    submission_assets/policy_heatmap.svg
    submission_assets/policy_agreement.txt
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from control_agents import QLearningAgent
from cpu_scheduling_env import CPUSchedulingEnv
from dp_solver import DynamicProgrammingSolver
from rl_utils import greedy_action_from_Q, greedy_policy_from_Q

OUT_DIR = Path("submission_assets")
MAX_STEPS = 100
TRAIN_EPISODES = 50_000

ACTION_COLOR = {
    "run_short": "#3a78b5",
    "run_medium": "#3aa54a",
    "run_long": "#e08a2b",
    "idle": "#cccccc",
}
ACTION_SYMBOL = {"run_short": "S", "run_medium": "M", "run_long": "L", "idle": "·"}
ACTIONS = ["run_short", "run_medium", "run_long", "idle"]
MODE_NAMES = {0: "none", 1: "short", 2: "medium", 3: "long"}


def policy_action(policy_or_Q, env, state):
    if isinstance(policy_or_Q, dict) and state in policy_or_Q and isinstance(policy_or_Q[state], dict):
        ps = policy_or_Q[state]
        # Either a Q-table dict or a probability policy
        if set(ps.keys()) <= set(env.actions) and abs(sum(ps.values()) - 1.0) < 1e-6:
            return max(ps, key=ps.get)
    return greedy_action_from_Q(env, policy_or_Q, state)


def state_grid(env):
    """Return list of (q_S, q_M, q_L) tuples in lex order (length 27)."""
    triples = []
    for qs in range(env.max_queue + 1):
        for qm in range(env.max_queue + 1):
            for ql in range(env.max_queue + 1):
                triples.append((qs, qm, ql))
    return triples


def build_policy_matrix(env, policy):
    triples = state_grid(env)
    modes = [0, 1, 2, 3]
    rows = []
    for mode in modes:
        row = []
        for (qs, qm, ql) in triples:
            state = (qs, qm, ql, mode)
            row.append(policy[state])
        rows.append(row)
    return rows, triples, modes


def policy_action_from_probs(probs):
    return max(probs, key=probs.get)


def draw_heatmap(ax, env, policy_rows, triples, modes, title):
    n_cols = len(triples)
    n_rows = len(modes)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    for r, mode in enumerate(modes):
        for c, _trip in enumerate(triples):
            probs = policy_rows[r][c]
            action = policy_action_from_probs(probs)
            ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                       facecolor=ACTION_COLOR[action],
                                       edgecolor="white", linewidth=0.5))
            ax.text(c + 0.5, r + 0.5, ACTION_SYMBOL[action],
                    ha="center", va="center", fontsize=7,
                    color="white" if action != "idle" else "#333333")
    ax.set_yticks([m + 0.5 for m in range(n_rows)])
    ax.set_yticklabels([f"mode={MODE_NAMES[m]}" for m in modes], fontsize=9)
    ax.set_xticks([])
    ax.set_title(title, fontsize=11)
    # Vertical separators every 9 cells (each new q_S block)
    for x in range(9, n_cols, 9):
        ax.axvline(x, color="#444", linewidth=0.8)


def draw_disagreement(ax, env, vi_rows, q_rows, triples, modes):
    n_cols = len(triples)
    n_rows = len(modes)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    n_disagree = 0
    for r in range(n_rows):
        for c in range(n_cols):
            a_vi = policy_action_from_probs(vi_rows[r][c])
            a_q = policy_action_from_probs(q_rows[r][c])
            if a_vi == a_q:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor="#f3f3f3",
                                           edgecolor="white", linewidth=0.5))
            else:
                n_disagree += 1
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor="#d33",
                                           edgecolor="white", linewidth=0.5))
                ax.text(c + 0.5, r + 0.5,
                        f"{ACTION_SYMBOL[a_vi]}→{ACTION_SYMBOL[a_q]}",
                        ha="center", va="center", fontsize=6, color="white")
    ax.set_yticks([m + 0.5 for m in range(n_rows)])
    ax.set_yticklabels([f"mode={MODE_NAMES[m]}" for m in modes], fontsize=9)
    ax.set_xticks([])
    total = n_cols * n_rows
    ax.set_title(f"(c) Disagreement (VI → Q-learning)   "
                 f"{n_disagree}/{total} cells differ "
                 f"({(total - n_disagree) / total * 100:.1f}% agree)", fontsize=11)
    return n_disagree


def main():
    env = CPUSchedulingEnv(max_queue=2, arrival_probs=(0.35, 0.25, 0.15), seed=2)
    gamma, alpha, epsilon = 0.95, 0.12, 0.08

    print("Solving Value Iteration...")
    vi_policy, _ = DynamicProgrammingSolver(env, gamma=gamma, theta=1e-4).value_iteration()

    print(f"Training Q-learning ({TRAIN_EPISODES} episodes)...")
    random.seed(0); env.rng.seed(0)
    qa = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon, max_steps=MAX_STEPS)
    qa.train(num_episodes=TRAIN_EPISODES, start_state=env.start_state)
    q_policy = greedy_policy_from_Q(env, qa.Q)

    vi_rows, triples, modes = build_policy_matrix(env, vi_policy)
    q_rows, _, _ = build_policy_matrix(env, q_policy)

    fig, axes = plt.subplots(3, 1, figsize=(13, 6.0))
    draw_heatmap(axes[0], env, vi_rows, triples, modes,
                 "(a) Value Iteration policy")
    draw_heatmap(axes[1], env, q_rows, triples, modes,
                 "(b) Q-learning policy (50k episodes)")
    n_disagree = draw_disagreement(axes[2], env, vi_rows, q_rows, triples, modes)

    # Legend
    handles = [mpatches.Patch(color=ACTION_COLOR[a], label=a) for a in ACTIONS]
    fig.legend(handles=handles, ncol=4, loc="upper center", fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.text(0.5, 0.005,
             "columns = 27 queue states (q_S, q_M, q_L), lex-ordered  ·  "
             "vertical separators mark q_S boundaries  ·  cell symbol = greedy action",
             ha="center", fontsize=8, color="#555")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "policy_heatmap.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "policy_heatmap.svg", bbox_inches="tight")
    plt.close(fig)

    total = len(triples) * len(modes)
    agreement = (total - n_disagree) / total
    txt = (
        f"Policy agreement between Value Iteration and Q-learning\n"
        f"  total states           : {total}\n"
        f"  agreeing states        : {total - n_disagree}\n"
        f"  disagreeing states     : {n_disagree}\n"
        f"  action-agreement rate  : {agreement:.4f}  ({agreement * 100:.2f}%)\n"
    )
    (OUT_DIR / "policy_agreement.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"Wrote {OUT_DIR / 'policy_heatmap.png'} / .svg")


if __name__ == "__main__":
    main()
