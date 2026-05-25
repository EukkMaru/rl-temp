"""Ablation studies for the CPU scheduling MDP.

Four ablations:
    1. switch_cost  ∈ {0.0, 0.3, 1.0}
    2. State without 'mode' (Markov is broken — partial observation)
    3. Arrival distribution shift  (short-heavy vs. long-heavy)
    4. Hyperparameter grid (α × ε)  at γ = 0.95

For each ablation we re-train Q-learning under the *same* 50k budget
and compare to the appropriate baseline (Value Iteration on the modified
env, plus a few heuristics for context).

Outputs:
    submission_assets/ablations.csv
    submission_assets/ablation_switch_cost.png
    submission_assets/ablation_arrival.png
    submission_assets/ablation_no_mode.png
    submission_assets/ablation_hparam_grid.png
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from control_agents import QLearningAgent
from cpu_scheduling_env import CPUSchedulingEnv
from dp_solver import DynamicProgrammingSolver
from rl_utils import greedy_action_from_Q, greedy_policy_from_Q, make_Q
from scheduling_policies import (
    cpu_longest_job_first_policy,
    cpu_max_queue_policy,
    cpu_shortest_job_first_policy,
    cpu_sticky_policy,
)
from scheduling_reporting import evaluate_scheduling_policy

OUT_DIR = Path("submission_assets")
TRAIN_EPISODES = 50_000
MAX_STEPS = 100
EVAL_EPISODES = 500
EVAL_SEED = 12345

GAMMA, ALPHA, EPSILON = 0.95, 0.12, 0.08


def fresh_env(switch_cost=0.30, arrival_probs=(0.35, 0.25, 0.15)):
    return CPUSchedulingEnv(
        max_queue=2,
        arrival_probs=arrival_probs,
        switch_cost=switch_cost,
        seed=2,
    )


def train_qlearning(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON,
                    num_episodes=TRAIN_EPISODES):
    random.seed(0)
    env.rng.seed(0)
    agent = QLearningAgent(env, gamma=gamma, alpha=alpha,
                           epsilon=epsilon, max_steps=MAX_STEPS)
    agent.train(num_episodes=num_episodes, start_state=env.start_state)
    return greedy_policy_from_Q(env, agent.Q), agent


def solve_vi(env, gamma=GAMMA):
    return DynamicProgrammingSolver(env, gamma=gamma, theta=1e-4).value_iteration()[0]


def evaluate(env, policy):
    return evaluate_scheduling_policy(env, policy, num_episodes=EVAL_EPISODES,
                                      max_steps=MAX_STEPS, seed=EVAL_SEED)


# ---------------------------------------------------------------------------
# Ablation 1: switch_cost sweep
# ---------------------------------------------------------------------------

def ablation_switch_cost():
    print("\n=== Ablation 1: switch_cost ===")
    rows = []
    settings = [0.0, 0.3, 1.0]
    for sc in settings:
        env = fresh_env(switch_cost=sc)
        vi_pol = solve_vi(env)
        q_pol, _ = train_qlearning(env)
        sjf_pol = cpu_shortest_job_first_policy(env)
        sticky_pol = cpu_sticky_policy(env)

        vi_eval = evaluate(env, vi_pol)
        q_eval = evaluate(env, q_pol)
        sjf_eval = evaluate(env, sjf_pol)
        sticky_eval = evaluate(env, sticky_pol)

        rows.append({
            "switch_cost": sc,
            "vi": vi_eval["avg_return"], "vi_err": vi_eval["stderr"],
            "q": q_eval["avg_return"], "q_err": q_eval["stderr"],
            "sjf": sjf_eval["avg_return"], "sjf_err": sjf_eval["stderr"],
            "sticky": sticky_eval["avg_return"], "sticky_err": sticky_eval["stderr"],
            "q_sticky_share": _sticky_share(env, q_pol),
            "vi_sticky_share": _sticky_share(env, vi_pol),
        })
        print(f"  sc={sc:.2f}: VI={vi_eval['avg_return']:.2f}, "
              f"Q={q_eval['avg_return']:.2f}, "
              f"SJF={sjf_eval['avg_return']:.2f}, "
              f"Sticky={sticky_eval['avg_return']:.2f}")

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"switch={r['switch_cost']:.1f}" for r in rows]
    x = np.arange(len(labels))
    w = 0.2
    methods = [
        ("SJF", "sjf", "#888"),
        ("Sticky", "sticky", "#9b59b6"),
        ("VI (oracle)", "vi", "#2f6f9f"),
        ("Q-learning", "q", "#c7532c"),
    ]
    for i, (name, key, color) in enumerate(methods):
        vals = [r[key] for r in rows]
        errs = [r[f"{key}_err"] for r in rows]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, label=name, color=color, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("avg return (500 eps)")
    ax.set_title("Ablation 1: effect of switch_cost on policy ranking")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(0, min(r["sjf"] for r in rows)) - 5, max(r["vi"] for r in rows) + 10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_switch_cost.png", dpi=140)
    fig.savefig(OUT_DIR / "ablation_switch_cost.svg")
    plt.close(fig)
    return rows


def _sticky_share(env, policy):
    """Fraction of (state) cells where the greedy action equals the previous mode action."""
    mode_action = {1: "run_short", 2: "run_medium", 3: "run_long"}
    n_with_mode = 0
    n_sticky = 0
    for state in env.states:
        _qs, _qm, _ql, mode = state
        if mode == 0:
            continue
        n_with_mode += 1
        a = max(policy[state], key=policy[state].get)
        if a == mode_action[mode]:
            n_sticky += 1
    return n_sticky / n_with_mode if n_with_mode else 0.0


# ---------------------------------------------------------------------------
# Ablation 2: state w/o mode  (partial observation)
# ---------------------------------------------------------------------------

def _mask_mode(state):
    return (state[0], state[1], state[2], 0)


def train_qlearning_no_mode(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON,
                            num_episodes=TRAIN_EPISODES):
    """Train Q-learning whose Q-table is keyed only by (q_S, q_M, q_L);
    the environment is unchanged (mode still affects reward/transitions)."""
    random.seed(0)
    env.rng.seed(0)
    Q = make_Q(env)
    # We will key Q by masked state; build a separate dict
    Q_masked = defaultdict(lambda: {a: 0.0 for a in env.actions})

    def greedy_a(masked_state):
        vals = Q_masked[masked_state]
        return max(env.actions, key=lambda a: (vals[a], -env.actions.index(a)))

    def eps_greedy_a(masked_state):
        if random.random() < epsilon:
            return random.choice(env.actions)
        return greedy_a(masked_state)

    for _ in range(num_episodes):
        state = env.start_state
        for _step in range(MAX_STEPS):
            masked = _mask_mode(state)
            action = eps_greedy_a(masked)
            next_state, reward, done = env.step(state, action)
            next_masked = _mask_mode(next_state)
            best_next = max(Q_masked[next_masked].values())
            target = reward if done else reward + gamma * best_next
            Q_masked[masked][action] += alpha * (target - Q_masked[masked][action])
            state = next_state
            if done:
                break

    # Lift to a 'true-state' policy that always uses the masked Q
    policy = {}
    for state in env.states:
        m = _mask_mode(state)
        a = max(env.actions, key=lambda act: (Q_masked[m][act], -env.actions.index(act)))
        policy[state] = {act: 0.0 for act in env.actions}
        policy[state][a] = 1.0
    return policy


def ablation_no_mode():
    print("\n=== Ablation 2: state without mode ===")
    env = fresh_env()
    vi_pol = solve_vi(env)
    full_q_pol, _ = train_qlearning(env)
    masked_q_pol = train_qlearning_no_mode(env)

    vi_eval = evaluate(env, vi_pol)
    full_eval = evaluate(env, full_q_pol)
    masked_eval = evaluate(env, masked_q_pol)

    print(f"  VI                : {vi_eval['avg_return']:.2f}")
    print(f"  Q-learning (full) : {full_eval['avg_return']:.2f}")
    print(f"  Q-learning (no-mode): {masked_eval['avg_return']:.2f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    methods = ["VI\n(oracle)", "Q-learning\n(full state)", "Q-learning\n(no mode)"]
    vals = [vi_eval["avg_return"], full_eval["avg_return"], masked_eval["avg_return"]]
    errs = [vi_eval["stderr"], full_eval["stderr"], masked_eval["stderr"]]
    colors = ["#2f6f9f", "#c7532c", "#888"]
    ax.bar(methods, vals, yerr=errs, color=colors, capsize=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("avg return (500 eps)")
    ax.set_title("Ablation 2: removing 'mode' breaks Markov ⇒ performance drops")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(vals) - 5, max(vals) + 5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_no_mode.png", dpi=140)
    fig.savefig(OUT_DIR / "ablation_no_mode.svg")
    plt.close(fig)

    return {
        "vi": vi_eval["avg_return"], "vi_err": vi_eval["stderr"],
        "q_full": full_eval["avg_return"], "q_full_err": full_eval["stderr"],
        "q_no_mode": masked_eval["avg_return"], "q_no_mode_err": masked_eval["stderr"],
    }


# ---------------------------------------------------------------------------
# Ablation 3: arrival distribution shift
# ---------------------------------------------------------------------------

def ablation_arrival():
    print("\n=== Ablation 3: arrival distribution shift ===")
    settings = [
        ("baseline", (0.35, 0.25, 0.15)),
        ("short-heavy", (0.5, 0.2, 0.1)),
        ("long-heavy", (0.1, 0.2, 0.5)),
    ]
    rows = []
    for name, probs in settings:
        env = fresh_env(arrival_probs=probs)
        vi_pol = solve_vi(env)
        q_pol, _ = train_qlearning(env)
        sjf_pol = cpu_shortest_job_first_policy(env)
        ljf_pol = cpu_longest_job_first_policy(env)
        mq_pol = cpu_max_queue_policy(env)

        rows.append({
            "regime": name,
            "probs": probs,
            "vi": evaluate(env, vi_pol)["avg_return"],
            "q": evaluate(env, q_pol)["avg_return"],
            "sjf": evaluate(env, sjf_pol)["avg_return"],
            "ljf": evaluate(env, ljf_pol)["avg_return"],
            "mq": evaluate(env, mq_pol)["avg_return"],
        })
        print(f"  {name:>12s} ({probs}):  "
              f"VI={rows[-1]['vi']:.2f}  Q={rows[-1]['q']:.2f}  "
              f"SJF={rows[-1]['sjf']:.2f}  LJF={rows[-1]['ljf']:.2f}  "
              f"MaxQ={rows[-1]['mq']:.2f}")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    methods = [("SJF", "sjf", "#888"),
               ("LJF", "ljf", "#5da55d"),
               ("MaxQueue", "mq", "#9b59b6"),
               ("VI", "vi", "#2f6f9f"),
               ("Q-learning", "q", "#c7532c")]
    x = np.arange(len(rows))
    w = 0.16
    for i, (name, key, color) in enumerate(methods):
        ax.bar(x + (i - 2) * w, [r[key] for r in rows], w,
               label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['regime']}\n{r['probs']}" for r in rows], fontsize=9)
    ax.set_ylabel("avg return (500 eps)")
    ax.set_title("Ablation 3: heuristics specialise; RL adapts to the arrival regime")
    ax.legend(loc="lower right", fontsize=9, ncol=5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_arrival.png", dpi=140)
    fig.savefig(OUT_DIR / "ablation_arrival.svg")
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# Ablation 4: hyperparameter grid α × ε  (γ = 0.95 fixed)
# ---------------------------------------------------------------------------

def ablation_hparam_grid():
    print("\n=== Ablation 4: hyperparameter grid (α × ε) ===")
    alphas = [0.05, 0.12, 0.30]
    epsilons = [0.02, 0.08, 0.20]
    grid = np.zeros((len(alphas), len(epsilons)))

    for i, a in enumerate(alphas):
        for j, e in enumerate(epsilons):
            env = fresh_env()
            random.seed(0); env.rng.seed(0)
            agent = QLearningAgent(env, gamma=GAMMA, alpha=a,
                                   epsilon=e, max_steps=MAX_STEPS)
            agent.train(num_episodes=TRAIN_EPISODES, start_state=env.start_state)
            pol = greedy_policy_from_Q(env, agent.Q)
            ret = evaluate(env, pol)["avg_return"]
            grid[i, j] = ret
            print(f"  α={a:.2f}, ε={e:.2f}: return={ret:.2f}")

    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    vmin, vmax = grid.min(), grid.max()
    im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"ε={e}" for e in epsilons])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"α={a}" for a in alphas])
    for i in range(len(alphas)):
        for j in range(len(epsilons)):
            color = "white" if grid[i, j] < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=10)
    plt.colorbar(im, ax=ax, label="avg return")
    ax.set_title("Ablation 4: Q-learning sensitivity to α and ε\n(γ=0.95, 50k episodes)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_hparam_grid.png", dpi=140)
    fig.savefig(OUT_DIR / "ablation_hparam_grid.svg")
    plt.close(fig)
    return {"alphas": alphas, "epsilons": epsilons, "grid": grid.tolist()}


# ---------------------------------------------------------------------------
# Collect all
# ---------------------------------------------------------------------------

def write_csv(path, switch_rows, no_mode, arrival_rows, hparam):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# ablation 1: switch_cost"])
        w.writerow(["switch_cost", "VI", "Q-learning", "SJF", "Sticky",
                    "Q_sticky_share", "VI_sticky_share"])
        for r in switch_rows:
            w.writerow([r["switch_cost"], f"{r['vi']:.4f}", f"{r['q']:.4f}",
                        f"{r['sjf']:.4f}", f"{r['sticky']:.4f}",
                        f"{r['q_sticky_share']:.4f}", f"{r['vi_sticky_share']:.4f}"])
        w.writerow([])
        w.writerow(["# ablation 2: state without mode"])
        w.writerow(["VI", "Q-learning (full)", "Q-learning (no mode)"])
        w.writerow([f"{no_mode['vi']:.4f}", f"{no_mode['q_full']:.4f}",
                    f"{no_mode['q_no_mode']:.4f}"])
        w.writerow([])
        w.writerow(["# ablation 3: arrival distribution"])
        w.writerow(["regime", "probs", "VI", "Q-learning", "SJF", "LJF", "MaxQueue"])
        for r in arrival_rows:
            w.writerow([r["regime"], str(r["probs"]),
                        f"{r['vi']:.4f}", f"{r['q']:.4f}", f"{r['sjf']:.4f}",
                        f"{r['ljf']:.4f}", f"{r['mq']:.4f}"])
        w.writerow([])
        w.writerow(["# ablation 4: hyperparameter grid (rows = alpha, cols = epsilon)"])
        w.writerow(["alpha\\epsilon"] + hparam["epsilons"])
        for i, a in enumerate(hparam["alphas"]):
            w.writerow([a] + [f"{v:.4f}" for v in hparam["grid"][i]])


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sc = ablation_switch_cost()
    nm = ablation_no_mode()
    ar = ablation_arrival()
    hp = ablation_hparam_grid()
    write_csv(OUT_DIR / "ablations.csv", sc, nm, ar, hp)
    print(f"\nWrote {OUT_DIR / 'ablations.csv'} and 4 ablation figures.")


if __name__ == "__main__":
    main()
