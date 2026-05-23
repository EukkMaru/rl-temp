# EL5001 Project 01 - CPU Scheduling MDP

## Purpose

This project models CPU job scheduling as a finite Markov Decision Process (MDP). At every time step, the scheduler chooses which job class to run while balancing throughput, queue waiting cost, context-switching cost, and idle cost.

This is a real-world sequential decision problem because a CPU scheduler must repeatedly make decisions under uncertainty. Running a job now changes future queues, while new jobs arrive randomly. Reinforcement learning is appropriate because the best action depends on the current queue state and on long-term effects, not only on immediate completion reward.

The comparison includes fixed CPU scheduling rules, Value Iteration, Monte Carlo Control, SARSA, and Q-learning. We use Q-learning as the selected model-free RL method.

## MDP Formulation

State:

```text
(short_queue, medium_queue, long_queue, current_mode)
```

`current_mode` records the job class that ran most recently:

```text
0 = idle/none, 1 = short, 2 = medium, 3 = long
```

Actions:

```text
run_short, run_medium, run_long, idle
```

The three run actions execute one job from the corresponding queue if possible. The idle action leaves the CPU idle for one time step.

Reward:

- Positive reward for completing a short, medium, or long job.
- Negative waiting cost for jobs left in the queues.
- Negative switching cost when the CPU changes from one job class to another.
- Negative idle cost when the scheduler chooses `idle`.
- Negative invalid-action penalty when the scheduler tries to run an empty queue.

The default reward parameters are:

```text
service_rewards = (2.0, 2.5, 3.0)
waiting_costs   = (1.0, 1.0, 1.0)
switch_cost     = 0.30
invalid_penalty = 2.00
idle_penalty    = 0.10
```

Transition:

1. The selected action is applied first.
2. Queue lengths and CPU mode are updated.
3. New jobs arrive stochastically according to fixed Bernoulli arrival probabilities.
4. Queue lengths are capped by `max_queue`.

The main experiment uses:

```text
max_queue = 2
arrival_probs = (0.35, 0.25, 0.15)
states = 3 * 3 * 3 * 4 = 108
actions = 4
gamma = 0.95
```

## Algorithms

Heuristic baselines:

- Shortest Job First
- Longest Job First
- Max Queue
- Sticky mode

Dynamic Programming baseline:

- Value Iteration

Model-free RL baselines:

- Monte Carlo Control
- SARSA

Selected method:

- Q-learning

No function approximation, neural networks, or non-tabular models are used.

## Source Files

```text
cpu_scheduling_env.py             CPU scheduling MDP environment
scheduling_common.py              Shared finite-MDP environment utilities
dp_solver.py                      Value Iteration implementation
control_agents.py                 MC Control, SARSA, and Q-learning
rl_utils.py                       Tabular Q-table and policy helpers
scheduling_policies.py            Handwritten heuristic baselines
scheduling_reporting.py           Evaluation and result-table helpers
experiment_scheduling.py          Main numerical experiment
visualize_scheduling_policies.py  Text rollout visualization
make_submission_assets.py         Result CSV and bar chart generator
make_mdp_graph.py                 State-space visualization generator
make_policy_demo.py               Interactive rollout demo generator
```

## How to Run

Python 3.10 or newer is recommended. No external package is required.

Run the result table:

```text
python experiment_scheduling.py
```

Print the text-based rollout visualization:

```text
python visualize_scheduling_policies.py
```

Save the visualization to a text file:

```text
python visualize_scheduling_policies.py > policy_visualization.txt
```

Generate result files:

```text
python make_submission_assets.py
```

This creates:

```text
submission_assets/policy_results.csv
submission_assets/policy_results.svg
```

Generate visual demos:

```text
python make_mdp_graph.py
python make_policy_demo.py
```

This creates:

```text
submission_assets/full_mdp_graph.svg
submission_assets/full_mdp_graph_3d.html
submission_assets/policy_demo.html
```

## Results

The values below are average undiscounted returns over 500 fixed-seed evaluation episodes. Higher is better.

CPU Job Scheduling:

```text
SJF                        105.80
LJF                        112.79
MaxQueue                    97.93
Sticky                     107.28
Value Iteration            113.48
MC Control                  71.55
SARSA                      112.04
Q-learning (Selected)      112.96
```

The selected Q-learning policy beats the best non-DP baseline by `0.17` average return and is only `0.52` below Value Iteration.

## Discussion

Value Iteration has the exact transition model, so it is best read as a model-based reference point rather than a realistic online learner. Q-learning only learns from sampled interaction. Therefore, the important result is that Q-learning gets close to Value Iteration while beating the handwritten rules and the other model-free learners.

The result is reasonable because long jobs have the largest completion reward and low arrival probability, but ignoring shorter queues creates waiting penalties. The learned policy often prioritizes high-value long jobs, avoids invalid actions, and still clears short or medium queues when waiting cost becomes important. The current CPU mode is included in the state so the agent can learn when switching is worth the cost.

In this setup, Q-learning is the best non-DP method. It slightly improves over the strongest heuristic baseline and stays close to the Value Iteration score.

MC Control, SARSA, and Q-learning are trained with the same 50,000-episode budget in the main experiment. This makes the model-free comparison fair: Q-learning is not advantaged by extra training episodes.
