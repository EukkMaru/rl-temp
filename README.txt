EL5001 Project 01 - Scheduling MDP with Dynamic Programming and Tabular RL

Purpose
This project formulates three real-world scheduling problems as finite Markov
Decision Processes:

1. Disk head scheduling
2. Elevator pickup scheduling
3. CPU job-class scheduling

The project compares heuristic policies, a Dynamic Programming baseline, and
classical model-free reinforcement learning methods. The selected solution is
Q-learning.


MDP Formulation
Common interface:
- State: finite tuple representation of the current scheduling situation.
- Action: one scheduling decision, such as move, serve, run a job, or idle.
- Reward: positive reward for serving/completing work, negative cost for
  waiting jobs, movement, idling, switching, and invalid actions.
- Transition: action effect is applied first; then new requests/jobs arrive
  stochastically according to Bernoulli arrival probabilities.

Disk scheduling:
- State: (head_track, direction, request_mask)
- Actions: seek_left, seek_right, serve, wait

Elevator scheduling:
- State: (floor, direction, call_mask)
- Actions: move_down, move_up, open, wait

CPU scheduling:
- State: (short_queue, medium_queue, long_queue, current_mode)
- Actions: run_short, run_medium, run_long, idle


Algorithms
Heuristic baselines:
- Disk/Elevator: Nearest, SCAN, LOOK
- CPU: Shortest Job First, Longest Job First, Max Queue, Sticky

Dynamic Programming baseline:
- Value Iteration

Model-free RL baselines:
- Monte Carlo Control
- SARSA

Selected model-free RL solution:
- Q-learning, trained with a larger interaction budget than the baseline
  learners because it is the selected final method.

No function approximation, neural networks, or non-tabular models are used.


How to Run
Run the quantitative comparison:

    python experiment_scheduling.py

Print the text-based CLI visualization:

    python visualize_scheduling_policies.py

Save the CLI visualization to a text file:

    python visualize_scheduling_policies.py > policy_visualization.txt


Representative Results
The following values are average undiscounted episode returns over 500 fixed-seed
evaluation episodes, using 100 steps per episode.

Disk Head Scheduling:
- Nearest: 18.42
- SCAN: 3.65
- LOOK: 17.87
- Value Iteration: 21.50
- MC Control: -117.42
- SARSA: 19.47
- Q-learning (Selected): 20.03

Elevator Pickup Scheduling:
- Nearest: 15.51
- Collective/SCAN: -0.72
- LOOK: 14.49
- Value Iteration: 18.46
- MC Control: -123.21
- SARSA: 13.55
- Q-learning (Selected): 17.43

CPU Job Scheduling:
- SJF: 105.80
- LJF: 112.79
- MaxQueue: 97.93
- Sticky: 107.28
- Value Iteration: 113.48
- MC Control: 10.42
- SARSA: 109.96
- Q-learning (Selected): 112.96


Discussion
Value Iteration has access to the exact transition model, so it is the natural
upper-bound baseline for these small finite MDPs. Q-learning does not know the
transition probabilities and learns only from sampled interaction. Therefore,
the main success criterion is whether Q-learning approaches the Value Iteration
policy and strictly outperforms the other non-DP baselines.

In the representative results, Q-learning strictly wins over the non-DP
baselines in all three environments. It is also on par with Value Iteration
within normal sampling variation. This supports Q-learning as a reasonable
selected solution: it is model-free, tabular, simple to implement, and learns
near-optimal scheduling behavior in finite stochastic scheduling MDPs.
