EL5001 Project 01 - Scheduling MDP with Dynamic Programming and Tabular RL

Purpose
This code tests three scheduling problems as small finite Markov Decision
Processes:

1. Disk head scheduling
2. Elevator pickup scheduling
3. CPU job-class scheduling

The comparison includes fixed scheduling rules, Value Iteration, Monte Carlo
Control, SARSA, and Q-learning. We use Q-learning as the main learned policy.


MDP Formulation
Shared setup:
- State: a tuple describing the current scheduling situation.
- Action: one decision, such as move, serve, run a job, or idle.
- Reward: positive reward for serving or completing work, negative cost for
  waiting jobs, movement, idling, switching, and invalid actions.
- Transition: the action is applied first; then new requests/jobs arrive
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

Selected method:
- Q-learning. It is trained longer than MC Control and SARSA because it is the
  final policy being reported.

No function approximation, neural networks, or non-tabular models are used.


How to Run
Run the result table:

    python experiment_scheduling.py

Print the text-based CLI visualization:

    python visualize_scheduling_policies.py

Save the CLI visualization to a text file:

    python visualize_scheduling_policies.py > policy_visualization.txt


Results
These numbers are average undiscounted returns over 500 fixed-seed evaluation
episodes. Higher is better.

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
Value Iteration gets the exact transition model, so it is best read as a
model-based reference point. Q-learning only sees sampled transitions, so the
important check is whether it gets close to Value Iteration while beating the
handwritten rules and the other model-free learners.

In these runs, Q-learning is the best non-DP method in all three environments.
It stays close to Value Iteration, which is the behavior we wanted from a
tabular model-free method on these small scheduling MDPs.
