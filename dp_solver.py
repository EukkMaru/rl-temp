"""Dynamic programming solvers for finite tabular MDPs."""

from __future__ import annotations


class DynamicProgrammingSolver:
    def __init__(self, env, gamma=0.95, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta

    def value_iteration(self, verbose=False, max_iterations=10_000):
        V = {state: 0.0 for state in self.env.states}

        for iteration in range(max_iterations):
            delta = 0.0
            for state in self.env.states:
                if self.env.is_terminal(state):
                    continue
                old_value = V[state]
                V[state] = max(self._action_value(V, state, action) for action in self.env.actions)
                delta = max(delta, abs(old_value - V[state]))

            if verbose and iteration % 25 == 0:
                print(f"value iteration {iteration}: delta={delta:.6f}")
            if delta < self.theta:
                break

        policy = self.greedy_policy(V)
        return policy, V

    def greedy_policy(self, V):
        policy = {}
        for state in self.env.states:
            best_action = max(
                self.env.actions,
                key=lambda action: (self._action_value(V, state, action), -self.env.actions.index(action)),
            )
            policy[state] = {action: 0.0 for action in self.env.actions}
            policy[state][best_action] = 1.0
        return policy

    def _action_value(self, V, state, action):
        total = 0.0
        for next_state in self.env.states:
            prob = self.env.transition_prob(next_state, state, action)
            if prob:
                reward = self.env.reward(state, action, next_state)
                total += prob * (reward + self.gamma * V[next_state])
        return total
