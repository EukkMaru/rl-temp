"""Classical tabular model-free control agents."""

from __future__ import annotations

from collections import defaultdict

from rl_utils import epsilon_greedy_action, make_Q


class SARSAAgent:
    """On-policy one-step SARSA control."""

    def __init__(self, env, gamma=0.95, alpha=0.10, epsilon=0.10, max_steps=200):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.Q = make_Q(env)
        self.episode_returns = []

    def train(self, num_episodes=1000, start_state=None, print_every=0):
        if start_state is None:
            start_state = self.env.start_state

        for episode in range(1, num_episodes + 1):
            state = start_state
            action = epsilon_greedy_action(self.env, self.Q, state, self.epsilon)
            total_reward = 0.0

            for _ in range(self.max_steps):
                next_state, reward, done = self.env.step(state, action)
                next_action = epsilon_greedy_action(self.env, self.Q, next_state, self.epsilon)
                target = reward if done else reward + self.gamma * self.Q[next_state][next_action]
                self.Q[state][action] += self.alpha * (target - self.Q[state][action])
                total_reward += reward
                state, action = next_state, next_action
                if done:
                    break

            self.episode_returns.append(total_reward)
            if print_every and episode % print_every == 0:
                print(f"SARSA episode {episode}: return={total_reward:.2f}")


class QLearningAgent:
    """Off-policy one-step Q-learning control."""

    def __init__(self, env, gamma=0.95, alpha=0.10, epsilon=0.10, max_steps=200):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.Q = make_Q(env)
        self.episode_returns = []

    def train(self, num_episodes=1000, start_state=None, print_every=0):
        if start_state is None:
            start_state = self.env.start_state

        for episode in range(1, num_episodes + 1):
            state = start_state
            total_reward = 0.0

            for _ in range(self.max_steps):
                action = epsilon_greedy_action(self.env, self.Q, state, self.epsilon)
                next_state, reward, done = self.env.step(state, action)
                best_next = max(self.Q[next_state].values())
                target = reward if done else reward + self.gamma * best_next
                self.Q[state][action] += self.alpha * (target - self.Q[state][action])
                total_reward += reward
                state = next_state
                if done:
                    break

            self.episode_returns.append(total_reward)
            if print_every and episode % print_every == 0:
                print(f"Q-learning episode {episode}: return={total_reward:.2f}")


class MonteCarloControlAgent:
    """First-visit epsilon-soft Monte Carlo control."""

    def __init__(self, env, gamma=0.95, epsilon=0.10, max_steps=200):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.Q = make_Q(env)
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.episode_returns = []

    def train(self, num_episodes=1000, start_state=None, print_every=0):
        if start_state is None:
            start_state = self.env.start_state

        for episode_idx in range(1, num_episodes + 1):
            episode = self._generate_episode(start_state)
            G = 0.0
            visited = set()

            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                key = (state, action)
                if key in visited:
                    continue
                visited.add(key)
                self.returns_sum[key] += G
                self.returns_count[key] += 1
                self.Q[state][action] = self.returns_sum[key] / self.returns_count[key]

            total_reward = sum(reward for _state, _action, reward in episode)
            self.episode_returns.append(total_reward)
            if print_every and episode_idx % print_every == 0:
                print(f"MC Control episode {episode_idx}: return={total_reward:.2f}")

    def _generate_episode(self, start_state):
        state = start_state
        episode = []
        for _ in range(self.max_steps):
            action = epsilon_greedy_action(self.env, self.Q, state, self.epsilon)
            next_state, reward, done = self.env.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode
