import gymnasium as gym
import numpy as np


class Agent:
    """
    Disallow missing docstrings.

    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.bins = [
            np.linspace(observation_space.low[i], observation_space.high[i], 20)
            for i in range(self.observation_space.shape[0])
        ]
        self.q_table = np.zeros((*[len(b) + 1 for b in self.bins], self.action_space.n))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def discretize(self, observation):
        return tuple(np.digitize(obs, bin) for obs, bin in zip(observation, self.bins))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Disallow missing docstrings.

        """
        discrete_obs = self.discretize(observation)
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[discrete_obs])

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Disallow missing docstrings.

        """
        discrete_obs = self.discretize(observation)
        if terminated or truncated:
            self.q_table[discrete_obs] = reward
        else:
            max_future_q = np.max(self.q_table[discrete_obs])
            current_q = self.q_table[discrete_obs]
            new_q = (1 - self.alpha) * current_q + self.alpha * (
                reward + self.gamma * max_future_q
            )
            self.q_table[discrete_obs] = new_q

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
