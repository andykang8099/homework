import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class Agent:
    """
    Disallow missing docstrings.

    """
    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(self.observation_space.shape[0], self.action_space.n).to(self.device)
        self.target_network = DQN(self.observation_space.shape[0], self.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.prev_observation = None
        self.prev_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
    Disallow missing docstrings.

        """
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            state = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.q_network(state)
            return torch.argmax(action_values).item()

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
        if self.prev_observation is not None and self.prev_action is not None:
            self.memory.append((self.prev_observation, self.prev_action, reward, observation, terminated))
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.vstack(states).astype(np.float32)).to(self.device)
        actions = torch.tensor([action for action in actions]).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.tensor(np.vstack(next_states).astype(np.float32)).to(self.device)
        dones = torch.tensor([float(done) for done in dones]).to(self.device)

        curr_Q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_Q = self.target_network(next_states).detach().max(1)[0]
        target_Q = rewards + (1 - dones) * self.gamma * next_Q

        loss = self.loss_fn(curr_Q, target_Q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, observation, action):
        self.prev_observation = observation
        self.prev_action = action





