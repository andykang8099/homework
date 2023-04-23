import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)  # get batch size explicitly
        x = x.view(batch_size, -1)  # flatten input tensor
        return self.layers(x)


class Agent:
    """
    Agent
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = []
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(self.observation_space.shape[0], self.action_space.n).to(
            self.device
        )
        self.target_network = DQN(
            self.observation_space.shape[0], self.action_space.n
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Selects an action to take given an observation using an epsilon-greedy policy.

        Parameters:
            observation (array-like): Current observation of the environment.

        Returns:
            action (int): Action to take.
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
        Updates the Q-values of the agent based on the observed reward and transition.

        Parameters:
            observation (array-like): Current observation of the environment.
            reward (float): Reward received from the environment.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        obs_batch, reward_batch, done_batch, next_obs_batch = zip(
            *[self.memory[i] for i in batch]
        )
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        reward_batch = (
            torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        done_batch = (
            torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(
            self.device
        )

        # Compute Q-values and target Q-values
        q_values = self.q_network(obs_batch)
        action_batch = torch.argmax(q_values, dim=1, keepdim=True)
        q_values = q_values.gather(1, action_batch)
        next_q_values = self.target_network(next_obs_batch).max(1)[0].detach()
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Update Q-values using the Bellman equation
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network parameters
        self.update_target_network()
