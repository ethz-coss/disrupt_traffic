import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.mlp import MLP
from gym.spaces import Box, Discrete

BUFFER_SIZE = int(1e5)  # replay buffer size
# GAMMA = 0.999           # discount factor
GAMMA = 0.8
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN:
    """ Actor (Policy) Model."""

    def __init__(self, observation_space, action_space, seed=2, gamma=0.99, lr=5e-4,
                 epsilon_min=0.05, epsilon_max=1, batch_size=64, buffer_size=5e5,
                 load=False):
        num_observations = observation_space.shape[0]
        if isinstance(action_space, Box):
            num_actions = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            num_actions = action_space.n
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # The first model makes the predictions for Q-values which are used to make a action.
        self.net_local = MLP(num_observations, num_actions,
                             seed=seed).to(device)

        # Build a target model for the prediction of future rewards.
        self.net_target = MLP(
            num_observations, num_actions, seed=seed).to(device)
        self.net_target.load_state_dict(self.net_local.state_dict())

        if load:
            self.net_local.load_state_dict(torch.load(
                load, map_location=torch.device('cpu')))
            self.net_local.eval()

            self.net_target.load_state_dict(torch.load(
                load, map_location=torch.device('cpu')))
            self.net_target.eval()

        self.optimizer = optim.Adam(
            self.net_local.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(batch_size=batch_size)
        self.step_count = 0


    def act(self, state, epsilon=0, **kwargs):
        # Use epsilon-greedy for exploration
        if epsilon > np.random.random():
            # Take random action
            action = np.random.choice(self.num_actions)
        else:
            state = state.unsqueeze(0)
            self.net_local.eval()
            with torch.no_grad():
                # Predict action Q-values from state
                action_probs = self.net_local(state)
                # Take best action
            self.net_local.train()
            action = action_probs.max(1)[1].item()
        return action

    def optimize_model(self, gamma=GAMMA, tau=TAU, criterion=None):
        """Update value parameters using given batch of experience tuples.

        Params
        =======

        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples

        gamma (float): discount factor
        """
        if len(self.memory) < self.batch_size:
            return 0
        if criterion is None:
            criterion = nn.MSELoss()

        states, actions, rewards, next_states, dones = self.memory.sample()

        self.net_local.train()
        self.net_target.eval()

        predicted_targets = self.net_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.net_target(next_states).detach().max(1)[
                0].unsqueeze(1)

        labels = rewards + (gamma * labels_next * ~dones)

        # .detach() ->  Returns a new Tensor, detached from the current graph.

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        # ------------------- update target network ------------------- #
        soft_update(self.net_local, self.net_target, TAU)

        return loss.item()

    def save(self, log_path, flag):
        if flag is None:
            prefix = 'reward'
        elif flag:
            prefix = 'throughput'
        else:
            prefix = 'time'
        torch.save(self.net_local.state_dict(),
                   log_path + f'/{prefix}_q_net.pt')
        torch.save(self.net_target.state_dict(),
                   log_path + f'/{prefix}_target_net.pt')


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    =======
        local model (PyTorch model): weights will be copied from
        target model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter

    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(tau*local_param.data +
                                (1-tau)*target_param.data)


class ReplayMemory(object):
    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=64, seed=42):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        samples = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = map(
            torch.stack, zip(*samples))
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __add__(self, other):
        self.memory += other.memory
        return self
