import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.mlp import MLP

BUFFER_SIZE = int(1e5)  # replay buffer size
# GAMMA = 0.999           # discount factor
GAMMA = 0.8
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    "Interacts with the environment"

    def __init__(self, num_observations, num_actions, gamma=0.99, epsilon_min=0.05, epsilon_max=1, batch_size=64, buffer_size=5e5):
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # The first model makes the predictions for Q-values which are used to make a action.
        self.net_local = MLP(num_observations, num_actions).to(device)
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every `update_target_network` steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        self.net_target = MLP(num_observations, num_actions).to(device)
        self.net_target.load_state_dict(self.net_local.state_dict())
        # Deepmind paper used RMSProp however then Adam optimizer is faster
        self.optimizer = optim.Adam(self.net_local.parameters(), lr=1e-3)
        self.memory = ReplayMemory(buffer_size, batch_size=batch_size)
        self.step_count = 0

    def step(self, action, state, state_next, reward, done):
        # Save actions and states in replay buffer
        self.memory.push(action, state, state_next, reward, done)

        self.step_count += 1
        # Update every `train_freq` frame if `batch_size` samples available
        if self.step_count % train_freq == 0 and len(self.memory) > self.batch_size:
            # sample the replay buffer
            experience_sample = self.memory.sample()
            self.optimize_model(experience_sample)

    def act(self, state, epsilon=0):
        # Use epsilon-greedy for exploration
        if epsilon > np.random.random():
            # Take random action
            action = torch.tensor(np.random.choice(
                self.num_actions), device=device).view(1, 1)
            # action = np.random.choice(self.num_actions)
        else:
            self.net_local.eval()
            with torch.no_grad():
                # Predict action Q-values from state
                action_probs = self.net_local(
                    torch.from_numpy(state).unsqueeze(0))
                # Take best action
                action = action_probs.max(1)[1].view(1, 1)
            self.net_local.train()
        return action

class DQN(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size, action_size, gamma=0.8, seed=2, fc1_unit=128,
                 fc2_unit = 64):
        """
         Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(DQN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        self.gamma = 0.8
    
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def optimize_model(experiences, net_local, net_target, optimizer, gamma=GAMMA, tau=TAU, criterion=None):
    """Update value parameters using given batch of experience tuples.

    Params
    =======

    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples

    gamma (float): discount factor
    """
    if criterion is None:
        criterion = nn.MSELoss()

    states, actions, rewards, next_states, dones = experiences

    net_local.train()
    net_target.eval()

    predicted_targets = net_local(states).gather(1, actions)

    with torch.no_grad():
        labels_next = net_target(next_states).detach().max(1)[0].unsqueeze(1)

    labels = rewards + (gamma * labels_next * ~dones)

    # .detach() ->  Returns a new Tensor, detached from the current graph.

    loss = criterion(predicted_targets, labels).to(device)
    optimizer.zero_grad()
    loss.backward()

    # for param in net_local.parameters():
    #     param.grad.data.clamp_(-1, 1)
    # torch.nn.utils.clip_grad.clip_grad_norm_(net_local.parameters(), 10)

    optimizer.step()

    # ------------------- update target network ------------------- #
    soft_update(net_local, net_target, TAU)

    return loss.item()


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

        states, actions, rewards, next_states, dones = map(torch.stack, zip(*samples))
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __add__(self, other):
        self.memory += other.memory
        return self