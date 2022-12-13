import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from copy import deepcopy
import itertools
from torch.optim import Adam
from models.dqn import ReplayMemory

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

GAMMA = 0.99
TAU = 1e-3              # for soft update of target parameters


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action -
                        F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] +
                     list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space#.shape[0]
        act_dim = action_space#.shape[0]
        act_limit = 1 #action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()


class SAC:
    def __init__(self, observation_space, action_space, seed=2, gamma=0.99,
                 # lr=5e-4,batch_size=64,
                 epsilon_min=0.05, epsilon_max=1,  buffer_size=5e5,
                 ac_kwargs=dict(),
                 steps_per_epoch=4000, epochs=100, replay_size=int(1e6),
                 polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
                 update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
                 logger_kwargs=dict(), save_freq=1,
                 load=False):
        self.polyak = polyak
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = action_space

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(observation_space, action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        self.memory = ReplayMemory(buffer_size=int(buffer_size), batch_size=batch_size)


    def act(self, state, epsilon=0, **kwargs):
        # Use epsilon-greedy for exploration
        if epsilon > np.random.random():
            # action = np.random.choice(self.num_actions)
            action_probs = np.random.random(self.num_actions)
        else:
            action_probs = self.ac.act(state, deterministic=False)
            # action = np.argmax(action_probs)

        return action_probs
        
    def optimize_model(self, gamma=GAMMA, tau=TAU, criterion=None):
        if len(self.memory) < self.batch_size:
            return 0
        update_every = 10 # FIXME: hardcode
        losses = []
        for j in range(update_every):
            states, actions, rewards, next_states, dones = self.memory.sample()

            l= self.update(states, actions, rewards, next_states, dones)
            losses.append(l)
        return sum(losses)/len(losses)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, states, actions, rewards, next_states, dones):
        o, a, r, o2, d = states, actions, rewards, next_states, dones

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
            backup = r + self.gamma * ~d * (q_pi_targ - self.alpha * logp_a2) # bool tensor

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, states, actions, rewards, next_states, dones):
        o = states
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update(self, states, actions, rewards, next_states, dones):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(states, actions, rewards, next_states, dones)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(states, actions, rewards, next_states, dones)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_pi.item()

    def save(self, log_path, flag):
        if flag is None:
            prefix = 'reward'
        elif flag:
            prefix = 'throughput'
        else:
            prefix = 'time'
        torch.save(self.ac.q1.state_dict(),
                   log_path + f'/{prefix}_q_net.pt')
        torch.save(self.ac.q2.state_dict(),
                   log_path + f'/{prefix}_target_net.pt')

