from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Policy stocastica con distribuzione Gaussiana squashata (tanh),
    come nel paper SAC originale (Haarnoja et al. 2018)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden_sizes], output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            pi_action = mu
            logp_pi = None
        else:
            dist = Normal(mu, std)
            pi_action = dist.rsample()
            if with_logprob:
                logp_pi = dist.log_prob(pi_action).sum(axis=-1)
                # correzione per lo squashing tanh (vedi appendice C del paper SAC)
                logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
            else:
                logp_pi = None

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = _mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)


class TwinQCritic(nn.Module):
    """Due Q-network indipendenti per ridurre l'overestimation bias (clipped double-Q)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = QCritic(obs_dim, act_dim, hidden_sizes)
        self.q2 = QCritic(obs_dim, act_dim, hidden_sizes)

    def forward(self, obs, act):
        return self.q1(obs, act), self.q2(obs, act)
