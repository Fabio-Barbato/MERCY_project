from __future__ import annotations
import copy
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .base_agent import BaseAgent
from .networks import GaussianActor, TwinQCritic
from .config import SACConfig


class SACAgent(BaseAgent):
    def __init__(self, config: SACConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.actor = GaussianActor(config.obs_dim, config.act_dim, config.hidden_sizes).to(self.device)
        self.critic = TwinQCritic(config.obs_dim, config.act_dim, config.hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim = Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.lr)

        self.auto_alpha = config.auto_alpha
        if self.auto_alpha:
            self.target_entropy = config.target_entropy if config.target_entropy is not None else -float(config.act_dim)
            self.log_alpha = torch.tensor(float(np.log(config.init_alpha)), device=self.device, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.log_alpha = torch.tensor(float(np.log(config.init_alpha)), device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        # ---- critic update ----
        with torch.no_grad():
            next_action, next_logp = self.actor(next_obs, with_logprob=True)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha.detach() * next_logp
            backup = rew + self.cfg.gamma * (1 - done) * q_t

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- actor update ----
        for p in self.critic.parameters():
            p.requires_grad = False

        pi_action, logp_pi = self.actor(obs, with_logprob=True)
        q1_pi, q2_pi = self.critic(obs, pi_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # ---- alpha (temperatura entropia) update ----
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_loss_val = alpha_loss.item()
        else:
            alpha_loss_val = 0.0

        # ---- soft update del target critic ----
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)

        return dict(
            critic_loss=critic_loss.item(),
            actor_loss=actor_loss.item(),
            alpha_loss=alpha_loss_val,
            alpha=self.alpha.item(),
        )

    def save(self, path: str) -> None:
        torch.save(dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            critic_target=self.critic_target.state_dict(),
            log_alpha=self.log_alpha.detach().clone(),
            cfg=self.cfg,
        ), path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"])

    def export_actor(self, path: str, example_obs: np.ndarray) -> None:
        """Esporta la sola policy (actor) in TorchScript, in modalita'
        deterministica (media della Gaussiana, niente sampling: per il
        deploy si vuole un comportamento riproducibile). Il file risultante
        NON dipende da questo package per essere eseguito: basta
        `torch.jit.load(path)`."""
        self.actor.eval()

        class DeterministicWrapper(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                action, _ = self.actor(obs, deterministic=True, with_logprob=False)
                return action

        wrapper = DeterministicWrapper(self.actor)
        example = torch.as_tensor(example_obs, dtype=torch.float32).unsqueeze(0)
        traced = torch.jit.trace(wrapper, example)
        traced.save(path)
        self.actor.train()
