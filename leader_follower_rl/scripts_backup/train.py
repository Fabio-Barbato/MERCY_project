"""
Training completamente isolato: nessun import di rclpy, webots o
qualsiasi libreria legata all'ambiente reale. Gira su un laptop
qualunque, anche senza VM/Webots installati, e in pochi minuti da
un'idea al primo checkpoint da valutare.

Uso:
    python scripts/train.py --algo sac --total-steps 200000
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_core.config import SACConfig, TrainConfig
from rl_core.agent_factory import make_agent
from rl_core.replay_buffer import ReplayBuffer
from envs.leader_follower_env import LeaderFollowerEnv


def evaluate(agent, env, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, trunc, ep_ret = False, False, 0.0
        while not (done or trunc):
            action = agent.select_action(obs, deterministic=True)
            obs, rew, done, trunc, _ = env.step(action)
            ep_ret += rew
        returns.append(ep_ret)
    return float(np.mean(returns))


def run_training(tcfg: TrainConfig, verbose: bool = True):
    np.random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)

    env = LeaderFollowerEnv(seed=tcfg.seed)
    eval_env = LeaderFollowerEnv(seed=tcfg.seed + 1000)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    acfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit)
    agent = make_agent(tcfg.algo, config=acfg)

    buffer = ReplayBuffer(obs_dim, act_dim, capacity=tcfg.buffer_capacity)

    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(tcfg.log_dir)

    obs, _ = env.reset()
    ep_ret = 0.0
    best_eval = -np.inf
    metrics = {}

    for step in range(1, tcfg.total_steps + 1):
        if step < tcfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, rew, terminated, truncated, _ = env.step(action)
        # nota: solo "terminated" viene passato come done al bootstrap del target Q;
        # un "truncated" per timeout non deve azzerare il valore futuro stimato.
        buffer.add(obs, action, rew, next_obs, float(terminated))

        obs = next_obs
        ep_ret += rew

        if terminated or truncated:
            writer.add_scalar("train/episode_return", ep_ret, step)
            obs, _ = env.reset()
            ep_ret = 0.0

        if step >= tcfg.update_after and step % tcfg.update_every == 0 and len(buffer) >= tcfg.batch_size:
            for _ in range(tcfg.update_every):
                batch = buffer.sample(tcfg.batch_size)
                metrics = agent.update(batch)
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, step)

        if step % tcfg.eval_every == 0:
            avg_ret = evaluate(agent, eval_env, tcfg.eval_episodes)
            writer.add_scalar("eval/avg_return", avg_ret, step)
            if verbose:
                print(f"[step {step}] eval avg return: {avg_ret:.2f}")
            if avg_ret > best_eval:
                best_eval = avg_ret
                agent.save(os.path.join(tcfg.checkpoint_dir, "best.pt"))

    agent.save(os.path.join(tcfg.checkpoint_dir, "final.pt"))
    if verbose:
        print("Training completato. Checkpoint in:", tcfg.checkpoint_dir)
    return agent, dict(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="sac")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--start-steps", type=int, default=5_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs")
    args = parser.parse_args()

    tcfg = TrainConfig(
        algo=args.algo, total_steps=args.total_steps, start_steps=args.start_steps,
        update_after=args.update_after, update_every=args.update_every,
        batch_size=args.batch_size, buffer_capacity=args.buffer_capacity,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        seed=args.seed, checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
    )
    run_training(tcfg)


if __name__ == "__main__":
    main()
