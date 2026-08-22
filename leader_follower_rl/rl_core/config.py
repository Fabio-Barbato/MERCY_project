from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SACConfig:
    obs_dim: int
    act_dim: int
    hidden_sizes: Tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    auto_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: Optional[float] = None  # default: -act_dim
    device: str = "cpu"


@dataclass
class TrainConfig:
    algo: str = "sac"
    total_steps: int = 200_000
    start_steps: int = 5_000       # step iniziali con azioni random per esplorazione
    update_after: int = 1_000
    update_every: int = 50
    batch_size: int = 256
    buffer_capacity: int = 500_000
    eval_every: int = 5_000
    eval_episodes: int = 5
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
