"""
Esporta la policy addestrata (SOLO l'actor) in TorchScript e, opzionalmente,
in ONNX: il formato pronto per essere caricato dal layer di inferenza
(inference/policy_runtime.py) SENZA bisogno di questo repository ne'
del package rl_core. Questo file e' il confine netto tra "ambiente di
training" e "ambiente di deploy" (ROS2 nella VM / Webots su macOS).

Uso:
    python scripts/export_policy.py --checkpoint checkpoints/best.pt \
        --obs-dim 9 --act-dim 2 --act-limit 0.6 --out policy.pt --onnx
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_core.config import SACConfig
from rl_core.agent_factory import make_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--algo", default="sac")
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--act-dim", type=int, required=True)
    parser.add_argument("--out", default="policy.pt")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()

    cfg = SACConfig(obs_dim=args.obs_dim, act_dim=args.act_dim)
    agent = make_agent(args.algo, config=cfg)
    agent.load(args.checkpoint)

    example_obs = np.zeros((args.obs_dim,), dtype=np.float32)
    agent.export_actor(args.out, example_obs)
    print(f"Policy TorchScript esportata in: {args.out}")

    if args.onnx:
        onnx_path = os.path.splitext(args.out)[0] + ".onnx"
        agent.actor.eval()
        dummy = torch.zeros((1, args.obs_dim), dtype=torch.float32)

        class Det(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                a, _ = self.actor(obs, deterministic=True, with_logprob=False)
                return a

        torch.onnx.export(
            Det(agent.actor), dummy, onnx_path,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,  # esportatore "legacy": evita la dipendenza opzionale onnxscript
        )
        print(f"Policy ONNX esportata in: {onnx_path}")


if __name__ == "__main__":
    main()
