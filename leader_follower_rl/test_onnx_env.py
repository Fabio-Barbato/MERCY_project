import numpy as np
import torch
import onnxruntime as ort

from envs.leader_follower_env import LeaderFollowerEnv
from rl_core.config import SACConfig
from rl_core.agent_factory import make_agent


# ---------------------------------------------------------
# 1. Carica il checkpoint PyTorch
# ---------------------------------------------------------

cfg = SACConfig(
    obs_dim=9,
    act_dim=2,
)

agent = make_agent("sac", config=cfg)
agent.load("checkpoints/best.pt")
agent.actor.eval()


# ---------------------------------------------------------
# 2. Carica ONNX
# ---------------------------------------------------------

session = ort.InferenceSession(
    "policy.onnx",
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# ---------------------------------------------------------
# 3. Ambiente reale di training
# ---------------------------------------------------------

env = LeaderFollowerEnv(seed=123)

obs, _ = env.reset()


# ---------------------------------------------------------
# 4. Confronto lungo una traiettoria reale
# ---------------------------------------------------------

max_error = 0.0
mean_error = 0.0
num_steps = 100

print("=== TEST ONNX SU TRAIETTORIA REALE ===")
print()

for step in range(num_steps):

    # -------------------------
    # PyTorch
    # -------------------------

    obs_t = torch.from_numpy(obs).unsqueeze(0)

    with torch.no_grad():
        action_pt, _ = agent.actor(
            obs_t,
            deterministic=True,
            with_logprob=False,
        )

    action_pt = action_pt.cpu().numpy()


    # -------------------------
    # ONNX
    # -------------------------

    action_onnx = session.run(
        [output_name],
        {
            input_name: obs.reshape(1, 9)
        },
    )[0]


    # -------------------------
    # Errore
    # -------------------------

    error = np.abs(action_pt - action_onnx)

    max_error = max(
        max_error,
        float(error.max()),
    )

    mean_error += float(error.mean())


    # -------------------------
    # Usa la stessa azione
    # per far evolvere l'ambiente
    # -------------------------

    action = action_pt[0]

    obs, reward, terminated, truncated, info = env.step(action)


    if terminated or truncated:
        obs, _ = env.reset()


mean_error /= num_steps


# ---------------------------------------------------------
# 5. Risultati
# ---------------------------------------------------------

print("Numero step :", num_steps)
print(f"Errore medio: {mean_error:.10e}")
print(f"Errore max  : {max_error:.10e}")
print()

if max_error < 1e-4:
    print("OK: PyTorch e ONNX sono equivalenti.")
else:
    print("ATTENZIONE: differenza superiore alla soglia.")
