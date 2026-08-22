import numpy as np
import torch
import onnxruntime as ort

from rl_core.config import SACConfig
from rl_core.agent_factory import make_agent

cfg = SACConfig(
    obs_dim=9,
    act_dim=2,
)

agent = make_agent("sac", config=cfg)
agent.load("checkpoints/best.pt")
agent.actor.eval()

session = ort.InferenceSession(
    "policy.onnx",
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX input :", input_name)
print("ONNX output:", output_name)
print("Input shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)

rng = np.random.default_rng(42)

observations = rng.normal(
    size=(100, 9)
).astype(np.float32)

max_error = 0.0
total_error = 0.0

for obs in observations:
    obs_t = torch.from_numpy(obs).unsqueeze(0)

    with torch.no_grad():
        action_pt, _ = agent.actor(
            obs_t,
            deterministic=True,
            with_logprob=False,
        )

    action_pt = action_pt.cpu().numpy()

    action_onnx = session.run(
        [output_name],
        {input_name: obs.reshape(1, 9)},
    )[0]

    error = np.abs(action_pt - action_onnx)

    max_error = max(max_error, float(error.max()))
    total_error += float(error.mean())

mean_error = total_error / len(observations)

print()
print("=== CONFRONTO PYTORCH vs ONNX ===")
print(f"Numero test : {len(observations)}")
print(f"Errore medio: {mean_error:.10e}")
print(f"Errore max  : {max_error:.10e}")
