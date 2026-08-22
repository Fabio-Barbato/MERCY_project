"""
Runtime di inferenza MINIMALE: nessuna dipendenza da rl_core, gymnasium,
ROS2 o Webots. Le uniche dipendenze sono numpy + UN backend di
inferenza a scelta (torch oppure onnxruntime).

Questo e' l'UNICO file che il codice ROS2/Webots deve importare per
usare la policy addestrata: e' il confine netto tra "training" e "deploy",
e la ragione per cui l'integrazione e' modulare (vedi integration/).
"""
from __future__ import annotations
import numpy as np


class TorchScriptPolicy:
    """Backend basato su TorchScript (richiede solo `torch`, non l'intero rl_core)."""

    def __init__(self, model_path: str):
        import torch  # import locale: solo chi sceglie questo backend paga il costo di torch
        self._torch = torch
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            obs_t = self._torch.as_tensor(obs, dtype=self._torch.float32).unsqueeze(0)
            action = self.model(obs_t)
        return action.squeeze(0).numpy()


class OnnxPolicy:
    """Backend basato su ONNX Runtime: nessuna dipendenza da PyTorch.
    Consigliato per il deploy finale su robot/Webots: onnxruntime e'
    molto piu' leggero di torch e ha binari per macOS/Linux/ARM."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        (action,) = self.session.run(None, {self.input_name: obs})
        return action[0]


def load_policy(model_path: str):
    """Factory: sceglie il backend in base all'estensione del file."""
    if model_path.endswith(".onnx"):
        return OnnxPolicy(model_path)
    return TorchScriptPolicy(model_path)
