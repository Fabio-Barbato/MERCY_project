"""
Interfaccia comune a tutti gli algoritmi RL supportati.

Qualsiasi algoritmo (SAC, TD3, DDPG, PPO, ...) implementato in questo
package deve rispettare questa interfaccia, in modo che:
  - lo script di training (scripts/train.py) sia identico per tutti gli algoritmi
  - lo script di export (scripts/export_policy.py) sappia sempre come
    estrarre la sola rete "actor" da esportare per l'inferenza
  - il layer di integrazione (ROS2 / Webots) non debba mai sapere quale
    algoritmo ha addestrato la policy che sta eseguendo
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class BaseAgent(ABC):
    """Interfaccia minima per un agente RL off-policy con azioni continue."""

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Ritorna un'azione continua data un'osservazione (singola, non a batch)."""

    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Esegue uno step di ottimizzazione su un batch dal replay buffer.
        Ritorna un dizionario di metriche/loss per il logging."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Salva tutti i pesi necessari a riprendere il training (actor+critic+optimizer)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Carica lo stato salvato con save()."""

    @abstractmethod
    def export_actor(self, path: str, example_obs: np.ndarray) -> None:
        """Esporta SOLO la rete actor, in un formato adatto all'inferenza
        (TorchScript). Usata dal layer di deploy, mai dal training."""
