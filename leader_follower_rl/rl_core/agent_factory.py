from __future__ import annotations
from typing import Dict, Type
from .base_agent import BaseAgent
from .sac_agent import SACAgent

_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "sac": SACAgent,
    # "td3": TD3Agent,      # da aggiungere in futuro implementando BaseAgent
    # "ddpg": DDPGAgent,
}


def make_agent(algo: str, **kwargs) -> BaseAgent:
    algo = algo.lower()
    if algo not in _REGISTRY:
        raise ValueError(f"Algoritmo '{algo}' non registrato. Disponibili: {list(_REGISTRY)}")
    return _REGISTRY[algo](**kwargs)


def register_agent(name: str, cls: Type[BaseAgent]) -> None:
    """Permette di aggiungere nuovi algoritmi senza modificare questo file
    (es. da uno script esterno: register_agent('td3', TD3Agent))."""
    _REGISTRY[name.lower()] = cls
