"""Continual learning baseline implementations (LwF, ER)."""
from src.foal.baselines.lwf import LwFAgent, LwFClassifier
from src.foal.baselines.er import ERAgent, ReplayBuffer

__all__ = [
    "LwFAgent",
    "LwFClassifier",
    "ERAgent",
    "ReplayBuffer",
]
