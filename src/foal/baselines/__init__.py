"""Continual learning baseline implementations (LwF, Official-Protocol LwF, ER)."""
from src.foal.baselines.er import ERAgent, ReplayBuffer
from src.foal.baselines.lwf import LwFAgent, LwFClassifier
from src.foal.baselines.official_lwf import (
    OfficialLwFAgent,
    OfficialLwFClassifier,
    official_loss_fn_kd,
)

__all__ = [
    "LwFAgent",
    "LwFClassifier",
    "ERAgent",
    "ReplayBuffer",
    "OfficialLwFAgent",
    "OfficialLwFClassifier",
    "official_loss_fn_kd",
]
