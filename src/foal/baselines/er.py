"""Experience Replay (ER) Baseline for Continual Learning.

Reference:
  Chaudhry et al., "Continual Learning with Tiny Episodic Memories", ICRA 2019.
  NeurIPS 2024 F-OAL Table 2 Reference on CIFAR-100:
    A_avg = 90.1%, A_last = 84.6%, F = 20.7% (Replay-based, Buffer = 5000)
  Official Author Command:
    python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000 --learning_rate 0.001
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.foal.baselines.lwf import LwFClassifier


class ReplayBuffer:
    """Reservoir sampling replay buffer storing feature representations and labels."""

    def __init__(self, capacity: int = 5000, seed: int = 42):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.features: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.total_seen: int = 0

    def __len__(self) -> int:
        return len(self.labels)

    def add(self, feature: torch.Tensor, label: int) -> None:
        """Add a sample using standard reservoir sampling (uniform random replacement)."""
        feat_cpu = feature.detach().cpu().clone()
        lbl_int = int(label)

        if len(self.labels) < self.capacity:
            self.features.append(feat_cpu)
            self.labels.append(lbl_int)
        else:
            # Reservoir replacement probability = capacity / total_seen
            idx = self.rng.randint(0, self.total_seen)
            if idx < self.capacity:
                self.features[idx] = feat_cpu
                self.labels[idx] = lbl_int
        self.total_seen += 1

    def add_batch(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a batch of samples into the replay memory."""
        for feat, lbl in zip(features, labels):
            self.add(feat, int(lbl))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a uniform random batch from the replay memory."""
        n = len(self.labels)
        if n == 0:
            return torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        k = min(batch_size, n)
        indices = self.rng.sample(range(n), k)
        feat_batch = torch.stack([self.features[i] for i in indices], dim=0)
        label_batch = torch.tensor([self.labels[i] for i in indices], dtype=torch.long)
        return feat_batch, label_batch


class ERAgent:
    """Experience Replay agent operating on feature representations with episodic memory."""

    def __init__(
        self,
        feature_dim: int,
        initial_classes: int = 10,
        buffer_capacity: int = 5000,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.feature_dim = feature_dim
        self.num_classes = initial_classes
        self.buffer_capacity = buffer_capacity
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device)

        self.model = LwFClassifier(feature_dim, initial_classes).to(self.device)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self._init_optimizer()

    def _init_optimizer(self) -> None:
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def before_task(self, task_id: int, total_seen_classes: int) -> None:
        """Expand output head for newly introduced classes."""
        if task_id > 0 and total_seen_classes > self.num_classes:
            self.model.expand_classes(total_seen_classes, device=self.device)
            self.num_classes = total_seen_classes
            self._init_optimizer()

    def train_batch(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Process incoming stream batch, retrieve buffer exemplars, and update model and memory."""
        self.model.train()
        self.optimizer.zero_grad()

        stream_features = features.to(self.device)
        stream_labels = labels.to(self.device)

        # Retrieve random exemplars from memory buffer
        if len(self.buffer) > 0:
            mem_feats, mem_labels = self.buffer.sample(len(stream_labels))
            comb_features = torch.cat([stream_features, mem_feats.to(self.device)], dim=0)
            comb_labels = torch.cat([stream_labels, mem_labels.to(self.device)], dim=0)
        else:
            comb_features = stream_features
            comb_labels = stream_labels

        logits = self.model(comb_features)
        loss = F.cross_entropy(logits, comb_labels)
        loss.backward()
        self.optimizer.step()

        # Update replay buffer with newly seen stream samples
        self.buffer.add_batch(features, labels)

        return {
            "loss_total": float(loss.item()),
            "buffer_occupancy": len(self.buffer),
        }

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features.to(self.device))
            return logits.argmax(dim=1).cpu()

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor, batch_size: int = 128) -> float:
        self.model.eval()
        correct = 0
        total = len(labels)
        if total == 0:
            return float("nan")

        with torch.no_grad():
            for i in range(0, total, batch_size):
                feat_b = features[i:i + batch_size].to(self.device)
                lbl_b = labels[i:i + batch_size]
                preds = self.model(feat_b).argmax(dim=1).cpu()
                correct += int((preds == lbl_b).sum())
        return correct / total
