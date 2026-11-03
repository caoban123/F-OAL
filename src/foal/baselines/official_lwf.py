"""Official-Protocol Paper-Faithful Learning without Forgetting (LwF) Baseline.

Directly mirrors the official F-OAL implementation:
  Repository: https://github.com/liuyuchen-cz/F-OAL @ 73e4ac1
  Files: agents/lwf.py, utils/kd_manager.py, agents/base.py, general_main.py

Key Protocol Attributes (NeurIPS 2024 Table 2 Parity):
  1. Static 100-class output space (nn.Linear(feature_dim, num_classes)).
  2. Optimizer: Pure SGD (lr=0.001, momentum=0.0, weight_decay=0.0).
  3. Dynamic task-weighted loss formulation:
       L_total = (1 / (t + 1)) * L_new + (1 - 1 / (t + 1)) * L_old
  4. Full-spectrum Knowledge Distillation over all 100 output logits with T=2.0.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def official_loss_fn_kd(scores: torch.Tensor, target_scores: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Official distillation loss matching utils/kd_manager.py:loss_fn_kd."""
    log_scores_norm = F.log_softmax(scores / temperature, dim=1)
    targets_norm = F.softmax(target_scores / temperature, dim=1)
    kd_loss = (-1.0 * targets_norm * log_scores_norm).sum(dim=1).mean() * (temperature ** 2)
    return kd_loss


class OfficialLwFClassifier(nn.Module):
    """Linear classifier head with full output dimensionality (100 classes for CIFAR-100)."""

    def __init__(self, in_features: int = 768, num_classes: int = 100):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class OfficialLwFAgent:
    """Paper-faithful LwF agent matching liuyuchen-cz/F-OAL agents/lwf.py."""

    def __init__(
        self,
        feature_dim: int = 768,
        total_classes: int = 100,
        learning_rate: float = 0.001,
        temperature: float = 2.0,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = torch.device(device)

        self.model = OfficialLwFClassifier(feature_dim, total_classes).to(self.device)
        self.teacher_model: Optional[OfficialLwFClassifier] = None
        self.task_seen: int = 0

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def before_task(self, task_id: int, total_seen_classes: int = 100) -> None:
        """Prepare for task: snapshot teacher model if moving past Task 0."""
        if task_id > 0:
            self.teacher_model = copy.deepcopy(self.model)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        else:
            self.teacher_model = None

    def after_task(self) -> None:
        """Increment task counter matching official agents/base.py:after_train."""
        self.task_seen += 1

    def compute_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """Compute task-weighted CE and full-spectrum KD loss matching agents/lwf.py:53-55."""
        features = features.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(features)

        # 1. New task classification loss
        loss_new = F.cross_entropy(logits, labels)

        # 2. Distillation loss from teacher snapshot
        if self.teacher_model is not None and self.task_seen > 0:
            with torch.no_grad():
                prev_logits = self.teacher_model(features)
            loss_old = official_loss_fn_kd(logits, prev_logits, temperature=self.temperature)
        else:
            loss_old = torch.tensor(0.0, device=self.device)

        # 3. Dynamic loss weighting: 1 / (t + 1) * loss_new + (1 - 1 / (t + 1)) * loss_old
        w_new = 1.0 / (self.task_seen + 1)
        w_old = 1.0 - w_new
        total_loss = w_new * loss_new + w_old * loss_old

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_new": float(loss_new.item()),
            "loss_old": float(loss_old.item()) if isinstance(loss_old, torch.Tensor) else float(loss_old),
            "weight_new": w_new,
            "weight_old": w_old,
            "task_seen": self.task_seen,
        }
        return total_loss, metrics

    def train_batch(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Perform one optimization step with SGD."""
        self.model.train()
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(features, labels)
        loss.backward()
        self.optimizer.step()
        return metrics

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
                lbl_b = labels[i:i + batch_size].to(self.device)
                preds = self.model(feat_b).argmax(dim=1)
                correct += int((preds == lbl_b).sum().item())
        return correct / total
