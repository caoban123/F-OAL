"""Adapted Learning without Forgetting (LwF) Baseline for Continual Learning.

Reference:
  Li & Hoiem, "Learning without Forgetting", TPAMI 2018.
  NeurIPS 2024 F-OAL Table 2 References:
    - CIFAR-100: A_avg = 69.3%, A_last = 64.8%, F = 1.3% (Exemplar-free, Buffer = 0)
    - DTD:       A_avg = 55.8%, A_last = 48.2%, F = 12.6% (Exemplar-free, Buffer = 0)
  Section IX.13 Methodology (Adapted LwF):
    - Dynamic Classifier Head: nn.Linear(768, num_classes) expanded per task
    - Static loss weighting: L = L_CE(active classes) + 1.0 * L_KD(old classes, T = 2.0)
    - Optimizer: AdamW (lr = 0.001, weight_decay = 1e-4)
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LwFClassifier(nn.Module):
    """Linear classifier head with incremental output expansion and distillation support."""

    def __init__(self, in_features: int, num_classes: int = 10):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def expand_classes(self, new_total_classes: int, device: torch.device | str = "cpu") -> None:
        """Expand output logits to accommodate newly introduced classes while preserving learned weights."""
        if new_total_classes <= self.num_classes:
            return

        old_weight = self.fc.weight.data
        old_bias = self.fc.bias.data
        new_fc = nn.Linear(self.in_features, new_total_classes, bias=True).to(device)

        with torch.no_grad():
            new_fc.weight.data[: self.num_classes] = old_weight
            new_fc.bias.data[: self.num_classes] = old_bias

        self.fc = new_fc
        self.num_classes = new_total_classes


class LwFAgent:
    """Adapted Learning without Forgetting agent operating on extracted representations."""

    def __init__(
        self,
        feature_dim: int,
        initial_classes: int = 10,
        learning_rate: float = 0.001,
        temperature: float = 2.0,
        distill_lambda: float = 1.0,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.feature_dim = feature_dim
        self.num_classes = initial_classes
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.distill_lambda = distill_lambda
        self.weight_decay = weight_decay
        self.device = torch.device(device)

        self.model = LwFClassifier(feature_dim, initial_classes).to(self.device)
        self.prev_model: Optional[LwFClassifier] = None
        self.prev_num_classes: int = 0
        self._init_optimizer()

    def _init_optimizer(self) -> None:
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def before_task(self, task_id: int, total_seen_classes: int) -> None:
        """Prepare agent for the next task: freeze previous snapshot and expand output head."""
        if task_id > 0:
            self.prev_model = copy.deepcopy(self.model)
            self.prev_model.eval()
            for p in self.prev_model.parameters():
                p.requires_grad = False
            self.prev_num_classes = self.num_classes

            self.model.expand_classes(total_seen_classes, device=self.device)
            self.num_classes = total_seen_classes
            self._init_optimizer()
        else:
            self.prev_model = None
            self.prev_num_classes = 0

    def compute_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """Compute combined Cross-Entropy (new classes) and Knowledge Distillation (old classes) loss."""
        features = features.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(features)

        ce_loss = F.cross_entropy(logits, labels)
        total_loss = ce_loss
        distill_loss_val = 0.0

        if self.prev_model is not None and self.prev_num_classes > 0:
            with torch.no_grad():
                prev_logits = self.prev_model(features)[:, : self.prev_num_classes]
                prev_soft_targets = F.softmax(prev_logits / self.temperature, dim=1)

            curr_old_logits = logits[:, : self.prev_num_classes]
            curr_log_soft = F.log_softmax(curr_old_logits / self.temperature, dim=1)

            distill_loss = F.kl_div(curr_log_soft, prev_soft_targets, reduction="batchmean") * (
                self.temperature ** 2
            )
            distill_loss = torch.clamp(distill_loss, min=0.0)
            total_loss = ce_loss + self.distill_lambda * distill_loss
            distill_loss_val = float(distill_loss.item())

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_ce": float(ce_loss.item()),
            "loss_distill": distill_loss_val,
        }
        return total_loss, metrics

    def train_batch(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Perform one optimization step on a mini-batch."""
        self.model.train()
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(features, labels)
        loss.backward()
        self.optimizer.step()
        return metrics

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Produce predicted class indices."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features.to(self.device))
            return logits.argmax(dim=1).cpu()

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor, batch_size: int = 128) -> float:
        """Evaluate accuracy on provided feature representations."""
        self.model.eval()
        correct = 0
        total = len(labels)
        if total == 0:
            return float("nan")

        with torch.no_grad():
            for i in range(0, total, batch_size):
                feat_b = features[i : i + batch_size].to(self.device)
                lbl_b = labels[i : i + batch_size]
                preds = self.model(feat_b).argmax(dim=1).cpu()
                correct += int((preds == lbl_b).sum())
        return correct / total
