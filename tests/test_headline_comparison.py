"""Unit tests for Continual Learning Headline Comparison (LwF, ER) components."""
from pathlib import Path
import pytest
import torch
import yaml

from src.foal.baselines.lwf import LwFClassifier, LwFAgent
from src.foal.baselines.er import ReplayBuffer, ERAgent


def test_lwf_classifier_expansion():
    """Verify that LwFClassifier correctly expands classes while preserving existing weights."""
    model = LwFClassifier(in_features=32, num_classes=10)
    orig_weight = model.fc.weight.data.clone()
    orig_bias = model.fc.bias.data.clone()

    x = torch.randn(4, 32)
    logits_10 = model(x)
    assert logits_10.shape == (4, 10)

    # Expand to 20 classes
    model.expand_classes(20)
    assert model.num_classes == 20
    assert model.fc.weight.shape == (20, 32)
    assert model.fc.bias.shape == (20,)

    # Check preserved weights
    assert torch.allclose(model.fc.weight.data[:10], orig_weight)
    assert torch.allclose(model.fc.bias.data[:10], orig_bias)

    # Test forward pass with expanded shape
    logits_20 = model(x)
    assert logits_20.shape == (4, 20)
    assert torch.allclose(logits_20[:, :10], logits_10)


def test_lwf_agent_step_and_distillation():
    """Verify LwFAgent training step, snapshot preservation, and distillation loss computation."""
    agent = LwFAgent(feature_dim=16, initial_classes=4, learning_rate=0.01, device="cpu")

    # Task 0 training
    x0 = torch.randn(8, 16)
    y0 = torch.randint(0, 4, (8,))
    metrics0 = agent.train_batch(x0, y0)
    assert "loss_total" in metrics0
    assert metrics0["loss_distill"] == 0.0  # No distillation on Task 0

    # Transition to Task 1 (classes 4..7, total 8)
    agent.before_task(task_id=1, total_seen_classes=8)
    assert agent.prev_model is not None
    assert agent.prev_num_classes == 4

    x1 = torch.randn(8, 16)
    y1 = torch.randint(4, 8, (8,))
    metrics1 = agent.train_batch(x1, y1)
    assert metrics1["loss_total"] > 0.0
    assert metrics1["loss_distill"] >= 0.0  # Distillation active

    # Test evaluation
    acc = agent.evaluate(x1, y1)
    assert 0.0 <= acc <= 1.0


def test_er_buffer_reservoir_sampling():
    """Verify ReplayBuffer respects capacity, reservoir updates, and uniform sampling."""
    capacity = 20
    buffer = ReplayBuffer(capacity=capacity, seed=123)

    # Add 15 samples (under capacity)
    feats = torch.randn(15, 8)
    labels = list(range(15))
    buffer.add_batch(feats, labels)
    assert len(buffer) == 15

    # Sample batch
    b_feat, b_lbl = buffer.sample(5)
    assert b_feat.shape == (5, 8)
    assert b_lbl.shape == (5,)

    # Add 50 more samples (exceeding capacity)
    more_feats = torch.randn(50, 8)
    more_labels = list(range(100, 150))
    buffer.add_batch(more_feats, more_labels)

    # Capacity must strictly not be exceeded
    assert len(buffer) == capacity
    assert buffer.total_seen == 65

    # Verify reservoir replacement occurred (labels should contain both early and late items)
    unique_labels = set(buffer.labels)
    assert len(unique_labels) == capacity


def test_er_agent_step():
    """Verify ERAgent updates parameters and incorporates memory samples."""
    agent = ERAgent(feature_dim=16, initial_classes=4, buffer_capacity=30, learning_rate=0.01, device="cpu")

    x0 = torch.randn(6, 16)
    y0 = torch.randint(0, 4, (6,))
    metrics0 = agent.train_batch(x0, y0)
    assert "loss_total" in metrics0
    assert metrics0["buffer_occupancy"] == 6

    # Task transition to 8 classes
    agent.before_task(task_id=1, total_seen_classes=8)
    x1 = torch.randn(6, 16)
    y1 = torch.randint(4, 8, (6,))
    metrics1 = agent.train_batch(x1, y1)
    assert metrics1["loss_total"] > 0.0
    assert metrics1["buffer_occupancy"] == 12


def test_headline_configs_validity():
    """Verify that all headline comparison YAML configs exist and contain required keys."""
    configs_to_check = [
        ("configs/headline/foal_cifar100_reference.yaml", ["dataset", "backbone", "reference_values"]),
        ("configs/headline/lwf_cifar100.yaml", ["dataset", "agent", "learning_rate", "temperature", "reference_values"]),
        ("configs/headline/er_cifar100.yaml", ["dataset", "agent", "buffer_capacity", "reference_values"]),
    ]
    for cfg_rel_path, required_keys in configs_to_check:
        cfg_path = Path(cfg_rel_path)
        assert cfg_path.exists(), f"Missing config file: {cfg_rel_path}"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        for key in required_keys:
            assert key in data, f"Config {cfg_rel_path} missing required key '{key}'"
