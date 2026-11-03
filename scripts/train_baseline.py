"""Train Continual Learning Baselines (LwF, ER) on CIFAR-100 with F-OAL protocol parity.

Usage:
    python scripts/train_baseline.py --config configs/headline/lwf_cifar100.yaml
    python scripts/train_baseline.py --config configs/headline/er_cifar100.yaml
    python scripts/train_baseline.py --config configs/headline/lwf_cifar100.yaml --max-tasks 2 --limit-train-per-task 20 --limit-test-per-task 10
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.foal.datasets import (
    load_cifar100,
    make_class_order,
    make_task_specs,
    IndexedImageDataset,
    cifar100_transforms,
    limit_indices_per_task,
)
from src.foal.backbone import FrozenViT
from src.foal.baselines.lwf import LwFAgent
from src.foal.baselines.er import ERAgent
from src.foal.metrics import summary_from_task_matrix, format_summary_percentages
from src.foal.environment import get_environment_info


def resolve_device(requested_device: str | None, config_device: str) -> str:
    target = (requested_device or config_device).lower()
    if target == "cuda" or target.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{target}' was requested, but torch.cuda.is_available() is False."
            )
        return target
    elif target == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return target


def extract_task_features(
    encoder: FrozenViT,
    dataset: IndexedImageDataset,
    batch_size: int,
    device: str,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Extract features using FrozenViT."""
    t0 = time.monotonic()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    feat_chunks = []
    label_chunks = []
    with torch.no_grad():
        for x, y in loader:
            feats = encoder(x.to(device)).cpu()
            feat_chunks.append(feats)
            label_chunks.append(y.cpu())
    feats_all = torch.cat(feat_chunks, dim=0) if feat_chunks else torch.empty((0, encoder.feature_dim))
    labels_all = torch.cat(label_chunks, dim=0) if label_chunks else torch.empty((0,), dtype=torch.long)
    elapsed = round(time.monotonic() - t0, 4)
    return feats_all, labels_all, elapsed


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # Config resolution
    agent_type = str(getattr(args, "agent", None) or cfg.get("agent", "lwf")).lower()
    seed = int(getattr(args, "seed", None) or cfg.get("seed", 42))
    class_order_seed = int(getattr(args, "class_order_seed", None) or cfg.get("class_order_seed", seed))
    max_tasks = int(getattr(args, "max_tasks", None) or cfg.get("max_tasks", 10))
    limit_train = getattr(args, "limit_train_per_task", None)
    if limit_train is None:
        limit_train = cfg.get("limit_train_per_task")
    if limit_train is not None:
        limit_train = int(limit_train)

    limit_test = getattr(args, "limit_test_per_task", None)
    if limit_test is None:
        limit_test = cfg.get("limit_test_per_task")
    if limit_test is not None:
        limit_test = int(limit_test)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_str = resolve_device(args.device, cfg.get("device", "auto"))
    print(f"[{agent_type.upper()}] Running baseline with device={device_str}, seed={seed}, max_tasks={max_tasks}")

    # Dataset & ordering
    raw_train, raw_test = load_cifar100(root=cfg.get("data_dir", "./data"))
    labels_train = [int(item["fine_label"]) for item in raw_train]
    labels_test = [int(item["fine_label"]) for item in raw_test]

    explicit_order = cfg.get("explicit_class_order")
    order = make_class_order(num_classes=100, seed=class_order_seed, explicit=explicit_order)
    classes_per_task = int(cfg.get("classes_per_task", 10))
    specs = make_task_specs(
        train_labels=labels_train,
        test_labels=labels_test,
        order=order,
        classes_per_task=classes_per_task,
    )

    # Transforms & Backbone
    prep = cfg.get("preprocessing", {})
    interp = prep.get("interpolation", "bilinear")
    norm = bool(prep.get("normalize", False))
    transform = cifar100_transforms(size=224, interpolation=interp, normalize=norm)

    backbone_name = cfg.get("backbone", "vit_base_patch16_224.augreg2_in21k_ft_in1k")
    feature_fusion = bool(cfg.get("feature_fusion", True))
    encoder = FrozenViT(name=backbone_name, pretrained=True, feature_fusion=feature_fusion).to(device_str)
    feat_dim = encoder.feature_dim
    print(f"[{agent_type.upper()}] FrozenViT initialized: {backbone_name}, feature_fusion={feature_fusion}, dim={feat_dim}")

    # Agent setup
    lr = float(cfg.get("learning_rate", 0.001))
    batch_size = int(cfg.get("batch_size", 10))
    if agent_type == "lwf":
        temperature = float(cfg.get("temperature", 2.0))
        distill_lambda = float(cfg.get("distill_lambda", 1.0))
        agent = LwFAgent(
            feature_dim=feat_dim,
            initial_classes=classes_per_task,
            learning_rate=lr,
            temperature=temperature,
            distill_lambda=distill_lambda,
            device=device_str,
        )
    elif agent_type == "er":
        buffer_capacity = int(cfg.get("buffer_capacity", 5000))
        agent = ERAgent(
            feature_dim=feat_dim,
            initial_classes=classes_per_task,
            buffer_capacity=buffer_capacity,
            learning_rate=lr,
            seed=seed,
            device=device_str,
        )
    elif agent_type in ["official_lwf", "official-lwf"]:
        from src.foal.baselines.official_lwf import OfficialLwFAgent
        temperature = float(cfg.get("temperature", 2.0))
        momentum = float(cfg.get("momentum", 0.0))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        agent = OfficialLwFAgent(
            feature_dim=feat_dim,
            total_classes=int(cfg.get("num_classes", 100)),
            learning_rate=lr,
            temperature=temperature,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device_str,
        )
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'. Supported: 'lwf', 'er', 'official_lwf'.")

    # Evaluation matrix R[i, j]: performance on task j after training on task i
    task_matrix: list[list[float]] = []
    stage_accuracies: list[float] = []

    total_training_seconds = 0.0
    active_specs = specs[:max_tasks]

    # Pre-extract test features for all tasks to make evaluation instant
    print("Preparing test datasets...")
    test_task_datasets = []
    for t_id, spec in enumerate(active_specs):
        # Build test dataset for this specific task
        task_orig_labels = set(spec.new_original_labels)
        task_test_ids = [idx for idx in spec.seen_test_indices if labels_test[idx] in task_orig_labels]
        if limit_test is not None:
            task_test_ids = limit_indices_per_task(task_test_ids, limit_test, seed + t_id)
        # Mapping from original label to incremental label
        seen_order = order[: (t_id + 1) * classes_per_task]
        lbl_map = {lbl: idx for idx, lbl in enumerate(seen_order)}
        t_ds = IndexedImageDataset(raw_test, task_test_ids, lbl_map, transform)
        t_feats, t_lbls, _ = extract_task_features(encoder, t_ds, batch_size=64, device=device_str)
        test_task_datasets.append((t_feats, t_lbls))

    print(f"\n--- Starting Continual Learning Stream ({len(active_specs)} Tasks) ---")
    for task_id, spec in enumerate(active_specs):
        t_start = time.monotonic()
        total_seen_classes = spec.total_seen_classes
        agent.before_task(task_id, total_seen_classes)

        # Prepare train data
        train_ids = list(spec.train_indices)
        if limit_train is not None:
            train_ids = limit_indices_per_task(train_ids, limit_train, seed + task_id)

        seen_order = order[:total_seen_classes]
        lbl_map = {lbl: idx for idx, lbl in enumerate(seen_order)}
        train_ds = IndexedImageDataset(raw_train, train_ids, lbl_map, transform)

        # Extract features
        train_feats, train_lbls, _ = extract_task_features(encoder, train_ds, batch_size=64, device=device_str)

        # Train single pass on streaming mini-batches (batch_size 10)
        n_samples = len(train_lbls)
        for b_start in range(0, n_samples, batch_size):
            b_feat = train_feats[b_start:b_start + batch_size]
            b_lbl = train_lbls[b_start:b_start + batch_size]
            agent.train_batch(b_feat, b_lbl)

        if hasattr(agent, "after_task"):
            agent.after_task()

        task_elapsed = time.monotonic() - t_start
        total_training_seconds += task_elapsed

        # Evaluate on all tasks seen so far: j = 0 .. task_id
        row_accuracies = []
        for j in range(task_id + 1):
            eval_feats, eval_lbls = test_task_datasets[j]
            # Since model outputs up to total_seen_classes, compute accuracy
            acc_j = agent.evaluate(eval_feats, eval_lbls, batch_size=128)
            row_accuracies.append(acc_j)

        task_matrix.append(row_accuracies)
        stage_acc = float(np.mean(row_accuracies))
        stage_accuracies.append(stage_acc)
        print(f"Task {task_id:02d} Completed ({task_elapsed:.2f}s) | Stage Acc: {stage_acc * 100:.2f}% | Seen Tasks: {[round(a*100, 2) for a in row_accuracies]}")

    # Compute continual learning metrics
    summary = summary_from_task_matrix(task_matrix)
    summary['a_avg_pct'] = float(summary['average_incremental_accuracy'] * 100)
    summary['a_last_pct'] = float(summary['last_accuracy'] * 100)
    summary['f_past_pct'] = float(summary['forgetting_past_tasks'] * 100)
    summary['f_code_pct'] = float(summary['forgetting_official_code'] * 100)
    formatted = format_summary_percentages(summary)

    print("\n" + "=" * 60)
    print(f"FINAL BENCHMARK RESULTS: {agent_type.upper()}")
    print("=" * 60)
    print(f"Average Accuracy (A_avg):     {summary['a_avg_pct']:.4f}% (Paper ref: {'69.3%' if agent_type=='lwf' else '90.1%'})")
    print(f"Last Task Accuracy (A_last): {summary['a_last_pct']:.4f}% (Paper ref: {'64.8%' if agent_type=='lwf' else '84.6%'})")
    print(f"Forgetting Rate (F_past):    {summary['f_past_pct']:.4f}% (Paper ref: {'1.3%' if agent_type=='lwf' else '20.7%'})")
    print(f"Total Stream Time:           {total_training_seconds:.2f}s")
    print("=" * 60)

    # Save output artifacts
    output_dir = Path(cfg.get("output_dir", f"runs/headline/{agent_type}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result_payload = {
        "agent": agent_type,
        "config": cfg,
        "metrics_summary": summary,
        "formatted_summary": formatted,
        "task_matrix": task_matrix,
        "stage_accuracies": stage_accuracies,
        "total_training_seconds": total_training_seconds,
        "environment": get_environment_info(),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    with open(output_dir / "task_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in task_matrix:
            writer.writerow([f"{v:.6f}" for v in row])

    print(f"Artifacts saved to {output_dir}")
    return result_payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Continual Learning Baseline (LwF, ER)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--agent", type=str, choices=["lwf", "er"], default=None, help="Agent override")
    p.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    p.add_argument("--seed", type=int, default=None, help="Random seed override")
    p.add_argument("--max-tasks", type=int, default=None, help="Cap number of tasks")
    p.add_argument("--limit-train-per-task", type=int, default=None, help="Limit training samples per task")
    p.add_argument("--limit-test-per-task", type=int, default=None, help="Limit test samples per task")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-offline", action="store_true", help="Log W&B offline")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_baseline(args)
