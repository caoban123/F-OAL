"""Train Continual Learning Baselines (Adapted LwF, Official-Protocol LwF, ER) on DTD or CIFAR-100.

Follows Section IX (Sections 12-15) of docs/CIFAR100_COMPLETE_WORK_REPORT.pdf with strict F-OAL protocol parity:
  - Frozen ViT-B/16 Feature Fusion representations (D = 768)
  - Bilinear resize (224x224), Raw [0, 1] preprocessing
  - One-pass streaming mini-batches (batch_size = 10)
  - Native Weights & Biases (wandb) tracking and feature caching

Usage:
    python scripts/train_baseline.py --config configs/headline/lwf_dtd.yaml --wandb
    python scripts/train_baseline.py --config configs/headline/lwf_dtd_official_protocol.yaml --wandb
    python scripts/train_baseline.py --config configs/headline/er_dtd.yaml --wandb
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.foal.backbone import FrozenViT
from src.foal.baselines.er import ERAgent
from src.foal.baselines.lwf import LwFAgent
from src.foal.baselines.official_lwf import OfficialLwFAgent
from src.foal.datasets import (
    IndexedImageDataset,
    cifar100_transforms,
    limit_indices_per_task,
    load_cifar100,
    make_class_order,
    make_task_specs,
)
from src.foal.dtd_datasets import (
    DTD_CLASS_NAMES,
    load_dtd,
    make_dtd_class_order,
    make_dtd_task_specs,
)
from src.foal.environment import get_environment_info
from src.foal.metrics import format_summary_percentages, summary_from_task_matrix
from scripts.train import compute_task_split_fingerprint


def resolve_device(requested_device: str | None, config_device: str) -> str:
    """Resolve compute device with strict guard against silent GPU-to-CPU fallback."""
    target = (requested_device or config_device).lower()
    if target == "cuda" or target.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{target}' was requested, but torch.cuda.is_available() is False."
            )
        return target
    if target == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return target


def resolve_seed(cli_value: Any, config_value: Any, default_seed: int) -> int:
    """Pick the seed with precedence CLI value > config value > default.

    Args:
        cli_value: Seed passed on the command line, or None.
        config_value: Seed read from the YAML config, or None.
        default_seed: Fallback seed.

    Returns:
        The resolved integer seed.
    """
    if cli_value is not None:
        return int(cli_value)
    if config_value is not None:
        return int(config_value)
    return int(default_seed)


def get_resolved_pretrained_cfg(encoder: FrozenViT) -> dict[str, Any]:
    """Extract JSON-serializable pretrained backbone configuration from timm."""
    info: dict[str, Any] = {}
    try:
        pcfg = getattr(encoder.network, "pretrained_cfg", None) or {}
        for k in (
            "hf_hub_id",
            "tag",
            "architecture",
            "url",
            "input_size",
            "interpolation",
            "mean",
            "std",
            "num_classes",
        ):
            if k in pcfg and pcfg[k] is not None:
                val = pcfg[k]
                info[k] = list(val) if isinstance(val, tuple) else val
    except Exception:
        pass
    return info


def extract_task_features(
    encoder: FrozenViT,
    dataset: IndexedImageDataset,
    batch_size: int,
    device: str,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Extract features using FrozenViT without gradients."""
    t0 = time.monotonic()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    feat_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            feats = encoder(x.to(device)).cpu()
            feat_chunks.append(feats)
            label_chunks.append(y.cpu())
    feats_all = torch.cat(feat_chunks, dim=0) if feat_chunks else torch.empty((0, encoder.feature_dim))
    labels_all = torch.cat(label_chunks, dim=0) if label_chunks else torch.empty((0,), dtype=torch.long)
    elapsed = round(time.monotonic() - t0, 4)
    return feats_all, labels_all, elapsed


def extract_or_load_baseline_features(
    *,
    cache_dir: Optional[Path],
    cache_keys: Sequence[str],
    encoder: FrozenViT,
    dataset: IndexedImageDataset,
    batch_size: int,
    num_workers: int,
    device: str,
    preprocessing: dict[str, Any],
    dataset_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Load validated feature cache from disk if available, else extract and save."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        indices_sha = hashlib.sha256(json.dumps(dataset.indices).encode("utf-8")).hexdigest()

        for key in cache_keys:
            cache_file = cache_dir / f"{key}.pt"
            if cache_file.exists():
                saved = torch.load(cache_file, map_location="cpu", weights_only=False)
                valid = (
                    saved.get("indices_sha256") == indices_sha
                    and saved.get("backbone_name") == encoder.model_name
                    and saved.get("feature_fusion") == encoder.feature_fusion
                    and saved.get("pretrained_tag") == getattr(encoder, "pretrained_tag", None)
                    and saved.get("dataset_name") == dataset_name
                    and saved.get("preprocessing") == preprocessing
                    and len(saved.get("features", [])) == len(dataset)
                )
                if valid:
                    return (
                        saved["features"],
                        saved["labels"],
                        float(saved.get("extraction_seconds", 0.0)),
                        True,
                    )

        feats_cpu, labels_cpu, elapsed = extract_task_features(
            encoder=encoder,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )
        primary_cache_file = cache_dir / f"{cache_keys[0]}.pt"
        torch.save(
            {
                "backbone_name": encoder.model_name,
                "pretrained_tag": getattr(encoder, "pretrained_tag", "unknown"),
                "feature_fusion": encoder.feature_fusion,
                "dataset_name": dataset_name,
                "preprocessing": preprocessing,
                "indices_sha256": indices_sha,
                "features_sha256": hashlib.sha256(feats_cpu.contiguous().numpy().tobytes()).hexdigest(),
                "extraction_seconds": elapsed,
                "features": feats_cpu,
                "labels": labels_cpu,
            },
            primary_cache_file,
        )
        return feats_cpu, labels_cpu, elapsed, False

    feats_cpu, labels_cpu, elapsed = extract_task_features(
        encoder=encoder,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    return feats_cpu, labels_cpu, elapsed, False


def _load_legacy_local_dtd_cache(
    cache_root: Path,
    orig_indices: Sequence[int],
    orig_labels: Sequence[int],
    lbl_map: Dict[int, int],
    is_train: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Optional offline helper to read pre-extracted tensors from results/dtd_feature_cache/."""
    split_name = "train" if is_train else "test"
    cache_path = cache_root / f"{split_name}_augreg2_bilinear_raw_ff.pt"
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached_feats = payload["features"]
    cached_labels = payload["labels"].tolist()
    cached_indices = payload["sample_indices"].tolist()

    idx_to_pos = {int(ds_idx): pos for pos, ds_idx in enumerate(cached_indices)}
    if all(int(idx) in idx_to_pos for idx in orig_indices):
        positions = [idx_to_pos[int(idx)] for idx in orig_indices]
        feats = cached_feats[positions]
        mapped_labels = torch.tensor([lbl_map[int(orig_labels[idx])] for idx in orig_indices], dtype=torch.long)
        return feats, mapped_labels

    # Fallback: match per-class count from legacy cache when train indices came from a different RNG permutation
    class_to_positions: Dict[int, List[int]] = {}
    for pos, raw_lbl in enumerate(cached_labels):
        class_to_positions.setdefault(int(raw_lbl), []).append(pos)
    class_cursor: Dict[int, int] = {}
    selected_positions: List[int] = []
    mapped_lbls: List[int] = []
    for idx in orig_indices:
        raw_lbl = int(orig_labels[idx])
        cur = class_cursor.get(raw_lbl, 0)
        avail = class_to_positions.get(raw_lbl, [])
        if cur >= len(avail):
            return None
        selected_positions.append(avail[cur])
        class_cursor[raw_lbl] = cur + 1
        mapped_lbls.append(lbl_map[raw_lbl])
    return cached_feats[selected_positions], torch.tensor(mapped_lbls, dtype=torch.long)


def _aggregate_batch_metrics(batch_metrics_list: Sequence[dict[str, float]]) -> dict[str, float]:
    """Compute per-task average of batch-level training metrics for W&B logging."""
    if not batch_metrics_list:
        return {}
    agg: dict[str, float] = {}
    keys = set().union(*(m.keys() for m in batch_metrics_list))
    for k in sorted(keys):
        vals = [float(m[k]) for m in batch_metrics_list if k in m and isinstance(m[k], (int, float))]
        if vals:
            if k in ("buffer_occupancy", "task_seen"):
                agg[f"train/{k}"] = vals[-1]
            else:
                agg[f"train/{k}"] = float(sum(vals) / len(vals))
    return agg


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate one continual-learning baseline (ER, LwF or official-protocol LwF).

    Uses the same frozen ViT feature pipeline and task stream as F-OAL, trains the
    baseline head online, evaluates after each task and writes matrix, metadata and
    metrics (optionally logging to W&B).

    Args:
        args: Parsed CLI arguments (config, agent type, seeds, device, output dir).

    Returns:
        Result payload dict with metrics and the task accuracy matrix.
    """
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # Configuration & seed resolution
    agent_type = str(getattr(args, "agent", None) or cfg.get("agent", "lwf")).lower().replace("-", "_")
    seed = resolve_seed(getattr(args, "seed", None), cfg.get("seed"), 42)
    class_order_seed = resolve_seed(getattr(args, "class_order_seed", None), cfg.get("class_order_seed"), seed)
    stream_seed = resolve_seed(getattr(args, "stream_seed", None), cfg.get("stream_seed"), seed)

    dataset_cfg_name = str(cfg.get("dataset", "cifar100")).lower()
    is_dtd = "dtd" in dataset_cfg_name

    default_max_tasks = 10
    max_tasks = int(getattr(args, "max_tasks", None) or cfg.get("max_tasks", default_max_tasks))

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

    # Set deterministic seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_str = resolve_device(getattr(args, "device", None), cfg.get("device", "auto"))
    print(
        f"[{agent_type.upper()}] Running baseline on dataset={dataset_cfg_name}, "
        f"device={device_str}, seed={seed}, max_tasks={max_tasks}"
    )

    # Dataset loading & task specification parity
    data_root = cfg.get("data_dir", "./data")
    if is_dtd:
        raw_train, raw_test = load_dtd(root=data_root)
        labels_train = list(raw_train.fine_label)
        labels_test = list(raw_test.fine_label)
        total_classes = int(cfg.get("total_classes", 47))
        num_classes = int(cfg.get("num_classes", 40))
        classes_per_task = int(cfg.get("classes_per_task", 4))
        samples_per_class = cfg.get("samples_per_class", 40)
        explicit_order = cfg.get("explicit_class_order")
        if explicit_order is not None:
            order = list(explicit_order)
        else:
            order = make_dtd_class_order(
                num_classes_to_use=num_classes,
                total_classes=total_classes,
                seed=class_order_seed,
            )
        specs = make_dtd_task_specs(
            train_labels=labels_train,
            test_labels=labels_test,
            order=order,
            classes_per_task=classes_per_task,
            samples_per_class=samples_per_class,
            seed=class_order_seed,
        )
        dataset_source_name = str(cfg.get("dataset_source", "tanganke/dtd"))
    else:
        raw_train, raw_test = load_cifar100(root=data_root)
        labels_train = [int(item["fine_label"]) for item in raw_train]
        labels_test = [int(item["fine_label"]) for item in raw_test]
        num_classes = int(cfg.get("num_classes", 100))
        classes_per_task = int(cfg.get("classes_per_task", 10))
        samples_per_class = None
        explicit_order = cfg.get("explicit_class_order")
        order = make_class_order(num_classes=num_classes, seed=class_order_seed, explicit=explicit_order)
        specs = make_task_specs(
            train_labels=labels_train,
            test_labels=labels_test,
            order=order,
            classes_per_task=classes_per_task,
        )
        dataset_source_name = str(cfg.get("dataset_source", "uoft-cs/cifar100"))

    if not (1 <= max_tasks <= len(specs)):
        raise ValueError(f"max_tasks must be between 1 and {len(specs)}, got {max_tasks}")
    active_specs = specs[:max_tasks]

    # Preprocessing & Backbone
    prep = cfg.get("preprocessing", {}) or {}
    interp = str(prep.get("interpolation", "bilinear")).lower()
    norm = bool(prep.get("normalize", False))
    image_size = int(prep.get("size", 224))
    mean_tuple = tuple(float(v) for v in prep.get("mean", (0.5, 0.5, 0.5)))
    std_tuple = tuple(float(v) for v in prep.get("std", (0.5, 0.5, 0.5)))
    transform = cifar100_transforms(
        size=image_size,
        mean=mean_tuple,
        std=std_tuple,
        interpolation=interp,
        normalize=norm,
    )

    backbone_name = str(cfg.get("backbone", "vit_base_patch16_224.augreg2_in21k_ft_in1k"))
    feature_fusion = bool(cfg.get("feature_fusion", True))
    use_local_dtd_cache = bool(getattr(args, "use_local_dtd_cache", False))
    local_dtd_cache_dir = Path(getattr(args, "local_dtd_cache_dir", "results/dtd_feature_cache"))

    encoder: Optional[FrozenViT] = None
    pretrained_cfg_info: dict[str, Any] = {}
    if use_local_dtd_cache and is_dtd and (local_dtd_cache_dir / "train_augreg2_bilinear_raw_ff.pt").exists():
        feat_dim = 768
        print(f"[{agent_type.upper()}] Using pre-extracted local DTD feature cache from {local_dtd_cache_dir} (dim={feat_dim})")
    else:
        encoder = FrozenViT(name=backbone_name, pretrained=True, feature_fusion=feature_fusion).to(device_str)
        feat_dim = encoder.feature_dim
        pretrained_cfg_info = get_resolved_pretrained_cfg(encoder)
        print(
            f"[{agent_type.upper()}] FrozenViT initialized: {backbone_name}, "
            f"feature_fusion={feature_fusion}, dim={feat_dim}"
        )

    # Agent setup
    lr = float(cfg.get("learning_rate", 0.001))
    batch_size = int(cfg.get("batch_size", 10))
    if agent_type == "lwf":
        temperature = float(cfg.get("temperature", 2.0))
        distill_lambda = float(cfg.get("distill_lambda", 1.0))
        weight_decay = float(cfg.get("weight_decay", 1e-4))
        agent = LwFAgent(
            feature_dim=feat_dim,
            initial_classes=classes_per_task,
            learning_rate=lr,
            temperature=temperature,
            distill_lambda=distill_lambda,
            weight_decay=weight_decay,
            device=device_str,
        )
    elif agent_type == "er":
        buffer_capacity = int(cfg.get("buffer_capacity", 5000))
        weight_decay = float(cfg.get("weight_decay", 1e-4))
        agent = ERAgent(
            feature_dim=feat_dim,
            initial_classes=classes_per_task,
            buffer_capacity=buffer_capacity,
            learning_rate=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=device_str,
        )
    elif agent_type == "official_lwf":
        temperature = float(cfg.get("temperature", 2.0))
        momentum = float(cfg.get("momentum", 0.0))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        agent = OfficialLwFAgent(
            feature_dim=feat_dim,
            total_classes=num_classes,
            learning_rate=lr,
            temperature=temperature,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device_str,
        )
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'. Supported: 'lwf', 'er', 'official_lwf'.")

    # Output dir & provenance metadata
    output_dir_override = getattr(args, "output_dir", None)
    default_out = f"runs/headline_dtd/{agent_type}" if is_dtd else f"runs/headline/{agent_type}"
    output_dir = Path(output_dir_override or cfg.get("output_dir", default_out))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = getattr(args, "run_name", None) or f"headline_{'dtd' if is_dtd else 'cifar100'}_{agent_type}_seed{seed}"
    env_info = get_environment_info()
    task_split_fp = compute_task_split_fingerprint(
        order=order,
        specs=specs,
        max_tasks=max_tasks,
        limit_train=limit_train,
        limit_test=limit_test,
        stream_seed=stream_seed,
        labels_test=labels_test,
    )

    feature_cache_dir = Path(args.feature_cache_dir) if getattr(args, "feature_cache_dir", None) else None
    prep_meta = {
        "image_size": image_size,
        "normalize": norm,
        "mean": list(mean_tuple) if norm else None,
        "std": list(std_tuple) if norm else None,
        "interpolation": interp,
    }

    metadata = {
        "paper": "F-OAL NeurIPS 2024",
        "section_methodology": "Section IX (Adapted LwF, Adapted ER, Official-Protocol LwF)",
        "dataset": "DTD" if is_dtd else "CIFAR-100",
        "dataset_source": dataset_source_name,
        "agent": agent_type,
        "run_name": run_name,
        "git_commit": env_info.get("git", {}).get("commit"),
        "seeds": {
            "base": seed,
            "class_order": class_order_seed,
            "stream": stream_seed,
        },
        "model_config": {
            "backbone": backbone_name,
            "pretrained": True,
            "pretrained_cfg": pretrained_cfg_info,
            "feature_fusion": feature_fusion,
            "feature_dim": feat_dim,
            "agent": agent_type,
            "optimizer": str(cfg.get("optimizer", "sgd" if agent_type == "official_lwf" else "adamw")),
            "learning_rate": lr,
            "batch_size": batch_size,
        },
        "training_protocol": {
            "dataset": dataset_source_name,
            "dataset_type": "dtd" if is_dtd else "cifar100",
            "num_classes": num_classes,
            "classes_per_task": classes_per_task,
            "samples_per_class": samples_per_class,
            "max_tasks": max_tasks,
            "limit_train_per_task": limit_train,
            "limit_test_per_task": limit_test,
            "batch_size": batch_size,
            "device": device_str,
            "one_pass": True,
            "feature_cache_enabled": feature_cache_dir is not None,
            "used_legacy_dtd_feature_cache": bool(use_local_dtd_cache and encoder is None),
        },
        "task_split_fingerprint": task_split_fp,
        "class_order": order,
        "class_names": [DTD_CLASS_NAMES[c] for c in order] if is_dtd else None,
        "environment": env_info,
        "config": cfg,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Native Weights & Biases integration
    logging_cfg = cfg.get("logging", {}) or {}
    wandb_run = None
    use_wandb = (
        bool(getattr(args, "wandb", False)) or bool(getattr(args, "wandb_offline", False)) or bool(logging_cfg.get("use_wandb", False))
    ) and os.environ.get("WANDB_MODE") != "disabled"
    if use_wandb:
        try:
            import wandb

            if getattr(args, "wandb_offline", False):
                os.environ["WANDB_MODE"] = "offline"

            base_url = (
                getattr(args, "wandb_base_url", None)
                or logging_cfg.get("base_url")
                or os.environ.get("WANDB_BASE_URL")
            )
            if base_url:
                os.environ["WANDB_BASE_URL"] = str(base_url)

            project = (
                getattr(args, "wandb_project", None)
                or logging_cfg.get("project")
                or "hcmus-foal-reproduction"
            )
            entity = getattr(args, "wandb_entity", None) or logging_cfg.get("entity")
            default_group = "headline_comparison_dtd" if is_dtd else "headline_comparison"
            group = getattr(args, "wandb_group", None) or logging_cfg.get("group") or default_group
            tags = [
                "dtd" if is_dtd else "cifar100",
                "headline",
                "baseline",
                agent_type,
                f"seed_{seed}",
                "section_ix",
            ]
            if getattr(args, "wandb_tags", None):
                tags.extend([t.strip() for t in str(args.wandb_tags).split(",") if t.strip()])

            wandb_run = wandb.init(
                project=project,
                entity=entity,
                group=group,
                name=run_name,
                config=metadata,
                tags=tags,
                reinit=True,
            )
        except Exception as exc:
            print(f"Warning: W&B init failed ({exc}). Continuing training without W&B.")
            wandb_run = None

    # Pre-extract test features for all tasks
    mapping = {orig: idx for idx, orig in enumerate(order)}
    bb_slug = backbone_name.replace("/", "_").replace(".", "_")
    interp_tag = "bicubic" if interp == "bicubic" else "bilinear"
    norm_tag = "norm" if norm else "raw"
    ff_tag = "ff" if feature_fusion else "noff"
    prefix_ds = "dtd_" if is_dtd else ""

    print("Preparing test datasets...")
    test_task_datasets: list[Tuple[torch.Tensor, torch.Tensor]] = []
    total_feature_extraction_seconds = 0.0

    for t_id, spec in enumerate(active_specs):
        task_orig_labels = set(spec.new_original_labels)
        task_test_ids = [idx for idx in spec.seen_test_indices if int(labels_test[idx]) in task_orig_labels]
        if limit_test is not None:
            task_test_ids = limit_indices_per_task(task_test_ids, limit_test, stream_seed + t_id)

        cached_loaded = None
        if encoder is None:
            cached_loaded = _load_legacy_local_dtd_cache(
                local_dtd_cache_dir, task_test_ids, labels_test, mapping, is_train=False
            )
        if cached_loaded is not None:
            t_feats, t_lbls = cached_loaded
        else:
            if encoder is None:
                encoder = FrozenViT(name=backbone_name, pretrained=True, feature_fusion=feature_fusion).to(device_str)
            t_ds = IndexedImageDataset(raw_test, task_test_ids, mapping, transform)
            test_keys = [
                f"test_{prefix_ds}{bb_slug}_{interp_tag}_{norm_tag}_{ff_tag}_c{class_order_seed}_s{stream_seed}_t{t_id + 1:02d}_te{limit_test or 'full'}",
                f"test_{bb_slug}_{interp_tag}_{norm_tag}_{ff_tag}_c{class_order_seed}_s{stream_seed}_t{t_id + 1:02d}_te{limit_test or 'full'}",
            ]
            t_feats, t_lbls, ext_sec, _ = extract_or_load_baseline_features(
                cache_dir=feature_cache_dir,
                cache_keys=test_keys,
                encoder=encoder,
                dataset=t_ds,
                batch_size=64,
                num_workers=0,
                device=device_str,
                preprocessing=prep_meta,
                dataset_name=dataset_source_name,
            )
            total_feature_extraction_seconds += ext_sec
        test_task_datasets.append((t_feats, t_lbls))

    # Continual learning stream
    task_matrix: list[list[float]] = []
    stage_accuracies: list[float] = []
    total_training_seconds = 0.0
    total_train_and_eval_seconds = 0.0
    samples_seen_total = 0
    peak_gpu_memory_mb = 0.0
    if device_str.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    stream_start = time.monotonic()
    print(f"\n--- Starting Continual Learning Stream ({len(active_specs)} Tasks) ---")
    try:
        for task_id, spec in enumerate(active_specs):
            t_start = time.monotonic()
            total_seen_classes = spec.total_seen_classes
            agent.before_task(task_id, total_seen_classes)

            train_ids = list(spec.train_indices)
            if limit_train is not None:
                train_ids = limit_indices_per_task(train_ids, limit_train, stream_seed + task_id)

            cached_loaded = None
            if encoder is None:
                cached_loaded = _load_legacy_local_dtd_cache(
                    local_dtd_cache_dir, train_ids, labels_train, mapping, is_train=True
                )
            if cached_loaded is not None:
                train_feats, train_lbls = cached_loaded
            else:
                if encoder is None:
                    encoder = FrozenViT(name=backbone_name, pretrained=True, feature_fusion=feature_fusion).to(
                        device_str
                    )
                train_ds = IndexedImageDataset(raw_train, train_ids, mapping, transform)
                train_keys = [
                    f"train_{prefix_ds}{bb_slug}_{interp_tag}_{norm_tag}_{ff_tag}_c{class_order_seed}_s{stream_seed}_t{task_id + 1:02d}_tr{limit_train or 'full'}",
                    f"train_{bb_slug}_{interp_tag}_{norm_tag}_{ff_tag}_c{class_order_seed}_s{stream_seed}_t{task_id + 1:02d}_tr{limit_train or 'full'}",
                ]
                train_feats, train_lbls, ext_sec, _ = extract_or_load_baseline_features(
                    cache_dir=feature_cache_dir,
                    cache_keys=train_keys,
                    encoder=encoder,
                    dataset=train_ds,
                    batch_size=64,
                    num_workers=0,
                    device=device_str,
                    preprocessing=prep_meta,
                    dataset_name=dataset_source_name,
                )
                total_feature_extraction_seconds += ext_sec

            # Train single-pass on shuffled streaming mini-batches matching train.py/train_dtd.py
            t_opt_start = time.monotonic()
            gen = torch.Generator().manual_seed(stream_seed + task_id)
            feat_ds = TensorDataset(train_feats, train_lbls)
            stream_loader = DataLoader(
                feat_ds,
                batch_size=batch_size,
                shuffle=True,
                generator=gen,
                drop_last=False,
                num_workers=0,
            )
            batch_logs: list[dict[str, float]] = []
            for b_feat, b_lbl in stream_loader:
                b_metrics = agent.train_batch(b_feat, b_lbl)
                if isinstance(b_metrics, dict):
                    batch_logs.append(b_metrics)

            samples_seen_total += len(train_lbls)

            if hasattr(agent, "after_task"):
                agent.after_task()

            # Evaluate on all seen tasks j = 0 .. task_id
            row_accuracies: list[float] = []
            for j in range(task_id + 1):
                eval_feats, eval_lbls = test_task_datasets[j]
                acc_j = agent.evaluate(eval_feats, eval_lbls, batch_size=128)
                row_accuracies.append(acc_j)

            total_train_and_eval_seconds += time.monotonic() - t_opt_start
            task_elapsed = time.monotonic() - t_start
            total_training_seconds += task_elapsed

            if device_str.startswith("cuda"):
                current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, current_peak)

            task_matrix.append(row_accuracies)
            stage_acc = float(np.mean(row_accuracies))
            stage_accuracies.append(stage_acc)
            running_avg_acc = float(np.mean(stage_accuracies))

            print(
                f"Task {task_id + 1:02d}/{max_tasks:02d} Completed ({task_elapsed:.2f}s) | "
                f"Stage Acc: {stage_acc * 100:.2f}% | "
                f"Seen Tasks: {[round(a * 100, 2) for a in row_accuracies]}"
            )

            if wandb_run:
                try:
                    step_metrics: dict[str, Any] = {
                        "task": task_id + 1,
                        "stage_accuracy": stage_acc,
                        "stage_accuracy_pct": stage_acc * 100.0,
                        "average_incremental_accuracy_so_far": running_avg_acc,
                        "average_incremental_accuracy_so_far_pct": running_avg_acc * 100.0,
                        "diagonal_new_task_accuracy_pct": row_accuracies[-1] * 100.0,
                        "samples_seen": samples_seen_total,
                        "duration_seconds": round(time.monotonic() - stream_start, 3),
                    }
                    step_metrics.update(_aggregate_batch_metrics(batch_logs))
                    for j, acc_val in enumerate(row_accuracies):
                        step_metrics[f"task_acc/task_{j + 1}"] = acc_val
                        step_metrics[f"task_acc_pct/task_{j + 1}"] = acc_val * 100.0
                    wandb_run.log(step_metrics, step=task_id + 1)
                except Exception as exc:
                    print(f"Warning: W&B step log error: {exc}")

        # Compute continual learning metrics
        summary = summary_from_task_matrix(task_matrix)
        summary["a_avg_pct"] = float(summary["average_incremental_accuracy"] * 100.0)
        summary["a_last_pct"] = float(summary["last_accuracy"] * 100.0)
        summary["f_past_pct"] = float(summary["forgetting_past_tasks"] * 100.0)
        summary["f_code_pct"] = float(summary["forgetting_official_code"] * 100.0)
        summary["total_duration_seconds"] = round(time.monotonic() - stream_start, 4)
        summary["feature_extraction_seconds"] = round(total_feature_extraction_seconds, 4)
        summary["train_and_eval_seconds"] = round(total_train_and_eval_seconds, 4)
        if device_str.startswith("cuda"):
            summary["peak_gpu_memory_mb"] = round(peak_gpu_memory_mb, 2)
            summary["peak_gpu_memory_gb"] = round(peak_gpu_memory_mb / 1024.0, 3)

        formatted = format_summary_percentages(summary)

        paper_ref = (cfg.get("reference_values", {}) or {}).get("paper_reported", {})
        ref_a_avg = f"{paper_ref.get('a_avg_pct', 55.8 if is_dtd and 'lwf' in agent_type else (73.2 if is_dtd else 69.3))}%"
        ref_a_last = f"{paper_ref.get('a_last_pct', 48.2 if is_dtd and 'lwf' in agent_type else (68.1 if is_dtd else 64.8))}%"
        ref_f = f"{paper_ref.get('forgetting_pct', 12.6 if is_dtd and 'lwf' in agent_type else (18.5 if is_dtd else 1.3))}%"

        print("\n" + "=" * 65)
        print(f"FINAL BENCHMARK RESULTS ({'DTD' if is_dtd else 'CIFAR-100'}): {agent_type.upper()}")
        print("=" * 65)
        print(f"Average Accuracy (A_avg):     {summary['a_avg_pct']:.4f}% (Paper ref: {ref_a_avg})")
        print(f"Last Task Accuracy (A_last): {summary['a_last_pct']:.4f}% (Paper ref: {ref_a_last})")
        print(f"Forgetting (F_past):         {summary['f_past_pct']:.4f}% (Paper ref: {ref_f})")
        print(f"Forgetting (F_code):         {summary['f_code_pct']:.4f}%")
        print(f"Total Stream Time:           {total_training_seconds:.2f}s")
        print("=" * 65)

        # Save output artifacts
        result_payload = {
            "dataset": "DTD" if is_dtd else "CIFAR-100",
            "agent": agent_type,
            "run_name": run_name,
            "config": cfg,
            "metadata": metadata,
            "metrics_summary": summary,
            "formatted_summary": formatted,
            "task_matrix": task_matrix,
            "stage_accuracies": stage_accuracies,
            "total_training_seconds": total_training_seconds,
            "environment": env_info,
        }
        (output_dir / "summary.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        metrics_payload = {"task_matrix": task_matrix, **summary}
        (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        (output_dir / "matrix.json").write_text(json.dumps(task_matrix, indent=2), encoding="utf-8")

        with (output_dir / "task_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in task_matrix:
                writer.writerow([f"{v:.6f}" for v in row])

        with (output_dir / "task_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "task", "accuracy"])
            for stage, row in enumerate(task_matrix, start=1):
                for t_idx, val in enumerate(row, start=1):
                    writer.writerow([stage, t_idx, val])

        if wandb_run:
            try:
                import wandb

                summary_payload = {
                    "summary/average_incremental_accuracy": summary["average_incremental_accuracy"],
                    "summary/average_incremental_accuracy_pct": summary["a_avg_pct"],
                    "summary/last_accuracy": summary["last_accuracy"],
                    "summary/last_accuracy_pct": summary["a_last_pct"],
                    "summary/forgetting_past_tasks": summary["forgetting_past_tasks"],
                    "summary/forgetting_past_tasks_pct": summary["f_past_pct"],
                    "summary/forgetting_official_code": summary["forgetting_official_code"],
                    "summary/forgetting_official_code_pct": summary["f_code_pct"],
                    "summary/total_duration_seconds": summary["total_duration_seconds"],
                    "final/average_incremental_accuracy": summary["a_avg_pct"],
                    "final/last_accuracy": summary["a_last_pct"],
                    "final/forgetting_past_tasks": summary["f_past_pct"],
                    "final/forgetting_official_code": summary["f_code_pct"],
                    "final/training_time_seconds": total_training_seconds,
                    "comparison/average_incremental_accuracy_pct": summary["a_avg_pct"],
                }
                for t_idx, acc in enumerate(summary.get("final_task_accuracies", []), start=1):
                    summary_payload[f"final_task_accuracies/task_{t_idx:02d}"] = acc * 100.0
                wandb_run.summary.update(summary_payload)

                columns = [f"Task_{i + 1:02d}" for i in range(max_tasks)]
                table_rows = []
                for row_idx, row in enumerate(task_matrix):
                    padded_row: list[Any] = [f"Stage_{row_idx + 1:02d}"]
                    for col_idx in range(max_tasks):
                        if col_idx <= row_idx and col_idx < len(row):
                            padded_row.append(round(row[col_idx] * 100.0, 2))
                        else:
                            padded_row.append(None)
                    table_rows.append(padded_row)
                wandb_run.log({"task_accuracy_matrix": wandb.Table(columns=["Stage"] + columns, data=table_rows)})

                artifact = wandb.Artifact(
                    name=f"{run_name}_results",
                    type="evaluation",
                    description=f"Headline baseline evaluation results for {run_name}",
                )
                for fn in ("summary.json", "metrics.json", "metadata.json", "task_matrix.csv", "task_accuracy.csv"):
                    fp = output_dir / fn
                    if fp.exists():
                        artifact.add_file(str(fp))
                wandb_run.log_artifact(artifact)
            except Exception as exc:
                print(f"Warning: W&B summary/artifact error: {exc}")

        print(f"Artifacts saved to {output_dir}")
        return result_payload
    finally:
        if wandb_run:
            try:
                wandb_run.finish()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the baseline runner.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    p = argparse.ArgumentParser(description="Run Continual Learning Baseline (LwF, Official LwF, ER)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--agent",
        type=str,
        choices=["lwf", "er", "official_lwf", "official-lwf"],
        default=None,
        help="Agent override",
    )
    p.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    p.add_argument("--run-name", type=str, default=None, help="Run name for W&B and metadata")
    p.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    p.add_argument("--seed", type=int, default=None, help="Random seed override")
    p.add_argument("--class-order-seed", type=int, default=None, help="Class order seed override")
    p.add_argument("--stream-seed", type=int, default=None, help="Stream shuffle seed override")
    p.add_argument("--max-tasks", type=int, default=None, help="Cap number of tasks")
    p.add_argument("--limit-train-per-task", type=int, default=None, help="Limit training samples per task")
    p.add_argument("--limit-test-per-task", type=int, default=None, help="Limit test samples per task")
    p.add_argument("--feature-cache-dir", type=str, default=None, help="Directory for cached ViT task features")
    p.add_argument(
        "--use-local-dtd-cache",
        action="store_true",
        help="Load pre-extracted features from results/dtd_feature_cache if present",
    )
    p.add_argument(
        "--local-dtd-cache-dir",
        type=str,
        default="results/dtd_feature_cache",
        help="Path to pre-extracted local DTD feature cache directory",
    )
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-offline", action="store_true", help="Log W&B offline")
    p.add_argument("--wandb-project", type=str, default=None, help="W&B project override")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity override")
    p.add_argument("--wandb-group", type=str, default=None, help="W&B group override")
    p.add_argument("--wandb-base-url", type=str, default=None, help="W&B base URL override")
    p.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated extra W&B tags")
    return p


if __name__ == "__main__":
    parser = build_parser()
    run_baseline(parser.parse_args())
