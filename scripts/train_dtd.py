"""Train one-pass F-OAL on DTD with comprehensive experiment tracking.

Adapted from scripts/train.py for the DTD (Describable Textures Dataset).

Usage:
    python -m scripts.train_dtd --config configs/dtd.yaml
    python -m scripts.train_dtd --config configs/dtd.yaml --device cpu
    python -m scripts.train_dtd --config configs/dtd.yaml --max-tasks 2 --limit-train-per-task 50
    python -m scripts.train_dtd --config configs/dtd.yaml --no-feature-fusion
    python -m scripts.train_dtd --config configs/dtd.yaml --no-projection
"""
from __future__ import annotations
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import csv
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.foal.dtd_datasets import (
    load_dtd, make_dtd_class_order, make_dtd_task_specs,
    filter_indices_by_classes, DTD_CLASS_NAMES,
)
from src.foal.datasets import (
    IndexedImageDataset, cifar100_transforms, limit_indices_per_task,
)
from src.foal.model import FOALModel
from src.foal.projection import VALID_PROJECTION_INITIALIZATIONS
from src.foal.metrics import summary_from_task_matrix, format_summary_percentages
from src.foal.environment import get_environment_info


def resolve_device(requested_device: str | None, config_device: str) -> str:
    target = (requested_device or config_device).lower()
    if target == 'cuda' or target.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{target}' was explicitly requested, but torch.cuda.is_available() is False."
            )
        return target
    elif target == 'auto':
        resolved = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device set to 'auto'. Resolved to: {resolved.upper()}")
        return resolved
    return target


def evaluate_task(model: FOALModel, loader: DataLoader, device: str) -> float:
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            scores = model(x.to(device))
            correct += int((scores.argmax(dim=1).cpu() == y).sum())
            total += len(y)
    return correct / total if total else float('nan')


def evaluate_task_cached_features(model: FOALModel, features: torch.Tensor, labels: torch.Tensor,
                                  batch_size: int, device: str) -> float:
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            h_batch = features[start:start + batch_size].to(device)
            y_batch = labels[start:start + batch_size]
            z_batch = model.projection(h_batch) if model.projection is not None else h_batch
            scores = model.classifier.scores(z_batch)
            correct += int((scores.argmax(dim=1).cpu() == y_batch).sum())
            total += len(y_batch)
    return correct / total if total else float('nan')


def resolve_seed(cli_value: Any, config_value: Any, default_seed: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    if config_value is not None:
        return int(config_value)
    return int(default_seed)


def resolve_projection_settings(cfg: dict[str, Any], args: argparse.Namespace, base_seed: int) -> dict[str, Any]:
    model_cfg = cfg.get('model', {}) or {}
    proj_cfg = cfg.get('projection', {}) or {}
    dimension = int(proj_cfg.get('dimension', model_cfg.get('projection_dim', 1000)))
    cli_init = getattr(args, 'projection_init', None)
    initialization = str(
        cli_init
        or proj_cfg.get('initialization')
        or model_cfg.get('projection_init')
        or 'official_uniform'
    )
    if initialization not in VALID_PROJECTION_INITIALIZATIONS:
        raise ValueError(
            f"Unsupported projection initialization '{initialization}'. "
            f"Expected one of {VALID_PROJECTION_INITIALIZATIONS}"
        )
    gaussian_std = float(proj_cfg.get('gaussian_std', model_cfg.get('gaussian_std', 1.0)))
    proj_seed_cfg = proj_cfg.get('seed') if proj_cfg.get('seed') is not None else cfg.get('projection_seed')
    seed = resolve_seed(getattr(args, 'projection_seed', None), proj_seed_cfg, base_seed)
    return {
        'dimension': dimension,
        'initialization': initialization,
        'gaussian_std': gaussian_std,
        'seed': seed,
    }


def extract_or_load_task_features(
    *,
    cache_dir: Path,
    cache_key: str,
    model: FOALModel,
    dataset: IndexedImageDataset,
    batch_size: int,
    num_workers: int,
    device: str,
    preprocessing: dict[str, Any] | None = None,
    dataset_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Extract frozen ViT features once or load from verified cache on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{cache_key}.pt'
    indices_sha = hashlib.sha256(json.dumps(dataset.indices).encode('utf-8')).hexdigest()

    if cache_file.exists():
        saved = torch.load(cache_file, map_location='cpu', weights_only=False)
        valid = (
            saved.get('indices_sha256') == indices_sha
            and saved.get('backbone_name') == model.backbone_name
            and saved.get('feature_fusion') == model.feature_fusion_enabled
            and saved.get('pretrained_tag') == getattr(model.encoder, 'pretrained_tag', None)
            and (dataset_name is None or saved.get('dataset_name') == dataset_name)
            and (preprocessing is None or saved.get('preprocessing') == preprocessing)
            and len(saved.get('features', [])) == len(dataset)
        )
        if valid:
            return saved['features'], saved['labels'], float(saved.get('extraction_seconds', 0.0)), True
        else:
            print(f"Warning: Cache {cache_file.name} validation failed. Re-extracting...")

    t0 = time.monotonic()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=num_workers)
    feat_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            feats = model.encoder(x.to(device)).cpu()
            feat_chunks.append(feats)
            label_chunks.append(y.cpu())
    features_cpu = torch.cat(feat_chunks, dim=0) if feat_chunks else torch.empty((0, model.encoder.feature_dim))
    labels_cpu = torch.cat(label_chunks, dim=0) if label_chunks else torch.empty((0,), dtype=torch.long)
    elapsed = round(time.monotonic() - t0, 4)

    torch.save({
        'backbone_name': model.backbone_name,
        'pretrained_tag': getattr(model.encoder, 'pretrained_tag', 'unknown'),
        'feature_fusion': model.feature_fusion_enabled,
        'dataset_name': dataset_name,
        'preprocessing': preprocessing,
        'indices_sha256': indices_sha,
        'features_sha256': hashlib.sha256(features_cpu.contiguous().numpy().tobytes()).hexdigest(),
        'extraction_seconds': elapsed,
        'features': features_cpu,
        'labels': labels_cpu,
    }, cache_file)
    return features_cpu, labels_cpu, elapsed, False


def get_resolved_pretrained_cfg(model: FOALModel) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        pcfg = getattr(model.encoder.network, 'pretrained_cfg', None) or {}
        for k in ('hf_hub_id', 'tag', 'architecture', 'url', 'input_size', 'interpolation', 'mean', 'std', 'num_classes'):
            if k in pcfg and pcfg[k] is not None:
                val = pcfg[k]
                info[k] = list(val) if isinstance(val, tuple) else val
    except Exception:
        pass
    return info


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))

    # Seed management
    base_seed = resolve_seed(args.seed, cfg.get('seed'), 42)
    class_order_seed = resolve_seed(args.class_order_seed, cfg.get('class_order_seed'), base_seed)
    stream_seed = resolve_seed(args.stream_seed, cfg.get('stream_seed'), base_seed)
    proj_settings = resolve_projection_settings(cfg, args, base_seed)
    projection_seed = proj_settings['seed']
    projection_dim = proj_settings['dimension']
    projection_init = proj_settings['initialization']
    gaussian_std = proj_settings['gaussian_std']

    # Seed global random generators
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    device = resolve_device(args.device, cfg['training'].get('device', 'auto'))
    data_cfg = cfg['dataset']
    model_cfg = cfg['model']

    # ===== DTD-specific data loading =====
    print(f"Loading DTD dataset...")
    train_source, test_source = load_dtd()
    labels_train = train_source.fine_label
    labels_test = test_source.fine_label

    total_classes = int(data_cfg.get('total_classes', 47))
    num_classes = int(data_cfg.get('num_classes', 40))
    classes_per_task = int(data_cfg.get('classes_per_task', 4))

    # Generate class order for DTD (select subset + permute)
    order = make_dtd_class_order(
        num_classes_to_use=num_classes,
        total_classes=total_classes,
        seed=class_order_seed,
    )

    # Filter training and test data to only include selected classes
    selected_classes = set(order)
    print(f"Selected {len(selected_classes)} DTD classes: {sorted(selected_classes)}")
    print(f"Class names: {[DTD_CLASS_NAMES[c] for c in sorted(selected_classes)]}")

    # Create task specs
    specs = make_dtd_task_specs(labels_train, labels_test, order, classes_per_task)

    max_tasks = args.max_tasks or len(specs)
    if not (1 <= max_tasks <= len(specs)):
        raise ValueError(f'max-tasks must be between 1 and {len(specs)}')

    # Create label mapping: original class index -> incremental index
    mapping = {original: i for i, original in enumerate(order)}

    # Preprocessing
    normalize_val = data_cfg.get('normalize', data_cfg.get('normalization', True))
    normalize_bool = bool(normalize_val) if normalize_val is not None else True
    mean_val = data_cfg.get('mean', (0.5, 0.5, 0.5))
    std_val = data_cfg.get('std', (0.5, 0.5, 0.5))
    mean_tuple = tuple(float(v) for v in mean_val) if mean_val is not None else (0.5, 0.5, 0.5)
    std_tuple = tuple(float(v) for v in std_val) if std_val is not None else (0.5, 0.5, 0.5)
    interp_str = str(data_cfg.get('interpolation', 'bicubic')).lower()
    trans = cifar100_transforms(
        size=int(data_cfg.get('image_size', 224)),
        mean=mean_tuple, std=std_tuple,
        interpolation=interp_str,
        normalize=normalize_bool,
    )

    use_fusion = bool(model_cfg['feature_fusion']) and not args.no_feature_fusion
    use_projection = bool(model_cfg['smooth_projection']) and not args.no_projection
    rls_dtype = model_cfg.get('rls_dtype', 'float64')

    # Method classification
    if use_fusion and use_projection:
        method = 'foal'
    elif not use_fusion and use_projection:
        method = 'foal_no_fusion'
    elif use_fusion and not use_projection:
        method = 'foal_no_projection'
    else:
        method = 'foal_neither'

    # Experiment type
    if args.experiment_type:
        experiment_type = args.experiment_type
    elif max_tasks < len(specs) or args.limit_train_per_task or args.limit_test_per_task:
        experiment_type = 'pilot'
    elif not use_fusion or not use_projection:
        experiment_type = 'ablation'
    else:
        experiment_type = 'reproduction'

    # Build model
    model = FOALModel(
        backbone_name=model_cfg['backbone'],
        pretrained=bool(model_cfg['pretrained']),
        feature_fusion=use_fusion,
        smooth_projection=use_projection,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
        projection_init=projection_init,
        gaussian_std=gaussian_std,
        gamma=float(model_cfg['gamma']),
        rls_dtype=rls_dtype,
        device=device,
    )
    model.eval()

    run_name = args.run_name or f'dtd_{method}_seed{base_seed}_{experiment_type}'
    if getattr(args, 'output_dir', None):
        outdir = Path(args.output_dir)
    else:
        outdir = Path(cfg['logging'].get('results_dir', 'results')) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Environment and metadata
    env_info = get_environment_info()
    projection_meta = model.projection.get_metadata() if model.projection is not None else None
    pretrained_cfg_info = get_resolved_pretrained_cfg(model)

    feature_cache_dir = Path(args.feature_cache_dir) if getattr(args, 'feature_cache_dir', None) else None

    metadata = {
        'paper': 'F-OAL NeurIPS 2024',
        'dataset': 'DTD',
        'method': method,
        'experiment_type': experiment_type,
        'run_name': run_name,
        'git_commit': env_info.get('git', {}).get('commit'),
        'seeds': {
            'base': base_seed,
            'class_order': class_order_seed,
            'projection': projection_seed,
            'stream': stream_seed,
        },
        'model_config': {
            'backbone': model_cfg['backbone'],
            'pretrained': bool(model_cfg['pretrained']),
            'pretrained_cfg': pretrained_cfg_info,
            'feature_fusion': use_fusion,
            'smooth_projection': use_projection,
            'projection_dim': projection_dim,
            'projection_init': projection_init,
            'gaussian_std': gaussian_std,
            'gamma': float(model_cfg['gamma']),
            'rls_dtype': rls_dtype,
        },
        'projection_metadata': projection_meta,
        'training_protocol': {
            'dataset_source': data_cfg['name'],
            'dataset_type': 'dtd',
            'total_classes': total_classes,
            'num_classes_used': num_classes,
            'classes_per_task': classes_per_task,
            'max_tasks': max_tasks,
            'limit_train_per_task': args.limit_train_per_task,
            'limit_test_per_task': args.limit_test_per_task,
            'batch_size': int(cfg['training']['batch_size']),
            'device': device,
            'one_pass': True,
            'feature_cache_enabled': feature_cache_dir is not None,
        },
        'class_order': order,
        'selected_classes': sorted(selected_classes),
        'class_names': [DTD_CLASS_NAMES[c] for c in order],
        'environment': env_info,
        'config': cfg,
        'paper_reference': {
            'full_foal_a_avg': 82.8,
            'without_ff_a_avg': 80.5,
            'without_sp_a_avg': 79.3,
            'without_both_a_avg': 71.2,
            'training_time_seconds': 8,
        },
    }
    (outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # W&B
    wandb_run = None
    use_wandb = args.wandb or cfg['logging'].get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            if args.wandb_offline:
                os.environ['WANDB_MODE'] = 'offline'

            base_url = getattr(args, 'wandb_base_url', None) or cfg['logging'].get('base_url') or os.environ.get('WANDB_BASE_URL')
            if base_url:
                os.environ['WANDB_BASE_URL'] = base_url

            project = args.wandb_project or cfg['logging'].get('project', 'hcmus-foal-reproduction')
            entity = args.wandb_entity or cfg['logging'].get('entity')
            group = args.wandb_group or cfg['logging'].get('group') or f'dtd_{method}_{experiment_type}'
            tags = ['dtd', experiment_type, method, f'seed_{base_seed}', f'proj_{projection_init}']
            if args.wandb_tags:
                tags.extend([t.strip() for t in args.wandb_tags.split(',') if t.strip()])
            wandb_run = wandb.init(project=project, entity=entity, group=group,
                                   name=run_name, config=metadata, tags=tags)
        except Exception as exc:
            print(f'Warning: W&B init failed: {exc}. Continuing without W&B.')
            wandb_run = None

    # ===== Training loop =====
    matrix: list[list[float]] = []
    seen_specs = []
    cached_test_sets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    start_time = time.monotonic()
    total_feature_extraction_seconds = 0.0
    total_rls_and_eval_seconds = 0.0
    batch_size = int(cfg['training']['batch_size'])
    num_workers = int(cfg['training']['num_workers'])

    initial_proj_sha = projection_meta['matrix_sha256'] if projection_meta is not None else None

    prep_meta = {
        'image_size': int(data_cfg.get('image_size', 224)),
        'normalize': normalize_bool,
        'mean': list(mean_tuple) if normalize_bool else None,
        'std': list(std_tuple) if normalize_bool else None,
        'interpolation': interp_str,
    }
    dataset_name = str(data_cfg.get('name', 'tanganke/dtd'))

    # GPU memory tracking
    peak_gpu_memory_mb = 0.0
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    try:
        for spec in specs[:max_tasks]:
            task_start = time.monotonic()
            model.classifier.expand_classes(spec.total_seen_classes)

            chosen_train_ids = limit_indices_per_task(
                spec.train_indices, args.limit_train_per_task, stream_seed + spec.task_id
            )
            training_ds = IndexedImageDataset(train_source, chosen_train_ids, mapping, trans)
            gen = torch.Generator().manual_seed(stream_seed + spec.task_id)

            # Print task info
            print(f"\n--- Task {spec.task_id + 1}/{max_tasks} ---")
            print(f"  New classes: {spec.new_original_labels} "
                  f"({[DTD_CLASS_NAMES[c] for c in spec.new_original_labels]})")
            print(f"  Train samples: {len(chosen_train_ids)}, "
                  f"Total seen classes: {spec.total_seen_classes}")

            if feature_cache_dir is not None:
                bb_slug = model.backbone_name.replace('/', '_').replace('.', '_')
                interp_tag = interp_str
                norm_tag = "norm" if normalize_bool else "raw"
                ff_tag = "ff" if use_fusion else "noff"
                cache_prefix = (
                    f"dtd_{bb_slug}_{interp_tag}_{norm_tag}_{ff_tag}_c{class_order_seed}"
                    f"_s{stream_seed}_t{spec.task_id + 1:02d}"
                    f"_tr{args.limit_train_per_task or 'full'}"
                )
                train_feats, train_labels, ext_sec, from_cache = extract_or_load_task_features(
                    cache_dir=feature_cache_dir,
                    cache_key=f'train_{cache_prefix}',
                    model=model,
                    dataset=training_ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                    preprocessing=prep_meta,
                    dataset_name=dataset_name,
                )
                total_feature_extraction_seconds += ext_sec
                t_rls = time.monotonic()
                feat_ds = TensorDataset(train_feats, train_labels)
                loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True,
                                    generator=gen, drop_last=False, num_workers=0)
                with torch.no_grad():
                    for h_batch, y_batch in loader:
                        h_dev = h_batch.to(device)
                        z_dev = model.projection(h_dev) if model.projection is not None else h_dev
                        model.classifier.update(z_dev, y_batch.to(device))
                total_rls_and_eval_seconds += (time.monotonic() - t_rls)
            else:
                t_step = time.monotonic()
                loader = DataLoader(training_ds, batch_size=batch_size,
                                    shuffle=True, generator=gen, drop_last=False,
                                    num_workers=num_workers)
                for x, y in loader:
                    model.partial_fit(x.to(device), y.to(device))
                total_rls_and_eval_seconds += (time.monotonic() - t_step)

            # Evaluate on all seen tasks
            seen_specs.append(spec)
            per_task: list[float] = []

            for previous_spec in seen_specs:
                task_test_ids = [idx for idx in previous_spec.seen_test_indices
                                 if int(labels_test[idx]) in set(previous_spec.new_original_labels)]
                if args.limit_test_per_task is not None:
                    task_test_ids = limit_indices_per_task(
                        task_test_ids, args.limit_test_per_task, stream_seed + previous_spec.task_id
                    )

                if feature_cache_dir is not None:
                    if previous_spec.task_id not in cached_test_sets:
                        test_ds = IndexedImageDataset(test_source, task_test_ids, mapping, trans)
                        bb_slug = model.backbone_name.replace('/', '_').replace('.', '_')
                        cache_prefix = (
                            f"dtd_{bb_slug}_{interp_str}_{norm_tag}_{ff_tag}_c{class_order_seed}"
                            f"_s{stream_seed}_t{previous_spec.task_id + 1:02d}"
                            f"_te{args.limit_test_per_task or 'full'}"
                        )
                        t_feats, t_labels, ext_sec, _ = extract_or_load_task_features(
                            cache_dir=feature_cache_dir,
                            cache_key=f'test_{cache_prefix}',
                            model=model,
                            dataset=test_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            device=device,
                            preprocessing=prep_meta,
                            dataset_name=dataset_name,
                        )
                        total_feature_extraction_seconds += ext_sec
                        cached_test_sets[previous_spec.task_id] = (t_feats, t_labels)
                    t_eval = time.monotonic()
                    t_feats, t_labels = cached_test_sets[previous_spec.task_id]
                    per_task.append(evaluate_task_cached_features(model, t_feats, t_labels, batch_size, device))
                    total_rls_and_eval_seconds += (time.monotonic() - t_eval)
                else:
                    t_eval = time.monotonic()
                    test_ds = IndexedImageDataset(test_source, task_test_ids, mapping, trans)
                    test_loader = DataLoader(test_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
                    per_task.append(evaluate_task(model, test_loader, device))
                    total_rls_and_eval_seconds += (time.monotonic() - t_eval)

            # Verify projection remained frozen
            if model.projection is not None and initial_proj_sha is not None:
                current_proj_sha = model.projection.get_metadata()['matrix_sha256']
                if current_proj_sha != initial_proj_sha:
                    raise RuntimeError('Projection matrix P changed during training!')

            matrix.append(per_task)
            stage_acc = sum(per_task) / len(per_task)
            elapsed = round(time.monotonic() - start_time, 2)
            task_elapsed = round(time.monotonic() - task_start, 2)

            # Track GPU memory
            if device.startswith('cuda'):
                current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, current_peak)

            record = {
                'task': spec.task_id + 1,
                'task_accuracy': per_task,
                'average_accuracy': stage_acc,
                'seen_classes': spec.total_seen_classes,
                'samples_seen': model.classifier.samples_seen,
                'task_duration_seconds': task_elapsed,
                'total_duration_seconds': elapsed,
            }
            print(json.dumps(record))

            if wandb_run:
                try:
                    step_metrics = {
                        'task': spec.task_id + 1,
                        'stage_accuracy': stage_acc,
                        'stage_accuracy_pct': stage_acc * 100,
                        'samples_seen': model.classifier.samples_seen,
                        'duration_seconds': elapsed,
                    }
                    for i, acc_val in enumerate(per_task):
                        step_metrics[f'task_acc/task_{i+1}'] = acc_val
                        step_metrics[f'task_acc_pct/task_{i+1}'] = acc_val * 100
                    wandb_run.log(step_metrics)
                except Exception as exc:
                    print(f'Warning: W&B step log error: {exc}')

            # Save checkpoint
            checkpoint_path = outdir / f'task_{spec.task_id+1:02d}.pt'
            torch.save({
                'model_config': {
                    'backbone_name': model_cfg['backbone'],
                    'feature_fusion': use_fusion,
                    'smooth_projection': use_projection,
                    'projection_dim': projection_dim,
                    'projection_seed': projection_seed,
                    'projection_init': projection_init,
                    'gaussian_std': gaussian_std,
                    'gamma': float(model_cfg['gamma']),
                    'rls_dtype': rls_dtype,
                },
                'projection_state': model.projection.state_dict() if model.projection else None,
                'classifier_state': model.classifier.state_dict(),
                'class_order': order,
                'task_number': spec.task_id + 1,
                'metadata': metadata,
                'config': cfg,
            }, checkpoint_path)

        # Compute final metrics
        metrics = summary_from_task_matrix(matrix)
        total_duration = round(time.monotonic() - start_time, 2)
        metrics['total_duration_seconds'] = total_duration
        metrics['feature_extraction_seconds'] = round(total_feature_extraction_seconds, 2)
        metrics['rls_and_eval_seconds'] = round(total_rls_and_eval_seconds, 2)
        if device.startswith('cuda'):
            metrics['peak_gpu_memory_mb'] = round(peak_gpu_memory_mb, 2)
            metrics['peak_gpu_memory_gb'] = round(peak_gpu_memory_mb / 1024, 3)

        # Save metrics
        metrics_payload = {'task_matrix': matrix, **metrics}
        (outdir / 'metrics.json').write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')
        summary_payload = {
            'accuracy_bar': metrics['average_incremental_accuracy'] * 100.0,
            'a_10': metrics['last_accuracy'] * 100.0,
            'f_10_past': metrics['forgetting_past_tasks'] * 100.0,
            'f_10_code': metrics['forgetting_official_code'] * 100.0,
            **metrics,
        }
        (outdir / 'summary.json').write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
        (outdir / 'matrix.json').write_text(json.dumps(matrix, indent=2), encoding='utf-8')

        with (outdir / 'task_accuracy.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['stage', 'task', 'accuracy'])
            for stage, row in enumerate(matrix, start=1):
                for i, val in enumerate(row, start=1):
                    writer.writerow([stage, i, val])

        formatted_pcts = format_summary_percentages(metrics)
        print('\n' + '=' * 70)
        print(f'DTD F-OAL EXPERIMENT RESULTS ({method})')
        print('=' * 70)
        print(f"Method:                  {method}")
        print(f"Feature Fusion:          {'ON' if use_fusion else 'OFF'}")
        print(f"Smooth Projection:       {'ON' if use_projection else 'OFF'}")
        print(f"Backbone:                {model_cfg['backbone']}")
        print(f"Projection Init:         {projection_init}")
        print(f"Seed:                    {base_seed}")
        print(f"Tasks:                   {max_tasks}")
        print(f"Classes per task:        {classes_per_task}")
        print(f"Total classes used:      {num_classes}")
        print(f"---")
        print(f"Avg Incr Accuracy:       {formatted_pcts.get('average_incremental_accuracy')}")
        print(f"Last Accuracy:           {formatted_pcts.get('last_accuracy')}")
        print(f"Forgetting (past-task):  {formatted_pcts.get('forgetting_past_tasks')}")
        print(f"Forgetting (official):   {formatted_pcts.get('forgetting_official_code')}")
        print(f"Total Duration:          {total_duration}s")
        if device.startswith('cuda'):
            print(f"Peak GPU Memory:         {metrics.get('peak_gpu_memory_gb', 'N/A')} GB")
        print('=' * 70)

        if wandb_run:
            try:
                wandb_run.log({
                    'summary/average_incremental_accuracy': metrics['average_incremental_accuracy'],
                    'summary/average_incremental_accuracy_pct': metrics['average_incremental_accuracy'] * 100,
                    'summary/last_accuracy': metrics['last_accuracy'],
                    'summary/last_accuracy_pct': metrics['last_accuracy'] * 100,
                    'summary/forgetting_past_tasks': metrics['forgetting_past_tasks'],
                    'summary/forgetting_past_tasks_pct': metrics['forgetting_past_tasks'] * 100,
                    'summary/forgetting_official_code': metrics['forgetting_official_code'],
                    'summary/forgetting_official_code_pct': metrics['forgetting_official_code'] * 100,
                    'summary/total_duration_seconds': total_duration,
                })
                artifact = wandb.Artifact(name=f'{run_name}_results', type='evaluation',
                                          description=f'DTD evaluation results for {run_name}')
                artifact.add_file(str(outdir / 'metrics.json'))
                artifact.add_file(str(outdir / 'metadata.json'))
                artifact.add_file(str(outdir / 'task_accuracy.csv'))
                wandb_run.log_artifact(artifact)
            except Exception as exc:
                print(f'Warning: W&B summary/artifact error: {exc}')

        return metrics
    finally:
        if wandb_run:
            try:
                wandb_run.finish()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='Train one-pass F-OAL on DTD.')
    parser.add_argument('--config', default='configs/dtd.yaml')
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--limit-train-per-task', type=int, default=None)
    parser.add_argument('--limit-test-per-task', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--class-order-seed', type=int, default=None)
    parser.add_argument('--projection-seed', type=int, default=None)
    parser.add_argument('--projection-init', default=None,
                        choices=list(VALID_PROJECTION_INITIALIZATIONS))
    parser.add_argument('--stream-seed', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--experiment-type', default=None,
                        choices=['pilot', 'reproduction', 'ablation'])
    parser.add_argument('--no-feature-fusion', action='store_true')
    parser.add_argument('--no-projection', action='store_true')
    parser.add_argument('--feature-cache-dir', default=None)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-offline', action='store_true')
    parser.add_argument('--wandb-project', default=None)
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-base-url', default=None)
    parser.add_argument('--wandb-group', default=None)
    parser.add_argument('--wandb-tags', default=None)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
