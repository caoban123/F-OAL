"""Train one-pass F-OAL on CIFAR-100 with comprehensive experiment tracking.

Usage:
    python -m scripts.train --config configs/cifar100.yaml
    python -m scripts.train --config configs/c1_p1_official_uniform.yaml
    python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100
    python -m scripts.train --config configs/cifar100.yaml --wandb --wandb-offline
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
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.foal.datasets import (load_cifar100, make_class_order, make_task_specs,
                              IndexedImageDataset, cifar100_transforms, limit_indices_per_task)
from src.foal.dtd_datasets import load_dtd, make_dtd_class_order, make_dtd_task_specs
from src.foal.model import FOALModel
from src.foal.projection import VALID_PROJECTION_INITIALIZATIONS
from src.foal.metrics import summary_from_task_matrix, format_summary_percentages
from src.foal.environment import get_environment_info


def resolve_device(requested_device: str | None, config_device: str) -> str:
    """Resolve compute device with strict validation against silent fallback.

    If 'cuda' is explicitly requested (via CLI or config) but CUDA is unavailable,
    raise RuntimeError immediately instead of silently executing on CPU.
    """
    target = (requested_device or config_device).lower()
    if target == 'cuda' or target.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{target}' was explicitly requested, but torch.cuda.is_available() is False. "
                "Aborting to prevent unintentional CPU execution on GPU workloads. "
                "To run on CPU, specify --device cpu or set training.device: cpu in config."
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
    """Resolve random seed with strict precedence: CLI arg > Config value > Default."""
    if cli_value is not None:
        return int(cli_value)
    if config_value is not None:
        return int(config_value)
    return int(default_seed)


def resolve_projection_settings(cfg: dict[str, Any], args: argparse.Namespace, base_seed: int) -> dict[str, Any]:
    """Resolve Smooth Projection configuration from YAML (`projection:` or `model:`) and CLI overrides."""
    model_cfg = cfg.get('model', {}) or {}
    proj_cfg = cfg.get('projection', {}) or {}

    dimension = int(proj_cfg.get('dimension', model_cfg.get('projection_dim', 1000)))
    cli_init = getattr(args, 'projection_init', None)
    initialization = str(
        cli_init
        or proj_cfg.get('initialization')
        or model_cfg.get('projection_init')
        or 'gaussian_std1'
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


def compute_task_split_fingerprint(order: Sequence[int], specs: Sequence[Any], max_tasks: int,
                                   limit_train: int | None, limit_test: int | None,
                                   stream_seed: int, labels_test: Sequence[int]) -> dict[str, Any]:
    """Compute a deterministic SHA256 fingerprint of the class order and per-task sample indices."""
    task_entries = []
    for spec in specs[:max_tasks]:
        chosen_train_ids = limit_indices_per_task(
            spec.train_indices, limit_train, stream_seed + spec.task_id
        )
        task_test_ids = [
            idx for idx in spec.seen_test_indices
            if int(labels_test[idx]) in set(spec.new_original_labels)
        ]
        if limit_test is not None:
            task_test_ids = limit_indices_per_task(
                task_test_ids, limit_test, stream_seed + spec.task_id
            )
        train_sha = hashlib.sha256(json.dumps(chosen_train_ids).encode('utf-8')).hexdigest()
        test_sha = hashlib.sha256(json.dumps(task_test_ids).encode('utf-8')).hexdigest()
        task_entries.append({
            'task_id': int(spec.task_id),
            'new_original_labels': list(spec.new_original_labels),
            'num_train_samples': len(chosen_train_ids),
            'num_test_samples': len(task_test_ids),
            'train_indices_sha256': train_sha,
            'test_indices_sha256': test_sha,
        })
    payload = {
        'class_order': list(order),
        'max_tasks': int(max_tasks),
        'limit_train_per_task': limit_train,
        'limit_test_per_task': limit_test,
        'stream_seed': int(stream_seed),
        'tasks': task_entries,
    }
    overall_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
    return {
        'sha256': overall_sha,
        'class_order_sha256': hashlib.sha256(json.dumps(list(order)).encode('utf-8')).hexdigest(),
        'num_tasks': int(max_tasks),
        'tasks': task_entries,
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
    """Extract frozen ViT features once or load from verified cache on disk.

    Returns (features_cpu, labels_cpu, extraction_seconds, loaded_from_cache).
    """
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
            print(f"Warning: Cache {cache_file.name} failed integrity/provenance validation. Re-extracting features...")

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
    """Extract JSON-serializable pretrained backbone configuration from timm."""
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

    # Granular seed management with strict RNG isolation (A12 / C1 Section 7.1)
    base_seed = resolve_seed(args.seed, cfg.get('seed'), 42)
    class_order_seed = resolve_seed(args.class_order_seed, cfg.get('class_order_seed'), base_seed)
    stream_seed = resolve_seed(args.stream_seed, cfg.get('stream_seed'), base_seed)
    proj_settings = resolve_projection_settings(cfg, args, base_seed)
    projection_seed = proj_settings['seed']
    projection_dim = proj_settings['dimension']
    projection_init = proj_settings['initialization']
    gaussian_std = proj_settings['gaussian_std']

    # Seed global random generators with base_seed
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    device = resolve_device(args.device, cfg['training'].get('device', 'auto'))
    data_cfg = cfg['dataset']
    model_cfg = cfg['model']

    dataset_type = str(data_cfg.get('type', '')).lower()
    dataset_name = str(data_cfg.get('name', '')).lower()
    is_dtd = dataset_type == 'dtd' or 'dtd' in dataset_name

    if is_dtd:
        train_source, test_source = load_dtd()
        labels_train = train_source.fine_label
        labels_test = test_source.fine_label
        total_dtd_classes = int(data_cfg.get('total_classes', 47))
        num_classes = int(data_cfg.get('num_classes', 40))
        classes_per_task = int(data_cfg.get('classes_per_task', 4))
        if data_cfg.get('class_order') is not None:
            order = list(data_cfg['class_order'])
        else:
            order = make_dtd_class_order(
                num_classes_to_use=num_classes,
                total_classes=total_dtd_classes,
                seed=class_order_seed,
            )
        specs = make_dtd_task_specs(labels_train, labels_test, order, classes_per_task)
    else:
        train_source, test_source = load_cifar100()
        labels_train = train_source['fine_label']
        labels_test = test_source['fine_label']

        order = make_class_order(int(data_cfg['num_classes']), class_order_seed, data_cfg.get('class_order'))
        specs = make_task_specs(labels_train, labels_test, order, int(data_cfg['classes_per_task']))

    max_tasks = args.max_tasks or len(specs)
    if not (1 <= max_tasks <= len(specs)):
        raise ValueError(f'max-tasks must be between 1 and {len(specs)}')

    mapping = {original: i for i, original in enumerate(order)}
    normalize_val = data_cfg.get('normalize', data_cfg.get('normalization', True))
    normalize_bool = bool(normalize_val) if normalize_val is not None else True
    mean_val = data_cfg.get('mean', (0.5, 0.5, 0.5))
    std_val = data_cfg.get('std', (0.5, 0.5, 0.5))
    mean_tuple = tuple(float(v) for v in mean_val) if mean_val is not None else (0.5, 0.5, 0.5)
    std_tuple = tuple(float(v) for v in std_val) if std_val is not None else (0.5, 0.5, 0.5)
    interp_str = str(data_cfg.get('interpolation', 'bicubic')).lower()
    trans = cifar100_transforms(size=int(data_cfg.get('image_size', 224)),
                               mean=mean_tuple, std=std_tuple,
                               interpolation=interp_str,
                               normalize=normalize_bool)

    use_fusion = bool(model_cfg['feature_fusion']) and not args.no_feature_fusion
    use_projection = bool(model_cfg['smooth_projection']) and not args.no_projection
    rls_dtype = model_cfg.get('rls_dtype', 'float64')

    # Experiment identity and method classification
    if use_fusion and use_projection:
        method = 'foal'
    elif not use_fusion and use_projection:
        method = 'foal_no_fusion'
    elif use_fusion and not use_projection:
        method = 'foal_no_projection'
    else:
        method = 'foal_neither'

    # Determine experiment type
    if args.experiment_type:
        experiment_type = args.experiment_type
    elif max_tasks < len(specs) or args.limit_train_per_task or args.limit_test_per_task:
        experiment_type = 'pilot'
    elif not use_fusion or not use_projection:
        experiment_type = 'ablation'
    else:
        experiment_type = 'reproduction'

    model = FOALModel(backbone_name=model_cfg['backbone'],
                      pretrained=bool(model_cfg['pretrained']),
                      feature_fusion=use_fusion,
                      smooth_projection=use_projection,
                      projection_dim=projection_dim,
                      projection_seed=projection_seed,
                      projection_init=projection_init,
                      gaussian_std=gaussian_std,
                      gamma=float(model_cfg['gamma']),
                      rls_dtype=rls_dtype,
                      device=device)
    model.eval()

    run_name = args.run_name or f'{method}_seed{base_seed}_{experiment_type}'
    if getattr(args, 'output_dir', None):
        outdir = Path(args.output_dir)
    else:
        outdir = Path(cfg['logging'].get('results_dir', 'results')) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Capture complete environment, projection metadata, and task split fingerprint (Section 8.3)
    env_info = get_environment_info()
    projection_meta = model.projection.get_metadata() if model.projection is not None else None
    pretrained_cfg_info = get_resolved_pretrained_cfg(model)
    task_split_fp = compute_task_split_fingerprint(
        order=order,
        specs=specs,
        max_tasks=max_tasks,
        limit_train=args.limit_train_per_task,
        limit_test=args.limit_test_per_task,
        stream_seed=stream_seed,
        labels_test=labels_test,
    )

    feature_cache_dir = Path(args.feature_cache_dir) if getattr(args, 'feature_cache_dir', None) else None

    metadata = {
        'paper': 'F-OAL NeurIPS 2024',
        'method': method,
        'experiment_type': experiment_type,
        'run_name': run_name,
        'git_commit': env_info.get('git', {}).get('commit'),
        'seeds': {
            'base': base_seed,
            'dataset': base_seed,
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
            'dataset': data_cfg['name'],
            'num_classes': int(data_cfg['num_classes']),
            'classes_per_task': int(data_cfg['classes_per_task']),
            'max_tasks': max_tasks,
            'limit_train_per_task': args.limit_train_per_task,
            'limit_test_per_task': args.limit_test_per_task,
            'batch_size': int(cfg['training']['batch_size']),
            'device': device,
            'one_pass': True,
            'feature_cache_enabled': feature_cache_dir is not None,
        },
        'task_split_fingerprint': task_split_fp,
        'class_order': order,
        'environment': env_info,
        'config': cfg,
    }
    (outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # Weights & Biases experiment tracking (A13 / C1 Section 11.2)
    wandb_run = None
    use_wandb = (args.wandb or cfg['logging'].get('use_wandb', False)) and os.environ.get('WANDB_MODE') != 'disabled'
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
            group = args.wandb_group or f'{method}_{experiment_type}'
            tags = [experiment_type, method, f'seed_{base_seed}', f'proj_{projection_init}']
            if args.wandb_tags:
                tags.extend([t.strip() for t in args.wandb_tags.split(',') if t.strip()])

            wandb_run = wandb.init(project=project, entity=entity, group=group,
                                   name=run_name, config=metadata, tags=tags)
        except Exception as exc:
            print(f'Warning: Weights & Biases initialization failed: {exc}. Continuing training without W&B.')
            wandb_run = None

    matrix: list[list[float]] = []
    seen_specs = []
    cached_test_sets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    start_time = time.monotonic()
    total_feature_extraction_seconds = 0.0
    total_rls_and_eval_seconds = 0.0
    batch_size = int(cfg['training']['batch_size'])
    num_workers = int(cfg['training']['num_workers'])

    # Initial projection matrix fingerprint to verify P stays strictly frozen across all stages
    initial_proj_sha = projection_meta['matrix_sha256'] if projection_meta is not None else None

    prep_meta = {
        'image_size': int(cfg['dataset'].get('image_size', 224)),
        'normalize': normalize_bool,
        'mean': list(mean_tuple) if normalize_bool else None,
        'std': list(std_tuple) if normalize_bool else None,
        'interpolation': interp_str,
    }
    dataset_name = str(cfg['dataset'].get('name', 'uoft-cs/cifar100'))

    try:
        for spec in specs[:max_tasks]:
            model.classifier.expand_classes(spec.total_seen_classes)

            chosen_train_ids = limit_indices_per_task(
                spec.train_indices, args.limit_train_per_task, stream_seed + spec.task_id
            )
            training_ds = IndexedImageDataset(train_source, chosen_train_ids, mapping, trans)
            gen = torch.Generator().manual_seed(stream_seed + spec.task_id)

            if feature_cache_dir is not None:
                bb_slug = model.backbone_name.replace('/', '_').replace('.', '_')
                interp_tag = "bicubic" if interp_str == "bicubic" else "bilinear"
                norm_tag = "norm" if normalize_bool else "raw"
                ff_tag = "ff" if use_fusion else "noff"
                cache_prefix = (
                    f"{bb_slug}_{interp_tag}_{norm_tag}_{ff_tag}_c{class_order_seed}_s{stream_seed}_t{spec.task_id + 1:02d}"
                    f"_tr{args.limit_train_per_task or 'full'}"
                )
                train_feats, train_labels, ext_sec, _ = extract_or_load_task_features(
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
                        interp_tag = "bicubic" if interp_str == "bicubic" else "bilinear"
                        norm_tag = "norm" if normalize_bool else "raw"
                        ff_tag = "ff" if use_fusion else "noff"
                        cache_prefix = (
                            f"{bb_slug}_{interp_tag}_{norm_tag}_{ff_tag}_c{class_order_seed}_s{stream_seed}_t{previous_spec.task_id + 1:02d}"
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

            # Verify projection matrix P remained completely unchanged after task
            if model.projection is not None and initial_proj_sha is not None:
                current_proj_sha = model.projection.get_metadata()['matrix_sha256']
                if current_proj_sha != initial_proj_sha:
                    raise RuntimeError('Projection matrix P changed during training! Must remain strictly frozen.')

            matrix.append(per_task)
            stage_acc = sum(per_task) / len(per_task)
            elapsed = round(time.monotonic() - start_time, 2)

            record = {
                'task': spec.task_id + 1,
                'task_accuracy': per_task,
                'average_accuracy': stage_acc,
                'seen_classes': spec.total_seen_classes,
                'samples_seen': model.classifier.samples_seen,
                'duration_seconds': elapsed,
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

            # Save checkpoint with complete reproducible configuration
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

        # Compute formal metrics (including both C0 forgetting definitions)
        metrics = summary_from_task_matrix(matrix)
        total_duration = round(time.monotonic() - start_time, 2)
        metrics['total_duration_seconds'] = total_duration
        metrics['feature_extraction_seconds'] = round(total_feature_extraction_seconds, 2)
        metrics['rls_and_eval_seconds'] = round(total_rls_and_eval_seconds, 2)

        # Save metrics and summary
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
        print('SUMMARY (ratios):', json.dumps(metrics))
        print(
            f"SUMMARY (percentage): Avg Incr Acc: {formatted_pcts.get('average_incremental_accuracy')}, "
            f"Last Acc: {formatted_pcts.get('last_accuracy')}, "
            f"Forgetting (past-task): {formatted_pcts.get('forgetting_past_tasks')}, "
            f"Forgetting (official-code): {formatted_pcts.get('forgetting_official_code')}"
        )

        if wandb_run:
            try:
                wandb_run.log({
                    'summary/average_incremental_accuracy': metrics['average_incremental_accuracy'],
                    'summary/average_incremental_accuracy_pct': metrics['average_incremental_accuracy'] * 100,
                    'summary/last_accuracy': metrics['last_accuracy'],
                    'summary/last_accuracy_pct': metrics['last_accuracy'] * 100,
                    'summary/forgetting': metrics['forgetting'],
                    'summary/forgetting_pct': metrics['forgetting'] * 100,
                    'summary/forgetting_past_tasks': metrics['forgetting_past_tasks'],
                    'summary/forgetting_past_tasks_pct': metrics['forgetting_past_tasks'] * 100,
                    'summary/forgetting_official_code': metrics['forgetting_official_code'],
                    'summary/forgetting_official_code_pct': metrics['forgetting_official_code'] * 100,
                    'summary/total_duration_seconds': total_duration,
                    'summary/feature_extraction_seconds': metrics['feature_extraction_seconds'],
                    'summary/rls_and_eval_seconds': metrics['rls_and_eval_seconds'],
                })
                # Upload evaluation artifacts
                artifact = wandb.Artifact(name=f'{run_name}_results', type='evaluation',
                                          description=f'Evaluation results for {run_name}')
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
    parser = argparse.ArgumentParser(description='Train one-pass F-OAL on CIFAR-100.')
    parser.add_argument('--config', default='configs/cifar100.yaml')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Pilot only: limit number of sequential tasks')
    parser.add_argument('--limit-train-per-task', type=int, default=None,
                        help='Pilot only: limit training samples per task')
    parser.add_argument('--limit-test-per-task', type=int, default=None,
                        help='Pilot only: limit test samples evaluated per task')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base random seed (overrides config seed)')
    parser.add_argument('--class-order-seed', type=int, default=None,
                        help='Dedicated seed for dataset class permutation')
    parser.add_argument('--projection-seed', type=int, default=None,
                        help='Dedicated seed for random projection matrix')
    parser.add_argument('--projection-init', default=None,
                        choices=list(VALID_PROJECTION_INITIALIZATIONS),
                        help='Smooth Projection initialization strategy (gaussian_std1, official_uniform, gaussian_matched_variance)')
    parser.add_argument('--stream-seed', type=int, default=None,
                        help='Dedicated seed for data stream ordering and sampling')
    parser.add_argument('--device', default=None,
                        help="Device to use ('cpu', 'cuda', etc.)")
    parser.add_argument('--experiment-type', default=None,
                        choices=['pilot', 'reproduction', 'ablation', 'additional-analysis', 'controlled-c1'],
                        help='Classification of the experiment')
    parser.add_argument('--no-feature-fusion', action='store_true',
                        help='Ablation: disable feature fusion')
    parser.add_argument('--no-projection', action='store_true',
                        help='Ablation: disable smooth projection')
    parser.add_argument('--feature-cache-dir', default=None,
                        help='Optional directory to cache deterministic frozen ViT features across runs')
    parser.add_argument('--run-name', default=None,
                        help='Custom name for this experiment run')
    parser.add_argument('--output-dir', default=None,
                        help='Explicit directory to save results, checkpoints, and metrics')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases experiment logging')
    parser.add_argument('--wandb-offline', action='store_true',
                        help='Log to Weights & Biases in offline mode')
    parser.add_argument('--wandb-project', default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', default=None,
                        help='Weights & Biases entity/username')
    parser.add_argument('--wandb-base-url', default=None,
                        help='Weights & Biases base API URL (e.g. https://api.forge.coreweave.com)')
    parser.add_argument('--wandb-group', default=None,
                        help='Weights & Biases experiment group')
    parser.add_argument('--wandb-tags', default=None,
                        help='Comma-separated tags for Weights & Biases')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
