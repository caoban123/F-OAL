"""Master execution suite for all DTD experiments in F-OAL.

Executes:
1. Table 4 Ablation Study:
   - A0: Full F-OAL (FF=True, SP=True, D=1000)
   - A1: w/o Feature Fusion (FF=False, SP=True, D=1000)
   - A2: w/o Smooth Projection (FF=True, SP=False, D=768)
   - A3: w/o Both (FF=False, SP=False, D=768)
2. Diagnostic C1 (Projection Initialization):
   - P0: Gaussian std=1.0
   - P1: Official Uniform U[-1/sqrt(d), 1/sqrt(d)]
   - P2: Matched Gaussian N(0, 1/(3d))
3. Diagnostic C2 (Backbone Pretraining):
   - B0: augreg_in1k (ImageNet-1K)
   - B1: augreg2_in21k_ft_in1k (ImageNet-21K -> 1K)
4. Diagnostic C3 (Preprocessing Alignment):
   - C3-A: Bicubic, Normalized [-1, 1]
   - C3-D: Bilinear, Raw [0, 1]
5. Multi-Seed Stability:
   - Seeds: 42, 123, 2026 (Mean +/- Std)

Uses cached features from results/dtd_feature_cache/ for execution.
Saves all metrics, task matrices, metadata, and checkpoints into results/dtd_experiments/.
"""
from __future__ import annotations
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import csv
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.foal.projection import SmoothProjection
from src.foal.rls import OnlineRidgeRLS
from src.foal.metrics import (
    summary_from_task_matrix,
    format_summary_percentages,
    aggregate_multi_seed_metrics,
)
from src.foal.environment import get_environment_info
from src.foal.dtd_datasets import DTD_CLASS_NAMES


def run_experiment_on_cached_features(
    *,
    train_cache_path: Path,
    test_cache_path: Path,
    class_order: list[int],
    classes_per_task: int = 4,
    use_projection: bool = True,
    projection_dim: int = 1000,
    projection_init: str = 'official_uniform',
    projection_seed: int = 42,
    gaussian_std: float = 1.0,
    gamma: float = 1.0,
    batch_size: int = 10,
    stream_seed: int = 42,
    device: str = 'cpu',
) -> tuple[dict, list[list[float]], float]:
    """Runs sequential 10-task online class-incremental learning on pre-cached features."""
    train_data = torch.load(train_cache_path, map_location='cpu', weights_only=False)
    test_data = torch.load(test_cache_path, map_location='cpu', weights_only=False)

    train_feats = train_data['features']
    train_labels = train_data['labels']
    test_feats = test_data['features']
    test_labels = test_data['labels']

    input_dim = train_feats.shape[1]

    # Initialize projection
    if use_projection:
        projection = SmoothProjection(
            input_dim=input_dim,
            output_dim=projection_dim,
            seed=projection_seed,
            gaussian_std=gaussian_std,
            initialization=projection_init,
        ).to(device)
        classifier_dim = projection_dim
    else:
        projection = None
        classifier_dim = input_dim

    # Initialize classifier
    classifier = OnlineRidgeRLS(
        feature_dim=classifier_dim,
        gamma=gamma,
        device=device,
        dtype=torch.float64,
    )

    # Class mapping: original class id -> incremental class index [0..39]
    label_to_incr = {orig: idx for idx, orig in enumerate(class_order)}
    num_tasks = len(class_order) // classes_per_task

    matrix = []
    seen_classes = []
    t0 = time.monotonic()

    for task_idx in range(num_tasks):
        start_cls = task_idx * classes_per_task
        end_cls = start_cls + classes_per_task
        task_classes = class_order[start_cls:end_cls]
        seen_classes.extend(task_classes)

        # Expand classifier output dimensions
        classifier.expand_classes(len(seen_classes))

        # Filter training samples for current task
        task_train_mask = torch.tensor([l.item() in task_classes for l in train_labels])
        t_feats = train_feats[task_train_mask]
        t_labels_orig = train_labels[task_train_mask]
        t_labels_incr = torch.tensor([label_to_incr[l.item()] for l in t_labels_orig], dtype=torch.long)

        # DataLoader with reproducible shuffling
        gen = torch.Generator().manual_seed(stream_seed + task_idx)
        dataset = TensorDataset(t_feats, t_labels_incr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False)

        # Update classifier batch-by-batch
        with torch.no_grad():
            for h_batch, y_batch in loader:
                h_dev = h_batch.to(device)
                z_dev = projection(h_dev) if projection is not None else h_dev
                classifier.update(z_dev, y_batch.to(device))

        # Evaluate on all seen tasks
        stage_accs = []
        for prev_task_idx in range(task_idx + 1):
            prev_start = prev_task_idx * classes_per_task
            prev_end = prev_start + classes_per_task
            prev_classes = set(class_order[prev_start:prev_end])

            prev_test_mask = torch.tensor([l.item() in prev_classes for l in test_labels])
            p_feats = test_feats[prev_test_mask]
            p_labels_orig = test_labels[prev_test_mask]
            p_labels_incr = torch.tensor([label_to_incr[l.item()] for l in p_labels_orig], dtype=torch.long)

            correct = 0
            total = len(p_labels_incr)
            with torch.no_grad():
                for start in range(0, total, 64):
                    h_b = p_feats[start:start+64].to(device)
                    y_b = p_labels_incr[start:start+64]
                    z_b = projection(h_b) if projection is not None else h_b
                    preds = classifier.predict(z_b).cpu()
                    correct += int((preds == y_b).sum())
            stage_accs.append(correct / total if total > 0 else 0.0)

        matrix.append(stage_accs)

    elapsed = round(time.monotonic() - t0, 4)
    metrics = summary_from_task_matrix(matrix)
    metrics['execution_seconds'] = elapsed
    return metrics, matrix, elapsed


def save_run_artifacts(out_dir: Path, run_id: str, metadata: dict, metrics: dict, matrix: list[list[float]]):
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {'task_matrix': matrix, **metrics}
    (run_dir / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    with (run_dir / 'task_accuracy.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'task', 'accuracy'])
        for stage, row in enumerate(matrix, start=1):
            for t_idx, val in enumerate(row, start=1):
                writer.writerow([stage, t_idx, val])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default='results/dtd_feature_cache')
    parser.add_argument('--out-dir', default='results/dtd_experiments')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = cache_dir / 'cache_manifest.json'
    if not manifest_file.exists():
        raise FileNotFoundError(f"Cache manifest not found at {manifest_file}. Run prepare_dtd_cache.py first!")

    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    class_order = manifest['class_order']
    env_info = get_environment_info()

    print("=" * 80)
    print("EXECUTING COMPREHENSIVE DTD EXPERIMENTAL SUITE")
    print("=" * 80)

    all_results = []

    # =========================================================================
    # PART 1: TABLE 4 ABLATION STUDY (Primary Paper Table 4 Reproduction)
    # =========================================================================
    print("\n>>> PART 1: TABLE 4 ABLATION STUDY (2x2 Factorial Decomposition) <<<")
    ablation_configs = [
        ('A0_Full_FOAL', 'train_augreg2_bilinear_raw_ff.pt', 'test_augreg2_bilinear_raw_ff.pt', True, 1000, 'official_uniform'),
        ('A1_without_FF', 'train_augreg2_bilinear_raw_noff.pt', 'test_augreg2_bilinear_raw_noff.pt', True, 1000, 'official_uniform'),
        ('A2_without_SP', 'train_augreg2_bilinear_raw_ff.pt', 'test_augreg2_bilinear_raw_ff.pt', False, 768, 'none'),
        ('A3_without_Both', 'train_augreg2_bilinear_raw_noff.pt', 'test_augreg2_bilinear_raw_noff.pt', False, 768, 'none'),
    ]

    for run_id, tr_file, te_file, use_sp, dim, pinit in ablation_configs:
        tr_path = cache_dir / tr_file
        te_path = cache_dir / te_file
        metrics, matrix, elapsed = run_experiment_on_cached_features(
            train_cache_path=tr_path,
            test_cache_path=te_path,
            class_order=class_order,
            use_projection=use_sp,
            projection_dim=dim,
            projection_init=pinit if use_sp else 'official_uniform',
            device=args.device,
        )
        meta = {
            'experiment_type': 'table4_ablation',
            'run_id': run_id,
            'feature_fusion': '_ff.pt' in tr_file,
            'smooth_projection': use_sp,
            'dimension': dim,
            'projection_init': pinit,
            'backbone': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            'preprocessing': 'bilinear_raw',
            'seed': 42,
            'environment': env_info,
        }
        save_run_artifacts(out_dir, run_id, meta, metrics, matrix)
        p_res = {
            'category': 'Table 4 Ablation',
            'run_id': run_id,
            'FF': 'ON' if '_ff.pt' in tr_file else 'OFF',
            'SP': 'ON' if use_sp else 'OFF',
            'Dim': dim,
            'A_avg': f"{metrics['average_incremental_accuracy']*100:.2f}%",
            'A_last': f"{metrics['last_accuracy']*100:.2f}%",
            'F_past': f"{metrics['forgetting_past_tasks']*100:.2f}%",
            'F_code': f"{metrics['forgetting_official_code']*100:.2f}%",
            'Time_s': f"{elapsed:.2f}s",
        }
        all_results.append(p_res)
        print(f"  {run_id:<20} | A_avg: {p_res['A_avg']} | A_last: {p_res['A_last']} | F_past: {p_res['F_past']} | Time: {p_res['Time_s']}")

    # =========================================================================
    # PART 2: DIAGNOSTIC C1 (Smooth Projection Initialization Investigation)
    # =========================================================================
    print("\n>>> PART 2: DIAGNOSTIC C1 (Projection Initialization Investigation) <<<")
    c1_configs = [
        ('C1_P0_Gaussian_std1', 'gaussian_std1', 1.0),
        ('C1_P1_Uniform_Official', 'official_uniform', 1.0),
        ('C1_P2_Gaussian_Matched', 'gaussian_matched_variance', 1.0),
    ]
    tr_path = cache_dir / 'train_augreg2_bilinear_raw_ff.pt'
    te_path = cache_dir / 'test_augreg2_bilinear_raw_ff.pt'

    for run_id, pinit, gstd in c1_configs:
        metrics, matrix, elapsed = run_experiment_on_cached_features(
            train_cache_path=tr_path,
            test_cache_path=te_path,
            class_order=class_order,
            use_projection=True,
            projection_dim=1000,
            projection_init=pinit,
            gaussian_std=gstd,
            device=args.device,
        )
        meta = {
            'experiment_type': 'c1_projection_initialization',
            'run_id': run_id,
            'projection_init': pinit,
            'gaussian_std': gstd,
            'seed': 42,
            'environment': env_info,
        }
        save_run_artifacts(out_dir, run_id, meta, metrics, matrix)
        p_res = {
            'category': 'C1 Projection Init',
            'run_id': run_id,
            'FF': 'ON',
            'SP': 'ON',
            'Dim': 1000,
            'A_avg': f"{metrics['average_incremental_accuracy']*100:.2f}%",
            'A_last': f"{metrics['last_accuracy']*100:.2f}%",
            'F_past': f"{metrics['forgetting_past_tasks']*100:.2f}%",
            'F_code': f"{metrics['forgetting_official_code']*100:.2f}%",
            'Time_s': f"{elapsed:.2f}s",
        }
        all_results.append(p_res)
        print(f"  {run_id:<25} | A_avg: {p_res['A_avg']} | A_last: {p_res['A_last']} | F_past: {p_res['F_past']} | Time: {p_res['Time_s']}")

    # =========================================================================
    # PART 3: DIAGNOSTIC C2 (Pretrained Backbone Checkpoint Alignment)
    # =========================================================================
    print("\n>>> PART 3: DIAGNOSTIC C2 (Backbone Checkpoint Alignment) <<<")
    c2_configs = [
        ('C2_B0_augreg_in1k', 'train_augreg_in1k_bilinear_raw_ff.pt', 'test_augreg_in1k_bilinear_raw_ff.pt'),
        ('C2_B1_augreg2_in21k_ft_in1k', 'train_augreg2_bilinear_raw_ff.pt', 'test_augreg2_bilinear_raw_ff.pt'),
    ]

    for run_id, tr_file, te_file in c2_configs:
        tr_path = cache_dir / tr_file
        te_path = cache_dir / te_file
        if not (tr_path.exists() and te_path.exists()):
            print(f"  [Skipped] Cache files for {run_id} not available.")
            continue
        metrics, matrix, elapsed = run_experiment_on_cached_features(
            train_cache_path=tr_path,
            test_cache_path=te_path,
            class_order=class_order,
            use_projection=True,
            projection_dim=1000,
            projection_init='official_uniform',
            device=args.device,
        )
        meta = {
            'experiment_type': 'c2_backbone_alignment',
            'run_id': run_id,
            'backbone': 'augreg_in1k' if 'in1k' in tr_file and 'in21k' not in tr_file else 'augreg2_in21k_ft_in1k',
            'seed': 42,
            'environment': env_info,
        }
        save_run_artifacts(out_dir, run_id, meta, metrics, matrix)
        p_res = {
            'category': 'C2 Backbone Alignment',
            'run_id': run_id,
            'FF': 'ON',
            'SP': 'ON',
            'Dim': 1000,
            'A_avg': f"{metrics['average_incremental_accuracy']*100:.2f}%",
            'A_last': f"{metrics['last_accuracy']*100:.2f}%",
            'F_past': f"{metrics['forgetting_past_tasks']*100:.2f}%",
            'F_code': f"{metrics['forgetting_official_code']*100:.2f}%",
            'Time_s': f"{elapsed:.2f}s",
        }
        all_results.append(p_res)
        print(f"  {run_id:<25} | A_avg: {p_res['A_avg']} | A_last: {p_res['A_last']} | F_past: {p_res['F_past']} | Time: {p_res['Time_s']}")

    # =========================================================================
    # PART 4: DIAGNOSTIC C3 (Preprocessing Alignment)
    # =========================================================================
    print("\n>>> PART 4: DIAGNOSTIC C3 (Preprocessing Alignment) <<<")
    c3_configs = [
        ('C3_A_Bicubic_Norm', 'train_augreg2_bicubic_norm_ff.pt', 'test_augreg2_bicubic_norm_ff.pt'),
        ('C3_D_Bilinear_Raw', 'train_augreg2_bilinear_raw_ff.pt', 'test_augreg2_bilinear_raw_ff.pt'),
    ]

    for run_id, tr_file, te_file in c3_configs:
        tr_path = cache_dir / tr_file
        te_path = cache_dir / te_file
        if not (tr_path.exists() and te_path.exists()):
            continue
        metrics, matrix, elapsed = run_experiment_on_cached_features(
            train_cache_path=tr_path,
            test_cache_path=te_path,
            class_order=class_order,
            use_projection=True,
            projection_dim=1000,
            projection_init='official_uniform',
            device=args.device,
        )
        meta = {
            'experiment_type': 'c3_preprocessing_alignment',
            'run_id': run_id,
            'preprocessing': 'bicubic_norm' if 'bicubic' in tr_file else 'bilinear_raw',
            'seed': 42,
            'environment': env_info,
        }
        save_run_artifacts(out_dir, run_id, meta, metrics, matrix)
        p_res = {
            'category': 'C3 Preprocessing',
            'run_id': run_id,
            'FF': 'ON',
            'SP': 'ON',
            'Dim': 1000,
            'A_avg': f"{metrics['average_incremental_accuracy']*100:.2f}%",
            'A_last': f"{metrics['last_accuracy']*100:.2f}%",
            'F_past': f"{metrics['forgetting_past_tasks']*100:.2f}%",
            'F_code': f"{metrics['forgetting_official_code']*100:.2f}%",
            'Time_s': f"{elapsed:.2f}s",
        }
        all_results.append(p_res)
        print(f"  {run_id:<25} | A_avg: {p_res['A_avg']} | A_last: {p_res['A_last']} | F_past: {p_res['F_past']} | Time: {p_res['Time_s']}")

    # =========================================================================
    # PART 5: MULTI-SEED REPLICATION (Seeds 42, 123, 2026)
    # =========================================================================
    print("\n>>> PART 5: MULTI-SEED REPLICATION (Seeds 42, 123, 2026) <<<")
    tr_path = cache_dir / 'train_augreg2_bilinear_raw_ff.pt'
    te_path = cache_dir / 'test_augreg2_bilinear_raw_ff.pt'

    multi_seed_metrics = []
    for seed in [42, 123, 2026]:
        run_id = f"MultiSeed_seed{seed}"
        metrics, matrix, elapsed = run_experiment_on_cached_features(
            train_cache_path=tr_path,
            test_cache_path=te_path,
            class_order=class_order,
            use_projection=True,
            projection_dim=1000,
            projection_init='official_uniform',
            projection_seed=seed,
            stream_seed=seed,
            device=args.device,
        )
        multi_seed_metrics.append(metrics)
        meta = {
            'experiment_type': 'multi_seed',
            'run_id': run_id,
            'seed': seed,
            'environment': env_info,
        }
        save_run_artifacts(out_dir, run_id, meta, metrics, matrix)
        p_res = {
            'category': 'Multi-Seed',
            'run_id': run_id,
            'FF': 'ON',
            'SP': 'ON',
            'Dim': 1000,
            'A_avg': f"{metrics['average_incremental_accuracy']*100:.2f}%",
            'A_last': f"{metrics['last_accuracy']*100:.2f}%",
            'F_past': f"{metrics['forgetting_past_tasks']*100:.2f}%",
            'F_code': f"{metrics['forgetting_official_code']*100:.2f}%",
            'Time_s': f"{elapsed:.2f}s",
        }
        all_results.append(p_res)
        print(f"  {run_id:<25} | A_avg: {p_res['A_avg']} | A_last: {p_res['A_last']} | F_past: {p_res['F_past']} | Time: {p_res['Time_s']}")

    agg = aggregate_multi_seed_metrics(multi_seed_metrics)
    print(f"\nMulti-Seed Aggregation (N=3):")
    print(f"  A_avg Mean +/- Std: {agg.get('average_incremental_accuracy_display')}")
    print(f"  A_last Mean +/- Std: {agg.get('last_accuracy_display')}")
    print(f"  F_past Mean +/- Std: {agg.get('forgetting_past_tasks_display')}")

    # =========================================================================
    # SUMMARY TABLE AND EXPORT
    # =========================================================================
    summary_path = out_dir / 'dtd_experiment_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'run_id', 'FF', 'SP', 'Dim', 'A_avg', 'A_last', 'F_past', 'F_code', 'Time_s'])
        writer.writeheader()
        writer.writerows(all_results)

    json_summary = {
        'results': all_results,
        'multi_seed_aggregation': agg,
        'paper_ground_truth': {
            'dataset': 'DTD',
            'full_foal_a_avg': 82.8,
            'without_ff_a_avg': 80.5,
            'without_sp_a_avg': 79.3,
            'without_both_a_avg': 71.2,
            'ease_a_avg': 76.0,
            'slca_a_avg': 72.8,
            'training_time_seconds': 8,
        }
    }
    (out_dir / 'dtd_experiment_summary.json').write_text(json.dumps(json_summary, indent=2), encoding='utf-8')
    print("\n" + "=" * 80)
    print(f"SUMMARY SAVED TO: {summary_path.resolve()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
