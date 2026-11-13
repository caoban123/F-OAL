"""Run matched-input Smooth Projection diagnostics on real DTD ViT features.

Extracts frozen `vit_base_patch16_224.augreg_in1k` feature vectors h from a fixed,
deterministic, representative multi-class subset of DTD training data
(default: 200 samples = 20 samples/class across first 10 classes of the DTD 40-class order, seed=42),
and evaluates P0 (`gaussian_std1`), P1 (`official_uniform`), and P2 (`gaussian_matched_variance`)
on the exact same feature tensor h.

Usage:
    python -m scripts.run_c1_dtd_diagnostics --output-dir results/dtd_c1_diagnostics
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.foal.backbone import FrozenViT
from src.foal.datasets import (
    IndexedImageDataset,
    cifar100_transforms,
)
from src.foal.dtd_datasets import (
    load_dtd,
    make_dtd_class_order,
    make_dtd_task_specs,
)
from src.foal.diagnostics import compare_c1_projections_on_features
from src.foal.environment import get_environment_info


def select_deterministic_multiclass_subset(
    train_labels: list[int] | np.ndarray,
    class_order: list[int],
    num_classes_subset: int = 10,
    samples_per_class: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Select a balanced, deterministic subset of DTD training indices across multiple classes."""
    labels_arr = np.asarray(train_labels)
    selected_classes = list(class_order[:num_classes_subset])
    rng = np.random.default_rng(seed)

    selected_indices: list[int] = []
    per_class_counts: dict[int, int] = {}
    for cls_id in selected_classes:
        cls_indices = np.flatnonzero(labels_arr == cls_id).tolist()
        chosen = rng.choice(cls_indices, size=min(samples_per_class, len(cls_indices)), replace=False).tolist()
        selected_indices.extend(int(i) for i in chosen)
        per_class_counts[int(cls_id)] = len(chosen)

    indices_sha = hashlib.sha256(json.dumps(selected_indices).encode('utf-8')).hexdigest()
    return {
        'selected_classes': selected_classes,
        'samples_per_class': int(samples_per_class),
        'total_samples': len(selected_indices),
        'selected_indices': selected_indices,
        'selected_indices_sha256': indices_sha,
        'per_class_counts': per_class_counts,
        'split': 'train_only_no_test_leakage',
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_source, test_source = load_dtd()
    order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=args.class_order_seed)
    _ = make_dtd_task_specs(
        train_source['fine_label'],
        test_source['fine_label'],
        order,
        classes_per_task=4,
        samples_per_class=40,
        seed=args.class_order_seed,
    )
    mapping = {orig: i for i, orig in enumerate(order)}

    subset_info = select_deterministic_multiclass_subset(
        train_labels=train_source['fine_label'],
        class_order=order,
        num_classes_subset=args.num_classes_subset,
        samples_per_class=args.samples_per_class,
        seed=args.subset_seed,
    )

    trans = cifar100_transforms(size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation='bicubic')
    ds = IndexedImageDataset(train_source, subset_info['selected_indices'], mapping, trans)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    encoder = FrozenViT(name=args.backbone, pretrained=True, feature_fusion=True).to(device)
    encoder.eval()

    feat_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for x, _ in loader:
            feat_chunks.append(encoder(x.to(device)).cpu())
    features = torch.cat(feat_chunks, dim=0)

    comparison = compare_c1_projections_on_features(
        features=features,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        gamma=args.gamma,
    )

    env_info = get_environment_info()
    report = {
        'experiment': 'C1_DTD_Matched_Input_Projection_Diagnostics',
        'dataset': 'tanganke/dtd',
        'backbone': args.backbone,
        'feature_fusion': True,
        'preprocessing': {
            'image_size': 224,
            'interpolation': 'bicubic',
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        },
        'seeds': {
            'class_order_seed': int(args.class_order_seed),
            'subset_seed': int(args.subset_seed),
            'projection_seed': int(args.projection_seed),
        },
        'subset': subset_info,
        'diagnostics': comparison,
        'environment': env_info,
    }

    json_path = outdir / 'c1_dtd_matched_input_diagnostics.json'
    json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    csv_path = outdir / 'c1_dtd_matched_input_diagnostics_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'config_id',
            'initialization',
            'theoretical_std',
            'measured_weight_std',
            'pre_sigmoid_mean',
            'pre_sigmoid_std',
            'pre_sigmoid_min',
            'pre_sigmoid_max',
            'pre_sigmoid_q01',
            'pre_sigmoid_q50',
            'pre_sigmoid_q99',
            'post_sigmoid_mean',
            'post_sigmoid_std',
            'frac_z_lt_0_01',
            'frac_z_gt_0_99',
            'total_saturation_frac',
            'near_const_std_lt_1e3_count',
            'reg_gram_cond_number',
            'effective_rank_unreg',
            'effective_rank_reg',
            'matrix_sha256',
        ])
        label_map = {
            'gaussian_std1': 'P0_gaussian_std1',
            'official_uniform': 'P1_official_uniform',
            'gaussian_matched_variance': 'P2_gaussian_matched_variance',
        }
        for init_name, d in comparison['configurations'].items():
            pm = d['projection_metadata']
            u = d['pre_sigmoid_u']
            z = d['post_sigmoid_z']
            ns = d['numerical_stability']
            writer.writerow([
                label_map.get(init_name, init_name),
                init_name,
                f"{pm['theoretical_std']:.8f}",
                f"{pm['measured_std']:.8f}",
                f"{u['mean']:.6f}",
                f"{u['std']:.6f}",
                f"{u['min']:.6f}",
                f"{u['max']:.6f}",
                f"{u['quantiles']['q01']:.6f}",
                f"{u['quantiles']['q50']:.6f}",
                f"{u['quantiles']['q99']:.6f}",
                f"{z['mean']:.6f}",
                f"{z['std']:.6f}",
                f"{z['fraction_z_lt_0_01']:.6f}",
                f"{z['fraction_z_gt_0_99']:.6f}",
                f"{z['total_saturation_fraction']:.6f}",
                z['near_constant_features_std_lt_1e3_count'],
                f"{ns['regularized_gram_condition_number']:.4f}",
                f"{ns['effective_rank_unregularized']:.2f}",
                f"{ns['effective_rank_regularized']:.2f}",
                pm['matrix_sha256'],
            ])

    print('\n' + '=' * 105)
    print(f"C1 DTD MATCHED-INPUT PROJECTION DIAGNOSTICS (N={subset_info['total_samples']} real DTD images, {args.backbone})")
    print('=' * 105)
    print(f"{'Configuration':<30} {'Weight Std':>11} {'u=hP Std':>10} {'z=sig(u) Std':>13} {'Sat (<0.01|>0.99)':>18} {'EffRank(G)':>11} {'Cond(G+I)':>12}")
    print('-' * 105)
    for init_name, d in comparison['configurations'].items():
        pm = d['projection_metadata']
        u = d['pre_sigmoid_u']
        z = d['post_sigmoid_z']
        ns = d['numerical_stability']
        sat_pct = z['total_saturation_fraction'] * 100.0
        print(
            f"{init_name:<30} {pm['measured_std']:>11.6f} {u['std']:>10.4f} "
            f"{z['std']:>13.4f} {sat_pct:>17.2f}% "
            f"{ns['effective_rank_unregularized']:>11.2f} {ns['regularized_gram_condition_number']:>12.2f}"
        )
    print('=' * 105)
    print(f'Saved JSON report to: {json_path}')
    print(f'Saved CSV summary to: {csv_path}\n')
    return report


def main():
    p = argparse.ArgumentParser(description='Run matched-input Smooth Projection diagnostics on DTD.')
    p.add_argument('--backbone', default='vit_base_patch16_224.augreg_in1k')
    p.add_argument('--projection-dim', type=int, default=1000)
    p.add_argument('--projection-seed', type=int, default=42)
    p.add_argument('--class-order-seed', type=int, default=42)
    p.add_argument('--subset-seed', type=int, default=42)
    p.add_argument('--num-classes-subset', type=int, default=10)
    p.add_argument('--samples-per-class', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=20)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--output-dir', default='results/dtd_c1_diagnostics')
    run_diagnostics(p.parse_args())


if __name__ == '__main__':
    main()
