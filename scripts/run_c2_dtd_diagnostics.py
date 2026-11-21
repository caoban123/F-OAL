"""Execution script for C2 Matched-Input Backbone Diagnostics on DTD.

Extracts features from B0 (vit_base_patch16_224.augreg_in1k) and B1 (vit_base_patch16_224.augreg2_in21k_ft_in1k)
on the exact same DTD images, evaluates representation geometry (L2 norms, centered covariance rank,
class separation via cosine similarity), and evaluates downstream projection activations under identical
projection matrix P (Projection Equality, seed 42, D=1000, official_uniform).
"""

from __future__ import annotations

import argparse
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
from src.foal.datasets import IndexedImageDataset, cifar100_transforms
from src.foal.dtd_datasets import load_dtd, make_dtd_class_order
from src.foal.diagnostics import compare_c2_backbone_conditions
from src.foal.environment import get_environment_info


def parse_args():
    parser = argparse.ArgumentParser(description="Run C2 Matched-Input Backbone Diagnostics on DTD")
    parser.add_argument("--output-dir", default="results/dtd_c2_diagnostics", help="Directory to save diagnostics JSON")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to evaluate (default 200)")
    parser.add_argument("--samples-per-class", type=int, default=20, help="Samples per class if stratified")
    parser.add_argument("--num-classes-subset", type=int, default=10, help="Number of classes in subset")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--batch-size", type=int, default=50, help="Extraction batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running C2 DTD Diagnostics on device: {device}")

    # Load DTD dataset
    train_source, _ = load_dtd()
    order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=args.seed)
    mapping = {orig: i for i, orig in enumerate(order)}
    transform = cifar100_transforms(size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation="bicubic")

    # Select balanced deterministic subset across first classes of DTD order
    labels_arr = np.asarray(train_source['fine_label'])
    selected_classes = list(order[:args.num_classes_subset])
    rng = np.random.default_rng(args.seed)

    selected_indices: list[int] = []
    for cls_id in selected_classes:
        cls_indices = np.flatnonzero(labels_arr == cls_id).tolist()
        chosen = rng.choice(cls_indices, size=min(args.samples_per_class, len(cls_indices)), replace=False).tolist()
        selected_indices.extend(int(i) for i in chosen)

    if len(selected_indices) > args.num_samples:
        selected_indices = selected_indices[:args.num_samples]

    subset_ds = IndexedImageDataset(train_source, selected_indices, mapping, transform)
    loader = DataLoader(subset_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Loaded {len(subset_ds)} DTD samples across {len(selected_classes)} classes for matched-input diagnostics.")

    # 1. Extract B0 features
    print("\n--- Extracting B0 Features (vit_base_patch16_224.augreg_in1k) ---")
    b0_vit = FrozenViT("vit_base_patch16_224.augreg_in1k", pretrained=True, feature_fusion=True).to(device)
    b0_feats_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            b0_feats_list.append(b0_vit(x.to(device)).cpu())
            labels_list.append(y.cpu())
    b0_feats = torch.cat(b0_feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    print(f"B0 features extracted: {b0_feats.shape}")
    del b0_vit
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. Extract B1 features
    print("\n--- Extracting B1 Features (vit_base_patch16_224.augreg2_in21k_ft_in1k) ---")
    b1_vit = FrozenViT("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, feature_fusion=True).to(device)
    b1_feats_list = []
    with torch.no_grad():
        for x, _ in loader:
            b1_feats_list.append(b1_vit(x.to(device)).cpu())
    b1_feats = torch.cat(b1_feats_list, dim=0)
    print(f"B1 features extracted: {b1_feats.shape}")
    del b1_vit
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Run comparative diagnostics
    print("\n--- Computing Comparative Diagnostics ---")
    diag_report = compare_c2_backbone_conditions(
        b0_features=b0_feats,
        b1_features=b1_feats,
        labels=labels,
        projection_dim=1000,
        projection_seed=42,
        gamma=1.0,
    )
    diag_report["environment"] = get_environment_info()
    diag_report["dataset"] = "tanganke/dtd"
    diag_report["num_samples"] = len(subset_ds)
    diag_report["selected_indices_sha256"] = hashlib.sha256(json.dumps(selected_indices).encode('utf-8')).hexdigest()

    # Save JSON report
    report_path = out_dir / "c2_dtd_matched_backbone_diagnostics.json"
    report_path.write_text(json.dumps(diag_report, indent=2), encoding="utf-8")
    print(f"Saved JSON report to: {report_path}")

    # Print summary table
    b0_res = diag_report["b0_augreg_in1k"]
    b1_res = diag_report["b1_augreg2_in21k_ft_in1k"]

    print("\n" + "=" * 75)
    print("C2 DTD MATCHED-BACKBONE DIAGNOSTICS SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<35} | {'B0 (augreg_in1k)':<18} | {'B1 (augreg2_in21k_ft)':<18}")
    print("-" * 75)
    print(f"{'L2 Norm (Mean ± Std)':<35} | {b0_res['features']['l2_norm']['mean']:.2f} ± {b0_res['features']['l2_norm']['std']:.2f} | {b1_res['features']['l2_norm']['mean']:.2f} ± {b1_res['features']['l2_norm']['std']:.2f}")
    print(f"{'Centered Covariance Erank':<35} | {b0_res['features']['centered_covariance']['effective_rank']:.2f} | {b1_res['features']['centered_covariance']['effective_rank']:.2f}")
    if b0_res["features"]["class_separation"]:
        print(f"{'Intra-Class Cosine Sim':<35} | {b0_res['features']['class_separation']['intra_class_cosine_mean']:.4f} | {b1_res['features']['class_separation']['intra_class_cosine_mean']:.4f}")
        print(f"{'Inter-Class Cosine Sim':<35} | {b0_res['features']['class_separation']['inter_class_cosine_mean']:.4f} | {b1_res['features']['class_separation']['inter_class_cosine_mean']:.4f}")
        print(f"{'Class Separation Ratio (Intra/Inter)':<35} | {b0_res['features']['class_separation']['separation_ratio']:.3f} | {b1_res['features']['class_separation']['separation_ratio']:.3f}")
    print(f"{'Pre-sigmoid u Std':<35} | {b0_res['projection']['pre_sigmoid_u']['std']:.4f} | {b1_res['projection']['pre_sigmoid_u']['std']:.4f}")
    print(f"{'Sigmoid Saturation (<0.01 | >0.99)':<35} | {b0_res['projection']['post_sigmoid_z']['total_saturation_fraction']*100:.2f}% | {b1_res['projection']['post_sigmoid_z']['total_saturation_fraction']*100:.2f}%")
    print(f"{'Regularized Gram Cond Number':<35} | {b0_res['projection']['numerical_stability']['regularized_gram_condition_number']:.2f} | {b1_res['projection']['numerical_stability']['regularized_gram_condition_number']:.2f}")
    print(f"{'Projection Matrix Equality SHA256':<35} | {diag_report['projection_matrix_sha256'][:16]}... | {diag_report['projection_matrix_sha256'][:16]}...")
    print("=" * 75)
    return diag_report


if __name__ == "__main__":
    main()
