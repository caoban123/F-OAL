"""
Execution script for C2 Matched-Input Backbone Diagnostics.

Extracts features from B0 (vit_base_patch16_224.augreg_in1k) and B1 (vit_base_patch16_224.augreg2_in21k_ft_in1k)
on the exact same CIFAR-100 images, evaluates representation geometry (L2 norms, centered covariance rank,
class separation via cosine similarity), and evaluates downstream projection activations under identical
projection matrix P (Projection Equality, seed 42, D=1000, official_uniform).
"""

import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.foal.backbone import FrozenViT
from src.foal.datasets import CIFAR100OnlineDataset, IndexedImageDataset
from src.foal.diagnostics import compare_c2_backbone_conditions


def parse_args():
    parser = argparse.ArgumentParser(description="Run C2 Matched-Input Backbone Diagnostics")
    parser.add_argument("--output-dir", default="results/c2_diagnostics", help="Directory to save diagnostics JSON")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to evaluate (default 200)")
    parser.add_argument("--batch-size", type=int, default=50, help="Extraction batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running C2 Diagnostics on device: {device}")

    # Load CIFAR-100 dataset
    dataset_handler = CIFAR100OnlineDataset(root="artifacts/data", seed=42)
    train_source = dataset_handler.get_train_source()
    mapping = dataset_handler.get_label_mapping()
    transform = dataset_handler.get_transforms()

    # Deterministic subset of N samples
    chosen_indices = list(range(min(args.num_samples, len(train_source))))
    subset_ds = IndexedImageDataset(train_source, chosen_indices, mapping, transform)
    loader = DataLoader(subset_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Loaded {len(subset_ds)} CIFAR-100 samples for matched-input diagnostics.")

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

    # Save JSON report
    report_path = out_dir / "c2_matched_backbone_diagnostics.json"
    report_path.write_text(json.dumps(diag_report, indent=2), encoding="utf-8")
    print(f"Saved JSON report to: {report_path}")

    # Print summary table
    b0_res = diag_report["b0_augreg_in1k"]
    b1_res = diag_report["b1_augreg2_in21k_ft_in1k"]

    print("\n" + "=" * 75)
    print("C2 MATCHED-BACKBONE DIAGNOSTICS SUMMARY")
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


if __name__ == "__main__":
    main()
