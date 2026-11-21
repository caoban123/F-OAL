"""C3 Preprocessing Alignment Diagnostics on Matched DTD Samples.

Evaluates real DTD test samples across the 2x2 factorial preprocessing design:
  - C3-A: Bicubic, Normalize [-1, 1]
  - C3-B: Bilinear, Normalize [-1, 1]
  - C3-C: Bicubic, Raw [0, 1]
  - C3-D: Bilinear, Raw [0, 1]
Also compares C3-D against the official author transform.

Records:
  - Exact sample indices, selection seed (42), and indices SHA256
  - True class labels, count of unique classes, and class composition
  - Mathematical effective rank with explicit sample-size bounds (min(N-1, d))
  - Pre-activation statistics and Sigmoid saturation fraction
  - Pairwise cosine similarities between preprocessing conditions
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torchvision.transforms as tv_transforms

from src.foal.dtd_datasets import load_dtd
from src.foal.datasets import cifar100_transforms
from src.foal.backbone import FrozenViT
from src.foal.projection import SmoothProjection
from src.foal.diagnostics import compute_backbone_feature_diagnostics, compute_projection_diagnostics
from src.foal.environment import get_environment_info


def get_official_author_transform():
    """Official author transform from setup_elements.py."""
    return tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
    ])


def run_diagnostics(num_samples: int = 200, seed: int = 42, device: str = 'cpu', output_path: str | None = None) -> dict:
    diag_type = "SMOKE_TEST_ONLY" if num_samples < 50 else "FULL_MATCHED_DIAGNOSTICS"
    
    # 1. Load DTD test data
    _, test_source = load_dtd()
    rng = np.random.default_rng(seed)
    total_avail = len(test_source)
    n_samples = min(num_samples, total_avail)
    sample_indices = rng.choice(total_avail, size=n_samples, replace=False).tolist()
    sample_indices_sha = hashlib.sha256(json.dumps(sample_indices).encode('utf-8')).hexdigest()
    
    # Extract labels and compute class composition
    sample_labels = [int(test_source[idx]['fine_label']) for idx in sample_indices]
    unique_classes = sorted(list(set(sample_labels)))
    class_counts = {int(k): int(v) for k, v in collections.Counter(sample_labels).items()}
    labels_tensor = torch.tensor(sample_labels, dtype=torch.long)
    
    print("=" * 80)
    print("C3 DTD PREPROCESSING MATCHED DIAGNOSTICS")
    print(f"Type              : {diag_type}")
    print(f"Sample Count (N)  : {n_samples}")
    print(f"Unique Classes    : {len(unique_classes)} / 47")
    print(f"Sample Seed       : {seed}")
    print(f"Indices SHA256    : {sample_indices_sha[:16]}...")
    print(f"Compute Device    : {device}")
    if n_samples < 50:
        print(f"WARNING: Smoke test mode (N={n_samples}). Effective rank centered covariance is mathematically")
        print(f"         bounded by min(N-1, d) = {min(n_samples-1, 768)}! Not representative of full feature space.")
    print("=" * 80)
    
    # 2. Build 4 condition transforms + official transform
    transforms_dict = {
        'C3-A': cifar100_transforms(size=224, interpolation='bicubic', normalize=True),
        'C3-B': cifar100_transforms(size=224, interpolation='bilinear', normalize=True),
        'C3-C': cifar100_transforms(size=224, interpolation='bicubic', normalize=False),
        'C3-D': cifar100_transforms(size=224, interpolation='bilinear', normalize=False),
        'Official': get_official_author_transform(),
    }
    
    images_pil = [test_source[idx]['img'].convert('RGB') for idx in sample_indices]
    
    # 3. Transform images and evaluate tensor stats
    tensors_dict = {}
    tensor_stats = {}
    for name, trans in transforms_dict.items():
        batch = torch.stack([trans(img) for img in images_pil], dim=0)  # [N, 3, 224, 224]
        tensors_dict[name] = batch
        tensor_stats[name] = {
            'min': float(batch.min()),
            'max': float(batch.max()),
            'mean': float(batch.mean()),
            'std': float(batch.std()),
        }
    
    # Check pixel difference between C3-D and Official
    diff_d_official = (tensors_dict['C3-D'] - tensors_dict['Official']).abs()
    max_d_official_diff = float(diff_d_official.max())
    mean_d_official_diff = float(diff_d_official.mean())
    print(f"C3-D vs Official transform max pixel abs diff: {max_d_official_diff:.2e}")
    print(f"C3-D vs Official transform mean pixel abs diff: {mean_d_official_diff:.2e}")
    
    # 4. Load ViT backbone
    backbone_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    print(f"Loading backbone: {backbone_name} ...")
    encoder = FrozenViT(name=backbone_name, pretrained=True, feature_fusion=True)
    encoder.eval()
    encoder.to(device)
    
    # 5. Extract features for all conditions
    features_dict = {}
    feature_stats = {}
    for name, batch_tensor in tensors_dict.items():
        feats_list = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                sub = batch_tensor[i:i+batch_size].to(device)
                f = encoder(sub).cpu()
                feats_list.append(f)
        features = torch.cat(feats_list, dim=0)  # [N, 768]
        features_dict[name] = features
        
        diag = compute_backbone_feature_diagnostics(features, labels=labels_tensor)
        feature_stats[name] = {
            'norm_mean': diag['l2_norm']['mean'],
            'norm_std': diag['l2_norm']['std'],
            'r_eff': diag['centered_covariance']['effective_rank'],
            'r_eff_max_bound': min(n_samples - 1, 768),
            'class_separation': diag.get('class_separation'),
        }
    
    # 6. Pairwise cosine similarity across conditions
    def pairwise_cosine(f1, f2):
        f1_norm = f1 / (torch.norm(f1, dim=-1, keepdim=True) + 1e-9)
        f2_norm = f2 / (torch.norm(f2, dim=-1, keepdim=True) + 1e-9)
        sims = (f1_norm * f2_norm).sum(dim=-1)
        return float(sims.mean()), float(sims.std()), float(sims.min())
    
    sim_a_b = pairwise_cosine(features_dict['C3-A'], features_dict['C3-B'])
    sim_a_c = pairwise_cosine(features_dict['C3-A'], features_dict['C3-C'])
    sim_a_d = pairwise_cosine(features_dict['C3-A'], features_dict['C3-D'])
    sim_b_d = pairwise_cosine(features_dict['C3-B'], features_dict['C3-D'])
    sim_c_d = pairwise_cosine(features_dict['C3-C'], features_dict['C3-D'])
    sim_d_off = pairwise_cosine(features_dict['C3-D'], features_dict['Official'])
    
    # 7. Smooth projection stats
    proj = SmoothProjection(
        input_dim=768,
        output_dim=1000,
        initialization='official_uniform',
        seed=42,
    )
    p_meta = proj.get_metadata()
    p_sha = p_meta['matrix_sha256']
    print(f"Smooth projection matrix SHA256: {p_sha}")
    
    projection_stats = {}
    for name, feats in features_dict.items():
        p_diag = compute_projection_diagnostics(feats, proj, gamma=1.0)
        projection_stats[name] = {
            'pre_act_mean': p_diag['pre_sigmoid_u']['mean'],
            'pre_act_std': p_diag['pre_sigmoid_u']['std'],
            'pre_act_min': p_diag['pre_sigmoid_u']['min'],
            'pre_act_max': p_diag['pre_sigmoid_u']['max'],
            'sat_ratio': p_diag['post_sigmoid_z']['total_saturation_fraction'],
        }
    
    results = {
        'diagnostics_type': diag_type,
        'dataset': 'tanganke/dtd',
        'num_samples': n_samples,
        'sample_selection_seed': seed,
        'sample_indices_sha256': sample_indices_sha,
        'sample_indices': sample_indices,
        'num_unique_classes': len(unique_classes),
        'unique_classes': unique_classes,
        'class_composition': class_counts,
        'device': device,
        'backbone': backbone_name,
        'projection_sha256': p_sha,
        'tensor_stats': tensor_stats,
        'c3_d_vs_official_pixel_diff': {
            'max_abs': max_d_official_diff,
            'mean_abs': mean_d_official_diff,
        },
        'feature_stats': feature_stats,
        'cosine_similarities': {
            'C3-A_vs_C3-B (Interp effect)': {'mean': sim_a_b[0], 'std': sim_a_b[1], 'min': sim_a_b[2]},
            'C3-A_vs_C3-C (Norm effect)': {'mean': sim_a_c[0], 'std': sim_a_c[1], 'min': sim_a_c[2]},
            'C3-A_vs_C3-D (Total effect)': {'mean': sim_a_d[0], 'std': sim_a_d[1], 'min': sim_a_d[2]},
            'C3-B_vs_C3-D (Norm in Bilinear)': {'mean': sim_b_d[0], 'std': sim_b_d[1], 'min': sim_b_d[2]},
            'C3-C_vs_C3-D (Interp in Raw)': {'mean': sim_c_d[0], 'std': sim_c_d[1], 'min': sim_c_d[2]},
            'C3-D_vs_Official': {'mean': sim_d_off[0], 'std': sim_d_off[1], 'min': sim_d_off[2]},
        },
        'projection_stats': projection_stats,
        'environment': get_environment_info(),
    }
    
    print(f"\n--- Summary of Tensor & Feature Diagnostics [{diag_type}, N={n_samples}] ---")
    print(f"{'Condition':<10} | {'Pixel Range':<18} | {'Feat Norm':<12} | {'r_eff':<8} | {'u std':<8} | {'Sat %':<8}")
    print("-" * 75)
    for k in ['C3-A', 'C3-B', 'C3-C', 'C3-D', 'Official']:
        t = tensor_stats[k]
        f = feature_stats[k]
        p = projection_stats[k]
        rng_str = f"[{t['min']:.2f}, {t['max']:.2f}]"
        print(f"{k:<10} | {rng_str:<18} | {f['norm_mean']:<12.2f} | {f['r_eff']:<8.2f} | {p['pre_act_std']:<8.3f} | {p['sat_ratio']*100:<7.2f}%")
        
    print("\n--- Pairwise Cosine Similarities Between Conditions ---")
    for pair, sim in results['cosine_similarities'].items():
        print(f"  {pair:<36}: mean={sim['mean']:.5f}, min={sim['min']:.5f}")
        
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f"\nSaved diagnostics results to: {output_path}")
        
    return results


def main():
    parser = argparse.ArgumentParser(description="C3 Preprocessing Matched Diagnostics on DTD")
    parser.add_argument('--num-samples', type=int, default=200, help="Number of test samples")
    parser.add_argument('--seed', type=int, default=42, help="Sample selection seed")
    parser.add_argument('--device', type=str, default='cpu', help="Compute device")
    parser.add_argument('--output', type=str, default='results/dtd_c3_analysis/c3_dtd_matched_diagnostics.json')
    args = parser.parse_args()
    run_diagnostics(num_samples=args.num_samples, seed=args.seed, device=args.device, output_path=args.output)


if __name__ == '__main__':
    main()
