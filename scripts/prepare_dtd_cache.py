"""Pre-extract and cache frozen ViT representations for DTD dataset.

Extracts both Feature-Fused (FF) and Non-Feature-Fused (No-FF) representations
in a single forward pass to optimize extraction time on CPU.

Extracts for:
1. Primary backbone: vit_base_patch16_224.augreg2_in21k_ft_in1k
   - Bilinear, Raw [0, 1] (official-aligned / C3-D)
   - Bicubic, Normalized [-1, 1] (standard / C3-A)
2. Control backbone: vit_base_patch16_224.augreg_in1k (for C2 comparison)

Saves all feature tensors into results/dtd_feature_cache/.
"""
from __future__ import annotations
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm

from src.foal.dtd_datasets import load_dtd, make_dtd_class_order, DTD_CLASS_NAMES
from src.foal.datasets import cifar100_transforms
from src.foal.feature_fusion import fuse_cls_features


class DualFeatureExtractor:
    """Extracts both Feature-Fused and Non-Fused features in a single forward pass."""
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.network = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.network.eval()
        self.network.to(device)
        for p in self.network.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fused_features, no_fusion_features) both of shape [B, D]."""
        collected = []
        hooks = []
        def save_block(_module, _inputs, output):
            collected.append(output if isinstance(output, torch.Tensor) else output[0])

        try:
            for block in self.network.blocks:
                hooks.append(block.register_forward_hook(save_block))
            _ = self.network.forward_features(x.to(self.device))
        finally:
            for hook in hooks:
                hook.remove()

        # Fused: arithmetic mean of CLS tokens across all 12 blocks + norm
        fused = fuse_cls_features(collected)
        if hasattr(self.network, 'norm') and self.network.norm is not None:
            fused = self.network.norm(fused)

        # No-FF: only the 12th block's CLS token + norm
        last_block = collected[-1]
        last_cls = last_block[:, 0, :]
        if hasattr(self.network, 'norm') and self.network.norm is not None:
            last_cls = self.network.norm(last_cls)

        return fused.cpu(), last_cls.cpu()


class DTDSubsetDataset(Dataset):
    """Subsets DTD samples to the selected classes and limits train samples to 40/class."""
    def __init__(self, source, selected_classes: list[int],
                 samples_per_class: int | None,
                 transform, seed: int = 42):
        self.source = source
        self.transform = transform
        self.indices = []
        self.labels = []

        labels_arr = np.asarray(source.fine_label)
        rng = np.random.default_rng(seed)

        for cls_id in selected_classes:
            cls_indices = np.flatnonzero(labels_arr == cls_id).tolist()
            if samples_per_class is not None and len(cls_indices) > samples_per_class:
                chosen = rng.choice(cls_indices, size=samples_per_class, replace=False).tolist()
            else:
                chosen = cls_indices
            self.indices.extend(chosen)
            self.labels.extend([cls_id] * len(chosen))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.source[self.indices[idx]]
        img = sample['img']
        label = self.labels[idx]
        return self.transform(img), label, self.indices[idx]


def extract_and_save(extractor: DualFeatureExtractor,
                     dataset: DTDSubsetDataset,
                     batch_size: int,
                     cache_dir: Path,
                     prefix: str):
    """Extracts features and saves FF and No-FF files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ff_path = cache_dir / f"{prefix}_ff.pt"
    noff_path = cache_dir / f"{prefix}_noff.pt"

    if ff_path.exists() and noff_path.exists():
        print(f"  [Cache Hit] {prefix} already exists. Skipping.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ff_list = []
    noff_list = []
    labels_list = []
    indices_list = []

    t0 = time.time()
    for batch_idx, (x, y, idxs) in enumerate(loader):
        ff, noff = extractor.extract_batch(x)
        ff_list.append(ff)
        noff_list.append(noff)
        labels_list.append(y)
        indices_list.append(idxs)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            done = min((batch_idx + 1) * batch_size, len(dataset))
            elapsed = time.time() - t0
            print(f"    Extracted {done}/{len(dataset)} samples ({done/elapsed:.1f} samples/s)...")

    all_ff = torch.cat(ff_list, dim=0)
    all_noff = torch.cat(noff_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    all_indices = torch.cat(indices_list, dim=0)
    total_time = time.time() - t0

    torch.save({
        'features': all_ff,
        'labels': all_labels,
        'sample_indices': all_indices,
        'extraction_seconds': total_time,
        'feature_dim': all_ff.shape[1],
        'feature_fusion': True,
    }, ff_path)

    torch.save({
        'features': all_noff,
        'labels': all_labels,
        'sample_indices': all_indices,
        'extraction_seconds': total_time,
        'feature_dim': all_noff.shape[1],
        'feature_fusion': False,
    }, noff_path)

    print(f"  [Saved] {ff_path.name} & {noff_path.name} in {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default='results/dtd_feature_cache')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print("=" * 70)
    print("DTD FEATURE CACHING PIPELINE (ONE-PASS DUAL EXTRACTOR)")
    print("=" * 70)

    # 1. Load DTD dataset
    print("Loading DTD dataset...")
    train_source, test_source = load_dtd()
    order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=args.seed)
    selected_classes = sorted(list(set(order)))
    print(f"Selected 40 classes (seed {args.seed}): {selected_classes}")

    # Transforms to pre-extract:
    # 1. Bilinear + Raw [0, 1] (Official Aligned / C3-D / Default)
    trans_bilinear_raw = cifar100_transforms(size=224, interpolation='bilinear', normalize=False)
    # 2. Bicubic + Normalized [-1, 1] (Standard / C3-A)
    trans_bicubic_norm = cifar100_transforms(size=224, interpolation='bicubic', normalize=True)

    # Subsets: 40 train/class (1,600 total), 40 test/class (1,600 total)
    train_sub_bilinear_raw = DTDSubsetDataset(train_source, selected_classes, samples_per_class=40,
                                              transform=trans_bilinear_raw, seed=args.seed)
    test_sub_bilinear_raw = DTDSubsetDataset(test_source, selected_classes, samples_per_class=40,
                                             transform=trans_bilinear_raw, seed=args.seed)

    print(f"Train subset size: {len(train_sub_bilinear_raw)} (40 per class * 40 classes)")
    print(f"Test subset size:  {len(test_sub_bilinear_raw)} (40 per class * 40 classes)")

    # Backbone 1: Primary ViT-B (augreg2_in21k_ft_in1k)
    print("\n--- Extracting Primary Backbone (augreg2_in21k_ft_in1k) ---")
    ext_primary = DualFeatureExtractor('vit_base_patch16_224.augreg2_in21k_ft_in1k', device=args.device)

    print("Extracting train features (Bilinear, Raw)...")
    extract_and_save(ext_primary, train_sub_bilinear_raw, args.batch_size, cache_dir,
                     "train_augreg2_bilinear_raw")

    print("Extracting test features (Bilinear, Raw)...")
    extract_and_save(ext_primary, test_sub_bilinear_raw, args.batch_size, cache_dir,
                     "test_augreg2_bilinear_raw")

    # Extract Bicubic + Norm for Preprocessing Comparison (C3)
    train_sub_bicubic_norm = DTDSubsetDataset(train_source, selected_classes, samples_per_class=40,
                                              transform=trans_bicubic_norm, seed=args.seed)
    test_sub_bicubic_norm = DTDSubsetDataset(test_source, selected_classes, samples_per_class=40,
                                             transform=trans_bicubic_norm, seed=args.seed)

    print("\nExtracting train features (Bicubic, Normalized)...")
    extract_and_save(ext_primary, train_sub_bicubic_norm, args.batch_size, cache_dir,
                     "train_augreg2_bicubic_norm")

    print("Extracting test features (Bicubic, Normalized)...")
    extract_and_save(ext_primary, test_sub_bicubic_norm, args.batch_size, cache_dir,
                     "test_augreg2_bicubic_norm")

    # Backbone 2: Control ViT-B (augreg_in1k) for C2 Backbone study
    print("\n--- Extracting Control Backbone (augreg_in1k) for C2 Study ---")
    ext_control = DualFeatureExtractor('vit_base_patch16_224.augreg_in1k', device=args.device)
    print("Extracting train features (augreg_in1k, Bilinear, Raw)...")
    extract_and_save(ext_control, train_sub_bilinear_raw, args.batch_size, cache_dir,
                     "train_augreg_in1k_bilinear_raw")
    print("Extracting test features (augreg_in1k, Bilinear, Raw)...")
    extract_and_save(ext_control, test_sub_bilinear_raw, args.batch_size, cache_dir,
                     "test_augreg_in1k_bilinear_raw")

    # Save metadata
    manifest = {
        'num_classes': 40,
        'selected_classes': selected_classes,
        'class_order': order,
        'train_samples_per_class': 40,
        'test_samples_per_class': 40,
        'seed': args.seed,
        'primary_backbone': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        'control_backbone': 'vit_base_patch16_224.augreg_in1k',
        'cache_files': [f.name for f in cache_dir.glob('*.pt')],
    }
    (cache_dir / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print("\n" + "=" * 70)
    print("ALL DTD FEATURES CACHED SUCCESSFULLY!")
    print(f"Location: {cache_dir.resolve()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
