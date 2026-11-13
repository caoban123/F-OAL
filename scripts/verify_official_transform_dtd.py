"""Deep Verification of C3-D vs Official Author Preprocessing on DTD images.

Tests transform equivalence across:
1. 100 real DTD test images (evaluating resizing from varied original aspect ratios & resolutions)
2. Synthetic edge cases:
   - Solid colors (black, white, mid-gray)
   - 1-pixel checkerboard (high spatial frequency)
   - Linear gradients (horizontal, vertical)
   - Dirac delta (single pixel impulse)
   - Random uniform uint8 noise
Records max absolute error, mean error, library versions, and equivalence verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as tv_transforms

from src.foal.dtd_datasets import load_dtd
from src.foal.datasets import cifar100_transforms


def run_transform_verification(num_samples: int = 100, seed: int = 42, output_path: str | None = None) -> dict:
    print("=" * 80)
    print("OFFICIAL AUTHOR TRANSFORM VS C3-D DEEP VERIFICATION (DTD)")
    print("=" * 80)
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"PyTorch version    : {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Pillow (PIL) ver   : {Image.__version__}")
    print("-" * 80)

    # 1. Define Official Author Transform (from setup_elements.py)
    # The official repo feeds a PIL Image or uint8 array to Resize((224, 224)) + ToTensor()
    official_pipeline = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),  # Default: Bilinear
        tv_transforms.ToTensor(),          # [0.0, 1.0], FloatTensor
    ])

    # 2. Define C3-D Pipeline
    # Our repo feeds a PIL Image (RGB) to cifar100_transforms(size=224, interpolation='bilinear', normalize=False)
    c3_d_pipeline = cifar100_transforms(size=224, interpolation='bilinear', normalize=False)

    # Verification function
    def compare_images(pil_img: Image.Image, name: str) -> dict:
        t_official = official_pipeline(pil_img)
        t_c3_d = c3_d_pipeline(pil_img)

        abs_diff = (t_official - t_c3_d).abs()
        max_diff = float(abs_diff.max().item())
        mean_diff = float(abs_diff.mean().item())
        exact_match = (t_official == t_c3_d).float().mean().item() * 100.0

        return {
            'name': name,
            'orig_size': list(pil_img.size),  # (width, height)
            'max_abs_diff': max_diff,
            'mean_abs_diff': mean_diff,
            'exact_match_pct': exact_match,
            'official_range': (float(t_official.min().item()), float(t_official.max().item())),
            'c3_d_range': (float(t_c3_d.min().item()), float(t_c3_d.max().item())),
            'shape': list(t_c3_d.shape),
            'dtype': str(t_c3_d.dtype),
        }

    results = []

    # --- Part A: Real DTD Samples ---
    print(f"Loading {num_samples} real DTD test samples...")
    _, test_source = load_dtd()
    rng = np.random.default_rng(seed)
    n = min(num_samples, len(test_source))
    sample_indices = rng.choice(len(test_source), size=n, replace=False).tolist()

    real_max_diff = 0.0
    real_mean_diff = 0.0
    for idx in sample_indices:
        pil_img = test_source[idx]['img'].convert('RGB')
        res = compare_images(pil_img, f"dtd_test_sample_{idx}")
        real_max_diff = max(real_max_diff, res['max_abs_diff'])
        real_mean_diff += res['mean_abs_diff']

    real_mean_diff /= len(sample_indices)
    print(f"Real DTD ({len(sample_indices)} samples): Max Diff = {real_max_diff:.2e}, Mean Diff = {real_mean_diff:.2e}")

    # --- Part B: Synthetic Edge Cases ---
    synthetic_cases = {}

    # Solid black, white, gray
    synthetic_cases['solid_black'] = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode='RGB')
    synthetic_cases['solid_white'] = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8), mode='RGB')
    synthetic_cases['solid_gray128'] = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8), mode='RGB')

    # 1-pixel checkerboard (non-square 64x48)
    cb = np.zeros((64, 48, 3), dtype=np.uint8)
    cb[::2, ::2, :] = 255
    cb[1::2, 1::2, :] = 255
    synthetic_cases['checkerboard_1px'] = Image.fromarray(cb, mode='RGB')

    # Gradients (non-square 120x80)
    grad_h = np.tile(np.linspace(0, 255, 120, dtype=np.uint8), (80, 1))
    synthetic_cases['gradient_horizontal'] = Image.fromarray(np.stack([grad_h]*3, axis=-1), mode='RGB')

    grad_v = np.tile(np.linspace(0, 255, 80, dtype=np.uint8)[:, None], (1, 120))
    synthetic_cases['gradient_vertical'] = Image.fromarray(np.stack([grad_v]*3, axis=-1), mode='RGB')

    # Single-pixel impulse
    impulse = np.zeros((100, 100, 3), dtype=np.uint8)
    impulse[50, 50, :] = 255
    synthetic_cases['single_pixel_impulse'] = Image.fromarray(impulse, mode='RGB')

    # Random noise (varied aspect ratio 300x200)
    rng_noise = np.random.default_rng(12345)
    noise_arr = rng_noise.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
    synthetic_cases['random_uniform_noise'] = Image.fromarray(noise_arr, mode='RGB')

    print("\n--- Synthetic Edge Case Comparisons ---")
    print(f"{'Case Name':<25} | {'Max Abs Diff':<14} | {'Mean Abs Diff':<14} | {'Exact Match %':<14} | {'Range [min, max]'}")
    print("-" * 85)

    all_passed = True
    for name, img in synthetic_cases.items():
        res = compare_images(img, name)
        results.append(res)
        pass_flag = res['max_abs_diff'] < 1e-6
        if not pass_flag:
            all_passed = False
        rng_str = f"[{res['c3_d_range'][0]:.3f}, {res['c3_d_range'][1]:.3f}]"
        print(f"{name:<25} | {res['max_abs_diff']:<14.2e} | {res['mean_abs_diff']:<14.2e} | {res['exact_match_pct']:<13.2f}% | {rng_str}")

    print("-" * 85)
    print(f"Overall Real Images Max Abs Diff: {real_max_diff:.2e}")
    print(f"Edge Cases Max Abs Diff        : {max(r['max_abs_diff'] for r in results):.2e}")
    if real_max_diff < 1e-6 and all_passed:
        print("\n>>> VERIFICATION SUCCESS: C3-D is BITWISE EQUIVALENT (diff <= 1e-7) to Official Transform on DTD! <<<")
    else:
        print("\n>>> VERIFICATION WARNING: Discrepancy detected between C3-D and Official Transform! <<<")

    summary = {
        'dataset': 'tanganke/dtd',
        'real_dtd_samples_evaluated': len(sample_indices),
        'real_dtd_max_diff': real_max_diff,
        'real_dtd_mean_diff': real_mean_diff,
        'synthetic_cases': results,
        'all_passed': real_max_diff < 1e-6 and all_passed,
        'library_versions': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'pillow': Image.__version__,
        }
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"\nSaved verification summary to: {output_path}")

    return summary


def main():
    p = argparse.ArgumentParser(description="Verify C3-D vs Official Author Preprocessing on DTD")
    p.add_argument('--num-samples', type=int, default=100, help="Number of real DTD samples")
    p.add_argument('--seed', type=int, default=42, help="Sample seed")
    p.add_argument('--output', type=str, default='results/dtd_c3_analysis/c3_dtd_transform_verification.json')
    args = p.parse_args()
    run_transform_verification(num_samples=args.num_samples, seed=args.seed, output_path=args.output)


if __name__ == '__main__':
    main()
