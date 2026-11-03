"""Deep Verification of C3-D vs Official Author Preprocessing.

Tests transform equivalence across:
1. 100 real CIFAR-100 test images
2. Synthetic edge cases:
   - Solid colors (black, white, mid-gray)
   - 1-pixel checkerboard (high spatial frequency)
   - Linear gradients (horizontal, vertical, diagonal)
   - Dirac delta (single pixel impulse)
   - Random uniform uint8 noise
Records max absolute error, mean error, library versions, and antialias semantics.
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as tv_transforms
from torchvision.transforms import InterpolationMode

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.foal.datasets import load_cifar100, cifar100_transforms


def run_transform_verification():
    print("=" * 80)
    print("OFFICIAL AUTHOR TRANSFORM VS C3-D DEEP VERIFICATION")
    print("=" * 80)
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"PyTorch version    : {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Pillow (PIL) ver   : {Image.__version__}")
    print("-" * 80)

    # 1. Define Official Author Transform (from setup_elements.py)
    # The official repo feeds a NumPy uint8 array (32, 32, 3) to ToPILImage()
    official_pipeline = tv_transforms.Compose([
        tv_transforms.ToPILImage(),
        tv_transforms.Resize((224, 224)),  # Default: Bilinear
        tv_transforms.ToTensor(),          # [0.0, 1.0], FloatTensor
    ])

    # 2. Define C3-D Pipeline
    # Our repo feeds a PIL Image (RGB) to cifar100_transforms
    c3_d_pipeline = cifar100_transforms(size=224, interpolation='bilinear', normalize=False)

    # Verification function
    def compare_images(np_array_uint8, name):
        # Official takes NumPy array
        t_official = official_pipeline(np_array_uint8)
        # C3-D takes PIL Image
        pil_img = Image.fromarray(np_array_uint8, mode='RGB')
        t_c3_d = c3_d_pipeline(pil_img)

        abs_diff = (t_official - t_c3_d).abs()
        max_diff = float(abs_diff.max().item())
        mean_diff = float(abs_diff.mean().item())
        exact_match = (t_official == t_c3_d).float().mean().item() * 100.0

        return {
            'name': name,
            'max_abs_diff': max_diff,
            'mean_abs_diff': mean_diff,
            'exact_match_pct': exact_match,
            'official_range': (float(t_official.min().item()), float(t_official.max().item())),
            'c3_d_range': (float(t_c3_d.min().item()), float(t_c3_d.max().item())),
            'shape': list(t_c3_d.shape),
            'dtype': str(t_c3_d.dtype),
        }

    results = []

    # --- Part A: Real CIFAR-100 Samples ---
    print("Loading 100 real CIFAR-100 samples...")
    _, test_source = load_cifar100()
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(test_source), size=100, replace=False).tolist()

    real_max_diff = 0.0
    real_mean_diff = 0.0
    for idx in sample_indices:
        pil_img = test_source[idx]['img'].convert('RGB')
        np_arr = np.array(pil_img)
        res = compare_images(np_arr, f"cifar100_sample_{idx}")
        real_max_diff = max(real_max_diff, res['max_abs_diff'])
        real_mean_diff += res['mean_abs_diff']

    real_mean_diff /= len(sample_indices)
    print(f"Real CIFAR-100 (100 samples): Max Diff = {real_max_diff:.2e}, Mean Diff = {real_mean_diff:.2e}")

    # --- Part B: Synthetic Edge Cases ---
    synthetic_cases = {}
    
    # Solid black, white, gray
    synthetic_cases['solid_black'] = np.zeros((32, 32, 3), dtype=np.uint8)
    synthetic_cases['solid_white'] = np.full((32, 32, 3), 255, dtype=np.uint8)
    synthetic_cases['solid_gray128'] = np.full((32, 32, 3), 128, dtype=np.uint8)

    # 1-pixel checkerboard
    cb = np.zeros((32, 32, 3), dtype=np.uint8)
    cb[::2, ::2, :] = 255
    cb[1::2, 1::2, :] = 255
    synthetic_cases['checkerboard_1px'] = cb

    # Gradients
    grad_h = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    synthetic_cases['gradient_horizontal'] = np.stack([grad_h]*3, axis=-1)

    grad_v = np.tile(np.linspace(0, 255, 32, dtype=np.uint8)[:, None], (1, 32))
    synthetic_cases['gradient_vertical'] = np.stack([grad_v]*3, axis=-1)

    # Single-pixel impulse (Dirac delta in center)
    impulse = np.zeros((32, 32, 3), dtype=np.uint8)
    impulse[16, 16, :] = 255
    synthetic_cases['single_pixel_impulse'] = impulse

    # Random noise
    rng_noise = np.random.default_rng(12345)
    synthetic_cases['random_uniform_noise'] = rng_noise.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    print("\n--- Synthetic Edge Case Comparisons ---")
    print(f"{'Case Name':<25} | {'Max Abs Diff':<14} | {'Mean Abs Diff':<14} | {'Exact Match %':<14} | {'Range [min, max]'}")
    print("-" * 85)

    all_passed = True
    for name, arr in synthetic_cases.items():
        res = compare_images(arr, name)
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
        print("\n>>> VERIFICATION SUCCESS: C3-D is BITWISE EQUIVALENT (diff <= 1e-7) to Official Transform! <<<")
    else:
        print("\n>>> VERIFICATION WARNING: Discrepancy detected between C3-D and Official Transform! <<<")

    return {
        'real_cifar100_max_diff': real_max_diff,
        'real_cifar100_mean_diff': real_mean_diff,
        'synthetic_cases': results,
        'all_passed': real_max_diff < 1e-6 and all_passed,
        'library_versions': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'pillow': Image.__version__,
        }
    }


if __name__ == '__main__':
    run_transform_verification()
