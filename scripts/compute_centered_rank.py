import math
from pathlib import Path
import sys
import torch
from src.foal.projection import SmoothProjection

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cache_dir = Path("results/c1_pilot_feature_cache")
t1 = torch.load(cache_dir / "train_c42_s42_t01_tr100.pt", map_location="cpu")
t2 = torch.load(cache_dir / "train_c42_s42_t02_tr100.pt", map_location="cpu")

feats = torch.cat([t1["features"], t2["features"]], dim=0) # 200 x 768
print(f"Loaded features shape: {feats.shape}")

def calc_erank(matrix_psd):
    eigvals = torch.linalg.eigvalsh(matrix_psd)
    eigvals = torch.clamp(eigvals, min=0.0)
    total = eigvals.sum().item()
    if total <= 0:
        return 0.0
    p = eigvals / total
    p_pos = p[p > 1e-15]
    h = -(p_pos * torch.log(p_pos)).sum().item()
    return math.exp(h)

variants = [
    ("P0: gaussian_std1", "gaussian_std1"),
    ("P1: official_uniform", "official_uniform"),
    ("P2: gaussian_matched_variance", "gaussian_matched_variance"),
]

print("\n--- SPECTRAL & RANK DIAGNOSTICS: UNCENTERED VS CENTERED ---")
for label, mode in variants:
    proj = SmoothProjection(768, 1000, seed=42, initialization=mode)
    with torch.no_grad():
        u = proj.pre_activation(feats.float())
        z = proj(feats.float()) # N x D
        N = z.shape[0]

        # 1. Uncentered Gram G = Z^T Z
        gram_uncentered = z.T @ z
        erank_uncentered = calc_erank(gram_uncentered)

        # 2. Centered Covariance C = (Z - mean)^T (Z - mean) / (N - 1)
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov_centered = (z_centered.T @ z_centered) / (N - 1)
        erank_centered = calc_erank(cov_centered)

        # 3. Ratio of top singular value to trace
        eigs_uncentered = torch.linalg.eigvalsh(gram_uncentered).clamp(min=0.0)
        eigs_centered = torch.linalg.eigvalsh(cov_centered).clamp(min=0.0)
        top_uncentered_pct = (eigs_uncentered[-1] / eigs_uncentered.sum()).item() * 100
        top_centered_pct = (eigs_centered[-1] / eigs_centered.sum()).item() * 100

        print(f"\n{label}:")
        print(f"  Pre-sigmoid u: mean={u.mean():.4f}, std={u.std():.4f}")
        print(f"  Post-sigmoid z: mean={z.mean():.4f}, std={z.std():.4f}")
        print(f"  Uncentered Gram Erank   : {erank_uncentered:.2f} (Top eigval carries {top_uncentered_pct:.2f}% of trace)")
        print(f"  Centered Covariance Erank: {erank_centered:.2f} (Top eigval carries {top_centered_pct:.2f}% of trace)")
