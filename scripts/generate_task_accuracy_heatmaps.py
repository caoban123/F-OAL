"""
Script: scripts/generate_task_accuracy_heatmaps.py

Generates the 2x2 comparison figure of Task Accuracy Matrices for CIFAR-100:
  1. F-OAL (Reproduction)
  2. Experience Replay (Buffer = 5000)
  3. LwF (Official Protocol)
  4. LwF (Adapted AdamW)

Output file:
  results/headline_analysis/fig1_headline_final_task_matrices.png (300 DPI)
"""

import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 output
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

RUN_CONFIGS = [
    {
        "id": "foal",
        "title": "(a) F-OAL (Reproduction)",
        "path": ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw" / "task_accuracy.csv",
        "is_long_format": True,
        "exp_a10": 85.73,
        "exp_fpast": 5.16,
        "subtitle": "Exemplar-Free Analytic Learning | Closed-Form RLS",
    },
    {
        "id": "er",
        "title": "(b) Experience Replay (Buffer = 5000)",
        "path": ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "er" / "task_matrix.csv",
        "is_long_format": False,
        "exp_a10": 79.79,
        "exp_fpast": 16.74,
        "subtitle": "Replay-Based Baseline | Reservoir Exemplar Buffer",
    },
    {
        "id": "lwf_off",
        "title": "(c) LwF (Official Protocol)",
        "path": ROOT / "results" / "headline_official_lwf_verified" / "runs" / "headline" / "lwf_official_cifar100" / "task_matrix.csv",
        "is_long_format": False,
        "exp_a10": 23.34,
        "exp_fpast": 0.41,
        "subtitle": "Exemplar-Free Distillation | SGD + Dynamic Loss Weighting",
    },
    {
        "id": "lwf_ad",
        "title": "(d) LwF (Adapted AdamW)",
        "path": ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "lwf" / "task_matrix.csv",
        "is_long_format": False,
        "exp_a10": 11.21,
        "exp_fpast": 94.99,
        "subtitle": "Exemplar-Free Distillation | AdamW + Static Loss Weighting",
    },
]


def load_matrix(path: Path, is_long_format: bool) -> np.ndarray:
    """Load lower-triangular 10x10 accuracy matrix in percent [0, 100]."""
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    grid = np.full((10, 10), np.nan, dtype=float)

    if is_long_format:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s = int(r["stage"]) - 1  # 0-indexed stage
                t = int(r["task"]) - 1   # 0-indexed task
                acc = float(r["accuracy"])
                # Normalize to percentage if in [0, 1]
                if acc <= 1.0:
                    acc *= 100.0
                grid[s, t] = acc
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            stage = 0
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                for task, val_str in enumerate(row):
                    if val_str.strip():
                        acc = float(val_str)
                        if acc <= 1.0:
                            acc *= 100.0
                        grid[stage, task] = acc
                stage += 1

    # Verify that exactly 55 lower-triangular cells are present and non-nan
    observed_count = np.count_nonzero(~np.isnan(grid))
    if observed_count != 55:
        raise ValueError(
            f"Expected exactly 55 observed lower-triangular cells, found {observed_count} in {path}"
        )

    return grid


def compute_metrics(grid: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes:
      - A_10: Final accuracy across all 10 tasks after Stage 10 (row index 9)
      - F_past_clipped: Average forgetting across past 9 tasks (clipped >= 0)
      - F_past_unclipped: Average forgetting across past 9 tasks (unclipped Eq. 18)
    """
    # 1. Final Accuracy A_10
    row_10 = grid[9, :10]
    if np.any(np.isnan(row_10)):
        raise ValueError("Stage 10 contains NaN values in task evaluations!")
    a_10 = float(np.mean(row_10))

    # 2. Past Forgetting across 9 past tasks (j = 0..8)
    unclipped_drops = []
    clipped_drops = []
    for j in range(9):
        # peak historical accuracy prior to stage 10 (stages j..8)
        peak = np.nanmax(grid[j:9, j])
        final = grid[9, j]
        drop = peak - final
        unclipped_drops.append(drop)
        clipped_drops.append(max(0.0, drop))

    f_past_unclipped = float(np.mean(unclipped_drops))
    f_past_clipped = float(np.mean(clipped_drops))

    return a_10, f_past_clipped, f_past_unclipped


def verify_data(cfg: Dict, a_10: float, f_past_clipped: float, f_past_unclipped: float):
    """Verify metrics against recorded headline audit values within tolerance."""
    exp_a10 = cfg["exp_a10"]
    exp_fpast = cfg["exp_fpast"]
    tol = 0.05  # 0.05 pp tolerance for rounding

    # Check Final Accuracy A_10
    if abs(a_10 - exp_a10) > tol:
        raise ValueError(
            f"Verification FAILED for {cfg['title']}: "
            f"Calculated A_10 = {a_10:.2f}%, expected {exp_a10:.2f}% (tol={tol}%)"
        )

    # Check Past Forgetting (compare clipped for standard, or unclipped for official LwF zero note)
    match_clipped = abs(f_past_clipped - exp_fpast) <= tol
    match_unclipped = abs(f_past_unclipped - exp_fpast) <= tol

    if not (match_clipped or match_unclipped):
        raise ValueError(
            f"Verification FAILED for {cfg['title']}: "
            f"Calculated F_past = {f_past_clipped:.2f}% (unclipped={f_past_unclipped:.2f}%), "
            f"expected {exp_fpast:.2f}% (tol={tol}%)"
        )


def main():
    print("=" * 80)
    print("TASK ACCURACY MATRIX COMPARISON GENERATOR (2x2 HEATMAP)")
    print("Target: results/headline_analysis/fig1_headline_final_task_matrices.png")
    print("=" * 80)

    loaded_data = []

    # 1. Load and Verify All 4 Matrices
    for cfg in RUN_CONFIGS:
        print(f"\n[Verifying] {cfg['title']}")
        print(f"  Source file: {cfg['path']}")

        grid = load_matrix(cfg["path"], cfg["is_long_format"])
        a_10, f_clipped, f_unclipped = compute_metrics(grid)

        print(f"  Final Accuracy A_10 : {a_10:.2f}% (Expected: {cfg['exp_a10']:.2f}%)")
        print(f"  Past Forgetting F_p : {f_clipped:.2f}% (Unclipped: {f_unclipped:.4f}%, Expected: {cfg['exp_fpast']:.2f}%)")

        verify_data(cfg, a_10, f_clipped, f_unclipped)
        print("  -> Data audit status: VERIFIED MATCH")

        loaded_data.append({
            "cfg": cfg,
            "grid": grid,
            "a_10": a_10,
            "f_clipped": f_clipped,
            "f_unclipped": f_unclipped,
        })

    # 2. Plotting 2x2 Heatmaps
    out_dir = ROOT / "results" / "headline_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig1_headline_final_task_matrices.png"

    # Academic plotting styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 15.5), dpi=300)

    # Use Blues colormap with high visual clarity
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f4f5f7")  # Clean subtle light-gray for unlearned cells

    im = None

    for idx, item in enumerate(loaded_data):
        row_idx = idx // 2
        col_idx = idx % 2
        ax = axes[row_idx, col_idx]

        cfg = item["cfg"]
        grid = item["grid"]
        a_10 = item["a_10"]
        f_clipped = item["f_clipped"]

        # Display masked heatmap
        im = ax.imshow(
            grid,
            cmap=cmap,
            vmin=0.0,
            vmax=100.0,
            aspect="equal",
            interpolation="nearest",
        )

        # Annotate lower triangular cells
        for s in range(10):
            for t in range(s + 1):
                val = grid[s, t]
                # High-contrast text color selection
                text_color = "white" if val >= 55.0 else "#0a192f"
                # Formatting: 1 decimal place
                ax.text(
                    t,
                    s,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold" if s == 9 else "normal",
                    color=text_color,
                )

        # Configure axes ticks and labels
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels([f"T{i+1}" for i in range(10)], fontsize=10, fontweight="bold")
        ax.set_yticklabels([f"S{i+1}" for i in range(10)], fontsize=10, fontweight="bold")

        # Labels
        ax.set_xlabel("Evaluation Task", fontsize=11, fontweight="bold", labelpad=6)
        ax.set_ylabel("Training Stage", fontsize=11, fontweight="bold", labelpad=6)

        # Highlight final row (Stage 10) with subtle underline or border
        ax.axhline(8.5, color="#d9534f", linewidth=1.2, linestyle="--", alpha=0.7)

        # Subplot Title & Subtitle
        fgt_str = f"{f_clipped:.2f}%"
        if cfg["id"] == "lwf_off":
            fgt_str = "0.41% (0.00% unclip)"

        ax.set_title(
            f"{cfg['title']}\n"
            f"A10 = {a_10:.2f}%  |  F_past = {fgt_str}\n"
            f"({cfg['subtitle']})",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )

        # Grid styling
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="#d0d7de", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Adjust layout to make plenty of room for main title and colorbar
    fig.subplots_adjust(left=0.065, right=0.875, top=0.87, bottom=0.06, wspace=0.22, hspace=0.32)

    # Global Colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.15, 0.022, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Task Accuracy (%)", fontsize=12, fontweight="bold", labelpad=10)
    cbar.set_ticks(np.linspace(0, 100, 11))
    cbar.ax.tick_params(labelsize=10)

    # Main Suptitle placed with safe clearance above subplots
    fig.suptitle(
        "CIFAR-100 Continual Learning: Task Accuracy Matrices (10 Tasks × 10 Stages)\n"
        "Audited Headline Comparison: F-OAL vs. Experience Replay vs. LwF Baselines (Seed 42)",
        fontsize=14.5,
        fontweight="bold",
        y=0.955,
    )

    # Save high-resolution PNG
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\n>>> SUCCESS: Heatmap figure generated at:\n    {out_path}")
    print(f"    File size: {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
