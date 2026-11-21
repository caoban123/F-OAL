"""Generate Table 4 Paper vs. Reproduction (Seed 42) Horizontal Dumbbell Plot.

Reads the original verified Seed 42 `metrics.json` artifacts directly from disk,
verifies exact values, computes `gap_pp` and `relative_deviation`, and exports:
    results/headline_analysis/fig_table4_paper_vs_reproduction.png (300 DPI)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

TABLE4_SPECS = [
    {
        "label": "Full (FF+SP)",
        "paper_pct": 91.10,
        "metrics_path": ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw" / "metrics.json",
    },
    {
        "label": "Without FF",
        "paper_pct": 90.60,
        "metrics_path": ROOT / "results" / "table4_full_gpu_verified" / "results" / "table4_a1_without_ff" / "metrics.json",
    },
    {
        "label": "Without SP",
        "paper_pct": 90.70,
        "metrics_path": ROOT / "results" / "table4_full_gpu_verified" / "results" / "table4_a2_without_sp" / "metrics.json",
    },
    {
        "label": "Without FF & SP",
        "paper_pct": 90.60,
        "metrics_path": ROOT / "results" / "table4_full_gpu_verified" / "results" / "table4_a3_without_both" / "metrics.json",
    },
]


def load_verified_rows() -> list[dict]:
    rows = []
    for spec in TABLE4_SPECS:
        m_path = spec["metrics_path"]
        if not m_path.exists():
            raise FileNotFoundError(f"Verified Seed 42 artifact missing: {m_path}")
        metrics = json.loads(m_path.read_text(encoding="utf-8"))
        repro_pct = float(metrics["average_incremental_accuracy"]) * 100.0
        paper_pct = float(spec["paper_pct"])
        gap_pp = repro_pct - paper_pct
        rel_dev = 100.0 * gap_pp / paper_pct
        rows.append({
            "label": spec["label"],
            "paper_pct": paper_pct,
            "repro_pct": repro_pct,
            "gap_pp": gap_pp,
            "rel_dev": rel_dev,
        })
    return rows


def plot_dumbbell(rows: list[dict], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=300)

    # Order from top to bottom: Full at top
    n = len(rows)
    y_positions = np.arange(n - 1, -1, -1, dtype=float)

    paper_color = "#1f4e78"
    repro_color = "#d95f02"
    line_color = "#9aa5b1"

    for y, row in zip(y_positions, rows):
        p_val = row["paper_pct"]
        r_val = row["repro_pct"]
        gap_pp = row["gap_pp"]
        rel_dev = row["rel_dev"]

        # Connecting dumbbell line
        ax.plot([r_val, p_val], [y, y], color=line_color, linewidth=3.2, zorder=2)

        # Endpoints
        ax.scatter([p_val], [y], color=paper_color, marker="o", s=145, zorder=4,
                   edgecolors="white", linewidth=1.2,
                   label="Paper (F-OAL, NeurIPS 2024, Table 4)" if y == y_positions[0] else "")
        ax.scatter([r_val], [y], color=repro_color, marker="D", s=115, zorder=5,
                   edgecolors="white", linewidth=1.2,
                   label="Our Reproduction (CIFAR-100, 10 tasks, seed 42)" if y == y_positions[0] else "")

        # Value labels at each point
        ax.text(r_val - 0.045, y, f"{r_val:.4f}%", ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=repro_color)
        ax.text(p_val + 0.045, y, f"{p_val:.2f}%", ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=paper_color)

        # Gap (pp) & Relative Deviation (%) centered above dumbbell
        mid_x = 0.5 * (r_val + p_val)
        ax.text(
            mid_x,
            y + 0.21,
            f"{gap_pp:+.4f} pp  ({rel_dev:+.4f}% rel.)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#24292f",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f6f8fa", edgecolor="#d0d7de", linewidth=0.8),
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=11, fontweight="bold")
    ax.set_ylim(-0.55, n - 0.35)
    ax.set_xlim(89.30, 91.65)

    ax.set_xlabel("Average Incremental Accuracy (%, zoomed scale: 89.30%–91.65%; non-zero axis)",
                  fontsize=10.5, fontweight="bold", labelpad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#d0d7de", fontsize=9.5)

    fig.suptitle(
        "Table 4 Reproduction — Published vs. Reproduced Results on CIFAR-100",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )
    ax.set_title(
        "Paper: F-OAL, NeurIPS 2024, Table 4   |   Reproduction: CIFAR-100, 10 tasks, seed 42 (C3-D Bilinear Raw)",
        fontsize=10,
        color="#444444",
        pad=12,
    )
    fig.text(
        0.5,
        0.015,
        "Note: Accuracy axis is zoomed to highlight sub-percentage-point differences (not zero-based). "
        "gap_pp = reproduction − paper; relative_deviation = 100 × gap_pp / paper.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    fig.tight_layout(rect=[0.01, 0.04, 0.99, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    rows = load_verified_rows()
    print("=== VERIFIED TABLE 4 SEED 42 VS PAPER ===")
    for r in rows:
        print(
            f"{r['label']:<16} | Paper={r['paper_pct']:.2f}% | Repro={r['repro_pct']:.4f}% "
            f"| gap={r['gap_pp']:+.4f} pp | rel_dev={r['rel_dev']:+.4f}%"
        )

    out_path = ROOT / "results" / "headline_analysis" / "fig_table4_paper_vs_reproduction.png"
    plot_dumbbell(rows, out_path)
    print(f"\nSaved 300 DPI dumbbell plot to: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
