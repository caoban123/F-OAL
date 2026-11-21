"""Generate Section IX Headline Comparison Analysis & Figures for DTD.

Reconciles 55-element 10x10 task accuracy matrices across:
  1. Full F-OAL (A0 / C3-D)      [results/dtd/dtd_a0_full]
  2. Experience Replay (M=5000)  [runs/headline_dtd/er]
  3. Official-Protocol LwF (M=0) [runs/headline_dtd/lwf_official_dtd]
  4. Adapted LwF (M=0, AdamW)    [runs/headline_dtd/lwf]

Outputs CSVs, JSON manifests, and 6 publication-grade 300-DPI figures to
results/headline_dtd_analysis/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.foal.metrics import summary_from_task_matrix


def load_task_matrix(run_dir: Path) -> Optional[List[List[float]]]:
    """Load lower-triangular 10x10 task accuracy matrix (values in [0, 1]) from run_dir."""
    matrix_json = run_dir / "matrix.json"
    summary_json = run_dir / "summary.json"
    metrics_json = run_dir / "metrics.json"
    task_matrix_csv = run_dir / "task_matrix.csv"
    task_acc_csv = run_dir / "task_accuracy.csv"

    if matrix_json.exists():
        mat = json.loads(matrix_json.read_text(encoding="utf-8"))
        return [[float(x) for x in row] for row in mat]
    if summary_json.exists():
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        if "task_matrix" in data:
            return [[float(x) for x in row] for row in data["task_matrix"]]
    if metrics_json.exists():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        if "task_matrix" in data:
            return [[float(x) for x in row] for row in data["task_matrix"]]
    if task_matrix_csv.exists():
        mat: List[List[float]] = []
        with task_matrix_csv.open("r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                mat.append([float(x) for x in row if x.strip() != ""])
        return mat
    if task_acc_csv.exists():
        records: List[Tuple[int, int, float]] = []
        with task_acc_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                records.append((int(r["stage"]), int(r["task"]), float(r["accuracy"])))
        if not records:
            return None
        num_stages = max(s for s, _, _ in records)
        mat = [[0.0] * i for i in range(1, num_stages + 1)]
        for s, t, acc in records:
            mat[s - 1][t - 1] = acc
        return mat
    return None


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _raw_evidence_file(run_dir: Path) -> Optional[Path]:
    for fn in ("task_matrix.csv", "task_accuracy.csv", "metrics.json", "summary.json"):
        p = run_dir / fn
        if p.exists():
            return p
    return None


def analyze_run(
    *,
    key: str,
    label: str,
    classification: str,
    paradigm: str,
    buffer_size: int,
    run_dir: Path,
    paper_a_avg: Optional[float],
    paper_a_last: Optional[float],
    paper_f: Optional[float],
    default_optimizer: str,
    default_arch: str,
    notes: str,
) -> Optional[Dict[str, Any]]:
    mat = load_task_matrix(run_dir)
    if mat is None:
        return None

    summary = summary_from_task_matrix(mat)
    meta = _load_json_if_exists(run_dir / "metadata.json")
    run_summary = _load_json_if_exists(run_dir / "summary.json")
    run_metrics = _load_json_if_exists(run_dir / "metrics.json")

    num_tasks = len(mat)
    assert sum(len(r) for r in mat) == num_tasks * (num_tasks + 1) // 2

    a_avg_pct = summary["average_incremental_accuracy"] * 100.0
    a_last_pct = summary["last_accuracy"] * 100.0
    f_past_pct = summary["forgetting_past_tasks"] * 100.0
    f_code_pct = summary["forgetting_official_code"] * 100.0
    stage_accs_pct = [s * 100.0 for s in summary["stage_accuracies"]]
    diag_accs_pct = [mat[i][i] * 100.0 for i in range(num_tasks)]

    # Verify reconciliation against stored summary/metrics if present
    stored_metrics = run_summary.get("metrics_summary", run_metrics)
    if "average_incremental_accuracy" in stored_metrics:
        diff = abs(summary["average_incremental_accuracy"] - float(stored_metrics["average_incremental_accuracy"]))
        if diff >= 1e-9:
            raise ValueError(f"Reconciliation mismatch for {key}: diff={diff}")

    duration_s = float(
        run_summary.get("total_duration_seconds")
        or run_metrics.get("total_duration_seconds")
        or run_summary.get("total_training_seconds")
        or stored_metrics.get("total_duration_seconds")
        or 0.0
    )
    hw_gpu = (
        meta.get("environment", {}).get("hardware", {}).get("gpu_name")
        or meta.get("environment", {}).get("gpu_name")
        or "Unknown"
    )
    git_sha = meta.get("git_commit") or meta.get("environment", {}).get("git", {}).get("commit") or "unknown"
    split_sha = meta.get("task_split_fingerprint", {}).get("sha256", "unknown")
    ev_file = _raw_evidence_file(run_dir)

    return {
        "key": key,
        "label": label,
        "classification": classification,
        "paradigm": paradigm,
        "buffer_size": buffer_size,
        "replay": buffer_size > 0,
        "run_dir": str(run_dir),
        "matrix": mat,
        "num_tasks": num_tasks,
        "a_avg_pct": a_avg_pct,
        "a_last_pct": a_last_pct,
        "f_past_pct": f_past_pct,
        "f_code_pct": f_code_pct,
        "stage_accs_pct": stage_accs_pct,
        "diag_accs_pct": diag_accs_pct,
        "task_forgetting_past_pct": [f * 100.0 for f in summary["task_forgetting_past_tasks"]],
        "paper_a_avg": paper_a_avg,
        "paper_a_last": paper_a_last,
        "paper_f": paper_f,
        "duration_s": duration_s,
        "hardware": hw_gpu,
        "git_sha": git_sha,
        "split_sha256": split_sha,
        "optimizer": meta.get("model_config", {}).get("optimizer", default_optimizer),
        "classifier_arch": default_arch,
        "notes": notes,
        "evidence_file": str(ev_file) if ev_file else "",
        "evidence_sha256": hashlib.sha256(ev_file.read_bytes()).hexdigest() if ev_file else "",
    }


def _plot_fig1_task_matrices(runs: List[Dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 15.5), dpi=300)
    cmap = plt.cm.Blues.copy()
    if hasattr(cmap, "with_extremes"):
        cmap = cmap.with_extremes(bad="#f4f5f7")
    else:
        cmap.set_bad(color="#f4f5f7")

    panel_map = {
        "foal": (0, 0, "(a) F-OAL Full (A0 / C3-D)", "Exemplar-Free Analytic Learning | Closed-Form RLS"),
        "er": (0, 1, "(b) Experience Replay (Buffer = 5000)", "Replay-Based Baseline | Reservoir Exemplar Buffer"),
        "official_lwf": (1, 0, "(c) LwF (Official Protocol)", "Exemplar-Free Distillation | SGD + Dynamic Loss 1/(t+1)"),
        "adapted_lwf": (1, 1, "(d) LwF (Adapted AdamW)", "Exemplar-Free Distillation | AdamW + Static 1:1 Loss"),
    }
    by_key = {r["key"]: r for r in runs}
    im = None

    for key, (r_idx, c_idx, title, subtitle) in panel_map.items():
        ax = axes[r_idx, c_idx]
        if key not in by_key:
            ax.axis("off")
            ax.set_title(f"{title}\n[Run Pending]", fontsize=11.5, fontweight="bold")
            continue

        run = by_key[key]
        n = run["num_tasks"]
        grid = np.full((n, n), np.nan, dtype=float)
        for s_idx, row in enumerate(run["matrix"]):
            for t_idx, val in enumerate(row):
                grid[s_idx, t_idx] = val * 100.0

        im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=100.0, aspect="equal", interpolation="nearest")
        for s in range(n):
            for t in range(s + 1):
                val = grid[s, t]
                text_color = "white" if val >= 55.0 else "#0a192f"
                ax.text(
                    t,
                    s,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold" if s == n - 1 else "normal",
                    color=text_color,
                )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"T{i + 1}" for i in range(n)], fontsize=10, fontweight="bold")
        ax.set_yticklabels([f"S{i + 1}" for i in range(n)], fontsize=10, fontweight="bold")
        ax.set_xlabel("Evaluation Task", fontsize=11, fontweight="bold", labelpad=6)
        ax.set_ylabel("Training Stage", fontsize=11, fontweight="bold", labelpad=6)
        ax.axhline(n - 1.5, color="#d9534f", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_title(
            f"{title}\n"
            f"A_avg = {run['a_avg_pct']:.2f}%  |  A10 = {run['a_last_pct']:.2f}%  |  F_past = {run['f_past_pct']:.2f}%\n"
            f"({subtitle})",
            fontsize=11.0,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#d0d7de", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.subplots_adjust(left=0.065, right=0.875, top=0.87, bottom=0.06, wspace=0.22, hspace=0.34)
    if im is not None:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.022, 0.68])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Task Accuracy (%)", fontsize=12, fontweight="bold", labelpad=10)
        cbar.set_ticks(np.linspace(0, 100, 11))
    fig.suptitle(
        "DTD Continual Learning: Task Accuracy Matrices (10 Tasks x 4 Classes, Seed 42)\n"
        "Section IX Headline Comparison: F-OAL vs. Experience Replay vs. LwF Baselines",
        fontsize=14.0,
        fontweight="bold",
        y=0.955,
    )
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_fig2_foal_vs_baselines(runs: List[Dict[str, Any]], out_path: Path) -> None:
    colors_map = {
        "foal": "#1f77b4",
        "er": "#2ca02c",
        "official_lwf": "#d62728",
        "adapted_lwf": "#ff7f0e",
    }
    labels = [r["label"] for r in runs]
    a_avg = [r["a_avg_pct"] for r in runs]
    a_last = [r["a_last_pct"] for r in runs]
    f_past = [r["f_past_pct"] for r in runs]
    colors = [colors_map.get(r["key"], "#4a5568") for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=300)
    x = np.arange(len(runs))

    panels = [
        (axes[0], a_avg, "Average Incremental Accuracy A_bar (%)", "Higher is Better (A_bar)", 105),
        (axes[1], a_last, "Last Stage Accuracy A_10 (%)", "Higher is Better (A_last)", 105),
        (axes[2], f_past, "Past-Tasks Forgetting F_past (%)", "Lower is Better (F_past)", 108),
    ]
    for ax, vals, ylabel, title, ylim in panels:
        bars = ax.bar(x, vals, color=colors, width=0.58, edgecolor="black", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.4,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=16, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, ylim)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.suptitle(
        "DTD Headline Comparison: F-OAL vs. Replay (ER) and Distillation (LwF) Baselines",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_fig3_adapted_vs_official_lwf(runs: List[Dict[str, Any]], out_path: Path) -> None:
    by_key = {r["key"]: r for r in runs}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=300)

    for key, style, marker, color in [
        ("adapted_lwf", "--", "v", "#ff7f0e"),
        ("official_lwf", "-", "^", "#d62728"),
    ]:
        if key not in by_key:
            continue
        r = by_key[key]
        stages = list(range(1, r["num_tasks"] + 1))
        ax1.plot(
            stages,
            r["stage_accs_pct"],
            linestyle=style,
            marker=marker,
            linewidth=2.2,
            color=color,
            label=f"{r['label']} (A_bar={r['a_avg_pct']:.2f}%, F_past={r['f_past_pct']:.2f}%)",
        )
        ax2.plot(
            stages,
            r["diag_accs_pct"],
            linestyle=style,
            marker=marker,
            linewidth=2.2,
            color=color,
            label=f"{r['label']} Diagonal R_(j,j)",
        )

    ax1.axhline(55.8, color="#7f7f7f", linestyle=":", linewidth=1.4, label="Paper LwF A_bar (55.8%)")
    ax1.set_title("Stage Incremental Accuracy A_k (Recency vs. Rigidity)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Task Stage (k)", fontsize=10.5)
    ax1.set_ylabel("Incremental Accuracy (%)", fontsize=10.5)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(range(1, 11))
    ax1.grid(True, linestyle="--", alpha=0.55)
    ax1.legend(fontsize=8.5, loc="upper right")

    ax2.set_title("New-Task Plasticity R_(j,j) at Introduction", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Newly Introduced Task (j)", fontsize=10.5)
    ax2.set_ylabel("Immediate Accuracy on New Task (%)", fontsize=10.5)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(range(1, 11))
    ax2.grid(True, linestyle="--", alpha=0.55)
    ax2.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        "Section IX Stability-Plasticity Analysis on DTD: Adapted LwF vs. Official-Protocol LwF",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_fig4_accuracy_progression(runs: List[Dict[str, Any]], out_path: Path) -> None:
    style_map = {
        "foal": ("-", "o", "#1f77b4", 2.5),
        "er": ("-", "s", "#2ca02c", 2.2),
        "official_lwf": ("-", "^", "#d62728", 2.2),
        "adapted_lwf": ("--", "v", "#ff7f0e", 1.8),
    }
    plt.figure(figsize=(10, 5.5), dpi=300)
    for r in runs:
        ls, mk, col, lw = style_map.get(r["key"], ("-", "o", "#4a5568", 1.8))
        stages = list(range(1, r["num_tasks"] + 1))
        plt.plot(
            stages,
            r["stage_accs_pct"],
            linestyle=ls,
            marker=mk,
            color=col,
            linewidth=lw,
            label=f"{r['label']} (A_bar = {r['a_avg_pct']:.2f}%)",
        )
    plt.xlabel("Task Stage (10 tasks x 4 classes = 40 DTD texture classes)", fontsize=11)
    plt.ylabel("Stage Incremental Accuracy (%)", fontsize=11)
    plt.title("DTD Incremental Accuracy Trajectory Across 10 Stages (Seed 42)", fontsize=12.5, fontweight="bold")
    plt.xticks(range(1, 11))
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=9.5, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_fig5_paper_vs_reproduced(runs: List[Dict[str, Any]], out_path: Path) -> None:
    color_map = {
        "foal": "#1f77b4",
        "er": "#2ca02c",
        "official_lwf": "#d62728",
        "adapted_lwf": "#ff7f0e",
    }
    labels = [r["label"] for r in runs]
    paper_vals = [r["paper_a_avg"] if r["paper_a_avg"] is not None else 0.0 for r in runs]
    ours_vals = [r["a_avg_pct"] for r in runs]
    ours_colors = [color_map.get(r["key"], "#4a5568") for r in runs]

    x = np.arange(len(runs))
    w = 0.35
    plt.figure(figsize=(9.5, 5.2), dpi=300)
    plt.bar(x - w / 2, paper_vals, w, label="Paper Table 2 (DTD)", color="#7f7f7f", alpha=0.85)
    plt.bar(x + w / 2, ours_vals, w, label="Our Reproduction (Seed 42)", color=ours_colors)
    for i in range(len(runs)):
        plt.text(x[i] - w / 2, paper_vals[i] + 1.0, f"{paper_vals[i]:.1f}%", ha="center", fontsize=9)
        plt.text(
            x[i] + w / 2,
            ours_vals[i] + 1.0,
            f"{ours_vals[i]:.2f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    plt.ylabel("Average Incremental Accuracy (%)", fontsize=11)
    plt.title("DTD Average Incremental Accuracy: NeurIPS 2024 Table 2 vs. Reproduction", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_fig6_forgetting_comparison(runs: List[Dict[str, Any]], out_path: Path) -> None:
    color_map = {
        "foal": "#1f77b4",
        "er": "#2ca02c",
        "official_lwf": "#d62728",
        "adapted_lwf": "#ff7f0e",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=300)

    tasks = list(range(1, 10))
    for r in runs:
        if len(r["task_forgetting_past_pct"]) >= 9:
            ax1.plot(
                tasks,
                r["task_forgetting_past_pct"][:9],
                marker="o",
                linewidth=2.0,
                color=color_map.get(r["key"], "#4a5568"),
                label=f"{r['label']} (Mean F_past={r['f_past_pct']:.2f}%)",
            )
    ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
    ax1.set_title("Per-Task Forgetting f_(10, j) Across Past Tasks 1..9", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Past Task (j)", fontsize=10.5)
    ax1.set_ylabel("Signed Forgetting (%)", fontsize=10.5)
    ax1.set_xticks(tasks)
    ax1.grid(True, linestyle="--", alpha=0.55)
    ax1.legend(fontsize=8.5, loc="best")

    x = np.arange(len(runs))
    w = 0.35
    f_past_vals = [r["f_past_pct"] for r in runs]
    f_code_vals = [r["f_code_pct"] for r in runs]
    ax2.bar(x - w / 2, f_past_vals, w, label="F_past (Paper Eq. 17, T-1)", color="#2b6cb0")
    ax2.bar(x + w / 2, f_code_vals, w, label="F_code (Author Code Eq. 9, T)", color="#90cdf4")
    for i in range(len(runs)):
        ax2.text(x[i] - w / 2, f_past_vals[i] + 1.0, f"{f_past_vals[i]:.2f}%", ha="center", fontsize=8.5)
        ax2.text(x[i] + w / 2, f_code_vals[i] + 1.0, f"{f_code_vals[i]:.2f}%", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["label"] for r in runs], rotation=14, ha="right", fontsize=9)
    ax2.set_ylabel("Average Forgetting (%)", fontsize=10.5)
    ax2.set_title("C0 Metric Comparison: F_past vs. F_code on DTD", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, max(max(f_past_vals + f_code_vals) + 12.0, 25.0))
    ax2.grid(True, linestyle="--", alpha=0.55, axis="y")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "DTD Catastrophic Forgetting Forensic Breakdown (Section IX + C0)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        dict(
            key="foal",
            label="F-OAL (Full, A0)",
            classification="OUR_INDEPENDENT_FOAL",
            paradigm="Analytic Forward-Only",
            buffer_size=0,
            run_dir=Path(args.foal_dir),
            paper_a_avg=82.8,
            paper_a_last=None,
            paper_f=None,
            default_optimizer="Analytic Ridge RLS (gamma=1.0)",
            default_arch="SmoothProjection(768->1000) + OnlineRidgeRLS",
            notes="Canonical DTD A0 / C3-D run (Seed 42). Zero replay, zero backprop.",
        ),
        dict(
            key="er",
            label="ER (Buffer=5000)",
            classification="OUR_ADAPTED_ER",
            paradigm="Experience Replay",
            buffer_size=5000,
            run_dir=Path(args.er_dir),
            paper_a_avg=73.2,
            paper_a_last=68.1,
            paper_f=18.5,
            default_optimizer="AdamW (lr=0.001, wd=1e-4)",
            default_arch="Dynamic nn.Linear(768, seen_classes) + ReservoirBuffer(5000)",
            notes="Adapted ER baseline on frozen ViT-B/16 Feature Fusion representations (D=768).",
        ),
        dict(
            key="official_lwf",
            label="LwF (Official Protocol)",
            classification="OUR_OFFICIAL_PROTOCOL_LWF",
            paradigm="Exemplar-Free Distillation",
            buffer_size=0,
            run_dir=Path(args.lwf_official_dir),
            paper_a_avg=55.8,
            paper_a_last=48.2,
            paper_f=12.6,
            default_optimizer="SGD (lr=0.001, momentum=0.0, wd=0.0)",
            default_arch="Static nn.Linear(768, 40)",
            notes="Author protocol: dynamic 1/(t+1) loss weighting, static 40-class head, full KD (T=2.0), pure SGD.",
        ),
        dict(
            key="adapted_lwf",
            label="LwF (Adapted AdamW)",
            classification="OUR_ADAPTED_LWF",
            paradigm="Exemplar-Free Distillation",
            buffer_size=0,
            run_dir=Path(args.lwf_adapted_dir),
            paper_a_avg=55.8,
            paper_a_last=48.2,
            paper_f=12.6,
            default_optimizer="AdamW (lr=0.001, wd=1e-4)",
            default_arch="Dynamic nn.Linear(768, seen_classes)",
            notes="Adapted LwF baseline with dynamic head and static 1.0/1.0 loss weighting.",
        ),
    ]

    analyzed_runs: List[Dict[str, Any]] = []
    missing_keys: List[str] = []
    for sp in specs:
        res = analyze_run(**sp)
        if res is None:
            missing_keys.append(sp["key"])
        else:
            analyzed_runs.append(res)

    if args.strict and missing_keys:
        raise RuntimeError(f"Missing required DTD runs in strict mode: {missing_keys}")
    if not analyzed_runs:
        raise RuntimeError("No DTD runs found to analyze!")

    # Export dtd_headline_final_metrics.json & .csv
    metrics_rows = [
        {
            "Method": r["label"],
            "Key": r["key"],
            "Classification": r["classification"],
            "Paradigm": r["paradigm"],
            "Buffer_Size": r["buffer_size"],
            "Replay": r["replay"],
            "A_avg_pct": round(r["a_avg_pct"], 6),
            "A_last_pct": round(r["a_last_pct"], 6),
            "F_past_pct": round(r["f_past_pct"], 6),
            "F_code_pct": round(r["f_code_pct"], 6),
            "Training_Time_s": round(r["duration_s"], 3),
            "Hardware": r["hardware"],
            "Feature_Dim": 768,
            "Classifier_Architecture": r["classifier_arch"],
            "Optimizer": r["optimizer"],
            "Git_SHA": r["git_sha"],
            "Split_Fingerprint_SHA256": r["split_sha256"],
            "Notes": r["notes"],
        }
        for r in analyzed_runs
    ]
    (out_dir / "dtd_headline_final_metrics.json").write_text(
        json.dumps(metrics_rows, indent=2), encoding="utf-8"
    )
    with (out_dir / "dtd_headline_final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    # Export dtd_headline_paper_comparison.csv
    paper_rows = []
    for r in analyzed_runs:
        p_avg = r["paper_a_avg"]
        p_last = r["paper_a_last"]
        p_f = r["paper_f"]
        paper_rows.append(
            {
                "Method": r["label"],
                "Buffer": r["buffer_size"],
                "Paper_A_avg": p_avg if p_avg is not None else "",
                "Ours_A_avg": round(r["a_avg_pct"], 4),
                "Delta_A_avg": round(r["a_avg_pct"] - p_avg, 4) if p_avg is not None else "",
                "Paper_A_last": p_last if p_last is not None else "",
                "Ours_A_last": round(r["a_last_pct"], 4),
                "Delta_A_last": round(r["a_last_pct"] - p_last, 4) if p_last is not None else "",
                "Paper_F": p_f if p_f is not None else "",
                "Ours_F_past": round(r["f_past_pct"], 4),
                "Ours_F_code": round(r["f_code_pct"], 4),
                "Delta_F_past": round(r["f_past_pct"] - p_f, 4) if p_f is not None else "",
            }
        )
    with (out_dir / "dtd_headline_paper_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paper_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paper_rows)

    # Export dtd_headline_protocol_matrix.csv
    protocol_rows = [
        {
            "Dimension": "Dataset",
            "FOAL_Full": "DTD (40/47 classes, 10x4, 40 train/test/cls)",
            "ER_Adapted": "DTD (40/47 classes, 10x4, 40 train/test/cls)",
            "LwF_Official": "DTD (40/47 classes, 10x4, 40 train/test/cls)",
            "LwF_Adapted": "DTD (40/47 classes, 10x4, 40 train/test/cls)",
        },
        {
            "Dimension": "Backbone",
            "FOAL_Full": "vit_base_patch16_224.augreg2_in21k_ft_in1k (Frozen)",
            "ER_Adapted": "vit_base_patch16_224.augreg2_in21k_ft_in1k (Frozen)",
            "LwF_Official": "vit_base_patch16_224.augreg2_in21k_ft_in1k (Frozen)",
            "LwF_Adapted": "vit_base_patch16_224.augreg2_in21k_ft_in1k (Frozen)",
        },
        {
            "Dimension": "Feature_Extraction",
            "FOAL_Full": "12-block CLS average + LayerNorm (D=768)",
            "ER_Adapted": "12-block CLS average + LayerNorm (D=768)",
            "LwF_Official": "12-block CLS average + LayerNorm (D=768)",
            "LwF_Adapted": "12-block CLS average + LayerNorm (D=768)",
        },
        {
            "Dimension": "Classifier_Head",
            "FOAL_Full": "SmoothProjection(768->1000) + Analytic Ridge RLS",
            "ER_Adapted": "Dynamic nn.Linear(768, seen_classes)",
            "LwF_Official": "Static nn.Linear(768, 40)",
            "LwF_Adapted": "Dynamic nn.Linear(768, seen_classes)",
        },
        {
            "Dimension": "Loss_Weighting",
            "FOAL_Full": "Closed-form L2 Ridge (gamma=1.0)",
            "ER_Adapted": "Cross-Entropy on stream + replay mini-batch",
            "LwF_Official": "Dynamic: 1/(t+1) L_new + (1 - 1/(t+1)) L_old (T=2.0)",
            "LwF_Adapted": "Static: 1.0 L_new + 1.0 L_old (T=2.0)",
        },
        {
            "Dimension": "Optimizer",
            "FOAL_Full": "Woodbury RLS (float64)",
            "ER_Adapted": "AdamW (lr=0.001, wd=1e-4)",
            "LwF_Official": "SGD (lr=0.001, momentum=0.0, wd=0.0)",
            "LwF_Adapted": "AdamW (lr=0.001, wd=1e-4)",
        },
        {
            "Dimension": "Replay_Buffer",
            "FOAL_Full": "0",
            "ER_Adapted": "5000 (Reservoir Sampling)",
            "LwF_Official": "0",
            "LwF_Adapted": "0",
        },
    ]
    with (out_dir / "dtd_headline_protocol_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(protocol_rows[0].keys()))
        writer.writeheader()
        writer.writerows(protocol_rows)

    # Export dtd_headline_stage_accuracies.csv
    num_stages = max(r["num_tasks"] for r in analyzed_runs)
    stage_rows = []
    by_key = {r["key"]: r for r in analyzed_runs}
    for s in range(num_stages):
        row = {"Stage": s + 1, "Num_Seen_Classes": (s + 1) * 4}
        for key in ("foal", "er", "official_lwf", "adapted_lwf"):
            if key in by_key and s < len(by_key[key]["stage_accs_pct"]):
                row[f"{key}_stage_pct"] = round(by_key[key]["stage_accs_pct"][s], 4)
                row[f"{key}_diag_pct"] = round(by_key[key]["diag_accs_pct"][s], 4)
            else:
                row[f"{key}_stage_pct"] = ""
                row[f"{key}_diag_pct"] = ""
        stage_rows.append(row)
    with (out_dir / "dtd_headline_stage_accuracies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stage_rows)

    # Export dtd_headline_task_forgetting.csv
    f_rows = []
    for j in range(max(0, num_stages - 1)):
        row = {"Task": j + 1}
        for key in ("foal", "er", "official_lwf", "adapted_lwf"):
            if key in by_key and j < len(by_key[key]["task_forgetting_past_pct"]):
                row[f"{key}_F_past_pct"] = round(by_key[key]["task_forgetting_past_pct"][j], 4)
            else:
                row[f"{key}_F_past_pct"] = ""
        f_rows.append(row)
    if f_rows:
        with (out_dir / "dtd_headline_task_forgetting.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(f_rows[0].keys()))
            writer.writeheader()
            writer.writerows(f_rows)

    # Generate all 6 figures
    _plot_fig1_task_matrices(analyzed_runs, out_dir / "fig1_dtd_headline_task_matrices.png")
    _plot_fig2_foal_vs_baselines(analyzed_runs, out_dir / "fig2_dtd_headline_foal_vs_baselines.png")
    _plot_fig3_adapted_vs_official_lwf(analyzed_runs, out_dir / "fig3_dtd_headline_adapted_vs_official_lwf.png")
    _plot_fig4_accuracy_progression(analyzed_runs, out_dir / "fig4_dtd_headline_accuracy_progression.png")
    _plot_fig5_paper_vs_reproduced(analyzed_runs, out_dir / "fig5_dtd_headline_paper_vs_reproduced.png")
    _plot_fig6_forgetting_comparison(analyzed_runs, out_dir / "fig6_dtd_headline_forgetting_comparison.png")

    # Export evidence manifest
    evidence_manifest = {
        "project": "F-OAL DTD Section IX Headline Comparison",
        "dataset": "DTD (tanganke/dtd, 40/47 classes, 10 tasks x 4 classes, 40 train / 40 test per class)",
        "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "feature_dim": 768,
        "feature_fusion": "12-block CLS average + LayerNorm (D=768)",
        "preprocessing": "Bilinear interpolation, unnormalized [0, 1]",
        "missing_runs": missing_keys,
        "runs": [
            {
                "key": r["key"],
                "method": r["label"],
                "classification": r["classification"],
                "run_dir": r["run_dir"],
                "evidence_file": r["evidence_file"],
                "evidence_sha256": r["evidence_sha256"],
                "split_sha256": r["split_sha256"],
                "git_sha": r["git_sha"],
                "metrics": {
                    "a_avg_pct": r["a_avg_pct"],
                    "a_last_pct": r["a_last_pct"],
                    "f_past_pct": r["f_past_pct"],
                    "f_code_pct": r["f_code_pct"],
                },
                "diagonal_new_task_accuracies_pct": r["diag_accs_pct"],
            }
            for r in analyzed_runs
        ],
    }
    (out_dir / "dtd_headline_evidence_manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2), encoding="utf-8"
    )

    # Print formatted summary table
    print("=" * 105)
    print(f"{'Method':<26} | {'Buffer':<8} | {'A_avg (%)':<10} | {'Paper A_avg':<11} | {'A_10 (%)':<10} | {'F_past (%)':<10} | {'F_code (%)':<10}")
    print("-" * 105)
    for r in analyzed_runs:
        p_str = f"{r['paper_a_avg']:.1f}%" if r["paper_a_avg"] is not None else "—"
        print(
            f"{r['label']:<26} | {r['buffer_size']:<8} | {r['a_avg_pct']:>8.2f}% | {p_str:>11} | "
            f"{r['a_last_pct']:>8.2f}% | {r['f_past_pct']:>8.2f}% | {r['f_code_pct']:>8.2f}%"
        )
    print("=" * 105)
    if missing_keys:
        print(f"Notice: Pending runs not yet found on disk: {missing_keys}")
    print(f"Analysis tables and 6 figures saved to {out_dir}")
    return evidence_manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Section IX DTD Headline Comparison analysis & figures.")
    p.add_argument("--foal-dir", default="results/dtd/dtd_a0_full", help="Directory for Full F-OAL DTD run")
    p.add_argument("--er-dir", default="runs/headline_dtd/er", help="Directory for ER DTD run")
    p.add_argument(
        "--lwf-official-dir",
        default="runs/headline_dtd/lwf_official_dtd",
        help="Directory for Official-Protocol LwF DTD run",
    )
    p.add_argument(
        "--lwf-adapted-dir",
        default="runs/headline_dtd/lwf",
        help="Directory for Adapted LwF DTD run",
    )
    p.add_argument(
        "--output-dir",
        default="results/headline_dtd_analysis",
        help="Output directory for CSV/JSON tables and PNG figures",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any of the 4 runs is missing",
    )
    return p


if __name__ == "__main__":
    run_analysis(build_parser().parse_args())
