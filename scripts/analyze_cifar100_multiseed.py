"""Statistical Analysis, Independent Audit, and Visualization for CIFAR-100 Multi-Seed Ablation.

Loads the 4 verified Seed 42 runs and the 8 new Seed 123 / 2026 runs, verifies
protocol parity and paired-comparison invariants, computes Mean ± Std (ddof=1),
paired component contributions (ΔFF, ΔSP, ΔBoth), 2x2 factorial effects, and
generates 4 publication-grade 300 DPI PNG figures.

Usage:
    python -m scripts.analyze_cifar100_multiseed
    python -m scripts.analyze_cifar100_multiseed --input-root results/cifar100_multiseed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cifar100_multiseed_ablation import (
    ABLATIONS,
    SEEDS,
    load_task_matrix_csv,
    validate_completed_run,
)
from src.foal.metrics import stage_accuracies, summary_from_task_matrix

ABLATION_ORDER = ["full", "no_ff", "no_sp", "no_ff_no_sp"]
ABLATION_LABELS = {
    "full": "Full F-OAL",
    "no_ff": "Without FF",
    "no_sp": "Without SP",
    "no_ff_no_sp": "Without FF & SP",
}
ABLATION_COLORS = {
    "full": "#1f77b4",
    "no_ff": "#ff7f0e",
    "no_sp": "#2ca02c",
    "no_ff_no_sp": "#d62728",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample_std) with ddof=1."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def load_all_runs(input_root: Path) -> dict[int, dict[str, dict[str, Any]]]:
    """Load and validate all 12 (seed x ablation) runs."""
    data: dict[int, dict[str, dict[str, Any]]] = {}

    for seed in SEEDS:
        data[seed] = {}
        seed_fps: set[str] = set()
        for abl_key in ABLATION_ORDER:
            if seed == 42:
                run_dir = ABLATIONS[abl_key]["seed42_dir"]
                require_ckpts = False
                if not run_dir.exists():
                    manifest_path = REPO_ROOT / "results" / "cifar100_multiseed" / "seed_manifest.json"
                    if not manifest_path.exists():
                        raise RuntimeError(f"Seed 42 artifacts and {manifest_path} both missing.")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    s42_rec = manifest.get("seed_42_verified_runs", {}).get(abl_key)
                    if not s42_rec or "stage_accuracies_pct" not in s42_rec:
                        raise RuntimeError(f"Seed 42 manifest missing verified record for '{abl_key}'.")
                    seed_fps.add(str(s42_rec["task_split_fingerprint"]))
                    data[seed][abl_key] = {**s42_rec}
                    continue
            else:
                run_dir = input_root / f"seed_{seed}" / abl_key
                require_ckpts = True

            ok, reasons, summary = validate_completed_run(
                run_dir,
                expected_seed=seed,
                ablation_key=abl_key,
                is_pilot=False,
                expected_tasks=10,
                require_checkpoints=require_ckpts,
            )
            if not ok or summary is None:
                raise RuntimeError(
                    f"Validation failed for seed={seed}, ablation={abl_key} ({run_dir}): {reasons}"
                )

            csv_path = run_dir / "task_accuracy.csv"
            matrix = load_task_matrix_csv(csv_path, expected_stages=10)
            stages_pct = [s * 100.0 for s in stage_accuracies(matrix)]

            seed_fps.add(str(summary["task_split_fingerprint"]))
            data[seed][abl_key] = {
                **summary,
                "csv_sha256": sha256_file(csv_path),
                "stage_accuracies_pct": stages_pct,
                "matrix": matrix,
            }

        if len(seed_fps) != 1:
            raise RuntimeError(
                f"Paired comparison invariant violated for seed={seed}: "
                f"distinct task_split_fingerprints={seed_fps}"
            )

    # Verify different seeds actually used distinct class orders / splits
    class_orders = {s: data[s]["full"]["class_order_sha256"] for s in SEEDS}
    if len(set(class_orders.values())) != len(SEEDS):
        raise RuntimeError(f"Expected distinct class orders across seeds, got: {class_orders}")

    return data


def compute_statistical_summary(data: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Compute Mean ± Std (ddof=1), paired differences, and 2x2 factorial effects."""
    metric_keys = {
        "average_incremental_accuracy_pct": "Average Incremental Accuracy (%)",
        "last_accuracy_pct": "Final Accuracy (%)",
        "forgetting_past_tasks_pct": "Past Forgetting (%)",
    }

    summary_tables: dict[str, dict[str, Any]] = {}
    for m_key, m_label in metric_keys.items():
        summary_tables[m_key] = {"label": m_label, "unit": "%", "rows": {}}
        for abl_key in ABLATION_ORDER:
            per_seed = {str(s): data[s][abl_key][m_key] for s in SEEDS}
            mean, std = sample_mean_std(list(per_seed.values()))
            summary_tables[m_key]["rows"][abl_key] = {
                "label": ABLATION_LABELS[abl_key],
                "per_seed": per_seed,
                "mean": mean,
                "std_ddof1": std,
                "mean_std_str": f"{mean:.4f} ± {std:.4f}%",
            }

    # Paired Effect Analysis across seeds (unit: percentage points, pp)
    paired_metrics: dict[str, Any] = {}
    for m_key in ("average_incremental_accuracy_pct", "last_accuracy_pct", "forgetting_past_tasks_pct"):
        delta_ff_seeds: dict[str, float] = {}
        delta_sp_seeds: dict[str, float] = {}
        delta_both_seeds: dict[str, float] = {}
        e_ff_main_seeds: dict[str, float] = {}
        e_sp_main_seeds: dict[str, float] = {}
        e_inter_seeds: dict[str, float] = {}

        for s in SEEDS:
            y_11 = data[s]["full"][m_key]
            y_01 = data[s]["no_ff"][m_key]
            y_10 = data[s]["no_sp"][m_key]
            y_00 = data[s]["no_ff_no_sp"][m_key]

            delta_ff_seeds[str(s)] = y_11 - y_01
            delta_sp_seeds[str(s)] = y_11 - y_10
            delta_both_seeds[str(s)] = y_11 - y_00

            # 2x2 factorial effects
            e_ff_main_seeds[str(s)] = 0.5 * ((y_11 - y_01) + (y_10 - y_00))
            e_sp_main_seeds[str(s)] = 0.5 * ((y_11 - y_10) + (y_01 - y_00))
            e_inter_seeds[str(s)] = (y_11 - y_01) - (y_10 - y_00)

        def summarize_contrast(per_seed_dict: dict[str, float]) -> dict[str, Any]:
            vals = list(per_seed_dict.values())
            mean, std = sample_mean_std(vals)
            pos_count = sum(1 for v in vals if v > 0.0)
            neg_count = sum(1 for v in vals if v < 0.0)
            return {
                "per_seed_pp": per_seed_dict,
                "mean_pp": mean,
                "std_ddof1_pp": std,
                "positive_seeds": pos_count,
                "negative_seeds": neg_count,
                "consistent_positive": pos_count == len(SEEDS),
                "consistent_negative": neg_count == len(SEEDS),
                "mean_std_str": f"{mean:+.4f} ± {std:.4f} pp",
            }

        paired_metrics[m_key] = {
            "delta_ff": summarize_contrast(delta_ff_seeds),
            "delta_sp": summarize_contrast(delta_sp_seeds),
            "delta_both": summarize_contrast(delta_both_seeds),
            "factorial_ff_main": summarize_contrast(e_ff_main_seeds),
            "factorial_sp_main": summarize_contrast(e_sp_main_seeds),
            "factorial_interaction": summarize_contrast(e_inter_seeds),
        }

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS,
        "sample_size_n": len(SEEDS),
        "std_ddof": 1,
        "summary_tables": summary_tables,
        "paired_effects": paired_metrics,
    }


def save_csv_outputs(stats: dict[str, Any], out_dir: Path) -> list[Path]:
    """Write summary and paired-effect CSV tables."""
    summary_csv = out_dir / "multiseed_summary_table.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "configuration", "seed_42", "seed_123", "seed_2026", "mean_pct", "std_ddof1_pct"])
        for m_key, m_block in stats["summary_tables"].items():
            for abl_key in ABLATION_ORDER:
                row = m_block["rows"][abl_key]
                writer.writerow([
                    m_key,
                    row["label"],
                    f"{row['per_seed']['42']:.6f}",
                    f"{row['per_seed']['123']:.6f}",
                    f"{row['per_seed']['2026']:.6f}",
                    f"{row['mean']:.6f}",
                    f"{row['std_ddof1']:.6f}",
                ])

    paired_csv = out_dir / "multiseed_paired_effects.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric", "contrast", "seed_42_pp", "seed_123_pp", "seed_2026_pp",
            "mean_pp", "std_ddof1_pp", "positive_seeds",
        ])
        for m_key, p_block in stats["paired_effects"].items():
            for c_key in ("delta_ff", "delta_sp", "delta_both", "factorial_ff_main", "factorial_sp_main", "factorial_interaction"):
                c = p_block[c_key]
                writer.writerow([
                    m_key,
                    c_key,
                    f"{c['per_seed_pp']['42']:.6f}",
                    f"{c['per_seed_pp']['123']:.6f}",
                    f"{c['per_seed_pp']['2026']:.6f}",
                    f"{c['mean_pp']:.6f}",
                    f"{c['std_ddof1_pp']:.6f}",
                    c["positive_seeds"],
                ])

    return [summary_csv, paired_csv]


def generate_figures(
    data: dict[int, dict[str, dict[str, Any]]],
    stats: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """Generate the 4 publication-grade 300 DPI PNG figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    saved_paths: list[Path] = []

    # Figure 1: Multi-Seed Ablation Accuracy (A_avg and A_10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    x = np.arange(len(ABLATION_ORDER))
    seed_markers = {42: "o", 123: "s", 2026: "^"}

    for ax_idx, (m_key, title) in enumerate([
        ("average_incremental_accuracy_pct", "Average Incremental Accuracy ($\\bar{A}$)"),
        ("last_accuracy_pct", "Final Accuracy ($A_{10}$)"),
    ]):
        ax = axes[ax_idx]
        rows = stats["summary_tables"][m_key]["rows"]
        means = [rows[k]["mean"] for k in ABLATION_ORDER]
        stds = [rows[k]["std_ddof1"] for k in ABLATION_ORDER]
        colors = [ABLATION_COLORS[k] for k in ABLATION_ORDER]

        ax.bar(x, means, yerr=stds, capsize=5, width=0.58, color=colors, alpha=0.78, edgecolor="#222222", linewidth=0.8)
        for s_idx, seed in enumerate(SEEDS):
            offset = (s_idx - 1) * 0.12
            ys = [rows[k]["per_seed"][str(seed)] for k in ABLATION_ORDER]
            ax.scatter(x + offset, ys, marker=seed_markers[seed], color="#111111", s=38, zorder=5, label=f"Seed {seed}")

        y_min = max(0.0, min(means) - max(stds) - 2.5)
        y_max = min(100.0, max(means) + max(stds) + 2.0)
        ax.set_ylim(y_min, y_max)
        for i, (m_val, s_val) in enumerate(zip(means, stds)):
            ax.text(i, m_val + s_val + 0.25, f"{m_val:.2f}±{s_val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_LABELS[k] for k in ABLATION_ORDER], fontsize=9.5, rotation=12)
        ax.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    fig.suptitle("CIFAR-100 Multi-Seed Table 4 Ablation Accuracy (Seeds 42, 123, 2026; ddof=1)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig1_path = out_dir / "fig1_multiseed_ablation_accuracy.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    saved_paths.append(fig1_path)

    # Figure 2: Paired Component Contributions (ΔFF, ΔSP, ΔBoth)
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    contrasts = [("delta_ff", "ΔFF (Full − No FF)"), ("delta_sp", "ΔSP (Full − No SP)"), ("delta_both", "ΔBoth (Full − Neither)")]
    p_block = stats["paired_effects"]["average_incremental_accuracy_pct"]
    cx = np.arange(len(contrasts))
    width = 0.2

    for s_idx, seed in enumerate(SEEDS):
        vals = [p_block[c_key]["per_seed_pp"][str(seed)] for c_key, _ in contrasts]
        ax.bar(cx + (s_idx - 1) * width, vals, width=width, label=f"Seed {seed}", alpha=0.85)

    mean_vals = [p_block[c_key]["mean_pp"] for c_key, _ in contrasts]
    std_vals = [p_block[c_key]["std_ddof1_pp"] for c_key, _ in contrasts]
    ax.errorbar(cx, mean_vals, yerr=std_vals, fmt="D", color="#111111", capsize=6, markersize=7, linewidth=1.6, label="Mean ± Std (ddof=1)", zorder=6)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)

    for i, (m_val, s_val) in enumerate(zip(mean_vals, std_vals)):
        ax.text(i + 0.28, m_val, f"{m_val:+.3f}±{s_val:.3f} pp", va="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(cx)
    ax.set_xticklabels([lbl for _, lbl in contrasts], fontsize=10, fontweight="bold")
    ax.set_ylabel("Paired Accuracy Gain (percentage points, pp)", fontsize=10.5, fontweight="bold")
    ax.set_title("Paired Component Contributions on Average Incremental Accuracy ($\\bar{A}$)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig2_path = out_dir / "fig2_paired_component_contributions.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    saved_paths.append(fig2_path)

    # Figure 3: Accuracy Progression Across 10 Tasks (Mean ± Std Band)
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    tasks = np.arange(1, 11)
    for abl_key in ABLATION_ORDER:
        curves = np.array([data[s][abl_key]["stage_accuracies_pct"] for s in SEEDS], dtype=float)
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0, ddof=1)
        color = ABLATION_COLORS[abl_key]
        ax.plot(tasks, mean_curve, marker="o", linewidth=2.0, color=color, label=ABLATION_LABELS[abl_key])
        ax.fill_between(tasks, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.14)

    ax.set_xticks(tasks)
    ax.set_xlabel("Incremental Training Stage (Task 1–10)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Seen-Class Average Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("CIFAR-100 Stage Accuracy Progression (Mean ± 1 Std across Seeds 42, 123, 2026)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig3_path = out_dir / "fig3_multiseed_accuracy_progression.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    saved_paths.append(fig3_path)

    # Figure 4: Accuracy vs Past Forgetting
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=300)
    acc_rows = stats["summary_tables"]["average_incremental_accuracy_pct"]["rows"]
    fgt_rows = stats["summary_tables"]["forgetting_past_tasks_pct"]["rows"]

    for abl_key in ABLATION_ORDER:
        a_m, a_s = acc_rows[abl_key]["mean"], acc_rows[abl_key]["std_ddof1"]
        f_m, f_s = fgt_rows[abl_key]["mean"], fgt_rows[abl_key]["std_ddof1"]
        color = ABLATION_COLORS[abl_key]
        ax.errorbar(
            f_m, a_m, xerr=f_s, yerr=a_s, fmt="o", markersize=9, capsize=5,
            color=color, linewidth=1.8, label=f"{ABLATION_LABELS[abl_key]} (A={a_m:.2f}%, F={f_m:.2f}%)",
        )
        for s in SEEDS:
            ax.scatter(
                fgt_rows[abl_key]["per_seed"][str(s)],
                acc_rows[abl_key]["per_seed"][str(s)],
                color=color, alpha=0.45, s=28,
            )

    ax.set_xlabel("Past Forgetting $F_{\\text{past}}$ (%, lower is better)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Average Incremental Accuracy $\\bar{A}$ (%, higher is better)", fontsize=10.5, fontweight="bold")
    ax.set_title("Accuracy vs. Past Forgetting across Configurations (3 Seeds)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=8.8)
    fig.tight_layout()
    fig4_path = out_dir / "fig4_accuracy_vs_forgetting.png"
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    saved_paths.append(fig4_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CIFAR-100 Multi-Seed Ablation Study.")
    parser.add_argument("--input-root", default=str(REPO_ROOT / "results" / "cifar100_multiseed"))
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-root>/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    out_dir = Path(args.output_dir) if args.output_dir else (input_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_runs(input_root)
    stats = compute_statistical_summary(data)

    summary_json = out_dir / "multiseed_summary.json"
    summary_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    csv_paths = save_csv_outputs(stats, out_dir)
    fig_paths = generate_figures(data, stats, out_dir)

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
        except ValueError:
            return str(p.resolve())

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_runs": {
            str(s): {
                k: {
                    "run_dir": data[s][k]["run_dir"],
                    "csv_sha256": data[s][k]["csv_sha256"],
                    "task_split_fingerprint": data[s][k]["task_split_fingerprint"],
                }
                for k in ABLATION_ORDER
            }
            for s in SEEDS
        },
        "outputs": {_rel(p): sha256_file(p) for p in [summary_json, *csv_paths, *fig_paths]},
    }
    (out_dir / "multiseed_evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("MULTI-SEED ANALYSIS COMPLETED")
    for abl_key in ABLATION_ORDER:
        r = stats["summary_tables"]["average_incremental_accuracy_pct"]["rows"][abl_key]
        print(f"  {r['label']:<18}: {r['mean_std_str']}")


if __name__ == "__main__":
    main()
