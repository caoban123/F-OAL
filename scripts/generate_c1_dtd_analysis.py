"""Generate C1 DTD Analysis Artifacts, Tables, and Figures.

Reads real `metrics.json` (and optional `metadata.json`) from C1 DTD training runs
(`p0_gaussian_std1`, `p1_official_uniform`, `p2_gaussian_matched_variance`) plus
optional matched-input diagnostics JSON, recomputes metrics from `task_matrix`
for verification, and emits comparison tables, CSVs, JSON summaries, and plots.

Usage:
    python -m scripts.generate_c1_dtd_analysis --results-dir results/dtd_c1_full --output-dir results/dtd_c1_analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.foal.metrics import summary_from_task_matrix


VARIANTS = [
    ("p0_gaussian_std1", "P0 (Gaussian std=1)", "P0: gaussian_std1", "Gaussian N(0, 1.0)", "1.0", "gaussian_std1"),
    ("p1_official_uniform", "P1 (Official Uniform)", "P1: official_uniform", "Uniform U(-1/sqrt(3d), +1/sqrt(3d))", "1/(3*768) ≈ 0.000434", "official_uniform"),
    ("p2_gaussian_matched_variance", "P2 (Matched Gaussian)", "P2: gaussian_matched_variance", "Gaussian N(0, 1/(3*768))", "1/(3*768) ≈ 0.000434", "gaussian_matched_variance"),
]


def load_run_bundle(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    recalc = summary_from_task_matrix(metrics["task_matrix"])
    return {"metrics": metrics, "metadata": meta, "recalculated": recalc}


def run_analysis(
    results_dir: Path,
    output_dir: Path,
    diag_file: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {folder: load_run_bundle(results_dir / folder) for folder, *_ in VARIANTS}

    diag_configs: dict[str, Any] = {}
    if diag_file is not None and diag_file.exists():
        diag_data = json.loads(diag_file.read_text(encoding="utf-8"))
        diag_configs = diag_data.get("diagnostics", {}).get("configurations", {})

    num_tasks = min(len(runs[folder]["recalculated"]["stage_accuracies"]) for folder, *_ in VARIANTS)

    # 1. Stage Accuracies Table
    stage_table = []
    for stage in range(1, num_tasks + 1):
        row: dict[str, Any] = {"Stage": stage}
        for folder, col_label, *_ in VARIANTS:
            row[col_label] = round(runs[folder]["recalculated"]["stage_accuracies"][stage - 1] * 100, 2)
        stage_table.append(row)
    df_stages = pd.DataFrame(stage_table)
    df_stages.to_csv(output_dir / "c1_dtd_stage_accuracies.csv", index=False)

    # 2. Per-Task Final Accuracy (final row of task_matrix)
    final_task_table = []
    for task in range(1, num_tasks + 1):
        row = {"Task": task}
        for folder, col_label, *_ in VARIANTS:
            short = col_label.split(" ")[0]
            row[short] = round(runs[folder]["metrics"]["task_matrix"][num_tasks - 1][task - 1] * 100, 2)
        final_task_table.append(row)
    df_final = pd.DataFrame(final_task_table)
    df_final.to_csv(output_dir / "c1_dtd_final_task_accuracies.csv", index=False)

    # 3. Per-Task Forgetting (Past-Tasks Formula)
    fgt_past_table = []
    for task in range(1, max(1, num_tasks)):
        row = {"Task": task}
        for folder, col_label, *_ in VARIANTS:
            short = col_label.split(" ")[0]
            vals = runs[folder]["recalculated"]["task_forgetting_past_tasks"]
            row[short] = round(vals[task - 1] * 100, 2) if task - 1 < len(vals) else 0.0
        fgt_past_table.append(row)
    df_fgt = pd.DataFrame(fgt_past_table)
    df_fgt.to_csv(output_dir / "c1_dtd_task_forgetting.csv", index=False)

    # 4. Comprehensive Summary Table
    comp_rows = []
    for folder, _col_label, variant_name, init_desc, var_scale, diag_key in VARIANTS:
        bundle = runs[folder]
        rec = bundle["recalculated"]
        m = bundle["metrics"]
        d_cfg = diag_configs.get(diag_key, {})
        u_std = d_cfg.get("pre_sigmoid_u", {}).get("std")
        sat = d_cfg.get("post_sigmoid_z", {}).get("total_saturation_fraction")
        comp_rows.append({
            "Variant": variant_name,
            "Initialization": init_desc,
            "Variance Scale": var_scale,
            "Avg Incr Acc (%)": round(rec["average_incremental_accuracy"] * 100, 4),
            "Last Acc (%)": round(rec["last_accuracy"] * 100, 4),
            "Fgt Past-Tasks (%)": round(rec["forgetting_past_tasks"] * 100, 4),
            "Fgt Official Code (%)": round(rec["forgetting_official_code"] * 100, 4),
            "Pre-sigmoid Std (Diagnostics)": round(u_std, 4) if u_std is not None else None,
            "Sigmoid Saturation (%)": round(sat * 100, 2) if sat is not None else None,
            "Feat Extraction Time (s)": round(m.get("feature_extraction_seconds", 0.0), 2),
            "RLS & Eval Time (s)": round(m.get("rls_and_eval_seconds", 0.0), 2),
            "Total Time (s)": round(m.get("total_duration_seconds", 0.0), 2),
        })

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(output_dir / "c1_dtd_full_comparison.csv", index=False)

    p0_acc = runs["p0_gaussian_std1"]["recalculated"]["average_incremental_accuracy"]
    p1_acc = runs["p1_official_uniform"]["recalculated"]["average_incremental_accuracy"]
    p2_acc = runs["p2_gaussian_matched_variance"]["recalculated"]["average_incremental_accuracy"]
    summary_payload = {
        "dataset": "tanganke/dtd",
        "num_tasks": num_tasks,
        "comparison_table": comp_rows,
        "deltas": {
            "p1_minus_p0_avg_acc_pp": round((p1_acc - p0_acc) * 100, 4),
            "p2_minus_p0_avg_acc_pp": round((p2_acc - p0_acc) * 100, 4),
            "p2_minus_p1_avg_acc_pp": round((p2_acc - p1_acc) * 100, 4),
        },
    }
    (output_dir / "c1_dtd_full_comparison.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("--- C1 DTD STAGE ACCURACY TRAJECTORY (%) ---")
    print(df_stages.to_string(index=False))
    print("\n--- C1 DTD COMPREHENSIVE COMPARISON TABLE ---")
    print(df_comp.to_string(index=False))

    # 5. Plot Stage Accuracy Comparison Chart
    plt.figure(figsize=(10, 6))
    stages = list(range(1, num_tasks + 1))
    styles = [
        ("p0_gaussian_std1", "P0: Gaussian std=1", "o", "#d9534f", "-"),
        ("p1_official_uniform", "P1: Official Uniform", "s", "#0275d8", "--"),
        ("p2_gaussian_matched_variance", "P2: Matched Gaussian", "^", "#5cb85c", ":"),
    ]
    all_vals = []
    for folder, name, marker, color, ls in styles:
        rec = runs[folder]["recalculated"]
        curve = [a * 100 for a in rec["stage_accuracies"][:num_tasks]]
        all_vals.extend(curve)
        lbl = f"{name} (A_avg={rec['average_incremental_accuracy']*100:.2f}%, A_last={rec['last_accuracy']*100:.2f}%)"
        plt.plot(stages, curve, marker=marker, color=color, linewidth=2.2, linestyle=ls, label=lbl)

    plt.title("DTD Incremental Accuracy across Stages (C1 Controlled Experiment)", fontsize=13, fontweight="bold")
    plt.xlabel("Incremental Task / Stage", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(stages)
    if all_vals:
        plt.ylim(max(0, min(all_vals) - 5), min(102, max(all_vals) + 5))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    chart_path = output_dir / "c1_dtd_stage_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nGenerated stage accuracy plot at: {chart_path}")
    return summary_payload


def main():
    p = argparse.ArgumentParser(description="Generate C1 DTD Analysis Tables and Plots")
    p.add_argument("--results-dir", type=Path, default=Path("results/dtd_c1_full"))
    p.add_argument("--output-dir", type=Path, default=Path("results/dtd_c1_analysis"))
    p.add_argument("--diag-file", type=Path, default=Path("results/dtd_c1_diagnostics/c1_dtd_matched_input_diagnostics.json"))
    args = p.parse_args()
    run_analysis(args.results_dir, args.output_dir, args.diag_file)


if __name__ == "__main__":
    main()
