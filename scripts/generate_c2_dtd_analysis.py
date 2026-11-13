"""Generate C2 DTD Analysis Artifacts, Tables, and Figures.

Reads real `metrics.json` and `metadata.json` from C2 DTD training runs
(`b0_augreg_in1k` and `b1_augreg2_in21k_ft_in1k`) plus C1-P1 reference,
computes representation effect deltas (B1 - B0), parses diagnostics,
and emits comparison tables, CSVs, JSON summaries, and plots.

Usage:
    python -m scripts.generate_c2_dtd_analysis --results-dir results/dtd_c2_full --output-dir results/dtd_c2_analysis
"""
from __future__ import annotations

import argparse
import csv
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


def load_bundle(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    recalc = summary_from_task_matrix(metrics["task_matrix"])
    return {"metrics": metrics, "metadata": meta, "recalculated": recalc}


def run_analysis(
    results_dir: Path,
    output_dir: Path,
    c1_p1_dir: Path | None = None,
    diag_file: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    b0_bundle = load_bundle(results_dir / "b0_augreg_in1k")
    b1_bundle = load_bundle(results_dir / "b1_augreg2_in21k_ft_in1k")

    if b0_bundle is None or b1_bundle is None:
        raise FileNotFoundError(f"Missing b0 or b1 results in {results_dir}")

    c1_bundle = load_bundle(c1_p1_dir) if c1_p1_dir and c1_p1_dir.exists() else None

    diag_data = {}
    if diag_file and diag_file.exists():
        diag_data = json.loads(diag_file.read_text(encoding="utf-8"))

    b0_rec = b0_bundle["recalculated"]
    b1_rec = b1_bundle["recalculated"]
    b0_m = b0_bundle["metrics"]
    b1_m = b1_bundle["metrics"]

    num_tasks = min(len(b0_rec["stage_accuracies"]), len(b1_rec["stage_accuracies"]))

    # 1. Comparison table
    conditions = [
        {
            "condition": "C2-B0 (Control)",
            "short_name": "B0: augreg_in1k",
            "backbone": "vit_base_patch16_224.augreg_in1k",
            "pretrain_data": "ImageNet-1k",
            "seed": 42,
            "gamma": 1.0,
            "avg_incremental_acc": round(b0_rec["average_incremental_accuracy"] * 100, 4),
            "last_acc": round(b0_rec["last_accuracy"] * 100, 4),
            "forgetting_past": round(b0_rec["forgetting_past_tasks"] * 100, 4),
            "forgetting_code": round(b0_rec["forgetting_official_code"] * 100, 4),
            "total_time_sec": round(b0_m.get("total_duration_seconds", 0.0), 2),
            "rls_time_sec": round(b0_m.get("rls_and_eval_seconds", 0.0), 2),
            "feat_time_sec": round(b0_m.get("feature_extraction_seconds", 0.0), 2),
        },
        {
            "condition": "C2-B1 (Treatment)",
            "short_name": "B1: augreg2_in21k_ft_in1k",
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "pretrain_data": "ImageNet-21k + IN-1k FT",
            "seed": 42,
            "gamma": 1.0,
            "avg_incremental_acc": round(b1_rec["average_incremental_accuracy"] * 100, 4),
            "last_acc": round(b1_rec["last_accuracy"] * 100, 4),
            "forgetting_past": round(b1_rec["forgetting_past_tasks"] * 100, 4),
            "forgetting_code": round(b1_rec["forgetting_official_code"] * 100, 4),
            "total_time_sec": round(b1_m.get("total_duration_seconds", 0.0), 2),
            "rls_time_sec": round(b1_m.get("rls_and_eval_seconds", 0.0), 2),
            "feat_time_sec": round(b1_m.get("feature_extraction_seconds", 0.0), 2),
        },
    ]

    if c1_bundle is not None:
        c1_rec = c1_bundle["recalculated"]
        c1_m = c1_bundle["metrics"]
        conditions.append({
            "condition": "C1-P1 (Reference Identity)",
            "short_name": "C1-P1 Reference",
            "backbone": "vit_base_patch16_224.augreg_in1k",
            "pretrain_data": "ImageNet-1k",
            "seed": 42,
            "gamma": 1.0,
            "avg_incremental_acc": round(c1_rec["average_incremental_accuracy"] * 100, 4),
            "last_acc": round(c1_rec["last_accuracy"] * 100, 4),
            "forgetting_past": round(c1_rec["forgetting_past_tasks"] * 100, 4),
            "forgetting_code": round(c1_rec["forgetting_official_code"] * 100, 4),
            "total_time_sec": round(c1_m.get("total_duration_seconds", 0.0), 2),
            "rls_time_sec": round(c1_m.get("rls_and_eval_seconds", 0.0), 2),
            "feat_time_sec": round(c1_m.get("feature_extraction_seconds", 0.0), 2),
        })

    csv_path = output_dir / "c2_dtd_full_comparison.csv"
    pd.DataFrame(conditions).to_csv(csv_path, index=False)

    deltas: dict[str, Any] = {
        "b1_minus_b0": {
            "avg_incremental_acc_pp": round((b1_rec["average_incremental_accuracy"] - b0_rec["average_incremental_accuracy"]) * 100, 4),
            "last_acc_pp": round((b1_rec["last_accuracy"] - b0_rec["last_accuracy"]) * 100, 4),
            "forgetting_past_pp": round((b1_rec["forgetting_past_tasks"] - b0_rec["forgetting_past_tasks"]) * 100, 4),
            "forgetting_code_pp": round((b1_rec["forgetting_official_code"] - b0_rec["forgetting_official_code"]) * 100, 4),
        }
    }
    if c1_bundle is not None:
        c1_rec = c1_bundle["recalculated"]
        deltas["b0_minus_c1_p1"] = {
            "avg_incremental_acc_pp": round((b0_rec["average_incremental_accuracy"] - c1_rec["average_incremental_accuracy"]) * 100, 4),
            "last_acc_pp": round((b0_rec["last_accuracy"] - c1_rec["last_accuracy"]) * 100, 4),
            "forgetting_past_pp": round((b0_rec["forgetting_past_tasks"] - c1_rec["forgetting_past_tasks"]) * 100, 4),
        }

    json_payload = {
        "dataset": "tanganke/dtd",
        "num_tasks": num_tasks,
        "comparison_table": conditions,
        "deltas": deltas,
    }
    (output_dir / "c2_dtd_full_comparison.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    # 2. Stage accuracies CSV
    stages_rows = []
    for s in range(num_tasks):
        s_b0 = b0_rec["stage_accuracies"][s]
        s_b1 = b1_rec["stage_accuracies"][s]
        row_dict = {
            "stage": s + 1,
            "c2_b0": round(s_b0 * 100, 2),
            "c2_b1": round(s_b1 * 100, 2),
            "delta_b1_b0_pp": round((s_b1 - s_b0) * 100, 2),
        }
        if c1_bundle is not None:
            row_dict["c1_p1_ref"] = round(c1_bundle["recalculated"]["stage_accuracies"][s] * 100, 2)
        stages_rows.append(row_dict)
    pd.DataFrame(stages_rows).to_csv(output_dir / "c2_dtd_stage_accuracies.csv", index=False)

    # 3. Final task accuracies CSV
    final_rows = []
    for t in range(num_tasks):
        acc_b0 = b0_bundle["metrics"]["task_matrix"][num_tasks - 1][t]
        acc_b1 = b1_bundle["metrics"]["task_matrix"][num_tasks - 1][t]
        row_dict = {
            "task": t + 1,
            "c2_b0_final_acc": round(acc_b0 * 100, 2),
            "c2_b1_final_acc": round(acc_b1 * 100, 2),
            "delta_b1_b0_pp": round((acc_b1 - acc_b0) * 100, 2),
        }
        if c1_bundle is not None:
            row_dict["c1_p1_final_acc"] = round(c1_bundle["metrics"]["task_matrix"][num_tasks - 1][t] * 100, 2)
        final_rows.append(row_dict)
    pd.DataFrame(final_rows).to_csv(output_dir / "c2_dtd_final_task_accuracies.csv", index=False)

    # 4. Task forgetting CSV
    fgt_rows = []
    for t in range(max(1, num_tasks - 1)):
        pf_b0 = b0_rec["task_forgetting_past_tasks"][t] if t < len(b0_rec["task_forgetting_past_tasks"]) else 0.0
        pf_b1 = b1_rec["task_forgetting_past_tasks"][t] if t < len(b1_rec["task_forgetting_past_tasks"]) else 0.0
        fgt_rows.append({
            "task": t + 1,
            "c2_b0_past_f": round(pf_b0 * 100, 2),
            "c2_b1_past_f": round(pf_b1 * 100, 2),
            "delta_past_f_pp": round((pf_b1 - pf_b0) * 100, 2),
        })
    pd.DataFrame(fgt_rows).to_csv(output_dir / "c2_dtd_task_forgetting.csv", index=False)

    # 5. Plot
    plt.figure(figsize=(10, 6))
    stages = list(range(1, num_tasks + 1))
    curve_b0 = [a * 100 for a in b0_rec["stage_accuracies"][:num_tasks]]
    curve_b1 = [a * 100 for a in b1_rec["stage_accuracies"][:num_tasks]]
    plt.plot(stages, curve_b0, marker="o", color="#0275d8", linewidth=2.2, label=f"B0: augreg_in1k (A_avg={b0_rec['average_incremental_accuracy']*100:.2f}%)")
    plt.plot(stages, curve_b1, marker="s", color="#5cb85c", linewidth=2.2, label=f"B1: augreg2_in21k_ft (A_avg={b1_rec['average_incremental_accuracy']*100:.2f}%)")
    if c1_bundle is not None:
        curve_c1 = [a * 100 for a in c1_bundle["recalculated"]["stage_accuracies"][:num_tasks]]
        plt.plot(stages, curve_c1, marker="x", color="#f0ad4e", linestyle=":", linewidth=1.8, label="C1-P1 Reference")

    plt.title("DTD Incremental Accuracy across Stages (C2 Controlled Experiment)", fontsize=13, fontweight="bold")
    plt.xlabel("Incremental Task / Stage", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(stages)
    vals = curve_b0 + curve_b1
    if vals:
        plt.ylim(max(0, min(vals) - 5), min(102, max(vals) + 5))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    chart_path = output_dir / "c2_dtd_stage_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print("--- C2 DTD COMPREHENSIVE COMPARISON TABLE ---")
    print(pd.DataFrame(conditions).to_string(index=False))
    print(f"\nSaved C2 analysis artifacts to {output_dir}")
    return json_payload


def main():
    p = argparse.ArgumentParser(description="Generate C2 DTD Analysis Tables and Plots")
    p.add_argument("--results-dir", type=Path, default=Path("results/dtd_c2_full"))
    p.add_argument("--output-dir", type=Path, default=Path("results/dtd_c2_analysis"))
    p.add_argument("--c1-p1-dir", type=Path, default=Path("results/dtd_c1_full/p1_official_uniform"))
    p.add_argument("--diag-file", type=Path, default=Path("results/dtd_c2_diagnostics/c2_dtd_matched_backbone_diagnostics.json"))
    args = p.parse_args()
    run_analysis(args.results_dir, args.output_dir, args.c1_p1_dir, args.diag_file)


if __name__ == "__main__":
    main()
