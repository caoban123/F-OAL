"""Generate C3 DTD Analysis Artifacts, Tables, and Figures.

Evaluates the 2x2 factorial preprocessing design on DTD:
  - C3-A: Bicubic, Normalized [-1, 1]
  - C3-B: Bilinear, Normalized [-1, 1]
  - C3-C: Bicubic, Raw [0, 1]
  - C3-D: Bilinear, Raw [0, 1] (Official Alignment)

Computes the 2x2 factorial contrasts:
  - Main effect of Interpolation (Bilinear vs Bicubic)
  - Main effect of Normalization (Raw vs Normalized)
  - Interaction effect
  - Total alignment effect (C3-D vs C3-A)
Emits comparison tables, CSVs, JSON summaries, and plots.

Usage:
    python -m scripts.generate_c3_dtd_analysis --results-dir results/dtd_c3_full --output-dir results/dtd_c3_analysis
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

CONDITIONS = [
    ("C3-A", "c3_a_bicubic_normalized", "bicubic", "normalized [-1, 1]"),
    ("C3-B", "c3_b_bilinear_normalized", "bilinear", "normalized [-1, 1]"),
    ("C3-C", "c3_c_bicubic_raw", "bicubic", "raw [0, 1]"),
    ("C3-D", "c3_d_bilinear_raw", "bilinear", "raw [0, 1]"),
]


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
    c2_b1_dir: Path | None = None,
    table4_a0_dir: Path | None = None,
    diag_file: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, Any] = {}
    for tag, folder, interp, norm in CONDITIONS:
        bundle = load_bundle(results_dir / folder)
        if bundle is None:
            # Check if C3-A can be loaded from c2_b1_dir as fallback
            if tag == "C3-A" and c2_b1_dir and (c2_b1_dir / "metrics.json").exists():
                bundle = load_bundle(c2_b1_dir)
            # Check if C3-D can be loaded from table4_a0_dir as fallback
            elif tag == "C3-D" and table4_a0_dir and (table4_a0_dir / "metrics.json").exists():
                bundle = load_bundle(table4_a0_dir)
        if bundle is None:
            raise FileNotFoundError(f"Missing run results for {tag} ({folder}) in {results_dir}")
        runs[tag] = bundle

    diag_data = {}
    if diag_file and diag_file.exists():
        diag_data = json.loads(diag_file.read_text(encoding="utf-8"))

    num_tasks = min(len(runs[tag]["recalculated"]["stage_accuracies"]) for tag, *_ in CONDITIONS)

    # 1. c3_dtd_stage_accuracies.csv
    stage_rows = []
    for s in range(num_tasks):
        row = {"stage": s + 1}
        for tag, folder, *_ in CONDITIONS:
            row[f"{tag.lower()}_{folder[5:]}"] = round(runs[tag]["recalculated"]["stage_accuracies"][s] * 100, 2)
        stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(output_dir / "c3_dtd_stage_accuracies.csv", index=False)

    # 2. c3_dtd_final_task_accuracies.csv
    final_rows = []
    for t in range(num_tasks):
        row = {"task": t + 1}
        for tag, folder, *_ in CONDITIONS:
            row[f"{tag.lower()}_{folder[5:]}"] = round(runs[tag]["metrics"]["task_matrix"][num_tasks - 1][t] * 100, 2)
        final_rows.append(row)
    pd.DataFrame(final_rows).to_csv(output_dir / "c3_dtd_final_task_accuracies.csv", index=False)

    # 3. c3_dtd_task_forgetting.csv
    fgt_rows = []
    for t in range(max(1, num_tasks - 1)):
        row = {"task": t + 1}
        for tag, folder, *_ in CONDITIONS:
            vals = runs[tag]["recalculated"]["task_forgetting_past_tasks"]
            row[f"{tag.lower()}_past"] = round(vals[t] * 100, 2) if t < len(vals) else 0.0
            cvals = runs[tag]["recalculated"]["task_forgetting_official_code"]
            row[f"{tag.lower()}_code"] = round(cvals[t] * 100, 2) if t < len(cvals) else 0.0
        fgt_rows.append(row)
    pd.DataFrame(fgt_rows).to_csv(output_dir / "c3_dtd_task_forgetting.csv", index=False)

    # 4. Factorial comparison table
    fact_rows = []
    for tag, folder, interp, norm in CONDITIONS:
        bundle = runs[tag]
        rec = bundle["recalculated"]
        m = bundle["metrics"]
        fact_rows.append({
            "condition": tag,
            "name": folder,
            "interpolation": interp,
            "normalization": norm,
            "avg_incremental_acc_pct": round(rec["average_incremental_accuracy"] * 100, 4),
            "last_acc_pct": round(rec["last_accuracy"] * 100, 4),
            "forgetting_past_pct": round(rec["forgetting_past_tasks"] * 100, 4),
            "forgetting_code_pct": round(rec["forgetting_official_code"] * 100, 4),
            "feat_sec": round(m.get("feature_extraction_seconds", 0.0), 2),
            "rls_sec": round(m.get("rls_and_eval_seconds", 0.0), 2),
            "total_sec": round(m.get("total_duration_seconds", 0.0), 2),
        })
    pd.DataFrame(fact_rows).to_csv(output_dir / "c3_dtd_factorial_comparison.csv", index=False)

    # 5. Factorial effects calculation
    a_acc = runs["C3-A"]["recalculated"]["average_incremental_accuracy"]
    b_acc = runs["C3-B"]["recalculated"]["average_incremental_accuracy"]
    c_acc = runs["C3-C"]["recalculated"]["average_incremental_accuracy"]
    d_acc = runs["C3-D"]["recalculated"]["average_incremental_accuracy"]

    delta_interp_norm = (b_acc - a_acc) * 100
    delta_interp_raw = (d_acc - c_acc) * 100
    main_interp = 0.5 * (delta_interp_norm + delta_interp_raw)

    delta_norm_bicubic = (c_acc - a_acc) * 100
    delta_norm_bilinear = (d_acc - b_acc) * 100
    main_norm = 0.5 * (delta_norm_bicubic + delta_norm_bilinear)

    interaction = delta_interp_raw - delta_interp_norm
    total_effect = (d_acc - a_acc) * 100

    factorial_summary = {
        "dataset": "tanganke/dtd",
        "num_tasks": num_tasks,
        "accuracies_pct": {
            "C3-A (Bicubic, Normalized)": round(a_acc * 100, 4),
            "C3-B (Bilinear, Normalized)": round(b_acc * 100, 4),
            "C3-C (Bicubic, Raw)": round(c_acc * 100, 4),
            "C3-D (Bilinear, Raw)": round(d_acc * 100, 4),
        },
        "effects_pp": {
            "main_effect_interpolation_bilinear_minus_bicubic": round(main_interp, 4),
            "main_effect_normalization_raw_minus_normalized": round(main_norm, 4),
            "interaction_effect": round(interaction, 4),
            "total_effect_c3_d_minus_c3_a": round(total_effect, 4),
            "interp_effect_in_normalized": round(delta_interp_norm, 4),
            "interp_effect_in_raw": round(delta_interp_raw, 4),
            "norm_effect_in_bicubic": round(delta_norm_bicubic, 4),
            "norm_effect_in_bilinear": round(delta_norm_bilinear, 4),
        },
    }
    (output_dir / "c3_dtd_factorial_summary.json").write_text(json.dumps(factorial_summary, indent=2), encoding="utf-8")

    # 6. Full comparison table & JSON
    full_table = []
    for r in fact_rows:
        full_table.append({
            "condition": r["condition"],
            "name": r["name"],
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "interpolation": r["interpolation"],
            "normalization": r["normalization"],
            "avg_incremental_acc_pct": r["avg_incremental_acc_pct"],
            "last_acc_pct": r["last_acc_pct"],
            "forgetting_past_pct": r["forgetting_past_pct"],
            "forgetting_code_pct": r["forgetting_code_pct"],
        })
    pd.DataFrame(full_table).to_csv(output_dir / "c3_dtd_full_comparison.csv", index=False)
    (output_dir / "c3_dtd_full_comparison.json").write_text(json.dumps(full_table, indent=2), encoding="utf-8")

    # 7. Plot
    plt.figure(figsize=(10, 6))
    stages = list(range(1, num_tasks + 1))
    colors = {"C3-A": "#0275d8", "C3-B": "#5bc0de", "C3-C": "#f0ad4e", "C3-D": "#5cb85c"}
    markers = {"C3-A": "o", "C3-B": "s", "C3-C": "^", "C3-D": "D"}
    all_vals = []
    for tag, folder, interp, norm in CONDITIONS:
        rec = runs[tag]["recalculated"]
        curve = [a * 100 for a in rec["stage_accuracies"][:num_tasks]]
        all_vals.extend(curve)
        lbl = f"{tag}: {interp}+{norm} (A_avg={rec['average_incremental_accuracy']*100:.2f}%)"
        plt.plot(stages, curve, marker=markers[tag], color=colors[tag], linewidth=2.0, label=lbl)

    plt.title("DTD Incremental Accuracy across Stages (C3 2x2 Factorial)", fontsize=13, fontweight="bold")
    plt.xlabel("Incremental Task / Stage", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(stages)
    if all_vals:
        plt.ylim(max(0, min(all_vals) - 5), min(102, max(all_vals) + 5))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    chart_path = output_dir / "c3_dtd_stage_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print("--- C3 DTD FACTORIAL SUMMARY ---")
    print(json.dumps(factorial_summary["effects_pp"], indent=2))
    print(f"\nSaved C3 analysis artifacts to {output_dir}")
    return factorial_summary


def main():
    p = argparse.ArgumentParser(description="Generate C3 DTD Analysis Tables and Plots")
    p.add_argument("--results-dir", type=Path, default=Path("results/dtd_c3_full"))
    p.add_argument("--output-dir", type=Path, default=Path("results/dtd_c3_analysis"))
    p.add_argument("--c2-b1-dir", type=Path, default=Path("results/dtd_c2_full/b1_augreg2_in21k_ft_in1k"))
    p.add_argument("--table4-a0-dir", type=Path, default=Path("results/dtd/dtd_a0_full"))
    p.add_argument("--diag-file", type=Path, default=Path("results/dtd_c3_analysis/c3_dtd_matched_diagnostics.json"))
    args = p.parse_args()
    run_analysis(args.results_dir, args.output_dir, args.c2_b1_dir, args.table4_a0_dir, args.diag_file)


if __name__ == "__main__":
    main()
