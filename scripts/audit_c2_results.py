"""
Comprehensive Scientific Audit of C2 GPU Experiments.
"""
import csv
import json
import math
from pathlib import Path
import sys

def parse_accuracy_matrix(csv_path: Path):
    matrix = [[] for _ in range(10)]
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            s = int(r['stage']) - 1
            t = int(r['task']) - 1
            while len(matrix[s]) <= t:
                matrix[s].append(0.0)
            matrix[s][t] = float(r['accuracy'])
    return matrix

def recalculate_metrics(matrix: list[list[float]]):
    num_tasks = len(matrix)
    stage_accuracies = []
    for t in range(num_tasks):
        row = matrix[t][:t+1]
        stage_accuracies.append(sum(row) / len(row))

    avg_incremental_acc = sum(stage_accuracies) / num_tasks
    last_acc = stage_accuracies[-1]

    # Past-task Forgetting (paper style, 9 tasks)
    past_forgetting_terms = []
    for j in range(num_tasks - 1):
        best_past = max(matrix[k][j] for k in range(j, num_tasks - 1))
        current = matrix[num_tasks - 1][j]
        past_forgetting_terms.append(best_past - current)
    f_past = sum(past_forgetting_terms) / len(past_forgetting_terms) if past_forgetting_terms else 0.0

    # Official-code Forgetting (10 tasks)
    code_forgetting_terms = []
    for j in range(num_tasks):
        best = max(matrix[k][j] for k in range(j, num_tasks))
        current = matrix[num_tasks - 1][j]
        code_forgetting_terms.append(best - current)
    f_code = sum(code_forgetting_terms) / len(code_forgetting_terms)

    return {
        "num_tasks": num_tasks,
        "matrix_entries_count": sum(len(matrix[t][:t+1]) for t in range(num_tasks)),
        "stage_accuracies": stage_accuracies,
        "average_incremental_accuracy": avg_incremental_acc,
        "last_accuracy": last_acc,
        "forgetting_past_tasks": f_past,
        "forgetting_official_code": f_code,
        "per_task_final_accuracy": [matrix[-1][j] for j in range(num_tasks)],
        "per_task_past_forgetting": past_forgetting_terms,
        "per_task_code_forgetting": code_forgetting_terms,
    }

def main():
    base_dir = Path("results/c2_full_gpu_verified/results")
    full_dir = base_dir / "c2_cifar100_full"
    diag_file = base_dir / "c2_diagnostics/c2_matched_backbone_diagnostics.json"

    b0_dir = full_dir / "b0_augreg_in1k"
    b1_dir = full_dir / "b1_augreg2_in21k_ft_in1k"

    print("=" * 80)
    print("C2 EXPERIMENT VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Load Metadatas
    with open(b0_dir / "metadata.json", "r", encoding="utf-8") as f:
        b0_meta = json.load(f)
    with open(b1_dir / "metadata.json", "r", encoding="utf-8") as f:
        b1_meta = json.load(f)

    # 2. Load and Recalculate Metrics
    b0_mat = parse_accuracy_matrix(b0_dir / "task_accuracy.csv")
    b1_mat = parse_accuracy_matrix(b1_dir / "task_accuracy.csv")

    b0_recalc = recalculate_metrics(b0_mat)
    b1_recalc = recalculate_metrics(b1_mat)

    with open(b0_dir / "metrics.json", "r", encoding="utf-8") as f:
        b0_exported = json.load(f)
    with open(b1_dir / "metrics.json", "r", encoding="utf-8") as f:
        b1_exported = json.load(f)

    # Audit Results
    audit = {
        "b0": {
            "metadata": b0_meta,
            "recalculated": b0_recalc,
            "exported": b0_exported,
            "matrix": b0_mat,
        },
        "b1": {
            "metadata": b1_meta,
            "recalculated": b1_recalc,
            "exported": b1_exported,
            "matrix": b1_mat,
        },
    }

    if diag_file.exists():
        with open(diag_file, "r", encoding="utf-8") as f:
            audit["diagnostics"] = json.load(f)

    out_audit = Path("results/c2_audit_parsed.json")
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"B0 Avg Incr Acc : {b0_recalc['average_incremental_accuracy']*100:.4f}% (Exported: {b0_exported['average_incremental_accuracy']*100:.4f}%)")
    print(f"B0 Last Acc     : {b0_recalc['last_accuracy']*100:.4f}% (Exported: {b0_exported['last_accuracy']*100:.4f}%)")
    print(f"B0 Forgetting P : {b0_recalc['forgetting_past_tasks']*100:.4f}% (Exported: {b0_exported.get('forgetting_past_tasks', b0_exported.get('forgetting'))*100:.4f}%)")
    print(f"B0 Forgetting C : {b0_recalc['forgetting_official_code']*100:.4f}% (Exported: {b0_exported.get('forgetting_official_code', 0.0)*100:.4f}%)")
    print("-" * 80)
    print(f"B1 Avg Incr Acc : {b1_recalc['average_incremental_accuracy']*100:.4f}% (Exported: {b1_exported['average_incremental_accuracy']*100:.4f}%)")
    print(f"B1 Last Acc     : {b1_recalc['last_accuracy']*100:.4f}% (Exported: {b1_exported['last_accuracy']*100:.4f}%)")
    print(f"B1 Forgetting P : {b1_recalc['forgetting_past_tasks']*100:.4f}% (Exported: {b1_exported.get('forgetting_past_tasks', b1_exported.get('forgetting'))*100:.4f}%)")
    print(f"B1 Forgetting C : {b1_recalc['forgetting_official_code']*100:.4f}% (Exported: {b1_exported.get('forgetting_official_code', 0.0)*100:.4f}%)")
    print("-" * 80)
    delta_avg = (b1_recalc['average_incremental_accuracy'] - b0_recalc['average_incremental_accuracy']) * 100
    delta_last = (b1_recalc['last_accuracy'] - b0_recalc['last_accuracy']) * 100
    delta_f_past = (b1_recalc['forgetting_past_tasks'] - b0_recalc['forgetting_past_tasks']) * 100
    delta_f_code = (b1_recalc['forgetting_official_code'] - b0_recalc['forgetting_official_code']) * 100
    print(f"Delta (B1 - B0) Avg Incr Acc : {delta_avg:+.4f} pp")
    print(f"Delta (B1 - B0) Last Acc     : {delta_last:+.4f} pp")
    print(f"Delta (B1 - B0) Forgetting P : {delta_f_past:+.4f} pp")
    print(f"Delta (B1 - B0) Forgetting C : {delta_f_code:+.4f} pp")
    print("=" * 80)

if __name__ == "__main__":
    main()
