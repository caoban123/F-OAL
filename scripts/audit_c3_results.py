import json
import csv
from pathlib import Path
import numpy as np

def audit_matrix(matrix):
    assert len(matrix) == 10, f"Expected 10 tasks, got {len(matrix)}"
    entries_count = 0
    for t_idx, row in enumerate(matrix):
        assert len(row) == t_idx + 1, f"Row {t_idx} should have {t_idx + 1} elements, got {len(row)}"
        for val in row:
            assert 0.0 <= val <= 1.0, f"Accuracy out of bounds [0, 1]: {val}"
            assert not np.isnan(val) and not np.isinf(val), f"Invalid numeric: {val}"
            entries_count += 1
    assert entries_count == 55, f"Expected 55 entries, got {entries_count}"
    
    stage_accuracies = [float(np.mean(row)) for row in matrix]
    avg_incremental_acc = float(np.mean(stage_accuracies))
    last_accuracy = float(stage_accuracies[-1])
    
    # Per-task forgetting
    final_row = matrix[-1]
    per_task_past_forgetting = []
    for j in range(9):
        # max over stages k from j to 8 (i.e. before stage 10)
        peak = max(matrix[k][j] for k in range(j, 9))
        per_task_past_forgetting.append(float(peak - final_row[j]))
        
    forgetting_past_tasks = float(np.mean(per_task_past_forgetting))
    
    # Official code forgetting includes task 10 (which has peak - final = 0)
    per_task_code_forgetting = []
    for j in range(10):
        peak = max(matrix[k][j] for k in range(j, 10))
        per_task_code_forgetting.append(float(peak - final_row[j]))
    forgetting_official_code = float(np.mean(per_task_code_forgetting))
    
    return {
        "entries_count": entries_count,
        "stage_accuracies": stage_accuracies,
        "average_incremental_accuracy": avg_incremental_acc,
        "last_accuracy": last_accuracy,
        "forgetting_past_tasks": forgetting_past_tasks,
        "forgetting_official_code": forgetting_official_code,
        "per_task_final_accuracy": [float(x) for x in final_row],
        "per_task_past_forgetting": per_task_past_forgetting,
        "per_task_code_forgetting": per_task_code_forgetting
    }

def main():
    base_dir = Path("results/c3_full_gpu_verified/results/c3_full_gpu")
    
    # 1. Load C2-B1 as C3-A
    with open("results/c2_audit_parsed.json", "r", encoding="utf-8") as f:
        c2 = json.load(f)
    matrix_a = c2["b1"]["matrix"]
    recalc_a = audit_matrix(matrix_a)
    
    # 2. Load C3-B, C3-C, C3-D
    runs = {
        "C3-A": {
            "name": "c3_a_bicubic_normalized",
            "condition": "Reference (Bicubic, Norm [-1, 1])",
            "provenance": "REUSED_VERIFIED_C2_B1",
            "matrix": matrix_a,
            "wandb_run_id": "ne1p16b5",
            "recalc": recalc_a,
            "exported": c2["b1"]["exported"]
        }
    }
    
    run_configs = {
        "C3-B": ("c3_b_bilinear_normalized", "Factor A (Bilinear, Norm [-1, 1])", "ts2wh4e3"),
        "C3-C": ("c3_c_bicubic_raw", "Factor B (Bicubic, Raw [0, 1])", "4m4uir7h"),
        "C3-D": ("c3_d_bilinear_raw", "Full Alignment (Bilinear, Raw [0, 1])", "3v53tmq5")
    }
    
    for cond_key, (dir_name, cond_desc, wandb_id) in run_configs.items():
        run_dir = base_dir / dir_name
        with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
            exp_metrics = json.load(f)
        with open(run_dir / "metadata.json", "r", encoding="utf-8") as f:
            exp_metadata = json.load(f)
            
        matrix = exp_metrics["task_matrix"]
        recalc = audit_matrix(matrix)
        
        runs[cond_key] = {
            "name": dir_name,
            "condition": cond_desc,
            "provenance": "FRESH_FULL_GPU_RUN",
            "matrix": matrix,
            "wandb_run_id": wandb_id,
            "recalc": recalc,
            "exported": exp_metrics,
            "metadata": exp_metadata
        }
        
    # Verify export vs recalculation
    reconciliation = {}
    for k, run in runs.items():
        recalc = run["recalc"]
        exp = run["exported"]
        diff_avg = abs(recalc["average_incremental_accuracy"] - exp["average_incremental_accuracy"])
        diff_last = abs(recalc["last_accuracy"] - exp["last_accuracy"])
        diff_fpast = abs(recalc["forgetting_past_tasks"] - exp["forgetting_past_tasks"])
        diff_fcode = abs(recalc["forgetting_official_code"] - exp["forgetting_official_code"])
        
        assert diff_avg < 1e-10, f"{k}: avg acc mismatch {diff_avg}"
        assert diff_last < 1e-10, f"{k}: last acc mismatch {diff_last}"
        assert diff_fpast < 1e-10, f"{k}: fpast mismatch {diff_fpast}"
        assert diff_fcode < 1e-10, f"{k}: fcode mismatch {diff_fcode}"
        
        reconciliation[k] = {
            "avg_inc_acc_match": True,
            "last_acc_match": True,
            "forgetting_past_match": True,
            "forgetting_code_match": True,
            "max_abs_diff": max(diff_avg, diff_last, diff_fpast, diff_fcode)
        }
        print(f"[{k}] Reconciliation verified! Max diff = {reconciliation[k]['max_abs_diff']:.2e}")

    # 3. Factorial Analysis
    A_avg = runs["C3-A"]["recalc"]["average_incremental_accuracy"]
    B_avg = runs["C3-B"]["recalc"]["average_incremental_accuracy"]
    C_avg = runs["C3-C"]["recalc"]["average_incremental_accuracy"]
    D_avg = runs["C3-D"]["recalc"]["average_incremental_accuracy"]
    
    A_last = runs["C3-A"]["recalc"]["last_accuracy"]
    B_last = runs["C3-B"]["recalc"]["last_accuracy"]
    C_last = runs["C3-C"]["recalc"]["last_accuracy"]
    D_last = runs["C3-D"]["recalc"]["last_accuracy"]
    
    A_fpast = runs["C3-A"]["recalc"]["forgetting_past_tasks"]
    B_fpast = runs["C3-B"]["recalc"]["forgetting_past_tasks"]
    C_fpast = runs["C3-C"]["recalc"]["forgetting_past_tasks"]
    D_fpast = runs["C3-D"]["recalc"]["forgetting_past_tasks"]

    def compute_effects(a, b, c, d):
        simple_interp_norm = b - a
        simple_interp_raw = d - c
        simple_norm_bicubic = c - a
        simple_norm_bilinear = d - b
        main_interp = (simple_interp_norm + simple_interp_raw) / 2.0
        main_norm = (simple_norm_bicubic + simple_norm_bilinear) / 2.0
        contrast_interact = d - c - b + a
        beta_interact = contrast_interact / 4.0
        delta_total = d - a
        assert abs((main_interp + main_norm) - delta_total) < 1e-12, "Factorial decomposition identity violated!"
        return {
            "simple_interp_in_norm": simple_interp_norm,
            "simple_interp_in_raw": simple_interp_raw,
            "simple_norm_in_bicubic": simple_norm_bicubic,
            "simple_norm_in_bilinear": simple_norm_bilinear,
            "main_interp": main_interp,
            "main_norm": main_norm,
            "contrast_interact": contrast_interact,
            "beta_interact": beta_interact,
            "delta_total": delta_total
        }
        
    factorial_effects = {
        "average_incremental_accuracy": compute_effects(A_avg, B_avg, C_avg, D_avg),
        "last_accuracy": compute_effects(A_last, B_last, C_last, D_last),
        "forgetting_past_tasks": compute_effects(A_fpast, B_fpast, C_fpast, D_fpast)
    }
    
    print("\n--- 2x2 Factorial Analysis Results (Average Incremental Accuracy) ---")
    eff = factorial_effects["average_incremental_accuracy"]
    print(f"C3-A: {A_avg*100:.4f}% | C3-B: {B_avg*100:.4f}% | C3-C: {C_avg*100:.4f}% | C3-D: {D_avg*100:.4f}%")
    print(f"Main effect of Interpolation (E_interp): {eff['main_interp']*100:+.4f} pp")
    print(f"Main effect of Normalization (E_norm):   {eff['main_norm']*100:+.4f} pp")
    print(f"Interaction contrast (E_IN):             {eff['contrast_interact']*100:+.4f} pp")
    print(f"Interaction beta (beta_IN):              {eff['beta_interact']*100:+.4f} pp")
    print(f"Total Preprocessing Shift (Delta_total): {eff['delta_total']*100:+.4f} pp")
    
    out_data = {
        "runs": runs,
        "reconciliation": reconciliation,
        "factorial_effects": factorial_effects
    }
    
    with open("results/c3_audit_parsed.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print("Saved results/c3_audit_parsed.json successfully!")

if __name__ == "__main__":
    main()
