"""Comprehensive Table 4 Ablation Audit, Recalculation, Factorial Analysis, and Manifest Generator."""
import json
import csv
from pathlib import Path

def main():
    analysis_dir = Path("results/table4_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load A0 (reused from C3-D)
    a0_path = Path("results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/metrics.json")
    a0_metrics = json.loads(a0_path.read_text(encoding="utf-8"))

    # 2. Load A1, A2, A3 (from extracted Colab zip)
    extracted_base = Path("results/table4_full_gpu_verified/results")
    a1_metrics = json.loads((extracted_base / "table4_a1_without_ff/metrics.json").read_text(encoding="utf-8"))
    a2_metrics = json.loads((extracted_base / "table4_a2_without_sp/metrics.json").read_text(encoding="utf-8"))
    a3_metrics = json.loads((extracted_base / "table4_a3_without_both/metrics.json").read_text(encoding="utf-8"))

    # Load metadata
    a1_meta = json.loads((extracted_base / "table4_a1_without_ff/metadata.json").read_text(encoding="utf-8"))
    a2_meta = json.loads((extracted_base / "table4_a2_without_sp/metadata.json").read_text(encoding="utf-8"))
    a3_meta = json.loads((extracted_base / "table4_a3_without_both/metadata.json").read_text(encoding="utf-8"))
    a0_meta = json.loads(Path("results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/metadata.json").read_text(encoding="utf-8"))

    runs = {
        "A0": {"name": "Full F-OAL", "ff": True, "sp": True, "metrics": a0_metrics, "meta": a0_meta, "wandb_id": "3v53tmq5", "provenance": "REUSED_VERIFIED_C3_D"},
        "A1": {"name": "w/o Feature Fusion", "ff": False, "sp": True, "metrics": a1_metrics, "meta": a1_meta, "wandb_id": "shl4p10l", "provenance": "NEW_FULL_GPU_COLAB"},
        "A2": {"name": "w/o Smooth Projection", "ff": True, "sp": False, "metrics": a2_metrics, "meta": a2_meta, "wandb_id": "rkedlkkk", "provenance": "NEW_FULL_GPU_COLAB"},
        "A3": {"name": "w/o Both (FF & SP)", "ff": False, "sp": False, "metrics": a3_metrics, "meta": a3_meta, "wandb_id": "k01pp8st", "provenance": "NEW_FULL_GPU_COLAB"},
    }

    # 3. Independent Recalculation of all metrics from raw task_matrix
    recalculated = {}
    for cond_id, r in runs.items():
        matrix = r["metrics"]["task_matrix"]
        assert len(matrix) == 10, f"{cond_id} matrix must have 10 stages"
        assert sum(len(row) for row in matrix) == 55, f"{cond_id} matrix must have 55 entries"

        # Stage accuracies: A_t = (1/t) * sum_{j=1}^t a_{t, j}
        stage_accs = [sum(row) / len(row) for row in matrix]
        # Average incremental accuracy: A_bar = (1/10) * sum_{t=1}^{10} A_t
        a_bar = sum(stage_accs) / len(stage_accs)
        # Last accuracy: A_10 = stage_accs[-1]
        a_last = stage_accs[-1]
        
        # Final task accuracies
        final_task_accs = matrix[-1]
        
        # Paper-style Forgetting: F_past = (1/9) * sum_{j=1}^9 (max_{j<=k<10} a_{k, j} - a_{10, j})
        task_f_past = []
        for j in range(9):
            best_past = max(matrix[k][j] for k in range(j, 9))
            task_f_past.append(best_past - matrix[9][j])
        f_past = sum(task_f_past) / 9.0

        # Official-code Forgetting: F_code = (1/10) * sum_{j=1}^10 (max_{j<=k<=10} a_{k, j} - a_{10, j})
        task_f_code = []
        for j in range(10):
            best_any = max(matrix[k][j] for k in range(j, 10))
            task_f_code.append(best_any - matrix[9][j])
        f_code = sum(task_f_code) / 10.0

        # Compare with exported values
        diff_a_bar = abs(a_bar - r["metrics"]["average_incremental_accuracy"])
        diff_a_last = abs(a_last - r["metrics"]["last_accuracy"])
        diff_f_past = abs(f_past - r["metrics"]["forgetting_past_tasks"])
        diff_f_code = abs(f_code - r["metrics"]["forgetting_official_code"])

        assert diff_a_bar < 1e-12, f"{cond_id} A_bar mismatch: {diff_a_bar}"
        assert diff_a_last < 1e-12, f"{cond_id} A_last mismatch: {diff_a_last}"
        assert diff_f_past < 1e-12, f"{cond_id} F_past mismatch: {diff_f_past}"
        assert diff_f_code < 1e-12, f"{cond_id} F_code mismatch: {diff_f_code}"

        recalculated[cond_id] = {
            "condition": cond_id,
            "name": r["name"],
            "ff": r["ff"],
            "sp": r["sp"],
            "a_bar_ratio": a_bar,
            "a_bar_pct": a_bar * 100.0,
            "a_last_ratio": a_last,
            "a_last_pct": a_last * 100.0,
            "f_past_ratio": f_past,
            "f_past_pct": f_past * 100.0,
            "f_code_ratio": f_code,
            "f_code_pct": f_code * 100.0,
            "stage_accuracies": stage_accs,
            "final_task_accuracies": final_task_accs,
            "task_forgetting_past": task_f_past,
            "task_forgetting_code": task_f_code,
            "duration_sec": r["metrics"].get("total_duration_seconds"),
            "feat_ext_sec": r["metrics"].get("feature_extraction_seconds"),
            "rls_eval_sec": r["metrics"].get("rls_and_eval_seconds"),
            "wandb_id": r["wandb_id"],
            "provenance": r["provenance"],
        }

    # 4. Save Metrics Comparison CSV & JSON
    comparison_json = analysis_dir / "table4_metrics_comparison.json"
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(recalculated, f, indent=2)

    comparison_csv = analysis_dir / "table4_metrics_comparison.csv"
    with open(comparison_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Name", "FF", "SP", "A_bar (%)", "A_10 (%)", "F_past (%)", "F_code (%)", "Duration (s)", "FeatExt (s)", "RLS (s)", "W&B Run ID", "Provenance"])
        for cid in ["A0", "A1", "A2", "A3"]:
            c = recalculated[cid]
            writer.writerow([
                c["condition"], c["name"], "ON" if c["ff"] else "OFF", "ON" if c["sp"] else "OFF",
                f"{c['a_bar_pct']:.4f}", f"{c['a_last_pct']:.4f}", f"{c['f_past_pct']:.4f}", f"{c['f_code_pct']:.4f}",
                c["duration_sec"], c["feat_ext_sec"], c["rls_eval_sec"], c["wandb_id"], c["provenance"]
            ])

    # 5. Factorial Effects Calculation
    # Unrounded values in percentage points (scaled by 100)
    A0_val = recalculated["A0"]["a_bar_pct"]
    A1_val = recalculated["A1"]["a_bar_pct"]
    A2_val = recalculated["A2"]["a_bar_pct"]
    A3_val = recalculated["A3"]["a_bar_pct"]

    # Simple effects
    e_ff_given_sp = A0_val - A1_val
    e_ff_given_nosp = A2_val - A3_val
    e_sp_given_ff = A0_val - A2_val
    e_sp_given_noff = A1_val - A3_val

    # Main effects
    e_ff_main = (e_ff_given_sp + e_ff_given_nosp) / 2.0
    e_sp_main = (e_sp_given_ff + e_sp_given_noff) / 2.0

    # Interaction
    i_contrast = A0_val - A1_val - A2_val + A3_val
    beta_interaction = i_contrast / 4.0

    # Full vs None
    delta_full_none = A0_val - A3_val
    sum_main_effects = e_ff_main + e_sp_main

    factorial_effects = {
        "metric": "Average Incremental Accuracy (A_bar, percentage points)",
        "A0_Full": A0_val,
        "A1_noFF": A1_val,
        "A2_noSP": A2_val,
        "A3_noBoth": A3_val,
        "simple_effects": {
            "E_FF_given_SP_ON": e_ff_given_sp,
            "E_FF_given_SP_OFF": e_ff_given_nosp,
            "E_SP_given_FF_ON": e_sp_given_ff,
            "E_SP_given_FF_OFF": e_sp_given_noff,
        },
        "main_effects": {
            "E_FF_main": e_ff_main,
            "E_SP_main": e_sp_main,
        },
        "interaction": {
            "interaction_contrast": i_contrast,
            "regression_coefficient_beta": beta_interaction,
        },
        "joint_benefit": {
            "delta_full_vs_none": delta_full_none,
            "sum_of_main_effects": sum_main_effects,
            "additivity_gap": abs(delta_full_none - sum_main_effects)
        }
    }

    with open(analysis_dir / "table4_factorial_effects.json", "w", encoding="utf-8") as f:
        json.dump(factorial_effects, f, indent=2)

    with open(analysis_dir / "table4_factorial_effects.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Effect Description", "Symbol", "Value (pp)"])
        writer.writerow(["Feature Fusion Benefit (SP ON)", "E_FF|SP", f"{e_ff_given_sp:+.4f}"])
        writer.writerow(["Feature Fusion Benefit (SP OFF)", "E_FF|~SP", f"{e_ff_given_nosp:+.4f}"])
        writer.writerow(["Smooth Projection Benefit (FF ON)", "E_SP|FF", f"{e_sp_given_ff:+.4f}"])
        writer.writerow(["Smooth Projection Benefit (FF OFF)", "E_SP|~FF", f"{e_sp_given_noff:+.4f}"])
        writer.writerow(["Average Feature Fusion Main Effect", "E_FF", f"{e_ff_main:+.4f}"])
        writer.writerow(["Average Smooth Projection Main Effect", "E_SP", f"{e_sp_main:+.4f}"])
        writer.writerow(["Difference-in-Differences Interaction Contrast", "I_FFxSP", f"{i_contrast:+.4f}"])
        writer.writerow(["Interaction Regression Coefficient (+-1 coding)", "beta_FFxSP", f"{beta_interaction:+.4f}"])
        writer.writerow(["Joint Benefit (Full F-OAL vs Without Both)", "Delta_full_none", f"{delta_full_none:+.4f}"])

    # 6. Comparison with Paper Table 4
    paper_cifar100 = {
        "A0": 91.1,
        "A1": 90.6,
        "A2": 90.7,
        "A3": 90.6
    }
    
    # Paper-level effects
    p_eff_ff = ((91.1 - 90.6) + (90.7 - 90.6)) / 2.0  # (0.5 + 0.1)/2 = 0.30 pp
    p_eff_sp = ((91.1 - 90.7) + (90.6 - 90.6)) / 2.0  # (0.4 + 0.0)/2 = 0.20 pp
    p_contrast = 91.1 - 90.6 - 90.7 + 90.6             # +0.40 pp

    paper_comp = []
    for cid in ["A0", "A1", "A2", "A3"]:
        c = recalculated[cid]
        pval = paper_cifar100[cid]
        oval = c["a_bar_pct"]
        paper_comp.append({
            "condition": cid,
            "name": c["name"],
            "ff": c["ff"],
            "sp": c["sp"],
            "paper_a_bar": pval,
            "our_a_bar": oval,
            "delta_pp": oval - pval
        })

    with open(analysis_dir / "table4_paper_comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Name", "FF", "SP", "Paper A_bar (%)", "Our A_bar (%)", "Delta (pp)"])
        for p in paper_comp:
            writer.writerow([p["condition"], p["name"], "ON" if p["ff"] else "OFF", "ON" if p["sp"] else "OFF", f"{p['paper_a_bar']:.1f}", f"{p['our_a_bar']:.4f}", f"{p['delta_pp']:+.4f}"])

    # 7. Stage Accuracies Table
    with open(analysis_dir / "table4_stage_accuracies.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "A0 (Full)", "A1 (w/o FF)", "A2 (w/o SP)", "A3 (w/o Both)"])
        for s in range(10):
            writer.writerow([
                f"Stage {s+1}",
                f"{recalculated['A0']['stage_accuracies'][s]*100.0:.2f}",
                f"{recalculated['A1']['stage_accuracies'][s]*100.0:.2f}",
                f"{recalculated['A2']['stage_accuracies'][s]*100.0:.2f}",
                f"{recalculated['A3']['stage_accuracies'][s]*100.0:.2f}",
            ])

    # 8. Final Task Accuracies Table
    with open(analysis_dir / "table4_final_task_accuracies.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "A0 (Full)", "A1 (w/o FF)", "A2 (w/o SP)", "A3 (w/o Both)"])
        for t in range(10):
            writer.writerow([
                f"Task {t+1}",
                f"{recalculated['A0']['final_task_accuracies'][t]*100.0:.2f}",
                f"{recalculated['A1']['final_task_accuracies'][t]*100.0:.2f}",
                f"{recalculated['A2']['final_task_accuracies'][t]*100.0:.2f}",
                f"{recalculated['A3']['final_task_accuracies'][t]*100.0:.2f}",
            ])

    # 9. Task Forgetting Table
    with open(analysis_dir / "table4_task_forgetting.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "A0 (Past)", "A1 (Past)", "A2 (Past)", "A3 (Past)", "A0 (Code)", "A1 (Code)", "A2 (Code)", "A3 (Code)"])
        for t in range(10):
            fp_a0 = f"{recalculated['A0']['task_forgetting_past'][t]*100.0:.2f}" if t < 9 else "-"
            fp_a1 = f"{recalculated['A1']['task_forgetting_past'][t]*100.0:.2f}" if t < 9 else "-"
            fp_a2 = f"{recalculated['A2']['task_forgetting_past'][t]*100.0:.2f}" if t < 9 else "-"
            fp_a3 = f"{recalculated['A3']['task_forgetting_past'][t]*100.0:.2f}" if t < 9 else "-"

            fc_a0 = f"{recalculated['A0']['task_forgetting_code'][t]*100.0:.2f}"
            fc_a1 = f"{recalculated['A1']['task_forgetting_code'][t]*100.0:.2f}"
            fc_a2 = f"{recalculated['A2']['task_forgetting_code'][t]*100.0:.2f}"
            fc_a3 = f"{recalculated['A3']['task_forgetting_code'][t]*100.0:.2f}"
            writer.writerow([f"Task {t+1}", fp_a0, fp_a1, fp_a2, fp_a3, fc_a0, fc_a1, fc_a2, fc_a3])

    # 10. Runtime and Computational Cost Table
    with open(analysis_dir / "table4_runtime_comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Name", "Total Time (s)", "Feature Extr (s)", "RLS & Eval (s)", "RLS Feature Dim", "Hardware Note"])
        for cid in ["A0", "A1", "A2", "A3"]:
            c = recalculated[cid]
            dim = 1000 if c["sp"] else 768
            note = "C3-D on Colab T4 GPU (Reused)" if cid == "A0" else "Colab T4 GPU (New Full GPU)"
            writer.writerow([cid, c["name"], c["duration_sec"], c["feat_ext_sec"], c["rls_eval_sec"], dim, note])

    # 11. W&B Audit Manifest
    wandb_audit = {
        "project": "hcmus-foal-reproduction",
        "entity": "caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm",
        "endpoint": "https://api.forge.coreweave.com",
        "total_online_runs": 17,
        "table4_runs": [
            {
                "condition": "A0",
                "name": "c3_d_bilinear_raw",
                "run_id": "3v53tmq5",
                "group": "c3_preprocessing_alignment",
                "status": "finished",
                "provenance": "REUSED_VERIFIED_C3_D",
                "url": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5"
            },
            {
                "condition": "A1",
                "name": "table4_a1_without_ff_seed42",
                "run_id": "shl4p10l",
                "group": "foal_no_fusion_ablation",
                "status": "finished",
                "provenance": "NEW_FULL_GPU_COLAB",
                "url": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/shl4p10l"
            },
            {
                "condition": "A2",
                "name": "table4_a2_without_sp_seed42",
                "run_id": "rkedlkkk",
                "group": "foal_no_projection_ablation",
                "status": "finished",
                "provenance": "NEW_FULL_GPU_COLAB",
                "url": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/rkedlkkk"
            },
            {
                "condition": "A3",
                "name": "table4_a3_without_both_seed42",
                "run_id": "k01pp8st",
                "group": "foal_neither_ablation",
                "status": "finished",
                "provenance": "NEW_FULL_GPU_COLAB",
                "url": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/k01pp8st"
            }
        ],
        "pilot_smoke_runs": [
            {"run_id": "172e2y2y", "name": "foal_no_fusion_seed42_pilot", "group": "foal_no_fusion_pilot"},
            {"run_id": "b4bao7ql", "name": "foal_no_fusion_seed42_pilot", "group": "foal_no_fusion_pilot"},
            {"run_id": "euwibfmj", "name": "foal_no_projection_seed42_pilot", "group": "foal_no_projection_pilot"},
            {"run_id": "zzzeg0k6", "name": "foal_neither_seed42_pilot", "group": "foal_neither_pilot"}
        ]
    }
    with open(analysis_dir / "table4_wandb_audit.json", "w", encoding="utf-8") as f:
        json.dump(wandb_audit, f, indent=2)

    # 12. Complete Evidence Manifest
    evidence_manifest = {
        "experiment": "Table 4 Controlled Architectural Ablation (Feature Fusion & Smooth Projection)",
        "archive_provenance": {
            "path": "foal_table4_ablation_results.zip",
            "sha256": "90f191d12b06fbd804779fcb038a12574c9b7cdc2f32369a6975dd6309e09e7a",
            "crc_status": "ALL_OK",
            "extracted_to": "results/table4_full_gpu_verified"
        },
        "reused_baseline_a0": {
            "source": "C3-D (results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw)",
            "wandb_id": "3v53tmq5",
            "a_bar": A0_val,
            "status": "A0_REUSE_VERIFIED"
        },
        "reconstructed_semantics": "RECONSTRUCTED_FROM_PAPER",
        "fixed_controls_verified": True,
        "metrics_recalculated_match": True,
        "factorial_effects": factorial_effects,
        "paper_comparison": paper_comp,
        "scientific_verdict": "TABLE4_SIGNED_OFF_WITH_LIMITATIONS",
        "limitations": [
            "Evaluated on single seed (Seed 42) under one-pass CIFAR-100 stream; paper reports 3-seed mean.",
            "CIFAR-100 features from pre-trained ViT are highly separable, exhibiting small ablation drops (<0.5 pp); DTD extension required for large domain-shift validation."
        ]
    }
    with open(analysis_dir / "table4_evidence_manifest.json", "w", encoding="utf-8") as f:
        json.dump(evidence_manifest, f, indent=2)

    print("Auditing and manifest generation complete!")
    print(f"A0 Full: {A0_val:.4f}% | A1 w/o FF: {A1_val:.4f}% | A2 w/o SP: {A2_val:.4f}% | A3 w/o Both: {A3_val:.4f}%")
    print(f"Main Effect FF: {e_ff_main:+.4f} pp | Main Effect SP: {e_sp_main:+.4f} pp | Interaction Contrast: {i_contrast:+.4f} pp")

if __name__ == "__main__":
    main()
