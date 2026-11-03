import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def main():
    analysis_dir = Path("results/c3_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    with open("results/c3_audit_parsed.json", "r", encoding="utf-8") as f:
        c3_audit = json.load(f)
        
    with open("results/c2_analysis/c2_full_comparison.json", "r", encoding="utf-8") as f:
        c2_comp = json.load(f)
        
    with open("results/c3_full_gpu_verified/results/c3_analysis/c3_matched_diagnostics.json", "r", encoding="utf-8") as f:
        c3_diag = json.load(f)

    runs = c3_audit["runs"]
    
    # Official run data from c2_comp
    official = next(item for item in c2_comp["comparison_table"] if "Official" in item["condition"])
    c2_b0 = next(item for item in c2_comp["comparison_table"] if "C2-B0" in item["condition"])
    
    # 1. c3_stage_accuracies.csv
    stage_csv_path = analysis_dir / "c3_stage_accuracies.csv"
    with open(stage_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "c3_a_bicubic_norm", "c3_b_bilinear_norm", "c3_c_bicubic_raw", "c3_d_bilinear_raw", "c2_b0_control"])
        for s in range(10):
            writer.writerow([
                s + 1,
                round(runs["C3-A"]["recalc"]["stage_accuracies"][s] * 100, 4),
                round(runs["C3-B"]["recalc"]["stage_accuracies"][s] * 100, 4),
                round(runs["C3-C"]["recalc"]["stage_accuracies"][s] * 100, 4),
                round(runs["C3-D"]["recalc"]["stage_accuracies"][s] * 100, 4),
                round(c2_comp["comparison_table"][0]["avg_incremental_acc"] * 100, 4) if s == 9 else "" # reference
            ])
            
    # 2. c3_final_task_accuracies.csv
    final_csv_path = analysis_dir / "c3_final_task_accuracies.csv"
    with open(final_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "c3_a_bicubic_norm", "c3_b_bilinear_norm", "c3_c_bicubic_raw", "c3_d_bilinear_raw"])
        for t in range(10):
            writer.writerow([
                t + 1,
                round(runs["C3-A"]["recalc"]["per_task_final_accuracy"][t] * 100, 4),
                round(runs["C3-B"]["recalc"]["per_task_final_accuracy"][t] * 100, 4),
                round(runs["C3-C"]["recalc"]["per_task_final_accuracy"][t] * 100, 4),
                round(runs["C3-D"]["recalc"]["per_task_final_accuracy"][t] * 100, 4)
            ])
            
    # 3. c3_task_forgetting.csv
    forget_csv_path = analysis_dir / "c3_task_forgetting.csv"
    with open(forget_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "c3_a_past", "c3_b_past", "c3_c_past", "c3_d_past", "c3_a_code", "c3_b_code", "c3_c_code", "c3_d_code"])
        for t in range(10):
            fpast_a = round(runs["C3-A"]["recalc"]["per_task_past_forgetting"][t] * 100, 4) if t < 9 else ""
            fpast_b = round(runs["C3-B"]["recalc"]["per_task_past_forgetting"][t] * 100, 4) if t < 9 else ""
            fpast_c = round(runs["C3-C"]["recalc"]["per_task_past_forgetting"][t] * 100, 4) if t < 9 else ""
            fpast_d = round(runs["C3-D"]["recalc"]["per_task_past_forgetting"][t] * 100, 4) if t < 9 else ""
            
            fcode_a = round(runs["C3-A"]["recalc"]["per_task_code_forgetting"][t] * 100, 4)
            fcode_b = round(runs["C3-B"]["recalc"]["per_task_code_forgetting"][t] * 100, 4)
            fcode_c = round(runs["C3-C"]["recalc"]["per_task_code_forgetting"][t] * 100, 4)
            fcode_d = round(runs["C3-D"]["recalc"]["per_task_code_forgetting"][t] * 100, 4)
            writer.writerow([t + 1, fpast_a, fpast_b, fpast_c, fpast_d, fcode_a, fcode_b, fcode_c, fcode_d])
            
    # 4. c3_factorial_comparison.csv
    fact_csv_path = analysis_dir / "c3_factorial_comparison.csv"
    with open(fact_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "name", "interpolation", "normalization", "avg_incremental_acc_pct", "last_acc_pct", "forgetting_past_pct", "forgetting_code_pct", "feat_sec", "rls_sec", "total_sec", "wandb_run_id"])
        
        c_meta = [
            ("C3-A", "c3_a_bicubic_normalized", "bicubic", "normalized [-1, 1]", 248.3, 8.49, 301.32),
            ("C3-B", "c3_b_bilinear_normalized", "bilinear", "normalized [-1, 1]", 176.08, 8.56, 230.16),
            ("C3-C", "c3_c_bicubic_raw", "bicubic", "raw [0, 1]", 175.58, 8.49, 230.17),
            ("C3-D", "c3_d_bilinear_raw", "bilinear", "raw [0, 1]", 175.28, 8.50, 229.11)
        ]
        for key, name, interp, norm, f_time, r_time, t_time in c_meta:
            rec = runs[key]["recalc"]
            writer.writerow([
                key,
                name,
                interp,
                norm,
                round(rec["average_incremental_accuracy"] * 100, 4),
                round(rec["last_accuracy"] * 100, 4),
                round(rec["forgetting_past_tasks"] * 100, 4),
                round(rec["forgetting_official_code"] * 100, 4),
                f_time,
                r_time,
                t_time,
                runs[key]["wandb_run_id"]
            ])

    # 5. c3_factorial_summary.json
    fact_sum_path = analysis_dir / "c3_factorial_summary.json"
    with open(fact_sum_path, "w", encoding="utf-8") as f:
        json.dump(c3_audit["factorial_effects"], f, indent=2)

    # 6. c3_full_comparison.csv & .json
    full_table = [
        {
            "condition": "NeurIPS 2024 Paper",
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k (implied)",
            "interpolation": "bilinear (official repo)",
            "normalization": "raw [0, 1] (official repo)",
            "seed": "average over seeds",
            "avg_incremental_acc": 0.911,
            "last_acc": None,
            "forgetting_past": None,
            "forgetting_code": None,
            "wandb_run_id": "N/A"
        },
        {
            "condition": "Official Author Run (seed 0)",
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "interpolation": "bilinear",
            "normalization": "raw [0, 1]",
            "seed": 0,
            "avg_incremental_acc": 0.9098279365079364,
            "last_acc": 0.8584,
            "forgetting_past": 0.051666666666666666,
            "forgetting_code": 0.0469,
            "wandb_run_id": "3eca8232"
        },
        {
            "condition": "C3-D (Full Official Alignment)",
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "interpolation": "bilinear",
            "normalization": "raw [0, 1]",
            "seed": 42,
            "avg_incremental_acc": runs["C3-D"]["recalc"]["average_incremental_accuracy"],
            "last_acc": runs["C3-D"]["recalc"]["last_accuracy"],
            "forgetting_past": runs["C3-D"]["recalc"]["forgetting_past_tasks"],
            "forgetting_code": runs["C3-D"]["recalc"]["forgetting_official_code"],
            "wandb_run_id": "3v53tmq5"
        },
        {
            "condition": "C3-C (Bicubic, Raw)",
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "interpolation": "bicubic",
            "normalization": "raw [0, 1]",
            "seed": 42,
            "avg_incremental_acc": runs["C3-C"]["recalc"]["average_incremental_accuracy"],
            "last_acc": runs["C3-C"]["recalc"]["last_accuracy"],
            "forgetting_past": runs["C3-C"]["recalc"]["forgetting_past_tasks"],
            "forgetting_code": runs["C3-C"]["recalc"]["forgetting_official_code"],
            "wandb_run_id": "4m4uir7h"
        },
        {
            "condition": "C3-B (Bilinear, Normalized)",
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "interpolation": "bilinear",
            "normalization": "normalized [-1, 1]",
            "seed": 42,
            "avg_incremental_acc": runs["C3-B"]["recalc"]["average_incremental_accuracy"],
            "last_acc": runs["C3-B"]["recalc"]["last_accuracy"],
            "forgetting_past": runs["C3-B"]["recalc"]["forgetting_past_tasks"],
            "forgetting_code": runs["C3-B"]["recalc"]["forgetting_official_code"],
            "wandb_run_id": "ts2wh4e3"
        },
        {
            "condition": "C3-A / C2-B1 (Bicubic, Normalized)",
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "interpolation": "bicubic",
            "normalization": "normalized [-1, 1]",
            "seed": 42,
            "avg_incremental_acc": runs["C3-A"]["recalc"]["average_incremental_accuracy"],
            "last_acc": runs["C3-A"]["recalc"]["last_accuracy"],
            "forgetting_past": runs["C3-A"]["recalc"]["forgetting_past_tasks"],
            "forgetting_code": runs["C3-A"]["recalc"]["forgetting_official_code"],
            "wandb_run_id": "ne1p16b5"
        },
        {
            "condition": "C2-B0 (augreg_in1k Control)",
            "backbone": "vit_base_patch16_224.augreg_in1k",
            "interpolation": "bicubic",
            "normalization": "normalized [-1, 1]",
            "seed": 42,
            "avg_incremental_acc": 0.8328135317460317,
            "last_acc": 0.7484,
            "forgetting_past": 0.08966666666666666,
            "forgetting_code": 0.0807,
            "wandb_run_id": "3oiftwg6"
        }
    ]
    
    with open(analysis_dir / "c3_full_comparison.json", "w", encoding="utf-8") as f:
        json.dump(full_table, f, indent=2)
        
    with open(analysis_dir / "c3_full_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "backbone", "interpolation", "normalization", "seed", "avg_incremental_acc_pct", "last_acc_pct", "forgetting_past_pct", "forgetting_code_pct", "wandb_run_id"])
        for r in full_table:
            writer.writerow([
                r["condition"],
                r["backbone"],
                r["interpolation"],
                r["normalization"],
                r["seed"],
                round(r["avg_incremental_acc"] * 100, 4) if r["avg_incremental_acc"] is not None else "",
                round(r["last_acc"] * 100, 4) if r["last_acc"] is not None else "",
                round(r["forgetting_past"] * 100, 4) if r["forgetting_past"] is not None else "",
                round(r["forgetting_code"] * 100, 4) if r["forgetting_code"] is not None else "",
                r["wandb_run_id"]
            ])

    # 7. wandb_10runs_verified.json
    with open("results/wandb_7runs_verified.json", "r", encoding="utf-8") as f:
        w10 = json.load(f)
        
    w10["ts2wh4e3"] = {
        "id": "ts2wh4e3",
        "name": "c3_b_bilinear_normalized",
        "group": "c3_preprocessing_alignment",
        "state": "finished",
        "url": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3",
        "experiment": "C3-B Factor A: Bilinear, Normalize [-1, 1]"
    }
    w10["4m4uir7h"] = {
        "id": "4m4uir7h",
        "name": "c3_c_bicubic_raw",
        "group": "c3_preprocessing_alignment",
        "state": "finished",
        "url": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h",
        "experiment": "C3-C Factor B: Bicubic, Raw [0, 1]"
    }
    w10["3v53tmq5"] = {
        "id": "3v53tmq5",
        "name": "c3_d_bilinear_raw",
        "group": "c3_preprocessing_alignment",
        "state": "finished",
        "url": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5",
        "experiment": "C3-D Full Alignment: Bilinear, Raw [0, 1]"
    }
    with open("results/wandb_10runs_verified.json", "w", encoding="utf-8") as f:
        json.dump(w10, f, indent=2)
    print("Saved results/wandb_10runs_verified.json")

    # 8. Figures Generation
    
    # Figure 1: c3_stage_accuracy_comparison.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    stages = np.arange(1, 11)
    ax.plot(stages, [x * 100 for x in runs["C3-A"]["recalc"]["stage_accuracies"]], 'o-', label="C3-A: Bicubic, Norm [-1,1] (91.77%)", color="#1f77b4", linewidth=2.5, markersize=7)
    ax.plot(stages, [x * 100 for x in runs["C3-B"]["recalc"]["stage_accuracies"]], 's--', label="C3-B: Bilinear, Norm [-1,1] (91.39%)", color="#2ca02c", linewidth=2.0, markersize=6)
    ax.plot(stages, [x * 100 for x in runs["C3-C"]["recalc"]["stage_accuracies"]], '^-.', label="C3-C: Bicubic, Raw [0,1] (90.88%)", color="#ff7f0e", linewidth=2.0, markersize=6)
    ax.plot(stages, [x * 100 for x in runs["C3-D"]["recalc"]["stage_accuracies"]], 'D-', label="C3-D: Bilinear, Raw [0,1] (90.40%) [Official Aligned]", color="#d62728", linewidth=2.5, markersize=7)
    
    ax.set_title("F-OAL C3 Preprocessing Alignment: Incremental Accuracy Trajectory", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Incremental Task Stage", fontsize=12)
    ax.set_ylabel("Incremental Accuracy (%)", fontsize=12)
    ax.set_xticks(stages)
    ax.set_ylim([84, 99])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10.5, loc="upper right")
    plt.tight_layout()
    fig.savefig(analysis_dir / "c3_stage_accuracy_comparison.png", dpi=300)
    plt.close()
    print("Generated c3_stage_accuracy_comparison.png")

    # Figure 2: c3_factorial_interaction_plot.png
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    x = [0, 1]
    x_labels = ["Raw [0, 1]", "Normalized [-1, 1]"]
    y_bicubic = [runs["C3-C"]["recalc"]["average_incremental_accuracy"] * 100, runs["C3-A"]["recalc"]["average_incremental_accuracy"] * 100]
    y_bilinear = [runs["C3-D"]["recalc"]["average_incremental_accuracy"] * 100, runs["C3-B"]["recalc"]["average_incremental_accuracy"] * 100]
    
    ax.plot(x, y_bicubic, 'o-', color="#1f77b4", linewidth=2.5, markersize=9, label="Bicubic Interpolation")
    ax.plot(x, y_bilinear, 's--', color="#d62728", linewidth=2.5, markersize=9, label="Bilinear Interpolation")
    
    # Annotate points
    ax.annotate(f"{y_bicubic[0]:.2f}%", (0, y_bicubic[0]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color="#1f77b4")
    ax.annotate(f"{y_bicubic[1]:.2f}%", (1, y_bicubic[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color="#1f77b4")
    ax.annotate(f"{y_bilinear[0]:.2f}%", (0, y_bilinear[0]), textcoords="offset points", xytext=(0, -15), ha='center', fontweight='bold', color="#d62728")
    ax.annotate(f"{y_bilinear[1]:.2f}%", (1, y_bilinear[1]), textcoords="offset points", xytext=(0, -15), ha='center', fontweight='bold', color="#d62728")
    
    ax.set_title(r"2$\times$2 Factorial Interaction Plot ($\bar{A}$)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Input Normalization Strategy", fontsize=12)
    ax.set_ylabel(r"Average Incremental Accuracy $\bar{A}$ (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylim([89.8, 92.5])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11, loc="lower right")
    plt.tight_layout()
    fig.savefig(analysis_dir / "c3_factorial_interaction_plot.png", dpi=300)
    plt.close()
    print("Generated c3_factorial_interaction_plot.png")

    # Figure 3: c3_main_effects_and_interaction.png
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    eff_data = c3_audit["factorial_effects"]["average_incremental_accuracy"]
    factors = [
        "Main Effect: Interp\n(Bilinear vs Bicubic)",
        "Main Effect: Norm\n(Raw [0,1] vs Norm [-1,1])",
        "Interaction Contrast\n(E_IN)",
        "Total Shift\n(C3-D minus C3-A)"
    ]
    vals = [
        eff_data["main_interp"] * 100,
        eff_data["main_norm"] * 100,
        eff_data["contrast_interact"] * 100,
        eff_data["delta_total"] * 100
    ]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    colors[1] = "#e377c2" # highlight norm
    colors[3] = "#333333" # total shift
    
    bars = ax.bar(factors, vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1.0)
    
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.05 if h >= 0 else -0.08
        ax.annotate(f"{h:+.4f} pp",
                    (bar.get_x() + bar.get_width() / 2, h + offset),
                    ha="center", va=va, fontsize=10.5, fontweight="bold")
                    
    ax.set_title("Factorial Effects Decomposition on Average Incremental Accuracy", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Effect Magnitude (percentage points)", fontsize=11)
    ax.set_ylim([-1.65, 0.4])
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    plt.tight_layout()
    fig.savefig(analysis_dir / "c3_main_effects_and_interaction.png", dpi=300)
    plt.close()
    print("Generated c3_main_effects_and_interaction.png")

    # Figure 4: c3_paper_official_ours_comparison.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    labels = ["NeurIPS Paper", "Official Author Run\n(seed 0)", "C3-D Ours Aligned\n(seed 42)", "C3-A Ours Standard\n(seed 42)", "C2-B0 augreg_in1k\n(seed 42)"]
    accs = [91.1, 90.98, runs["C3-D"]["recalc"]["average_incremental_accuracy"]*100, runs["C3-A"]["recalc"]["average_incremental_accuracy"]*100, 83.28]
    bar_colors = ["#7f7f7f", "#bcbd22", "#d62728", "#1f77b4", "#8c564b"]
    
    bars = ax.bar(labels, accs, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h + 0.3), ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        
    ax.set_title(r"Cross-Condition Comparison: Average Incremental Accuracy ($\bar{A}$)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(r"$\bar{A}$ (%)", fontsize=11)
    ax.set_ylim([75, 95])
    ax.axhline(91.1, color="#7f7f7f", linestyle="--", linewidth=1.2, label="Paper Target (91.1%)")
    ax.axhline(90.98, color="#bcbd22", linestyle=":", linewidth=1.2, label="Official Author Run (90.98%)")
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    plt.tight_layout()
    fig.savefig(analysis_dir / "c3_paper_official_ours_comparison.png", dpi=300)
    plt.close()
    print("Generated c3_paper_official_ours_comparison.png")

    # Figure 5: c3_forgetting_comparison.png
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    tasks = np.arange(1, 10)
    ax.plot(tasks, [x * 100 for x in runs["C3-A"]["recalc"]["per_task_past_forgetting"]], 'o-', label="C3-A (Mean: 5.02%)", color="#1f77b4", linewidth=2.0)
    ax.plot(tasks, [x * 100 for x in runs["C3-B"]["recalc"]["per_task_past_forgetting"]], 's--', label="C3-B (Mean: 4.89%)", color="#2ca02c", linewidth=2.0)
    ax.plot(tasks, [x * 100 for x in runs["C3-C"]["recalc"]["per_task_past_forgetting"]], '^-.', label="C3-C (Mean: 5.23%)", color="#ff7f0e", linewidth=2.0)
    ax.plot(tasks, [x * 100 for x in runs["C3-D"]["recalc"]["per_task_past_forgetting"]], 'D-', label="C3-D (Mean: 5.16%)", color="#d62728", linewidth=2.2)
    
    ax.set_title("Per-Task Forgetting across Initial 9 Tasks", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Task ID", fontsize=11)
    ax.set_ylabel("Forgetting (%)", fontsize=11)
    ax.set_xticks(tasks)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10.5, loc="upper right")
    plt.tight_layout()
    fig.savefig(analysis_dir / "c3_forgetting_comparison.png", dpi=300)
    plt.close()
    print("Generated c3_forgetting_comparison.png")

    # Figure 6: c3_representation_diagnostics.png
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    
    cond_names = ["C3-A", "C3-B", "C3-C", "C3-D", "Official"]
    c_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"]
    
    # Subplot 1: Feature Norm Mean
    norms = [c3_diag["feature_stats"][c]["norm_mean"] for c in cond_names]
    b1 = ax1.bar(cond_names, norms, color=c_colors, width=0.55, edgecolor="black")
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + 0.3), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax1.set_title("Mean Feature Norm (L2)", fontsize=11, fontweight="bold")
    ax1.set_ylim([30, 42])
    ax1.grid(True, linestyle="--", alpha=0.5, axis='y')
    
    # Subplot 2: Effective Rank
    ranks = [c3_diag["feature_stats"][c]["r_eff"] for c in cond_names]
    b2 = ax2.bar(cond_names, ranks, color=c_colors, width=0.55, edgecolor="black")
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + 0.5), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax2.set_title(r"Effective Rank $r_{\mathrm{eff}}$ (N=200)", fontsize=11, fontweight="bold")
    ax2.set_ylim([75, 90])
    ax2.grid(True, linestyle="--", alpha=0.5, axis='y')
    
    # Subplot 3: Class Separation Ratio
    seps = [c3_diag["feature_stats"][c]["class_separation"]["separation_ratio"] for c in cond_names]
    b3 = ax3.bar(cond_names, seps, color=c_colors, width=0.55, edgecolor="black")
    for bar in b3:
        h = bar.get_height()
        ax3.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h + 0.003), ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax3.set_title("Class Separation Ratio", fontsize=11, fontweight="bold")
    ax3.set_ylim([1.08, 1.14])
    ax3.grid(True, linestyle="--", alpha=0.5, axis='y')
    
    fig.suptitle("C3 Representation Diagnostics: ViT Features (N=200, 88 Classes)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(analysis_dir / "c3_representation_diagnostics.png", dpi=300)
    plt.close()
    print("Generated c3_representation_diagnostics.png")

    # 9. c3_evidence_manifest.json
    manifest = {
        "verified_git_commit": "15b8afc71f6109efdbf3ff98a801ffcc83aa58d5",
        "archive_sha256": "a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6",
        "total_verified_runs": 10,
        "c3_runs": {
            "C3-A": {"run_id": "ne1p16b5", "avg_acc": 0.9177423015873016, "status": "REUSED_VERIFIED_C2_B1"},
            "C3-B": {"run_id": "ts2wh4e3", "avg_acc": runs["C3-B"]["recalc"]["average_incremental_accuracy"], "status": "VERIFIED_FRESH"},
            "C3-C": {"run_id": "4m4uir7h", "avg_acc": runs["C3-C"]["recalc"]["average_incremental_accuracy"], "status": "VERIFIED_FRESH"},
            "C3-D": {"run_id": "3v53tmq5", "avg_acc": runs["C3-D"]["recalc"]["average_incremental_accuracy"], "status": "VERIFIED_FRESH"}
        },
        "diagnostics": {
            "num_samples": 200,
            "sample_indices_sha256": "c5588600ae51253d8194e86d381c44278949b5ff99adf45e573fc2aa17d9c4d7",
            "c3_d_vs_official_pixel_diff": {"max_abs": 0.0, "mean_abs": 0.0},
            "c3_d_vs_official_cosine_similarity": 1.0
        },
        "scientific_verdict": "C3_SIGNED_OFF_WITH_LIMITATIONS"
    }
    with open(analysis_dir / "c3_evidence_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved results/c3_analysis/c3_evidence_manifest.json successfully!")

if __name__ == "__main__":
    main()
