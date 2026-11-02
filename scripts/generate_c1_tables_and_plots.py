import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

audit_path = Path("results/c1_audit_parsed.json")
data = json.loads(audit_path.read_text(encoding="utf-8"))

# Stage Accuracies Table
stage_table = []
for stage in range(1, 11):
    row = {"Stage": stage}
    for v_key, label in [
        ("p0_gaussian_std1", "P0 (Gaussian std=1)"),
        ("p1_official_uniform", "P1 (Official Uniform)"),
        ("p2_gaussian_matched_variance", "P2 (Matched Gaussian)"),
    ]:
        row[label] = round(data[v_key]["recalculated"]["stage_accuracies"][stage - 1] * 100, 2)
    stage_table.append(row)

df_stages = pd.DataFrame(stage_table)
print("--- STAGE ACCURACY TRAJECTORY (%) ---")
print(df_stages.to_string(index=False))

# Per-Task Final Accuracy (Row 10 of matrix)
final_task_table = []
for task in range(1, 11):
    row = {"Task": task}
    for v_key, label in [
        ("p0_gaussian_std1", "P0"),
        ("p1_official_uniform", "P1"),
        ("p2_gaussian_matched_variance", "P2"),
    ]:
        row[label] = round(data[v_key]["task_matrix"][9][task - 1] * 100, 2)
    final_task_table.append(row)
df_final = pd.DataFrame(final_task_table)
print("\n--- FINAL TASK ACCURACY AT STAGE 10 (%) ---")
print(df_final.to_string(index=False))

# Per-Task Forgetting (Paper past-tasks formula)
fgt_past_table = []
for task in range(1, 10):
    row = {"Task": task}
    for v_key, label in [
        ("p0_gaussian_std1", "P0"),
        ("p1_official_uniform", "P1"),
        ("p2_gaussian_matched_variance", "P2"),
    ]:
        row[label] = round(data[v_key]["recalculated"]["task_forgetting_past_tasks"][task - 1] * 100, 2)
    fgt_past_table.append(row)
df_fgt = pd.DataFrame(fgt_past_table)
print("\n--- PER-TASK FORGETTING (Past-Tasks Formula) (%) ---")
print(df_fgt.to_string(index=False))

# Create analysis output dir
out_dir = Path("results/c1_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

df_stages.to_csv(out_dir / "c1_stage_accuracies.csv", index=False)
df_final.to_csv(out_dir / "c1_final_task_accuracies.csv", index=False)
df_fgt.to_csv(out_dir / "c1_task_forgetting.csv", index=False)

# Comprehensive Summary Table
comp_rows = [
    {
        "Variant": "P0: gaussian_std1",
        "Initialization": "Gaussian N(0, 1.0)",
        "Variance Scale": "1.0",
        "Avg Incr Acc (%)": f"{data['p0_gaussian_std1']['recalculated']['average_incremental_accuracy']*100:.2f}%",
        "Last Acc (%)": f"{data['p0_gaussian_std1']['recalculated']['last_accuracy']*100:.2f}%",
        "Fgt Past-Tasks (%)": f"{data['p0_gaussian_std1']['recalculated']['forgetting_past_tasks']*100:.2f}%",
        "Fgt Official Code (%)": f"{data['p0_gaussian_std1']['recalculated']['forgetting_official_code']*100:.2f}%",
        "Pre-sigmoid Std (Diagnostics)": "19.2844",
        "Sigmoid Saturation (<0.01 | >0.99)": "81.03%",
        "Feat Extraction Time (s)": f"{data['p0_gaussian_std1']['timing']['feature_extraction_seconds']:.2f}",
        "RLS & Eval Time (s)": f"{data['p0_gaussian_std1']['timing']['rls_and_eval_seconds']:.2f}",
        "Total Time (s)": f"{data['p0_gaussian_std1']['timing']['total_duration_seconds']:.2f}",
    },
    {
        "Variant": "P1: official_uniform",
        "Initialization": "Uniform U(-1/sqrt(d), +1/sqrt(d))",
        "Variance Scale": "1/(3*768) ≈ 0.000434",
        "Avg Incr Acc (%)": f"{data['p1_official_uniform']['recalculated']['average_incremental_accuracy']*100:.2f}%",
        "Last Acc (%)": f"{data['p1_official_uniform']['recalculated']['last_accuracy']*100:.2f}%",
        "Fgt Past-Tasks (%)": f"{data['p1_official_uniform']['recalculated']['forgetting_past_tasks']*100:.2f}%",
        "Fgt Official Code (%)": f"{data['p1_official_uniform']['recalculated']['forgetting_official_code']*100:.2f}%",
        "Pre-sigmoid Std (Diagnostics)": "0.3979",
        "Sigmoid Saturation (<0.01 | >0.99)": "0.00%",
        "Feat Extraction Time (s)": f"{data['p1_official_uniform']['timing']['feature_extraction_seconds']:.2f} (cached)",
        "RLS & Eval Time (s)": f"{data['p1_official_uniform']['timing']['rls_and_eval_seconds']:.2f}",
        "Total Time (s)": f"{data['p1_official_uniform']['timing']['total_duration_seconds']:.2f}",
    },
    {
        "Variant": "P2: gaussian_matched_variance",
        "Initialization": "Gaussian N(0, 1/(3*768))",
        "Variance Scale": "1/(3*768) ≈ 0.000434",
        "Avg Incr Acc (%)": f"{data['p2_gaussian_matched_variance']['recalculated']['average_incremental_accuracy']*100:.2f}%",
        "Last Acc (%)": f"{data['p2_gaussian_matched_variance']['recalculated']['last_accuracy']*100:.2f}%",
        "Fgt Past-Tasks (%)": f"{data['p2_gaussian_matched_variance']['recalculated']['forgetting_past_tasks']*100:.2f}%",
        "Fgt Official Code (%)": f"{data['p2_gaussian_matched_variance']['recalculated']['forgetting_official_code']*100:.2f}%",
        "Pre-sigmoid Std (Diagnostics)": "0.4018",
        "Sigmoid Saturation (<0.01 | >0.99)": "0.00%",
        "Feat Extraction Time (s)": f"{data['p2_gaussian_matched_variance']['timing']['feature_extraction_seconds']:.2f} (cached)",
        "RLS & Eval Time (s)": f"{data['p2_gaussian_matched_variance']['timing']['rls_and_eval_seconds']:.2f}",
        "Total Time (s)": f"{data['p2_gaussian_matched_variance']['timing']['total_duration_seconds']:.2f}",
    },
]

df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv(out_dir / "c1_full_comparison.csv", index=False)
(out_dir / "c1_full_comparison.json").write_text(json.dumps(comp_rows, indent=2), encoding="utf-8")
print("\n--- C1 COMPREHENSIVE COMPARISON TABLE ---")
print(df_comp.to_string(index=False))

# Plot Stage Accuracy Comparison Chart
plt.figure(figsize=(10, 6))
stages = list(range(1, 11))
p0_curve = [a * 100 for a in data["p0_gaussian_std1"]["recalculated"]["stage_accuracies"]]
p1_curve = [a * 100 for a in data["p1_official_uniform"]["recalculated"]["stage_accuracies"]]
p2_curve = [a * 100 for a in data["p2_gaussian_matched_variance"]["recalculated"]["stage_accuracies"]]

plt.plot(stages, p0_curve, marker="o", color="#d9534f", linewidth=2.2, label="P0: Gaussian std=1 (A_avg=79.33%, A_last=70.95%)")
plt.plot(stages, p1_curve, marker="s", color="#0275d8", linewidth=2.2, linestyle="--", label="P1: Official Uniform (A_avg=83.28%, A_last=74.84%)")
plt.plot(stages, p2_curve, marker="^", color="#5cb85c", linewidth=2.2, linestyle=":", label="P2: Matched Gaussian (A_avg=83.37%, A_last=74.99%)")

plt.title("CIFAR-100 Incremental Accuracy across Stages (C1 Controlled Experiment)", fontsize=13, fontweight="bold")
plt.xlabel("Incremental Task / Stage", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.xticks(stages)
plt.ylim(65, 95)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()
chart_path = out_dir / "c1_stage_accuracy_comparison.png"
plt.savefig(chart_path, dpi=300)
print(f"\nGenerated stage accuracy plot at: {chart_path}")
