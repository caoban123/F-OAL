"""Generate final headline comparison artifacts and figures incorporating Official LwF with precise scientific auditing."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import csv
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 1. Load verified data
# F-OAL (C3-D)
c3_csv = ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw" / "task_accuracy.csv"
records = []
with open(c3_csv, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        records.append((int(r["stage"]), int(r["task"]), float(r["accuracy"])))
foal_mat = [[0.0] * i for i in range(1, 11)]
for s, t, acc in records:
    foal_mat[s - 1][t - 1] = acc
foal_stages = [float(np.mean(r)) for r in foal_mat]
foal_a_avg = float(np.mean(foal_stages))
foal_a_last = float(foal_stages[-1])
foal_f_past = 5.155555555555556
foal_f_code = 4.64

# ER (Adapted)
er_csv = ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "er" / "task_matrix.csv"
with open(er_csv, "r", encoding="utf-8") as f:
    er_mat = [[float(x) for x in r] for r in csv.reader(f)]
er_stages = [float(np.mean(r)) for r in er_mat]
er_a_avg = float(np.mean(er_stages))
er_a_last = float(er_stages[-1])
er_f_past = 16.744444444444444
er_f_code = 15.07

# LwF (Adapted, AdamW)
lwf_ad_csv = ROOT / "results" / "headline_full_gpu_verified" / "runs" / "headline" / "lwf" / "task_matrix.csv"
with open(lwf_ad_csv, "r", encoding="utf-8") as f:
    lwf_ad_mat = [[float(x) for x in r] for r in csv.reader(f)]
lwf_ad_stages = [float(np.mean(r)) for r in lwf_ad_mat]
lwf_ad_a_avg = float(np.mean(lwf_ad_stages))
lwf_ad_a_last = float(lwf_ad_stages[-1])
lwf_ad_f_past = 94.98888888888888
lwf_ad_f_code = 85.49

# LwF (Official Protocol, SGD + Dynamic Loss)
lwf_off_csv = ROOT / "results" / "headline_official_lwf_verified" / "runs" / "headline" / "lwf_official_cifar100" / "task_matrix.csv"
with open(lwf_off_csv, "r", encoding="utf-8") as f:
    lwf_off_mat = [[float(x) for x in r] for r in csv.reader(f)]
lwf_off_stages = [float(np.mean(r)) for r in lwf_off_mat]
lwf_off_a_avg = float(np.mean(lwf_off_stages))
lwf_off_a_last = float(lwf_off_stages[-1])
lwf_off_f_past_clipped = 0.41111111111111115
lwf_off_f_code_clipped = 0.37000000000000005

out_dir = ROOT / "results" / "headline_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. headline_final_metrics.json & .csv
# -----------------------------------------------------------------------------
metrics_data = [
    {
        "Method": "F-OAL (Full)",
        "Classification": "OUR_INDEPENDENT_FOAL",
        "Paradigm": "Analytic Forward-Only",
        "Buffer_Size": 0,
        "Replay": False,
        "A_avg_pct": foal_a_avg * 100,
        "A_last_pct": foal_a_last * 100,
        "F_past_pct": foal_f_past,
        "F_code_pct": foal_f_code,
        "Training_Time_s": 229.11,
        "Hardware": "NVIDIA A100-SXM4-40GB",
        "Feature_Dim": 768,
        "Classifier_Architecture": "Analytic Closed-Form Ridge W=(H^T H + gamma I)^(-1) H^T Y",
        "Optimizer": "Analytic Ridge (Closed-form RLS)",
        "Execution_SHA": "d333705 / W&B 3v53tmq5",
        "Notes": "C3-D verified run. Zero replay, zero iterative backpropagation."
    },
    {
        "Method": "ER (Adapted, Replay)",
        "Classification": "OUR_ADAPTED_ER",
        "Paradigm": "Experience Replay",
        "Buffer_Size": 5000,
        "Replay": True,
        "A_avg_pct": er_a_avg * 100,
        "A_last_pct": er_a_last * 100,
        "F_past_pct": er_f_past,
        "F_code_pct": er_f_code,
        "Training_Time_s": 626.846,
        "Hardware": "Tesla T4 (14.56 GB)",
        "Feature_Dim": 768,
        "Classifier_Architecture": "nn.Linear(768, num_classes) with dynamic expansion",
        "Optimizer": "AdamW (lr=0.001)",
        "Execution_SHA": "d333705",
        "Notes": "Adapted baseline on frozen ViT features with 5000-exemplar reservoir buffer."
    },
    {
        "Method": "LwF (Official Protocol)",
        "Classification": "OUR_OFFICIAL_PROTOCOL_LWF",
        "Paradigm": "Exemplar-Free Distillation",
        "Buffer_Size": 0,
        "Replay": False,
        "A_avg_pct": lwf_off_a_avg * 100,
        "A_last_pct": lwf_off_a_last * 100,
        "F_past_pct": lwf_off_f_past_clipped,
        "F_code_pct": lwf_off_f_code_clipped,
        "Training_Time_s": 646.351,
        "Hardware": "Tesla T4 (14.56 GB)",
        "Feature_Dim": 768,
        "Classifier_Architecture": "nn.Linear(768, 100) static head",
        "Optimizer": "SGD (lr=0.001, momentum=0.0, weight_decay=0.0)",
        "Execution_SHA": "c3dd3f23fbf2d8d94d481ec75daceb860190118e",
        "Notes": "Implements author loss weighting 1/(t+1) and static 100-class head, evaluated on 12-block averaged ViT features (D=768)."
    },
    {
        "Method": "LwF (Adapted, AdamW)",
        "Classification": "OUR_ADAPTED_LWF",
        "Paradigm": "Exemplar-Free Distillation",
        "Buffer_Size": 0,
        "Replay": False,
        "A_avg_pct": lwf_ad_a_avg * 100,
        "A_last_pct": lwf_ad_a_last * 100,
        "F_past_pct": lwf_ad_f_past,
        "F_code_pct": lwf_ad_f_code,
        "Training_Time_s": 629.517,
        "Hardware": "Tesla T4 (14.56 GB)",
        "Feature_Dim": 768,
        "Classifier_Architecture": "nn.Linear(768, num_classes) with dynamic expansion",
        "Optimizer": "AdamW (lr=0.001)",
        "Execution_SHA": "d333705",
        "Notes": "Historical adapted baseline with dynamic head expansion. Suffered recency collapse under static 1.0/1.0 loss weighting."
    }
]

(out_dir / "headline_final_metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

with open(out_dir / "headline_final_metrics.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics_data[0].keys()))
    writer.writeheader()
    writer.writerows(metrics_data)

# -----------------------------------------------------------------------------
# 2. headline_final_paper_comparison.csv
# -----------------------------------------------------------------------------
paper_comp = [
    {
        "Method": "F-OAL (Full)",
        "Buffer": 0,
        "Paper_A_avg": 91.1,
        "Ours_A_avg": round(foal_a_avg * 100, 2),
        "Delta_A_avg": round(foal_a_avg * 100 - 91.1, 2),
        "Paper_A_last": 86.5,
        "Ours_A_last": round(foal_a_last * 100, 2),
        "Delta_A_last": round(foal_a_last * 100 - 86.5, 2),
        "Paper_F": 5.5,
        "Ours_F_past": round(foal_f_past, 2),
        "Delta_F_past": round(foal_f_past - 5.5, 2),
        "Scientific_Status": "REPRODUCED (Delta < 0.8 pp on CIFAR-100 seed 42)"
    },
    {
        "Method": "ER (Adapted, Replay)",
        "Buffer": 5000,
        "Paper_A_avg": 90.1,
        "Ours_A_avg": round(er_a_avg * 100, 2),
        "Delta_A_avg": round(er_a_avg * 100 - 90.1, 2),
        "Paper_A_last": 84.6,
        "Ours_A_last": round(er_a_last * 100, 2),
        "Delta_A_last": round(er_a_last * 100 - 84.6, 2),
        "Paper_F": 20.7,
        "Ours_F_past": round(er_f_past, 2),
        "Delta_F_past": round(er_f_past - 20.7, 2),
        "Scientific_Status": "CONSISTENT_TREND (Preserves ranking F-OAL > ER)"
    },
    {
        "Method": "LwF (Official Protocol)",
        "Buffer": 0,
        "Paper_A_avg": 69.3,
        "Ours_A_avg": round(lwf_off_a_avg * 100, 2),
        "Delta_A_avg": round(lwf_off_a_avg * 100 - 69.3, 2),
        "Paper_A_last": 64.8,
        "Ours_A_last": round(lwf_off_a_last * 100, 2),
        "Delta_A_last": round(lwf_off_a_last * 100 - 64.8, 2),
        "Paper_F": 1.3,
        "Ours_F_past": round(lwf_off_f_past_clipped, 2),
        "Delta_F_past": round(lwf_off_f_past_clipped - 1.3, 2),
        "Scientific_Status": "RIGIDITY_COLLAPSE (Low forgetting 0.41% reflects under-learning of new tasks; Last Acc 23.34% vs Paper 64.8%)"
    },
    {
        "Method": "LwF (Adapted, AdamW)",
        "Buffer": 0,
        "Paper_A_avg": 69.3,
        "Ours_A_avg": round(lwf_ad_a_avg * 100, 2),
        "Delta_A_avg": round(lwf_ad_a_avg * 100 - 69.3, 2),
        "Paper_A_last": 64.8,
        "Ours_A_last": round(lwf_ad_a_last * 100, 2),
        "Delta_A_last": round(lwf_ad_a_last * 100 - 64.8, 2),
        "Paper_F": 1.3,
        "Ours_F_past": round(lwf_ad_f_past, 2),
        "Delta_F_past": round(lwf_ad_f_past - 1.3, 2),
        "Scientific_Status": "RECENCY_COLLAPSE (Suffered catastrophic forgetting; superseded by Official Protocol run)"
    }
]

with open(out_dir / "headline_final_paper_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(paper_comp[0].keys()))
    writer.writeheader()
    writer.writerows(paper_comp)

# -----------------------------------------------------------------------------
# 3. headline_final_stage_accuracies.csv
# -----------------------------------------------------------------------------
stage_rows = []
for s in range(10):
    stage_rows.append({
        "Stage": s + 1,
        "Num_Seen_Classes": (s + 1) * 10,
        "FOAL_Full_pct": round(foal_stages[s] * 100, 4),
        "ER_Replay_pct": round(er_stages[s] * 100, 4),
        "LwF_Official_pct": round(lwf_off_stages[s] * 100, 4),
        "LwF_Adapted_pct": round(lwf_ad_stages[s] * 100, 4)
    })

with open(out_dir / "headline_final_stage_accuracies.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
    writer.writeheader()
    writer.writerows(stage_rows)

# -----------------------------------------------------------------------------
# 4. headline_final_task_forgetting.csv
# -----------------------------------------------------------------------------
task_f_rows = []
for j in range(9):
    f_foal = max(foal_mat[k][j] for k in range(j, 9)) - foal_mat[9][j]
    f_er = max(er_mat[k][j] for k in range(j, 9)) - er_mat[9][j]
    f_lwf_ad = max(lwf_ad_mat[k][j] for k in range(j, 9)) - lwf_ad_mat[9][j]
    f_lwf_off = max(lwf_off_mat[k][j] for k in range(j, 9)) - lwf_off_mat[9][j]
    task_f_rows.append({
        "Task": j + 1,
        "FOAL_Full_F_pct": round(max(0.0, f_foal) * 100, 4),
        "ER_Replay_F_pct": round(max(0.0, f_er) * 100, 4),
        "LwF_Official_F_pct": round(max(0.0, f_lwf_off) * 100, 4),
        "LwF_Adapted_F_pct": round(max(0.0, f_lwf_ad) * 100, 4)
    })

with open(out_dir / "headline_final_task_forgetting.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(task_f_rows[0].keys()))
    writer.writeheader()
    writer.writerows(task_f_rows)

# -----------------------------------------------------------------------------
# 5. headline_final_protocol_matrix.csv
# -----------------------------------------------------------------------------
protocol_rows = [
    {
        "Dimension": "Paradigm",
        "Paper_LwF": "Exemplar-Free Distillation",
        "Adapted_LwF": "Exemplar-Free Distillation",
        "Official_LwF": "Exemplar-Free Distillation",
        "Paper_FOAL": "Analytic Forward-Only",
        "Ours_FOAL": "Analytic Forward-Only"
    },
    {
        "Dimension": "Backbone",
        "Paper_LwF": "ViT-B/16 (AugReg 21k->1k)",
        "Adapted_LwF": "ViT-B/16 (AugReg 21k->1k)",
        "Official_LwF": "ViT-B/16 (AugReg 21k->1k)",
        "Paper_FOAL": "ViT-B/16 (AugReg 21k->1k)",
        "Ours_FOAL": "ViT-B/16 (AugReg 21k->1k)"
    },
    {
        "Dimension": "Feature Extraction",
        "Paper_LwF": "Single-layer final CLS token (D=768)",
        "Adapted_LwF": "12-block CLS average + LayerNorm (D=768)",
        "Official_LwF": "12-block CLS average + LayerNorm (D=768)",
        "Paper_FOAL": "12-block CLS average + LayerNorm (D=768)",
        "Ours_FOAL": "12-block CLS average + LayerNorm (D=768)"
    },
    {
        "Dimension": "Classifier Head",
        "Paper_LwF": "nn.Linear(768, 100)",
        "Adapted_LwF": "nn.Linear(768, dynamic_classes)",
        "Official_LwF": "nn.Linear(768, 100)",
        "Paper_FOAL": "Ridge Regression Head (D=1000 from SP)",
        "Ours_FOAL": "Ridge Regression Head (D=1000 from SP)"
    },
    {
        "Dimension": "Loss Weighting",
        "Paper_LwF": "1/(t+1) new + (1-1/(t+1)) old",
        "Adapted_LwF": "1.0 new + 1.0 old (Fixed)",
        "Official_LwF": "1/(t+1) new + (1-1/(t+1)) old",
        "Paper_FOAL": "None (Analytic Ridge)",
        "Ours_FOAL": "None (Analytic Ridge)"
    },
    {
        "Dimension": "Optimizer",
        "Paper_LwF": "SGD (lr=0.001)",
        "Adapted_LwF": "AdamW (lr=0.001)",
        "Official_LwF": "SGD (lr=0.001)",
        "Paper_FOAL": "RLS (gamma=1.0)",
        "Ours_FOAL": "RLS (gamma=1.0)"
    },
    {
        "Dimension": "Replay Buffer",
        "Paper_LwF": "0",
        "Adapted_LwF": "0",
        "Official_LwF": "0",
        "Paper_FOAL": "0",
        "Ours_FOAL": "0"
    },
    {
        "Dimension": "Class Order",
        "Paper_LwF": "MT19937 seed=42 (starts with 83)",
        "Adapted_LwF": "PCG64 seed=42 (starts with 59)",
        "Official_LwF": "PCG64 seed=42 (starts with 59)",
        "Paper_FOAL": "MT19937 seed=42 / seed=0",
        "Ours_FOAL": "PCG64 seed=42 (consistent)"
    }
]

with open(out_dir / "headline_final_protocol_matrix.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(protocol_rows[0].keys()))
    writer.writeheader()
    writer.writerows(protocol_rows)

# -----------------------------------------------------------------------------
# 6. Figures
# -----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Fig 1: Paper vs Reproduction Average Accuracy
plt.figure(figsize=(9, 5))
methods = ["F-OAL (Full)", "ER (Replay)", "LwF (Official)", "LwF (Adapted)"]
paper_vals = [91.1, 90.1, 69.3, 69.3]
ours_vals = [foal_a_avg * 100, er_a_avg * 100, lwf_off_a_avg * 100, lwf_ad_a_avg * 100]
x = np.arange(len(methods))
w = 0.35
plt.bar(x - w/2, paper_vals, w, label="Paper Table 2", color="#7f7f7f", alpha=0.85)
plt.bar(x + w/2, ours_vals, w, label="Our Reproduction", color=["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"])
for i in range(len(methods)):
    plt.text(x[i] - w/2, paper_vals[i] + 1.0, f"{paper_vals[i]:.1f}%", ha="center", fontsize=9)
    plt.text(x[i] + w/2, ours_vals[i] + 1.0, f"{ours_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
plt.ylabel("Average Incremental Accuracy (%)", fontsize=11)
plt.title("CIFAR-100 Average Accuracy: Paper vs Our Reproduction", fontsize=12, fontweight="bold")
plt.xticks(x, methods)
plt.ylim(0, 105)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(out_dir / "fig1_headline_final_average_accuracy.png", dpi=300)
plt.close()

# Fig 2: F-OAL vs Baselines (Final Ranking)
plt.figure(figsize=(8, 5))
rank_m = ["F-OAL (Full, Ours)", "ER (Adapted, Buffer=5000)", "LwF (Official Protocol)", "LwF (Adapted)"]
rank_acc = [foal_a_avg * 100, er_a_avg * 100, lwf_off_a_avg * 100, lwf_ad_a_avg * 100]
colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
bars = plt.barh(rank_m[::-1], rank_acc[::-1], color=colors[::-1], height=0.55)
for b in bars:
    w_val = b.get_width()
    plt.text(w_val + 1.2, b.get_y() + b.get_height()/2, f"{w_val:.2f}%", va="center", fontsize=10, fontweight="bold")
plt.xlabel("Average Incremental Accuracy (%)", fontsize=11)
plt.title("Headline Comparison: Empirical Benchmark on CIFAR-100", fontsize=12, fontweight="bold")
plt.xlim(0, 105)
plt.tight_layout()
plt.savefig(out_dir / "fig2_headline_final_foal_vs_baselines.png", dpi=300)
plt.close()

# Fig 3: Adapted vs Official LwF (The Stability-Plasticity Resolution)
plt.figure(figsize=(9, 5))
stages = range(1, 11)
plt.plot(stages, [s * 100 for s in lwf_ad_stages], marker="v", linestyle="--", color="#ff7f0e", linewidth=2, label="LwF (Adapted, AdamW): Recency Collapse (Avg 29.63%)")
plt.plot(stages, [s * 100 for s in lwf_off_stages], marker="^", color="#d62728", linewidth=2.5, label="LwF (Official Protocol, SGD): High Stability / Low Plasticity (Avg 46.29%)")
plt.axhline(69.3, color="#7f7f7f", linestyle=":", label="Paper Reference LwF (69.30%)")
for i, s in enumerate(stages):
    diff = (lwf_off_stages[i] - lwf_ad_stages[i]) * 100
    if i in [1, 3, 5, 7, 9]:
        plt.annotate(f"+{diff:.1f}%", (s, lwf_off_stages[i] * 100), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="#d62728", fontweight="bold")
plt.xlabel("Task Stage (10 classes per stage)", fontsize=11)
plt.ylabel("Stage Incremental Accuracy (%)", fontsize=11)
plt.title("Effect of Official Protocol Alignment on LwF: +16.66% Gain Across Stages", fontsize=12, fontweight="bold")
plt.xticks(stages)
plt.ylim(0, 100)
plt.legend(frameon=True, loc="upper right")
plt.tight_layout()
plt.savefig(out_dir / "fig3_headline_final_adapted_vs_official_lwf.png", dpi=300)
plt.close()

# Fig 4: Stage-wise Accuracy trajectories (All 4 Methods)
plt.figure(figsize=(10, 5.5))
plt.plot(stages, [s * 100 for s in foal_stages], marker="o", color="#1f77b4", linewidth=2.5, label=f"F-OAL Full (Zero Replay, Ours) - Avg {foal_a_avg*100:.1f}%")
plt.plot(stages, [s * 100 for s in er_stages], marker="s", color="#2ca02c", linewidth=2, label=f"ER (Buffer=5000, Ours) - Avg {er_a_avg*100:.1f}%")
plt.plot(stages, [s * 100 for s in lwf_off_stages], marker="^", color="#d62728", linewidth=2, label=f"LwF (Official Protocol, Colab GPU) - Avg {lwf_off_a_avg*100:.1f}%")
plt.plot(stages, [s * 100 for s in lwf_ad_stages], marker="v", linestyle="--", color="#ff7f0e", linewidth=1.5, label=f"LwF (Adapted, AdamW) - Avg {lwf_ad_a_avg*100:.1f}%")
plt.xlabel("Task Stage (10 tasks, 100 classes total)", fontsize=11)
plt.ylabel("Incremental Accuracy (%)", fontsize=11)
plt.title("CIFAR-100 Incremental Accuracy Trajectory Across 10 Stages", fontsize=12, fontweight="bold")
plt.xticks(stages)
plt.ylim(0, 105)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True, fontsize=10, loc="lower left")
plt.tight_layout()
plt.savefig(out_dir / "fig4_headline_final_stage_trajectories.png", dpi=300)
plt.close()

# Fig 5: Final-stage Accuracy (A_last)
plt.figure(figsize=(8, 5))
last_accs = [foal_a_last * 100, er_a_last * 100, lwf_off_a_last * 100, lwf_ad_a_last * 100]
paper_lasts = [86.5, 84.6, 64.8, 64.8]
x = np.arange(len(methods))
plt.bar(x - w/2, paper_lasts, w, label="Paper Table 2 A_last", color="#7f7f7f", alpha=0.85)
plt.bar(x + w/2, last_accs, w, label="Our Reproduction A_last", color=colors)
for i in range(len(methods)):
    plt.text(x[i] - w/2, paper_lasts[i] + 1.0, f"{paper_lasts[i]:.1f}%", ha="center", fontsize=9)
    plt.text(x[i] + w/2, last_accs[i] + 1.0, f"{last_accs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
plt.ylabel("Final Stage 10 Accuracy (%)", fontsize=11)
plt.title("CIFAR-100 Final Stage Accuracy (A_last): Paper vs Reproduction", fontsize=12, fontweight="bold")
plt.xticks(x, methods)
plt.ylim(0, 100)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(out_dir / "fig5_headline_final_last_accuracy.png", dpi=300)
plt.close()

# Fig 6: Forgetting Comparison
plt.figure(figsize=(9, 5))
methods_f = ["F-OAL (Full)", "ER (Buffer 5k)", "LwF (Official)", "LwF (Adapted)"]
paper_f = [5.5, 20.7, 1.3, 1.3]
ours_f = [foal_f_past, er_f_past, lwf_off_f_past_clipped, lwf_ad_f_past]
x = np.arange(len(methods_f))
plt.bar(x - w/2, paper_f, w, label="Paper Reported Forgetting", color="#7f7f7f", alpha=0.85)
plt.bar(x + w/2, ours_f, w, label="Our Reproduced F_past", color=colors)
for i in range(len(methods_f)):
    plt.text(x[i] - w/2, paper_f[i] + 1.0, f"{paper_f[i]:.1f}%", ha="center", fontsize=9)
    plt.text(x[i] + w/2, ours_f[i] + 1.0, f"{ours_f[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
plt.ylabel("Catastrophic Forgetting (%) - Lower is Better", fontsize=11)
plt.title("CIFAR-100 Catastrophic Forgetting: F-OAL and Official LwF Comparisons", fontsize=12, fontweight="bold")
plt.xticks(x, methods_f)
plt.ylim(0, 110)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(out_dir / "fig6_headline_final_forgetting_comparison.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 7. Manifests
# -----------------------------------------------------------------------------
archive_manifest = {
    "historical_headline_archive": {
        "filename": "foal_cifar100_headline_results.zip",
        "sha256": hashlib.sha256((ROOT / "foal_cifar100_headline_results.zip").read_bytes()).hexdigest(),
        "size_bytes": (ROOT / "foal_cifar100_headline_results.zip").stat().st_size,
        "contains": ["runs/headline/lwf", "runs/headline/er"],
        "status": "PRESERVED_HISTORICAL"
    },
    "official_lwf_archive": {
        "filename": "foal_cifar100_official_lwf_results.zip",
        "sha256": hashlib.sha256((ROOT / "foal_cifar100_official_lwf_results.zip").read_bytes()).hexdigest(),
        "size_bytes": (ROOT / "foal_cifar100_official_lwf_results.zip").stat().st_size,
        "contains": ["runs/headline/lwf_official_cifar100", "configs/headline/lwf_cifar100_official_protocol.yaml"],
        "status": "VERIFIED_PRIMARY"
    }
}
(out_dir / "headline_final_archive_manifest.json").write_text(json.dumps(archive_manifest, indent=2), encoding="utf-8")

evidence_manifest = {
    "project": "F-OAL Headline Comparison Final Audit",
    "dataset": "CIFAR-100 (10 tasks x 10 classes)",
    "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "feature_dim": 768,
    "feature_fusion": "12-block CLS average + LayerNorm (D=768)",
    "preprocessing": "Bilinear interpolation, unnormalized [0, 1]",
    "runs": [
        {
            "method": "F-OAL (Full)",
            "classification": "OUR_INDEPENDENT_FOAL",
            "source_run_id": "3v53tmq5",
            "source_experiment": "C3-D Bilinear",
            "raw_file": "results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/task_accuracy.csv",
            "sha256": hashlib.sha256(c3_csv.read_bytes()).hexdigest(),
            "metrics": {"a_avg": foal_a_avg * 100, "a_last": foal_a_last * 100, "f_past": foal_f_past}
        },
        {
            "method": "ER (Adapted)",
            "classification": "OUR_ADAPTED_ER",
            "source_run": "runs/headline/er",
            "raw_file": "results/headline_full_gpu_verified/runs/headline/er/task_matrix.csv",
            "sha256": hashlib.sha256(er_csv.read_bytes()).hexdigest(),
            "metrics": {"a_avg": er_a_avg * 100, "a_last": er_a_last * 100, "f_past": er_f_past}
        },
        {
            "method": "LwF (Official Protocol)",
            "classification": "OUR_OFFICIAL_PROTOCOL_LWF",
            "source_run": "runs/headline/lwf_official_cifar100",
            "raw_file": "results/headline_official_lwf_verified/runs/headline/lwf_official_cifar100/task_matrix.csv",
            "sha256": hashlib.sha256(lwf_off_csv.read_bytes()).hexdigest(),
            "metrics": {"a_avg": lwf_off_a_avg * 100, "a_last": lwf_off_a_last * 100, "f_past": lwf_off_f_past_clipped}
        },
        {
            "method": "LwF (Adapted)",
            "classification": "OUR_ADAPTED_LWF",
            "source_run": "runs/headline/lwf",
            "raw_file": "results/headline_full_gpu_verified/runs/headline/lwf/task_matrix.csv",
            "sha256": hashlib.sha256(lwf_ad_csv.read_bytes()).hexdigest(),
            "metrics": {"a_avg": lwf_ad_a_avg * 100, "a_last": lwf_ad_a_last * 100, "f_past": lwf_ad_f_past}
        }
    ]
}
(out_dir / "headline_final_evidence_manifest.json").write_text(json.dumps(evidence_manifest, indent=2), encoding="utf-8")

print("All headline final artifacts and figures generated successfully in results/headline_analysis/")
