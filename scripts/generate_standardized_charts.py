"""
Generate 4 Standardized Scientific Comparison Charts for F-OAL Reproduction.

Chart 1: C1 Stage Accuracy Trajectory (P0 vs P1 vs P2, % scale, seed 42)
Chart 2: Average Incremental Accuracy Comparison (Paper, Official seed 0, Historical Ours seed 42, C1 P0/P1/P2 seed 42)
Chart 3: Forgetting Comparison (Dual Subplots: Paper Eq 17-18 past-tasks vs Official 10-task code formula)
Chart 4: Projection Initialization Effects (Multi-panel: Pre-sigmoid Std, Sigmoid Saturation %, and Accuracy %)
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Set UTF-8 encoding
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

out_dir = Path("results/c1_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

# Set global plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})

# ==============================================================================
# CHART 1: C1 Stage Accuracy Trajectory
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
stages = list(range(1, 11))
p0_stages = [89.60, 88.00, 85.10, 81.55, 79.10, 76.93, 75.76, 74.03, 72.30, 70.95]
p1_stages = [92.70, 91.60, 88.60, 85.62, 83.74, 81.50, 80.06, 77.76, 76.39, 74.84]
p2_stages = [93.40, 91.60, 88.50, 85.58, 83.70, 81.62, 80.01, 78.00, 76.34, 74.99]

ax.plot(stages, p0_stages, marker="o", color="#d9534f", linewidth=2.0, label="P0: Gaussian std=1 (Avg=79.33%, Last=70.95%)")
ax.plot(stages, p1_stages, marker="s", color="#0275d8", linewidth=2.0, linestyle="--", label="P1: Official Uniform (Avg=83.28%, Last=74.84%)")
ax.plot(stages, p2_stages, marker="^", color="#5cb85c", linewidth=2.0, linestyle=":", label="P2: Matched Gaussian (Avg=83.37%, Last=74.99%)")

ax.set_title("Chart 1 — C1 Stage Accuracy Trajectory across 10 Tasks (CIFAR-100, seed 42)", fontweight="bold")
ax.set_xlabel("Incremental Stage / Task")
ax.set_ylabel("Stage Accuracy (%)")
ax.set_xticks(stages)
ax.set_ylim(65, 96)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
plt.tight_layout()
chart1_path = out_dir / "chart1_c1_stage_accuracy.png"
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"Saved: {chart1_path}")

# ==============================================================================
# CHART 2: Average Incremental Accuracy Comparison (Across 6 Benchmarks)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
benchmarks = [
    "Paper\n(3-run mean)",
    "Official Run\n(seed 0, n=1)",
    "Historical Ours\n(seed 42, n=1)",
    "C1: P0 Gaussian\n(seed 42, n=1)",
    "C1: P1 Uniform\n(seed 42, n=1)",
    "C1: P2 Matched G\n(seed 42, n=1)",
]
avg_accs = [91.10, 90.98, 79.33, 79.33, 83.28, 83.37]
last_accs = [86.50, 85.84, 70.92, 70.95, 74.84, 74.99]

x = np.arange(len(benchmarks))
width = 0.35

rects1 = ax.bar(x - width/2, avg_accs, width, label="Average Incremental Acc (A_avg %)", color="#2b5c8f")
rects2 = ax.bar(x + width/2, last_accs, width, label="Last Task Acc (A_last %)", color="#e07a5f")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Chart 2 — Accuracy Comparison across Paper, Official, and Reproduction Runs\n(Note: Official uses seed 0; Ours/C1 use seed 42. Non-controlled across seeds)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks)
ax.set_ylim(60, 100)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="upper right")

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
chart2_path = out_dir / "chart2_average_accuracy_comparison.png"
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved: {chart2_path}")

# ==============================================================================
# CHART 3: Forgetting Comparison (Dual Subplots: Past Tasks vs Official Code)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

runs_fgt = ["Official (s0)", "Hist Ours (s42)", "C1 P0 (s42)", "C1 P1 (s42)", "C1 P2 (s42)"]
fgt_past = [5.17, 9.21, 9.17, 8.97, 8.81]
fgt_code = [4.69, 8.29, 8.25, 8.07, 7.93]

colors = ["#457b9d", "#e63946", "#e63946", "#2a9d8f", "#2a9d8f"]

bars1 = ax1.bar(runs_fgt, fgt_past, color=colors, width=0.55)
ax1.set_title("Formula 1: Paper Eq. (17–18)\n(Denominator = T-1 = 9 past tasks)", fontweight="bold")
ax1.set_ylabel("Forgetting (%) [Lower is better]")
ax1.set_ylim(0, 11)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.tick_params(axis="x", rotation=15)

for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

bars2 = ax2.bar(runs_fgt, fgt_code, color=colors, width=0.55)
ax2.set_title("Formula 2: Official Code\n(Denominator = T = 10 tasks, clamped)", fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.tick_params(axis="x", rotation=15)

for b in bars2:
    h = b.get_height()
    ax2.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.suptitle("Chart 3 — Forgetting Metric Comparison under Both Mathematical Formulations", fontsize=12, fontweight="bold")
plt.tight_layout()
chart3_path = out_dir / "chart3_forgetting_comparison.png"
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"Saved: {chart3_path}")

# ==============================================================================
# CHART 4: Projection Initialization Effects (Multi-panel)
# ==============================================================================
fig, (ax_u, ax_sat, ax_acc) = plt.subplots(1, 3, figsize=(12, 4.5))

conds = ["P0 (std=1)", "P1 (Uniform)", "P2 (Matched G)"]
cond_colors = ["#d9534f", "#0275d8", "#5cb85c"]

u_std = [19.2844, 0.3979, 0.4018]
sat_pct = [81.03, 0.00, 0.00]
avg_acc = [79.33, 83.28, 83.37]

# Panel 1: Pre-sigmoid std
ax_u.bar(conds, u_std, color=cond_colors, width=0.5)
ax_u.set_title("Pre-sigmoid Std: std(hP)\n[Signal magnitude]", fontweight="bold")
ax_u.set_ylabel("Standard Deviation")
ax_u.grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(u_std):
    ax_u.text(i, v + 0.4, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold")

# Panel 2: Sigmoid saturation %
ax_sat.bar(conds, sat_pct, color=cond_colors, width=0.5)
ax_sat.set_title("Sigmoid Saturation (%)\n[z < 0.01 or z > 0.99]", fontweight="bold")
ax_sat.set_ylabel("Saturation Percentage (%)")
ax_sat.set_ylim(0, 100)
ax_sat.grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(sat_pct):
    ax_sat.text(i, v + 2.0, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

# Panel 3: Incremental Accuracy
ax_acc.bar(conds, avg_acc, color=cond_colors, width=0.5)
ax_acc.set_title("Average Incr. Accuracy (%)\n[Downstream RLS performance]", fontweight="bold")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.set_ylim(70, 90)
ax_acc.grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(avg_acc):
    ax_acc.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=8.5, fontweight="bold")

plt.suptitle("Chart 4 — Mechanism Analysis: Impact of Projection Variance on Saturation & Performance", fontsize=12, fontweight="bold")
plt.tight_layout()
chart4_path = out_dir / "chart4_projection_effects.png"
plt.savefig(chart4_path, dpi=300)
plt.close()
print(f"Saved: {chart4_path}")

print("\nAll 4 standardized comparison charts generated successfully in results/c1_analysis/!")
