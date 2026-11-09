"""Generate publication-quality charts for DTD Ablation Study matching W&B report layout."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path("docs/F-OAL _ Thí nghiệm loại bỏ thành phần trên DTD/charts")
out_dir.mkdir(parents=True, exist_ok=True)

# Data from verified Colab runs
conditions = ["A0 (Full F-OAL)", "A1 (w/o FF)", "A2 (w/o SP)", "A3 (w/o Both)"]
labels_short = ["A0\n(Full)", "A1\n(-FF)", "A2\n(-SP)", "A3\n(-Both)"]
paper_vals = [82.8, 80.5, 79.3, 71.2]
our_vals = [87.196, 85.008, 84.348, 81.104]
a10_vals = [79.563, 78.063, 79.125, 77.563]
f_past_vals = [7.986, 7.014, 5.764, 5.556]
f_code_vals = [7.188, 6.313, 5.188, 5.125]

trajectories = {
    "A0 (Full)": [98.75, 92.81, 93.96, 91.56, 87.88, 84.48, 81.88, 81.02, 80.07, 79.56],
    "A1 (w/o FF)": [96.25, 89.69, 90.00, 87.97, 85.62, 83.12, 80.89, 79.92, 78.54, 78.06],
    "A2 (w/o SP)": [98.75, 90.94, 89.17, 87.81, 85.25, 82.92, 80.00, 79.06, 80.14, 79.12],
    "A3 (w/o Both)": [95.62, 77.81, 77.29, 82.81, 82.25, 80.62, 79.02, 79.84, 78.19, 77.56],
}

# --- Chart 1: Bar chart comparing Paper Target vs Our Reproduction ---
plt.figure(figsize=(7, 4.5), dpi=300)
x = np.arange(len(labels_short))
width = 0.35

plt.bar(x - width/2, paper_vals, width, label='Paper Target (Table 4)', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=0.8)
plt.bar(x + width/2, our_vals, width, label='Our Reproduction (Colab GPU)', color='#50E3C2', alpha=0.9, edgecolor='black', linewidth=0.8)

for i in range(len(x)):
    plt.text(x[i] - width/2, paper_vals[i] + 0.8, f"{paper_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    plt.text(x[i] + width/2, our_vals[i] + 0.8, f"{our_vals[i]:.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.ylabel('Average Incremental Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('DTD Table 4 Ablation: Paper Target vs. Reproduction', fontsize=12, fontweight='bold')
plt.xticks(x, labels_short, fontsize=10)
plt.ylim(65, 95)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
plt.savefig(out_dir / "Section-6-Panel-0-accuracy-comparison.png")
plt.close()

# --- Chart 2: Stage Accuracy Progression Trajectories ---
plt.figure(figsize=(7, 4.5), dpi=300)
stages = list(range(1, 11))
markers = ['o', 's', '^', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for (name, traj), m, c in zip(trajectories.items(), markers, colors):
    plt.plot(stages, traj, marker=m, linewidth=2, markersize=5, label=name, color=c)

plt.xlabel('Incremental Stage (1 to 10 Tasks)', fontsize=11, fontweight='bold')
plt.ylabel('Stage Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('Stage Accuracy Progression Trajectories across 10 Tasks', fontsize=12, fontweight='bold')
plt.xticks(stages)
plt.ylim(70, 102)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')
plt.tight_layout()
plt.savefig(out_dir / "Section-6-Panel-1-stage-trajectories.png")
plt.close()

# --- Chart 3: 2x2 Factorial Interaction Chart ---
plt.figure(figsize=(6, 4.5), dpi=300)
# Factorial levels: SP Off vs SP On
plt.plot(["SP OFF", "SP ON"], [our_vals[3], our_vals[1]], marker='o', linewidth=2.2, label='FF OFF', color='#d62728')
plt.plot(["SP OFF", "SP ON"], [our_vals[2], our_vals[0]], marker='s', linewidth=2.2, label='FF ON', color='#1f77b4')

plt.ylabel('Average Incremental Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('2x2 Factorial Interaction: Feature Fusion & Smooth Projection', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(78, 90)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()
plt.savefig(out_dir / "Section-11-Panel-0-factorial-interaction.png")
plt.close()

# --- Chart 4: Forgetting Metrics Comparison ---
plt.figure(figsize=(6, 4.5), dpi=300)
plt.bar(x - width/2, f_past_vals, width, label='F_past (Eq. 8, Peak)', color='#FF9F43', alpha=0.9, edgecolor='black', linewidth=0.8)
plt.bar(x + width/2, f_code_vals, width, label='F_code (Eq. 9, Post-task)', color='#EE5253', alpha=0.9, edgecolor='black', linewidth=0.8)

for i in range(len(x)):
    plt.text(x[i] - width/2, f_past_vals[i] + 0.15, f"{f_past_vals[i]:.2f}%", ha='center', va='bottom', fontsize=8)
    plt.text(x[i] + width/2, f_code_vals[i] + 0.15, f"{f_code_vals[i]:.2f}%", ha='center', va='bottom', fontsize=8)

plt.ylabel('Forgetting Rate (%)', fontsize=11, fontweight='bold')
plt.title('Forgetting Comparison across Ablation Conditions (C0 Metric Parity)', fontsize=12, fontweight='bold')
plt.xticks(x, labels_short, fontsize=10)
plt.ylim(0, 10.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
plt.savefig(out_dir / "Section-11-Panel-1-forgetting-comparison.png")
plt.close()

print("All charts successfully generated in:", out_dir.resolve())
