"""Generate publication-quality charts for DTD Ablation Study matching W&B report layout.

Updated with exact 1,600 samples (40 samples/class) and multi-seed (N=3) results.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path("docs/F-OAL _ Thí nghiệm loại bỏ thành phần trên DTD/charts")
out_dir.mkdir(parents=True, exist_ok=True)

# Data from verified 1,600-sample multi-seed runs (Seeds 42, 123, 2026)
conditions = ["A0 (Full F-OAL)", "A1 (w/o FF)", "A2 (w/o SP)", "A3 (w/o Both)"]
labels_short = ["A0\n(Full)", "A1\n(-FF)", "A2\n(-SP)", "A3\n(-Both)"]
paper_vals = [82.8, 80.5, 79.3, 71.2]
our_vals = [85.13, 81.91, 78.64, 72.97]
our_stds = [0.11, 0.56, 0.00, 0.00]
a10_vals = [77.54, 74.69, 73.62, 71.69]
f_past_vals = [7.31, 6.74, 6.87, 4.93]
f_code_vals = [6.62, 6.06, 6.50, 5.06]

trajectories = {
    "A0 (Full)": [98.75, 92.81, 92.71, 88.75, 85.00, 81.46, 79.20, 78.44, 78.19, 77.25],
    "A1 (w/o FF)": [95.00, 90.00, 88.75, 84.53, 81.75, 79.48, 78.21, 77.42, 75.69, 74.69],
    "A2 (w/o SP)": [98.75, 91.56, 88.33, 77.81, 69.75, 69.06, 70.62, 73.44, 73.47, 73.62],
    "A3 (w/o Both)": [95.00, 86.56, 81.67, 68.44, 57.25, 63.44, 65.36, 69.06, 71.25, 71.69],
}

# --- Chart 1: Bar chart comparing Paper Target vs Our Reproduction ---
plt.figure(figsize=(7, 4.5), dpi=300)
x = np.arange(len(labels_short))
width = 0.35

plt.bar(x - width/2, paper_vals, width, label='Paper Target (Table 4)', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=0.8)
plt.bar(x + width/2, our_vals, width, yerr=our_stds, capsize=4, label='Our Multi-Seed (N=3, 1,600 samples)', color='#50E3C2', alpha=0.9, edgecolor='black', linewidth=0.8)

for i in range(len(x)):
    plt.text(x[i] - width/2, paper_vals[i] + 0.8, f"{paper_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    txt = f"{our_vals[i]:.2f}%" if our_stds[i] == 0 else f"{our_vals[i]:.2f}%\n±{our_stds[i]:.2f}"
    plt.text(x[i] + width/2, our_vals[i] + 0.8, txt, ha='center', va='bottom', fontsize=8.0, fontweight='bold')

plt.ylabel('Average Incremental Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('DTD Table 4 Ablation: Paper Target vs. Multi-Seed Reproduction', fontsize=12, fontweight='bold')
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
plt.title('Stage Accuracy Progression Trajectories across 10 Tasks (160 samples/task)', fontsize=11, fontweight='bold')
plt.xticks(stages)
plt.ylim(50, 102)
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
plt.ylim(70, 88)
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
plt.title('Forgetting Comparison across Ablation Conditions (40 samples/class)', fontsize=11, fontweight='bold')
plt.xticks(x, labels_short, fontsize=10)
plt.ylim(0, 9.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
plt.savefig(out_dir / "Section-11-Panel-1-forgetting-comparison.png")
plt.close()

print("Successfully regenerated all 4 charts in docs/F-OAL _ Thí nghiệm loại bỏ thành phần trên DTD/charts/")
