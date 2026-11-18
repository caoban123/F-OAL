"""Generate publication-quality charts for DTD Ablation Study matching W&B report layout.

Reads directly from the official decoupled 3-seed dataset in results/dtd/
(Seeds 42, 123, 2026; 1,600 train / 1,600 test samples, 40 samples/class).
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DTD_RESULTS_DIR = Path("results/dtd")
out_dir = Path("docs/F-OAL _ Thí nghiệm loại bỏ thành phần trên DTD/charts")
out_dir.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("A0", "A0 (Full F-OAL)", "a0", "dtd_a0_full", 82.8),
    ("A1", "A1 (w/o FF)", "a1", "dtd_a1_without_ff", 80.5),
    ("A2", "A2 (w/o SP)", "a2", "dtd_a2_without_sp", 79.3),
    ("A3", "A3 (w/o Both)", "a3", "dtd_a3_without_both", 71.2),
]
SEEDS = [42, 123, 2026]

labels_short = ["A0\n(Full)", "A1\n(-FF)", "A2\n(-SP)", "A3\n(-Both)"]
paper_vals = [c[4] for c in CONDITIONS]

our_vals = []
our_stds = []
f_past_vals = []
f_past_stds = []
f_code_vals = []
f_code_stds = []
trajectories = {}
trajectories_std = {}

for cid, label, prefix, base_folder, _ in CONDITIONS:
    abars = []
    fpasts = []
    fcodes = []
    stage_rows = []
    for s in SEEDS:
        rdir = DTD_RESULTS_DIR / (base_folder if s == 42 else f"{prefix}_seed{s}")
        m = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        abars.append(m["average_incremental_accuracy"] * 100.0)
        fpasts.append(m["forgetting_past_tasks"] * 100.0)
        fcodes.append(m["forgetting_official_code"] * 100.0)
        stage_rows.append([v * 100.0 for v in m["stage_accuracies"]])
    our_vals.append(float(np.mean(abars)))
    our_stds.append(float(np.std(abars, ddof=1)))
    f_past_vals.append(float(np.mean(fpasts)))
    f_past_stds.append(float(np.std(fpasts, ddof=1)))
    f_code_vals.append(float(np.mean(fcodes)))
    f_code_stds.append(float(np.std(fcodes, ddof=1)))
    trajectories[label] = np.mean(stage_rows, axis=0)
    trajectories_std[label] = np.std(stage_rows, axis=0, ddof=1)

# --- Chart 1: Bar chart comparing Paper Target vs Our Official Multi-Seed Reproduction ---
plt.figure(figsize=(7.2, 4.6), dpi=300)
x = np.arange(len(labels_short))
width = 0.35

plt.bar(x - width/2, paper_vals, width, label='Paper Target (Table 4)', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=0.8)
plt.bar(x + width/2, our_vals, width, yerr=our_stds, capsize=4, label='Official Reproduction (N=3 Seeds: 42, 123, 2026)', color='#50E3C2', alpha=0.9, edgecolor='black', linewidth=0.8)

for i in range(len(x)):
    plt.text(x[i] - width/2, paper_vals[i] + 0.6, f"{paper_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    txt = f"{our_vals[i]:.2f}%\n±{our_stds[i]:.2f}"
    plt.text(x[i] + width/2, our_vals[i] + our_stds[i] + 0.5, txt, ha='center', va='bottom', fontsize=8.0, fontweight='bold')

plt.ylabel('Average Incremental Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('DTD Table 4 Ablation: Paper Target vs. Official 3-Seed Mean', fontsize=12, fontweight='bold')
plt.xticks(x, labels_short, fontsize=10)
plt.ylim(65, 95)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
plt.savefig(out_dir / "Section-6-Panel-0-accuracy-comparison.png")
plt.close()

# --- Chart 2: Stage Accuracy Progression Trajectories (3-seed mean +/- std band) ---
plt.figure(figsize=(7.2, 4.6), dpi=300)
stages = np.arange(1, 11)
markers = ['o', 's', '^', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for (name, traj), m, c in zip(trajectories.items(), markers, colors):
    std_band = trajectories_std[name]
    plt.plot(stages, traj, marker=m, linewidth=2, markersize=5, label=name, color=c)
    plt.fill_between(stages, traj - std_band, traj + std_band, color=c, alpha=0.14)

plt.xlabel('Incremental Stage (1 to 10 Tasks)', fontsize=11, fontweight='bold')
plt.ylabel('Stage Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('Stage Accuracy Trajectories across 10 Tasks (Mean ± Std, N=3 Seeds)', fontsize=11, fontweight='bold')
plt.xticks(stages)
plt.ylim(55, 102)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')
plt.tight_layout()
plt.savefig(out_dir / "Section-6-Panel-1-stage-trajectories.png")
plt.close()

# --- Chart 3: 2x2 Factorial Interaction Chart (Mean +/- Std) ---
plt.figure(figsize=(6.2, 4.5), dpi=300)
plt.errorbar(["SP OFF", "SP ON"], [our_vals[3], our_vals[1]], yerr=[our_stds[3], our_stds[1]], marker='o', capsize=4, linewidth=2.2, label='FF OFF', color='#d62728')
plt.errorbar(["SP OFF", "SP ON"], [our_vals[2], our_vals[0]], yerr=[our_stds[2], our_stds[0]], marker='s', capsize=4, linewidth=2.2, label='FF ON', color='#1f77b4')

plt.ylabel('Average Incremental Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('2x2 Factorial Interaction: FF & SP (Mean ± Std, N=3)', fontsize=11.5, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(70, 90)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()
plt.savefig(out_dir / "Section-11-Panel-0-factorial-interaction.png")
plt.close()

# --- Chart 4: Forgetting Metrics Comparison (3-seed mean +/- std) ---
plt.figure(figsize=(6.5, 4.6), dpi=300)
plt.bar(x - width/2, f_past_vals, width, yerr=f_past_stds, capsize=3, label='F_past (Eq. 8, Peak)', color='#FF9F43', alpha=0.9, edgecolor='black', linewidth=0.8)
plt.bar(x + width/2, f_code_vals, width, yerr=f_code_stds, capsize=3, label='F_code (Eq. 9, Post-task)', color='#EE5253', alpha=0.9, edgecolor='black', linewidth=0.8)

for i in range(len(x)):
    plt.text(x[i] - width/2, f_past_vals[i] + f_past_stds[i] + 0.15, f"{f_past_vals[i]:.2f}%", ha='center', va='bottom', fontsize=8)
    plt.text(x[i] + width/2, f_code_vals[i] + f_code_stds[i] + 0.15, f"{f_code_vals[i]:.2f}%", ha='center', va='bottom', fontsize=8)

plt.ylabel('Forgetting Rate (%)', fontsize=11, fontweight='bold')
plt.title('Forgetting Comparison across Ablation Conditions (N=3 Mean ± Std)', fontsize=11, fontweight='bold')
plt.xticks(x, labels_short, fontsize=10)
plt.ylim(0, 10.8)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
plt.savefig(out_dir / "Section-11-Panel-1-forgetting-comparison.png")
plt.close()

print("Successfully regenerated all 4 charts in docs/F-OAL _ Thí nghiệm loại bỏ thành phần trên DTD/charts/ from results/dtd/.")
