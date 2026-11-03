"""Generate publication-quality figures for Table 4 Ablation Study."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

analysis_dir = Path("results/table4_analysis")
analysis_dir.mkdir(parents=True, exist_ok=True)

# Load data
with open(analysis_dir / "table4_metrics_comparison.json", "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

with open(analysis_dir / "table4_factorial_effects.json", "r", encoding="utf-8") as f:
    fact_data = json.load(f)

# Colors
c_a0 = '#1f77b4' # Full
c_a1 = '#ff7f0e' # w/o FF
c_a2 = '#2ca02c' # w/o SP
c_a3 = '#d62728' # w/o Both

# -------------------------------------------------------------
# Figure 1: Average Incremental Accuracy Comparison across A0..A3
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
conds = ['A0 (Full F-OAL)\n[C3-D Reused]', 'A1 (w/o FF)\n[New GPU]', 'A2 (w/o SP)\n[New GPU]', 'A3 (w/o Both)\n[New GPU]']
accs = [metrics_data[k]['a_bar_pct'] for k in ['A0', 'A1', 'A2', 'A3']]
colors = [c_a0, c_a1, c_a2, c_a3]

bars = ax.bar(conds, accs, color=colors, width=0.55, edgecolor='black', linewidth=1.2, zorder=3)
ax.set_ylim(88.0, 91.5)
ax.set_ylabel('Average Incremental Accuracy $\\bar{A}$ (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 1: F-OAL Table 4 Architectural Ablations on CIFAR-100 (Seed 42)', fontsize=13, fontweight='bold', pad=15)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(analysis_dir / "fig1_table4_accuracy_comparison.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 2: Paper Table 4 vs Our Reproduction
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(4)
width = 0.35

paper_vals = [91.1, 90.6, 90.7, 90.6]
our_vals = [metrics_data[k]['a_bar_pct'] for k in ['A0', 'A1', 'A2', 'A3']]
labels = ['Full F-OAL', 'w/o Feature Fusion', 'w/o Smooth Projection', 'w/o Both']

rects1 = ax.bar(x - width/2, paper_vals, width, label='Paper Table 4 (NeurIPS 2024)', color='#4a7c59', edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, our_vals, width, label='Our Reproduction (Seed 42)', color='#33658a', edgecolor='black', linewidth=1)

ax.set_ylim(88.0, 92.5)
ax.set_ylabel('Average Accuracy $\\bar{A}$ (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 2: Paper Table 4 vs. Our Controlled Reproduction (CIFAR-100)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11, loc='upper right')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(analysis_dir / "fig2_paper_vs_our_table4.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 3: 2x2 Factorial Interaction Plot (FF x SP)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
# X-axis: Smooth Projection (OFF vs ON)
# Lines: Feature Fusion (OFF vs ON)
sp_states = ['Smooth Projection OFF', 'Smooth Projection ON']
ff_on_line = [metrics_data['A2']['a_bar_pct'], metrics_data['A0']['a_bar_pct']] # A2 (FF on, SP off) -> A0 (FF on, SP on)
ff_off_line = [metrics_data['A3']['a_bar_pct'], metrics_data['A1']['a_bar_pct']] # A3 (FF off, SP off) -> A1 (FF off, SP on)

ax.plot(sp_states, ff_on_line, marker='o', markersize=10, linewidth=2.5, color='#1f77b4', label='Feature Fusion ON', zorder=4)
ax.plot(sp_states, ff_off_line, marker='s', markersize=10, linewidth=2.5, color='#ff7f0e', linestyle='--', label='Feature Fusion OFF', zorder=4)

ax.set_ylabel('Average Incremental Accuracy $\\bar{A}$ (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 3: Feature Fusion $\\times$ Smooth Projection Interaction Plot', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(89.5, 90.8)
ax.legend(fontsize=11, loc='upper left')

# Annotate points
ax.annotate(f'A2: {ff_on_line[0]:.2f}%', (0, ff_on_line[0]), textcoords="offset points", xytext=(-20, 8), fontweight='bold')
ax.annotate(f'A0: {ff_on_line[1]:.2f}%', (1, ff_on_line[1]), textcoords="offset points", xytext=(-20, 8), fontweight='bold')
ax.annotate(f'A3: {ff_off_line[0]:.2f}%', (0, ff_off_line[0]), textcoords="offset points", xytext=(-20, -15), fontweight='bold')
ax.annotate(f'A1: {ff_off_line[1]:.2f}%', (1, ff_off_line[1]), textcoords="offset points", xytext=(-20, -15), fontweight='bold')

plt.tight_layout()
fig.savefig(analysis_dir / "fig3_ff_sp_interaction.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 4: Factorial Main Effects and Interaction Contrast
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
effects = [
    fact_data['main_effects']['E_FF_main'],
    fact_data['main_effects']['E_SP_main'],
    fact_data['interaction']['interaction_contrast'],
    fact_data['joint_benefit']['delta_full_vs_none']
]
effect_labels = [
    'Feature Fusion\nMain Effect ($E_{FF}$)',
    'Smooth Projection\nMain Effect ($E_{SP}$)',
    'Interaction Contrast\n($I_{FF \\times SP}$)',
    'Joint Benefit\n($A_0 - A_3$)'
]
colors = ['#2ca02c', '#17becf', '#e377c2', '#9467bd']

bars = ax.bar(effect_labels, effects, color=colors, width=0.5, edgecolor='black', linewidth=1.2, zorder=3)
ax.axhline(0, color='black', linewidth=1, linestyle='-', zorder=2)
ax.set_ylabel('Effect Size (Percentage Points, pp)', fontsize=12, fontweight='bold')
ax.set_title('Figure 4: Factorial Main Effects & Interaction Contrasts on $\\bar{A}$', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(-0.25, 0.60)

for bar in bars:
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    offset = 4 if h >= 0 else -12
    ax.annotate(f'{h:+.4f} pp',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center', va=va, fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(analysis_dir / "fig4_factorial_effects.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 5: Stage Accuracy Trajectories (1..10)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
stages = list(range(1, 11))

ax.plot(stages, [v*100 for v in metrics_data['A0']['stage_accuracies']], marker='o', linewidth=2.2, color=c_a0, label='A0 (Full F-OAL)')
ax.plot(stages, [v*100 for v in metrics_data['A1']['stage_accuracies']], marker='s', linewidth=2.2, color=c_a1, label='A1 (w/o FF)')
ax.plot(stages, [v*100 for v in metrics_data['A2']['stage_accuracies']], marker='^', linewidth=2.2, color=c_a2, label='A2 (w/o SP)')
ax.plot(stages, [v*100 for v in metrics_data['A3']['stage_accuracies']], marker='d', linewidth=2.2, color=c_a3, label='A3 (w/o Both)')

ax.set_xlabel('Incremental Stage (10 Classes Added per Stage)', fontsize=12, fontweight='bold')
ax.set_ylabel('Stage Accuracy $A_t$ (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 5: Incremental Stage Accuracy Trajectories across 10 Tasks', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(stages)
ax.set_ylim(84.0, 97.5)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
fig.savefig(analysis_dir / "fig5_stage_accuracy_trajectories.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 6: Forgetting Metrics Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4)
width = 0.35

f_past_vals = [metrics_data[k]['f_past_pct'] for k in ['A0', 'A1', 'A2', 'A3']]
f_code_vals = [metrics_data[k]['f_code_pct'] for k in ['A0', 'A1', 'A2', 'A3']]
cond_labels = ['A0 (Full)', 'A1 (w/o FF)', 'A2 (w/o SP)', 'A3 (w/o Both)']

rects1 = ax.bar(x - width/2, f_past_vals, width, label='Paper-style Forgetting $F_{10}^{past}$ (9 tasks)', color='#e76f51', edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, f_code_vals, width, label='Official-code Forgetting $F_{10}^{code}$ (10 tasks)', color='#f4a261', edgecolor='black', linewidth=1)

ax.set_ylabel('Forgetting Rate (%) [Lower is Better]', fontsize=12, fontweight='bold')
ax.set_title('Figure 6: Continual Learning Forgetting Rates Across Ablations', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(cond_labels, fontsize=11)
ax.set_ylim(0.0, 7.0)
ax.legend(fontsize=10, loc='upper right')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(analysis_dir / "fig6_forgetting_comparison.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 7: Computational Efficiency: RLS Inversion Duration
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
rls_times = [metrics_data[k]['rls_eval_sec'] for k in ['A0', 'A1', 'A2', 'A3']]
dims = [1000, 1000, 768, 768]
labels = [f"A0 (Full, D=1000)\n{rls_times[0]:.1f}s", f"A1 (w/o FF, D=1000)\n{rls_times[1]:.1f}s", f"A2 (w/o SP, D=768)\n{rls_times[2]:.1f}s", f"A3 (w/o Both, D=768)\n{rls_times[3]:.1f}s"]

bars = ax.bar(labels, rls_times, color=['#3a86ff', '#8338ec', '#ff006e', '#fb5607'], width=0.5, edgecolor='black', linewidth=1.2)
ax.set_ylabel('RLS Inversion & Eval Duration (s)', fontsize=12, fontweight='bold')
ax.set_title('Figure 7: Computational Cost vs. RLS Feature Dimension', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 20)

for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.2f} s', (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(analysis_dir / "fig7_runtime_comparison.png")
plt.close(fig)

print("All 7 scientific figures successfully generated in results/table4_analysis/!")
