"""Generate official DTD Table 4 summary, factorial decomposition, and plots from results/dtd/.

Reads directly from the 12 canonical decoupled GPU runs in results/dtd/:
  - Seed 42:   dtd_a0_full, dtd_a1_without_ff, dtd_a2_without_sp, dtd_a3_without_both
  - Seed 123:  a0_seed123, a1_seed123, a2_seed123, a3_seed123
  - Seed 2026: a0_seed2026, a1_seed2026, a2_seed2026, a3_seed2026

Outputs:
  - results/dtd/dtd_table4_summary.json
  - results/dtd_table4_analysis/* (CSVs, JSONs, manifest, and 4 PNG figures)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DTD_RESULTS_DIR = Path("results/dtd")
ANALYSIS_DIR = Path("results/dtd_table4_analysis")

CONDITIONS = [
    ("A0", "a0", "dtd_a0_full", "Full F-OAL", True, True, 1000, 82.8),
    ("A1", "a1", "dtd_a1_without_ff", "w/o Feature Fusion", False, True, 1000, 80.5),
    ("A2", "a2", "dtd_a2_without_sp", "w/o Smooth Projection", True, False, 768, 79.3),
    ("A3", "a3", "dtd_a3_without_both", "w/o Both", False, False, 768, 71.2),
]

SEEDS = [42, 123, 2026]


def run_folder_for(prefix: str, base_folder: str, seed: int) -> Path:
    if seed == 42:
        return DTD_RESULTS_DIR / base_folder
    return DTD_RESULTS_DIR / f"{prefix}_seed{seed}"


def load_canonical_runs() -> dict[str, dict[int, dict[str, Any]]]:
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    for cid, prefix, base_folder, *_ in CONDITIONS:
        runs[cid] = {}
        for seed in SEEDS:
            rdir = run_folder_for(prefix, base_folder, seed)
            metrics = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
            meta = json.loads((rdir / "metadata.json").read_text(encoding="utf-8"))
            runs[cid][seed] = {
                "dir": str(rdir).replace("\\", "/"),
                "metrics": metrics,
                "metadata": meta,
            }
    return runs


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def compute_factorial_dict(a0: float, a1: float, a2: float, a3: float) -> dict[str, float]:
    e_ff_sp = a0 - a1
    e_ff_nosp = a2 - a3
    e_sp_ff = a0 - a2
    e_sp_noff = a1 - a3
    e_ff = 0.5 * (e_ff_sp + e_ff_nosp)
    e_sp = 0.5 * (e_sp_ff + e_sp_noff)
    i_did = e_ff_sp - e_ff_nosp
    i_half = 0.5 * i_did
    beta = i_did / 4.0
    joint = a0 - a3
    return {
        "E_FF|SP": e_ff_sp,
        "E_FF|~SP": e_ff_nosp,
        "E_SP|FF": e_sp_ff,
        "E_SP|~FF": e_sp_noff,
        "E_FF": e_ff,
        "E_SP": e_sp,
        "I_FFxSP_DiD": i_did,
        "I_FFxSP_half": i_half,
        "beta_FFxSP": beta,
        "Delta_full_none": joint,
    }


def generate_outputs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_canonical_runs()

    # 1. Update results/dtd/dtd_table4_summary.json
    table4_summary = []
    cond_stats: dict[str, dict[str, Any]] = {}

    for cid, prefix, base_folder, name, ff, sp, dim, paper_abar in CONDITIONS:
        m42 = runs[cid][42]["metrics"]
        m123 = runs[cid][123]["metrics"]
        m2026 = runs[cid][2026]["metrics"]

        abars = [m["average_incremental_accuracy"] * 100.0 for m in (m42, m123, m2026)]
        alasts = [m["last_accuracy"] * 100.0 for m in (m42, m123, m2026)]
        fpasts = [m["forgetting_past_tasks"] * 100.0 for m in (m42, m123, m2026)]
        fcodes = [m["forgetting_official_code"] * 100.0 for m in (m42, m123, m2026)]

        abar_mean, abar_std = mean_std(abars)
        alast_mean, alast_std = mean_std(alasts)
        fpast_mean, fpast_std = mean_std(fpasts)
        fcode_mean, fcode_std = mean_std(fcodes)

        stage_stack = np.array([m["stage_accuracies"] for m in (m42, m123, m2026)]) * 100.0
        stage_mean = np.mean(stage_stack, axis=0)
        stage_std = np.std(stage_stack, axis=0, ddof=1)

        final_tasks_42 = [m42["task_matrix"][9][j] for j in range(10)]
        final_tasks_mean = np.mean(
            np.array([[m["task_matrix"][9][j] for j in range(10)] for m in (m42, m123, m2026)]),
            axis=0,
        )

        label_map = {
            "A0": "A0 (Full F-OAL)",
            "A1": "A1 (w/o FF)",
            "A2": "A2 (w/o SP)",
            "A3": "A3 (w/o Both)",
        }

        table4_summary.append({
            "condition": label_map[cid],
            "feature_fusion": "ON" if ff else "OFF",
            "smooth_projection": "ON" if sp else "OFF",
            "paper_a_bar": paper_abar,
            "our_a_bar": round(abars[0], 4),
            "delta": round(abars[0] - paper_abar, 4),
            "a_10": round(alasts[0], 4),
            "f_past": round(fpasts[0], 4),
            "f_code": round(fcodes[0], 4),
            "seed42_a_bar": round(abars[0], 4),
            "seed123_a_bar": round(abars[1], 4),
            "seed2026_a_bar": round(abars[2], 4),
            "multiseed_a_bar_mean": round(abar_mean, 4),
            "multiseed_a_bar_std": round(abar_std, 4),
            "multiseed_delta_vs_paper": round(abar_mean - paper_abar, 4),
            "multiseed_a_10_mean": round(alast_mean, 4),
            "multiseed_a_10_std": round(alast_std, 4),
            "multiseed_f_past_mean": round(fpast_mean, 4),
            "multiseed_f_past_std": round(fpast_std, 4),
            "multiseed_f_code_mean": round(fcode_mean, 4),
            "multiseed_f_code_std": round(fcode_std, 4),
        })

        cond_stats[cid] = {
            "condition": cid,
            "name": name,
            "ff": ff,
            "sp": sp,
            "dim": dim,
            "paper_a_bar_pct": paper_abar,
            "seed42": {
                "a_bar_pct": abars[0],
                "a_last_pct": alasts[0],
                "f_past_pct": fpasts[0],
                "f_code_pct": fcodes[0],
                "stage_accuracies": m42["stage_accuracies"],
                "final_task_accuracies": final_tasks_42,
                "total_duration_seconds": m42["total_duration_seconds"],
                "feature_extraction_seconds": m42["feature_extraction_seconds"],
                "rls_and_eval_seconds": m42["rls_and_eval_seconds"],
                "run_dir": runs[cid][42]["dir"],
            },
            "seed123": {
                "a_bar_pct": abars[1],
                "a_last_pct": alasts[1],
                "f_past_pct": fpasts[1],
                "f_code_pct": fcodes[1],
                "stage_accuracies": m123["stage_accuracies"],
                "total_duration_seconds": m123["total_duration_seconds"],
                "feature_extraction_seconds": m123["feature_extraction_seconds"],
                "rls_and_eval_seconds": m123["rls_and_eval_seconds"],
                "run_dir": runs[cid][123]["dir"],
            },
            "seed2026": {
                "a_bar_pct": abars[2],
                "a_last_pct": alasts[2],
                "f_past_pct": fpasts[2],
                "f_code_pct": fcodes[2],
                "stage_accuracies": m2026["stage_accuracies"],
                "total_duration_seconds": m2026["total_duration_seconds"],
                "feature_extraction_seconds": m2026["feature_extraction_seconds"],
                "rls_and_eval_seconds": m2026["rls_and_eval_seconds"],
                "run_dir": runs[cid][2026]["dir"],
            },
            "multiseed_mean": {
                "n": 3,
                "seeds": SEEDS,
                "a_bar_pct": abar_mean,
                "a_bar_std_pct": abar_std,
                "a_last_pct": alast_mean,
                "a_last_std_pct": alast_std,
                "f_past_pct": fpast_mean,
                "f_past_std_pct": fpast_std,
                "f_code_pct": fcode_mean,
                "f_code_std_pct": fcode_std,
                "stage_accuracies_pct": [float(v) for v in stage_mean],
                "stage_accuracies_std_pct": [float(v) for v in stage_std],
                "final_task_accuracies_ratio": [float(v) for v in final_tasks_mean],
            },
        }

    (DTD_RESULTS_DIR / "dtd_table4_summary.json").write_text(
        json.dumps(table4_summary, indent=2), encoding="utf-8"
    )

    # 2. Write table4_metrics_comparison.json and .csv
    (ANALYSIS_DIR / "table4_metrics_comparison.json").write_text(
        json.dumps(cond_stats, indent=2), encoding="utf-8"
    )
    with open(ANALYSIS_DIR / "table4_metrics_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Condition", "Name", "Feature Fusion", "Smooth Projection", "Dimension",
            "Seed 42 A_bar (%)", "Seed 123 A_bar (%)", "Seed 2026 A_bar (%)",
            "Multi-Seed Mean A_bar (%)", "Multi-Seed Std (%)",
            "Multi-Seed Mean A_10 (%)", "Multi-Seed Std A_10 (%)",
            "Multi-Seed Mean F_past (%)", "Multi-Seed Std F_past (%)",
            "Multi-Seed Mean F_code (%)", "Multi-Seed Std F_code (%)",
        ])
        for cid, *_ in CONDITIONS:
            d = cond_stats[cid]
            ms = d["multiseed_mean"]
            w.writerow([
                cid, d["name"], d["ff"], d["sp"], d["dim"],
                f"{d['seed42']['a_bar_pct']:.2f}%",
                f"{d['seed123']['a_bar_pct']:.2f}%",
                f"{d['seed2026']['a_bar_pct']:.2f}%",
                f"{ms['a_bar_pct']:.2f}%", f"{ms['a_bar_std_pct']:.2f}%",
                f"{ms['a_last_pct']:.2f}%", f"{ms['a_last_std_pct']:.2f}%",
                f"{ms['f_past_pct']:.2f}%", f"{ms['f_past_std_pct']:.2f}%",
                f"{ms['f_code_pct']:.2f}%", f"{ms['f_code_std_pct']:.2f}%",
            ])

    # 3. Factorial Effects (per-seed + 3-seed mean +/- std)
    per_seed_fact = {}
    for seed in SEEDS:
        key = f"seed{seed}"
        per_seed_fact[seed] = compute_factorial_dict(
            cond_stats["A0"][key]["a_bar_pct"],
            cond_stats["A1"][key]["a_bar_pct"],
            cond_stats["A2"][key]["a_bar_pct"],
            cond_stats["A3"][key]["a_bar_pct"],
        )

    effect_meta = [
        ("Feature Fusion Benefit (SP ON)", "E_FF|SP"),
        ("Feature Fusion Benefit (SP OFF)", "E_FF|~SP"),
        ("Smooth Projection Benefit (FF ON)", "E_SP|FF"),
        ("Smooth Projection Benefit (FF OFF)", "E_SP|~FF"),
        ("Average Feature Fusion Main Effect", "E_FF"),
        ("Average Smooth Projection Main Effect", "E_SP"),
        ("Difference-in-Differences Interaction Contrast", "I_FFxSP_DiD"),
        ("Half-Interaction Contrast (0.5 * I_DiD)", "I_FFxSP_half"),
        ("Interaction Regression Coefficient (+-1 coding)", "beta_FFxSP"),
        ("Joint Benefit (Full F-OAL vs Without Both)", "Delta_full_none"),
    ]
    fact_rows = []
    for desc, sym in effect_meta:
        vals = [per_seed_fact[s][sym] for s in SEEDS]
        m, sd = mean_std(vals)
        fact_rows.append({
            "Effect Description": desc,
            "Symbol": sym,
            "Seed 42 (pp)": f"{vals[0]:+.4f}",
            "Seed 123 (pp)": f"{vals[1]:+.4f}",
            "Seed 2026 (pp)": f"{vals[2]:+.4f}",
            "Mean (pp)": f"{m:+.4f}",
            "Std (pp)": f"{sd:.4f}",
        })

    (ANALYSIS_DIR / "table4_factorial_effects.json").write_text(
        json.dumps(fact_rows, indent=2), encoding="utf-8"
    )
    with open(ANALYSIS_DIR / "table4_factorial_effects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Effect Description", "Symbol", "Seed 42 (pp)", "Seed 123 (pp)", "Seed 2026 (pp)", "Mean (pp)", "Std (pp)"])
        for row in fact_rows:
            w.writerow([row[k] for k in ["Effect Description", "Symbol", "Seed 42 (pp)", "Seed 123 (pp)", "Seed 2026 (pp)", "Mean (pp)", "Std (pp)"]])

    # 4. Stage Accuracies & Final Task Accuracies CSVs (Seed 42 + 3-seed mean)
    with open(ANALYSIS_DIR / "table4_stage_accuracies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "A0_seed42", "A1_seed42", "A2_seed42", "A3_seed42", "A0_mean", "A1_mean", "A2_mean", "A3_mean"])
        for s in range(10):
            w.writerow([
                s + 1,
                *[cond_stats[c]["seed42"]["stage_accuracies"][s] for c in ("A0", "A1", "A2", "A3")],
                *[cond_stats[c]["multiseed_mean"]["stage_accuracies_pct"][s] / 100.0 for c in ("A0", "A1", "A2", "A3")],
            ])

    with open(ANALYSIS_DIR / "table4_final_task_accuracies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "A0_seed42", "A1_seed42", "A2_seed42", "A3_seed42", "A0_mean", "A1_mean", "A2_mean", "A3_mean"])
        for t in range(10):
            w.writerow([
                t + 1,
                *[cond_stats[c]["seed42"]["final_task_accuracies"][t] for c in ("A0", "A1", "A2", "A3")],
                *[cond_stats[c]["multiseed_mean"]["final_task_accuracies_ratio"][t] for c in ("A0", "A1", "A2", "A3")],
            ])

    # 5. Paper Comparison CSV
    with open(ANALYSIS_DIR / "table4_paper_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Condition", "Paper Table 4 DTD (%)", "Seed 42 DTD (%)", "Seed 42 Delta (pp)",
            "Multi-Seed Mean DTD (%)", "Multi-Seed Std (%)", "Multi-Seed Delta (pp)",
        ])
        for cid, *_ in CONDITIONS:
            d = cond_stats[cid]
            p = d["paper_a_bar_pct"]
            s42 = d["seed42"]["a_bar_pct"]
            ms_m = d["multiseed_mean"]["a_bar_pct"]
            ms_sd = d["multiseed_mean"]["a_bar_std_pct"]
            w.writerow([
                cid, f"{p:.1f}%", f"{s42:.2f}%", f"{s42 - p:+.2f} pp",
                f"{ms_m:.2f}%", f"{ms_sd:.2f}%", f"{ms_m - p:+.2f} pp",
            ])

    # 6. Runtime Comparison CSV (from all 12 runs on Tesla T4 GPU)
    with open(ANALYSIS_DIR / "table4_runtime_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Condition", "Seed", "Hardware", "Total Wall-Clock (s)",
            "Feature Extraction Recorded (s)", "RLS & Eval (s)", "Cache Status",
        ])
        for cid, *_ in CONDITIONS:
            for seed in SEEDS:
                skey = f"seed{seed}"
                sd = cond_stats[cid][skey]
                cache_status = "Reused cached ViT features" if cid in ("A2", "A3") else "Extracted & cached ViT features"
                w.writerow([
                    cid, seed, "NVIDIA Tesla T4 (Google Colab)",
                    f"{sd['total_duration_seconds']:.2f}",
                    f"{sd['feature_extraction_seconds']:.2f}",
                    f"{sd['rls_and_eval_seconds']:.2f}",
                    cache_status,
                ])

    # 7. Evidence Manifest JSON
    evidence_manifest = {
        "canonical_source": "results/dtd/ (12 decoupled runs: A0-A3 x Seeds 42, 123, 2026)",
        "dataset": "DTD (tanganke/dtd)",
        "protocol": "10 tasks x 4 classes (40 of 47 classes), 40 train & 40 test samples/class (1,600 train / 1,600 test), one-pass stream, batch size 10, gamma 1.0, float64",
        "hardware": "NVIDIA Tesla T4 (Google Colab)",
        "software": "Python 3.13.16, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, timm 1.0.31, datasets 4.8.5",
        "git_commit": "ca42e67265e3c6be485113216e51b488bb3775a9",
        "conditions": cond_stats,
        "factorial_effects": fact_rows,
    }
    (ANALYSIS_DIR / "table4_evidence_manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2), encoding="utf-8"
    )

    # 8. Generate 4 PNG Figures in results/dtd_table4_analysis/
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "figure.dpi": 300,
    })

    # Fig 1: Paper vs Multi-Seed Reproduction
    labels_short = ["A0\n(Full)", "A1\n(-FF)", "A2\n(-SP)", "A3\n(-Both)"]
    paper_vals = [cond_stats[c]["paper_a_bar_pct"] for c in ("A0", "A1", "A2", "A3")]
    our_means = [cond_stats[c]["multiseed_mean"]["a_bar_pct"] for c in ("A0", "A1", "A2", "A3")]
    our_stds = [cond_stats[c]["multiseed_mean"]["a_bar_std_pct"] for c in ("A0", "A1", "A2", "A3")]
    seed42_vals = [cond_stats[c]["seed42"]["a_bar_pct"] for c in ("A0", "A1", "A2", "A3")]

    x = np.arange(len(labels_short))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.bar(x - width / 2, paper_vals, width, label="Paper Target (Table 4)", color="#4A90E2", edgecolor="black", linewidth=0.8)
    ax.bar(x + width / 2, our_means, width, yerr=our_stds, capsize=4, label="Official Reproduction (N=3 Seeds: 42, 123, 2026)", color="#2ECC71", alpha=0.9, edgecolor="black", linewidth=0.8)
    for i in range(len(x)):
        ax.text(x[i] - width / 2, paper_vals[i] + 0.6, f"{paper_vals[i]:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(x[i] + width / 2, our_means[i] + our_stds[i] + 0.6, f"{our_means[i]:.2f}%\n±{our_stds[i]:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.set_ylabel("Average Incremental Accuracy (%)", fontweight="bold")
    ax.set_title("DTD Table 4 Ablation: Paper Target vs. Official Decoupled 3-Seed Mean", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short)
    ax.set_ylim(65, 95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "fig1_table4_accuracy_comparison.png", bbox_inches="tight")
    plt.close()

    # Fig 2: 10x10 Accuracy Matrix Heatmap for A0 Seed 42 (canonical reference)
    t_mat = runs["A0"][42]["metrics"]["task_matrix"]
    mat_full = np.full((10, 10), np.nan)
    for r_i, row in enumerate(t_mat):
        for c_i, val in enumerate(row):
            mat_full[r_i, c_i] = val * 100.0
    fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=300)
    sns.heatmap(
        mat_full, mask=np.isnan(mat_full), annot=True, fmt=".1f",
        cmap=sns.color_palette("YlGnBu", as_cmap=True), vmin=60, vmax=100,
        cbar_kws={"label": "Test Accuracy (%)"}, linewidths=0.5, linecolor="white",
        square=True, ax=ax,
    )
    ax.set_title("DTD Official Full F-OAL (A0, Seed 42): 10x10 Stage-by-Task Accuracy Matrix (%)", fontweight="bold", pad=12)
    ax.set_xlabel("Evaluated Task (4 texture classes/task)", labelpad=8)
    ax.set_ylabel("Completed Incremental Stage", labelpad=8)
    ax.set_xticklabels([f"T{i}" for i in range(1, 11)])
    ax.set_yticklabels([f"S{i}" for i in range(1, 11)], rotation=0)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "fig2_accuracy_matrix_heatmap.png", bbox_inches="tight")
    plt.close()

    # Fig 3: 2x2 Factorial Interaction & Main Effects (Mean +/- Std across 3 seeds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=300)
    ax1.errorbar(["SP OFF", "SP ON"], [our_means[3], our_means[1]], yerr=[our_stds[3], our_stds[1]], marker="o", linewidth=2.2, capsize=4, label="FF OFF (A3 -> A1)", color="#d62728")
    ax1.errorbar(["SP OFF", "SP ON"], [our_means[2], our_means[0]], yerr=[our_stds[2], our_stds[0]], marker="s", linewidth=2.2, capsize=4, label="FF ON (A2 -> A0)", color="#1f77b4")
    ax1.set_ylabel("Average Incremental Accuracy (%)", fontweight="bold")
    ax1.set_title("(a) 2x2 Factorial Interaction (Mean ± Std, N=3)", fontweight="bold")
    ax1.set_ylim(70, 90)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)

    eff_keys = ["E_FF", "E_SP", "I_FFxSP_DiD", "Delta_full_none"]
    eff_labels = ["Main FF\n($E_{FF}$)", "Main SP\n($E_{SP}$)", "Interaction\n($I_{DiD}$)", "Joint Drop\n($A0 - A3$)"]
    eff_means = [mean_std([per_seed_fact[s][k] for s in SEEDS])[0] for k in eff_keys]
    eff_stds = [mean_std([per_seed_fact[s][k] for s in SEEDS])[1] for k in eff_keys]
    eff_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]
    bars = ax2.bar(eff_labels, eff_means, yerr=eff_stds, capsize=4, color=eff_colors, edgecolor="black", width=0.52)
    ax2.axhline(0, color="gray", linewidth=0.9)
    ax2.set_ylabel("Effect Magnitude (pp)", fontweight="bold")
    ax2.set_title("(b) Factorial Effects Decomposition (Mean ± Std, N=3)", fontweight="bold")
    ax2.set_ylim(-4.2, 13.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, m_val, s_val in zip(bars, eff_means, eff_stds):
        if m_val >= 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, m_val + s_val + 0.35, f"{m_val:+.2f}\n±{s_val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, m_val - s_val - 0.35, f"{m_val:+.2f}\n±{s_val:.2f}", ha="center", va="top", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "fig3_ff_sp_interaction.png", bbox_inches="tight")
    plt.close()

    # Fig 5: Stage Accuracy Trajectories (Mean +/- Std shaded across 3 seeds)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=300)
    stages = np.arange(1, 11)
    palette = [("A0", "A0 (Full F-OAL)", "#1f77b4", "o-"), ("A1", "A1 (w/o FF)", "#ff7f0e", "s--"), ("A2", "A2 (w/o SP)", "#2ca02c", "^-."), ("A3", "A3 (w/o Both)", "#d62728", "D:")]
    for cid, label, col, style in palette:
        m_arr = np.asarray(cond_stats[cid]["multiseed_mean"]["stage_accuracies_pct"])
        s_arr = np.asarray(cond_stats[cid]["multiseed_mean"]["stage_accuracies_std_pct"])
        ax.plot(stages, m_arr, style, label=label, color=col, linewidth=2.2, markersize=5)
        ax.fill_between(stages, m_arr - s_arr, m_arr + s_arr, color=col, alpha=0.14)
    ax.set_xlabel("Incremental Stage (1 to 10 Tasks, 40 classes total)", fontweight="bold")
    ax.set_ylabel("Stage Accuracy (%)", fontweight="bold")
    ax.set_title("DTD Table 4 Stage Accuracy Trajectories (Mean ± Std across Seeds 42, 123, 2026)", fontweight="bold")
    ax.set_xticks(stages)
    ax.set_ylim(55, 102)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "fig5_stage_accuracy_trajectories.png", bbox_inches="tight")
    plt.close()

    print("Successfully updated results/dtd/dtd_table4_summary.json and results/dtd_table4_analysis/.")


if __name__ == "__main__":
    generate_outputs()
