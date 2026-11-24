"""Generate 300-DPI publication figures for Lab 2 Empirical Demonstration & Portrait A2 Poster.

Reads JSON outputs from `results/lab2_empirical/` across available datasets (`cifar100`, `dtd`,
`aircraft`) and produces standardized academic figures:
1. `fig_e1_longtail_degradation_and_recovery.png`: Tail-Class & Average Accuracy vs `rho` (`Mean +/- Std`).
2. `fig_e1_weight_norm_collapse_vs_paper_fig3.png`: Classifier column L2 norm `||w_c||_2` vs class sample
   count `N_c` (direct empirical refutation of Paper Section 4.6 Figure 3 on imbalanced streams).
3. `fig_e3_quadratic_scaling_reality.png`: Measured RLS wall-clock time & memory vs `D`, overlaid with
   theoretical `O(D^2)` curve.
4. `fig_e3_overparam_double_descent_heatmap.png`: Side-by-side `(D x gamma)` heatmaps contrasting
   Sample-Rich (`V = 50,000`) vs Small-Sample (`V = 1,600`) regimes with FD2 `gamma_auto` overlay.
5. `fig_e2_corruption_robustness.png` (generated automatically whenever E2 corruption results exist).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "vanilla_foal": "#C0392B",       # Crimson red (failure baseline)
    "norm_calibrated": "#2980B9",    # Academic blue (FD1b)
    "balanced_woodbury": "#1E8449",  # Emerald green (FD1a+b)
    "fd2_l2norm_adaptive": "#1E8449",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })


def plot_e1_figures(e1_json_path: Path, out_dir: Path) -> list[Path]:
    """Render Figure 1A (Long-Tail Degradation & FD1 Recovery) and Figure 1B (Weight-Norm Collapse)."""
    data = json.loads(e1_json_path.read_text(encoding="utf-8"))
    ds = data["dataset"].upper()
    agg = data["aggregated"]
    runs = data["runs"]
    created: list[Path] = []

    # Figure 1A: 2-panel (Left: Few-Shot Tail Accuracy vs rho; Right: Incremental A_bar vs rho)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    methods = [
        ("vanilla_foal", "Vanilla F-OAL (Paper MSE)"),
        ("norm_calibrated", "F-OAL + Power-Norm Calib. (FD1b)"),
        ("balanced_woodbury", "Balanced Woodbury RLS (FD1a+b)"),
    ]

    for m_id, label in methods:
        rows = sorted([r for r in agg if r["method"] == m_id], key=lambda x: x["rho"])
        if not rows:
            continue
        rhos = [r["rho"] for r in rows]
        tail_m = [r["tail_acc_mean"] for r in rows]
        tail_s = [r["tail_acc_std"] for r in rows]
        abar_m = [r["average_accuracy_mean"] for r in rows]
        abar_s = [r["average_accuracy_std"] for r in rows]
        color = PALETTE.get(m_id, "#333333")
        marker = "o" if m_id == "vanilla_foal" else ("s" if m_id == "norm_calibrated" else "^")

        axes[0].errorbar(rhos, tail_m, yerr=tail_s, label=label, color=color, marker=marker, lw=2.2, capsize=3.5)
        axes[1].errorbar(rhos, abar_m, yerr=abar_s, label=label, color=color, marker=marker, lw=2.2, capsize=3.5)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Imbalance Ratio $\\rho = N_{\\max} / N_{\\min}$ (log scale)")
    axes[0].set_ylabel("Few-Shot (Tail 1/3 Classes) Accuracy (%)")
    axes[0].set_title(f"({ds}) Minority-Class Collapse & FD1 Recovery")
    axes[0].grid(True, alpha=0.28, linestyle="--")
    axes[0].legend(loc="lower left", frameon=True)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Imbalance Ratio $\\rho = N_{\\max} / N_{\\min}$ (log scale)")
    axes[1].set_ylabel("Cumulative Average Accuracy $\\bar{A}$ (%)")
    axes[1].set_title(f"({ds}) Overall OCIL $\\bar{{A}}$ Across 10 Tasks (3 Seeds)")
    axes[1].grid(True, alpha=0.28, linestyle="--")
    axes[1].legend(loc="lower left", frameon=True)

    fig.tight_layout()
    p1 = out_dir / f"fig_e1_{data['dataset']}_longtail_curves.png"
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created.append(p1)

    # Figure 1B: Direct Refutation of Paper Figure 3 (||w_c||_2 at rho=1 vs max rho)
    max_rho = max(data["rhos"])
    seed0 = data["seeds"][0]
    bal_van = next((r for r in runs if r["method"] == "vanilla_foal" and r["rho"] == 1.0 and r["seed"] == seed0 and "quotas" in r), None)
    lt_van = next((r for r in runs if r["method"] == "vanilla_foal" and r["rho"] == max_rho and r["seed"] == seed0 and "quotas" in r), None)
    lt_bal = next((r for r in runs if r["method"] == "balanced_woodbury" and r["rho"] == max_rho and r["seed"] == seed0 and "quotas" in r), None)

    if lt_van is not None and lt_bal is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        quotas = np.array(lt_van["quotas"], dtype=np.float64)
        order_idx = np.argsort(-quotas)
        ranks = np.arange(1, len(quotas) + 1)

        if bal_van is not None:
            bn = np.array(bal_van["raw_column_norms"])
            axes[0].plot(ranks, bn / bn.mean(), color="#7F8C8D", ls="--", lw=1.8, label="Balanced ($\\rho=1$, Paper Fig. 3 regime)")

        vn = np.array(lt_van["raw_column_norms"])[order_idx]
        axes[0].plot(ranks, vn / vn.mean(), color=PALETTE["vanilla_foal"], lw=2.2, label=f"Vanilla F-OAL ($\\rho={int(max_rho)}$) — Collapses!")

        en = np.array(lt_bal["effective_column_norms"])[order_idx]
        axes[0].plot(ranks, en / en.mean(), color=PALETTE["balanced_woodbury"], lw=2.2, label=f"Balanced Woodbury + Calib. ($\\rho={int(max_rho)}$)")

        axes[0].set_xlabel("Classes Sorted by Sample Count $N_c$ (Head $\\rightarrow$ Tail)")
        axes[0].set_ylabel("Normalized Classifier Norm $\\|\\mathbf{w}_c\\|_2 / \\overline{\\|\\mathbf{w}\\|_2}$")
        axes[0].set_title(f"({ds}) Refuting Paper Fig. 3 Under Imbalance ($\\rho={int(max_rho)}$)")
        axes[0].grid(True, alpha=0.28, linestyle="--")
        axes[0].legend(loc="upper right", frameon=True)

        axes[1].scatter(quotas, lt_van["per_class_acc"], alpha=0.65, s=26, color=PALETTE["vanilla_foal"], label="Vanilla F-OAL")
        axes[1].scatter(quotas, lt_bal["per_class_acc"], alpha=0.65, s=26, color=PALETTE["balanced_woodbury"], marker="^", label="Balanced Woodbury (FD1)")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Training Samples per Class $N_c$ (log scale)")
        axes[1].set_ylabel("Final Test Accuracy of Class $c$ (%)")
        axes[1].set_title(f"({ds}) Per-Class Test Recall vs. Sample Count $N_c$")
        axes[1].grid(True, alpha=0.28, linestyle="--")
        axes[1].legend(loc="lower right", frameon=True)

        fig.tight_layout()
        p2 = out_dir / f"fig_e1_{data['dataset']}_weight_norm_refutation.png"
        fig.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(p2)

    return created


def plot_e3_figures(e3_json_path: Path, out_dir: Path) -> list[Path]:
    """Render Figure 2B-1 (Quadratic O(D^2) Scaling) and Figure 2B-2 ((D x gamma) Heatmaps)."""
    data = json.loads(e3_json_path.read_text(encoding="utf-8"))
    ds = data["dataset"].upper()
    s_runs = data["scaling_runs"]
    g_runs = data["grid_runs"]
    alpha = data["empirical_scaling_exponent_alpha"]
    created: list[Path] = []

    dims = sorted({r["projection_dim"] for r in s_runs})
    t_means = [float(np.mean([r["rls_update_seconds"] for r in s_runs if r["projection_dim"] == d])) for d in dims]
    t_stds = [float(np.std([r["rls_update_seconds"] for r in s_runs if r["projection_dim"] == d], ddof=1)) if len([r for r in s_runs if r["projection_dim"] == d]) > 1 else 0.0 for d in dims]
    mems = [next(r["r_matrix_mb"] for r in s_runs if r["projection_dim"] == d) for d in dims]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].errorbar(dims, t_means, yerr=t_stds, marker="o", lw=2.3, color="#8E44AD", capsize=3.5, label=f"Measured RLS Time (slope $\\alpha={alpha:.2f}$)")
    ref_idx = dims.index(1000) if 1000 in dims else 0
    quad_ref = [t_means[ref_idx] * ((d / dims[ref_idx]) ** 2) for d in dims]
    axes[0].plot(dims, quad_ref, ls="--", lw=1.8, color="#2C3E50", label="Exact Quadratic Law $\\Theta(D^2)$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Projection Width $D$ (log scale)")
    axes[0].set_ylabel("Total 10-Task RLS Update Time (s, log scale)")
    axes[0].set_title(f"({ds}) Empirical $\\Theta(V D^2)$ Compute Scaling")
    axes[0].grid(True, which="both", alpha=0.25, linestyle="--")
    axes[0].legend(frameon=True)

    axes[1].plot(dims, mems, marker="s", lw=2.3, color="#D35400", label="$\\mathbf{R} \\in \\mathbb{R}^{D \\times D}$ Footprint (float64)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Projection Width $D$ (log scale)")
    axes[1].set_ylabel("Autocorrelation Matrix $\\mathbf{R}$ Memory (MB)")
    axes[1].set_title("Exact $\\Theta(D^2)$ Memory Growth of Matrix $\\mathbf{R}$")
    axes[1].grid(True, which="both", alpha=0.25, linestyle="--")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    p_scale = out_dir / f"fig_e3_{data['dataset']}_quadratic_scaling.png"
    fig.savefig(p_scale, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created.append(p_scale)

    # Heatmaps across regimes
    regimes = sorted({r["regime"] for r in g_runs})
    fixed_rows = [r for r in g_runs if r["gamma_policy"].startswith("fixed_")]
    g_dims = sorted({r["projection_dim"] for r in fixed_rows})
    g_vals = sorted({r["resolved_gamma"] for r in fixed_rows})

    if g_dims and g_vals:
        fig, axes = plt.subplots(1, len(regimes), figsize=(6.0 * len(regimes), 4.6), squeeze=False)
        for idx, reg in enumerate(regimes):
            ax = axes[0, idx]
            mat = np.full((len(g_dims), len(g_vals)), np.nan)
            for i_d, d_val in enumerate(g_dims):
                for j_g, g_val in enumerate(g_vals):
                    match = next(
                        (r for r in fixed_rows if r["regime"] == reg and r["projection_dim"] == d_val and abs(r["resolved_gamma"] - g_val) < 1e-9),
                        None,
                    )
                    if match is not None:
                        mat[i_d, j_g] = match["average_accuracy"]

            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", origin="lower")
            ax.set_xticks(range(len(g_vals)))
            ax.set_xticklabels([f"{g:g}" for g in g_vals])
            ax.set_yticks(range(len(g_dims)))
            ax.set_yticklabels([str(d) for d in g_dims])
            ax.set_xlabel("Fixed Ridge Regularizer $\\gamma$")
            ax.set_ylabel("Projection Dimension $D$")
            v_tot = next((r["total_train_samples"] for r in g_runs if r["regime"] == reg), 0)
            ax.set_title(f"{ds} [{reg}] ($V={v_tot:,}$) — $\\bar{{A}}$ (%)")

            for i_d in range(len(g_dims)):
                for j_g in range(len(g_vals)):
                    if not np.isnan(mat[i_d, j_g]):
                        ax.text(j_g, i_d, f"{mat[i_d, j_g]:.1f}", ha="center", va="center", fontsize=8.5, color="black")

            # Annotate FD2 trace_auto performance on the right margin of each row
            for i_d, d_val in enumerate(g_dims):
                auto_r = next(
                    (r for r in g_runs if r["regime"] == reg and r["projection_dim"] == d_val and r["gamma_policy"] == "trace_auto"),
                    None,
                )
                if auto_r is not None:
                    ax.set_yticklabels(
                        [
                            f"{d} (FD2: {next((x['average_accuracy'] for x in g_runs if x['regime'] == reg and x['projection_dim'] == d and x['gamma_policy'] == 'trace_auto'), 0.0):.1f}%)"
                            for d in g_dims
                        ]
                    )
                    break

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        p_grid = out_dir / f"fig_e3_{data['dataset']}_overparam_heatmap.png"
        fig.savefig(p_grid, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(p_grid)

    return created


def plot_e2_figures(e2_json_path: Path, out_dir: Path) -> list[Path]:
    """Render Figure 2A (Covariate-Shift Corruption Curves & Feature Cosine Drift)."""
    data = json.loads(e2_json_path.read_text(encoding="utf-8"))
    ds = data["dataset"].upper()
    runs = data["runs"]
    corrs = sorted({r["corruption"] for r in runs})
    if not corrs:
        return []

    fig, axes = plt.subplots(1, len(corrs), figsize=(4.6 * len(corrs), 4.2), squeeze=False)
    for idx, corr in enumerate(corrs):
        ax = axes[0, idx]
        for m_id, label, ls in (
            ("vanilla_foal", "Vanilla F-OAL ($\\gamma=1$)", "-"),
            ("fd2_l2norm_adaptive", "FD2 ($L_2$-PreNorm + $\\gamma_{\\mathrm{auto}}$)", "--"),
        ):
            sub = sorted([r for r in runs if r["corruption"] == corr and r["method"] == m_id], key=lambda x: x["severity"])
            if not sub:
                continue
            sevs = [r["severity"] for r in sub]
            shift_acc = [r["test_shift_avg_acc"] for r in sub]
            stream_acc = [r["corrupted_stream_avg_acc"] for r in sub]
            color = PALETTE.get(m_id, "#333333")
            ax.plot(sevs, shift_acc, marker="o", ls=ls, lw=2.0, color=color, label=f"{label} [Clean->Corrupt]")
            ax.plot(sevs, stream_acc, marker="^", ls=":", lw=1.8, color=color, alpha=0.8, label=f"{label} [Corrupt Stream]")

        ax.set_xlabel("Corruption Severity $s \\in \\{0..5\\}$")
        ax.set_ylabel("Cumulative Average Accuracy $\\bar{A}$ (%)")
        ax.set_title(f"{ds} — {corr.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.28, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=8, frameon=True)

    fig.tight_layout()
    p_corr = out_dir / f"fig_e2_{data['dataset']}_corruption_curves.png"
    fig.savefig(p_corr, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [p_corr]


def main() -> None:
    """CLI entry point generating all available Lab 2 figures."""
    parser = argparse.ArgumentParser(description="Generate 300-DPI Lab 2 Empirical Demonstration figures.")
    parser.add_argument("--input-root", default="results/lab2_empirical")
    parser.add_argument("--output-dir", default="results/lab2_empirical/figures")
    args = parser.parse_args()

    _setup_style()
    in_root = Path(args.input_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for p in sorted(in_root.glob("e1_longtail/*/e1_longtail_summary.json")):
        generated.extend(plot_e1_figures(p, out_dir))
    for p in sorted(in_root.glob("e3_scaling_gamma/*/e3_scaling_gamma_summary.json")):
        generated.extend(plot_e3_figures(p, out_dir))
    for p in sorted(in_root.glob("e2_corruptions/*/e2_corruptions_summary.json")):
        generated.extend(plot_e2_figures(p, out_dir))

    print(f"[Lab2-Figures] Generated {len(generated)} figure(s) in {out_dir}:")
    for fp in generated:
        print(f"  - {fp.as_posix()}")


if __name__ == "__main__":
    main()
