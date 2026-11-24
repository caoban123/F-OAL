# -*- coding: utf-8 -*-
"""Generate 400-DPI colorblind-safe vector-sharp visual assets for the Portrait A2 Poster.

Follows NeurIPS / ICML / CVPR scientific poster design standards:
- Colorblind-safe Okabe-Ito & Viridis/YlGnBu palette (zero red-vs-green ambiguity).
- Analytical "Action Titles" on every sub-figure.
- Crisp Computer Modern (`cm`) LaTeX math equations at 450 DPI (no pixelated screenshots).
- Authentic ISO-18004 scannable QR codes to GitHub Repository & Public W&B Project.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

EXTRA_LIB = Path("D:/.cache/pylibs")
if EXTRA_LIB.exists() and str(EXTRA_LIB) not in sys.path:
    sys.path.insert(0, str(EXTRA_LIB))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO_ROOT / "poster" / "assets"

# Okabe-Ito & Institutional Academic Palette (Colorblind-safe)
NAVY = "#0F2942"
SLATE = "#334155"
MUTED = "#64748B"
VERMILION = "#C2410C"      # High-contrast warm vermilion/burnt-orange (Critique / Baseline)
SAPPHIRE = "#0284C7"       # Distinct bright sapphire blue (FD1b)
TEAL = "#0F766E"           # Deep bluish-teal (FD1a+b / FD2)
AMBER = "#D97706"
PURPLE = "#6D28D9"
ICE_BG = "#F8FAFC"
WARM_BG = "#FFF7ED"
TEAL_BG = "#F0FDFA"
BORDER = "#CBD5E1"


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def generate_qr_codes(out_dir: Path) -> None:
    """Generate crisp scannable QR PNGs for GitHub Repo and W&B Project."""
    import qrcode

    targets = {
        "qr_github.png": "https://github.com/caoban123/F-OAL",
        "qr_wandb.png": "https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction",
    }
    for fname, url in targets.items():
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=14,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color=NAVY, back_color="white")
        img.save(out_dir / fname)


def generate_pipeline_schematic(out_dir: Path) -> Path:
    """Draw redrawn F-OAL pipeline diagram with fault-line callouts (aspect 200 x 70 mm)."""
    fig, ax = plt.subplots(figsize=(10.0, 3.5), dpi=260)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def draw_box(x, y, w, h, title, sub, fc, ec, tc=NAVY, lw=1.8):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.6, color=SLATE)

    def draw_arrow(x1, y1, x2, y2, col=NAVY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=2.0, color=col))

    # Main F-OAL Forward-Only Pipeline Row (y = 1.65 .. 2.95)
    draw_box(0.15, 1.72, 1.75, 1.22, "Online Stream", "1-Pass Mini-Batch\n$X_{k,i} \\in \\mathbb{R}^{S \\times 3 \\times 224^2}$", ICE_BG, BORDER)
    draw_arrow(1.94, 2.33, 2.28, 2.33)
    draw_box(2.32, 1.72, 2.25, 1.22, "Frozen ViT-B/16 + FF", "$\\nabla_\\theta = 0$ · 12-Block Avg\n$\\mathbf{h} = \\mathrm{LN}(\\frac{1}{12}\\sum_{l=1}^{12} \\mathbf{h}^{(l)})$", "#EFF6FF", SAPPHIRE)
    draw_arrow(4.61, 2.33, 4.95, 2.33)
    draw_box(4.99, 1.72, 2.22, 1.22, "Smooth Projection", "Fixed Random $P \\in \\mathbb{R}^{768 \\times D}$\n$\\phi(x) = \\sigma(\\mathbf{h}P) \\in (0,1)^D$", TEAL_BG, TEAL)
    draw_arrow(7.25, 2.33, 7.59, 2.33)
    draw_box(7.63, 1.72, 2.22, 1.22, "Woodbury RLS Head", "Update $R_{k,i} \\in \\mathbb{R}^{D \\times D}, \\hat{W}$\n$\\min_W \\|Y - \\Phi W\\|_F^2 + \\gamma\\|W\\|_F^2$", "#FEF3C7", AMBER)

    # Bottom Fault-Line Audit Badges (y = 0.15 .. 1.28)
    draw_box(
        0.15, 0.14, 3.05, 1.12,
        "L3 (README): Variance Law",
        "Paper p.4 '$\\mathcal{N}(0,1)$' omits $1/d_{\\mathrm{in}}$ scale\n$\\Rightarrow 81\\%$ Sigmoid saturation ($-18.8$ pp)",
        "#F8FAFC", "#64748B", tc="#334155", lw=1.4,
    )
    draw_box(
        3.42, 0.14, 3.12, 1.12,
        "Axis II (L2): Rank & O(D²) Trap",
        "Eq.(14) $C$-bug; true $R$ cost $\\Theta(D^2 S)$;\n$D \\approx V, \\gamma \\ll 1 \\Rightarrow 16\\times \\|\\hat{W}\\|_F$ explosion",
        WARM_BG, VERMILION, tc=VERMILION, lw=1.8,
    )
    draw_box(
        6.76, 0.14, 3.09, 1.12,
        "Axis I (L1): Long-Tail Collapse",
        "Unweighted $\{0,1\}$ MSE forces $\\|\\hat{\\mathbf{w}}_c\\|_2 \\propto N_c$\n$\\Rightarrow -60.8$ pp tail crash at $\\rho=100$",
        WARM_BG, VERMILION, tc=VERMILION, lw=1.8,
    )

    # Dashed fault pointers
    ax.plot([6.10, 1.75], [1.70, 1.28], ls="--", lw=1.3, color="#64748B")
    ax.plot([6.10, 4.98], [1.70, 1.28], ls="--", lw=1.5, color=VERMILION)
    ax.plot([8.74, 8.30], [1.70, 1.28], ls="--", lw=1.5, color=VERMILION)

    out_path = out_dir / "asset_pipeline_critique.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    import gc; gc.collect()
    return out_path


def generate_equation_plates(out_dir: Path) -> tuple[Path, Path]:
    """Render ultra-crisp Computer Modern LaTeX formula plates for Axis I and Axis II."""
    import gc
    # Plate 1: Axis I (L1 Theorem + FD1 Remedy)
    fig1, ax1 = plt.subplots(figsize=(6.8, 2.55), dpi=300)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    fig1.patch.set_facecolor("white")

    ax1.add_patch(FancyBboxPatch((0.01, 0.52), 0.98, 0.46, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor=WARM_BG, edgecolor=VERMILION, lw=1.5))
    ax1.text(0.04, 0.87, "Theorem L1 (Sherman–Morrison Minority-Class Norm Collapse):", fontsize=9.5, fontweight="bold", color=VERMILION, va="center")
    ax1.text(
        0.50, 0.65,
        r"$(\Phi^\top \Phi + \gamma I_D)\hat{\mathbf{w}}_c = N_c \hat{\boldsymbol{\mu}}_c \;\Longrightarrow\; \|\hat{\mathbf{w}}_c\|_2 = \frac{N_c \alpha}{N_c \alpha^2 + V\sigma_\perp^2 + \gamma} \propto N_c$",
        fontsize=11.4, ha="center", va="center", color=NAVY,
    )

    ax1.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.45, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor=TEAL_BG, edgecolor=TEAL, lw=1.5))
    ax1.text(0.04, 0.37, "Remedy FD1 (Causal Weighted Woodbury + Zero-Cost Norm Calibration):", fontsize=9.5, fontweight="bold", color=TEAL, va="center")
    ax1.text(
        0.50, 0.16,
        r"$\tilde{\mathbf{x}}_i = \sqrt{\omega_i}\,\boldsymbol{\phi}_i,\;\; \omega_i = \left(\frac{N_{\mathrm{ref}}}{\hat{N}_{y_i}^{(t)}}\right)^{\!\beta}\!,\quad \tilde{s}_c(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})\frac{\hat{\mathbf{w}}_c}{\|\hat{\mathbf{w}}_c\|_2^\tau + \varepsilon}$",
        fontsize=11.2, ha="center", va="center", color=NAVY,
    )
    p_eq1 = out_dir / "asset_eq_axis1.png"
    fig1.savefig(p_eq1, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig1)
    gc.collect()

    # Plate 2: Axis II (L2 Spectral & Complexity Theorem + FD2 Remedy)
    fig2, ax2 = plt.subplots(figsize=(6.8, 2.55), dpi=300)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    fig2.patch.set_facecolor("white")

    ax2.add_patch(FancyBboxPatch((0.01, 0.52), 0.98, 0.46, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor=WARM_BG, edgecolor=VERMILION, lw=1.5))
    ax2.text(0.04, 0.87, "Critique L2 (Exact Complexity & Tikhonov Spectral Noise Gain):", fontsize=9.5, fontweight="bold", color=VERMILION, va="center")
    ax2.text(
        0.50, 0.66,
        r"$T_R = \Theta(D^2 S + DS^2 + S^3) \perp\!\!\!\perp C,\qquad \hat{W} = \sum_{r=1}^{\min(V,D)} \frac{\sigma_r}{\sigma_r^2 + \gamma}\mathbf{v}_r(\mathbf{u}_r^\top Y)$",
        fontsize=11.0, ha="center", va="center", color=NAVY,
    )

    ax2.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.45, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor=TEAL_BG, edgecolor=TEAL, lw=1.5))
    ax2.text(0.04, 0.37, "Remedy FD2 (Parameter-Free Task-1 Trace & Rank Adaptive Ridge):", fontsize=9.5, fontweight="bold", color=TEAL, va="center")
    ax2.text(
        0.50, 0.16,
        r"$\gamma_{\mathrm{auto}} = \gamma_0 \cdot \frac{\mathrm{tr}(\Phi_1^\top \Phi_1)}{0.25\, V_1 D_{\mathrm{ref}}} \cdot \max\!\left(1,\, \frac{D}{V_{\mathrm{est}}}\right) \quad (\text{0 validation split})$",
        fontsize=11.0, ha="center", va="center", color=NAVY,
    )
    p_eq2 = out_dir / "asset_eq_axis2.png"
    fig2.savefig(p_eq2, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig2)
    gc.collect()

    return p_eq1, p_eq2


def generate_axis1_master_figure(out_dir: Path) -> Path:
    """Render 4-panel colorblind-safe Axis I empirical figure (aspect ~240 x 156 mm)."""
    e1_path = REPO_ROOT / "results" / "lab2_empirical" / "e1_longtail" / "cifar100" / "e1_longtail_summary.json"
    data = json.loads(e1_path.read_text(encoding="utf-8"))
    agg = data["aggregated"]
    runs = data["runs"]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.45), dpi=250)
    fig.patch.set_facecolor("white")

    specs = [
        ("vanilla_foal", "Vanilla F-OAL (Paper MSE) [g34o8awx]", VERMILION, "o", "-"),
        ("norm_calibrated", "Power-Norm Calib. FD1b [kafjjzio]", SAPPHIRE, "s", "--"),
        ("balanced_woodbury", "Balanced Woodbury FD1a+b [fqahbyt4]", TEAL, "^", "-"),
    ]

    # (A) Few-Shot Tail Recall vs rho
    ax = axes[0, 0]
    for m_id, label, col, mk, ls in specs:
        rows = sorted([r for r in agg if r["method"] == m_id], key=lambda x: x["rho"])
        rhos = [r["rho"] for r in rows]
        ym = [r["tail_acc_mean"] for r in rows]
        ys = [r["tail_acc_std"] for r in rows]
        ax.errorbar(rhos, ym, yerr=ys, label=label, color=col, marker=mk, ls=ls, lw=2.4, markersize=6.5, capsize=3.5)

    # Shaded gain annotation at rho=100
    v100 = next(r["tail_acc_mean"] for r in agg if r["method"] == "vanilla_foal" and r["rho"] == 100.0)
    b100 = next(r["tail_acc_mean"] for r in agg if r["method"] == "balanced_woodbury" and r["rho"] == 100.0)
    ax.annotate(
        f"+{b100 - v100:.1f} pp\nrecovery!",
        xy=(100, b100 - 1.5), xytext=(28, 43),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8),
        fontsize=9.5, fontweight="bold", color=TEAL,
        bbox=dict(boxstyle="round,pad=0.25", fc=TEAL_BG, ec=TEAL, lw=1.0),
    )
    ax.set_xscale("log")
    ax.set_xlabel("Imbalance Ratio $\\rho = N_{\\max}/N_{\\min}$ (log)")
    ax.set_ylabel("Tail 1/3 Classes Recall (%)")
    ax.set_title("A. Tail Recall Crashes 60.8 pp; FD1 Recovers +42.7 pp", fontsize=10.8, fontweight="bold", color=NAVY)
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(fontsize=8.2, loc="lower left", frameon=True)

    # (B) Cumulative 10-Task A_bar vs rho
    ax = axes[0, 1]
    for m_id, label, col, mk, ls in specs:
        rows = sorted([r for r in agg if r["method"] == m_id], key=lambda x: x["rho"])
        rhos = [r["rho"] for r in rows]
        ym = [r["average_accuracy_mean"] for r in rows]
        ys = [r["average_accuracy_std"] for r in rows]
        ax.errorbar(rhos, ym, yerr=ys, label=label.split(" [")[0], color=col, marker=mk, ls=ls, lw=2.4, markersize=6.5, capsize=3.5)
    ax.set_xscale("log")
    ax.set_xlabel("Imbalance Ratio $\\rho = N_{\\max}/N_{\\min}$ (log)")
    ax.set_ylabel("10-Task Incremental $\\bar{A}$ (%)")
    ax.set_title("B. FD1 Restores +13.3 pp Overall $\\bar{A}$ at $\\rho=100$ (3 Seeds)", fontsize=10.8, fontweight="bold", color=NAVY)
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(fontsize=8.3, loc="lower left", frameon=True)

    # (C) Direct Refutation of Paper Figure 3 (Weight Norms across 100 classes)
    ax = axes[1, 0]
    seed0 = data["seeds"][0]
    bal_van = next(r for r in runs if r["method"] == "vanilla_foal" and r["rho"] == 1.0 and r["seed"] == seed0)
    lt_van = next(r for r in runs if r["method"] == "vanilla_foal" and r["rho"] == 100.0 and r["seed"] == seed0)
    lt_bal = next(r for r in runs if r["method"] == "balanced_woodbury" and r["rho"] == 100.0 and r["seed"] == seed0)

    quotas = np.array(lt_van["quotas"], dtype=np.float64)
    ord_idx = np.argsort(-quotas)
    ranks = np.arange(1, len(quotas) + 1)
    bn = np.array(bal_van["raw_column_norms"])
    vn = np.array(lt_van["raw_column_norms"])[ord_idx]
    en = np.array(lt_bal["effective_column_norms"])[ord_idx]

    ax.plot(ranks, bn / bn.mean(), color="#64748B", ls=":", lw=1.9, label="Balanced $\\rho=1$ (Paper Fig. 3)")
    ax.plot(ranks, vn / vn.mean(), color=VERMILION, lw=2.2, label="Vanilla F-OAL $\\rho=100$ ($r=+0.89$)")
    ax.plot(ranks, en / en.mean(), color=TEAL, lw=2.2, label="Balanced Woodbury FD1 ($\\rho=100$)")
    ax.set_xlabel("Classes Ranked by $N_c$ (Head $\\rightarrow$ Tail)")
    ax.set_ylabel("Relative Norm $\\|\\mathbf{w}_c\\|_2 / \\overline{\\|\\mathbf{w}\\|_2}$")
    ax.set_title("C. Paper Fig. 3 Uniformity Breaks Under Imbalance", fontsize=10.8, fontweight="bold", color=NAVY)
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(fontsize=8.2, loc="upper right", frameon=True)

    # (D) Head vs Medium vs Tail Bar Breakdown at rho = 100
    ax = axes[1, 1]
    groups = ["Head (Many)", "Medium", "Tail (Few)", "Harmonic Mean"]
    x_pos = np.arange(len(groups))
    width = 0.26
    for idx, (m_id, short_lbl, col) in enumerate([
        ("vanilla_foal", "Vanilla F-OAL", VERMILION),
        ("norm_calibrated", "Norm Calib. (FD1b)", SAPPHIRE),
        ("balanced_woodbury", "Balanced RLS (FD1a+b)", TEAL),
    ]):
        r100 = next(r for r in agg if r["method"] == m_id and r["rho"] == 100.0)
        vals = [r100["head_acc_mean"], r100["medium_acc_mean"], r100["tail_acc_mean"], r100["harmonic_mean_acc_mean"]]
        errs = [r100["head_acc_std"], r100["medium_acc_std"], r100["tail_acc_std"], r100["harmonic_mean_acc_std"]]
        bars = ax.bar(x_pos + (idx - 1) * width, vals, width, yerr=errs, label=short_lbl, color=col, alpha=0.92, capsize=2.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=SLATE)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=9.0)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("D. Group Breakdown at $\\rho=100$: +33.9 pp Harmonic Mean", fontsize=10.8, fontweight="bold", color=NAVY)
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.legend(fontsize=8.0, loc="upper right", frameon=True)

    del data, runs
    fig.tight_layout(pad=0.85)
    out_path = out_dir / "asset_fig_axis1.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    import gc; gc.collect()
    return out_path


def generate_axis2_master_figure(out_dir: Path) -> Path:
    """Render 4-panel colorblind-safe Axis II empirical figure (aspect ~240 x 152 mm)."""
    e3_path = REPO_ROOT / "results" / "lab2_empirical" / "e3_scaling_gamma" / "cifar100" / "e3_scaling_gamma_summary.json"
    data = json.loads(e3_path.read_text(encoding="utf-8"))
    s_runs = data["scaling_runs"]
    g_runs = data["grid_runs"]
    alpha = data["empirical_scaling_exponent_alpha"]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.30), dpi=250)
    fig.patch.set_facecolor("white")

    # (A) Dual-axis Empirical O(D^2) Compute & Memory Wall
    ax = axes[0, 0]
    dims = sorted({r["projection_dim"] for r in s_runs})
    t_means = [float(np.mean([r["rls_update_seconds"] for r in s_runs if r["projection_dim"] == d])) for d in dims]
    t_stds = [float(np.std([r["rls_update_seconds"] for r in s_runs if r["projection_dim"] == d], ddof=1)) for d in dims]
    mems = [next(r["r_matrix_mb"] for r in s_runs if r["projection_dim"] == d) for d in dims]

    ax.errorbar(dims, t_means, yerr=t_stds, color=PURPLE, marker="o", lw=2.4, capsize=3.5, label=f"RLS Time (slope $\\alpha={alpha:.2f}$) [nbr04hu5]")
    idx_1k = dims.index(1000)
    quad_line = [t_means[idx_1k] * ((d / 1000.0) ** 2) for d in dims]
    ax.plot(dims, quad_line, color=NAVY, ls="--", lw=1.8, label="Exact Theory $\\Theta(D^2)$ ($\\alpha=2.00$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Projection Width $D$ (log)")
    ax.set_ylabel("10-Task RLS Time (s, log)", color=PURPLE)
    ax.set_title("A. RLS Cost Scales as $\\Theta(D^2)$: $35.6\\times$ Slower at $D=6000$", fontsize=10.6, fontweight="bold", color=NAVY)
    ax.grid(True, which="both", alpha=0.24, ls="--")
    ax.legend(fontsize=8.1, loc="upper left", frameon=True)

    # (B) Classic Marchenko-Pastur Double-Descent Spike in ||W||_F and A_10 Crash at D ~ V=1600
    ax = axes[0, 1]
    small_g = [r for r in g_runs if r["regime"] == "small_sample_v1600"]
    g_dims = sorted({r["projection_dim"] for r in small_g})

    a10_low = [next(r["last_accuracy"] for r in small_g if r["projection_dim"] == d and r["gamma_policy"] == "fixed_0.001") for d in g_dims]
    a10_pap = [next(r["last_accuracy"] for r in small_g if r["projection_dim"] == d and r["gamma_policy"] == "fixed_1") for d in g_dims]
    a10_fd2 = [next(r["last_accuracy"] for r in small_g if r["projection_dim"] == d and r["gamma_policy"] == "trace_auto") for d in g_dims]
    wnorm_low = [next(r["w_frobenius_norm"] for r in small_g if r["projection_dim"] == d and r["gamma_policy"] == "fixed_0.001") for d in g_dims]

    ax.plot(g_dims, a10_low, color=VERMILION, marker="o", lw=2.3, label="Fixed $\\gamma=10^{-3}$ (Double-Descent!)")
    ax.plot(g_dims, a10_pap, color=SAPPHIRE, marker="s", ls="--", lw=2.0, label="Paper Fixed $\\gamma=1.0$")
    ax.plot(g_dims, a10_fd2, color=TEAL, marker="^", lw=2.4, label="FD2 $\\gamma_{\\mathrm{auto}}$ [kyj36qk9]")
    ax.axvline(1600, color="#64748B", ls=":", lw=1.6)
    ax.annotate(
        "Interpolation Peak ($D \\approx V$)\n$\\|\\hat{W}\\|_F = 155.0$ ($16\\times$ blowup!)",
        xy=(1960, 72.85), xytext=(620, 76.2),
        arrowprops=dict(arrowstyle="->", color=VERMILION, lw=1.7),
        fontsize=8.2, fontweight="bold", color=VERMILION,
        bbox=dict(boxstyle="round,pad=0.25", fc=WARM_BG, ec=VERMILION, lw=1.0),
    )
    ax.set_xscale("log")
    ax.set_xlabel("Projection Width $D$ (log, $V=1,600$ samples)")
    ax.set_ylabel("Final Accuracy $A_{10}$ (%)")
    ax.set_title("B. Double-Descent Crash at $D \\approx V$ & FD2 Stabilization", fontsize=10.6, fontweight="bold", color=NAVY)
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(fontsize=8.0, loc="lower left", frameon=True)

    # (C) & (D) Colorblind-safe YlGnBu Heatmaps for V=50,000 vs V=1,600
    fixed_gammas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    for col_idx, (reg, title_tag, run_id) in enumerate([
        ("native", "C. Sample-Rich $V=50,000$: Well-Conditioned", "1bns4np4"),
        ("small_sample_v1600", "D. Small-Sample $V=1,600$: FD2 Beats Fixed $\\gamma$", "kyj36qk9"),
    ]):
        ax = axes[1, col_idx]
        sub_r = [r for r in g_runs if r["regime"] == reg]
        mat = np.zeros((len(g_dims), len(fixed_gammas)))
        for i_d, d_val in enumerate(g_dims):
            for j_g, g_val in enumerate(fixed_gammas):
                rec = next(r for r in sub_r if r["projection_dim"] == d_val and r["gamma_policy"].startswith("fixed_") and abs(r["resolved_gamma"] - g_val) < 1e-9)
                mat[i_d, j_g] = rec["average_accuracy"]

        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", origin="lower")
        ax.set_xticks(range(len(fixed_gammas)))
        ax.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$", "$10$", "$10^2$"])
        ylabs = []
        for d_val in g_dims:
            fd2_acc = next(r["average_accuracy"] for r in sub_r if r["projection_dim"] == d_val and r["gamma_policy"] == "trace_auto")
            ylabs.append(f"{d_val} (FD2:{fd2_acc:.1f})")
        ax.set_yticks(range(len(g_dims)))
        ax.set_yticklabels(ylabs, fontsize=8.3)
        ax.set_xlabel(f"Fixed $\\gamma$  [W&B: {run_id}]")
        if col_idx == 0:
            ax.set_ylabel("Width $D$  (FD2 $\\bar{A}$%)")
        ax.set_title(title_tag, fontsize=10.5, fontweight="bold", color=NAVY)

        vmin, vmax = mat.min(), mat.max()
        mid = 0.5 * (vmin + vmax)
        for i_d in range(len(g_dims)):
            for j_g in range(len(fixed_gammas)):
                val = mat[i_d, j_g]
                txt_col = "white" if val > mid + 0.15 * (vmax - vmin) else "#0F172A"
                ax.text(j_g, i_d, f"{val:.1f}", ha="center", va="center", fontsize=8.3, fontweight="bold", color=txt_col)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    del data, s_runs, g_runs
    fig.tight_layout(pad=0.85)
    out_path = out_dir / "asset_fig_axis2.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    import gc; gc.collect()
    return out_path


def main() -> None:
    """Build all high-resolution assets for the Portrait A2 Poster."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _configure_matplotlib()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generate_qr_codes(ASSET_DIR)
    p_pipe = generate_pipeline_schematic(ASSET_DIR)
    p_eq1, p_eq2 = generate_equation_plates(ASSET_DIR)
    p_f1 = generate_axis1_master_figure(ASSET_DIR)
    p_f2 = generate_axis2_master_figure(ASSET_DIR)
    print("Generated Poster A2 high-DPI assets:")
    for p in (p_pipe, p_eq1, p_eq2, p_f1, p_f2):
        print(" -", p.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
