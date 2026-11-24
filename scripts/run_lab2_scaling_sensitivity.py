"""Lab 2 Experiment E3: Quadratic O(D^2) Scaling & (D x gamma) Over-Parameterization Stress-Test.

Tests Hypothesis H2b (`docs/CRITICAL_ANALYSIS_REPORT.md` Section 2.4, 4.2 & 5.2):
1. Part A (Complexity Scaling Sweep): Evaluates projection widths `D in {250, 500, 1000, 2000, 4000, 8000}`
   across multiple seeds, measuring exact RLS update wall-clock time, matrix `R` memory footprint (`float64`),
   and empirical log-log scaling exponent `alpha` in `T_RLS ~ D^alpha`.
2. Part B (Small-Sample vs Sample-Rich `(D x gamma)` Grid & FD2 Adaptive Ridge):
   Sweeps `D in {250, 500, 1000, 2000, 4000}` x `gamma in {1e-3, 1e-2, 1e-1, 1, 10, 100}` plus
   automatic FD2 policies (`trace_scaled_gamma` and `gcv_gamma`), comparing sample-rich (`V = 50,000`)
   against small-sample (`V_small = 1,600` or native DTD/Aircraft) regimes where `D > V` triggers
   rank-deficient double-descent collapse at small `gamma`.

Usage:
    python -m scripts.run_lab2_scaling_sensitivity --dataset cifar100 --seeds 42 123 2026
    python -m scripts.run_lab2_scaling_sensitivity --dataset dtd --seeds 42 123 2026 --wandb
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lab2_common import (
    DEFAULT_WANDB_PROJECT,
    build_remapped_pools,
    extract_or_load_backbone_features,
    init_lab2_wandb_run,
    make_dataset_class_order,
)
from src.foal.critical_analysis import (
    compute_spectral_diagnostics,
    compute_trace_scaled_gamma,
    select_gcv_gamma,
)
from src.foal.metrics import summary_from_task_matrix
from src.foal.projection import SmoothProjection
from src.foal.rls import OnlineRidgeRLS


def run_ocil_at_dim_gamma(
    bundle: dict[str, Any],
    *,
    projection_dim: int,
    gamma_policy: str | float,
    seed: int,
    max_train_per_class: int | None = None,
    batch_size: int = 10,
    compute_spectrum: bool = False,
) -> dict[str, Any]:
    """Run one 10-task OCIL trial for a specific projection width ``D`` and ``gamma`` policy.

    Args:
        bundle: Backbone feature bundle from ``lab2_common``.
        projection_dim: SmoothProjection output dimension ``D``.
        gamma_policy: Positive float ``gamma``, or ``'trace_auto'`` / ``'gcv_auto'`` (FD2).
        seed: Random seed for class order, SmoothProjection, and batch order.
        max_train_per_class: Optional sample cap per class to emulate small-sample regime ``V < D``.
        batch_size: Streaming mini-batch size (default ``10``).
        compute_spectrum: Whether to compute exact spectral conditioning diagnostics.

    Returns:
        Dictionary of accuracy, forgetting, latency, memory, and spectral metrics.
    """
    dataset = bundle["dataset"]
    num_classes = int(bundle["num_classes"])
    cpt = int(bundle["classes_per_task"])
    num_tasks = num_classes // cpt

    cap = max_train_per_class if max_train_per_class is not None else (40 if dataset == "dtd" else None)
    class_order = make_dataset_class_order(dataset, seed=seed)
    tr_pool, te_pool = build_remapped_pools(
        bundle["train_orig_labels"],
        bundle["test_orig_labels"],
        class_order,
        max_train_per_class=cap,
        seed=seed,
    )

    proj = SmoothProjection(
        input_dim=bundle["train_features"].shape[1],
        output_dim=projection_dim,
        seed=seed,
        initialization="official_uniform",
    )
    with torch.no_grad():
        tr_z_all = proj(bundle["train_features"])
        te_z_all = proj(bundle["test_features"])

    # Prepare per-task train and test tensors
    task_tr_z: list[torch.Tensor] = []
    task_tr_y: list[torch.Tensor] = []
    task_te_z: list[torch.Tensor] = []
    task_te_y: list[torch.Tensor] = []
    total_train_samples = 0

    for stage in range(num_tasks):
        stage_classes = range(stage * cpt, (stage + 1) * cpt)
        tr_idxs = [i for c in stage_classes for i in tr_pool[c]]
        tr_lbls = [c for c in stage_classes for _ in tr_pool[c]]
        rng = np.random.default_rng(int(seed) * 1009 + stage)
        perm = rng.permutation(len(tr_idxs))
        task_tr_z.append(tr_z_all[[tr_idxs[p] for p in perm]])
        task_tr_y.append(torch.tensor([tr_lbls[p] for p in perm], dtype=torch.long))
        total_train_samples += len(tr_idxs)

        te_idxs = [i for c in stage_classes for i in te_pool[c]]
        te_lbls = [c for c in stage_classes for _ in te_pool[c]]
        task_te_z.append(te_z_all[te_idxs])
        task_te_y.append(torch.tensor(te_lbls, dtype=torch.long))

    # Resolve gamma (fixed scalar vs Task-1 FD2 adaptive rules)
    if isinstance(gamma_policy, str):
        if gamma_policy == "trace_auto":
            resolved_gamma = compute_trace_scaled_gamma(
                task_tr_z[0],
                gamma_0=1.0,
                total_expected_samples=total_train_samples,
                ref_dim=1000,
            )
            policy_name = "trace_auto"
        elif gamma_policy == "gcv_auto":
            gcv_info = select_gcv_gamma(
                task_tr_z[0],
                task_tr_y[0],
                num_classes_task1=cpt,
                horizon_tasks=float(num_tasks),
            )
            resolved_gamma = gcv_info["best_gamma_stream"]
            policy_name = "gcv_auto"
        else:
            raise ValueError(f"Unknown gamma_policy '{gamma_policy}'")
    else:
        resolved_gamma = float(gamma_policy)
        policy_name = f"fixed_{resolved_gamma:g}"

    clf = OnlineRidgeRLS(feature_dim=projection_dim, gamma=resolved_gamma, dtype=torch.float64)
    matrix: list[list[float]] = []

    rls_update_seconds = 0.0
    total_batches = 0

    for stage in range(num_tasks):
        clf.expand_classes((stage + 1) * cpt)
        z_s = task_tr_z[stage]
        y_s = task_tr_y[stage]

        t_up = time.perf_counter()
        for start in range(0, len(y_s), batch_size):
            clf.update(z_s[start:start + batch_size], y_s[start:start + batch_size])
            total_batches += 1
        rls_update_seconds += time.perf_counter() - t_up

        stage_row: list[float] = []
        for eval_t in range(stage + 1):
            preds_t = clf.predict(task_te_z[eval_t])
            stage_row.append(float((preds_t == task_te_y[eval_t]).to(torch.float64).mean().item()))
        matrix.append(stage_row)

    inc = summary_from_task_matrix(matrix)
    r_mem_mb = (projection_dim * projection_dim * 8) / (1024.0 * 1024.0)
    w_mem_mb = (projection_dim * num_classes * 8) / (1024.0 * 1024.0)
    w_fro_norm = float(torch.linalg.matrix_norm(clf.W, ord="fro").item())

    spec_info: dict[str, float] = {}
    if compute_spectrum:
        all_tr_z = torch.cat(task_tr_z, dim=0)
        # Subsample at most 2,500 rows when V=50,000 so SVD stays fast on CPU
        if all_tr_z.shape[0] > 2500:
            g_rng = np.random.default_rng(42)
            sub_idx = sorted(g_rng.choice(all_tr_z.shape[0], size=2500, replace=False).tolist())
            spec_info = compute_spectral_diagnostics(all_tr_z[sub_idx], gamma=resolved_gamma)
        else:
            spec_info = compute_spectral_diagnostics(all_tr_z, gamma=resolved_gamma)

    return {
        "dataset": dataset,
        "total_train_samples": int(total_train_samples),
        "projection_dim": int(projection_dim),
        "gamma_policy": policy_name,
        "resolved_gamma": round(float(resolved_gamma), 6),
        "seed": int(seed),
        "average_accuracy": round(float(inc["average_incremental_accuracy"]) * 100.0, 4),
        "last_accuracy": round(float(inc["last_accuracy"]) * 100.0, 4),
        "forgetting_past": round(float(inc["forgetting"]) * 100.0, 4),
        "rls_update_seconds": round(float(rls_update_seconds), 4),
        "ms_per_batch": round((rls_update_seconds * 1000.0) / max(total_batches, 1), 4),
        "r_matrix_mb": round(r_mem_mb, 3),
        "total_state_mb": round(r_mem_mb + w_mem_mb, 3),
        "w_frobenius_norm": round(w_fro_norm, 4),
        **{f"spec_{k}": round(float(v), 4) for k, v in spec_info.items()},
    }


def fit_loglog_exponent(dims: list[float], times: list[float]) -> float:
    """Estimate asymptotic power-law exponent `alpha` in `T ~ D^alpha` for `D >= 1000`."""
    pairs = [(math.log(d), math.log(max(t, 1e-6))) for d, t in zip(dims, times) if d >= 1000]
    if len(pairs) < 2:
        pairs = [(math.log(d), math.log(max(t, 1e-6))) for d, t in zip(dims, times)]
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)
    slope, _ = np.polyfit(xs, ys, 1)
    return round(float(slope), 4)


def main() -> None:
    """CLI entry point for Lab 2 Experiment E3 (Scaling & Regularization Sensitivity)."""
    parser = argparse.ArgumentParser(description="Run Lab 2 E3 Scaling & (D x gamma) Sensitivity Sweep.")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "dtd", "aircraft"])
    parser.add_argument("--scaling-dims", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000, 6000])
    parser.add_argument("--grid-dims", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000])
    parser.add_argument("--gammas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size ( RLS solution is batch-size invariant! )")
    parser.add_argument("--small-sample-cap", type=int, default=16, help="Samples/class for CIFAR small-sample V=1600 contrast")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/lab2_empirical/e3_scaling_gamma")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Lab2-E3] Loading backbone features for '{args.dataset}'...")
    bundle = extract_or_load_backbone_features(
        args.dataset,
        device=device,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    # Part A: Wall-clock & memory scaling sweep across D at gamma=1.0 (multi-seed)
    print("\n[Lab2-E3A] Running O(D^2) Complexity Scaling Sweep...")
    scaling_runs: list[dict[str, Any]] = []
    for d_dim in args.scaling_dims:
        for seed in args.seeds:
            rec = run_ocil_at_dim_gamma(
                bundle,
                projection_dim=d_dim,
                gamma_policy=1.0,
                seed=seed,
                batch_size=args.batch_size,
                compute_spectrum=(seed == args.seeds[0] and d_dim <= 4000),
            )
            scaling_runs.append(rec)
            print(
                f"  D={d_dim:5d} | seed={seed:4d} -> A_bar={rec['average_accuracy']:6.2f}% | "
                f"RLS_time={rec['rls_update_seconds']:6.3f}s ({rec['ms_per_batch']:6.2f} ms/batch) | R_MB={rec['r_matrix_mb']:6.2f}"
            )

    mean_times = [
        float(np.mean([r["rls_update_seconds"] for r in scaling_runs if r["projection_dim"] == d]))
        for d in args.scaling_dims
    ]
    emp_exponent = fit_loglog_exponent([float(d) for d in args.scaling_dims], mean_times)
    print(f"  => Empirical log-log scaling exponent alpha (D >= 1000): {emp_exponent:.3f} (Theory: 2.000)")

    # Part B: (D x gamma) grid + FD2 adaptive policies in native regime AND small-sample V=1600 regime (if cifar100)
    regimes = [("native", None)]
    if args.dataset == "cifar100" and args.small_sample_cap > 0:
        regimes.append((f"small_sample_v{args.small_sample_cap * 100}", args.small_sample_cap))

    grid_runs: list[dict[str, Any]] = []
    policies: list[str | float] = [*args.gammas, "trace_auto", "gcv_auto"]

    for regime_name, cap in regimes:
        print(f"\n[Lab2-E3B] Running (D x gamma) Grid for regime='{regime_name}' (seed={args.seeds[0]})...")
        for d_dim in args.grid_dims:
            for pol in policies:
                rec = run_ocil_at_dim_gamma(
                    bundle,
                    projection_dim=d_dim,
                    gamma_policy=pol,
                    seed=args.seeds[0],
                    max_train_per_class=cap,
                    batch_size=args.batch_size,
                    compute_spectrum=(cap is not None and pol in (1e-3, 1.0, "trace_auto")),
                )
                rec["regime"] = regime_name
                grid_runs.append(rec)
                print(
                    f"  [{regime_name:18s}] D={d_dim:4d} | policy={str(pol):10s} (gamma={rec['resolved_gamma']:8.4f}) "
                    f"-> A_bar={rec['average_accuracy']:6.2f}% | A_10={rec['last_accuracy']:6.2f}% | ||W||_F={rec['w_frobenius_norm']:8.2f}"
                )

                wb_run = init_lab2_wandb_run(
                    enabled=args.wandb,
                    offline=args.wandb_offline,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=f"lab2_e3_{args.dataset}_{regime_name}_D{d_dim}_{rec['gamma_policy']}",
                    group=f"lab2_e3_scaling_{args.dataset}",
                    extra_tags=["e3-scaling-gamma", args.dataset, regime_name],
                    config={
                        "experiment": "E3_scaling_and_gamma_grid",
                        "dataset": args.dataset,
                        "regime": regime_name,
                        "projection_dim": d_dim,
                        "gamma_policy": rec["gamma_policy"],
                        "resolved_gamma": rec["resolved_gamma"],
                        "seed": args.seeds[0],
                        "batch_size": args.batch_size,
                        "provenance": bundle["provenance"],
                    },
                )
                if wb_run is not None:
                    wb_run.log({
                        "average_accuracy": rec["average_accuracy"],
                        "last_accuracy": rec["last_accuracy"],
                        "forgetting_past": rec["forgetting_past"],
                        "rls_update_seconds": rec["rls_update_seconds"],
                        "r_matrix_mb": rec["r_matrix_mb"],
                        "w_frobenius_norm": rec["w_frobenius_norm"],
                    })
                    wb_run.finish()

    payload = {
        "experiment": "Lab2_E3_Scaling_and_Gamma_Sensitivity",
        "dataset": args.dataset,
        "provenance": bundle["provenance"],
        "empirical_scaling_exponent_alpha": emp_exponent,
        "scaling_runs": scaling_runs,
        "grid_runs": grid_runs,
    }
    (out_dir / "e3_scaling_gamma_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if grid_runs:
        all_keys = sorted({k for r in grid_runs for k in r.keys()})
        with (out_dir / "e3_grid_runs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(grid_runs)

    print(f"[Lab2-E3] Saved full summary to {out_dir / 'e3_scaling_gamma_summary.json'}")


if __name__ == "__main__":
    main()
