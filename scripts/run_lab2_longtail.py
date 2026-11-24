"""Lab 2 Experiment E1: Long-Tail Class-Imbalanced OCIL Stress-Test & FD1 Mitigation.

Tests Hypothesis H1 (`docs/CRITICAL_ANALYSIS_REPORT.md` Section 4.1 & 5.1):
1. Evaluates Vanilla F-OAL across exponential imbalance ratios `rho in {1, 5, 10, 25, 50, 100}`
   over multiple random seeds, measuring cumulative average accuracy `A_bar`, final accuracy `A_T`,
   Head / Medium / Tail class accuracies, and classifier column L2 norms `||w_c||_2` vs `N_c`
   (refuting the generality of Paper Figure 3 beyond balanced benchmarks).
2. Evaluates two zero-replay analytic remedies (FD1):
   - `norm_calibrated`: post-hoc column power-norm calibration (`norm_tau = 0.75`, 0 training cost).
   - `balanced_woodbury`: causal online inverse-frequency Weighted Woodbury RLS + norm calibration.

Usage:
    python -m scripts.run_lab2_longtail --dataset cifar100 --seeds 42 123 2026
    python -m scripts.run_lab2_longtail --dataset dtd --seeds 42 123 2026 --wandb
"""
from __future__ import annotations

import argparse
import csv
import json
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
    BalancedOnlineRidgeRLS,
    evaluate_longtail_groups,
    make_longtail_class_counts,
    subsample_longtail_indices,
)
from src.foal.metrics import summary_from_task_matrix
from src.foal.projection import SmoothProjection

DEFAULT_RHOS_BY_DATASET = {
    "cifar100": [1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
    "dtd": [1.0, 5.0, 10.0, 20.0, 40.0],
    "aircraft": [1.0, 5.0, 10.0, 20.0, 33.0],
}

METHODS = (
    {"id": "vanilla_foal", "label": "Vanilla F-OAL (Paper)", "weight_mode": "none", "weight_power": 0.0, "norm_tau": 0.0},
    {"id": "norm_calibrated", "label": "F-OAL + Norm Calibration (FD1b)", "weight_mode": "none", "weight_power": 0.0, "norm_tau": 0.70},
    {"id": "balanced_woodbury", "label": "Balanced Woodbury RLS (FD1a+b)", "weight_mode": "online_running", "weight_power": 0.65, "norm_tau": 0.55},
)


def run_single_longtail_trial(
    bundle: dict[str, Any],
    *,
    rho: float,
    seed: int,
    method_spec: dict[str, Any],
    projection_dim: int = 1000,
    gamma: float = 1.0,
    batch_size: int = 10,
    longtail_mode: str = "permuted_global",
) -> dict[str, Any]:
    """Execute one 10-task one-pass OCIL run under a specified imbalance ratio and method.

    Args:
        bundle: Backbone feature bundle from ``lab2_common``.
        rho: Imbalance ratio ``N_max / N_min``.
        seed: Random seed controlling class order, quota assignment, projection, and batch shuffle.
        method_spec: Dictionary configuring ``BalancedOnlineRidgeRLS``.
        projection_dim: SmoothProjection output width ``D``.
        gamma: Ridge regularization parameter.
        batch_size: Streaming mini-batch size (paper default ``10``).
        longtail_mode: Quota assignment strategy across classes.

    Returns:
        Dictionary of incremental, group-wise, and weight-norm diagnostics.
    """
    dataset = bundle["dataset"]
    num_classes = int(bundle["num_classes"])
    cpt = int(bundle["classes_per_task"])
    num_tasks = num_classes // cpt
    n_max = int(bundle["max_samples_per_class"])

    class_order = make_dataset_class_order(dataset, seed=seed)
    cap = 40 if dataset == "dtd" else None
    tr_pool, te_pool = build_remapped_pools(
        bundle["train_orig_labels"],
        bundle["test_orig_labels"],
        class_order,
        max_train_per_class=cap,
        seed=seed,
    )

    quotas = make_longtail_class_counts(
        num_classes,
        n_max,
        rho,
        min_samples=1,
        mode=longtail_mode,
        classes_per_task=cpt,
        quota_seed=seed,
    )
    selected_tr, split_sha = subsample_longtail_indices(tr_pool, quotas, seed=seed)

    proj = SmoothProjection(
        input_dim=bundle["train_features"].shape[1],
        output_dim=projection_dim,
        seed=seed,
        initialization="official_uniform",
    )
    with torch.no_grad():
        tr_z_all = proj(bundle["train_features"])
        te_z_all = proj(bundle["test_features"])

    clf = BalancedOnlineRidgeRLS(
        feature_dim=projection_dim,
        gamma=gamma,
        weight_mode=method_spec["weight_mode"],
        weight_power=method_spec["weight_power"],
        ref_count=float(n_max) * 0.5,
        norm_tau=method_spec["norm_tau"],
        dtype=torch.float64,
    )

    # Pre-slice test sets per task
    task_test_z: list[torch.Tensor] = []
    task_test_y: list[torch.Tensor] = []
    for t_idx in range(num_tasks):
        t_classes = range(t_idx * cpt, (t_idx + 1) * cpt)
        idxs = [i for c in t_classes for i in te_pool[c]]
        labels = [c for c in t_classes for _ in te_pool[c]]
        task_test_z.append(te_z_all[idxs])
        task_test_y.append(torch.tensor(labels, dtype=torch.long))

    t0 = time.perf_counter()
    matrix: list[list[float]] = []

    for stage in range(num_tasks):
        seen_classes = (stage + 1) * cpt
        clf.expand_classes(seen_classes)

        stage_classes = range(stage * cpt, seen_classes)
        tr_idxs = [i for c in stage_classes for i in selected_tr[c]]
        tr_lbls = [c for c in stage_classes for _ in selected_tr[c]]

        rng = np.random.default_rng(int(seed) * 1009 + stage)
        perm = rng.permutation(len(tr_idxs))
        z_stage = tr_z_all[[tr_idxs[p] for p in perm]]
        y_stage = torch.tensor([tr_lbls[p] for p in perm], dtype=torch.long)

        for start in range(0, len(y_stage), batch_size):
            clf.update(z_stage[start:start + batch_size], y_stage[start:start + batch_size])

        stage_row: list[float] = []
        for eval_t in range(stage + 1):
            preds_t = clf.predict(task_test_z[eval_t])
            stage_row.append(float((preds_t == task_test_y[eval_t]).to(torch.float64).mean().item()))
        matrix.append(stage_row)

    rls_seconds = round(time.perf_counter() - t0, 4)

    # Final stage full-test evaluation across Head / Medium / Tail groups
    all_te_z = torch.cat(task_test_z, dim=0)
    all_te_y = torch.cat(task_test_y, dim=0)
    final_preds = clf.predict(all_te_z).numpy()
    group_metrics = evaluate_longtail_groups(final_preds, all_te_y.numpy(), quotas)

    inc_summary = summary_from_task_matrix(matrix)
    raw_norms = clf.column_norms().numpy()
    eff_norms = torch.linalg.vector_norm(clf.effective_weights(), ord=2, dim=0).cpu().numpy()

    corr_raw = float(np.corrcoef(quotas, raw_norms)[0, 1]) if rho > 1.0 else 0.0
    corr_eff = float(np.corrcoef(quotas, eff_norms)[0, 1]) if (rho > 1.0 and np.std(eff_norms) > 1e-12) else 0.0

    return {
        "dataset": dataset,
        "method": method_spec["id"],
        "method_label": method_spec["label"],
        "rho": float(rho),
        "seed": int(seed),
        "longtail_mode": longtail_mode,
        "total_train_samples": int(sum(quotas)),
        "min_class_quota": int(min(quotas)),
        "max_class_quota": int(max(quotas)),
        "split_sha256": split_sha,
        "average_accuracy": float(inc_summary["average_incremental_accuracy"]) * 100.0,
        "last_accuracy": float(inc_summary["last_accuracy"]) * 100.0,
        "forgetting_past": float(inc_summary["forgetting"]) * 100.0,
        "head_acc": float(group_metrics["head_acc"]) * 100.0,
        "medium_acc": float(group_metrics["medium_acc"]) * 100.0,
        "tail_acc": float(group_metrics["tail_acc"]) * 100.0,
        "harmonic_mean_acc": float(group_metrics["harmonic_mean"]) * 100.0,
        "norm_quota_pearson_raw": corr_raw,
        "norm_quota_pearson_effective": corr_eff,
        "rls_seconds": rls_seconds,
        "quotas": quotas,
        "raw_column_norms": [round(float(x), 6) for x in raw_norms],
        "effective_column_norms": [round(float(x), 6) for x in eff_norms],
        "per_class_acc": [round(float(x) * 100.0, 4) for x in group_metrics["per_class_acc"]],
    }


def aggregate_longtail_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed long-tail runs into sample ``Mean +/- Std`` (`ddof=1`) rows."""
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec["dataset"], rec["method"], float(rec["rho"]))
        grouped.setdefault(key, []).append(rec)

    agg_rows: list[dict[str, Any]] = []
    metric_keys = (
        "average_accuracy",
        "last_accuracy",
        "forgetting_past",
        "head_acc",
        "medium_acc",
        "tail_acc",
        "harmonic_mean_acc",
        "norm_quota_pearson_raw",
    )
    for (dataset, method, rho), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "method_label": items[0]["method_label"],
            "rho": rho,
            "num_seeds": len(items),
            "total_train_samples": items[0]["total_train_samples"],
        }
        for mk in metric_keys:
            vals = np.array([it[mk] for it in items], dtype=np.float64)
            row[f"{mk}_mean"] = round(float(vals.mean()), 4)
            row[f"{mk}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)
        agg_rows.append(row)
    return agg_rows


def main() -> None:
    """CLI entry point for Lab 2 Experiment E1 (Long-Tail OCIL)."""
    parser = argparse.ArgumentParser(description="Run Lab 2 E1 Long-Tail OCIL Stress-Test & FD1 Remedies.")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "dtd", "aircraft"])
    parser.add_argument("--rhos", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--projection-dim", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--mode", default="permuted_global", choices=["permuted_global", "within_task", "monotonic"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/lab2_empirical/e1_longtail")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )
    rhos = args.rhos if args.rhos is not None else DEFAULT_RHOS_BY_DATASET[args.dataset]
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Lab2-E1] Loading backbone features for dataset='{args.dataset}' (device={device})...")
    bundle = extract_or_load_backbone_features(
        args.dataset,
        device=device,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    all_records: list[dict[str, Any]] = []
    for rho in rhos:
        for m_spec in METHODS:
            for seed in args.seeds:
                rec = run_single_longtail_trial(
                    bundle,
                    rho=rho,
                    seed=seed,
                    method_spec=m_spec,
                    projection_dim=args.projection_dim,
                    gamma=args.gamma,
                    batch_size=args.batch_size,
                    longtail_mode=args.mode,
                )
                all_records.append(rec)
                print(
                    f"  rho={rho:5.1f} | {m_spec['id']:17s} | seed={seed:4d} -> "
                    f"A_bar={rec['average_accuracy']:6.2f}% | A_10={rec['last_accuracy']:6.2f}% | "
                    f"Head={rec['head_acc']:6.2f}% | Tail={rec['tail_acc']:6.2f}% | r(||w||,N)={rec['norm_quota_pearson_raw']:+.3f}"
                )

                wb_run = init_lab2_wandb_run(
                    enabled=args.wandb,
                    offline=args.wandb_offline,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=f"lab2_e1_{args.dataset}_{m_spec['id']}_rho{int(rho)}_s{seed}",
                    group=f"lab2_e1_longtail_{args.dataset}",
                    extra_tags=["e1-longtail", args.dataset, m_spec["id"]],
                    config={
                        "experiment": "E1_longtail_ocil",
                        "dataset": args.dataset,
                        "method": m_spec["id"],
                        "imbalance_ratio_rho": rho,
                        "seed": seed,
                        "projection_dim": args.projection_dim,
                        "gamma": args.gamma,
                        "batch_size": args.batch_size,
                        "longtail_mode": args.mode,
                        "provenance": bundle["provenance"],
                    },
                )
                if wb_run is not None:
                    wb_run.log({
                        "average_accuracy": rec["average_accuracy"],
                        "last_accuracy": rec["last_accuracy"],
                        "forgetting_past": rec["forgetting_past"],
                        "head_acc": rec["head_acc"],
                        "medium_acc": rec["medium_acc"],
                        "tail_acc": rec["tail_acc"],
                        "harmonic_mean_acc": rec["harmonic_mean_acc"],
                        "norm_quota_pearson_raw": rec["norm_quota_pearson_raw"],
                    })
                    wb_run.finish()

    agg_rows = aggregate_longtail_runs(all_records)

    # Save detailed JSON (keeping full norm arrays for seed=42 representative plots) and CSV summary
    compact_records = []
    for r in all_records:
        rc = dict(r)
        if r["seed"] != args.seeds[0]:
            rc.pop("quotas", None)
            rc.pop("raw_column_norms", None)
            rc.pop("effective_column_norms", None)
            rc.pop("per_class_acc", None)
        compact_records.append(rc)

    summary_payload = {
        "experiment": "Lab2_E1_LongTail_OCIL",
        "dataset": args.dataset,
        "provenance": bundle["provenance"],
        "seeds": args.seeds,
        "rhos": rhos,
        "longtail_mode": args.mode,
        "aggregated": agg_rows,
        "runs": compact_records,
    }
    (out_dir / "e1_longtail_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "e1_longtail_aggregated.csv"
    if agg_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)

    print(f"[Lab2-E1] Saved aggregated summary to {csv_path}")


if __name__ == "__main__":
    main()
