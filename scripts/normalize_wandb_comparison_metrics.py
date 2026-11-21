"""
Script: scripts/normalize_wandb_comparison_metrics.py

Standardizes comparison metrics across the 4 core CIFAR-100 runs in W&B project
'hcmus-foal-reproduction' by populating unified summary keys:
  1. Average Incremental Accuracy:
     Key: comparison/average_incremental_accuracy_pct
  2. Past Forgetting (9 past tasks, K=10):
     Key: comparison/forgetting_past_tasks_pct

Target Runs:
1. F-OAL: c3_d_bilinear_raw (ID: 3v53tmq5)
   - Accuracy Source   : summary/average_incremental_accuracy_pct
   - Forgetting Source : summary/forgetting_past_tasks_pct
2. ER: headline_er_replay_buffer5000_seed42 (ID: xlcme1c3)
   - Accuracy Source   : final/average_incremental_accuracy
   - Forgetting Source : final/forgetting_past_tasks
3. Official LwF: headline_lwf_official_protocol_seed42 (ID: c5pjewk4)
   - Accuracy Source   : final/average_incremental_accuracy
   - Forgetting Source : final/forgetting_past_tasks
4. Adapted LwF: headline_lwf_adapted_adamw_seed42 (ID: ifl3rz2u)
   - Accuracy Source   : final/average_incremental_accuracy
   - Forgetting Source : final/forgetting_past_tasks
"""

import argparse
import csv
import logging
import math
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Ensure UTF-8 output
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("wandb").setLevel(logging.ERROR)

RUN_SPECS: List[Dict[str, Any]] = [
    {
        "method": "F-OAL (Full)",
        "run_id": "3v53tmq5",
        "expected_name": "c3_d_bilinear_raw",
        "matrix_path": Path(
            "results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/task_accuracy.csv"
        ),
        "metrics": {
            "accuracy": {
                "source_key": "summary/average_incremental_accuracy_pct",
                "target_key": "comparison/average_incremental_accuracy_pct",
                "unit": "%",
            },
            "forgetting": {
                "source_key": "summary/forgetting_past_tasks_pct",
                "target_key": "comparison/forgetting_past_tasks_pct",
                "unit": "%",
            },
        },
    },
    {
        "method": "ER (Buffer=5000 Replay)",
        "run_id": "xlcme1c3",
        "expected_name": "headline_er_replay_buffer5000_seed42",
        "matrix_path": Path(
            "results/headline_full_gpu_verified/runs/headline/er/task_matrix.csv"
        ),
        "metrics": {
            "accuracy": {
                "source_key": "final/average_incremental_accuracy",
                "target_key": "comparison/average_incremental_accuracy_pct",
                "unit": "%",
            },
            "forgetting": {
                "source_key": "final/forgetting_past_tasks",
                "target_key": "comparison/forgetting_past_tasks_pct",
                "unit": "%",
            },
        },
    },
    {
        "method": "Official LwF (SGD + Dynamic)",
        "run_id": "c5pjewk4",
        "expected_name": "headline_lwf_official_protocol_seed42",
        "matrix_path": Path(
            "results/headline_official_lwf_verified/runs/headline/lwf_official_cifar100/task_matrix.csv"
        ),
        "metrics": {
            "accuracy": {
                "source_key": "final/average_incremental_accuracy",
                "target_key": "comparison/average_incremental_accuracy_pct",
                "unit": "%",
            },
            "forgetting": {
                "source_key": "final/forgetting_past_tasks",
                "target_key": "comparison/forgetting_past_tasks_pct",
                "unit": "%",
            },
        },
    },
    {
        "method": "Adapted LwF (AdamW + Static)",
        "run_id": "ifl3rz2u",
        "expected_name": "headline_lwf_adapted_adamw_seed42",
        "matrix_path": Path(
            "results/headline_full_gpu_verified/runs/headline/lwf/task_matrix.csv"
        ),
        "metrics": {
            "accuracy": {
                "source_key": "final/average_incremental_accuracy",
                "target_key": "comparison/average_incremental_accuracy_pct",
                "unit": "%",
            },
            "forgetting": {
                "source_key": "final/forgetting_past_tasks",
                "target_key": "comparison/forgetting_past_tasks_pct",
                "unit": "%",
            },
        },
    },
]


def load_task_matrix(matrix_path: Path) -> List[List[float]]:
    """Load lower-triangular task accuracy matrix from CSV."""
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    rows: List[List[float]] = []
    with open(matrix_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if "stage" in first_line:
            # Long format (stage, task, accuracy)
            reader = csv.DictReader(f)
            for r in reader:
                s, t, acc = int(r["stage"]), int(r["task"]), float(r["accuracy"])
                while len(rows) < s:
                    rows.append([])
                rows[s - 1].append(acc)
        else:
            # Square/triangular CSV
            reader = csv.reader(f)
            for line in reader:
                if line and not line[0].startswith("#"):
                    rows.append([float(x) for x in line if x.strip() != ""])
    return rows


def compute_past_forgetting_from_matrix(
    matrix: List[List[float]],
) -> Dict[str, float]:
    """
    Compute Past Forgetting across the 9 past tasks (K=10).
    Returns both unclipped (Eq. 17-18) and non-negative clipped definitions.
    """
    num_stages = len(matrix)
    if num_stages <= 1:
        return {"unclipped": 0.0, "clipped": 0.0, "official_code": 0.0}

    past_tasks = num_stages - 1
    unclipped_drops = []
    clipped_drops = []

    for j in range(past_tasks):
        peak_acc = max(matrix[t][j] for t in range(j, num_stages - 1))
        final_acc = matrix[-1][j]
        drop = peak_acc - final_acc
        unclipped_drops.append(drop)
        clipped_drops.append(max(0.0, drop))

    # Official code definition (experiment/metrics.py: denominator=num_stages, peak includes final stage)
    code_drops = []
    for j in range(num_stages):
        peak_acc = max(matrix[t][j] for t in range(j, num_stages))
        final_acc = matrix[-1][j]
        code_drops.append(max(0.0, peak_acc - final_acc))

    return {
        "unclipped": (sum(unclipped_drops) / len(unclipped_drops)) * 100.0,
        "clipped": (sum(clipped_drops) / len(clipped_drops)) * 100.0,
        "official_code": (sum(code_drops) / len(code_drops)) * 100.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize comparison metrics on W&B for CIFAR-100"
    )
    parser.add_argument(
        "--entity",
        default="caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm",
        help="W&B entity name",
    )
    parser.add_argument(
        "--project",
        default="hcmus-foal-reproduction",
        help="W&B project name",
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "forgetting", "all"],
        default="forgetting",
        help="Metric family to standardize: 'accuracy', 'forgetting', or 'all'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and check metrics without making updates to W&B",
    )
    return parser.parse_args()


def process_metric(
    api: Any,
    run: Any,
    run_spec: Dict[str, Any],
    metric_type: str,
    dry_run: bool,
) -> Dict[str, Any]:
    metric_cfg = run_spec["metrics"][metric_type]
    source_key = metric_cfg["source_key"]
    target_key = metric_cfg["target_key"]
    run_id = run_spec["run_id"]
    method = run_spec["method"]
    actual_name = run.name

    # 1. Check source key exists
    if source_key not in run.summary:
        raise KeyError(
            f"Source metric '{source_key}' not found in run '{actual_name}' (ID: {run_id})."
        )

    raw_val = run.summary[source_key]

    # 2. Check numeric and finite
    if not isinstance(raw_val, (int, float)) or not math.isfinite(raw_val):
        raise ValueError(
            f"Metric '{source_key}' in run {run_id} is not numeric/finite: {raw_val}"
        )

    val_float = float(raw_val)

    # 3. Check percentage unit [0.0, 100.0]
    if not (0.0 <= val_float <= 100.0):
        raise ValueError(
            f"Metric '{source_key}' value {val_float} outside percentage range [0, 100]!"
        )

    # 4. Matrix cross-validation for forgetting
    matrix_info = ""
    if metric_type == "forgetting":
        mat = load_task_matrix(run_spec["matrix_path"])
        f_calcs = compute_past_forgetting_from_matrix(mat)
        f_unclipped = f_calcs["unclipped"]
        f_clipped = f_calcs["clipped"]
        f_code = f_calcs["official_code"]

        # Check match with matrix unclipped (which is Eq. 17 & 18 implemented in codebase)
        diff_unclipped = abs(val_float - f_unclipped)
        if diff_unclipped > 1e-4:
            raise ValueError(
                f"Discrepancy detected for {method}! W&B source {val_float:.6f}% != matrix {f_unclipped:.6f}%"
            )

        if abs(f_unclipped - f_clipped) > 1e-4:
            matrix_info = f"Matrix: unclipped={f_unclipped:.4f}%, clipped={f_clipped:.4f}%, code={f_code:.4f}%"
        else:
            matrix_info = f"Matrix: {f_unclipped:.4f}% (matches clipped)"

    existing_target_val = run.summary.get(target_key)
    is_already_set = False
    if existing_target_val is not None:
        if abs(float(existing_target_val) - val_float) < 1e-9:
            is_already_set = True

    status = "PENDING"
    if dry_run:
        status = "DRY_RUN_CHECKED" if not is_already_set else "DRY_RUN_ALREADY_EXISTS"
        return {
            "method": method,
            "name": actual_name,
            "id": run_id,
            "metric_type": metric_type,
            "target_key": target_key,
            "source_key": source_key,
            "source_val": val_float,
            "before_val": existing_target_val,
            "after_val": val_float if not is_already_set else existing_target_val,
            "matrix_info": matrix_info,
            "status": status,
        }

    # 5. Live update with verification
    if is_already_set:
        status = "ALREADY_UP_TO_DATE"
        verified_val = existing_target_val
    else:
        run.summary[target_key] = val_float
        run.summary.update()
        status = "UPDATED_AND_VERIFIED"

        # Re-fetch from API to verify
        run_path = f"{run.entity}/{run.project}/{run_id}"
        re_run = api.run(run_path)
        verified_val = re_run.summary.get(target_key)
        if verified_val is None or abs(float(verified_val) - val_float) >= 1e-9:
            raise RuntimeError(
                f"Verification failed for {actual_name}! Expected {val_float}, got {verified_val}"
            )

    return {
        "method": method,
        "name": actual_name,
        "id": run_id,
        "metric_type": metric_type,
        "target_key": target_key,
        "source_key": source_key,
        "source_val": val_float,
        "before_val": existing_target_val,
        "after_val": verified_val,
        "matrix_info": matrix_info,
        "status": status,
    }


def main():
    args = parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb package is not installed.")
        sys.exit(1)

    metrics_to_run = (
        ["accuracy", "forgetting"]
        if args.metric == "all"
        else [args.metric]
    )

    print("=" * 85)
    print("W&B COMPARISON METRICS NORMALIZATION")
    print(f"Target Entity : {args.entity}")
    print(f"Target Project: {args.project}")
    print(f"Metrics       : {metrics_to_run}")
    print(f"Mode          : {'DRY RUN (Read-Only)' if args.dry_run else 'LIVE UPDATE'}")
    print("=" * 85)

    api = wandb.Api()
    report_rows = []

    for run_spec in RUN_SPECS:
        run_id = run_spec["run_id"]
        expected_name = run_spec["expected_name"]
        run_path = f"{args.entity}/{args.project}/{run_id}"

        try:
            run = api.run(run_path)
        except Exception as e:
            print(f"ERROR: Could not fetch run at {run_path}: {e}")
            sys.exit(1)

        actual_name = run.name
        if actual_name != expected_name:
            print(
                f"WARNING: Run ID {run_id} name '{actual_name}' != expected '{expected_name}'."
            )

        for m_type in metrics_to_run:
            try:
                row = process_metric(api, run, run_spec, m_type, args.dry_run)
                report_rows.append(row)
            except Exception as e:
                print(f"ERROR processing {m_type} for {run_spec['method']} ({run_id}): {e}")
                sys.exit(1)

    # Print Summary Report Table
    print("\n" + "=" * 125)
    print(
        f"{'Method':<25} | {'Run ID':<10} | {'Metric Target':<38} | {'Before':<10} | {'After (Target)':<14} | {'Status'}"
    )
    print("-" * 125)
    for r in report_rows:
        b_str = "None" if r["before_val"] is None else f"{r['before_val']:.4f}%"
        a_str = f"{r['after_val']:.4f}%"
        print(
            f"{r['method']:<25} | {r['id']:<10} | {r['target_key']:<38} | {b_str:<10} | {a_str:<14} | {r['status']}"
        )
        if r.get("matrix_info"):
            print(f"  --> {r['matrix_info']}")
    print("=" * 125)

    print("\n[VERIFICATION SUMMARY]")
    all_ok = all(
        r["status"] in [
            "UPDATED_AND_VERIFIED",
            "ALREADY_UP_TO_DATE",
            "DRY_RUN_CHECKED",
            "DRY_RUN_ALREADY_EXISTS",
        ]
        for r in report_rows
    )
    print(f"All runs successfully verified: {all_ok}")
    print(f"Processed metrics: {metrics_to_run}\n")


if __name__ == "__main__":
    main()
