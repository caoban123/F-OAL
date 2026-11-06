"""
Script: scripts/normalize_wandb_comparison_metrics.py

Standardizes Average Incremental Accuracy for the 4 core CIFAR-100 runs
in W&B project 'hcmus-foal-reproduction' by populating a unified summary key:
    comparison/average_incremental_accuracy_pct

Target Runs:
1. F-OAL: c3_d_bilinear_raw (ID: 3v53tmq5)
   Source: summary/average_incremental_accuracy_pct
2. ER: headline_er_replay_buffer5000_seed42 (ID: xlcme1c3)
   Source: final/average_incremental_accuracy
3. Official LwF: headline_lwf_official_protocol_seed42 (ID: c5pjewk4)
   Source: final/average_incremental_accuracy
4. Adapted LwF: headline_lwf_adapted_adamw_seed42 (ID: ifl3rz2u)
   Source: final/average_incremental_accuracy
"""

import argparse
import sys
from typing import Dict, Any, List

# Ensure UTF-8 output
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TARGET_KEY = "comparison/average_incremental_accuracy_pct"

TARGET_RUNS: List[Dict[str, str]] = [
    {
        "method": "F-OAL (Full)",
        "run_id": "3v53tmq5",
        "expected_name": "c3_d_bilinear_raw",
        "source_key": "summary/average_incremental_accuracy_pct",
    },
    {
        "method": "ER (Buffer=5000 Replay)",
        "run_id": "xlcme1c3",
        "expected_name": "headline_er_replay_buffer5000_seed42",
        "source_key": "final/average_incremental_accuracy",
    },
    {
        "method": "Official LwF (SGD + Dynamic)",
        "run_id": "c5pjewk4",
        "expected_name": "headline_lwf_official_protocol_seed42",
        "source_key": "final/average_incremental_accuracy",
    },
    {
        "method": "Adapted LwF (AdamW + Static)",
        "run_id": "ifl3rz2u",
        "expected_name": "headline_lwf_adapted_adamw_seed42",
        "source_key": "final/average_incremental_accuracy",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize comparison/average_incremental_accuracy_pct on W&B"
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
        "--dry-run",
        action="store_true",
        help="Read and check metrics without making updates to W&B",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb package is not installed.")
        sys.exit(1)

    print("=" * 80)
    print("W&B COMPARISON METRICS NORMALIZATION")
    print(f"Target Entity : {args.entity}")
    print(f"Target Project: {args.project}")
    print(f"Target Key    : {TARGET_KEY}")
    print(f"Mode          : {'DRY RUN (Read-Only)' if args.dry_run else 'LIVE UPDATE'}")
    print("=" * 80)

    api = wandb.Api()
    report_rows = []

    for target in TARGET_RUNS:
        method = target["method"]
        run_id = target["run_id"]
        expected_name = target["expected_name"]
        source_key = target["source_key"]

        run_path = f"{args.entity}/{args.project}/{run_id}"

        # 1. Fetch Run from W&B
        try:
            run = api.run(run_path)
        except Exception as e:
            print(f"ERROR: Could not fetch run at {run_path}: {e}")
            sys.exit(1)

        actual_name = run.name
        if actual_name != expected_name:
            print(
                f"WARNING: Run ID {run_id} has name '{actual_name}', expected '{expected_name}'."
            )

        # 2. Check source key exists
        if source_key not in run.summary:
            print(
                f"ERROR: Source metric '{source_key}' not found in run '{actual_name}' (ID: {run_id})."
            )
            print(f"Available summary keys: {list(run.summary.keys())}")
            sys.exit(1)

        raw_val = run.summary[source_key]

        # 3. Validate value type and percentage range
        if not isinstance(raw_val, (int, float)):
            print(
                f"ERROR: Metric '{source_key}' in run {run_id} is not numeric: {raw_val} ({type(raw_val)})"
            )
            sys.exit(1)

        val_float = float(raw_val)

        # Confirm unit is percentage in [0.0, 100.0]
        if not (0.0 <= val_float <= 100.0):
            print(
                f"ERROR: Metric '{source_key}' value {val_float} outside valid percentage range [0, 100]!"
            )
            sys.exit(1)

        existing_target_val = run.summary.get(TARGET_KEY)

        # Check idempotency
        is_already_set = False
        if existing_target_val is not None:
            if abs(float(existing_target_val) - val_float) < 1e-9:
                is_already_set = True

        status = "PENDING"

        if args.dry_run:
            status = "DRY_RUN_CHECKED" if not is_already_set else "DRY_RUN_ALREADY_EXISTS"
            report_rows.append({
                "method": method,
                "name": actual_name,
                "id": run_id,
                "source_key": source_key,
                "source_val": val_float,
                "before_val": existing_target_val,
                "after_val": val_float if not is_already_set else existing_target_val,
                "status": status,
            })
            continue

        # 4. Perform Update if needed
        if is_already_set:
            status = "ALREADY_UP_TO_DATE"
            verified_val = existing_target_val
        else:
            # Update summary key
            run.summary[TARGET_KEY] = val_float
            run.summary.update()
            status = "UPDATED_AND_VERIFIED"

            # 5. Re-fetch from API to verify
            re_run = api.run(run_path)
            verified_val = re_run.summary.get(TARGET_KEY)
            if verified_val is None or abs(float(verified_val) - val_float) >= 1e-9:
                print(
                    f"FATAL: Verification failed for {actual_name}! Expected {val_float}, got {verified_val}"
                )
                sys.exit(1)

        report_rows.append({
            "method": method,
            "name": actual_name,
            "id": run_id,
            "source_key": source_key,
            "source_val": val_float,
            "before_val": existing_target_val,
            "after_val": verified_val,
            "status": status,
        })

    # Print Final Verification Table
    print("\n" + "=" * 105)
    print(
        f"{'Method':<25} | {'Run Name':<38} | {'Run ID':<10} | {'Before':<10} | {'After (Target)':<14} | {'Status'}"
    )
    print("-" * 105)
    for r in report_rows:
        b_str = "None" if r["before_val"] is None else f"{r['before_val']:.4f}%"
        a_str = f"{r['after_val']:.4f}%"
        print(
            f"{r['method']:<25} | {r['name']:<38} | {r['id']:<10} | {b_str:<10} | {a_str:<14} | {r['status']}"
        )
    print("=" * 105)

    print("\n[VERIFICATION SUMMARY]")
    all_ok = all(
        r["status"] in ["UPDATED_AND_VERIFIED", "ALREADY_UP_TO_DATE", "DRY_RUN_CHECKED", "DRY_RUN_ALREADY_EXISTS"]
        for r in report_rows
    )
    print(f"All 4 runs successfully verified: {all_ok}")
    print(f"Target key added: '{TARGET_KEY}'\n")


if __name__ == "__main__":
    main()
