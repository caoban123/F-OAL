"""
Script: scripts/create_wandb_accuracy_progression_run.py

Creates a post-hoc analysis-only W&B run that aggregates the Accuracy Progression
across 10 tasks for all four core CIFAR-100 methods in project 'hcmus-foal-reproduction':
  - F-OAL (c3_d_bilinear_raw, ID: 3v53tmq5)
  - ER (headline_er_replay_buffer5000_seed42, ID: xlcme1c3)
  - Official LwF (headline_lwf_official_protocol_seed42, ID: c5pjewk4)
  - Adapted LwF (headline_lwf_adapted_adamw_seed42, ID: ifl3rz2u)

Features:
  - Exact join on integer 'task' (1..10), ignoring raw '_step'.
  - Validates Task 10 accuracy against final Last Accuracy for all methods.
  - Idempotent: checks for existing run to avoid creating duplicates.
  - Logs unified metrics per task:
      foal_accuracy_pct
      er_accuracy_pct
      lwf_official_accuracy_pct
      lwf_adapted_accuracy_pct
  - No modification to original source runs, metrics, or artifacts.
  - No API keys logged or saved.
"""

import argparse
import logging
import sys
from typing import Dict, Any, List

# Ensure UTF-8 output
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("wandb").setLevel(logging.ERROR)

ANALYSIS_RUN_NAME = "analysis_baseline_accuracy_progression_seed42"
GROUP_NAME = "headline_comparison"
JOB_TYPE = "analysis"

SOURCE_RUNS = [
    {
        "method_key": "foal",
        "display_name": "F-OAL (Full)",
        "run_id": "3v53tmq5",
        "expected_name": "c3_d_bilinear_raw",
        "metric_key": "stage_accuracy_pct",
        "dest_metric": "foal_accuracy_pct",
        "final_acc_key": "summary/last_accuracy_pct",
    },
    {
        "method_key": "er",
        "display_name": "ER (Buffer=5000 Replay)",
        "run_id": "xlcme1c3",
        "expected_name": "headline_er_replay_buffer5000_seed42",
        "metric_key": "stage_accuracy",
        "dest_metric": "er_accuracy_pct",
        "final_acc_key": "final/last_accuracy",
    },
    {
        "method_key": "lwf_official",
        "display_name": "Official LwF (SGD + Dynamic)",
        "run_id": "c5pjewk4",
        "expected_name": "headline_lwf_official_protocol_seed42",
        "metric_key": "stage_accuracy",
        "dest_metric": "lwf_official_accuracy_pct",
        "final_acc_key": "final/last_accuracy",
    },
    {
        "method_key": "lwf_adapted",
        "display_name": "Adapted LwF (AdamW + Static)",
        "run_id": "ifl3rz2u",
        "expected_name": "headline_lwf_adapted_adamw_seed42",
        "metric_key": "stage_accuracy",
        "dest_metric": "lwf_adapted_accuracy_pct",
        "final_acc_key": "final/last_accuracy",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create W&B Accuracy Progression Analysis Run"
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
        help="Extract and verify data without creating W&B run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload and overwrite if analysis run already exists",
    )
    return parser.parse_args()


def extract_task_series(run, metric_key: str) -> Dict[int, float]:
    """
    Extracts {task: accuracy_pct} from run history, ensuring exact join on integer 'task'.
    """
    history_df = run.history(keys=["task", metric_key])
    if history_df.empty:
        raise ValueError(f"Run {run.id} has empty history for keys ['task', '{metric_key}']")

    task_series: Dict[int, float] = {}
    for _, row in history_df.iterrows():
        t_raw = row.get("task")
        v_raw = row.get(metric_key)
        if t_raw is not None and not (isinstance(t_raw, float) and str(t_raw) == "nan"):
            t_int = int(round(float(t_raw)))
            if v_raw is not None and not (isinstance(v_raw, float) and str(v_raw) == "nan"):
                val = float(v_raw)
                task_series[t_int] = val

    return task_series


def main():
    args = parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb package is not installed.")
        sys.exit(1)

    print("=" * 85)
    print("F-OAL W&B ACCURACY PROGRESSION ANALYSIS RUN BUILDER")
    print(f"Target Entity : {args.entity}")
    print(f"Target Project: {args.project}")
    print(f"Run Name      : {ANALYSIS_RUN_NAME}")
    print(f"Group / Type  : {GROUP_NAME} / {JOB_TYPE}")
    print(f"Mode          : {'DRY RUN (Read-Only)' if args.dry_run else 'LIVE W&B RUN'}")
    print("=" * 85)

    api = wandb.Api()

    # 1. Check Idempotency: does the analysis run already exist?
    existing_runs = api.runs(
        f"{args.entity}/{args.project}",
        filters={"display_name": ANALYSIS_RUN_NAME},
    )
    existing_run_list = list(existing_runs)
    if existing_run_list and not args.force and not args.dry_run:
        existing = existing_run_list[0]
        print(f"\n[IDEMPOTENCY CHECK] Analysis run '{ANALYSIS_RUN_NAME}' already exists!")
        print(f"  Run ID : {existing.id}")
        print(f"  Run URL: https://forge.coreweave.com/wandb/{args.entity}/{args.project}/runs/{existing.id}")
        print(f"  Status : {existing.state}")
        print("Use --force to recreate/re-upload if necessary.")
        print("=" * 85)
        return

    # 2. Extract and validate series from all four source runs
    extracted_series: Dict[str, Dict[int, float]] = {}
    final_verifications: List[Dict[str, Any]] = []

    for src in SOURCE_RUNS:
        run_path = f"{args.entity}/{args.project}/{src['run_id']}"
        try:
            r = api.run(run_path)
        except Exception as e:
            print(f"FATAL: Could not fetch run {run_path}: {e}")
            sys.exit(1)

        print(f"\nReading source run: {src['display_name']} ({r.id})...")
        print(f"  Metric Key: {src['metric_key']}")

        series = extract_task_series(r, src["metric_key"])
        extracted_series[src["method_key"]] = series

        # Check exactly 10 tasks present
        tasks_found = sorted(series.keys())
        expected_tasks = list(range(1, 11))
        if tasks_found != expected_tasks:
            print(
                f"FATAL: Run {src['run_id']} has tasks {tasks_found}, expected {expected_tasks}!"
            )
            sys.exit(1)

        # Check values are in percentage range [0.0, 100.0]
        for t, val in series.items():
            if not (0.0 <= val <= 100.0):
                print(
                    f"FATAL: Task {t} value {val} in run {src['run_id']} outside [0, 100]!"
                )
                sys.exit(1)

        # Verify Task 10 matches Final Accuracy in Summary
        task10_val = series[10]
        final_acc_raw = r.summary.get(src["final_acc_key"])
        if final_acc_raw is None:
            # Fallback check for alternate keys
            for alt_key in ["final/last_accuracy", "summary/last_accuracy_pct"]:
                if alt_key in r.summary:
                    final_acc_raw = r.summary[alt_key]
                    break

        if final_acc_raw is None:
            print(f"FATAL: Final accuracy key '{src['final_acc_key']}' not found in {r.id}!")
            sys.exit(1)

        final_acc_val = float(final_acc_raw)
        delta = abs(task10_val - final_acc_val)
        if delta > 1e-4:
            print(
                f"FATAL DIVERGENCE: Run {src['run_id']} Task 10 accuracy ({task10_val:.4f}%) "
                f"does not match final accuracy ({final_acc_val:.4f}%)! Delta: {delta:.6f}"
            )
            sys.exit(1)

        final_verifications.append({
            "method": src["display_name"],
            "run_id": src["run_id"],
            "task10_acc": task10_val,
            "final_acc": final_acc_val,
            "delta": delta,
            "status": "PASS",
        })

    print("\n" + "-" * 85)
    print("TASK 10 FINAL ACCURACY VERIFICATION GATES:")
    for v in final_verifications:
        print(
            f"  [PASS] {v['method']:<30} | Task 10: {v['task10_acc']:.4f}% | "
            f"Final Acc: {v['final_acc']:.4f}% | Delta: {v['delta']:.2e}"
        )
    print("-" * 85)

    # 3. Build unified 10-task dataset
    table_data = []
    print("\nUNIFIED ACCURACY PROGRESSION DATASET (TASKS 1..10):")
    print(
        f"{'Task':<6} | {'F-OAL (%)':<12} | {'ER (%)':<12} | {'Official LwF (%)':<18} | {'Adapted LwF (%)':<16}"
    )
    print("-" * 75)
    for task_id in range(1, 11):
        foal_v = extracted_series["foal"][task_id]
        er_v = extracted_series["er"][task_id]
        lwf_off_v = extracted_series["lwf_official"][task_id]
        lwf_adp_v = extracted_series["lwf_adapted"][task_id]

        print(
            f"{task_id:<6} | {foal_v:<12.4f} | {er_v:<12.4f} | {lwf_off_v:<18.4f} | {lwf_adp_v:<16.4f}"
        )
        table_data.append([task_id, foal_v, er_v, lwf_off_v, lwf_adp_v])

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Data validated. No W&B run created.")
        return

    # 4. Create and upload new W&B analysis-only run
    print("\nCreating W&B analysis run...")

    run_config = {
        "analysis_type": "cross_method_accuracy_progression",
        "nature": "post_hoc_aggregation_only",
        "is_training_run": False,
        "dataset": "cifar100",
        "num_tasks": 10,
        "classes_per_task": 10,
        "seed": 42,
        "join_key": "task (integer 1..10)",
        "source_runs": {
            "foal": {
                "name": "c3_d_bilinear_raw",
                "run_id": "3v53tmq5",
                "metric": "stage_accuracy_pct",
            },
            "er": {
                "name": "headline_er_replay_buffer5000_seed42",
                "run_id": "xlcme1c3",
                "metric": "stage_accuracy",
            },
            "lwf_official": {
                "name": "headline_lwf_official_protocol_seed42",
                "run_id": "c5pjewk4",
                "metric": "stage_accuracy",
            },
            "lwf_adapted": {
                "name": "headline_lwf_adapted_adamw_seed42",
                "run_id": "ifl3rz2u",
                "metric": "stage_accuracy",
            },
        },
        "description": (
            "Comparative Accuracy Progression across 10 tasks for CIFAR-100 headline benchmarks. "
            "Data extracted directly from verified GPU runs. For visualization and analysis only."
        ),
    }

    tags = [
        "analysis",
        "headline_comparison",
        "accuracy_progression",
        "comparative_visualization",
        "cifar100",
        "seed_42",
    ]

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=GROUP_NAME,
        job_type=JOB_TYPE,
        name=ANALYSIS_RUN_NAME,
        tags=tags,
        config=run_config,
        notes="Post-hoc comparative analysis run for Accuracy Progression over 10 tasks (Seed 42). Not a training run.",
        reinit=True,
    )

    # 5. Log 10 sequential history steps exactly by task
    for row in table_data:
        t_id, f_v, e_v, lo_v, la_v = row
        step_payload = {
            "task": t_id,
            "foal_accuracy_pct": f_v,
            "er_accuracy_pct": e_v,
            "lwf_official_accuracy_pct": lo_v,
            "lwf_adapted_accuracy_pct": la_v,
        }
        run.log(step_payload, step=t_id)

    # 6. Log final summary metrics
    run.summary.update({
        "final/foal_accuracy_pct": extracted_series["foal"][10],
        "final/er_accuracy_pct": extracted_series["er"][10],
        "final/lwf_official_accuracy_pct": extracted_series["lwf_official"][10],
        "final/lwf_adapted_accuracy_pct": extracted_series["lwf_adapted"][10],
        "num_tasks": 10,
    })

    # 7. Log W&B Table
    wandb_table = wandb.Table(
        columns=["Task", "F-OAL (%)", "ER (%)", "Official LwF (%)", "Adapted LwF (%)"],
        data=table_data,
    )
    run.log({"accuracy_progression_table": wandb_table})

    new_run_id = run.id
    new_run_url = f"https://forge.coreweave.com/wandb/{args.entity}/{args.project}/runs/{new_run_id}"
    run.finish()

    print("\n" + "=" * 85)
    print("SUCCESSFULLY CREATED AND UPLOADED ANALYSIS RUN!")
    print(f"  Run Name: {ANALYSIS_RUN_NAME}")
    print(f"  Run ID  : {new_run_id}")
    print(f"  Run URL : {new_run_url}")
    print("=" * 85)


if __name__ == "__main__":
    main()
