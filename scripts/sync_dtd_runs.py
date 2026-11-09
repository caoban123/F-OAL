"""Sync existing DTD Table 4 ablation results from local disk to Weights & Biases.

Usage:
    python scripts/sync_dtd_runs.py
    python scripts/sync_dtd_runs.py --entity caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def sync_runs(
    results_dir: Path = Path("results/dtd"),
    project: str = "hcmus-foal-reproduction",
    entity: str = "caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm",
    base_url: str = "https://api.forge.coreweave.com",
    group: str = "table4_ablation_dtd",
) -> None:
    import wandb

    if base_url:
        os.environ["WANDB_BASE_URL"] = base_url

    conditions = [
        ("A0", "dtd_a0_full", "Full F-OAL (FF On, SP On)", True, True, 1000),
        ("A1", "dtd_a1_without_ff", "w/o Feature Fusion (FF Off, SP On)", False, True, 1000),
        ("A2", "dtd_a2_without_sp", "w/o Smooth Projection (FF On, SP Off)", True, False, 768),
        ("A3", "dtd_a3_without_both", "w/o Both (FF Off, SP Off)", False, False, 768),
    ]

    print("=" * 80)
    print(f"Syncing DTD Table 4 Results to W&B")
    print(f"Target: {base_url} -> {entity}/{project} (Group: {group})")
    print("=" * 80)

    for cond, folder_name, desc, ff, sp, dim in conditions:
        run_dir = results_dir / folder_name
        metrics_file = run_dir / "metrics.json"
        metadata_file = run_dir / "metadata.json"
        csv_file = run_dir / "task_accuracy.csv"

        if not metrics_file.exists():
            print(f"[-] Skipping {cond} ({folder_name}): metrics.json not found in {run_dir}")
            continue

        print(f"\n[+] Syncing {cond}: {desc} from {run_dir}...")
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}

        run_name = f"{folder_name}_seed42"
        tags = ["dtd", "ablation", "table4", cond.lower(), f"ff_{ff}", f"sp_{sp}"]

        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=run_name,
            config=metadata,
            tags=tags,
            reinit=True,
        )

        # Log sequential task progression
        stage_accs = metrics.get("stage_accuracies", [])
        for task_idx, acc in enumerate(stage_accs, start=1):
            run.log({
                "task": task_idx,
                "stage_accuracy": acc,
                "stage_accuracy_pct": acc * 100.0 if acc <= 1.0 else acc,
                "samples_seen": task_idx * 160,
            })

        # Summary metrics
        a_bar = metrics.get("average_incremental_accuracy", 0.0)
        a_last = metrics.get("last_accuracy", 0.0)
        f_past = metrics.get("forgetting_past_tasks", 0.0)
        f_code = metrics.get("forgetting_official_code", 0.0)

        run.summary["summary/average_incremental_accuracy"] = a_bar if a_bar <= 1.0 else a_bar / 100.0
        run.summary["summary/average_incremental_accuracy_pct"] = a_bar * 100.0 if a_bar <= 1.0 else a_bar
        run.summary["summary/last_accuracy"] = a_last if a_last <= 1.0 else a_last / 100.0
        run.summary["summary/last_accuracy_pct"] = a_last * 100.0 if a_last <= 1.0 else a_last
        run.summary["summary/forgetting_past_tasks"] = f_past if f_past <= 1.0 else f_past / 100.0
        run.summary["summary/forgetting_past_tasks_pct"] = f_past * 100.0 if f_past <= 1.0 else f_past
        run.summary["summary/forgetting_official_code"] = f_code if f_code <= 1.0 else f_code / 100.0
        run.summary["summary/forgetting_official_code_pct"] = f_code * 100.0 if f_code <= 1.0 else f_code
        run.summary["summary/total_duration_seconds"] = metrics.get("total_duration_seconds", 0.0)

        # Upload evaluation artifacts
        art = wandb.Artifact(name=f"{run_name}_results", type="evaluation")
        art.add_file(str(metrics_file))
        if metadata_file.exists():
            art.add_file(str(metadata_file))
        if csv_file.exists():
            art.add_file(str(csv_file))
        run.log_artifact(art)

        try:
            run_url = getattr(run, "url", None) or getattr(run, "get_url", lambda: "")() or "uploaded"
            print(f"    SUCCESS: Synced {cond} -> {run_url}")
        except Exception:
            print(f"    SUCCESS: Synced {cond}")
        run.finish()

    print("\n" + "=" * 80)
    print("ALL AVAILABLE DTD RUNS SYNCED TO W&B SUCCESSFULLY!")
    print(f"View runs at: https://forge.coreweave.com/wandb/{entity}/{project}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Sync DTD results to Weights & Biases")
    parser.add_argument("--results-dir", default="results/dtd", help="Directory containing DTD condition folders")
    parser.add_argument("--entity", default="caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm", help="W&B entity")
    parser.add_argument("--project", default="hcmus-foal-reproduction", help="W&B project")
    parser.add_argument("--base-url", default="https://api.forge.coreweave.com", help="W&B base URL")
    parser.add_argument("--group", default="table4_ablation_dtd", help="W&B group")
    args = parser.parse_args()

    sync_runs(
        results_dir=Path(args.results_dir),
        project=args.project,
        entity=args.entity,
        base_url=args.base_url,
        group=args.group,
    )


if __name__ == "__main__":
    main()
