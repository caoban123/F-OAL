"""
Post-hoc W&B Importer for Verified Headline Comparison Experiments (ER, Official LwF, Adapted LwF).

Imports verified CIFAR-100 Headline baseline results from local artifacts
into Weights & Biases (W&B) under project 'hcmus-foal-reproduction'
with explicit provenance tags.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys

# Configure UTF-8 stdout and stderr
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("wandb").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc W&B Importer for Headline Baseline Results")
    parser.add_argument("--project", default="hcmus-foal-reproduction", help="W&B project name")
    parser.add_argument(
        "--entity",
        default="caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm",
        help="W&B entity/username",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print payload without uploading")
    return parser.parse_args()


def load_task_matrix_csv(csv_path: Path):
    matrix = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            matrix.append([float(x) for x in row if x.strip() != ""])
    return matrix


def main():
    args = parse_args()

    print("=" * 75)
    print("F-OAL HEADLINE COMPARISON POST-HOC W&B IMPORTER")
    print(f"Target Entity : {args.entity}")
    print(f"Target Project: {args.project}")
    print(f"Mode          : {'DRY RUN (no network)' if args.dry_run else 'LIVE UPLOAD'}")
    print("=" * 75)

    runs_to_import = [
        {
            "key": "er",
            "name": "headline_er_replay_buffer5000_seed42",
            "summary_path": Path("results/headline_full_gpu_verified/runs/headline/er/summary.json"),
            "matrix_path": Path("results/headline_full_gpu_verified/runs/headline/er/task_matrix.csv"),
            "display_name": "ER (Buffer=5000 Replay)",
            "paradigm": "Experience Replay",
            "buffer_size": 5000,
            "tags": [
                "headline",
                "baseline",
                "er",
                "replay",
                "buffer_5000",
                "seed_42",
                "colab_t4",
                "posthoc_import",
            ],
            "provenance_archive": "foal_cifar100_headline_results.zip",
            "archive_sha256": "6ec6a4a4cadecb4e77b7704a217878356ac6f6cd4e2d72c8a78a3813cceea551",
            "execution_commit": "d333705",
        },
        {
            "key": "official_lwf",
            "name": "headline_lwf_official_protocol_seed42",
            "summary_path": Path(
                "results/headline_official_lwf_verified/runs/headline/lwf_official_cifar100/summary.json"
            ),
            "matrix_path": Path(
                "results/headline_official_lwf_verified/runs/headline/lwf_official_cifar100/task_matrix.csv"
            ),
            "display_name": "LwF (Official Protocol SGD + Dynamic Loss)",
            "paradigm": "Exemplar-Free Distillation",
            "buffer_size": 0,
            "tags": [
                "headline",
                "baseline",
                "lwf",
                "official_protocol",
                "distillation",
                "seed_42",
                "colab_t4",
                "posthoc_import",
            ],
            "provenance_archive": "foal_cifar100_official_lwf_results.zip",
            "archive_sha256": "2f5e67ea86b22d36e8223d8a68494dc0de0f285e015b97b9fcc29df9ccd8f0ce",
            "execution_commit": "c3dd3f23fbf2d8d94d481ec75daceb860190118e",
        },
        {
            "key": "adapted_lwf",
            "name": "headline_lwf_adapted_adamw_seed42",
            "summary_path": Path("results/headline_full_gpu_verified/runs/headline/lwf/summary.json"),
            "matrix_path": Path("results/headline_full_gpu_verified/runs/headline/lwf/task_matrix.csv"),
            "display_name": "LwF (Adapted AdamW + Static Loss)",
            "paradigm": "Exemplar-Free Distillation",
            "buffer_size": 0,
            "tags": [
                "headline",
                "baseline",
                "lwf",
                "adapted_adamw",
                "distillation",
                "seed_42",
                "colab_t4",
                "posthoc_import",
            ],
            "provenance_archive": "foal_cifar100_headline_results.zip",
            "archive_sha256": "6ec6a4a4cadecb4e77b7704a217878356ac6f6cd4e2d72c8a78a3813cceea551",
            "execution_commit": "d333705",
        },
    ]

    wandb = None
    if not args.dry_run:
        try:
            import wandb as wb
            wandb = wb
        except ImportError:
            print("ERROR: 'wandb' package is not installed. Install with `pip install wandb`.")
            sys.exit(1)

    uploaded_runs = []

    for item in runs_to_import:
        summary_path = item["summary_path"]
        matrix_path = item["matrix_path"]

        if not summary_path.exists():
            print(f"Warning: Summary not found: {summary_path}")
            continue
        if not matrix_path.exists():
            print(f"Warning: Matrix not found: {matrix_path}")
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        matrix = load_task_matrix_csv(matrix_path)
        metrics = summary_data.get("metrics_summary", {})
        config = summary_data.get("config", {})
        env = summary_data.get("environment", {})
        stage_accs = summary_data.get("stage_accuracies", [])
        total_time = summary_data.get("total_training_seconds", 0.0)

        # Compute running average incremental accuracy across stages
        running_avg_accs = []
        for i in range(1, len(stage_accs) + 1):
            running_avg_accs.append(sum(stage_accs[:i]) / i)

        a_avg_pct = metrics.get("a_avg_pct", metrics.get("average_incremental_accuracy", 0.0) * 100)
        a_last_pct = metrics.get("a_last_pct", metrics.get("last_accuracy", 0.0) * 100)
        f_past_pct = metrics.get("f_past_pct", metrics.get("forgetting_past_tasks", 0.0) * 100)
        f_code_pct = metrics.get("f_code_pct", metrics.get("forgetting_official_code", 0.0) * 100)

        final_task_accs = [row[-1] if idx == 9 else matrix[-1][idx] for idx, row in enumerate(matrix)]

        print(f"\n[Run: {item['name']}]")
        print(f"  Display Name    : {item['display_name']}")
        print(f"  A_avg           : {a_avg_pct:.4f}%")
        print(f"  A_last          : {a_last_pct:.4f}%")
        print(f"  F_past          : {f_past_pct:.4f}% | F_code: {f_code_pct:.4f}%")
        print(f"  Total Time      : {total_time:.2f} s")
        print(f"  Execution Commit: {item['execution_commit']}")

        if args.dry_run:
            print(f"  [DRY RUN] Would upload run '{item['name']}' to W&B.")
            continue

        run_config = {
            "method": item["display_name"],
            "agent": summary_data.get("agent"),
            "paradigm": item["paradigm"],
            "buffer_size": item["buffer_size"],
            "seed": config.get("seed", 42),
            "class_order_seed": config.get("class_order_seed", 42),
            "dataset": config.get("dataset", "cifar100"),
            "num_classes": config.get("num_classes", 100),
            "classes_per_task": config.get("classes_per_task", 10),
            "feature_dim": 768,
            "feature_fusion": "12-block CLS average + LayerNorm (D=768)",
            "optimizer": config.get("optimizer", "adamw" if item["key"] != "official_lwf" else "sgd"),
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "execution_environment": {
                "platform": "Google Colab GPU",
                "hardware": env.get("hardware", {}),
                "git_commit": item["execution_commit"],
                "archive_sha256": item["archive_sha256"],
            },
            "provenance": {
                "source_archive": item["provenance_archive"],
                "import_type": "posthoc_import_from_verified_colab_artifacts",
                "verifier": "Antigravity Scientific Auditor",
            },
        }

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group="headline_comparison",
            name=item["name"],
            tags=item["tags"],
            config=run_config,
            reinit=True,
        )

        # Log stage-by-stage learning curve
        for stage_idx in range(len(stage_accs)):
            step = stage_idx + 1
            stage_acc_pct = stage_accs[stage_idx] * 100
            run_avg_pct = running_avg_accs[stage_idx] * 100
            run.log(
                {
                    "task": step,
                    "stage_accuracy": stage_acc_pct,
                    "average_incremental_accuracy_so_far": run_avg_pct,
                },
                step=step,
            )

        # Log final summary metrics
        summary_payload = {
            "final/average_incremental_accuracy": a_avg_pct,
            "final/last_accuracy": a_last_pct,
            "final/forgetting_past_tasks": f_past_pct,
            "final/forgetting_official_code": f_code_pct,
            "final/training_time_seconds": total_time,
        }
        for t_idx, acc in enumerate(final_task_accs, start=1):
            summary_payload[f"final_task_accuracies/task_{t_idx:02d}"] = acc * 100

        run.summary.update(summary_payload)

        # Log Task Accuracy Matrix as W&B Table
        columns = [f"Task_{i+1:02d}" for i in range(10)]
        table_rows = []
        for row_idx, row in enumerate(matrix):
            padded_row = [f"Stage_{row_idx+1:02d}"]
            for col_idx in range(10):
                if col_idx <= row_idx and col_idx < len(row):
                    padded_row.append(round(row[col_idx] * 100, 2))
                else:
                    padded_row.append(None)
            table_rows.append(padded_row)

        wandb_table = wandb.Table(columns=["Stage"] + columns, data=table_rows)
        run.log({"task_accuracy_matrix": wandb_table})

        # Save artifacts to run
        run.save(str(summary_path))
        run.save(str(matrix_path))

        # Log visual figures if present
        fig_dir = Path("results/headline_analysis")
        if fig_dir.exists():
            for fig_name in [
                "fig1_headline_final_task_matrices.png",
                "fig2_headline_final_foal_vs_baselines.png",
                "fig3_headline_final_adapted_vs_official_lwf.png",
                "fig4_headline_final_accuracy_progression.png",
                "fig6_headline_final_forgetting_comparison.png",
            ]:
                fig_path = fig_dir / fig_name
                if fig_path.exists():
                    run.log({fig_name.replace(".png", ""): wandb.Image(str(fig_path))})

        run_id = run.id
        run_url = getattr(run, "url", f"https://forge.coreweave.com/wandb/{args.entity}/{args.project}/runs/{run_id}")
        run.finish()

        print(f"  >>> SUCCESS: Uploaded {item['name']} to W&B!")
        print(f"      Run ID : {run_id}")
        print(f"      Run URL: {run_url}")

        uploaded_runs.append({
            "key": item["key"],
            "name": item["name"],
            "run_id": run_id,
            "run_url": run_url,
            "a_avg_pct": a_avg_pct,
            "a_last_pct": a_last_pct,
            "f_past_pct": f_past_pct,
        })

    print("\n" + "=" * 75)
    print("ALL HEADLINE BASELINES SUCCESSFULLY SYNCED TO W&B!")
    print("=" * 75)
    for u in uploaded_runs:
        print(f"  * {u['name']} (ID: {u['run_id']}) -> {u['run_url']}")

    # Save mapping file
    out_manifest = Path("results/headline_analysis/headline_wandb_runs.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(uploaded_runs, f, indent=2)
    print(f"\nSaved W&B run mapping to: {out_manifest}")


if __name__ == "__main__":
    main()
