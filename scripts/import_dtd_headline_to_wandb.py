"""W&B Importer & Figure Sync for DTD Section IX Headline Comparison Experiments.

Uploads or synchronizes DTD ER, Official-Protocol LwF, and Adapted LwF baseline
runs, 10x10 W&B Tables, artifact bundles, and Section IX comparative figures to
Weights & Biases (CoreWeave Forge or wandb.ai).

Usage:
    python scripts/import_dtd_headline_to_wandb.py --dry-run
    python scripts/import_dtd_headline_to_wandb.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("wandb").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W&B Importer for DTD Headline Baseline Results")
    parser.add_argument("--results-root", default="runs/headline_dtd", help="Root directory of DTD baseline runs")
    parser.add_argument(
        "--analysis-dir",
        default="results/headline_dtd_analysis",
        help="Directory containing Section IX figures and manifests",
    )
    parser.add_argument("--project", default="hcmus-foal-reproduction", help="W&B project name")
    parser.add_argument(
        "--entity",
        default="caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm",
        help="W&B entity/username",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.forge.coreweave.com",
        help="W&B base URL (CoreWeave Forge or https://api.wandb.ai)",
    )
    parser.add_argument("--group", default="headline_comparison_dtd", help="W&B group name")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without uploading")
    return parser.parse_args()


def load_task_matrix_csv(csv_path: Path) -> List[List[float]]:
    matrix: List[List[float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            matrix.append([float(x) for x in row if x.strip() != ""])
    return matrix


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    analysis_dir = Path(args.analysis_dir)

    if args.base_url and not args.dry_run:
        os.environ["WANDB_BASE_URL"] = args.base_url

    print("=" * 75)
    print("F-OAL DTD SECTION IX HEADLINE COMPARISON W&B SYNC")
    print(f"Target Base URL: {args.base_url}")
    print(f"Target Entity  : {args.entity}")
    print(f"Target Project : {args.project}")
    print(f"Target Group   : {args.group}")
    print(f"Mode           : {'DRY RUN (no network)' if args.dry_run else 'LIVE UPLOAD'}")
    print("=" * 75)

    runs_to_import = [
        {
            "key": "er",
            "name": "headline_dtd_er_replay_buffer5000_seed42",
            "run_dir": results_root / "er",
            "display_name": "ER (Buffer=5000 Replay, DTD)",
            "paradigm": "Experience Replay",
            "buffer_size": 5000,
            "tags": ["dtd", "headline", "baseline", "er", "replay", "buffer_5000", "seed_42", "section_ix"],
        },
        {
            "key": "official_lwf",
            "name": "headline_dtd_lwf_official_protocol_seed42",
            "run_dir": results_root / "lwf_official_dtd",
            "display_name": "LwF (Official Protocol SGD + Dynamic Loss, DTD)",
            "paradigm": "Exemplar-Free Distillation",
            "buffer_size": 0,
            "tags": ["dtd", "headline", "baseline", "lwf", "official_protocol", "distillation", "seed_42", "section_ix"],
        },
        {
            "key": "adapted_lwf",
            "name": "headline_dtd_lwf_adapted_adamw_seed42",
            "run_dir": results_root / "lwf",
            "display_name": "LwF (Adapted AdamW + Static Loss, DTD)",
            "paradigm": "Exemplar-Free Distillation",
            "buffer_size": 0,
            "tags": ["dtd", "headline", "baseline", "lwf", "adapted_adamw", "distillation", "seed_42", "section_ix"],
        },
    ]

    if not args.dry_run:
        import wandb

    uploaded_runs: List[Dict[str, Any]] = []

    for item in runs_to_import:
        run_dir = Path(item["run_dir"])
        summary_path = run_dir / "summary.json"
        matrix_path = run_dir / "task_matrix.csv"
        metadata_path = run_dir / "metadata.json"

        if not summary_path.exists() or not matrix_path.exists():
            print(f"\n[-] Skipping {item['name']}: missing summary.json or task_matrix.csv in {run_dir}")
            continue

        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        matrix = load_task_matrix_csv(matrix_path)
        metrics = summary_data.get("metrics_summary", {})
        config = summary_data.get("config", {})
        stage_accs = summary_data.get("stage_accuracies", [])
        total_time = summary_data.get("total_training_seconds", 0.0)
        final_task_accs = metrics.get("final_task_accuracies", [])

        a_avg_pct = metrics.get("a_avg_pct", metrics.get("average_incremental_accuracy", 0.0) * 100.0)
        a_last_pct = metrics.get("a_last_pct", metrics.get("last_accuracy", 0.0) * 100.0)
        f_past_pct = metrics.get("f_past_pct", metrics.get("forgetting_past_tasks", 0.0) * 100.0)
        f_code_pct = metrics.get("f_code_pct", metrics.get("forgetting_official_code", 0.0) * 100.0)

        print(f"\n[{item['key'].upper()}] {item['name']}")
        print(f"  A_avg={a_avg_pct:.4f}%, A_last={a_last_pct:.4f}%, F_past={f_past_pct:.4f}%, F_code={f_code_pct:.4f}%")

        if args.dry_run:
            uploaded_runs.append(
                {
                    "key": item["key"],
                    "name": item["name"],
                    "run_dir": str(run_dir),
                    "a_avg_pct": a_avg_pct,
                    "a_last_pct": a_last_pct,
                    "f_past_pct": f_past_pct,
                    "f_code_pct": f_code_pct,
                    "dry_run": True,
                }
            )
            continue

        run_config = {
            "method": item["display_name"],
            "paradigm": item["paradigm"],
            "buffer_size": item["buffer_size"],
            "dataset": "tanganke/dtd",
            "num_classes": config.get("num_classes", 40),
            "classes_per_task": config.get("classes_per_task", 4),
            "samples_per_class": config.get("samples_per_class", 40),
            "seed": config.get("seed", 42),
            "backbone": config.get("backbone", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
            "feature_dim": 768,
            "feature_fusion": "12-block CLS average + LayerNorm (D=768)",
            "optimizer": config.get("optimizer", "sgd" if item["key"] == "official_lwf" else "adamw"),
            "learning_rate": config.get("learning_rate", 0.001),
            "batch_size": config.get("batch_size", 10),
        }

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            name=item["name"],
            tags=item["tags"],
            config=run_config,
            reinit=True,
        )

        running_accs: List[float] = []
        for idx, s_acc in enumerate(stage_accs):
            step = idx + 1
            running_accs.append(float(s_acc))
            run_avg_pct = float(sum(running_accs) / len(running_accs)) * 100.0
            diag_acc_pct = float(matrix[idx][idx]) * 100.0 if idx < len(matrix) and idx < len(matrix[idx]) else 0.0
            run.log(
                {
                    "task": step,
                    "stage_accuracy": float(s_acc),
                    "stage_accuracy_pct": float(s_acc) * 100.0,
                    "average_incremental_accuracy_so_far_pct": run_avg_pct,
                    "diagonal_new_task_accuracy_pct": diag_acc_pct,
                },
                step=step,
            )

        summary_payload = {
            "summary/average_incremental_accuracy": a_avg_pct / 100.0,
            "summary/average_incremental_accuracy_pct": a_avg_pct,
            "summary/last_accuracy": a_last_pct / 100.0,
            "summary/last_accuracy_pct": a_last_pct,
            "summary/forgetting_past_tasks": f_past_pct / 100.0,
            "summary/forgetting_past_tasks_pct": f_past_pct,
            "summary/forgetting_official_code": f_code_pct / 100.0,
            "summary/forgetting_official_code_pct": f_code_pct,
            "final/average_incremental_accuracy": a_avg_pct,
            "final/last_accuracy": a_last_pct,
            "final/forgetting_past_tasks": f_past_pct,
            "final/forgetting_official_code": f_code_pct,
            "final/training_time_seconds": total_time,
            "comparison/average_incremental_accuracy_pct": a_avg_pct,
        }
        for t_idx, acc in enumerate(final_task_accs, start=1):
            summary_payload[f"final_task_accuracies/task_{t_idx:02d}"] = float(acc) * 100.0
        run.summary.update(summary_payload)

        n_tasks = len(matrix)
        columns = [f"Task_{i + 1:02d}" for i in range(n_tasks)]
        table_rows = []
        for row_idx, row in enumerate(matrix):
            padded_row: List[Any] = [f"Stage_{row_idx + 1:02d}"]
            for col_idx in range(n_tasks):
                padded_row.append(round(row[col_idx] * 100.0, 2) if col_idx <= row_idx and col_idx < len(row) else None)
            table_rows.append(padded_row)
        run.log({"task_accuracy_matrix": wandb.Table(columns=["Stage"] + columns, data=table_rows)})

        if analysis_dir.exists():
            for fig_name in (
                "fig1_dtd_headline_task_matrices.png",
                "fig2_dtd_headline_foal_vs_baselines.png",
                "fig3_dtd_headline_adapted_vs_official_lwf.png",
                "fig4_dtd_headline_accuracy_progression.png",
                "fig5_dtd_headline_paper_vs_reproduced.png",
                "fig6_dtd_headline_forgetting_comparison.png",
            ):
                fig_path = analysis_dir / fig_name
                if fig_path.exists():
                    run.log({fig_name.replace(".png", ""): wandb.Image(str(fig_path))})

        art = wandb.Artifact(name=f"{item['name']}_results", type="evaluation")
        for fp in (summary_path, matrix_path, metadata_path, run_dir / "metrics.json", run_dir / "task_accuracy.csv"):
            if fp.exists():
                art.add_file(str(fp))
        run.log_artifact(art)

        run_id = run.id
        run_url = getattr(run, "url", f"https://forge.coreweave.com/wandb/{args.entity}/{args.project}/runs/{run_id}")
        run.finish()
        print(f"  >>> Uploaded Run ID {run_id} -> {run_url}")
        uploaded_runs.append(
            {
                "key": item["key"],
                "name": item["name"],
                "run_id": run_id,
                "run_url": run_url,
                "a_avg_pct": a_avg_pct,
                "a_last_pct": a_last_pct,
                "f_past_pct": f_past_pct,
                "f_code_pct": f_code_pct,
            }
        )

    if not args.dry_run and uploaded_runs:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = analysis_dir / "dtd_headline_wandb_runs.json"
        out_manifest.write_text(json.dumps(uploaded_runs, indent=2), encoding="utf-8")
        print(f"\nSaved W&B run mapping to: {out_manifest}")


if __name__ == "__main__":
    main()
