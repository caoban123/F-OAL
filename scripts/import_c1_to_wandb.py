"""
Post-hoc W&B Importer for Verified Colab C1 GPU Experiments.

Imports verified C1 experiment results (P0, P1, P2) from local artifacts
into Weights & Biases (W&B) under project 'hcmus-foal-reproduction'
with explicit provenance tags.

Tags used:
  - c1_projection
  - colab_a100
  - posthoc_import_from_verified_colab_artifacts
  - reproduction

Usage:
  # Dry-run inspection (no wandb connection needed):
  python -m scripts.import_c1_to_wandb --dry-run

  # Live upload using environment variable WANDB_API_KEY or logged-in wandb:
  python -m scripts.import_c1_to_wandb --project hcmus-foal-reproduction
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Configure UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ZIP_SHA256 = "21eb4e977e739f8380b22f87e1db14937925b69d4a93b9285e7b49a6c2374cab"
COLAB_EXEC_COMMIT = "269f05ccaa8f12d1911e9d806449dccfb46591b9"


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc W&B Importer for C1 Results")
    parser.add_argument("--project", default="hcmus-foal-reproduction", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity/username")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without uploading")
    parser.add_argument("--parsed-json", default="results/c1_audit_parsed.json", help="Path to parsed audit JSON")
    parser.add_argument("--analysis-dir", default="results/c1_analysis", help="Path to c1_analysis directory")
    return parser.parse_args()


def load_c1_data(parsed_path: Path):
    if not parsed_path.exists():
        raise FileNotFoundError(f"Missing parsed audit file: {parsed_path}")
    with open(parsed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    parsed_path = Path(args.parsed_json)
    analysis_dir = Path(args.analysis_dir)
    data = load_c1_data(parsed_path)

    print("=" * 70)
    print("F-OAL C1 POST-HOC W&B IMPORTER")
    print(f"Source JSON   : {parsed_path}")
    print(f"Target Project: {args.project}")
    print(f"Colab Commit  : {COLAB_EXEC_COMMIT[:7]}")
    print(f"ZIP SHA256    : {ZIP_SHA256[:16]}...")
    print(f"Mode          : {'DRY RUN (no network)' if args.dry_run else 'LIVE UPLOAD'}")
    print("=" * 70)

    conditions = [
        ("p0_gaussian_std1", "p0", "gaussian_std1", "Gaussian N(0, 1.0)"),
        ("p1_official_uniform", "p1", "official_uniform", "Official Uniform U(-1/sqrt(d), +1/sqrt(d))"),
        ("p2_gaussian_matched_variance", "p2", "gaussian_matched_variance", "Gaussian Matched Variance"),
    ]

    wandb = None
    if not args.dry_run:
        try:
            import wandb as wb
            wandb = wb
        except ImportError:
            print("ERROR: 'wandb' package is not installed. Install with `pip install wandb` or use --dry-run.")
            sys.exit(1)

    for cond_key, p_id, init_mode, display_name in conditions:
        if cond_key not in data:
            print(f"Warning: condition {cond_key} not found in {parsed_path}")
            continue

        cond_data = data[cond_key]
        recalc = cond_data["recalculated"]
        stage_accs = [acc * 100 for acc in recalc["stage_accuracies"]]
        avg_acc = recalc["average_incremental_accuracy"] * 100
        last_acc = recalc["last_accuracy"] * 100
        fgt_past = recalc["forgetting_past_tasks"] * 100
        fgt_code = recalc["forgetting_official_code"] * 100
        matrix = cond_data["task_matrix"]
        t10_row = [acc * 100 for acc in matrix[-1]]

        run_name = f"c1_{p_id}_{init_mode}_seed42"
        run_config = {
            "experiment": "C1_controlled_smooth_projection",
            "condition_key": p_id,
            "condition_name": display_name,
            "initialization": init_mode,
            "seed": 42,
            "dataset": "cifar100",
            "backbone": "vit_base_patch16_224.augreg_in1k",
            "projection_dim": 1000,
            "gamma": 1.0,
            "batch_size": 10,
            "rls_dtype": "float64",
            "execution_environment": {
                "platform": "Google Colab GPU",
                "gpu": "NVIDIA A100-SXM4 (VRAM ~40-80GB)",
                "execution_commit": COLAB_EXEC_COMMIT,
                "artifact_zip_sha256": ZIP_SHA256,
            },
            "provenance": {
                "source": "foal_c1_projection_results.zip",
                "import_type": "posthoc_import_from_verified_colab_artifacts",
                "verifier": "Antigravity Scientific Auditor",
            },
        }

        tags = [
            "c1_projection",
            "colab_a100",
            "posthoc_import_from_verified_colab_artifacts",
            "reproduction",
            f"init_{init_mode}",
        ]

        print(f"\nProcessing condition: {cond_key} -> Run: {run_name}")
        print(f"  Avg Acc: {avg_acc:.2f}% | Last Acc: {last_acc:.2f}%")
        print(f"  Forgetting Past: {fgt_past:.2f}% | Forgetting Code: {fgt_code:.2f}%")

        if args.dry_run:
            print(f"  [DRY RUN] Would init W&B run '{run_name}' with tags: {tags}")
            print(f"  [DRY RUN] Would log 10 sequential stage accuracy steps: {[round(x, 2) for x in stage_accs]}")
            continue

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group="c1_projection_initialization",
            name=run_name,
            tags=tags,
            config=run_config,
            reinit=True,
        )

        # Log stage-by-stage metrics
        for stage_idx, acc in enumerate(stage_accs, start=1):
            step_payload = {
                "task": stage_idx,
                "stage_accuracy": acc,
            }
            run.log(step_payload, step=stage_idx)

        # Log final summary metrics
        summary_payload = {
            "final/average_incremental_accuracy": avg_acc,
            "final/last_task_accuracy": last_acc,
            "final/average_forgetting_past_tasks": fgt_past,
            "final/average_forgetting_official_code": fgt_code,
        }
        for t_idx, acc in enumerate(t10_row, start=1):
            summary_payload[f"final_task_accuracies/task_{t_idx:02d}"] = acc

        run.summary.update(summary_payload)

        # Log artifact table/chart if present
        plot_path = analysis_dir / "c1_stage_accuracy_comparison.png"
        if plot_path.exists():
            run.log({"stage_accuracy_plot": wandb.Image(str(plot_path))})

        run.finish()
        print(f"  [SUCCESS] Uploaded run {run_name} to W&B.")

    print("\n" + "=" * 70)
    print("W&B Import Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
