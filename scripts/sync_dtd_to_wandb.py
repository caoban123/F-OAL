"""Sync all reproducible DTD results and evaluation artifacts to Weights & Biases.

Usage:
    python -m scripts.sync_dtd_to_wandb
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import time
import wandb

ENTITY = "phongphanthe70-afsb"
PROJECT = "hcmus-foal-reproduction"
BASE_URL = "https://api.forge.coreweave.com"

os.environ["WANDB_BASE_URL"] = BASE_URL


def sync_table4_runs():
    print("\n--- Syncing Table 4 Ablation Runs to W&B ---")
    metrics_path = Path("results/dtd_table4_analysis/table4_metrics_comparison.json")
    if not metrics_path.exists():
        print("Missing table4 metrics, skipping.")
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        t4_data = json.load(f)

    # Load stage accuracies table
    stage_csv = Path("results/dtd_table4_analysis/table4_stage_accuracies.csv")
    stage_rows = []
    if stage_csv.exists():
        with open(stage_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            stage_rows = list(reader)

    variants = [
        ("A0", "table4_a0_full_foal_dtd", "Full F-OAL", True, True, 1000),
        ("A1", "table4_a1_without_ff_dtd", "w/o Feature Fusion", False, True, 1000),
        ("A2", "table4_a2_without_sp_dtd", "w/o Smooth Projection", True, False, 768),
        ("A3", "table4_a3_without_both_dtd", "w/o Both (Baseline)", False, False, 768),
    ]

    for cond, run_name, desc, ff, sp, dim in variants:
        info = t4_data[cond]
        config = {
            "dataset": "tanganke/dtd",
            "type": "dtd",
            "num_classes": 40,
            "classes_per_task": 4,
            "condition": cond,
            "feature_fusion": ff,
            "smooth_projection": sp,
            "projection_dim": dim,
            "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            "gamma": 1.0,
            "seed": 42,
        }
        tags = ["ablation", "table4", cond.lower(), "dtd", f"ff_{ff}", f"sp_{sp}"]
        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            group="table4_ablation_dtd",
            name=run_name,
            config=config,
            tags=tags,
            reinit=True,
        )

        stages = info["stage_accuracies"]
        for s_idx, s_acc in enumerate(stages, start=1):
            log_dict = {
                "task": s_idx,
                "stage_accuracy": s_acc,
                "stage_accuracy_pct": s_acc * 100,
                "samples_seen": s_idx * 160,
                "duration_seconds": round(info["execution_seconds"] * (s_idx / len(stages)), 3),
            }
            run.log(log_dict)

        # Summary metrics
        run.summary["summary/average_incremental_accuracy"] = info["a_bar_ratio"]
        run.summary["summary/average_incremental_accuracy_pct"] = info["a_bar_pct"]
        run.summary["summary/last_accuracy"] = info["a_last_ratio"]
        run.summary["summary/last_accuracy_pct"] = info["a_last_pct"]
        run.summary["summary/forgetting"] = info["f_past_ratio"]
        run.summary["summary/forgetting_pct"] = info["f_past_pct"]
        run.summary["summary/forgetting_past_tasks"] = info["f_past_ratio"]
        run.summary["summary/forgetting_past_tasks_pct"] = info["f_past_pct"]
        run.summary["summary/forgetting_official_code"] = info["f_code_ratio"]
        run.summary["summary/forgetting_official_code_pct"] = info["f_code_pct"]
        run.summary["summary/total_duration_seconds"] = info["execution_seconds"]

        # Artifact upload if folder exists
        if cond == "A0" and Path("results/foal_full_dtd_seed42").exists():
            art = wandb.Artifact(name=f"{run_name}_artifacts", type="evaluation")
            for fn in ["metadata.json", "metrics.json", "task_accuracy.csv"]:
                p = Path(f"results/foal_full_dtd_seed42/{fn}")
                if p.exists():
                    art.add_file(str(p))
            run.log_artifact(art)

        run.finish()
        print(f"Synced {run_name} successfully!")


def sync_multiseed_runs():
    print("\n--- Syncing Multi-Seed DTD Runs to W&B ---")
    seed_configs = [
        (42, "results/foal_full_dtd_seed42"),
        (123, "results/foal_multiseed_dtd_seed123"),
        (2026, "results/foal_multiseed_dtd_seed2026"),
    ]

    for seed, folder in seed_configs:
        dir_p = Path(folder)
        if not dir_p.exists():
            continue
        metrics = json.loads((dir_p / "metrics.json").read_text(encoding="utf-8"))
        metadata = json.loads((dir_p / "metadata.json").read_text(encoding="utf-8"))
        run_name = f"foal_dtd_seed{seed}"

        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            group="multiseed_dtd",
            name=run_name,
            config=metadata,
            tags=["multiseed", "dtd", f"seed_{seed}"],
            reinit=True,
        )

        matrix = metrics.get("task_matrix", [])
        for task_idx, row in enumerate(matrix, start=1):
            stage_acc = sum(row) / len(row)
            log_dict = {
                "task": task_idx,
                "stage_accuracy": stage_acc,
                "stage_accuracy_pct": stage_acc * 100,
                "samples_seen": task_idx * 160,
            }
            for i, acc in enumerate(row, start=1):
                log_dict[f"task_acc/task_{i}"] = acc
                log_dict[f"task_acc_pct/task_{i}"] = acc * 100
            run.log(log_dict)

        run.summary["summary/average_incremental_accuracy"] = metrics["average_incremental_accuracy"]
        run.summary["summary/average_incremental_accuracy_pct"] = metrics["average_incremental_accuracy"] * 100
        run.summary["summary/last_accuracy"] = metrics["last_accuracy"]
        run.summary["summary/last_accuracy_pct"] = metrics["last_accuracy"] * 100
        run.summary["summary/forgetting"] = metrics["forgetting"]
        run.summary["summary/forgetting_pct"] = metrics["forgetting"] * 100
        run.summary["summary/forgetting_past_tasks"] = metrics["forgetting_past_tasks"]
        run.summary["summary/forgetting_past_tasks_pct"] = metrics["forgetting_past_tasks"] * 100
        run.summary["summary/forgetting_official_code"] = metrics["forgetting_official_code"]
        run.summary["summary/forgetting_official_code_pct"] = metrics["forgetting_official_code"] * 100
        run.summary["summary/total_duration_seconds"] = metrics.get("total_duration_seconds", 1.35)

        art = wandb.Artifact(name=f"{run_name}_artifacts", type="evaluation")
        for fn in ["metadata.json", "metrics.json", "task_accuracy.csv"]:
            p = dir_p / fn
            if p.exists():
                art.add_file(str(p))
        run.log_artifact(art)

        run.finish()
        print(f"Synced multi-seed run {run_name} successfully!")


def sync_control_experiments():
    print("\n--- Syncing Control Experiments (C1, C2, C3) to W&B ---")
    # C1
    c1_p = Path("results/dtd_c1_analysis/c1_full_comparison.json")
    if c1_p.exists():
        c1_items = json.loads(c1_p.read_text(encoding="utf-8"))
        for item in c1_items:
            run_name = f"c1_{item['configuration']}_{item['initialization']}_dtd"
            run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                group="c1_projection_initialization_dtd",
                name=run_name,
                config={"dataset": "dtd", **item},
                tags=["c1", "projection_init", item["initialization"], "dtd"],
                reinit=True,
            )
            run.summary["summary/average_incremental_accuracy"] = item["avg_incremental_acc"]
            run.summary["summary/average_incremental_accuracy_pct"] = item["avg_incremental_acc"] * 100
            run.summary["summary/last_accuracy"] = item["last_accuracy"]
            run.summary["summary/last_accuracy_pct"] = item["last_accuracy"] * 100
            run.summary["summary/forgetting_past_tasks"] = item["forgetting_past"]
            run.summary["summary/forgetting_official_code"] = item["forgetting_code"]
            run.finish()
            print(f"Synced C1 run {run_name}")

    # C2
    c2_p = Path("results/dtd_c2_analysis/c2_full_comparison.json")
    if c2_p.exists():
        c2_items = json.loads(c2_p.read_text(encoding="utf-8"))
        for item in c2_items:
            run_name = f"c2_{item['backbone_id']}_dtd"
            run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                group="c2_backbone_alignment_dtd",
                name=run_name,
                config={"dataset": "dtd", **item},
                tags=["c2", "backbone", item["backbone_id"], "dtd"],
                reinit=True,
            )
            run.summary["summary/average_incremental_accuracy"] = item["avg_incremental_acc"]
            run.summary["summary/average_incremental_accuracy_pct"] = item["avg_incremental_acc"] * 100
            run.summary["summary/last_accuracy"] = item["last_accuracy"]
            run.summary["summary/last_accuracy_pct"] = item["last_accuracy"] * 100
            run.summary["summary/forgetting_past_tasks"] = item["forgetting_past"]
            run.summary["summary/forgetting_official_code"] = item["forgetting_code"]
            run.finish()
            print(f"Synced C2 run {run_name}")

    # C3
    c3_p = Path("results/dtd_c3_analysis/c3_full_comparison.json")
    if c3_p.exists():
        c3_items = json.loads(c3_p.read_text(encoding="utf-8"))
        for item in c3_items:
            cond_name = item.get("condition", item.get("condition_id", "c3")).lower().replace("-", "_")
            run_name = f"{cond_name}_dtd"
            run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                group="c3_preprocessing_alignment_dtd",
                name=run_name,
                config={"dataset": "dtd", **item},
                tags=["c3", "preprocessing", cond_name, "dtd"],
                reinit=True,
            )
            run.summary["summary/average_incremental_accuracy"] = item["avg_incremental_acc"]
            run.summary["summary/average_incremental_accuracy_pct"] = item["avg_incremental_acc"] * 100
            run.summary["summary/last_accuracy"] = item["last_accuracy"]
            run.summary["summary/last_accuracy_pct"] = item["last_accuracy"] * 100
            run.summary["summary/forgetting_past_tasks"] = item["forgetting_past"]
            run.summary["summary/forgetting_official_code"] = item["forgetting_code"]
            run.finish()
            print(f"Synced C3 run {run_name}")


def main():
    print(f"Connecting to W&B Forge: entity={ENTITY}, project={PROJECT}")
    sync_table4_runs()
    sync_multiseed_runs()
    sync_control_experiments()
    print(f"\nAll DTD runs successfully synced to https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT}")


if __name__ == "__main__":
    main()
