"""Resumable Multi-Seed Table 4 Ablation Runner for CIFAR-100 (Seeds 42, 123, 2026).

Reuses verified Seed 42 results without overwriting or re-running, and executes
the 8 new Full GPU runs (Seeds 123 and 2026 x {full, no_ff, no_sp, no_ff_no_sp})
with strict protocol parity, feature-cache isolation, and checkpoint resumption.

Usage:
    # Generate/verify seed manifest and check status
    python -m scripts.run_cifar100_multiseed_ablation --status-only

    # Pilot smoke test (2 tasks, 50 train / 50 test per task, separate dir & W&B group)
    python -m scripts.run_cifar100_multiseed_ablation --pilot --seeds 123 --ablations full no_ff

    # Run all 8 Full GPU experiments (Seeds 123 & 2026) with W&B and feature caching
    python -m scripts.run_cifar100_multiseed_ablation --device cuda --wandb
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.foal.metrics import stage_accuracies, summary_from_task_matrix

SEEDS: list[int] = [42, 123, 2026]
TARGET_NEW_SEEDS: list[int] = [123, 2026]

EXPECTED_BACKBONE = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
EXPECTED_INTERPOLATION = "bilinear"
EXPECTED_NORMALIZE = False
EXPECTED_PROJECTION_INIT = "official_uniform"
EXPECTED_PROJECTION_DIM = 1000
EXPECTED_GAMMA = 1.0
EXPECTED_RLS_DTYPE = "float64"

ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {
        "label": "Full F-OAL",
        "condition": "A0",
        "config_path": REPO_ROOT / "configs" / "table4_a0_full_foal_cifar100.yaml",
        "feature_fusion": True,
        "smooth_projection": True,
        "seed42_dir": REPO_ROOT / "results" / "c3_full_gpu_verified" / "results" / "c3_full_gpu" / "c3_d_bilinear_raw",
        "seed42_wandb_id": "3v53tmq5",
        "seed42_wandb_name": "c3_d_bilinear_raw",
        "seed42_wandb_group": "c3_preprocessing_alignment",
    },
    "no_ff": {
        "label": "Without FF",
        "condition": "A1",
        "config_path": REPO_ROOT / "configs" / "table4_a1_without_ff_cifar100.yaml",
        "feature_fusion": False,
        "smooth_projection": True,
        "seed42_dir": REPO_ROOT / "results" / "table4_full_gpu_verified" / "results" / "table4_a1_without_ff",
        "seed42_wandb_id": "shl4p10l",
        "seed42_wandb_name": "table4_a1_without_ff_seed42",
        "seed42_wandb_group": "foal_no_fusion_ablation",
    },
    "no_sp": {
        "label": "Without SP",
        "condition": "A2",
        "config_path": REPO_ROOT / "configs" / "table4_a2_without_sp_cifar100.yaml",
        "feature_fusion": True,
        "smooth_projection": False,
        "seed42_dir": REPO_ROOT / "results" / "table4_full_gpu_verified" / "results" / "table4_a2_without_sp",
        "seed42_wandb_id": "rkedlkkk",
        "seed42_wandb_name": "table4_a2_without_sp_seed42",
        "seed42_wandb_group": "foal_no_projection_ablation",
    },
    "no_ff_no_sp": {
        "label": "Without FF & SP",
        "condition": "A3",
        "config_path": REPO_ROOT / "configs" / "table4_a3_without_both_cifar100.yaml",
        "feature_fusion": False,
        "smooth_projection": False,
        "seed42_dir": REPO_ROOT / "results" / "table4_full_gpu_verified" / "results" / "table4_a3_without_both",
        "seed42_wandb_id": "k01pp8st",
        "seed42_wandb_name": "table4_a3_without_both_seed42",
        "seed42_wandb_group": "foal_neither_ablation",
    },
}


def load_task_matrix_csv(csv_path: Path, expected_stages: int = 10) -> list[list[float]]:
    """Load long-format `stage,task,accuracy` CSV into a lower-triangular matrix."""
    matrix: list[list[float]] = [[] for _ in range(expected_stages)]
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = int(row["stage"])
            task = int(row["task"])
            acc = float(row["accuracy"])
            if not (1 <= stage <= expected_stages) or not (1 <= task <= stage):
                raise ValueError(f"Invalid (stage={stage}, task={task}) entry in {csv_path}")
            if not math.isfinite(acc) or not (0.0 <= acc <= 1.0):
                raise ValueError(f"Invalid accuracy value {acc} in {csv_path}")
            if len(matrix[stage - 1]) != task - 1:
                raise ValueError(f"Out-of-order entry (stage={stage}, task={task}) in {csv_path}")
            matrix[stage - 1].append(acc)
    for stage_idx, row in enumerate(matrix, start=1):
        if len(row) != stage_idx:
            raise ValueError(f"Stage {stage_idx} in {csv_path} has {len(row)} entries, expected {stage_idx}")
    return matrix


def validate_completed_run(
    run_dir: Path,
    expected_seed: int,
    ablation_key: str,
    *,
    is_pilot: bool = False,
    expected_tasks: int = 10,
    require_checkpoints: bool = True,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Validate that a run directory is complete, non-contaminated, and protocol-compliant."""
    reasons: list[str] = []
    if ablation_key not in ABLATIONS:
        return False, [f"Unknown ablation '{ablation_key}'"], None

    ablation_cfg = ABLATIONS[ablation_key]
    meta_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    csv_path = run_dir / "task_accuracy.csv"

    for required in (meta_path, metrics_path, csv_path):
        if not required.exists():
            reasons.append(f"Missing required file: {required.name}")

    if require_checkpoints:
        for task_idx in range(1, expected_tasks + 1):
            ckpt = run_dir / f"task_{task_idx:02d}.pt"
            if not ckpt.exists():
                reasons.append(f"Missing checkpoint: {ckpt.name}")

    if reasons:
        return False, reasons, None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        matrix = load_task_matrix_csv(csv_path, expected_stages=expected_tasks)
    except Exception as exc:
        return False, [f"Failed to parse artifacts: {exc}"], None

    # Validate seeds
    seeds = meta.get("seeds", {})
    for s_key in ("base", "class_order", "projection", "stream"):
        if int(seeds.get(s_key, -1)) != int(expected_seed):
            reasons.append(f"Seed mismatch for '{s_key}': got {seeds.get(s_key)}, expected {expected_seed}")

    # Validate model & preprocessing protocol
    mc = meta.get("model_config", {})
    if mc.get("backbone") != EXPECTED_BACKBONE:
        reasons.append(f"Backbone mismatch: {mc.get('backbone')} != {EXPECTED_BACKBONE}")
    if bool(mc.get("feature_fusion")) != bool(ablation_cfg["feature_fusion"]):
        reasons.append(
            f"feature_fusion mismatch: {mc.get('feature_fusion')} != {ablation_cfg['feature_fusion']}"
        )
    if bool(mc.get("smooth_projection")) != bool(ablation_cfg["smooth_projection"]):
        reasons.append(
            f"smooth_projection mismatch: {mc.get('smooth_projection')} != {ablation_cfg['smooth_projection']}"
        )
    if ablation_cfg["smooth_projection"]:
        if mc.get("projection_init") != EXPECTED_PROJECTION_INIT:
            reasons.append(f"projection_init mismatch: {mc.get('projection_init')} != {EXPECTED_PROJECTION_INIT}")
        if int(mc.get("projection_dim", -1)) != EXPECTED_PROJECTION_DIM:
            reasons.append(f"projection_dim mismatch: {mc.get('projection_dim')} != {EXPECTED_PROJECTION_DIM}")
    if float(mc.get("gamma", -1.0)) != EXPECTED_GAMMA:
        reasons.append(f"gamma mismatch: {mc.get('gamma')} != {EXPECTED_GAMMA}")
    if mc.get("rls_dtype") != EXPECTED_RLS_DTYPE:
        reasons.append(f"rls_dtype mismatch: {mc.get('rls_dtype')} != {EXPECTED_RLS_DTYPE}")

    ds_cfg = meta.get("config", {}).get("dataset", {})
    if str(ds_cfg.get("interpolation", "")).lower() != EXPECTED_INTERPOLATION:
        reasons.append(f"interpolation mismatch: {ds_cfg.get('interpolation')} != {EXPECTED_INTERPOLATION}")
    if bool(ds_cfg.get("normalize", True)) != EXPECTED_NORMALIZE:
        reasons.append(f"normalize mismatch: {ds_cfg.get('normalize')} != {EXPECTED_NORMALIZE}")

    # Validate training protocol & guard against pilot contamination
    tp = meta.get("training_protocol", {})
    if int(tp.get("max_tasks", -1)) != expected_tasks:
        reasons.append(f"max_tasks mismatch: {tp.get('max_tasks')} != {expected_tasks}")
    if not is_pilot:
        if tp.get("limit_train_per_task") is not None:
            reasons.append(f"Pilot contamination: limit_train_per_task={tp.get('limit_train_per_task')}")
        if tp.get("limit_test_per_task") is not None:
            reasons.append(f"Pilot contamination: limit_test_per_task={tp.get('limit_test_per_task')}")
        if meta.get("experiment_type") == "pilot":
            reasons.append("Pilot contamination: experiment_type == 'pilot'")

    # Cross-check metrics against raw matrix
    recalc = summary_from_task_matrix(matrix)
    for key in ("average_incremental_accuracy", "last_accuracy", "forgetting_past_tasks", "forgetting_official_code"):
        val_m = metrics.get(key)
        val_r = recalc.get(key)
        if val_m is None or not math.isfinite(float(val_m)):
            reasons.append(f"Non-finite metric '{key}': {val_m}")
        elif abs(float(val_m) - float(val_r)) > 1e-9:
            reasons.append(f"Metric '{key}' mismatch: json={val_m} vs matrix={val_r}")

    if reasons:
        return False, reasons, None

    summary = {
        "run_dir": str(run_dir),
        "seed": expected_seed,
        "ablation": ablation_key,
        "condition": ablation_cfg["condition"],
        "task_split_fingerprint": meta.get("task_split_fingerprint", {}).get("sha256"),
        "class_order_sha256": meta.get("task_split_fingerprint", {}).get("class_order_sha256"),
        "projection_matrix_sha256": (meta.get("projection_metadata") or {}).get("matrix_sha256"),
        "average_incremental_accuracy_pct": float(recalc["average_incremental_accuracy"]) * 100.0,
        "last_accuracy_pct": float(recalc["last_accuracy"]) * 100.0,
        "forgetting_past_tasks_pct": float(recalc["forgetting_past_tasks"]) * 100.0,
        "forgetting_official_code_pct": float(recalc["forgetting_official_code"]) * 100.0,
        "total_duration_seconds": float(metrics.get("total_duration_seconds", 0.0)),
    }
    return True, [], summary


def build_seed_manifest(output_root: Path) -> dict[str, Any]:
    """Audit Seed 42 source artifacts (or load Git-tracked verified manifest on fresh clones)."""
    import hashlib

    all_s42_local = all(cfg["seed42_dir"].exists() for cfg in ABLATIONS.values())
    canonical_manifest_path = REPO_ROOT / "results" / "cifar100_multiseed" / "seed_manifest.json"

    if not all_s42_local:
        if not canonical_manifest_path.exists():
            raise RuntimeError(
                f"Neither local Seed 42 artifacts nor Git-tracked {canonical_manifest_path} found."
            )
        manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
        s42_recs = manifest.get("seed_42_verified_runs", {})
        if set(s42_recs.keys()) != set(ABLATIONS.keys()):
            raise RuntimeError(f"Invalid seed_manifest.json: missing ablations in {list(s42_recs.keys())}")
        fps = {str(s42_recs[k].get("task_split_fingerprint")) for k in ABLATIONS}
        if len(fps) != 1:
            raise RuntimeError(f"Seed 42 manifest ablations do not share a single task_split_fingerprint: {fps}")
        output_root.mkdir(parents=True, exist_ok=True)
        target_path = output_root / "seed_manifest.json"
        if target_path.resolve() != canonical_manifest_path.resolve():
            target_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    seed42_records: dict[str, Any] = {}
    seed42_fps: set[str] = set()

    for abl_key, abl_cfg in ABLATIONS.items():
        s42_dir = abl_cfg["seed42_dir"]
        ok, reasons, summary = validate_completed_run(
            s42_dir,
            expected_seed=42,
            ablation_key=abl_key,
            is_pilot=False,
            expected_tasks=10,
            require_checkpoints=False,  # verified zip archives omit .pt weights to keep repo lightweight
        )
        if not ok or summary is None:
            raise RuntimeError(f"Seed 42 audit failed for '{abl_key}' at {s42_dir}: {reasons}")
        csv_path = s42_dir / "task_accuracy.csv"
        matrix = load_task_matrix_csv(csv_path, expected_stages=10)
        stages_pct = [s * 100.0 for s in stage_accuracies(matrix)]
        csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        seed42_fps.add(str(summary["task_split_fingerprint"]))
        seed42_records[abl_key] = {
            **summary,
            "csv_sha256": csv_sha,
            "stage_accuracies_pct": stages_pct,
            "provenance": "REUSED_VERIFIED_SEED42",
            "wandb_run_id": abl_cfg["seed42_wandb_id"],
            "wandb_run_name": abl_cfg["seed42_wandb_name"],
            "wandb_group": abl_cfg["seed42_wandb_group"],
        }

    if len(seed42_fps) != 1:
        raise RuntimeError(f"Seed 42 ablations do not share a single task_split_fingerprint: {seed42_fps}")

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "CIFAR-100 Table 4 Multi-Seed Ablation Study",
        "seeds": SEEDS,
        "target_new_seeds": TARGET_NEW_SEEDS,
        "reused_seed": 42,
        "fixed_protocol": {
            "dataset": "uoft-cs/cifar100",
            "num_classes": 100,
            "classes_per_task": 10,
            "num_tasks": 10,
            "backbone": EXPECTED_BACKBONE,
            "interpolation": EXPECTED_INTERPOLATION,
            "normalize": EXPECTED_NORMALIZE,
            "projection_init": EXPECTED_PROJECTION_INIT,
            "projection_dim_when_enabled": EXPECTED_PROJECTION_DIM,
            "gamma": EXPECTED_GAMMA,
            "rls_dtype": EXPECTED_RLS_DTYPE,
            "batch_size": 10,
            "one_pass": True,
            "replay_buffer": 0,
        },
        "seed_policy": {
            "description": (
                "Each seed S in {42, 123, 2026} sets base_seed=S, class_order_seed=S, "
                "projection_seed=S, and stream_seed=S, matching the verified Seed 42 configuration."
            ),
            "controlled_sources": [
                "class_order: np.random.default_rng(class_order_seed).permutation(100)",
                "projection: isolated torch.Generator('cpu').manual_seed(projection_seed) when SP=True",
                "data_stream: torch.Generator().manual_seed(stream_seed + task_id) batch shuffling",
                "global_rngs: random.seed(S), np.random.seed(S), torch.manual_seed(S)",
            ],
            "paired_comparison_invariant": (
                "Within each seed S, all 4 ablations (full, no_ff, no_sp, no_ff_no_sp) share the "
                "exact same class_order_seed=S and stream_seed=S, guaranteeing identical task_split_fingerprint."
            ),
            "feature_cache_safety": (
                "Cache keys include backbone, interpolation, normalize, ff/noff tag, "
                "class_order_seed, stream_seed, task_id, and sample limit, plus SHA256 validation "
                "on indices and preprocessing. Therefore 'full'/'no_sp' share 'ff' cache within a seed, "
                "'no_ff'/'no_ff_no_sp' share 'noff' cache within a seed, and different seeds never collide."
            ),
        },
        "seed_42_verified_runs": seed42_records,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "seed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_single_experiment(
    *,
    seed: int,
    ablation_key: str,
    output_dir: Path,
    feature_cache_dir: Path | None,
    device: str,
    use_wandb: bool,
    wandb_offline: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_base_url: str,
    is_pilot: bool,
) -> dict[str, Any]:
    """Invoke `scripts.train.run` cleanly for a single (seed, ablation) pair."""
    from scripts.train import run as train_run

    abl_cfg = ABLATIONS[ablation_key]
    run_name = f"cifar100_{ablation_key}_seed{seed}" + ("_pilot" if is_pilot else "")
    wandb_group = "cifar100_table4_multiseed_pilot" if is_pilot else "cifar100_table4_multiseed"
    tags = f"multiseed,table4,{ablation_key},seed_{seed}" + (",pilot" if is_pilot else ",full")

    args = SimpleNamespace(
        config=str(abl_cfg["config_path"]),
        max_tasks=2 if is_pilot else None,
        limit_train_per_task=50 if is_pilot else None,
        limit_test_per_task=50 if is_pilot else None,
        seed=seed,
        class_order_seed=seed,
        projection_seed=seed,
        projection_init=EXPECTED_PROJECTION_INIT if abl_cfg["smooth_projection"] else None,
        stream_seed=seed,
        device=device,
        experiment_type="pilot" if is_pilot else ("reproduction" if ablation_key == "full" else "ablation"),
        no_feature_fusion=not abl_cfg["feature_fusion"],
        no_projection=not abl_cfg["smooth_projection"],
        feature_cache_dir=str(feature_cache_dir) if feature_cache_dir is not None else None,
        run_name=run_name,
        output_dir=str(output_dir),
        wandb=use_wandb,
        wandb_offline=wandb_offline,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_base_url=wandb_base_url,
        wandb_group=wandb_group,
        wandb_tags=tags,
    )
    return train_run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-100 Multi-Seed Table 4 Ablation Study.")
    parser.add_argument("--seeds", type=int, nargs="+", default=TARGET_NEW_SEEDS,
                        help="Seeds to execute (default: 123 2026; seed 42 is reused and protected)")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()),
                        choices=list(ABLATIONS.keys()),
                        help="Ablation conditions to run")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "cifar100_multiseed"),
                        help="Root directory for multi-seed outputs")
    parser.add_argument("--feature-cache-dir", default=str(REPO_ROOT / "artifacts" / "feature_cache_multiseed"),
                        help="Directory for deterministic frozen ViT feature cache (empty string disables)")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--pilot", action="store_true",
                        help="Run lightweight 2-task pilot into isolated pilot directory and W&B group")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if valid completed artifacts exist (never applies to seed 42)")
    parser.add_argument("--status-only", action="store_true",
                        help="Audit seed 42, build seed_manifest.json, and report status without training")
    parser.add_argument("--wandb", action="store_true", help="Log runs to Weights & Biases")
    parser.add_argument("--wandb-offline", action="store_true", help="Use W&B offline mode")
    parser.add_argument("--wandb-project", default="hcmus-foal-reproduction")
    parser.add_argument("--wandb-entity", default="caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm")
    parser.add_argument("--wandb-base-url", default="https://api.forge.coreweave.com")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.pilot:
        output_root = output_root.parent / f"{output_root.name}_pilot"

    # Protect seed 42 from accidental re-execution
    if not args.status_only and 42 in args.seeds:
        raise ValueError(
            "Seed 42 is already verified and must be reused without re-running or overwriting. "
            "Remove 42 from --seeds."
        )

    manifest = build_seed_manifest(Path(args.output_root))
    print(f"[Manifest] Seed 42 verified and seed_manifest.json written to {Path(args.output_root) / 'seed_manifest.json'}")

    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    expected_tasks = 2 if args.pilot else 10
    status_records: list[dict[str, Any]] = []

    total_runs = len(args.seeds) * len(args.ablations)
    run_idx = 0
    overall_start = time.monotonic()

    for seed in args.seeds:
        for abl_key in args.ablations:
            run_idx += 1
            run_dir = output_root / f"seed_{seed}" / abl_key
            ok, reasons, summary = validate_completed_run(
                run_dir,
                expected_seed=seed,
                ablation_key=abl_key,
                is_pilot=args.pilot,
                expected_tasks=expected_tasks,
                require_checkpoints=True,
            )

            if args.status_only:
                status_records.append({
                    "seed": seed,
                    "ablation": abl_key,
                    "status": "COMPLETED_VERIFIED" if ok else "PENDING",
                    "reasons": reasons,
                    "summary": summary,
                })
                state = "COMPLETED_VERIFIED" if ok else f"PENDING ({'; '.join(reasons[:2])})"
                print(f"[{run_idx}/{total_runs}] seed={seed} ablation={abl_key:<11} -> {state}")
                continue

            if ok and not args.force:
                print(
                    f"[{run_idx}/{total_runs}] SKIPPING seed={seed} ablation={abl_key} "
                    f"(verified complete: A_avg={summary['average_incremental_accuracy_pct']:.4f}%)"
                )
                status_records.append({"seed": seed, "ablation": abl_key, "status": "SKIPPED_VERIFIED", "summary": summary})
                continue

            print(f"\n[{run_idx}/{total_runs}] STARTING seed={seed} ablation={abl_key} ({'PILOT' if args.pilot else 'FULL'})")
            t0 = time.monotonic()
            run_single_experiment(
                seed=seed,
                ablation_key=abl_key,
                output_dir=run_dir,
                feature_cache_dir=feature_cache_dir,
                device=args.device,
                use_wandb=args.wandb,
                wandb_offline=args.wandb_offline,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_base_url=args.wandb_base_url,
                is_pilot=args.pilot,
            )
            dt = round(time.monotonic() - t0, 2)

            ok_after, reasons_after, summary_after = validate_completed_run(
                run_dir,
                expected_seed=seed,
                ablation_key=abl_key,
                is_pilot=args.pilot,
                expected_tasks=expected_tasks,
                require_checkpoints=True,
            )
            if not ok_after:
                raise RuntimeError(
                    f"Post-run validation failed for seed={seed} ablation={abl_key}: {reasons_after}"
                )
            print(
                f"[{run_idx}/{total_runs}] FINISHED seed={seed} ablation={abl_key} in {dt}s "
                f"(A_avg={summary_after['average_incremental_accuracy_pct']:.4f}%, "
                f"A_last={summary_after['last_accuracy_pct']:.2f}%, "
                f"F_past={summary_after['forgetting_past_tasks_pct']:.2f}%)"
            )
            status_records.append({
                "seed": seed,
                "ablation": abl_key,
                "status": "COMPLETED_NEW",
                "duration_seconds": dt,
                "summary": summary_after,
            })

    status_payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot": args.pilot,
        "total_elapsed_seconds": round(time.monotonic() - overall_start, 2),
        "runs": status_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
