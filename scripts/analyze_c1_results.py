import csv
import json
import math
from pathlib import Path
import sys
import torch

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path("results/c1_full_gpu_verified/results")
if not BASE_DIR.exists():
    BASE_DIR = Path("results/c1_full_gpu_verified")

print(f"Base results directory: {BASE_DIR}")
for p in BASE_DIR.iterdir():
    print(f" - {p.name} ({'dir' if p.is_dir() else 'file'})")

FULL_DIR = BASE_DIR / "c1_cifar100_full"
PILOT_DIR = BASE_DIR / "c1_pilot"
DIAG_DIR = BASE_DIR / "c1_diagnostics"

print("\nFull runs directory exists:", FULL_DIR.exists())
if FULL_DIR.exists():
    for p in FULL_DIR.iterdir():
        print(f"  - {p.name}")

# Import metrics calculation functions
from src.foal.metrics import (
    stage_accuracies,
    task_forgetting_past_tasks,
    forgetting_past_tasks,
    task_forgetting_official_code,
    forgetting_official_code,
    summary_from_task_matrix,
)

variants = ["p0_gaussian_std1", "p1_official_uniform", "p2_gaussian_matched_variance"]
results_data = {}

for v in variants:
    v_dir = FULL_DIR / v
    print(f"\n========================================================")
    print(f"AUDITING FULL RUN: {v}")
    print(f"========================================================")
    if not v_dir.exists():
        print(f"ERROR: {v_dir} does not exist!")
        continue

    meta_file = v_dir / "metadata.json"
    metrics_file = v_dir / "metrics.json"
    csv_file = v_dir / "task_accuracy.csv"

    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    metrics_reported = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}

    print("Metadata Summary:")
    print(f"  Run name       : {meta.get('run_name')}")
    print(f"  Experiment type: {meta.get('experiment_type')}")
    print(f"  Git commit     : {meta.get('git_commit')}")
    env = meta.get("environment", {})
    gpu_info = env.get("gpu", {})
    print(f"  GPU device     : {gpu_info.get('device_name')} (Count: {gpu_info.get('count')})")
    print(f"  CUDA version   : {gpu_info.get('cuda_version')}")
    print(f"  PyTorch        : {env.get('libraries', {}).get('torch')}")
    print(f"  timm           : {env.get('libraries', {}).get('timm')}")
    print(f"  Python         : {env.get('python', {}).get('version')}")
    print(f"  Platform       : {env.get('os', {}).get('platform')}")

    proj_meta = meta.get("projection_metadata", {})
    print("Projection Metadata:")
    print(f"  Initialization : {proj_meta.get('initialization')}")
    print(f"  Seed           : {proj_meta.get('seed')}")
    print(f"  Measured mean  : {proj_meta.get('measured_mean'):.8f}")
    print(f"  Measured std   : {proj_meta.get('measured_std'):.8f}")
    print(f"  Measured var   : {proj_meta.get('measured_variance'):.8f}")
    print(f"  SHA256         : {proj_meta.get('matrix_sha256')}")

    # Inspect Checkpoints
    ckpts = sorted(v_dir.glob("task_*.pt"))
    print(f"Checkpoints found: {len(ckpts)}/10")
    ckpt_hashes = []
    ckpt_classes = []
    all_ckpts_valid = True

    for ckpt_p in ckpts:
        size_mb = ckpt_p.stat().st_size / (1024**2)
        try:
            # Safe checkpoint inspection using torch.load
            ckpt_data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            task_num = ckpt_data.get("task_number")
            cls_order = ckpt_data.get("class_order", [])
            proj_st = ckpt_data.get("projection_state")
            cls_st = ckpt_data.get("classifier_state")

            if proj_st and "weight" in proj_st:
                pw = proj_st["weight"]
                import hashlib
                pw_sha = hashlib.sha256(pw.float().contiguous().numpy().tobytes()).hexdigest()
                ckpt_hashes.append(pw_sha)

            w_cls = cls_st.get("weight") if cls_st else None
            out_dim = w_cls.shape[0] if w_cls is not None else 0
            ckpt_classes.append(out_dim)

            # Check for NaN / Inf
            has_nan = torch.isnan(w_cls).any().item() if w_cls is not None else False
            has_inf = torch.isinf(w_cls).any().item() if w_cls is not None else False
            if has_nan or has_inf:
                print(f"  WARNING: Checkpoint {ckpt_p.name} contains NaN ({has_nan}) or Inf ({has_inf})!")
                all_ckpts_valid = False

        except Exception as e:
            print(f"  ERROR loading {ckpt_p.name}: {e}")
            all_ckpts_valid = False

    print(f"All 10 checkpoints loadable without NaN/Inf: {all_ckpts_valid}")
    print(f"Classifier expanding dimensions: {ckpt_classes}")
    unique_proj_hashes = set(ckpt_hashes)
    print(f"Projection matrix frozen across all 10 tasks: {len(unique_proj_hashes) == 1} (Unique hashes: {len(unique_proj_hashes)})")
    if unique_proj_hashes:
        print(f"  Checkpoint P hash matches metadata: {list(unique_proj_hashes)[0] == proj_meta.get('matrix_sha256')}")

    # Raw Task Matrix & Independent Recalculation
    matrix = metrics_reported.get("task_matrix", [])
    print(f"\nTask Matrix Dimensions: {len(matrix)} stages")
    is_lower_triangular = True
    total_entries = 0
    for s_idx, row in enumerate(matrix):
        total_entries += len(row)
        if len(row) != s_idx + 1:
            is_lower_triangular = False

    print(f"Is strict lower-triangular (1..10): {is_lower_triangular} (Total accuracy entries: {total_entries}/55)")

    # Recalculate
    rec_summary = summary_from_task_matrix(matrix)
    rec_stage_accs = rec_summary["stage_accuracies"]
    rec_avg_incr_acc = rec_summary["average_incremental_accuracy"]
    rec_last_acc = rec_summary["last_accuracy"]
    rec_fgt_past = rec_summary["forgetting_past_tasks"]
    rec_fgt_code = rec_summary["forgetting_official_code"]
    rec_task_fgt_past = rec_summary["task_forgetting_past_tasks"]
    rec_task_fgt_code = rec_summary["task_forgetting_official_code"]

    rep_avg = metrics_reported.get("average_incremental_accuracy")
    rep_last = metrics_reported.get("last_accuracy")
    rep_fgt_past = metrics_reported.get("forgetting_past_tasks", metrics_reported.get("forgetting"))
    rep_fgt_code = metrics_reported.get("forgetting_official_code")

    print("\nMetrics Verification (Reported vs Independently Recalculated):")
    print(f"  Avg Incremental Acc : Reported={rep_avg:.6f} | Recalculated={rec_avg_incr_acc:.6f} | Diff={abs(rep_avg - rec_avg_incr_acc):.2e}")
    print(f"  Last Accuracy (A10) : Reported={rep_last:.6f} | Recalculated={rec_last_acc:.6f} | Diff={abs(rep_last - rec_last_acc):.2e}")
    print(f"  Fgt (Past-Tasks T-1): Reported={rep_fgt_past:.6f} | Recalculated={rec_fgt_past:.6f} | Diff={abs(rep_fgt_past - rec_fgt_past):.2e}")
    print(f"  Fgt (Official Code) : Reported={rep_fgt_code:.6f} | Recalculated={rec_fgt_code:.6f} | Diff={abs(rep_fgt_code - rec_fgt_code):.2e}")

    timing = {
        "total_duration_seconds": metrics_reported.get("total_duration_seconds"),
        "feature_extraction_seconds": metrics_reported.get("feature_extraction_seconds"),
        "rls_and_eval_seconds": metrics_reported.get("rls_and_eval_seconds"),
    }
    print(f"\nTiming:")
    print(f"  Total duration      : {timing['total_duration_seconds']} s")
    print(f"  Feature extraction  : {timing['feature_extraction_seconds']} s")
    print(f"  RLS & Eval          : {timing['rls_and_eval_seconds']} s")

    results_data[v] = {
        "metadata": meta,
        "metrics_reported": metrics_reported,
        "recalculated": {
            "stage_accuracies": rec_stage_accs,
            "average_incremental_accuracy": rec_avg_incr_acc,
            "last_accuracy": rec_last_acc,
            "forgetting_past_tasks": rec_fgt_past,
            "forgetting_official_code": rec_fgt_code,
            "task_forgetting_past_tasks": rec_task_fgt_past,
            "task_forgetting_official_code": rec_task_fgt_code,
        },
        "task_matrix": matrix,
        "timing": timing,
        "checkpoints_valid": all_ckpts_valid,
        "frozen_p_verified": len(unique_proj_hashes) == 1,
        "matrix_sha256": proj_meta.get("matrix_sha256"),
    }

# Save audit results
Path("results/c1_audit_parsed.json").write_text(json.dumps(results_data, indent=2), encoding="utf-8")
print("\nAuditing complete! Output saved to results/c1_audit_parsed.json")
