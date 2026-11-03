import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Verify ZIP SHA256
zip_path = Path("foal_table4_ablation_results.zip")
expected_zip_sha256 = "90f191d12b06fbd804779fcb038a12574c9b7cdc2f32369a6975dd6309e09e7a"
actual_zip_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
assert actual_zip_sha256 == expected_zip_sha256, f"ZIP SHA256 mismatch: {actual_zip_sha256} vs {expected_zip_sha256}"
print(f"[AUDIT] foal_table4_ablation_results.zip SHA256 VERIFIED: {actual_zip_sha256}")

# 2. Reconstruct raw matrices from task_accuracy.csv
paths = {
    "A0_reused_c3_d": Path("results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw"),
    "A1_no_ff": Path("results/table4_full_gpu_verified/results/table4_a1_without_ff"),
    "A2_no_sp": Path("results/table4_full_gpu_verified/results/table4_a2_without_sp"),
    "A3_no_both": Path("results/table4_full_gpu_verified/results/table4_a3_without_both"),
}

raw_data = {}
for cond_key, dir_path in paths.items():
    csv_file = dir_path / "task_accuracy.csv"
    metrics_file = dir_path / "metrics.json"
    meta_file = dir_path / "metadata.json"
    
    assert csv_file.exists(), f"Missing CSV: {csv_file}"
    assert metrics_file.exists(), f"Missing metrics: {metrics_file}"
    assert meta_file.exists(), f"Missing metadata: {meta_file}"
    
    csv_sha256 = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    df = pd.read_csv(csv_file)
    assert len(df) == 55, f"{cond_key} CSV row count != 55 (found {len(df)})"
    
    # Verify stage/task pairs
    expected_pairs = [(s, t) for s in range(1, 11) for t in range(1, s + 1)]
    actual_pairs = list(zip(df["stage"], df["task"]))
    assert actual_pairs == expected_pairs, f"{cond_key} stage/task sequence mismatch"
    
    # Build 10x10 lower triangular matrix (0-indexed)
    R = np.zeros((10, 10))
    for _, row in df.iterrows():
        s = int(row["stage"]) - 1
        t = int(row["task"]) - 1
        R[s, t] = float(row["accuracy"])
        
    # Recompute stage accuracies: A_s = mean(R[s, 0..s])
    stage_accs = [float(np.mean(R[s, :s+1])) for s in range(10)]
    
    # Recompute average incremental accuracy: A_bar = mean(stage_accs)
    a_bar = float(np.mean(stage_accs))
    
    # Recompute last accuracy: A_10 = stage_accs[9] = mean(R[9, :10])
    a_last = float(stage_accs[9])
    
    # Recompute per-task forgetting:
    # F_past (9 tasks): f_j = max_{j <= s < 9} R[s, j] - R[9, j]
    task_f_past = []
    for j in range(9):
        best_past = float(np.max(R[j:9, j]))
        task_f_past.append(best_past - R[9, j])
    f_past = float(np.mean(task_f_past))
    
    # F_code (10 tasks): f_j = max_{j <= s <= 9} R[s, j] - R[9, j]
    task_f_code = []
    for j in range(10):
        best_any = float(np.max(R[j:10, j]))
        task_f_code.append(best_any - R[9, j])
    f_code = float(np.mean(task_f_code))
    
    # Load metrics.json for comparison
    with open(metrics_file, "r", encoding="utf-8") as f:
        m_json = json.load(f)
        
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
        
    # Check consistency against metrics.json
    diff_abar = abs(a_bar - m_json["average_incremental_accuracy"])
    diff_alast = abs(a_last - m_json["last_accuracy"])
    diff_fpast = abs(f_past - m_json["forgetting_past_tasks"])
    diff_fcode = abs(f_code - m_json["forgetting_official_code"])
    
    assert diff_abar < 1e-12, f"A_bar mismatch in {cond_key}: {diff_abar}"
    assert diff_alast < 1e-12, f"A_last mismatch in {cond_key}: {diff_alast}"
    assert diff_fpast < 1e-12, f"F_past mismatch in {cond_key}: {diff_fpast}"
    assert diff_fcode < 1e-12, f"F_code mismatch in {cond_key}: {diff_fcode}"
    
    raw_data[cond_key] = {
        "dir": str(dir_path),
        "csv_sha256": csv_sha256,
        "R": R,
        "stage_accs": stage_accs,
        "stage_mean": float(np.mean(stage_accs)),
        "a_bar": a_bar,
        "a_last": a_last,
        "task_f_past": task_f_past,
        "task_f_code": task_f_code,
        "f_past": f_past,
        "f_code": f_code,
        "metrics_json": m_json,
        "metadata_json": meta_json,
    }
    print(f"[AUDIT] {cond_key}: CSV SHA256={csv_sha256[:16]}... A_bar={a_bar*100:.6f}%, Stage Mean={np.mean(stage_accs)*100:.6f}%, A_last={a_last*100:.6f}%")

# Save detailed reconstruction to json
output_reconstruction = {
    cond_key: {
        "csv_sha256": v["csv_sha256"],
        "stage_accs_pct": [x * 100.0 for x in v["stage_accs"]],
        "stage_mean_pct": v["stage_mean"] * 100.0,
        "a_bar_pct": v["a_bar"] * 100.0,
        "a_last_pct": v["a_last"] * 100.0,
        "task_f_past_pct": [x * 100.0 for x in v["task_f_past"]],
        "task_f_code_pct": [x * 100.0 for x in v["task_f_code"]],
        "f_past_pct": v["f_past"] * 100.0,
        "f_code_pct": v["f_code"] * 100.0,
        "matrix_55_entries": [float(x) for x in v["R"][np.tril_indices(10)]],
    }
    for cond_key, v in raw_data.items()
}

with open("results/table4_analysis/forensic_reconstruction_raw.json", "w", encoding="utf-8") as f:
    json.dump(output_reconstruction, f, indent=2)

print("\n[AUDIT] Forensic matrix audit completed successfully and saved to results/table4_analysis/forensic_reconstruction_raw.json")
