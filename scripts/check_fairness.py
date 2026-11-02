import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

data = json.load(open("results/c1_audit_parsed.json", encoding="utf-8"))
p0_m = data["p0_gaussian_std1"]["metadata"]
p1_m = data["p1_official_uniform"]["metadata"]
p2_m = data["p2_gaussian_matched_variance"]["metadata"]

print("Class Order identical across all 3:", p0_m["class_order"] == p1_m["class_order"] == p2_m["class_order"])
print("First 10 classes:", p0_m["class_order"][:10])

print("\nModel config comparison:")
for k in ["backbone", "feature_fusion", "smooth_projection", "projection_dim", "gamma", "rls_dtype"]:
    v0 = p0_m["model_config"].get(k)
    v1 = p1_m["model_config"].get(k)
    v2 = p2_m["model_config"].get(k)
    print(f"  {k:20s}: P0={str(v0):10s} | P1={str(v1):10s} | P2={str(v2):10s} | Identical={v0 == v1 == v2}")

print("\nProjection Init:")
print("  P0:", p0_m["model_config"]["projection_init"])
print("  P1:", p1_m["model_config"]["projection_init"])
print("  P2:", p2_m["model_config"]["projection_init"])

print("\nTraining Protocol Config:")
for k in ["batch_size", "device", "one_pass", "feature_cache_enabled", "max_tasks", "classes_per_task"]:
    v0 = p0_m["training_protocol"].get(k)
    v1 = p1_m["training_protocol"].get(k)
    v2 = p2_m["training_protocol"].get(k)
    print(f"  {k:22s}: P0={str(v0):10s} | P1={str(v1):10s} | P2={str(v2):10s} | Identical={v0 == v1 == v2}")

print("\nSeeds:")
for k in ["base", "dataset", "class_order", "projection", "stream"]:
    v0 = p0_m["seeds"].get(k)
    v1 = p1_m["seeds"].get(k)
    v2 = p2_m["seeds"].get(k)
    print(f"  {k:15s}: P0={str(v0):5s} | P1={str(v1):5s} | P2={str(v2):5s} | Identical={v0 == v1 == v2}")
