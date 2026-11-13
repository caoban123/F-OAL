"""Generate Google Colab notebook for Controlled Experiment C2 (Backbone Alignment) on DTD."""
from __future__ import annotations

import json
from pathlib import Path


def make_md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()[:-1]] + ([source.splitlines()[-1]] if source.splitlines() else []),
    }


def make_code(source: str) -> dict:
    lines = source.splitlines()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }


def build_notebook() -> dict:
    cells = []

    cells.append(make_md("""# F-OAL Controlled Experiment C2 (DTD): Pretrained ViT Backbone Alignment

**Research Question:** In Forward-only Online Analytic Learning (F-OAL) on the Describable Textures Dataset (DTD), how much of the performance gap between the initial reproduction and official results is driven by the pretrained ViT representation:
- **Control (B0):** `vit_base_patch16_224.augreg_in1k` (ImageNet-1k pretraining only)
- **Treatment (B1):** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21k pretraining + ImageNet-1k fine-tuning)

**Strict Controlled Design (Single-Factor Isolation):**
- **B0:** `configs/dtd/c2_b0_augreg_in1k.yaml` — `vit_base_patch16_224.augreg_in1k`
- **B1:** `configs/dtd/c2_b1_augreg2_in21k_ft_in1k.yaml` — `vit_base_patch16_224.augreg2_in21k_ft_in1k`

**All other factors strictly locked:**
- **Dataset:** DTD (`tanganke/dtd`), 40 selected classes, 10 tasks x 4 classes, 40 train / 40 test per class (`samples_per_class: 40`, seed `42`)
- **Projection Equality:** Smooth Projection $P \\in \\mathbb{R}^{768 \\times 1000}$, `official_uniform`, seed `42` (exact matrix SHA256 match)
- **Preprocessing:** `bicubic` resize to 224x224, `normalize=true` (`mean=[0.5,0.5,0.5]`, `std=[0.5,0.5,0.5]`)
- **Classifier:** Exact float64 Woodbury RLS ($\\gamma = 1.0$), batch size 10, one-pass streaming
- **Chaining Guarantees:**
  - `C2-B0 == C1-P1` (B0 Reproducibility Guard)
  - `C2-B1 == C3-A` (Enables zero-compute reuse for C3 2x2 factorial baseline)"""))

    cells.append(make_code("""# === CELL 1: Environment Preflight (T4 / L4 / A100 GPU) ===
import os
import sys
import shutil
import platform
import subprocess
import torch

print(f"Python version   : {sys.version}")
print(f"Platform         : {platform.platform()}")
print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("FATAL: CUDA GPU is not available! Switch Colab runtime to GPU (T4 / L4 / A100) before running.")

device_name = torch.cuda.get_device_name(0)
device_props = torch.cuda.get_device_properties(0)
vram_gb = device_props.total_memory / (1024 ** 3)
print(f"GPU Device       : {device_name}")
print(f"CUDA version     : {torch.version.cuda}")
print(f"Total VRAM       : {vram_gb:.2f} GB")

free_bytes = shutil.disk_usage(".").free
print(f"Free Disk Space  : {free_bytes / (1024**3):.2f} GB")"""))

    cells.append(make_code("""# === CELL 2: Clone Group Repository & Verify C2 DTD Configs ===
from pathlib import Path
import yaml

REPO_DIR = Path("/content/F-OAL")
BRANCH_NAME = "experiment/dtd-ablation"
PINNED_COMMIT = None

if REPO_DIR.exists():
    print(f"Removing existing {REPO_DIR} for clean clone...")
    shutil.rmtree(REPO_DIR)

subprocess.run(["git", "clone", "-b", BRANCH_NAME, "https://github.com/caoban123/F-OAL.git", str(REPO_DIR)], check=True)
if PINNED_COMMIT:
    subprocess.run(["git", "checkout", PINNED_COMMIT], cwd=str(REPO_DIR), check=True)

commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), text=True).strip()
print(f"SUCCESS: Repository ready at commit {commit_hash} (branch: {BRANCH_NAME})")

os.chdir(str(REPO_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

CONFIGS = [
    "configs/dtd/c2_b0_augreg_in1k.yaml",
    "configs/dtd/c2_b1_augreg2_in21k_ft_in1k.yaml",
]
for cfg_path in CONFIGS:
    p = Path(cfg_path)
    assert p.exists(), f"Missing config: {cfg_path}"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    print(f"Loaded {cfg_path}: backbone={data['model']['backbone']}, init={data['model']['projection_init']}, interp={data['dataset']['interpolation']}, norm={data['dataset']['normalize']}")"""))

    cells.append(make_code("""# === CELL 3: Install Dependencies & Setup W&B Logging ===
!pip install -q timm pyyaml pytest datasets matplotlib pandas
!pip install -q --upgrade wandb

import wandb

WANDB_BASE_URL = "https://api.forge.coreweave.com"
WANDB_PROJECT = "hcmus-foal-reproduction"
WANDB_ENTITY = "caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm"
WANDB_GROUP = "c2_backbone_alignment_dtd"

os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL
print("Get your API key from Forge CoreWeave: https://forge.coreweave.com/wandb/authorize")
USE_WANDB = True
try:
    wandb.login(host=WANDB_BASE_URL)
    print("SUCCESS: W&B logged in.")
except Exception as exc:
    print(f"WARNING: W&B login failed ({exc}); continuing with --wandb-offline or local logging.")
    USE_WANDB = False"""))

    cells.append(make_code("""# === CELL 4: Run Unit & Single-Factor Isolation Tests ===
ret = subprocess.run([sys.executable, "-m", "pytest", "tests/test_c2_dtd.py", "-v"], check=False)
if ret.returncode != 0:
    raise RuntimeError("Unit/isolation tests failed! Inspect pytest output before running GPU experiments.")
print("All C2 DTD unit & protocol tests PASSED.")"""))

    cells.append(make_code("""# === CELL 5: Verify Smooth Projection Equality (B0 vs B1) ===
from src.foal.projection import SmoothProjection

p_b0 = SmoothProjection(768, 1000, initialization="official_uniform", seed=42)
p_b1 = SmoothProjection(768, 1000, initialization="official_uniform", seed=42)

sha_b0 = p_b0.get_metadata()["matrix_sha256"]
sha_b1 = p_b1.get_metadata()["matrix_sha256"]
assert sha_b0 == sha_b1, "Smooth projection matrices diverge across backbones!"
print(f"PROJECTION EQUALITY VERIFIED: Exact SHA256 = {sha_b0}")"""))

    cells.append(make_code("""# === CELL 6: Matched-Input Backbone Diagnostics on DTD ===
!python -m scripts.run_c2_dtd_diagnostics --device cuda --output-dir results/dtd_c2_diagnostics

import json
diag_json = json.loads(Path("results/dtd_c2_diagnostics/c2_dtd_matched_backbone_diagnostics.json").read_text(encoding="utf-8"))
b0_diag = diag_json["b0_augreg_in1k"]
b1_diag = diag_json["b1_augreg2_in21k_ft_in1k"]
print(f"B0 L2 Norm: {b0_diag['features']['l2_norm']['mean']:.2f} +/- {b0_diag['features']['l2_norm']['std']:.2f}, e-rank: {b0_diag['features']['centered_covariance']['effective_rank']:.2f}")
print(f"B1 L2 Norm: {b1_diag['features']['l2_norm']['mean']:.2f} +/- {b1_diag['features']['l2_norm']['std']:.2f}, e-rank: {b1_diag['features']['centered_covariance']['effective_rank']:.2f}")"""))

    cells.append(make_code("""# === CELL 7: Fast 2-Task End-to-End Pilot Verification ===
for tag, cfg_path in [
    ("b0", "configs/dtd/c2_b0_augreg_in1k.yaml"),
    ("b1", "configs/dtd/c2_b1_augreg2_in21k_ft_in1k.yaml"),
]:
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", cfg_path,
        "--output-dir", f"results/dtd_c2_pilot/{tag}",
        "--max-tasks", "2",
        "--limit-train-per-task", "20",
        "--limit-test-per-task", "20",
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)
print("All C2 DTD 2-task pilots PASSED.")"""))

    cells.append(make_code("""# === CELL 8: Execute Full 10-Task C2 Controlled Experiment (B0 vs B1) ===
# Note: Feature caches are isolated by backbone name automatically or via distinct cache dirs.
RUN_SPECS = [
    ("b0_augreg_in1k", "configs/dtd/c2_b0_augreg_in1k.yaml", "results/dtd_c2_feature_cache/b0"),
    ("b1_augreg2_in21k_ft_in1k", "configs/dtd/c2_b1_augreg2_in21k_ft_in1k.yaml", "results/dtd_c2_feature_cache/b1"),
]

for folder, cfg_path, cache_dir in RUN_SPECS:
    out_dir = f"results/dtd_c2_full/{folder}"
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", cfg_path,
        "--output-dir", out_dir,
        "--feature-cache-dir", cache_dir,
        "--device", "cuda",
    ]
    if USE_WANDB:
        cmd.extend(["--wandb", "--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY, "--wandb-group", WANDB_GROUP, "--wandb-base-url", WANDB_BASE_URL])
    print(f"\\n>>> Running {folder} -> {out_dir}")
    subprocess.run(cmd, check=True)"""))

    cells.append(make_code("""# === CELL 9: B0 Reproducibility Guard Verification ===
c1_p1_path = Path("results/dtd_c1_full/p1_official_uniform/metrics.json")
b0_path = Path("results/dtd_c2_full/b0_augreg_in1k/metrics.json")

if c1_p1_path.exists() and b0_path.exists():
    c1_p1_m = json.loads(c1_p1_path.read_text(encoding="utf-8"))
    b0_m = json.loads(b0_path.read_text(encoding="utf-8"))
    diff = abs(b0_m["average_incremental_accuracy"] - c1_p1_m["average_incremental_accuracy"])
    print(f"C2-B0 A_avg: {b0_m['average_incremental_accuracy']*100:.2f}% | C1-P1 A_avg: {c1_p1_m['average_incremental_accuracy']*100:.2f}% (diff: {diff:.6f})")
    assert diff < 1e-6, "B0 Reproducibility Guard Failed! C2-B0 must match C1-P1 under identical seed=42."
    print("SUCCESS: B0 Reproducibility Guard PASSED (exact match).")
else:
    print("C1-P1 results not present in local results; skipping cross-experiment exact equality check.")"""))

    cells.append(make_code("""# === CELL 10: Generate C2 DTD Analysis Tables, CSVs & Plots ===
!python -m scripts.generate_c2_dtd_analysis --results-dir results/dtd_c2_full --output-dir results/dtd_c2_analysis --c1-p1-dir results/dtd_c1_full/p1_official_uniform --diag-file results/dtd_c2_diagnostics/c2_dtd_matched_backbone_diagnostics.json

from IPython.display import Image, display
plot_file = Path("results/dtd_c2_analysis/c2_dtd_stage_accuracy_comparison.png")
if plot_file.exists():
    display(Image(filename=str(plot_file)))"""))

    cells.append(make_code("""# === CELL 11: Cross-Run Verification & Archive Artifacts ===
b0_meta = json.loads(Path("results/dtd_c2_full/b0_augreg_in1k/metadata.json").read_text(encoding="utf-8"))
b1_meta = json.loads(Path("results/dtd_c2_full/b1_augreg2_in21k_ft_in1k/metadata.json").read_text(encoding="utf-8"))

assert b0_meta["task_split_fingerprint"]["sha256"] == b1_meta["task_split_fingerprint"]["sha256"], "Split fingerprint mismatch across B0 and B1!"
print("Verified exact split SHA256 parity:", b0_meta["task_split_fingerprint"]["sha256"])

shutil.make_archive("dtd_c2_results_bundle", "zip", "results")
print("Created dtd_c2_results_bundle.zip. Ready for download.")
try:
    from google.colab import files
    files.download("dtd_c2_results_bundle.zip")
except ImportError:
    print("Not in Colab runtime; archive saved locally at dtd_c2_results_bundle.zip")"""))

    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


def main():
    nb = build_notebook()
    out_path = Path("notebooks/foal_c2_backbone_controlled_dtd.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
