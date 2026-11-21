"""Generate notebooks/foal_dtd_full_reproduction.ipynb for Google Colab / Kaggle.

Follows the structural guidelines of notebooks/foal_cifar100_full_reproduction.ipynb while aligning
with the standardized Colab setup, branch cloning, and W&B tracking conventions across all DTD notebooks:
1. Environment Preflight (!nvidia-smi, PyTorch, CUDA, VRAM, Disk)
2. Clone Group Repository (branch: experiment/dtd-ablation) & Verify DTD Full Config
3. Install Dependencies & Setup W&B Logging (Colab Secrets / API Key support)
4. Run Preflight Test Suite (pytest)
5. Single-Seed Full Reproduction (Seed 42) via scripts/train.py
6. Independent Checkpoint Evaluation via scripts/evaluate.py
7. Official Results Table & Summary via scripts/summarize_runs.py
8. Multi-Seed Full Reproduction (Optional - Phase P1.3) via scripts/run_seeds.py
9. Export & Archive Results (auto-download with fallback)
"""
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

    # Title & Benchmark Overview
    cells.append(make_md("""# F-OAL: Describable Textures Dataset (DTD) Full Reproduction (NeurIPS 2024)
## Phase P1.2 — Single-Seed Full Reproduction on GPU

This notebook implements the exact single-seed execution pipeline on GPU for reproducing the official DTD benchmark from:
> **F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning**  
> *Huiping Zhuang, Yuchen Liu, Run He, et al. NeurIPS 2024 (Main Conference Track)*

### Benchmark Targets (DTD 10-tasks, Table 2 & Table 4 of Paper)
- **Average Incremental Accuracy ($A_{\\text{avg}}$):** **82.8%** (Table 2 & Table 4)
- **Last Task Accuracy ($A_{\\text{last}}$):** ~76.1%
- **Average Forgetting ($F$):** ~8.6%
- **Training Time:** ~8 seconds on GPU (NeurIPS 2024 Table 3)

### Experimental Protocol
- **Backbone:** Frozen ViT-B/16 (`vit_base_patch16_224.augreg2_in21k_ft_in1k`, ImageNet-21k pretraining + ImageNet-1k fine-tuning)
- **Feature Fusion:** Multi-layer CLS token averaging (layers 0-11) + final LayerNorm
- **Expansion:** Smooth Random Projection to $D = 1000$, $\\gamma = 1.0$, `official_uniform` initialization
- **Learner:** Online Recursive Least Squares (Woodbury update, $\\gamma = 1.0$, float64)
- **Data Protocol:** 10 tasks $\\times$ 4 classes (40 classes total out of 47), streaming batch size 10, strictly **one-pass** (single epoch)
- **Sample Limits:** Exactly 40 train and 40 test samples per class (160 samples/task, 1,600 train and 1,600 test total, 100% paper alignment)
- **Preprocessing:** Bilinear interpolation, Raw [0, 1] tensor (Official author transform: `Resize((224, 224))` + `ToTensor()`)
- **Trial Seed:** Seed 42 (Single-seed evaluation; multi-seed is deferred to Phase P1.3)"""))

    # Section 1: Hardware & Environment Verification
    cells.append(make_md("""## 1. Hardware & Environment Verification
Ensure that an NVIDIA GPU (T4, P100, V100, A100, or L4) is active in this runtime."""))

    cells.append(make_code("!nvidia-smi"))

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

    # Section 2: Clone Repository & Verify Configs
    cells.append(make_md("""## 2. Clone Repository & Verify DTD Configuration
Clone the group repository on branch `experiment/dtd-ablation` and verify the full reproduction configuration."""))

    cells.append(make_code("""# === CELL 2: Clone Group Repository & Verify DTD Full Config ===
from pathlib import Path
import os
import shutil
import subprocess
import sys
import yaml

REPO_DIR = Path("/content/F-OAL")
BRANCH_NAME = "experiment/dtd-ablation"
PINNED_COMMIT = None  # Optional: pin exact commit SHA

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

CONFIG_PATH = "configs/dtd_full_reproduction.yaml"
p = Path(CONFIG_PATH)
assert p.exists(), f"Missing config: {CONFIG_PATH}"
data = yaml.safe_load(p.read_text(encoding="utf-8"))
print(f"Loaded {CONFIG_PATH}: backbone={data['model']['backbone']}, init={data['model']['projection_init']}, interp={data['dataset']['interpolation']}, norm={data['dataset']['normalize']}")"""))

    # Section 3: Install Dependencies & Setup W&B Logging
    cells.append(make_md("""## 3. Install Dependencies & Setup W&B Logging
Install core dependencies and configure Weights & Biases cloud logging via Colab Secrets (`WANDB_API_KEY`) or interactive authentication."""))

    cells.append(make_code("""# === CELL 3: Install Dependencies & Setup W&B Logging ===
!pip install -q timm pyyaml pytest datasets matplotlib pandas
!pip install -q --upgrade wandb
!pip install -q -e .

import os
import wandb

# 1. Retrieve W&B API Key from Colab Secrets or Environment
try:
    from google.colab import userdata
    if "WANDB_API_KEY" in userdata:
        os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
except Exception:
    pass

api_key = os.environ.get("WANDB_API_KEY", "").strip()
wandb_base_url = os.environ.get("WANDB_BASE_URL", "https://api.forge.coreweave.com")

WANDB_PROJECT = "hcmus-foal-reproduction"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm")
WANDB_GROUP = "dtd_full_reproduction"

USE_WANDB = False
if api_key:
    try:
        if "forge" in wandb_base_url or "coreweave" in wandb_base_url:
            wandb.login(key=api_key, host=wandb_base_url)
        else:
            wandb.login(key=api_key)
        USE_WANDB = True
        print("SUCCESS: Logged in to Weights & Biases (cloud tracking enabled).")
    except Exception as exc:
        print(f"WARNING: W&B login failed ({exc}); continuing with local logging.")
else:
    print("NOTE: WANDB_API_KEY not found in environment or Colab Secrets.")
    print("      To enable cloud tracking: add WANDB_API_KEY in Colab Secrets (key icon on left) or run `wandb.login(key='...')`.")
    print("      Continuing with local logging.")"""))

    # Section 4: Run Preflight Test Suite
    cells.append(make_md("""## 4. Run Preflight Test Suite
Verify mathematical correctness, Woodbury RLS updates, DTD dataset partitioning, and protocol isolation."""))

    cells.append(make_code("!python -m pytest tests/test_dtd_dataset.py tests/test_dtd_full_reproduction.py tests/test_rls.py tests/test_projection.py tests/test_metrics.py -v"))

    # Section 5: Single-Seed Full Reproduction (Seed 42)
    cells.append(make_md("""## 5. Single-Seed Full Reproduction (Seed 42)
Run F-OAL on the full DTD benchmark (10 tasks $\\times$ 4 classes, 40 train and 40 test samples/class, streaming batch size 10, one-pass).

Checkpoints are saved after every task (`task_01.pt` ... `task_10.pt`)."""))

    cells.append(make_code("""# === CELL 5: Execute Single-Seed Full Reproduction (Seed 42) ===
cmd = [
    sys.executable, "-m", "scripts.train",
    "--config", "configs/dtd_full_reproduction.yaml",
    "--seed", "42",
    "--device", "cuda",
    "--run-name", "foal_full_dtd_seed42",
]
if USE_WANDB:
    cmd.extend([
        "--wandb",
        "--wandb-project", WANDB_PROJECT,
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-group", WANDB_GROUP,
    ])
    if "forge" in wandb_base_url or "coreweave" in wandb_base_url:
        cmd.extend(["--wandb-base-url", wandb_base_url])
else:
    cmd.extend(["--wandb", "--wandb-offline"])

print(">>> Running Full DTD Reproduction (Seed 42)...")
subprocess.run(cmd, check=True)"""))

    # Section 6: Independent Checkpoint Evaluation
    cells.append(make_md("""## 6. Independent Checkpoint Evaluation
Verify checkpoint roundtrip and numerical stability by loading `task_10.pt` from disk into a fresh model instance and evaluating against all 10 tasks."""))

    cells.append(make_code("""!python -m scripts.evaluate \\
    --checkpoint results/foal_full_dtd_seed42/task_10.pt \\
    --device cuda"""))

    # Section 7: Official Results Table & Summary
    cells.append(make_md("""## 7. Official Results Table & Summary
Summarize the reproduction run using `scripts/summarize_runs.py`.
The summarizer strictly verifies run completeness, isolates protocols, and exports summary CSVs."""))

    cells.append(make_code("!python -m scripts.summarize_runs --results-dir results"))

    # Section 8: Multi-Seed Full Reproduction (Optional — Phase P1.3)
    cells.append(make_md("""## 8. Multi-Seed Full Reproduction (Optional — Phase P1.3)
> **NOTE:** In Phase P1.2, only Single-Seed (Seed 42) is executed.
> The multi-seed command below is intentionally commented out and should only be triggered during Phase P1.3."""))

    cells.append(make_code("""# Multi-seed orchestration is disabled in Phase P1.2.
# Uncomment to run in Phase P1.3:
# !python -m scripts.run_seeds \\
#     --seeds 42 123 2026 \\
#     --config configs/dtd_full_reproduction.yaml \\
#     --device cuda \\
#     --run-name-prefix foal_full_dtd_3seeds \\
#     --wandb \\
#     --wandb-offline \\
#     --skip-completed

# !python -m scripts.summarize_runs --results-dir results"""))

    # Section 9: Export & Archive Results
    cells.append(make_md("""## 9. Export & Archive Results
Compress all result CSVs, metrics, and metadata into a zip archive for persistent download (excluding large .pt weights)."""))

    cells.append(make_code("""import zipfile
from pathlib import Path

archive_name = "foal_dtd_single_seed42_results.zip"
target_dir = Path("results/foal_full_dtd_seed42")

with zipfile.ZipFile(archive_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    if target_dir.exists():
        for f in sorted(target_dir.rglob("*")):
            if f.is_file() and not f.name.endswith(".pt") and not f.name.endswith(".pth"):
                zf.write(f, arcname=str(f.relative_to(Path("results"))))
    for summary_file in Path("results").glob("summary_*.csv"):
        zf.write(summary_file, arcname=summary_file.name)

print(f"Archive verified: {archive_name} ({Path(archive_name).stat().st_size:,} bytes)")

# Trigger Automatic Browser Download in Google Colab
try:
    from google.colab import files
    print(f"Triggering automatic browser download for {archive_name}...")
    files.download(archive_name)
except Exception as exc:
    print(f"Automatic browser download not available or blocked ({exc}).")
    print(f"To download manually, open the Colab Files sidebar on the left and right-click -> Download '{archive_name}'")"""))

    return {
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {
            "accelerator": "GPU",
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "cells": cells,
    }


def main():
    nb = build_notebook()
    out_path = Path("notebooks/foal_dtd_full_reproduction.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote notebook to {out_path.resolve()} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
