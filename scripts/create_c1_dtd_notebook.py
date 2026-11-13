"""Generate Google Colab notebook for Controlled Experiment C1 (Smooth Projection Init) on DTD."""
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

    cells.append(make_md("""# F-OAL Controlled Experiment C1 (DTD): Smooth Projection Initialization & Variance Scaling

**Research Question:** In Forward-only Online Analytic Learning (F-OAL) on the Describable Textures Dataset (DTD), why did `gaussian_std1` degrade downstream performance vs the official `official_uniform` (`nn.Linear` Kaiming Uniform) projection? Is the mechanism:
- **Projection variance scaling** ($\\text{Var}(P_{ij}) = 1$ vs $1/(3 \\cdot 768) \\approx 4.34 \\times 10^{-4}$ avoiding Sigmoid saturation), OR
- **Uniform vs Gaussian distributional shape**?

**Strict Controlled Design (Single-Factor Isolation):**
- **P0:** `configs/dtd/c1_p0_gaussian_std1.yaml` — Gaussian $\\mathcal{N}(0, 1.0)$
- **P1:** `configs/dtd/c1_p1_official_uniform.yaml` — Official Uniform $\\mathcal{U}(-1/\\sqrt{3 \\cdot 768}, +1/\\sqrt{3 \\cdot 768})$
- **P2:** `configs/dtd/c1_p2_gaussian_matched_variance.yaml` — Variance-matched Gaussian $\\mathcal{N}(0, 1/(3 \\cdot 768))$

**All other factors fixed across P0, P1, P2:**
- **Dataset:** DTD (`tanganke/dtd`), 40 selected classes out of 47, 10 tasks x 4 classes, 40 train / 40 test samples per class (`samples_per_class: 40`, seed `42`)
- **Backbone:** `vit_base_patch16_224.augreg_in1k` (frozen, Feature Fusion enabled)
- **Preprocessing:** `bicubic` resize to 224x224, `normalize=true` (`mean=[0.5,0.5,0.5]`, `std=[0.5,0.5,0.5]`)
- **Classifier:** Exact float64 Woodbury RLS ($\\gamma = 1.0$), $D = 1000$, batch size 10, one-pass streaming
- **Chaining Identity:** `C1-P1 == C2-B0` (exact hyperparameter and seed parity)"""))

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

    cells.append(make_code("""# === CELL 2: Clone Group Repository & Verify C1 DTD Configs ===
from pathlib import Path
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

CONFIGS = [
    "configs/dtd/c1_p0_gaussian_std1.yaml",
    "configs/dtd/c1_p1_official_uniform.yaml",
    "configs/dtd/c1_p2_gaussian_matched_variance.yaml",
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
WANDB_GROUP = "c1_projection_initialization_dtd"

os.environ["WANDB_BASE_URL"] = WANDB_BASE_URL
print("Get your API key from Forge CoreWeave: https://forge.coreweave.com/wandb/authorize")
USE_WANDB = True
try:
    wandb.login(host=WANDB_BASE_URL)
    print("SUCCESS: W&B logged in.")
except Exception as exc:
    print(f"WARNING: W&B login failed ({exc}); continuing with --wandb-offline or local logging.")
    USE_WANDB = False"""))

    cells.append(make_code("""# === CELL 4: Run Unit & Config Integrity Tests ===
ret = subprocess.run([sys.executable, "-m", "pytest", "tests/test_c1_dtd.py", "tests/test_dtd_dataset.py", "-v"], check=False)
if ret.returncode != 0:
    raise RuntimeError("Unit/integrity tests failed! Inspect pytest output before running GPU experiments.")
print("All C1 DTD unit & protocol tests PASSED.")"""))

    cells.append(make_code("""# === CELL 5: Preload DTD Dataset & Backbone Weights + Verify Deterministic Splits ===
from src.foal.dtd_datasets import load_dtd, make_dtd_class_order, make_dtd_task_specs
from src.foal.backbone import FrozenViT

train_src, test_src = load_dtd()
order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=42)
tasks = make_dtd_task_specs(train_src["fine_label"], test_src["fine_label"], order, classes_per_task=4, samples_per_class=40, seed=42)
print(f"DTD loaded: train={len(train_src)}, test={len(test_src)}, tasks={len(tasks)}")
print(f"Task 1 train={len(tasks[0].train_indices)}, test={len(tasks[0].test_indices)}, class_order[:8]={order[:8]}")

print("Pre-caching ViT backbone vit_base_patch16_224.augreg_in1k ...")
_enc = FrozenViT("vit_base_patch16_224.augreg_in1k", pretrained=True, feature_fusion=True)
print(f"Backbone cached. Feature dim = {_enc.feature_dim}")
del _enc"""))

    cells.append(make_code("""# === CELL 6: Matched-Input Smooth Projection Diagnostics (N=200 Real DTD Images) ===
!python -m scripts.run_c1_dtd_diagnostics --device cuda --output-dir results/dtd_c1_diagnostics

import json
diag_json = json.loads(Path("results/dtd_c1_diagnostics/c1_dtd_matched_input_diagnostics.json").read_text(encoding="utf-8"))
for name, cfg in diag_json["diagnostics"]["configurations"].items():
    u_std = cfg["pre_sigmoid_u"]["std"]
    sat = cfg["post_sigmoid_z"]["total_saturation_fraction"] * 100.0
    cond = cfg["numerical_stability"]["regularized_gram_condition_number"]
    print(f"  {name:<26}: u_std={u_std:.4f}, saturation={sat:.2f}%, cond(G+I)={cond:.2f}")"""))

    cells.append(make_code("""# === CELL 7: Fast 2-Task End-to-End Pilot Verification ===
for tag, cfg_path in [
    ("p0", "configs/dtd/c1_p0_gaussian_std1.yaml"),
    ("p1", "configs/dtd/c1_p1_official_uniform.yaml"),
    ("p2", "configs/dtd/c1_p2_gaussian_matched_variance.yaml"),
]:
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", cfg_path,
        "--output-dir", f"results/dtd_c1_pilot/{tag}",
        "--max-tasks", "2",
        "--limit-train-per-task", "20",
        "--limit-test-per-task", "20",
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)
print("All 3 C1 DTD 2-task pilots PASSED.")"""))

    cells.append(make_code("""# === CELL 8: Execute Full 10-Task C1 Controlled Experiment (P0, P1, P2) ===
# Feature cache ensures FrozenViT forward pass runs ONCE and P1/P2 reuse the exact cached feature tensors.
CACHE_DIR = "results/dtd_c1_feature_cache"
RUN_SPECS = [
    ("p0_gaussian_std1", "configs/dtd/c1_p0_gaussian_std1.yaml"),
    ("p1_official_uniform", "configs/dtd/c1_p1_official_uniform.yaml"),
    ("p2_gaussian_matched_variance", "configs/dtd/c1_p2_gaussian_matched_variance.yaml"),
]

for folder, cfg_path in RUN_SPECS:
    out_dir = f"results/dtd_c1_full/{folder}"
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", cfg_path,
        "--output-dir", out_dir,
        "--feature-cache-dir", CACHE_DIR,
        "--device", "cuda",
    ]
    if USE_WANDB:
        cmd.extend(["--wandb", "--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY, "--wandb-group", WANDB_GROUP, "--wandb-base-url", WANDB_BASE_URL])
    print(f"\\n>>> Running {folder} -> {out_dir}")
    subprocess.run(cmd, check=True)"""))

    cells.append(make_code("""# === CELL 9: Generate C1 DTD Analysis Tables, CSVs & Plots ===
!python -m scripts.generate_c1_dtd_analysis --results-dir results/dtd_c1_full --output-dir results/dtd_c1_analysis --diag-file results/dtd_c1_diagnostics/c1_dtd_matched_input_diagnostics.json

from IPython.display import Image, display
plot_file = Path("results/dtd_c1_analysis/c1_dtd_stage_accuracy_comparison.png")
if plot_file.exists():
    display(Image(filename=str(plot_file)))"""))

    cells.append(make_code("""# === CELL 10: Cross-Run Verification & Archive Artifacts ===
import hashlib

p0_m = json.loads(Path("results/dtd_c1_full/p0_gaussian_std1/metrics.json").read_text(encoding="utf-8"))
p1_m = json.loads(Path("results/dtd_c1_full/p1_official_uniform/metrics.json").read_text(encoding="utf-8"))
p2_m = json.loads(Path("results/dtd_c1_full/p2_gaussian_matched_variance/metrics.json").read_text(encoding="utf-8"))

p0_meta = json.loads(Path("results/dtd_c1_full/p0_gaussian_std1/metadata.json").read_text(encoding="utf-8"))
p1_meta = json.loads(Path("results/dtd_c1_full/p1_official_uniform/metadata.json").read_text(encoding="utf-8"))
p2_meta = json.loads(Path("results/dtd_c1_full/p2_gaussian_matched_variance/metadata.json").read_text(encoding="utf-8"))

assert p0_meta["task_split_fingerprint"]["sha256"] == p1_meta["task_split_fingerprint"]["sha256"] == p2_meta["task_split_fingerprint"]["sha256"], "Split fingerprint mismatch!"
print("Verified exact split SHA256 parity across P0, P1, P2:", p1_meta["task_split_fingerprint"]["sha256"])

shutil.make_archive("dtd_c1_results_bundle", "zip", "results")
print("Created dtd_c1_results_bundle.zip. Ready for download from Colab.")
try:
    from google.colab import files
    files.download("dtd_c1_results_bundle.zip")
except ImportError:
    print("Not in Colab runtime; archive saved locally at dtd_c1_results_bundle.zip")"""))

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
    out_path = Path("notebooks/foal_c1_projection_controlled_dtd.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
