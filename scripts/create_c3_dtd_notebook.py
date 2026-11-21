"""Generate Google Colab notebook for Controlled Experiment C3 (Preprocessing Alignment) on DTD."""
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

    cells.append(make_md("""# F-OAL Controlled Experiment C3 (DTD): 2x2 Factorial Preprocessing Alignment

**Research Question:** Dissect the exact performance impact of input image preprocessing in Forward-only Online Analytic Learning (F-OAL) on the Describable Textures Dataset (DTD) across a complete 2x2 balanced factorial design:
- **Factor 1 (Interpolation):** `bicubic` vs `bilinear`
- **Factor 2 (Normalization):** `normalized [-1, 1]` vs `raw [0, 1]`

**The 4 Factorial Conditions:**
- **C3-A:** `configs/dtd/c3_a_bicubic_normalized.yaml` — Bicubic + Normalized `[-1, 1]` (Identity with `C2-B1`)
- **C3-B:** `configs/dtd/c3_b_bilinear_normalized.yaml` — Bilinear + Normalized `[-1, 1]`
- **C3-C:** `configs/dtd/c3_c_bicubic_raw.yaml` — Bicubic + Raw `[0, 1]`
- **C3-D:** `configs/dtd/c3_d_bilinear_raw.yaml` — Bilinear + Raw `[0, 1]` (Full Official Alignment & Table 4 A0 parity)

**All Other Factors Strictly Held Invariant:**
- **Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21k + 1k FT, frozen, Feature Fusion ON)
- **Smooth Projection:** $P \\in \\mathbb{R}^{768 \\times 1000}$, `official_uniform`, seed `42` (exact matrix SHA256 match)
- **Dataset & Protocol:** DTD (`tanganke/dtd`), 40 classes, 10 tasks x 4 classes, 40 train / 40 test per class, one-pass streaming, float64 RLS ($\\gamma = 1.0$)
- **Zero-Compute Optimization:** `C3-A` is hyperparameter-identical to `C2-B1`; can be reused when `C3_A_REUSE_APPROVED = True`!"""))

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

    cells.append(make_code("""# === CELL 2: Clone Group Repository & Verify C3 DTD Configs ===
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
    "configs/dtd/c3_a_bicubic_normalized.yaml",
    "configs/dtd/c3_b_bilinear_normalized.yaml",
    "configs/dtd/c3_c_bicubic_raw.yaml",
    "configs/dtd/c3_d_bilinear_raw.yaml",
]
for cfg_path in CONFIGS:
    p = Path(cfg_path)
    assert p.exists(), f"Missing config: {cfg_path}"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    print(f"Loaded {cfg_path}: interp={data['dataset']['interpolation']}, norm={data['dataset']['normalize']}")"""))

    cells.append(make_code("""# === CELL 3: Install Dependencies & Setup W&B Logging ===
!pip install -q timm pyyaml pytest datasets matplotlib pandas seaborn
!pip install -q --upgrade wandb

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
WANDB_GROUP = "c3_preprocessing_alignment_dtd"

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

    cells.append(make_code("""# === CELL 4: Run Unit & Factorial Isolation Tests ===
ret = subprocess.run([sys.executable, "-m", "pytest", "tests/test_c3_dtd.py", "-v"], check=False)
if ret.returncode != 0:
    raise RuntimeError("Unit/isolation tests failed! Inspect pytest output before running GPU experiments.")
print("All C3 DTD unit & protocol tests PASSED.")"""))

    cells.append(make_code("""# === CELL 5: Deep Transform Equivalence Verification (C3-D vs Official) ===
!python -m scripts.verify_official_transform_dtd --num-samples 100 --output results/dtd_c3_analysis/c3_dtd_transform_verification.json

import json
verif_data = json.loads(Path("results/dtd_c3_analysis/c3_dtd_transform_verification.json").read_text(encoding="utf-8"))
print(f"Real DTD max pixel diff : {verif_data['real_dtd_max_diff']:.2e}")
print(f"Edge cases all passed   : {verif_data['all_passed']}")
assert verif_data['all_passed'], "C3-D is NOT bitwise equivalent to Official Author transform!"
print("TRANSFORM PARITY VERIFIED: C3-D matches Official Author Preprocessing bitwise.")"""))

    cells.append(make_code("""# === CELL 6: Matched Preprocessing Diagnostics on DTD ===
!python -m scripts.run_c3_dtd_diagnostics --num-samples 200 --device cuda --output results/dtd_c3_analysis/c3_dtd_matched_diagnostics.json

diag_data = json.loads(Path("results/dtd_c3_analysis/c3_dtd_matched_diagnostics.json").read_text(encoding="utf-8"))
for pair, sim in diag_data["cosine_similarities"].items():
    print(f"  {pair:<36}: mean={sim['mean']:.5f}")"""))

    cells.append(make_code("""# === CELL 7: Fast 2-Task End-to-End Pilot Verification ===
for tag, cfg_path in [
    ("c3_a", "configs/dtd/c3_a_bicubic_normalized.yaml"),
    ("c3_b", "configs/dtd/c3_b_bilinear_normalized.yaml"),
    ("c3_c", "configs/dtd/c3_c_bicubic_raw.yaml"),
    ("c3_d", "configs/dtd/c3_d_bilinear_raw.yaml"),
]:
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config", cfg_path,
        "--output-dir", f"results/dtd_c3_pilot/{tag}",
        "--max-tasks", "2",
        "--limit-train-per-task", "20",
        "--limit-test-per-task", "20",
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)
print("All 4 C3 DTD 2-task pilots PASSED.")"""))

    cells.append(make_code("""# === CELL 8: C3-A Reuse Decision (Zero-Compute from C2-B1) ===
C3_A_REUSE_APPROVED = True  # Set to False to re-run C3-A from scratch
c2_b1_metrics = Path("results/dtd_c2_full/b1_augreg2_in21k_ft_in1k/metrics.json")
c3_a_dir = Path("results/dtd_c3_full/c3_a_bicubic_normalized")

if C3_A_REUSE_APPROVED and c2_b1_metrics.exists():
    c3_a_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(c2_b1_metrics, c3_a_dir / "metrics.json")
    meta_src = Path("results/dtd_c2_full/b1_augreg2_in21k_ft_in1k/metadata.json")
    if meta_src.exists():
        shutil.copy2(meta_src, c3_a_dir / "metadata.json")
    print(f"REUSED C2-B1 for C3-A (Zero Compute!). Copied artifacts to {c3_a_dir}")
else:
    print("C3-A will be trained from scratch in CELL 9.")"""))

    cells.append(make_code("""# === CELL 9: Execute Full 10-Task C3 Controlled Factorial Runs ===
RUN_SPECS = [
    ("c3_a_bicubic_normalized", "configs/dtd/c3_a_bicubic_normalized.yaml", "results/dtd_c3_feature_cache/c3_a"),
    ("c3_b_bilinear_normalized", "configs/dtd/c3_b_bilinear_normalized.yaml", "results/dtd_c3_feature_cache/c3_b"),
    ("c3_c_bicubic_raw", "configs/dtd/c3_c_bicubic_raw.yaml", "results/dtd_c3_feature_cache/c3_c"),
    ("c3_d_bilinear_raw", "configs/dtd/c3_d_bilinear_raw.yaml", "results/dtd_c3_feature_cache/c3_d"),
]

for folder, cfg_path, cache_dir in RUN_SPECS:
    out_dir = f"results/dtd_c3_full/{folder}"
    metrics_path = Path(out_dir) / "metrics.json"
    if metrics_path.exists():
        print(f"Skipping {folder} (metrics already exist at {metrics_path})")
        continue

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

    cells.append(make_code("""# === CELL 10: Generate C3 DTD 2x2 Factorial Analysis & Plots ===
!python -m scripts.generate_c3_dtd_analysis --results-dir results/dtd_c3_full --output-dir results/dtd_c3_analysis --table4-a0-dir results/dtd/dtd_a0_full --diag-file results/dtd_c3_analysis/c3_dtd_matched_diagnostics.json

from IPython.display import Image, display
plot_file = Path("results/dtd_c3_analysis/c3_dtd_stage_accuracy_comparison.png")
if plot_file.exists():
    display(Image(filename=str(plot_file)))"""))

    cells.append(make_code("""# === CELL 11: Cross-Condition Split & Matrix Verification ===
import zipfile

for c in ["c3_a_bicubic_normalized", "c3_b_bilinear_normalized", "c3_c_bicubic_raw", "c3_d_bilinear_raw"]:
    meta = json.loads(Path(f"results/dtd_c3_full/{c}/metadata.json").read_text(encoding="utf-8"))
    print(f"{c:<28}: split sha={meta['task_split_fingerprint']['sha256'][:16]}..., proj sha={meta['projection_metadata']['matrix_sha256'][:16]}...")

archive_name = "dtd_c3_results_bundle.zip"
with zipfile.ZipFile(archive_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for sub in ["dtd_c3_full", "dtd_c3_analysis"]:
        sub_dir = Path("results") / sub
        if sub_dir.exists():
            for f in sorted(sub_dir.rglob("*")):
                if f.is_file() and not f.name.endswith(".pt") and not f.name.endswith(".pth"):
                    zf.write(f, arcname=str(f.relative_to(Path("results"))))

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
    out_path = Path("notebooks/foal_c3_preprocessing_controlled_dtd.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
