"""Generate notebooks/foal_dtd_full_reproduction.ipynb for Google Colab / Kaggle.

Follows the exact structure of notebooks/foal_cifar100_full_reproduction.ipynb:
1. Hardware & Environment Verification (!nvidia-smi, PyTorch, CUDA)
2. Clone Repository & Install Dependencies
3. Preflight Unit Test Suite
4. Single-Seed Full Reproduction (Seed 42) via scripts/train.py
5. Independent Checkpoint Evaluation via scripts/evaluate.py
6. Official Results Table & Summary via scripts/summarize_runs.py
7. Multi-Seed Full Reproduction (Optional - Phase P1.3) via scripts/run_seeds.py
8. Export & Archive Results
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

    cells.append(make_code("""import sys
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    raise SystemError("CUDA GPU not detected! Please enable GPU accelerator in Kaggle/Colab runtime settings.")"""))

    # Section 2: Clone Repository & Install Dependencies
    cells.append(make_md("""## 2. Clone Repository & Install Dependencies
Clone the repository if not already in workspace and install all dependencies."""))

    cells.append(make_code("""# If running in Colab/Kaggle and repo is not cloned, uncomment:
# !git clone https://github.com/caoban123/F-OAL.git
# %cd F-OAL

!pip install -q timm datasets wandb pyyaml pytest matplotlib pandas
!pip install -q -e ."""))

    # Section 3: Run Preflight Test Suite
    cells.append(make_md("""## 3. Run Preflight Test Suite
Verify mathematical correctness, Woodbury RLS updates, DTD dataset partitioning, and protocol isolation."""))

    cells.append(make_code("!python -m pytest tests/test_dtd_dataset.py tests/test_dtd_full_reproduction.py tests/test_rls.py tests/test_projection.py tests/test_metrics.py -v"))

    # Section 4: Single-Seed Full Reproduction (Seed 42)
    cells.append(make_md("""## 4. Single-Seed Full Reproduction (Seed 42)
Run F-OAL on full DTD benchmark (10 tasks $\\times$ 4 classes, 40 train and 40 test samples/class, streaming batch size 10, one-pass).

Checkpoints are saved after every task (`task_01.pt` ... `task_10.pt`)."""))

    cells.append(make_code("""# Set W&B mode: offline by default to avoid hanging if no API key is provided.
# To sync to cloud, configure your API key or W&B credentials:
# import os; os.environ["WANDB_API_KEY"] = "your_api_key_here"

!python -m scripts.train \\
    --config configs/dtd_full_reproduction.yaml \\
    --seed 42 \\
    --device cuda \\
    --run-name foal_full_dtd_seed42 \\
    --wandb \\
    --wandb-offline"""))

    # Section 5: Independent Checkpoint Evaluation
    cells.append(make_md("""## 5. Independent Checkpoint Evaluation
Verify checkpoint roundtrip and numerical stability by loading `task_10.pt` from disk into a fresh model instance and evaluating against all 10 tasks."""))

    cells.append(make_code("""!python -m scripts.evaluate \\
    --checkpoint results/foal_full_dtd_seed42/task_10.pt \\
    --device cuda"""))

    # Section 6: Official Results Table & Summary
    cells.append(make_md("""## 6. Official Results Table & Summary
Summarize the reproduction run using `scripts/summarize_runs.py`.
The summarizer strictly verifies run completeness, isolates protocols, and exports summary CSVs."""))

    cells.append(make_code("!python -m scripts.summarize_runs --results-dir results"))

    # Section 7: Multi-Seed Full Reproduction (Optional — Phase P1.3)
    cells.append(make_md("""> **NOTE:** In Phase P1.2, only Single-Seed (Seed 42) is executed.
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

    # Section 8: Export & Archive Results
    cells.append(make_md("""## 8. Export & Archive Results
Compress all result checkpoints, CSVs, metrics, and metadata into a zip archive for persistent download."""))

    cells.append(make_code("!zip -r foal_dtd_single_seed42_results.zip results/"))

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
