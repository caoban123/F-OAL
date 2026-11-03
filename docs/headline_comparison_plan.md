# Headline Comparison Reproduction Plan: Baselines & Benchmark Alignment

## 1. Executive Summary & Objective

The primary objective of Experiment 1 (mandated under `REQUIRED_BY_HCMUS`) is to reproduce the **Headline Benchmark Comparison** reported in Table 2 and Table 3 of the F-OAL paper (NeurIPS 2024).

The paper evaluates Continual Learning methods across three key dimensions:
1. **Average Accuracy ($\bar{A}$)**: Mean accuracy over all tasks after learning each task incrementally.
2. **Final Task Accuracy ($A_T$) & Forgetting ($F_T$)**: Retention of historical classes.
3. **Training Time & Memory Footprint**: Computational efficiency in single-pass online streams.

This document defines the baseline selection, mathematical formulations, fairness boundaries, and execution roadmap.

---

## 2. Baseline Method Taxonomy & Selection

To provide a rigorous, balanced comparison within Colab GPU constraints (single NVIDIA T4/A100), we categorize baselines into four representative families:

| Family | Representative Baseline | Description | Replay Buffer | Training Mechanism | Compute Cost |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Analytic / Forward-Only** | **F-OAL (Ours / Paper)** | ViT feature extraction + Smooth Projection + Online RLS | 0 (Zero) | Closed-form recursive Woodbury updates | Minimal (~1 min) |
| **Linear Analytic** | **Linear Ridge (A3)** | ViT feature extraction (no projection) + Online RLS | 0 (Zero) | Closed-form recursive Woodbury updates | Minimal (~1 min) |
| **First-Order / Backpropagation** | **1-Pass SGD / AdamW** | Frozen ViT + Trainable Linear Classification Head | 0 (Zero) | Gradient descent, 1 epoch per task, batch size 10 | Low (~2 min) |
| **Exemplar Replay (Literature)** | **ER / DER++** | Buffer replay methods reported in NeurIPS 2024 Table 2 | 200–500 exemplars | Multi-task gradient replay | Literature Benchmark |

### 2.1 Baseline Specifications

1. **F-OAL (Full Method)**:
   - Evaluated under our verified pipeline (C3-D / A0).
   - Reported: $\bar{A} = 91.1\%$, Replicated: $\bar{A} = 90.40\%$.

2. **Linear Ridge Regression (F-OAL w/o Non-Linearity)**:
   - Direct online ridge regression on fused ViT features ($d=768$).
   - Serves as the strictly linear analytic lower bound.

3. **1-Pass Backpropagation (SGD / AdamW on Linear Head)**:
   - Frozen ViT backbone with a standard linear layer trained using Cross-Entropy loss via single-pass gradient descent.
   - Evaluates whether analytic closed-form learning outperforms gradient descent in the strict online single-pass setting (where SGD suffers from severe underfitting and catastrophic forgetting).

4. **Experience Replay & Advanced CL Baselines (Reference Parity)**:
   - Taken directly from published results in Table 2 of NeurIPS 2024:
     - Finetune: $\bar{A} = 48.7\%$ (severe forgetting)
     - EWC: $\bar{A} = 50.1\%$
     - LwF: $\bar{A} = 65.4\%$
     - ER ($M=500$): $\bar{A} = 78.2\%$
     - DER++ ($M=500$): $\bar{A} = 81.4\%$

---

## 3. Fairness Boundaries & Standardization

To ensure fair scientific comparison, all implemented methods must strictly adhere to the following protocol:
- **Identical Feature Extractor**: Frozen `vit_base_patch16_224.augreg2_in21k_ft_in1k`.
- **Identical Input Stream**: CIFAR-100, 10 tasks $\times$ 10 classes, identical class ordering (seed 42), identical sample presentation sequence.
- **Identical Preprocessing**: Bilinear resizing, Raw $[0, 1]$ tensor normalization.
- **Zero Exemplar Replay**: No raw images stored in memory across task boundaries (for online forward-only methods).
- **Single Pass Protocol**: Each sample is seen exactly once during stream execution.

---

## 4. Compute Budget & Implementation Milestones

- **Implementation Location**: `scripts/baselines/` and `src/foal/baselines/`.
- **Verification**: Unit tests in `tests/test_headline_baselines.py`.
- **Execution**: Colab notebook `notebooks/foal_headline_comparison.ipynb`.
- **Runtime Budget**: Under 15 minutes total on single GPU using cached ViT features.
