# Table 4 Ablation Study Plan: Feature Fusion & Smooth Projection

## 1. Scientific Objective & Official Paper Provenance

In Section 4.3 and **Table 4 (Page 10)** of the NeurIPS 2024 paper:
**"F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning"**,
the authors present an architectural ablation study isolating the effects of:
1. **Feature Fusion (FF)**: Averaging the `[CLS]` tokens of all 12 transformer encoder layers vs using only the final (12th) block `[CLS]` token.
2. **Smooth Projection (SP)**: Random linear mapping to dimension $D=1000$ with sigmoid non-linearity vs directly optimizing on the raw $d=768$ feature space.

This plan outlines the two-stage execution protocol:
- **Stage 1 (Primary Reproduction & Core Scope)**: Complete 4-condition factorial ablation on **CIFAR-100**.
- **Stage 2 (Optional Research Extension)**: Extension to **DTD** (Describable Textures Dataset) or other benchmark datasets.

---

## 2. Official Table 4 Reference Ground Truth (NeurIPS 2024, Page 10)

The table below reflects the exact figures reported in Table 4 of the NeurIPS 2024 paper across all six evaluated benchmarks:

| FF | SP | CIFAR-100 | CORe50 | FGVCAircraft | DTD | Tiny-ImageNet | Country211 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\checkmark$ | $\checkmark$ | **91.1** | **96.3** | **62.2** | **82.8** | **91.2** | **24.4** |
| $\times$ | $\checkmark$ | 90.6 | 95.3 | 60.9 | 80.5 | 91.4 | 21.3 |
| $\checkmark$ | $\times$ | 90.7 | 95.4 | 58.7 | 79.3 | 91.2 | 22.8 |
| $\times$ | $\times$ | 90.6 | 95.4 | 56.0 | 71.2 | 91.4 | 21.1 |

> [!IMPORTANT]
> **Audit Correction on Tiny-ImageNet:**
> In previous preliminary drafts, Tiny-ImageNet was incorrectly transcribed with CIFAR-10-like numbers (~68%). As verified directly from Page 10 of the NeurIPS 2024 PDF, Tiny-ImageNet achieves **91.2%** for Full F-OAL and **91.4%** without FF. In F-OAL, Tiny-ImageNet operates with pre-trained ViT features that yield >91% accuracy.

---

## 3. Theoretical Background & Ablation Hypotheses

### 3.1 Feature Fusion (FF)
- **Enabled (`FF=True`)**:
  Let $h^{(l)} \in \mathbb{R}^{d}$ denote the `[CLS]` token representation extracted from layer $l \in \{1, \dots, 12\}$.
  The fused representation is:
  $$h = \text{LayerNorm}\left(\frac{1}{12}\sum_{l=1}^{12} h^{(l)}\right) \in \mathbb{R}^{768}$$
- **Disabled (`FF=False`)**:
  The representation uses only the final block's representation:
  $$h = \text{LayerNorm}(h^{(12)}) \in \mathbb{R}^{768}$$
- **Hypothesis**: Multi-layer aggregation combines intermediate semantic features (edges, textures, mid-level parts) with task-invariant representations, reducing over-specialization and catastrophic forgetting in incremental stages.

### 3.2 Smooth Projection (SP)
- **Enabled (`SP=True`)**:
  With projection matrix $P \in \mathbb{R}^{768 \times 1000}$:
  $$z = \sigma(h P) \in (0, 1)^{1000}$$
- **Disabled (`SP=False`)**:
  The fused or single-layer feature is fed directly into the analytic classifier without projection or non-linearity:
  $$z = h \in \mathbb{R}^{768}$$
  The RLS covariance matrix is $R \in \mathbb{R}^{768 \times 768}$, and classifier weights are $W \in \mathbb{R}^{768 \times K}$.
- **Hypothesis**: In highly separable domains (e.g. natural images in CIFAR-100), the linear 768-dim features already possess strong linear separability. However, the smooth non-linear projection expands feature space capacity ($768 \to 1000$) and suppresses extreme outliers via the bounded sigmoid activation.

---

## 4. Experimental Protocol & 2x2 Factorial Matrix on CIFAR-100

### 4.1 Conditions on CIFAR-100

| ID | Condition Name | Feature Fusion (FF) | Smooth Projection (SP) | Classifier Dim ($D$) | Paper $\bar{A}$ | Expected Delta from Full | Config File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A0** | **Full F-OAL** | ON | ON | 1000 | **91.1%** | Baseline | `configs/table4_a0_full_foal_cifar100.yaml` |
| **A1** | **w/o Feature Fusion** | OFF | ON | 1000 | **90.6%** | $-0.5\text{ pp}$ | `configs/table4_a1_without_ff_cifar100.yaml` |
| **A2** | **w/o Smooth Projection** | ON | OFF | 768 | **90.7%** | $-0.4\text{ pp}$ | `configs/table4_a2_without_sp_cifar100.yaml` |
| **A3** | **w/o Both (FF & SP)** | OFF | OFF | 768 | **90.6%** | $-0.5\text{ pp}$ | `configs/table4_a3_without_both_cifar100.yaml` |

### 4.2 Fixed Controls Across All Conditions
- **Backbone**: `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21K pretrained, ImageNet-1K fine-tuned).
- **Preprocessing**: Bilinear resizing, Raw $[0, 1]$ tensor (verified in C3).
- **Optimization**: Online Ridge RLS, $\gamma = 1.0$, `dtype=float64`, `batch_size=10`.
- **Dataset**: CIFAR-100 (10 tasks $\times$ 10 classes), Seed 42.
- **Cache Isolation**: Train and test features cached with explicit `ff` vs `noff` tags to prevent cache contamination.

### 4.3 A0 Reuse Decision
As audited in `scripts/verify_a0_reuse.py`, Condition A0 is identical in every parameter and hash to C3-D (`3v53tmq5`, $\bar{A}=90.4016\%$).
- `A0_REUSE_APPROVED`: Reused directly into results without re-running on GPU.
- Compute requirement reduced by 25% (only A1, A2, A3 need GPU training).

---

## 5. Stage 2 Roadmap: DTD Dataset (Optional Research Extension)

In Table 4 of the paper, the authors show that while CIFAR-100 drops by only $0.5\text{ pp}$, DTD drops precipitously when components are removed:
- Full F-OAL: **82.8%**
- w/o FF: **80.5%** ($-2.3\text{ pp}$)
- w/o SP: **79.3%** ($-3.5\text{ pp}$)
- w/o Both: **71.2%** ($-11.6\text{ pp}$)

This indicates that Feature Fusion and Smooth Projection serve as critical inductive biases when pre-trained representations have high domain shift. Stage 2 will evaluate this hypothesis once Stage 1 and the Headline Comparison are completed.

---

## 6. Full GPU Execution Verification & Factorial Audit (Seed 42)

- **Verification Status**: `TABLE4_SIGNED_OFF_WITH_LIMITATIONS`
- **Execution Date**: 2026-11-03
- **Archive File**: `foal_table4_ablation_results.zip` (14,370 bytes, CRC32 status: `ALL_OK` across all 14 files)
- **Archive SHA256**: `90f191d12b06fbd804779fcb038a12574c9b7cdc2f32369a6975dd6309e09e7a`
- **Extracted Location**: `results/table4_full_gpu_verified/`
- **Analysis Artifacts & Figures**: `results/table4_analysis/` (7 publication-quality figures, 7 CSV manifests, 3 JSON manifests)

### 6.1 Hardware and Software Execution Environment
- **GPU (A1, A2, A3)**: Google Colab Tesla T4 (Total Memory: 14.56 GB, CUDA Capability: 7.5)
- **GPU (Reused A0)**: Google Colab NVIDIA A100-SXM4-40GB (from C3-D run `3v53tmq5`)
- **Software Stack**: Linux 6.6.159+, Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.30, wandb 0.31.0
- **Git Commit of Execution**: `d40310f936ea1ff9ddc667793c5c5d5e9c2dde4e` (branch `experiment/table4-ablation`)

### 6.2 Baseline A0 Reuse Audit
- **Reused Source**: C3-D run `3v53tmq5` (`results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/`)
- **Provenance Decision**: `REUSED_VERIFIED_C3_D` (Approved in preflight; bitwise identical configuration, random seeds, backbone, projection matrix $P$, and preprocessing).
- **A0 Raw Metrics**:
  - $ar{A} = 90.4016\%$
  - $A_{10} = 85.7300\%$
  - $F_{	ext{past}} = 5.1556\%$
  - $F_{	ext{code}} = 4.6400\%$
- **Matrix SHA256**: `68684dfb3d348c5838815a63fa45dd874bd84b3a427cd29d06f4c320fa5de1d7` (100% bitwise parity across all 55 matrix entries).

### 6.3 Table 4 Verification Matrix (CIFAR-100, 10 Tasks, Seed 42)

| Condition | FF | SP | Dim ($D$) | $ar{A}$ (%) | $A_{10}$ (%) | $F_{	ext{past}}$ (%) | $F_{	ext{code}}$ (%) | T_RLS/Eval (s) | GPU Hardware | W&B Run ID | W&B Run Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---| :---: | :---: |
| **A0 (Full F-OAL)** | $\checkmark$ | $\checkmark$ | 1000 | **90.4016%** | **85.7300%** | **5.1556%** | **4.6400%** | 8.50 | A100-SXM4-40GB | [`3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | Reused C3-D |
| **A1 (w/o FF)** | $	imes$ | $\checkmark$ | 1000 | **90.1004%** | **85.4800%** | **5.4444%** | **4.9000%** | 15.52 | Tesla T4 (14.56 GB) | [`shl4p10l`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/shl4p10l) | [A1 Run](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/shl4p10l) |
| **A2 (w/o SP)** | $\checkmark$ | $	imes$ | 768 | **90.3823%** | **85.7200%** | **5.2444%** | **4.7200%** | 11.59 | Tesla T4 (14.56 GB) | [`rkedlkkk`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/rkedlkkk) | [A2 Run](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/rkedlkkk) |
| **A3 (w/o Both)** | $	imes$ | $	imes$ | 768 | **89.9224%** | **85.4800%** | **5.3222%** | **4.7900%** | 11.97 | Tesla T4 (14.56 GB) | [`k01pp8st`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/k01pp8st) | [A3 Run](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/k01pp8st) |

*Note on Runtime Fairness:* A0 was executed on an NVIDIA A100 GPU (8.50s RLS & Eval); conditions A1, A2, and A3 were executed on the same Tesla T4 GPU. A1 vs A2/A3 is a controlled hardware comparison showing ~25% speedup when reducing dimension $D$ from 1000 to 768.

### 6.4 Stage-by-Stage Incremental Accuracy Trajectory ($A_k$, %)

| Stage ($k$) | Classes | A0 (Full F-OAL) | A1 (w/o FF) | A2 (w/o SP) | A3 (w/o Both) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 10 | 96.10% | 96.10% | 96.20% | 95.30% |
| **2** | 20 | 95.05% | 94.90% | 94.95% | 94.50% |
| **3** | 30 | 93.27% | 92.83% | 93.20% | 92.63% |
| **4** | 40 | 91.50% | 91.25% | 91.53% | 91.07% |
| **5** | 50 | 90.58% | 90.46% | 90.68% | 90.26% |
| **6** | 60 | 89.45% | 89.03% | 89.33% | 88.92% |
| **7** | 70 | 88.63% | 88.14% | 88.54% | 88.06% |
| **8** | 80 | 87.30% | 86.84% | 87.35% | 86.91% |
| **9** | 90 | 86.41% | 85.97% | 86.32% | 86.09% |
| **10** | 100 | 85.73% | 85.48% | 85.72% | 85.48% |
| **Mean ($ar{A}$)** | — | **90.4016%** | **90.1004%** | **90.3823%** | **89.9224%** |

*Forensic Identity Verified:* The exact mean of the 10 stage accuracies equals $ar{A}$ within $< 10^{-12}$ for every single condition.

### 6.5 Final Per-Task Forgetting Breakdown ($f_j^{10}$, %)

| Task ($j$) | Classes in Task | $f_j$ (A0: Full) | $f_j$ (A1: w/o FF) | $f_j$ (A2: w/o SP) | $f_j$ (A3: w/o Both) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1** | 0 – 9 | 16.30% | 16.40% | 16.30% | 15.80% |
| **Task 2** | 10 – 19 | 11.90% | 13.00% | 11.00% | 12.50% |
| **Task 3** | 20 – 29 | 4.90% | 5.80% | 5.30% | 6.00% |
| **Task 4** | 30 – 39 | 4.20% | 5.30% | 4.50% | 4.70% |
| **Task 5** | 40 – 49 | 3.00% | 3.60% | 3.70% | 3.80% |
| **Task 6** | 50 – 59 | 2.80% | 2.80% | 3.10% | 2.30% |
| **Task 7** | 60 – 69 | 1.50% | 1.30% | 1.20% | 1.20% |
| **Task 8** | 70 – 79 | 1.10% | 0.80% | 1.50% | 1.10% |
| **Task 9** | 80 – 89 | 0.70% | 0.00% | 0.60% | 0.50% |
| **Task 10** | 90 – 99 | 0.00% | 0.00% | 0.00% | 0.00% |
| **$F_{	ext{past}}$ (9 past tasks)** | — | **5.1556%** | **5.4444%** | **5.2444%** | **5.3222%** |
| **$F_{	ext{code}}$ (10 tasks)** | — | **4.6400%** | **4.9000%** | **4.7200%** | **4.7900%** |

### 6.6 2x2 Factorial Effects Decomposition

- **Simple Effects**:
  - $E_{	ext{FF}|	ext{SP}} = A0 - A1 = \mathbf{+0.3013	ext{ pp}}$
  - $E_{	ext{FF}|
eg	ext{SP}} = A2 - A3 = \mathbf{+0.4599	ext{ pp}}$
  - $E_{	ext{SP}|	ext{FF}} = A0 - A2 = \mathbf{+0.0193	ext{ pp}}$
  - $E_{	ext{SP}|
eg	ext{FF}} = A1 - A3 = \mathbf{+0.1780	ext{ pp}}$
- **Main Effects**:
  - $	ext{Main Effect of Feature Fusion } (E_{	ext{FF}}) = rac{0.3013 + 0.4599}{2} = \mathbf{+0.3806	ext{ pp}}$
  - $	ext{Main Effect of Smooth Projection } (E_{	ext{SP}}) = rac{0.0193 + 0.1780}{2} = \mathbf{+0.0987	ext{ pp}}$
- **Interaction Contrast**:
  - $I_{	ext{FF} 	imes 	ext{SP}} = A0 - A1 - A2 + A3 = \mathbf{-0.1587	ext{ pp}}$
  - Regression coefficient $eta_{	ext{int}} = rac{I}{4} = \mathbf{-0.0397	ext{ pp}}$
- **Joint Component Benefit**:
  - $\Delta_{	ext{Full - Both}} = A0 - A3 = \mathbf{+0.4793	ext{ pp}}$ (Exact identity verified: $E_{	ext{FF}} + E_{	ext{SP}} = 0.4793	ext{ pp}$).

### 6.7 Comparison with Official Published Table 4 (NeurIPS 2024, Page 10)

| Benchmark Condition | Official Paper $ar{A}$ | Our Reproduced $ar{A}$ (Seed 42) | Paper Delta ($\Delta_{	ext{paper}}$) | Our Delta ($\Delta_{	ext{our}}$) | Directional Alignment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A0: Full F-OAL** | **91.1%** | **90.4016%** | Baseline | Baseline | Baseline |
| **A1: w/o Feature Fusion** | **90.6%** | **90.1004%** | $-0.5	ext{ pp}$ | $-0.3013	ext{ pp}$ | $\checkmark$ Drop confirmed |
| **A2: w/o Smooth Projection** | **90.7%** | **90.3823%** | $-0.4	ext{ pp}$ | $-0.0193	ext{ pp}$ | $\checkmark$ Drop confirmed |
| **A3: w/o Both** | **90.6%** | **89.9224%** | $-0.5	ext{ pp}$ | $-0.4793	ext{ pp}$ | $\checkmark$ Lowest accuracy confirmed |

**Key Takeaways**:
1. **Trend Reproduction vs Numerical Reproduction**:
   - The paper's qualitative findings are corroborated: removing Feature Fusion and/or Smooth Projection degrades performance, and removing both results in the lowest accuracy ($89.92\%$).
   - On CIFAR-100 Seed 42, Feature Fusion provides the primary architectural benefit ($+0.38	ext{ pp}$ main effect), while Smooth Projection provides a modest boost ($+0.10	ext{ pp}$ main effect).
   - The $-0.1587	ext{ pp}$ interaction contrast reveals slight negative sub-additivity on CIFAR-100: when Feature Fusion is present, high ViT representation quality renders the smooth projection largely redundant for separating natural image classes.
2. **Computational Footprint**:
   - Reducing dimension from $D=1000$ to $D=768$ (conditions A2 and A3 on Tesla T4) cuts evaluation and RLS inversion duration from $15.52	ext{s}$ to $11.59	ext{s}$–$11.97	ext{s}$ ($pprox 25\%$ speedup), consistent with $O(D^2)$ complexity.

### 6.8 Weights & Biases Verification
All online runs are registered on CoreWeave Forge W&B:
- Project: `caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction`
- Base URL: `https://api.forge.coreweave.com`
- Total Online Runs in Project: 17 runs (10 baseline C0-C3 runs + 4 Colab pilot runs + 3 full Table 4 ablation runs).
- Ablation Runs:
  - A0: https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5 (Group: `c3_preprocessing_alignment`)
  - A1: https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/shl4p10l (Group: `foal_no_fusion_ablation`)
  - A2: https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/rkedlkkk (Group: `foal_no_projection_ablation`)
  - A3: https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/k01pp8st (Group: `foal_neither_ablation`)

### 6.9 Scientific Verdict & Limitations
- **Scientific Verdict**: `TABLE4_SIGNED_OFF_WITH_LIMITATIONS`.
- **Known Limitations**:
  - Evaluation conducted on single seed (`seed=42`). Multi-seed variance assessment ($\ge 3$ seeds) is recommended for publication-grade error bounds.
  - On CIFAR-100, the absolute magnitude of ablation drops ($-0.02	ext{ to } -0.48	ext{ pp}$) is modest due to the high quality of pretrained ViT representations. Cross-dataset validation on DTD or Aircraft is recommended for large domain shift.
  - A0 was executed on an NVIDIA A100 GPU whereas A1, A2, A3 were executed on Tesla T4; runtime comparisons across A0 and A1-A3 are hardware-confounded.
