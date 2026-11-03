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
