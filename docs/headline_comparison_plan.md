# Headline Comparison Reproduction Plan: Baselines & Benchmark Alignment

## 1. Executive Summary & Objective

The primary objective of Experiment 1 (mandated under `REQUIRED_BY_HCMUS`) is to reproduce the **Headline Benchmark Comparison** reported in **Table 2 (Page 7)** and Table 3 of the F-OAL paper (NeurIPS 2024).

The paper evaluates Continual Learning methods across three key metrics:
1. **Average Accuracy ($A_{avg}$)**: Mean incremental accuracy across all task stages.
2. **Last Task Accuracy ($A_{last}$)**: Final overall accuracy on all classes after completing the final task stage.
3. **Forgetting Rate ($F$)**: Metric quantifying accuracy degradation on earlier learned tasks.

---

## 2. Official Table 2 Baseline Ground Truth on CIFAR-100 (NeurIPS 2024, Page 7)

Table 2 in the paper evaluates twelve methods under the 10-task class-incremental CIFAR-100 benchmark. All methods employ a pre-trained Vision Transformer (`ViT-B/16`) backbone:

| Method | Venue & Ref | Replay Buffer? | $A_{avg}$ (%) $\uparrow$ | $A_{last}$ (%) $\uparrow$ | $F$ (%) $\downarrow$ | Category |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **iCaRL** | CVPR 2017 [31] | $\checkmark$ | 91.6 | 87.5 | 3.2 | Replay-based |
| **ER** | ICRA 2019 [12] | $\checkmark$ | 90.1 | 84.6 | 20.7 | Replay-based |
| **ASER** | AAAI 2021 [33] | $\checkmark$ | 87.2 | 82.0 | 16.5 | Replay-based |
| **SCR** | CVPR 2021 [25] | $\checkmark$ | 91.9 | 87.7 | 6.2 | Replay-based |
| **DVC** | CVPR 2022 [9] | $\checkmark$ | **92.4** | **87.8** | 8.2 | Replay-based (Best) |
| **PCR** | CVPR 2023 [21] | $\checkmark$ | 89.1 | 81.4 | 9.2 | Replay-based |
| **LwF** | TPAMI 2018 [20] | $\times$ | 69.3 | 64.8 | 1.3 | Exemplar-free |
| **EWC** | PNAS 2017 [17] | $\times$ | 49.9 | 25.2 | 67.4 | Exemplar-free |
| **EASE** | CVPR 2024 [42] | $\times$ | **91.1** | 85.4 | 6.1 | Exemplar-free |
| **LAE** | ICCV 2023 [8] | $\times$ | 79.1 | 75.6 | 11.8 | Exemplar-free |
| **SLCA** | ICCV 2023 [40] | $\times$ | 90.4 | 85.6 | 7.1 | Exemplar-free |
| **F-OAL (Paper)** | NeurIPS 2024 | $\times$ | **91.1** | **86.5** | **5.5** | Exemplar-free (Best Balance) |

> [!IMPORTANT]
> **Audit Corrections on Headline Baseline Figures:**
> - In preliminary notes, LwF was erroneously transcribed as 65.4% (actual: **69.3%**).
> - ER was erroneously transcribed as 78.2% (actual: **90.1%**).
> - EWC was listed as 50.1% (actual: **49.9%**).
> - **DER++ is NOT in Table 2.** The exemplar-based baselines evaluated in Table 2 are specifically iCaRL, ER, ASER, SCR, DVC, and PCR. DER++ may be referenced in broader CL literature or explored as an optional extension, but it is not an official Table 2 baseline.

---

## 3. Method Selection for Reproduction & Fairness Boundaries

Within the computational boundaries of Google Colab (single GPU, 1-pass online stream), our empirical Headline Comparison will implement and benchmark the following four representative paradigms:

| Paradigm | Selected Method | Buffer | Mechanism | Computational Cost |
| :--- | :--- | :---: | :--- | :---: |
| **Analytic Forward-Only (Proposed)** | **F-OAL (Full)** | 0 (Zero) | ViT + Feature Fusion + Smooth Projection + Online RLS | ~1 min |
| **Linear Analytic (Ablation A3)** | **Linear Ridge** | 0 (Zero) | ViT + Online Ridge RLS (no projection, no non-linearity) | ~1 min |
| **First-Order / Backprop** | **1-Pass SGD / AdamW** | 0 (Zero) | Frozen ViT + Fully Connected Head (Cross-Entropy, 1-pass) | ~2 min |
| **Literature Replay Benchmark** | **ER / iCaRL / DVC** | Exemplars | Literature figures from NeurIPS 2024 Table 2 | Reference Parity |

### 3.1 Strict Standardization Protocol
- **Frozen Backbone**: `vit_base_patch16_224.augreg2_in21k_ft_in1k`.
- **Input Stream**: CIFAR-100, 10 tasks $\times$ 10 classes, identical seed (42), identical class order.
- **Preprocessing**: Bilinear resizing, Raw $[0, 1]$ tensor normalization.
- **Single Pass Protocol**: Each training instance is seen exactly once during stream execution.
