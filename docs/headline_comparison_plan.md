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
> - **DER++ is NOT in Table 2.** The exemplar-based baselines evaluated in Table 2 are specifically iCaRL, ER, ASER, SCR, DVC, and PCR.

---

## 3. Prioritized Headline Baseline Suite (Table 2 Parity)

Following the objective CIFAR-100 completion checklist (`docs/cifar100_completion_checklist.md`), our Headline Comparison implements and directly benchmarks the two core continual learning paradigms evaluated in NeurIPS 2024 Table 2 on identical frozen ViT representations:

| Method | Venue & Type | Buffer Size | Hyperparameters / Provenance | Table 2 Ref $\bar{A}$ | Table 2 Ref $A_{10}$ | Table 2 Ref $F$ |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **F-OAL (Full)** | NeurIPS 2024 (Analytic) | 0 (Zero) | Official uniform SP, Feature Fusion, $\gamma=1.0$, float64 | **91.1%** | **86.5%** | **5.5%** |
| **ER (Replay)** | ICRA 2019 (Replay) | 5000 | Reservoir update, random retrieval, lr 0.001, AdamW | **90.1%** | **84.6%** | **20.7%** |
| **LwF (Distill)** | TPAMI 2018 (Distill) | 0 (Zero) | Distillation loss $T=2.0$, $\lambda=1.0$, lr 0.001, AdamW | **69.3%** | **64.8%** | **1.3%** |

### 3.1 Strict Standardization Protocol
- **Frozen Backbone**: `vit_base_patch16_224.augreg2_in21k_ft_in1k`.
- **Input Stream**: CIFAR-100, 10 tasks $\times$ 10 classes, seed 42, identical class order.
- **Preprocessing**: Bilinear resizing, Raw $[0, 1]$ tensor normalization.
- **Single Pass Protocol**: Each training instance is seen exactly once during stream execution (batch size 10).

---

### 3.2 Official Protocol Source Diff Findings (Nov 3, 2026)
Detailed source code comparison against official author implementation (`ref/official_foal/` pinned at `73e4ac1b4c965030973a2b77f05c57c2f44f49f5`) revealed key protocol parameters in the author's LwF baseline:
- **Dynamic Loss Weighting**: $\mathcal{L} = \frac{1}{t+1} \mathcal{L}_{\text{new}} + \left(1 - \frac{1}{t+1}\right) \mathcal{L}_{\text{old}}$. At Task 10, new task cross-entropy is scaled down to 0.10 while distillation loss is scaled up to 0.90, heavily suppressing recency bias.
- **Logit Scope for Distillation**: Author uses a static 100-class head (`nn.Linear(768, 100)`) and computes KD loss across all 100 classes (with softmax distillation), which prevents unconstrained growth of new class logits.
- **Optimizer Dynamics**: Author uses standard `SGD(lr=0.001, momentum=0.0, weight_decay=0.0)` instead of `AdamW`.
- **Dedicated Implementation**: Implemented `OfficialLwFAgent` in `src/foal/baselines/official_lwf.py` and configuration `configs/headline/lwf_cifar100_official_protocol.yaml` to ensure 100% fidelity.

---

## 4. Empirical Implementation & Verification Deliverables

1. **Baseline Agents**:
   - `src/foal/baselines/lwf.py`: Adapted LwF baseline (dynamic head expansion, AdamW, static weighting).
   - `src/foal/baselines/official_lwf.py`: Paper-faithful baseline matching official author code (static 100-class head, dynamic loss weighting $\frac{1}{t+1}$, SGD lr=0.001, full 100-class KD).
   - `src/foal/baselines/er.py`: Full ER implementation with reservoir memory buffer (capacity 5000) and uniform sampling.
2. **Unified Runner**:
   - `scripts/train_baseline.py`: Standalone CLI runner with full parity to `scripts/train.py`, evaluating cumulative task accuracy matrix $R_{i,j}$ and computing $A_{avg}$, $A_{last}$, $F_{past}$. Supports `lwf`, `official_lwf`, and `er`.
3. **Configurations**:
   - `configs/headline/foal_cifar100_reference.yaml`: Anchored to verified C3-D / A0 run (`3v53tmq5`, $\bar{A}=90.4016\%$, $A_{10}=85.7300\%$, $F_{code}=4.6400\%$).
   - `configs/headline/lwf_cifar100.yaml`: Adapted baseline config.
   - `configs/headline/lwf_cifar100_official_protocol.yaml`: Official protocol baseline config.
   - `configs/headline/er_cifar100.yaml`: Experience Replay config.
4. **Push-Button Colab Workflow**:
   - `notebooks/foal_cifar100_headline_final.ipynb`: End-to-end executable notebook with environment checks, pilot validation, full GPU runs (including official protocol cell 6b), comparative synthesis table, and results ZIP export.
5. **Acceptance Gate**:
   - `docs/cifar100_completion_checklist.md`: 15-item objective checklist in status `CIFAR100_NEEDS_OFFICIAL_LWF_RUN`.
   - `docs/headline_official_source_diff.md`: Comprehensive 12-dimension protocol diff auditing official author code line-by-line.
