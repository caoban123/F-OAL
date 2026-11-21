# F-OAL CIFAR-100 Scientific Reproduction & Forensic Audit Report

**Course Requirement:** Introduction to Machine Learning — HCMUS (Reproduction Project)  
**Lead Auditor & Author:** Senior ML Research Engineer & Scientific Reproducibility Auditor  
**Project Workspace:** `D:\Nhập môn ML\Paper`  
**Target Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024, pp. 1–14)  
**Official Repository:** [`https://github.com/liuyuchen-cz/F-OAL`](https://github.com/liuyuchen-cz/F-OAL)  
**Group Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL)  
**Development Branch:** `experiment/headline-comparison`  
**Base Stable Commit:** `747b828` (Merged Table 4 Ablation Suite)  
**Execution Commit (Official LwF GPU):** `c3dd3f2` (Descendant of preflight commit `33c058b`)  
**Audit Status:** `CIFAR100_REPORT_READY_WITH_LIMITATIONS`  

---

## 1. Executive Summary & Scientific Findings

This document presents the final scientific reproduction and forensic audit report for the **CIFAR-100 Benchmark** of the **F-OAL** continual learning framework. Across five systematic investigations (C0 Metric Alignment, C1 Random Projection Initialization, C2 Backbone Checkpoint Alignment, C3 Preprocessing Factorial Alignment, Table 4 Factorial Ablation, and Table 2 Headline Baseline Comparison), our team independently re-implemented the F-OAL pipeline, re-executed baseline methods on GPU hardware, and reconciled experimental outcomes against the published NeurIPS 2024 paper.

### Final Benchmark Summary Table (CIFAR-100, 10 Tasks $\times$ 10 Classes, Seed 42)

| Method | Method Classification | Replay Buffer | $\bar{A}$ (Ours) | $\bar{A}$ (Paper) | $\Delta \bar{A}$ | $A_{10}$ (Ours) | $A_{10}$ (Paper) | $\Delta A_{10}$ | $F_{\text{past}}$ (Ours) | $F$ (Paper) | Training Time | Hardware Environment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F-OAL (Full)** | `OUR_INDEPENDENT_FOAL` | 0 (Zero) | **90.40%** | **91.1%** | $-0.70$ pp | **85.73%** | **86.5%** | $-0.77$ pp | **5.16%** | **5.5%** | 229.11s | NVIDIA A100-SXM4 |
| **ER (Adapted)** | `OUR_ADAPTED_ER` | 5,000 | **86.67%** | **90.1%** | $-3.43$ pp | **79.79%** | **84.6%** | $-4.81$ pp | **16.74%** | **20.7%** | 626.85s | Tesla T4 GPU |
| **LwF (Official Protocol)** | `OUR_OFFICIAL_PROTOCOL_LWF` | 0 (Zero) | **46.29%** | **69.3%** | $-23.01$ pp | **23.34%** | **64.8%** | $-41.46$ pp | **0.41%** | **1.3%** | 646.35s | Tesla T4 GPU |
| **LwF (Adapted, AdamW)** | `OUR_ADAPTED_LWF` | 0 (Zero) | **29.63%** | **69.3%** | $-39.67$ pp | **11.21%** | **64.8%** | $-53.59$ pp | **94.99%** | **1.3%** | 629.52s | Tesla T4 GPU |

### Core Scientific Conclusions:
1. **F-OAL Reproduction on CIFAR-100:** F-OAL achieves $\bar{A} = 90.40\%$ and $A_{10} = 85.73\%$ with $5.16\%$ forgetting under a single seed (`seed=42`) without storing exemplars. This empirically reproduces the performance advantage of F-OAL over Experience Replay ($\bar{A} = 86.67\%$) and exemplar-free distillation ($\bar{A} = 46.29\%$) reported in Table 2 of the NeurIPS 2024 paper within the evaluated CIFAR-100 setup.
2. **Analysis of LwF Baselines (Adapted vs. Official-Inspired Protocol):**
   - **Adapted LwF (AdamW, dynamic head expansion, fixed 1.0/1.0 loss weighting):** Exhibited severe **recency collapse** ($F_{\text{past}} = 94.99\%$, $\bar{A} = 29.63\%$), remembering only the most recently trained task.
   - **Official-Protocol LwF (SGD $\text{lr}=0.001$, static 100-class head, dynamic weighting $\frac{1}{t+1} \mathcal{L}_{\text{new}} + (1 - \frac{1}{t+1}) \mathcal{L}_{\text{old}}$):** Preserved initial task accuracy ($90.1\% \to 86.4\%$), yielding low forgetting ($F_{\text{past}} = 0.41\%$, $F_{\text{code}} = 0.37\%$, consistent with the paper's reported $1.3\%$). Average accuracy rose by $+16.66\text{ pp}$ to $46.29\%$.
   - **Critical Scientific Caveat on LwF Forgetting:** The low forgetting score ($0.37\text{--}0.41\%$) is primarily an artifact of **catastrophic rigidity (under-learning)** on late tasks (Tasks 4–9 had initial accuracies $\le 5.9\%$, with Task 9 at $0.0\%$). When new classes are not adequately learned, they cannot disrupt earlier representations, resulting in artificially low forgetting metrics. The substantial gap between our final accuracy ($23.34\%$) and the paper's ($64.8\%$) remains an important reproduction divergence.
3. **Formal CIFAR-100 Status:** All planned GPU experiments for CIFAR-100 are concluded. No further GPU runs on CIFAR-100 are planned. Research compute shifts to DTD and FGVC-Aircraft.

---

## 2. Architecture & Feature Dimensionality Verification

### 2.1. Feature Fusion Specification ($D=768$, Not $1536$)
A discrepancy in earlier project documentation erroneously described Feature Fusion as a "concatenation of penultimate and final layers ($D=1536$)". Detailed code inspection of both the official author repository and our codebase confirms:
1. **Author Official Repository (`ref/official_foal/models/vit.py:22-33`):**
   ```python
   cumulative_output = 0
   for blk in self.vit.blocks:
       x = blk(x)
       cumulative_output += x[:, 0]  # Accumulate CLS token
   average_output = cumulative_output / len(self.vit.blocks)  # Mean across 12 blocks
   x = self.vit.norm(average_output)  # Final LayerNorm -> Output shape: [batch, 768]
   ```
2. **Our Implementation (`src/foal/feature_fusion.py` & `src/foal/backbone.py`):**
   ```python
   # fuse_cls_features averages CLS tokens across all 12 blocks:
   fused = torch.stack([f[:, 0, :] for f in block_features]).mean(dim=0)  # Shape: [batch, 768]
   fused = self.network.norm(fused)  # LayerNorm -> Feature dimension: 768
   ```
   **Confirmed Dimension:** In both the official implementation and our independent re-implementation, the ViT feature vector after Feature Fusion has dimensionality **$D = 768$**, exactly equal to `self.network.num_features`.

### 2.2. Classifier Dimensions in Official LwF
- **Input Dimension:** $D = 768$ (extracted from `FrozenViT` with Feature Fusion enabled).
- **Head Architecture:** `OfficialLwFClassifier(in_features=768, num_classes=100)` -> `nn.Linear(768, 100)`.
- **Author Baseline Difference:** In `ref/official_foal/models/pretrained.py:24-34`, the author's original `ModifiedViT` used standard single-layer final CLS tokens ($D=768$) rather than 12-block averaging. Because our baseline runner applied Feature Fusion to all methods for consistency across the comparison, our LwF run is classified as an **adapted/feature-aligned baseline** rather than a bitwise identical replication of the author's training script.

---

## 3. Execution Commit Provenance & Verification

The results archive `foal_cifar100_official_lwf_results.zip` recorded execution commit `c3dd3f23fbf2d8d94d481ec75daceb860190118e`. 

### Lineage & Code Diff Audit (`33c058b` vs. `c3dd3f2`):
- Commit `33c058b` was the preflight-verified commit containing the complete implementations of:
  - `src/foal/baselines/official_lwf.py`
  - `configs/headline/lwf_cifar100_official_protocol.yaml`
  - `scripts/train_baseline.py`
- Commit `c3dd3f2` is a direct child commit (`git log --oneline 33c058b..c3dd3f2`).
- The `git diff 33c058b..c3dd3f2` reveals that **only test decorators** were modified:
  ```diff
  --- a/tests/test_headline_comparison.py
  +++ b/tests/test_headline_comparison.py
  +@REQUIRES_HISTORICAL_ZIP
   def test_headline_zip_archive_and_manifest():
  ```
  `@pytest.mark.skipif` was applied to 8 historical archive tests to allow test execution on fresh Colab clones where git-ignored archives were absent.
- **Verification Verdict:** Zero algorithmic, architectural, or hyperparameter changes occurred between `33c058b` and `c3dd3f2`. The GPU execution on Google Colab ran the intended `OfficialLwFAgent` code.

---

## 4. Comprehensive Analysis of LwF Forgetting & Rigidity

In Continual Learning evaluation, forgetting metrics must be interpreted alongside forward learning ability (plasticity). The reported metrics for Official-Protocol LwF are:
- $\bar{A} = 46.2935\%$
- $A_{10} = 23.3400\%$
- $F_{\text{past}} = 0.4111\%$ (Equation 17, past tasks)
- $F_{\text{code}} = 0.3700\%$ (Equation 18, codebase formula)

### Forensic Inspection of the Raw Matrix (`task_matrix.csv`):
```text
Stage  1: [0.9010]
Stage  2: [0.8820, 0.6610]
Stage  3: [0.8790, 0.7320, 0.1930]
Stage  4: [0.8770, 0.7600, 0.2410, 0.1110]
Stage  5: [0.8740, 0.7740, 0.2740, 0.1430, 0.0080]
Stage  6: [0.8740, 0.7780, 0.3030, 0.1670, 0.0140, 0.0040]
Stage  7: [0.8770, 0.7830, 0.3240, 0.1850, 0.0280, 0.0050, 0.0000]
Stage  8: [0.8720, 0.7940, 0.3490, 0.2010, 0.0430, 0.0070, 0.0000, 0.0000]
Stage  9: [0.8690, 0.7980, 0.3610, 0.2150, 0.0510, 0.0070, 0.0010, 0.0000, 0.0000]
Stage 10: [0.8640, 0.7990, 0.3700, 0.2300, 0.0590, 0.0090, 0.0030, 0.0000, 0.0000, 0.0000]
```

### Key Scientific Observations:
1. **High Stability on Early Tasks:** Task 0 accuracy started at $90.10\%$ and dropped only to $86.40\%$ at Stage 10 ($\Delta = -3.70\text{ pp}$). Task 1 accuracy actually increased from $66.10\%$ to $79.90\%$ due to feature regularization.
2. **Catastrophic Rigidity on Late Tasks:**
   - As task index $t$ grows, the author's loss weighting assigns weight $\frac{1}{t+1}$ to new task cross-entropy and $\left(1 - \frac{1}{t+1}\right)$ to distillation from the teacher model.
   - At Stage 10 ($t=9$), new task loss has weight $0.10$ while distillation loss has weight $0.90$.
   - Under single-pass streaming (500 steps of batch size 10) with SGD $\text{lr}=10^{-3}$, this strong distillation constraint prevents the classifier weights from updating sufficiently on new classes.
   - Task 4 achieved initial accuracy of only $0.80\%$ ($0.0080$), and Tasks 7, 8, 9 achieved initial accuracies of $0.00\%$.
3. **Artifact of Near-Zero Forgetting:**
   - Because Tasks 4 through 9 never attained meaningful classification capability, they had negligible accuracy to lose in subsequent stages.
   - Therefore, the low forgetting score ($0.37\text{--}0.41\%$) reflects **a failure of plasticity (rigidity)** rather than healthy continual retention.
   - This explains why Last Accuracy ($23.34\%$) falls far short of the paper's reported $64.8\%$, representing a significant reproduction gap.

---

## 5. Hardware, Timing & Parity Disclosures

| Method | Hardware Platform | Total Training Duration | Timing Scope & Measurement Breakdown |
| :--- | :--- | :--- | :--- |
| **F-OAL (Full)** | NVIDIA A100-SXM4-40GB | **229.11s** | 175.28s feature extraction + 8.50s RLS solver & eval + 45.33s cache/I/O |
| **ER (Replay, Buffer=5000)** | NVIDIA Tesla T4 (14.56 GB) | **626.85s** | Online streaming loop (feature extraction + buffer sampling + SGD) |
| **LwF (Official Protocol)** | NVIDIA Tesla T4 (14.56 GB) | **646.35s** | Online streaming loop (feature extraction + KD computation + SGD) |
| **LwF (Adapted, AdamW)** | NVIDIA Tesla T4 (14.56 GB) | **629.52s** | Online streaming loop (feature extraction + dynamic head + AdamW) |

> [!WARNING]
> **Hardware Incommensurability Disclosure:**
> Direct speedup ratios between F-OAL and iterative baselines (e.g. earlier draft claims of an "11.7x speedup") are scientifically invalid because F-OAL was evaluated on an NVIDIA A100 GPU while baselines were evaluated on Tesla T4 GPUs. Within pure classifier solving time on matched features, F-OAL updates analytically in **8.50s**, demonstrating algorithmic efficiency, but total wall-clock times must not be compared directly as a controlled benchmark.

---

## 6. Project Scope Compliance (HCMUS Core Scope Self-Assessment)

Under the HCMUS evaluation rubric, the team selected the **Core Scope** option:
- **Mandatory Experiment 1 (Table 2 Headline Comparison):** Re-executed on CIFAR-100 with F-OAL Full ($90.40\%$), Experience Replay ($86.67\%$), and LwF ($46.29\%$ official protocol / $29.63\%$ adapted).
- **Mandatory Experiment 2 (Table 4 Factorial Ablation):** Re-executed on CIFAR-100 covering all four conditions: Full ($90.40\%$), w/o FF ($90.10\%$), w/o SP ($90.38\%$), and w/o Both ($89.92\%$). Merged into `main` at commit `747b828`.
- **Self-Designed Extensions:** Controlled single-factor diagnostic suites C1 (Projection distribution), C2 (Backbone checkpoint identification), and C3 (Preprocessing interpolation and pixel normalization).
- **Scope Boundary:** The original paper reported experiments on six datasets (CIFAR-100, DTD, FGVC-Aircraft, Food-101, Flowers-102, CUB-200). Our completed reproduction phase covers **CIFAR-100**. Parallel multi-dataset evaluations on DTD and FGVC-Aircraft represent ongoing extension work.

---

## 7. Artifact & Evidence Manifest

All final numerical artifacts and figures are versioned in [`results/headline_analysis/`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/):

| File | Type | Description |
| :--- | :---: | :--- |
| [`headline_final_metrics.csv`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_metrics.csv) | CSV | Master metrics table with feature dimension and architecture metadata. |
| [`headline_final_metrics.json`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_metrics.json) | JSON | Machine-readable metrics with full precision float64 values. |
| [`headline_final_paper_comparison.csv`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_paper_comparison.csv) | CSV | Comparison against NeurIPS 2024 Table 2 reference numbers with status annotations. |
| [`headline_final_stage_accuracies.csv`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_stage_accuracies.csv) | CSV | Incremental accuracy across all 10 stages for all 4 evaluated methods. |
| [`headline_final_task_forgetting.csv`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_task_forgetting.csv) | CSV | Catastrophic forgetting progression for Tasks 1 through 9. |
| [`headline_final_protocol_matrix.csv`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_protocol_matrix.csv) | CSV | Matrix mapping controlled protocol dimensions across methods. |
| [`headline_final_archive_manifest.json`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_archive_manifest.json) | JSON | SHA256 hashes of results archives. |
| [`headline_final_evidence_manifest.json`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/headline_final_evidence_manifest.json) | JSON | Traceability manifest linking raw files to reported metrics. |
| [`fig1_headline_final_average_accuracy.png`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/fig1_headline_final_average_accuracy.png) | Plot | Bar chart of paper vs. reproduced Average Accuracy. |
| [`fig2_headline_final_foal_vs_baselines.png`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/fig2_headline_final_foal_vs_baselines.png) | Plot | Comparative margin of F-OAL over baselines. |
| [`fig3_headline_final_adapted_vs_official_lwf.png`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/fig3_headline_final_adapted_vs_official_lwf.png) | Plot | Stage progression: Adapted vs. Official-Protocol LwF. |
| [`fig4_headline_final_stage_trajectories.png`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/fig4_headline_final_stage_trajectories.png) | Plot | Complete 10-stage trajectory curves for all 4 methods. |
| [`fig5_headline_final_last_accuracy.png`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/fig5_headline_final_last_accuracy.png) | Plot | Final Stage 10 Accuracy ($A_{\text{last}}$) comparison. |
| [`fig6_headline_final_forgetting_comparison.png`](file:///d:/Nhập môn ML/Paper/results/headline_analysis/fig6_headline_final_forgetting_comparison.png) | Plot | Catastrophic forgetting comparison ($F_{\text{past}}$). |

---

## 8. Limitations & Known Divergences

1. **Permutation Generator Discrepancy:** The author's code used `numpy.random.shuffle` (MT19937 RNG), whereas our setup used `numpy.random.default_rng(42).permutation(100)` (PCG64). Both are valid uniform permutations, but yield different class orders.
2. **Single-Seed Limitation:** All CIFAR-100 experiments were executed under `seed=42`. Multi-seed variance ($n=3$ or $n=5$) was not evaluated due to compute budget allocation.
3. **LwF Plasticity Deficit:** The official-protocol LwF baseline suffered from rigidity on late tasks under single-pass online SGD, explaining the gap between $23.34\%$ and the paper's $64.8\%$.

---

## 9. Conclusion & Transition to DTD and FGVC-Aircraft

The CIFAR-100 reproduction phase is concluded with documented limitations (**`CIFAR100_REPORT_READY_WITH_LIMITATIONS`**). 

The team moves to multi-dataset validation under [`docs/team_parallel_dataset_plan.md`](file:///d:/Nhập môn ML/Paper/docs/team_parallel_dataset_plan.md):
- **Member A (DTD Track):** `experiment/dtd-reproduction` (40 classes, 10 tasks $\times$ 4 classes). Test whether Smooth Projection prevents representation collapse when ViT features lack linear separability (paper reports $-11.6\text{ pp}$ drop without SP).
- **Member B (FGVC-Aircraft Track):** `experiment/fgvcaircraft-reproduction` (100 classes, 10 tasks $\times$ 10 classes). Test fine-grained class separation under Table 4 and investigate the Section 4.4 text discrepancy ($66.2\%$ vs. $62.2\%$).
