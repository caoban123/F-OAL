# F-OAL: Comprehensive Source-Level Baseline Protocol Parity Audit

**Project:** F-OAL Reproduction (NeurIPS 2024) — HCMUS  
**Auditor:** Senior ML Research Engineer, Continual Learning Researcher & Reproducibility Auditor  
**Date:** November 3, 2026  
**Workspace:** `D:\Nhập môn ML\Paper`  
**Current Branch:** `experiment/headline-comparison`  
**Official Repository:** `https://github.com/liuyuchen-cz/F-OAL`  
**Official Git SHA (Pinned):** `73e4ac1b4c965030973a2b77f05c57c2f44f49f5`  
**Latest Local Commit:** `1ef8dd9`  

---

## 1. Executive Summary & Audit Context

To establish complete scientific transparency for the **Headline Comparison** benchmarks (NeurIPS 2024 Table 2 on CIFAR-100), this audit performs a line-by-line forensic comparison between the official author baseline implementations in `ref/official_foal/` and our reproduced implementations in `src/foal/baselines/`.

### Verified Empirical Performance Summary:
- **F-OAL Full (Ours):** $\bar{A} = 90.4016\%$, $A_{10} = 85.7300\%$, $F_{\text{past}} = 5.1556\%$ (Paper: $91.1\%$, $86.5\%$, $5.5\%$, $\Delta \bar{A} = -0.70\text{ pp}$).
- **ER (Replay, $M=5000$):** $\bar{A} = 86.6703\%$, $A_{10} = 79.7900\%$, $F_{\text{past}} = 16.7444\%$ (Paper: $90.1\%$, $84.6\%$, $20.7\%$, $\Delta \bar{A} = -3.43\text{ pp}$).
- **LwF (Distillation, $M=0$):** $\bar{A} = 29.6295\%$, $A_{10} = 11.2100\%$, $F_{\text{past}} = 94.9889\%$ (Paper: $69.3\%$, $64.8\%$, $1.3\%$, $\Delta \bar{A} = -39.67\text{ pp}$).

The raw $10 \times 10$ matrices (55 entries each) have been verified to $< 10^{-12}$ mathematical precision. This audit diagnoses the source-level protocol and algorithmic divergence explaining the $-39.67\text{ pp}$ gap in LwF and $-3.43\text{ pp}$ gap in ER.

---

## 2. Line-by-Line Source Protocol Diff Matrix

Every dimension is classified into one of:
- `MATCHED`: Protocol or implementation is identical.
- `INTENTIONAL_METHOD_DIFFERENCE`: Difference dictated by core design differences between methods.
- `ADAPTED`: Methodologically sound adaptation to match frozen feature workflow.
- `MISMATCH`: Substantive implementation divergence from the author's official code.
- `UNKNOWN`: Not specified in paper or repository.

### 2.1. LwF (Learning without Forgetting) Comparison

| Dimension | Official Implementation (`liuyuchen-cz/F-OAL @ 73e4ac1`) | Adapted Baseline (`src/foal/baselines/lwf.py`) | Classification | Source Evidence & Line References |
|:---|:---|:---|:---:|:---|
| **Backbone Architecture** | `ModifiedViT` (`vit_base_patch16_224`, `timm`) | `FrozenViT` (`augreg2_in21k_ft_in1k`, `timm`) | `ADAPTED` | Official: `models/pretrained.py:28`<br>Ours: `src/foal/backbone.py:48-52` |
| **Backbone Trainability** | Trainable with SGD (`model.parameters()` in opt) | Strictly Frozen (`requires_grad = False`) | `MISMATCH` | Official: `utils/setup_elements.py:163`<br>Ours: `src/foal/backbone.py:51` |
| **Classifier Head Architecture** | Static 100-class linear head `nn.Linear(768, 100)` | Dynamically expanding head ($10 \to 20 \dots \to 100$) | `MISMATCH` | Official: `models/pretrained.py:33`<br>Ours: `src/foal/baselines/lwf.py:31-47` |
| **Optimizer** | `SGD` (no momentum, weight_decay = 0.0) | `AdamW` (lr=0.001, weight_decay=1e-4) | `MISMATCH` | Official: `general_main.py:62, 76`<br>Ours: `src/foal/baselines/lwf.py:76-80` |
| **Learning Rate** | `lr = 0.001` (CLI flag in README) | `learning_rate = 0.001` | `MATCHED` | Official: `README.md:22`<br>Ours: `configs/headline/lwf_cifar100.yaml:15` |
| **KD Output Scope** | Full 100-class output logits | Sliced to previously seen classes only (`[:C_old]`) | `MISMATCH` | Official: `utils/kd_manager.py:7-10`<br>Ours: `src/foal/baselines/lwf.py:119-123` |
| **Loss Weighting Scheme** | Dynamic: $\frac{1}{t+1} \mathcal{L}_{\text{new}} + (1 - \frac{1}{t+1}) \mathcal{L}_{\text{old}}$ | Static: $1.0 \times \mathcal{L}_{\text{new}} + 1.0 \times \mathcal{L}_{\text{old}}$ | `MISMATCH` | Official: `agents/lwf.py:55`<br>Ours: `src/foal/baselines/lwf.py:126` |
| **Temperature ($T$)** | $T = 2.0$ | $T = 2.0$ | `MATCHED` | Official: `utils/kd_manager.py:6`<br>Ours: `src/foal/baselines/lwf.py:57` |
| **Teacher Snapshot Timing** | Updated after each task (`after_train`) | Updated before each task (`before_task`) | `MATCHED` | Official: `agents/base.py:92`<br>Ours: `src/foal/baselines/lwf.py:85` |
| **Streaming Pass** | Exactly 1 epoch (`--epoch 1`) | Exactly 1 pass over stream mini-batches | `MATCHED` | Official: `general_main.py:67`<br>Ours: `scripts/train_baseline.py:209-214` |
| **Batch Size** | `--batch 10` | `batch_size: 10` | `MATCHED` | Official: `general_main.py:70`<br>Ours: `configs/headline/lwf_cifar100.yaml:14` |
| **Image Preprocessing** | Bilinear resize 224x224, Raw $[0, 1]$ | Bilinear resize 224x224, Raw $[0, 1]$ | `MATCHED` | Official: `utils/setup_elements.py:49-53`<br>Ours: `cifar100_transforms(norm=False)` |

---

### 2.2. ER (Experience Replay) Comparison

| Dimension | Official Implementation (`liuyuchen-cz/F-OAL @ 73e4ac1`) | Adapted Baseline (`src/foal/baselines/er.py`) | Classification | Source Evidence & Line References |
|:---|:---|:---|:---:|:---|
| **Backbone Architecture** | `ModifiedViT` (`vit_base_patch16_224`, `timm`) | `FrozenViT` (`augreg2_in21k_ft_in1k`, `timm`) | `ADAPTED` | Official: `models/pretrained.py:28`<br>Ours: `src/foal/backbone.py:48-52` |
| **Backbone Trainability** | Trainable with SGD (`model.parameters()` in opt) | Strictly Frozen (`requires_grad = False`) | `MISMATCH` | Official: `utils/setup_elements.py:163`<br>Ours: `src/foal/backbone.py:51` |
| **Buffer Storage Mechanism** | Raw pixel images `(5000, 3, 224, 224)` | Extracted feature vectors `(5000, 768)` | `ADAPTED` | Official: `utils/buffer/buffer.py:22`<br>Ours: `src/foal/baselines/er.py:28` |
| **Buffer Capacity** | 5,000 slots (`--mem_size 5000`) | 5,000 slots (`buffer_capacity: 5000`) | `MATCHED` | Official: `README.md:24`<br>Ours: `configs/headline/er_cifar100.yaml:17` |
| **Update Policy** | Reservoir sampling | Reservoir sampling | `MATCHED` | Official: `utils/buffer/reservoir_update.py:8-50`<br>Ours: `src/foal/baselines/er.py:35-50` |
| **Retrieval Policy** | Uniform random sample (`eps_mem_batch = 10`) | Uniform random sample (10 exemplars) | `MATCHED` | Official: `utils/buffer/random_retrieve.py:9`<br>Ours: `src/foal/baselines/er.py:56-66` |
| **Gradient Flow & Batching** | Dual backward (`loss.backward()` then `loss_mem.backward()`) | Combined batch `torch.cat([stream, mem])` | `ADAPTED` | Official: `agents/exp_replay.py:70, 92`<br>Ours: `src/foal/baselines/er.py:118-127` |
| **Optimizer** | `SGD(lr=0.001, weight_decay=0.0)` | `AdamW(lr=0.001, weight_decay=1e-4)` | `MISMATCH` | Official: `general_main.py:62`<br>Ours: `src/foal/baselines/er.py:94-98` |
| **Streaming Pass** | Exactly 1 epoch (`--epoch 1`) | Exactly 1 pass over stream mini-batches | `MATCHED` | Official: `general_main.py:67`<br>Ours: `scripts/train_baseline.py:209-214` |
| **Batch Size** | `--batch 10` | `batch_size: 10` | `MATCHED` | Official: `general_main.py:70`<br>Ours: `configs/headline/er_cifar100.yaml:14` |
| **Image Preprocessing** | Bilinear resize 224x224, Raw $[0, 1]$ | Bilinear resize 224x224, Raw $[0, 1]$ | `MATCHED` | Official: `utils/setup_elements.py:49-53`<br>Ours: `cifar100_transforms(norm=False)` |

---

## 3. Forensic Investigation of LwF Recency Bias Collapse

### 3.1. Concrete Code Evidence of Recency Collapse
The official author repository itself provides direct evidence that the authors actively monitored recency bias:
In `ref/official_foal/agents/lwf.py:75-100` and `ref/official_foal/agents/exp_replay.py:120-148`, the author code explicitly computes:
```python
weights = self.model.fc.weight.data.cpu()
row_norms = torch.norm(weights, p=2, dim=1)
# Plotting L2 norm per class categorized by Current Task (red), Completed (blue), Future (yellow)
plt.savefig(f"recencybiasER {self.k + 1}.pdf")
```
This proves that classifier norm imbalance across tasks was a central empirical concern of the original authors.

### 3.2. Three Compounding Causes of the 29.63% Result in the Adapted Baseline

1. **Unbalanced Loss Weighting in Stream Execution:**
   - In the official code (`agents/lwf.py:55`), the author uses:
     $$\text{loss} = \frac{1}{t+1} \mathcal{L}_{\text{new}} + \left(1 - \frac{1}{t+1}\right) \mathcal{L}_{\text{old}}$$
     By Task 10 ($t=9$), the official code weights the new cross-entropy loss by only $0.10$ and the distillation loss by $0.90$.
   - In our adapted baseline (`src/foal/baselines/lwf.py:126`), the cross-entropy loss was unweighted ($1.0 \times \mathcal{L}_{\text{new}} + 1.0 \times \mathcal{L}_{\text{old}}$). Over 500 batches, this continually suppressed old-class logits without counterweight.
2. **Logit Slicing Isolation:**
   - In our adapted baseline, `curr_old_logits = logits[:, :self.prev_num_classes]` was sliced before computing `F.log_softmax(curr_old_logits / T)`. Because softmax is shift-invariant, subtracting any large constant from all old logits produces zero distillation penalty, disconnecting old logit scales from new logit scales.
   - In the official code, `loss_fn_kd` operates over all 100 classes jointly.
3. **Optimizer Dynamics (AdamW vs. Pure SGD):**
   - In AdamW with learning rate 0.001, parameter updates scale inversely with gradient historical variance ($\frac{\eta}{\sqrt{v_t} + \epsilon}$). In 500 batches on a single linear head, AdamW drives old class weights rapidly downward.
   - In pure SGD with learning rate 0.001 (without momentum), parameter updates are directly proportional to the tiny gradient magnitude $\Delta w = -\eta \nabla \mathcal{L}$, preserving historical weights much more conservatively.

---

## 4. Runtime Breakdown Reconciliation

From the verified C3-D raw metrics (`metrics.json` and `output.log` of run `3v53tmq5` on NVIDIA A100 GPU):

$$\text{Total Duration (Recorded)} = \mathbf{229.11\text{ s}}$$

Detailed component audit:
- **`feature_extraction_seconds`:** **$175.28\text{ s}$** (Wall-clock time spent inside `extract_or_load_task_features` executing forward passes through the ViT encoder).
- **`rls_and_eval_seconds`:** **$8.50\text{ s}$** (Cumulative time executing analytic Woodbury recursive least squares matrix updates and evaluate passes on test features across all 10 stages).
- **Subtotal identified:** $175.28\text{ s} + 8.50\text{ s} = \mathbf{183.78\text{ s}}$.
- **Remaining Unallocated Difference:**
  $$229.11\text{ s} - 183.78\text{ s} = \mathbf{45.33\text{ s}}$$

### Source-Level Explanation of the 45.33s Unallocated Runtime:
Looking at lines 446–550 of `scripts/train.py`, the following operations occur outside the timed blocks:
1. **Feature Cache Disk Serialization (`torch.save`):** Writing 10 train `.pt` files and 10 test `.pt` files to disk and verifying disk cache hashes.
2. **PyTorch DataLoader Overhead:** Creating 10 training `DataLoader` instances and 55 evaluation `DataLoader` instances with index mapping.
3. **Projection Matrix Integrity Checks:** Computing cryptographic SHA256 hashes of the random projection matrix $P$ after each stage to guarantee zero parameter drift.
4. **W&B Metric Synchronization & Metadata Logging:** Writing JSON summaries and step records.

### Runtime Parity Retraction Verdict:
- F-OAL on NVIDIA A100: 229.11s (end-to-end task stream).
- LwF on Tesla T4: 629.52s (end-to-end task stream).
- ER on Tesla T4: 626.85s (end-to-end task stream).
- **Official Policy:** Because hardware platforms differ (A100 vs. T4) and the official repository benchmarks did not report FLOP-matched hardware times, **all cross-hardware speedup claims (such as "11.7x faster") are formally retracted**.

---

## 5. Weights & Biases Logging Status

- The existing Headline archive `foal_cifar100_headline_results.zip` (SHA256: `6ec6a4a4cadecb4e77b7704a217878356ac6f6cd4e2d72c8a78a3813cceea551`) was executed in an offline Colab environment.
- In accordance with our Anti-Hallucination policy, no synthetic online runs will be fabricated on CoreWeave Forge.
- The provenance classification for Headline Comparison is: **`OFFLINE_LOCAL_ARTIFACT_VERIFIED`**.

---

## 6. Official-Protocol Baseline Specifications & Rerun Plan

To maintain absolute scientific fidelity without discarding historical data, the codebase classifies results under two distinct tiers:

1. **`ADAPTED_BASELINE_EXISTING_RESULT` (Historical):**
   - Preserves our existing verified results: LwF = 29.63%, ER = 86.67%.
   - Fully documented as adapted baselines on frozen ViT features with AdamW.
2. **`OFFICIAL_PROTOCOL_BASELINE_PENDING` (Paper-Faithful):**
   - New configuration: `configs/headline/lwf_cifar100_official_protocol.yaml`.
   - Uses exact official parameters: SGD optimizer (`lr=0.001`, `momentum=0.0`, `weight_decay=0.0`), dynamic loss weighting $\frac{1}{t+1} \mathcal{L}_{\text{new}} + (1 - \frac{1}{t+1}) \mathcal{L}_{\text{old}}$, full 100-class logit distillation.
   - **ER Evaluation:** Since our adapted ER ($86.67\%$) is within $-3.43\text{ pp}$ of the paper ($90.1\%$) and successfully confirms that memory replay prevents catastrophic forgetting, **rerunning ER on Full GPU is scientifically optional** and may be deferred under compute constraints.
   - **LwF Evaluation:** Because LwF has a $-39.67\text{ pp}$ gap directly traceable to the optimizer and dynamic loss weighting mismatches, **running official-protocol LwF is the single necessary GPU rerun**.

---

## 7. Official Closeout Recommendation

**Scientific Recommendation:** **`CIFAR100_NEEDS_OFFICIAL_LWF_RUN`**

- The F-OAL core method reproduction is 100% complete and validated ($\bar{A} = 90.40\%$ vs $91.1\%$).
- The Table 4 architectural ablation is 100% complete and merged.
- Resolving the LwF headline comparison under the exact author SGD protocol will close CIFAR-100 definitively with minimal GPU compute (~10 minutes on a single T4 instance).
- Parallel cross-dataset work by teammates (DTD and FGVC-Aircraft) remains completely protected and unblocked.
