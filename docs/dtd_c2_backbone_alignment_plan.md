# F-OAL — C2 Controlled Pretrained ViT Backbone Alignment Plan on DTD (Describable Textures Dataset)

> **Status:** `SPECIFICATION_AND_CODE_COMPLETE`  
> **Target Dataset:** Describable Textures Dataset (DTD) via HuggingFace `tanganke/dtd` / `torchvision.datasets.DTD`  
> **Target Branch:** `experiment/dtd-ablation`  
> **Chaining Guarantees:**  
> - `C2-B0 == C1-P1` (B0 Reproducibility Guard)  
> - `C2-B1 == C3-A` (Zero-compute reuse for C3 2x2 factorial baseline)  

---

## 1. Research Question & Objective

**Research Question:**  
*Under strict single-factor control on the Describable Textures Dataset (DTD), how much of the performance gap in Forward-only Online Analytic Learning (F-OAL) is driven by the Pretrained Vision Transformer checkpoint: ImageNet-1k only (`vit_base_patch16_224.augreg_in1k`, B0) versus ImageNet-21k pretraining with ImageNet-1k fine-tuning (`vit_base_patch16_224.augreg2_in21k_ft_in1k`, B1)?*

### Context & Single-Factor Controlled Scope
- In F-OAL, the ViT encoder is 100% frozen (`param.requires_grad = False`). Downstream class-incremental learning depends entirely on the linear separability and semantic richness of the extracted frozen representation $h \in \mathbb{R}^{768}$.
- C2 isolates the backbone checkpoint as the **sole variable factor**, holding all dataset splits, class order, projection initialization, and preprocessing fixed.

---

## 2. Scientific Hypotheses

- **Hypothesis H1 (Representation Quality):** Pretraining on ImageNet-21k ($\approx 14$M images, 21k classes) provides richer texture and semantic features than ImageNet-1k ($\approx 1.28$M images). On DTD, which tests fine-grained texture recognition (e.g. banded, braided, bubbly), ImageNet-21k features are expected to exhibit:
  - Higher intra-class cosine similarity ($\mathcal{S}_{\text{intra}}$) relative to inter-class similarity ($\mathcal{S}_{\text{inter}}$).
  - Higher effective rank of centered covariance ($r_{\text{eff}}$), reducing representation collapse.
  - Reduced forgetting across sequential incremental tasks.

---

## 3. Single-Factor Control Matrix

| Component | B0 (`augreg_in1k`) | B1 (`augreg2_in21k_ft_in1k`) | Control Status |
|:---|:---|:---|:---:|
| **Backbone Identifier** | `vit_base_patch16_224.augreg_in1k` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | **Single Controlled Factor** |
| **Pretrained Dataset** | ImageNet-1k | ImageNet-21k + IN-1k FT | Factor Property |
| **Timm Pretrained Tag** | `augreg_in1k` | `augreg2_in21k_ft_in1k` | Factor Property |
| **Dataset** | DTD (`tanganke/dtd`) | DTD (`tanganke/dtd`) | Fixed 100% |
| **Classes / Tasks** | 40 classes, 10 tasks $\times$ 4 cls | 40 classes, 10 tasks $\times$ 4 cls | Fixed 100% |
| **Samples per Class** | 40 train / 40 test | 40 train / 40 test | Fixed 100% |
| **Preprocessing** | Bicubic, Normalized $[-1, 1]$ | Bicubic, Normalized $[-1, 1]$ | Fixed 100% |
| **Projection Matrix $P$** | `official_uniform`, seed $42$ | `official_uniform`, seed $42$ | **Exact SHA256 Match** |
| **Projection Dim $D$** | $D = 1000$ | $D = 1000$ | Fixed 100% |
| **Stream / Class Order Seeds**| $42$ / $42$ | $42$ / $42$ | Fixed 100% |
| **RLS Classifier** | float64, $\gamma = 1.0$, $B=10$ | float64, $\gamma = 1.0$, $B=10$ | Fixed 100% |

---

## 4. Verification Deliverables

1. **Config Files:** `configs/dtd/c2_b0_augreg_in1k.yaml`, `c2_b1_augreg2_in21k_ft_in1k.yaml`.
2. **Diagnostics Script:** `scripts/run_c2_dtd_diagnostics.py` comparing representation geometry (L2 norms, centered covariance rank, class separation) on matched DTD samples.
3. **Analysis & Plotting:** `scripts/generate_c2_dtd_analysis.py` computing $\Delta_{\text{B1}-\text{B0}}$, B0 Reproducibility Guard check, and emitting summary CSVs/plots.
4. **Notebook & Generator:** `scripts/create_c2_dtd_notebook.py` generating `notebooks/foal_c2_backbone_controlled_dtd.ipynb`.
5. **Unit & Protocol Tests:** `tests/test_c2_dtd.py` verifying config single-factor isolation, projection matrix SHA256 equality, and reproducibility assertions.
