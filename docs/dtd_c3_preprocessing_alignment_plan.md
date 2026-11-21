# F-OAL — C3 Controlled Input Preprocessing Alignment Plan on DTD ($2 \times 2$ Factorial Design)

> **Status:** `SPECIFICATION_AND_CODE_COMPLETE`  
> **Target Dataset:** Describable Textures Dataset (DTD) via HuggingFace `tanganke/dtd` / `torchvision.datasets.DTD`  
> **Target Branch:** `experiment/dtd-ablation`  
> **Chaining Guarantees:**  
> - `C3-A == C2-B1` (Zero-compute reuse via `C3_A_REUSE_APPROVED`)  
> - `C3-D == Table 4 A0` (`configs/dtd/dtd_a0_full.yaml` hyperparameter equivalence)  
> - `C3-D == Official Author Transform` (Bitwise equivalence verified by `scripts/verify_official_transform_dtd.py`)  

---

## 1. Research Question & Objective

**Research Question:**  
*In Forward-only Online Analytic Learning (F-OAL) on the Describable Textures Dataset (DTD), what is the quantitative impact of input image preprocessing on representation extraction and incremental classification accuracy across a complete $2 \times 2$ factorial design:*
1. **Factor 1 (Interpolation):** Bicubic vs Bilinear
2. **Factor 2 (Normalization):** Normalized $[-1, 1]$ vs Raw $[0, 1]$
*And what are the respective main effects ($\Delta_{\text{Interp}}, \Delta_{\text{Norm}}$) and interaction effect ($\Delta_{\text{Interact}}$)?*

### Context & Non-Square Image Nuance in DTD
- CIFAR-100 images are uniform $32 \times 32$ squares.
- In contrast, DTD images possess varied original resolutions and aspect ratios (e.g. $300 \times 300$, $640 \times 480$). Resizing to $224 \times 224$ involves aspect-ratio distortion and spatial interpolation across anisotropic dimensions.
- The official author repository (`utils/setup_elements.py:49-53`) uses:
  ```python
  transforms.Resize((224, 224))  # Bilinear default in torchvision.transforms.Resize
  transforms.ToTensor()          # Raw [0, 1] range, no Normalize step
  ```
- Evaluating both factors independently via a $2 \times 2$ design isolates the effect of high-frequency spatial filtering from input dynamic range scaling.

---

## 2. $2 \times 2$ Factorial Design Matrix

Fixed across all 4 conditions:
- **Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21k + 1k FT, frozen, Feature Fusion ON)
- **Smooth Projection:** $P \in \mathbb{R}^{768 \times 1000}$, `official_uniform`, seed $42$ (identical SHA256)
- **Dataset:** DTD (`tanganke/dtd`), 40 classes, 10 tasks $\times$ 4 cls, 40 train / 40 test per class, seed $42$
- **Classifier:** float64 Woodbury RLS ($\gamma = 1.0$), $B = 10$, one-pass streaming

| Condition | Interpolation (Factor 1) | Normalization (Factor 2) | Pixel Range | Scientific Role |
|:---|:---:|:---:|:---:|:---|
| **C3-A** | **Bicubic** | **Normalize** (`mean=0.5, std=0.5`) | $[-1.0, 1.0]$ | Baseline control (Identity with `C2-B1`) |
| **C3-B** | **Bilinear** | **Normalize** (`mean=0.5, std=0.5`) | $[-1.0, 1.0]$ | Isolates Interpolation in Normalized space |
| **C3-C** | **Bicubic** | **Raw Tensor** (No Normalization) | $[0.0, 1.0]$ | Isolates Normalization in Bicubic space |
| **C3-D** | **Bilinear** | **Raw Tensor** (No Normalization) | $[0.0, 1.0]$ | Official Author Alignment & Table 4 A0 parity |

### Factorial Contrasts:
- **Main Effect of Interpolation:**
  $$\Delta_{\text{Interp}} = \frac{1}{2} \left[ (\bar{A}_{\text{C3-B}} - \bar{A}_{\text{C3-A}}) + (\bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-C}}) \right]$$
- **Main Effect of Normalization:**
  $$\Delta_{\text{Norm}} = \frac{1}{2} \left[ (\bar{A}_{\text{C3-C}} - \bar{A}_{\text{C3-A}}) + (\bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-B}}) \right]$$
- **Interaction Effect:**
  $$\Delta_{\text{Interact}} = (\bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-C}}) - (\bar{A}_{\text{C3-B}} - \bar{A}_{\text{C3-A}})$$
- **Total Alignment Effect:**
  $$\Delta_{\text{Total}} = \bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-A}} = \Delta_{\text{Interp}} + \Delta_{\text{Norm}}$$

---

## 3. Verification Deliverables

1. **Config Files:** `configs/dtd/c3_a_bicubic_normalized.yaml`, `c3_b_bilinear_normalized.yaml`, `c3_c_bicubic_raw.yaml`, `c3_d_bilinear_raw.yaml`.
2. **Transform Deep Verification:** `scripts/verify_official_transform_dtd.py` proving bitwise equivalence between C3-D and Official transform on 100 real DTD images and synthetic edge cases.
3. **Diagnostics Script:** `scripts/run_c3_dtd_diagnostics.py` computing tensor stats, backbone feature diagnostics, and condition cosine similarities.
4. **Analysis & Plotting:** `scripts/generate_c3_dtd_analysis.py` calculating factorial contrasts and outputting comparison tables, CSVs, and plots.
5. **Notebook & Generator:** `scripts/create_c3_dtd_notebook.py` generating `notebooks/foal_c3_preprocessing_controlled_dtd.ipynb`.
6. **Unit & Protocol Tests:** `tests/test_c3_dtd.py` testing transform parity, factorial math identities, and config properties.
