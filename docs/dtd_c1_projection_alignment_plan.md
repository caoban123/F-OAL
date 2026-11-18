# F-OAL — C1 Controlled Smooth Projection Alignment Plan on DTD (Describable Textures Dataset)

> **Status:** `SPECIFICATION_AND_CODE_COMPLETE`  
> **Target Dataset:** Describable Textures Dataset (DTD) via HuggingFace `tanganke/dtd` / `torchvision.datasets.DTD`  
> **Target Branch:** `experiment/dtd-ablation`  
> **Chaining Guarantee:** `C1-P1 == C2-B0` (Identical hyperparameters, dataset split, and random seeds)  

---

## 1. Research Question & Objective

**Research Question:**  
*In Forward-only Online Analytic Learning (F-OAL) on the Describable Textures Dataset (DTD), why does standard Gaussian projection initialization $\mathcal{N}(0, 1.0)$ degrade incremental classification performance compared to the official Kaiming Uniform initialization? Is the degradation driven by:*
1. **Variance scaling** ($\text{Var}(P_{ij}) = 1.0$ vs $\text{Var}(P_{ij}) = \frac{1}{3d} = \frac{1}{3 \cdot 768} \approx 4.34 \times 10^{-4}$), which prevents post-projection Sigmoid saturation, OR
2. **Distributional shape** (Gaussian vs Uniform)?

### Context & Dataset Parameters (DTD)
- **Dataset:** DTD consists of 47 texture classes with 120 images per class (split into 40 train, 40 val, 40 test in the official 3-way split).
- **Paper & Group Protocol:** Table 4 of NeurIPS 2024 uses 40 classes out of 47, split into 10 tasks $\times$ 4 classes.
- **Samples per Class:** Exactly 40 train samples and 40 test samples per class (160 train samples/task, 160 test samples/task; 1,600 train samples and 1,600 test samples across 10 tasks).
- **Backbone in C1:** `vit_base_patch16_224.augreg_in1k` (ImageNet-1k pretrained, 100% frozen, Feature Fusion ON).
- **Preprocessing in C1:** `bicubic` resize to $224 \times 224$, `normalize=true` ($\mu = [0.5, 0.5, 0.5]$, $\sigma = [0.5, 0.5, 0.5]$).

---

## 2. Scientific Hypotheses

- **Hypothesis H2 (Variance Scaling):** When $h \in \mathbb{R}^{768}$ with $\|h\|_2 \approx 20\text{–}30$ is multiplied by $P \sim \mathcal{N}(0, 1.0)$, the pre-activation $u = hP$ has standard deviation $\sigma_u \approx 20\text{–}30$. Passing $u$ through Sigmoid $\sigma(u)$ pushes $>80\%$ of features into saturated tails ($z < 0.01$ or $z > 0.99$), zeroing out gradients and collapsing representation rank.
- **Hypothesis H2-Alt (Distributional Shape):** The bounded support of Uniform initialization $[-\frac{1}{\sqrt{3d}}, +\frac{1}{\sqrt{3d}}]$ prevents extreme outliers compared to Gaussian tails.

### Controlled Configurations
1. **P0 (`gaussian_std1`):** Gaussian $\mathcal{N}(0, 1.0)$
2. **P1 (`official_uniform`):** Kaiming Uniform $\mathcal{U}(-\frac{1}{\sqrt{3d}}, +\frac{1}{\sqrt{3d}})$
3. **P2 (`gaussian_matched_variance`):** Gaussian $\mathcal{N}(0, \frac{1}{3d})$ with matched variance $\sigma^2 = \frac{1}{3 \cdot 768}$

---

## 3. Single-Factor Control Matrix

| Component | P0 (`gaussian_std1`) | P1 (`official_uniform`) | P2 (`gaussian_matched_variance`) | Control Status |
|:---|:---|:---|:---|:---:|
| **Projection Init** | $\mathcal{N}(0, 1.0)$ | $\mathcal{U}(-\frac{1}{\sqrt{3d}}, +\frac{1}{\sqrt{3d}})$ | $\mathcal{N}(0, \frac{1}{3d})$ | **Single Controlled Factor** |
| **Theoretical Variance** | $1.0$ | $\frac{1}{2304} \approx 4.34 \times 10^{-4}$ | $\frac{1}{2304} \approx 4.34 \times 10^{-4}$ | Varied (P0 vs P1/P2) |
| **Dataset** | DTD (`tanganke/dtd`) | DTD (`tanganke/dtd`) | DTD (`tanganke/dtd`) | Fixed 100% |
| **Classes / Tasks** | 40 classes, 10 tasks $\times$ 4 cls | 40 classes, 10 tasks $\times$ 4 cls | 40 classes, 10 tasks $\times$ 4 cls | Fixed 100% |
| **Samples per Class** | 40 train / 40 test | 40 train / 40 test | 40 train / 40 test | Fixed 100% |
| **Backbone** | `vit_base_patch16_224.augreg_in1k` | `vit_base_patch16_224.augreg_in1k` | `vit_base_patch16_224.augreg_in1k` | Fixed 100% |
| **Preprocessing** | Bicubic, Normalized $[-1, 1]$ | Bicubic, Normalized $[-1, 1]$ | Bicubic, Normalized $[-1, 1]$ | Fixed 100% |
| **Projection Seed** | $42$ | $42$ | $42$ | Fixed 100% |
| **Stream / Class Order Seeds**| $42$ / $42$ | $42$ / $42$ | $42$ / $42$ | Fixed 100% |
| **RLS Classifier** | float64, $\gamma = 1.0$, $B=10$ | float64, $\gamma = 1.0$, $B=10$ | float64, $\gamma = 1.0$, $B=10$ | Fixed 100% |

---

## 4. Verification Deliverables

1. **Config Files:** `configs/dtd/c1_p0_gaussian_std1.yaml`, `c1_p1_official_uniform.yaml`, `c1_p2_gaussian_matched_variance.yaml`.
2. **Diagnostics Script:** `scripts/run_c1_dtd_diagnostics.py` measuring pre-sigmoid $u$, post-sigmoid $z$ saturation, and condition number $\text{Cond}(G + \gamma I)$ on matched DTD features.
3. **Analysis & Plotting:** `scripts/generate_c1_dtd_analysis.py` emitting summary CSVs, comparison JSON, and stage accuracy curves.
4. **Notebook & Generator:** `scripts/create_c1_dtd_notebook.py` generating `notebooks/foal_c1_projection_controlled_dtd.ipynb`.
5. **Unit & Protocol Tests:** `tests/test_c1_dtd.py` verifying config integrity, single-factor isolation, and mathematical variance calculations.
