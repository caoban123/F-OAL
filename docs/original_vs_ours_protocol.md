# Protocol & Code-Level Audit: Official F-OAL Repository (`liuyuchen-cz/F-OAL`) vs. Our Implementation (`src/foal`)

* **Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024)
* **Official Author Repository:** [`https://github.com/liuyuchen-cz/F-OAL`](https://github.com/liuyuchen-cz/F-OAL) — Pinned Commit: `73e4ac1b4c965030973a2b77f05c57c2f44f49f5` (`73e4ac1`)
* **Our Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL) — GPU Run Execution Commit: `aad7500c425bd13ece2fd7d4d21014616344ec81` (`aad7500`)
* **Audit Scope:** Phase D & Phase E (`A19` & `A20`) — Line-by-line comparison of all pipeline stages and scientific root-cause analysis of the $-11.77\text{ pp}$ Average Incremental Accuracy gap (`79.33%` vs. `91.10%`).

---

## 1. Executive Summary of Findings

A direct code audit of the official repository (`liuyuchen-cz/F-OAL`, commit `73e4ac1`) against our independent implementation (`src/foal/`, commit `aad7500`) revealed **five confirmed code-level protocol differences** (`H1`, `H2`, `H4`, `H5`, `H6`). None of these differences required modifying or mixing the two codebases to identify; all are verified directly from source files.

Most notably, **Hypothesis H2 (Smooth Projection Initialization Scaling)** is a **confirmed mathematical and implementation mismatch with high expected impact**:
1. **Official Code (`models/pretrained.py`, line 137):** Uses `self.linear_projection = nn.Linear(768, projection_size, bias=False)`. In PyTorch, `nn.Linear(768, 1000, bias=False)` initializes weights from a Kaiming uniform distribution $\mathcal{U}\!\left(-\frac{1}{\sqrt{768}}, +\frac{1}{\sqrt{768}}\right) = \mathcal{U}(-0.03608, +0.03608)$, which has standard deviation $\sigma_P = \frac{1}{\sqrt{3 \times 768}} \approx 0.02083$. Consequently, for a LayerNormed feature vector $h \in \mathbb{R}^{768}$ ($\text{std}(h) \approx 1$), the pre-sigmoid activation $u = h P$ has standard deviation $\text{std}(u) \approx \frac{1}{\sqrt{3}} \approx 0.577$, keeping features inside the **smooth, non-saturated regime** of $\sigma(u) = \frac{1}{1 + e^{-u}}$.
2. **Our Implementation (`src/foal/projection.py`, line 15):** Followed the literal text of Paper Section 3.2 (*"entries sampled from a Gaussian distribution"*) using standard normal initialization $\mathcal{N}(0, 1^2)$ (`gaussian_std = 1.0`). Because the input dimension is $d = 768$, the pre-sigmoid dot product $u = h P$ has standard deviation $\text{std}(u) \approx \sqrt{768} \approx 27.71$ (**48× larger than the official code**), pushing $\sigma(u)$ into hard binary saturation ($\approx 0$ or $\approx 1$) and degrading linear separability in the $1000$-dimensional projection space.

---

## 2. Side-by-Side Pipeline & Protocol Comparison

| Component | Paper Description (NeurIPS 2024) | Official Author Code (`liuyuchen-cz/F-OAL @ 73e4ac1`) | Our Implementation (`caoban123/F-OAL @ aad7500`) | Match Status |
|---|---|---|---|---|
| **Entrypoint** | — | `general_main.py` $\rightarrow$ `experiment/run.py::multiple_run` | `scripts/train.py::main` | Different CLI, equivalent role |
| **Backbone ID** | Pre-trained ViT-B | `timm.create_model('vit_base_patch16_224', pretrained=True)` (`models/pretrained.py:134`) | `timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)` (`src/foal/backbone.py:21`) | **MISMATCH (`H1`)** |
| **ViT Implementation** | 12 Transformer blocks | Custom `models/vit.py::VisionTransformer` copied into repo (`depth=12`, `embed_dim=768`, `qkv_bias=True`) | `timm` native `VisionTransformer` (`depth=12`, `embed_dim=768`) | Equivalent architecture; checkpoint tag depends on `timm` (`H1`) |
| **Feature Extraction & Fusion** | Average features across all 12 blocks + norm | `models/vit.py:263–269`: collects CLS token `x[:, 0]` after each block (`feature_list`), computes `torch.mean(torch.stack(feature_list, dim=0), dim=0)`, then applies `self.norm(x)` (`LayerNorm(768, eps=1e-6)`) | `src/foal/backbone.py:42–58`: collects `x[:, 0]` after each of the 12 blocks, computes `torch.stack(block_cls_features, dim=0).mean(dim=0)`, then applies `self.backbone.norm(fused)` | **EXACT MATCH (`H3` Verified Equivalent)** |
| **Smooth Projection Layer** | Random linear projection + Sigmoid ($D = 1000$) | `models/pretrained.py:137, 146`: `nn.Linear(768, 1000, bias=False)` initialized from $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$ ($\text{std} \approx 0.02083$), followed by `nn.Sigmoid()` | `src/foal/projection.py:15, 25`: `torch.randn(768, 1000) * 1.0` initialized from $\mathcal{N}(0, 1^2)$ ($\text{std} = 1.0$), followed by `torch.sigmoid(x @ self.weight)` | **MAJOR MISMATCH (`H2`)** |
| **Image Resize & Interpolation** | $224 \times 224$ | `utils/setup_elements.py:49–53`: `ToPILImage() -> Resize((224, 224))` (default **Bilinear** interpolation) `-> ToTensor()` | `src/foal/datasets.py:67–71`: `Resize((224, 224), InterpolationMode.BICUBIC) -> ToTensor()` (**Bicubic** interpolation) | **MISMATCH (`H4a`)** |
| **Input Normalization** | Not specified in paper text | `utils/setup_elements.py:49–53`: **No `Normalize` transform**; pixel values fed to ViT in **$[0.0, 1.0]$** range | `src/foal/datasets.py:70`: `Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`; pixel values in **$[-1.0, 1.0]$** range | **MAJOR MISMATCH (`H4b`)** |
| **Regularization $\gamma$ & $R_0$** | $\gamma = 1.0$, $R_0 = I_{1000}$ | `agents/FOAL.py:14`: `self.R = torch.eye(1000).cuda()` ($\gamma = 1.0$ implicit) | `src/foal/rls.py:18`: `torch.eye(1000, dtype=torch.float64) / gamma` with `gamma = 1.0` | **EXACT MATCH** |
| **RLS Update Formula & Solver** | $R_t = R_{t-1} - R_{t-1}H_t^\top(I + H_t R_{t-1} H_t^\top)^{-1} H_t R_{t-1}$ | `agents/FOAL.py:37–39`: Explicit `torch.pinverse(torch.eye(batch) + H @ R @ H.T)` in `float32` on GPU | `src/foal/rls.py:47–51`: `torch.linalg.solve(I_b + H @ K, K.T)` in `float64` + symmetry enforcement `0.5 * (R + R.T)` | **NUMERICAL VARIANT (`H5`)** |
| **Classifier Weight Matrix $W$** | $W \in \mathbb{R}^{1000 \times C}$ | `models/pretrained.py:138` & `agents/FOAL.py:15, 38–39`: Fixed shape `(1000, 100)` (`n_classes=100` from Task 1); unseen class columns stay `0` | `src/foal/rls.py:25–40`: Dynamically expanded `(1000, 10 * t)` for seen classes `0 .. 10t - 1` | **STRUCTURAL VARIANT (`H6a`)** |
| **Class Order & Random Seeds** | 10 tasks $\times$ 10 classes | `general_main.py:32` & `continuum/continuum.py:28`: `--seed 0`; for `run_id = r`, `np.random.seed(seed + r)` calls `np.random.permutation(100)` | `configs/cifar100.yaml`: `seed: 42`; `np.random.default_rng(42).permutation(100)` | **SEED / RNG MISMATCH (`H6b`)** |
| **DataLoader Options** | Batch size 10, one-pass | `agents/FOAL.py:22` & `continuum/data_utils.py:130`: `batch_size=10`, `shuffle=True`, `drop_last=True` (for both train and test; $5000 \bmod 10 = 0, 1000 \bmod 10 = 0$) | `src/foal/datasets.py:157`: `batch_size=10`, `shuffle=True` (train via generator), `shuffle=False` (test), `drop_last=False` | Minor (`5000` and `1000` are divisible by `10`, so 0 samples dropped) |
| **Forgetting Rate Formula** | Average forgetting after final task | `experiment/metrics.py:49–54`: Averages $\max_k A_{k,j} - A_{T,j}$ over **all $T = 10$ tasks** (including Task 10 where forgetting is $0$) | `src/foal/metrics.py:39–55`: Averages $\max_{k < T} A_{k,j} - A_{T,j}$ over **$T - 1 = 9$ past tasks** | **METRIC FORMULA MISMATCH (`H6c`)** |

---

## 3. Detailed Hypothesis Analysis (`H1`–`H6`)

### H1 — Pretrained ViT Checkpoint & `timm` Weight Tag
* **Status:** **CONFIRMED DIFFERENCE IN CODE**
* **Official Code Evidence (`models/pretrained.py:134–136`):**
  ```python
  vit_b_16 = timm.create_model('vit_base_patch16_224', pretrained=True)
  self.model = VisionTransformer(num_classes=1000)
  self.model.load_state_dict(vit_b_16.state_dict())
  ```
  In `timm==0.9.7` (pinned in the official `requirements.txt`) as well as `timm==1.0.30`, calling `timm.create_model('vit_base_patch16_224', pretrained=True)` without an explicit tag resolves to the default tag `.augreg2_in21k_ft_in1k` (or `.augreg_in21k_ft_in1k`), which is **pre-trained on ImageNet-21K and fine-tuned on ImageNet-1K**.
* **Our Code Evidence (`src/foal/backbone.py:21` & `configs/cifar100.yaml:13`):**
  Explicitly specifies `vit_base_patch16_224.augreg_in1k` (trained on ImageNet-1K only).
* **Expected Impact:** **Medium–High** ($+2\text{ pp}$ to $+5\text{ pp}$ on CIFAR-100 feature quality, as ImageNet-21K pre-training provides richer mid-level representations across the 12 transformer blocks).
* **How to Verify:** Run Cell 5 of `notebooks/foal_official_cifar100_reproduction.ipynb` to print `pretrained_cfg['hf_hub_id']` and `pretrained_cfg['tag']`, and run a 2-task pilot comparing `vit_base_patch16_224` vs `vit_base_patch16_224.augreg_in1k`.

---

### H2 — Smooth Projection Weight Distribution & Scaling
* **Status:** **CONFIRMED MAJOR MISMATCH IN CODE**
* **Official Code Evidence (`models/pretrained.py:137, 146`):**
  ```python
  self.linear_projection = nn.Linear(768, projection_size, bias=False)
  self.activation = nn.Sigmoid()
  ...
  new_activation = self.activation(self.linear_projection(feature_map))
  ```
  `nn.Linear(768, 1000, bias=False)` uses `init.kaiming_uniform_(self.weight, a=math.sqrt(5))`, which draws weights from $\mathcal{U}\!\left(-\frac{1}{\sqrt{768}}, +\frac{1}{\sqrt{768}}\right) = \mathcal{U}(-0.03608, +0.03608)$ with $\text{std}(P_{ij}) \approx 0.02083$.
* **Our Code Evidence (`src/foal/projection.py:15`):**
  ```python
  weight = torch.randn(input_dim, output_dim, generator=generator) * gaussian_std
  ```
  with `gaussian_std: float = 1.0`.
* **Mathematical Consequence:**
  For a normalized feature vector $h \in \mathbb{R}^{768}$ with unit variance per coordinate ($\mathbb{E}[h_k^2] \approx 1$), the pre-sigmoid dot product $u_j = \sum_{k=1}^{768} h_k P_{kj}$ has variance:
  $$\text{Var}(u_j) = 768 \cdot \text{Var}(P_{kj})$$
  * In **Official Code**: $\text{Var}(u_j) \approx 768 \times \frac{1}{3 \times 768} = \frac{1}{3} \implies \text{std}(u_j) \approx 0.577$. Over $95\%$ of activations lie within $[-1.15, +1.15]$, where $\sigma(u)$ is smooth and informative ($\sigma(u) \in [0.24, 0.76]$).
  * In **Our Code (`aad7500`)**: $\text{Var}(u_j) \approx 768 \times 1.0 = 768 \implies \text{std}(u_j) \approx 27.71$. Over $90\%$ of activations satisfy $|u_j| > 3.5$, saturating $\sigma(u_j)$ to $\approx 0$ or $\approx 1$ (acting like a hard random threshold/sign function rather than a smooth projection).
* **Expected Impact:** **Very High** (likely the single largest contributor to the $-11.77\text{ pp}$ gap).
* **How to Verify:** Run a controlled experiment on CIFAR-100 setting `gaussian_std = 1.0 / math.sqrt(768)` ($\approx 0.03608$) or `nn.Linear(768, 1000, bias=False)` initialization while keeping all other settings fixed.

---

### H3 — Feature Fusion Layer Indexing & LayerNorm Placement
* **Status:** **CONFIRMED IDENTICAL (RULED OUT AS ROOT CAUSE)**
* **Official Code Evidence (`models/vit.py:258–269`):**
  ```python
  feature_list = []
  x = self.patch_embed(x)
  cls_token = self.cls_token.expand(x.shape[0], -1, -1)
  x = torch.cat((cls_token, x), dim=1)
  x = self.pos_drop(x + self.pos_embed)
  for blk in self.blocks:
      x = blk(x)
      feature_list.append(x[:, 0])
  x = torch.mean(torch.stack(feature_list, dim=0), dim=0)
  x = self.norm(x)
  ```
* **Our Code Evidence (`src/foal/backbone.py:42–58`):**
  Extracts `x[:, 0]` after each of the 12 blocks in `self.backbone.blocks`, averages them via `torch.stack(block_cls_features, dim=0).mean(dim=0)`, and applies `self.backbone.norm(fused)` once at the end.
* **Expected Impact:** **None ($0.00\text{ pp}$)** — the mathematical operations are identical.

---

### H4 — Image Preprocessing, Resize Interpolation & Normalization
* **Status:** **CONFIRMED MISMATCH IN CODE**
* **Official Code Evidence (`utils/setup_elements.py:49–53`):**
  ```python
  'cifar100': transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])
  ```
  Uses PIL default bilinear interpolation and leaves tensors in $[0.0, 1.0]$ (no mean/std subtraction).
* **Our Code Evidence (`src/foal/datasets.py:67–71`):**
  Uses `InterpolationMode.BICUBIC` and `transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))` (scaling inputs to $[-1.0, 1.0]$).
* **Expected Impact:** **Low–Medium** ($0.5\text{ pp}$ to $2.5\text{ pp}$). Because `augreg` ViT models were trained with `mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)`, our normalization matches `timm`'s model card, whereas the official repo feeds unnormalized $[0, 1]$ tensors into ViT.
* **How to Verify:** Run a controlled 2-task / 10-task ablation toggling `mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]` + `bilinear` vs `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]` + `bicubic`.

---

### H5 — RLS Numerical Solver (`torch.pinverse` `float32` vs `torch.linalg.solve` `float64`)
* **Status:** **CONFIRMED IMPLEMENTATION VARIANT**
* **Official Code Evidence (`agents/FOAL.py:37–39`):**
  ```python
  self.R = self.R - self.R @ new_activation.T @ torch.pinverse(
      torch.eye(y.size(0)).cuda() + new_activation @ self.R @ new_activation.T
  ) @ new_activation @ self.R
  self.W = self.W + self.R @ new_activation.T @ (label_onehot - new_activation @ self.W)
  self.model.fc.weight = torch.nn.Parameter(self.W.t())
  ```
  Computed in `float32` on GPU using `torch.pinverse`.
* **Our Code Evidence (`src/foal/rls.py:47–51`):**
  Computed in `float64` using `torch.linalg.solve` and symmetrized at every step (`self.R = 0.5 * (R_next + R_next.T)`).
* **Expected Impact:** **Negligible on accuracy** ($< 0.1\text{ pp}$ when $H$ is well-conditioned), though `float32` on GPU is slightly faster than `float64` on T4 GPUs (where FP64 throughput is $1/32$ of FP32).

---

### H6 — Evaluation Protocol, Class Order RNG, and Forgetting Definition
* **Status:** **CONFIRMED MISMATCH IN METRIC DEFINITION & RNG**
* **Sub-finding H6a (Fixed 100-column $W$ vs Dynamic $W$):**
  In the official code, `self.model.fc` has 100 output units from Task 1 (`W` shape `1000 x 100`), with columns for unseen classes remaining identically `0.0`. Because targets are one-hot in $\{0, 1\}$, seen-class logits are predominantly positive, but if any test sample produces negative logits for all seen classes at stage $k < 10$, an unseen class with logit `0.0` will win `torch.max(logits, 1)` and be counted as an error. In our code, `W` only contains columns for the $10k$ seen classes at stage $k$. At Task 10 ($k = 10$), both have 100 columns.
* **Sub-finding H6b (Class Order RNG):**
  Official code uses `np.random.seed(0); np.random.permutation(100)` (legacy NumPy MT19937 generator with seed `0`). Our run used `np.random.default_rng(42).permutation(100)` (PCG64 generator with seed `42`).
* **Sub-finding H6c (10-Task vs 9-Task Forgetting Formula):**
  In `experiment/metrics.py:49–54`, the official code computes:
  ```python
  best_acc = np.max(end_task_acc_arr, axis=1)
  final_forgets = best_acc - end_acc
  avg_fgt = np.mean(final_forgets, axis=1)
  ```
  Because `end_task_acc_arr` has `10` columns (Tasks 1..10), `np.mean(final_forgets, axis=1)` divides the sum of task forgetting by **$T = 10$** (including Task 10, whose forgetting is identically $0$), whereas standard Continual Learning literature (and our `src/foal/metrics.py`) divides by **$T - 1 = 9$** past tasks!
  * Specifically, on our Seed 42 run, our 9-task forgetting is **$9.21\%$**, which corresponds to **$9.21\% \times \frac{9}{10} = 8.29\%$** under the official repository's 10-task formula (`0.92 pp` of the apparent forgetting gap is purely a denominator difference: $1/10$ vs $1/9$).

---

## 4. Prioritized Controlled Experiment Plan (Low Compute Cost)

To isolate and quantify each confirmed mismatch without modifying the baseline history (`aad7500`), execute the following sequence on Colab GPU:

| Step | Experiment ID | Change Relative to Previous Step | Compute Cost | Target Hypothesis |
|---:|---|---|---|---|
| **1** | `exp0_official_repo` | Run unmodified `liuyuchen-cz/F-OAL` (`73e4ac1`) via `notebooks/foal_official_cifar100_reproduction.ipynb` (`seed=0` and `seed=42`) | ~5–8 min on T4 | Establish exact Official Repo baseline on same T4 hardware |
| **2** | `exp1_proj_scale` | In our code, set Smooth Projection weight init to $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$ (or $\sigma = 1/\sqrt{3 \times 768} \approx 0.02083$) | ~28 min (full 10 tasks) or ~3 min (2-task pilot) | Measure exact $\Delta \bar{A}$ from fixing Smooth Projection saturation (`H2`) |
| **3** | `exp2_backbone_tag` | Switch backbone from `vit_base_patch16_224.augreg_in1k` to `vit_base_patch16_224` (default `augreg2_in21k_ft_in1k`) | ~28 min (full) / ~3 min (pilot) | Measure exact $\Delta \bar{A}$ from ImageNet-21K pre-training (`H1`) |
| **4** | `exp3_input_norm` | Compare `[0, 1]` bilinear transform vs `[-1, 1]` bicubic transform | ~3 min (pilot) | Measure sensitivity to input normalization (`H4`) |
| **5** | `exp4_seed_order` | Evaluate exact class order from `np.random.seed(0); np.random.permutation(100)` | ~28 min (full) | Separate class-order variance from algorithmic differences (`H6b`) |
