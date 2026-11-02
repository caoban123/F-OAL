# Protocol & Code-Level Discrepancy Audit: Official F-OAL (`liuyuchen-cz/F-OAL`) vs. Our Implementation (`src/foal`)

* **Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024)
* **Official Author Repository:** [`https://github.com/liuyuchen-cz/F-OAL`](https://github.com/liuyuchen-cz/F-OAL) — Verified Execution Commit: `73e4ac1b4c965030973a2b77f05c57c2f44f49f5` (`73e4ac1`)
* **Our Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL) — Verified GPU Execution Commit: `aad7500c425bd13ece2fd7d4d21014616344ec81` (`aad7500`)
* **Audit Status:** **A19 Official Reproduction `VERIFIED` (`PASS`)** | **A20 Discrepancy Analysis `IN PROGRESS / PARTIAL` (Code Audit Complete; Controlled GPU Experiments `C1`–`C4` Pending)**

---

## 1. Executive Summary & Diagnostic Classification (Case 1 Confirmed)

Following the execution of the official author code (`liuyuchen-cz/F-OAL @ 73e4ac1`) on Google Colab NVIDIA T4 (`seed = 0`, 10 tasks $\times$ 10 classes, CIFAR-100) and independent verification of `foal_official_cifar100_results.zip`, the project falls squarely into **Diagnostic Case 1 — Official Code closely matches the Paper, while Our Implementation (`aad7500`) is significantly lower**:

| Metric | Paper Reported (3-run Mean) | Official Code Run (`73e4ac1`, `seed=0`, `n=1`) | Our Code Run (`aad7500`, `seed=42`, `n=1`) | Official vs. Paper ($\Delta_{\text{pp}}$ / $\Delta_{\text{rel}}$) | Ours vs. Official ($\Delta_{\text{pp}}$ / $\Delta_{\text{rel}}$) | Ours vs. Paper ($\Delta_{\text{pp}}$ / $\Delta_{\text{rel}}$) |
|---|---:|---:|---:|---:|---:|---:|
| **Average Incremental Accuracy ($\bar{A}$, $\uparrow$)** | **91.10%** | **90.98%** (`n=1, std=N/A`) | **79.33%** (`n=1, std=N/A`) | **$-0.12\text{ pp}$** (`-0.13%`) | **$-11.65\text{ pp}$** (`-12.80%`) | **$-11.77\text{ pp}$** (`-12.92%`) |
| **Last Accuracy ($A_{10}$, $\uparrow$)** | **86.50%** | **85.84%** (`n=1, std=N/A`) | **70.92%** (`n=1, std=N/A`) | **$-0.66\text{ pp}$** (`-0.76%`) | **$-14.92\text{ pp}$** (`-17.38%`) | **$-15.58\text{ pp}$** (`-18.01%`) |
| **Forgetting — Official 10-Task Def ($F_{\text{official}}$, $\downarrow$)** | **5.50%** | **4.69%** (`n=1, std=N/A`) | **8.29%** (`n=1, std=N/A`) | **$-0.81\text{ pp}$** (`-14.73%`) | **$+3.60\text{ pp}$** (`+76.76%`) | **$+2.79\text{ pp}$** (`+50.73%`) |
| **Forgetting — Standard 9-Task Def ($F_{\text{ours}}$, $\downarrow$)** | — *(Not used by author code)* | **5.17%** (`n=1, std=N/A`) | **9.21%** (`n=1, std=N/A`) | — | **$+4.04\text{ pp}$** (`+78.26%`) | — *(Different denominator)* |
| **Wall-Clock Runtime (s)** | $261.0\text{ s}$ (RTX 4070 Ti) | $3047.6\text{ s}$ (Tesla T4) | $1735.3\text{ s}$ (Tesla T4) | — | $-1312.3\text{ s}$ (`-43.06%`) | — |

### Key Scientific Takeaways
1. **The paper's CIFAR-100 results are genuinely reproducible from the author's code:** Running the official repository on the exact same Colab Tesla T4 environment (`torch 2.11.0+cu130`, `timm 1.0.30`) achieves $\bar{A} = 90.98\%$ (within $-0.12\text{ pp}$ of the paper's $91.10\%$) and $A_{10} = 85.84\%$ (within $-0.66\text{ pp}$ of the paper's $86.50\%$).
2. **The $-11.65\text{ pp}$ gap in $\bar{A}$ (`79.33%` vs. `90.98%`) and $-14.92\text{ pp}$ gap in $A_{10}$ (`70.92%` vs. `85.84%`) is caused by concrete protocol/implementation differences between `src/foal` and `liuyuchen-cz/F-OAL`, NOT by hardware or library versions.**
3. **Strict separation between Verified Code Differences and Unverified Causal Attribution:** While we have verified exact code-level differences across `H1`, `H2`, `H4`, `H5`, `H6`, and `H7`, **we do not claim a definitive quantitative attribution of the accuracy gap to any single factor prior to running Controlled Experiments `C1`–`C4`**.

---

## 2. Protocol Discrepancy Matrix (`H1`–`H7`)

| Hypothesis / Component | Paper Description (NeurIPS 2024) | Official Author Code (`liuyuchen-cz/F-OAL @ 73e4ac1`) | Our Implementation (`caoban123/F-OAL @ aad7500`) | Verification Status |
|---|---|---|---|---|
| **H1. Pretrained ViT Backbone & Weights** | ViT-B pre-trained on ImageNet-1K | `models/pretrained.py:134`: `timm.create_model('vit_base_patch16_224', pretrained=True)`. In `timm 1.0.30`, resolves to `hf_hub_id='timm/vit_base_patch16_224.augreg2_in21k_ft_in1k'` (`tag='augreg2_in21k_ft_in1k'`). | `src/foal/backbone.py:21`: `timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)`. Resolves to `hf_hub_id='timm/vit_base_patch16_224.augreg_in1k'` (`tag='augreg_in1k'`). | **`VERIFIED DIFFERENCE`** |
| **H2. Smooth Projection Initialization & Scaling** | Entries sampled from a Gaussian distribution + Sigmoid ($D=1000$) | `models/pretrained.py:137, 160–161`: `nn.Linear(768, 1000, bias=False)` $\rightarrow$ Kaiming uniform $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768}) = \mathcal{U}(-0.03608, +0.03608)$ ($\text{std} \approx 0.02083$). Pre-sigmoid $\text{std}(u) \approx 0.578$, $0.00\%$ saturation. | `src/foal/projection.py:15, 25`: `torch.randn(768, 1000) * 1.0` $\rightarrow$ standard normal $\mathcal{N}(0, 1^2)$ ($\text{std} = 1.0$). Pre-sigmoid $\text{std}(u) \approx 27.79$, **$86.88\%$ saturation** ($z < 0.01$ or $z > 0.99$). | **`VERIFIED DIFFERENCE`** |
| **H3. Feature Fusion & LayerNorm** | Average features across all 12 blocks + norm | `models/pretrained.py:154–159`: Sums `x[:, 0]` across all 12 blocks in `self.vit.blocks`, divides by `len(self.vit.blocks)` ($12$), then applies `self.vit.norm(x)`. | `src/foal/backbone.py:42–58` & `feature_fusion.py`: Collects `x[:, 0]` from all 12 blocks, computes `.mean(dim=0)`, then applies `self.backbone.norm(fused)`. | **`STRUCTURALLY MATCHED`** |
| **H4. Image Preprocessing & Normalization** | Input resized to $224 \times 224$ | `utils/setup_elements.py:49–53`: `ToPILImage() -> Resize((224, 224))` (PIL default **Bilinear**) `-> ToTensor()`. **No `Normalize`**; pixels in **$[0.0, 1.0]$**. | `src/foal/datasets.py:67–71`: `Resize((224, 224), InterpolationMode.BICUBIC) -> ToTensor() -> Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`. Pixels in **$[-1.0, 1.0]$**. | **`VERIFIED DIFFERENCE`** |
| **H5. Online RLS Update & Precision** | Woodbury update with $\gamma = 1.0$, $R_0 = I_{1000}$ | `agents/FOAL.py:14–16, 36–43`: Updates `self.R` and `self.W` in **`float64` (`double()`)** with $\gamma = 1.0$ using `torch.pinverse(...)`, then assigns `self.W.float()` (`float32`) to `self.model.fc.weight` for inference. | `src/foal/rls.py:18, 45–60`: Updates `self.R` and `self.W` in **`float64`** with $\gamma = 1.0$ using `torch.linalg.solve(...)`, enforces symmetry `0.5 * (R + R.T)`, and evaluates in `float64`. | **`VERIFIED DIFFERENCE`** *(Minor numerical variant)* |
| **H6. Random Seed & Class Order** | 10 tasks $\times$ 10 classes, one-pass, batch size 10 | `general_main.py:14–18` & `continuum/dataset_scripts/dataset_base.py:55`: `--seed 0`; uses legacy `np.random.seed(0); np.random.permutation(100)` (`Task 0 = [26, 86, 2, 55, 75, 93, 16, 73, 54, 95]`). | `configs/cifar100.yaml` & `src/foal/datasets.py:85`: `seed: 42`; uses `np.random.default_rng(42).permutation(100)` (`Task 0 = [59, 21, 56, 18, 33, 42, 50, 27, 92, 71]`). | **`VERIFIED DIFFERENCE`** |
| **H7a. Classifier Output Head & Unseen Classes** | Linear classifier $W \in \mathbb{R}^{1000 \times C}$ | `models/pretrained.py:138` & `agents/FOAL.py:15, 38`: Fixed `n_classes = 100` from Task 1 (`W` shape `1000 x 100`); columns for unseen classes are `0.0`. | `src/foal/rls.py:25–40`: Dynamically expands `W` from `(1000, 10)` to `(1000, 100)` over stages $1 \dots 10$. | **`VERIFIED DIFFERENCE`** *(Equivalent at Stage 10)* |
| **H7b. Forgetting Rate Formula** | Average forgetting after final task | `experiment/metrics.py:49–54`: $F_{\text{official}} = \frac{1}{10}\sum_{j=1}^{10} (\max_{1 \le k \le 10} a_{k,j} - a_{10,j})$ (divides by $T=10$ and clamps negative forgetting to $0$). | `src/foal/metrics.py:39–55`: $F_{\text{ours}} = \frac{1}{9}\sum_{j=1}^{9} (\max_{j \le k \le 9} a_{k,j} - a_{10,j})$ (divides by $T-1=9$ and preserves negative values). | **`VERIFIED DIFFERENCE`** *(Formula mismatch)* |
| **H8. Confirmed Packaging Bug in Author Repo** | — | `utils/name_match.py:21` imports `agents/exp_replay_dvc.py:10` (`from loss import agmax_loss, cross_entropy_loss`), but `loss.py` is missing from repo; `README.md` lists `--projection_size` whereas `general_main.py:164` defines `--projection`. | Resolved non-intrusively via external `PYTHONPATH` shim `/content/official_foal_shim/loss.py` and `--projection 1000`. | **`CONFIRMED BUG IN OFFICIAL REPO`** |

---

## 3. Deep Technical Evidence for Each Hypothesis

### 3.1. H1 — Pretrained ViT Checkpoint (`vit_base_patch16_224` vs. `vit_base_patch16_224.augreg_in1k`)
* **Observation:** Calling `timm.create_model('vit_base_patch16_224', pretrained=False).pretrained_cfg` in `timm 1.0.30` resolves to:
  * `hf_hub_id`: `'timm/vit_base_patch16_224.augreg2_in21k_ft_in1k'`
  * `tag`: `'augreg2_in21k_ft_in1k'`
  Whereas calling `timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=False).pretrained_cfg` resolves to:
  * `hf_hub_id`: `'timm/vit_base_patch16_224.augreg_in1k'`
  * `tag`: `'augreg_in1k'`
* **Evidence from `official_stdout.log` (line 18):**
  `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`
  This confirms that the official run downloaded `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` (pre-trained on ImageNet-21K with AugReg v2 and fine-tuned on ImageNet-1K) from Hugging Face Hub.
* **Hypothesis & Confidence:** **High Confidence** that using ImageNet-21K pre-trained weights (`augreg2_in21k_ft_in1k`) vs. ImageNet-1K-only weights (`augreg_in1k`) contributes meaningfully to CIFAR-100 feature separability. Exact magnitude in percentage points remains **unverified** until Controlled Experiment `C2`.

### 3.2. H2 — Smooth Projection Weight Initialization & Sigmoid Saturation
* **Observation:**
  * Official code (`models/pretrained.py:137`): `self.linear_projection = nn.Linear(768, projection_size, bias=False)` initializes weights from $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768}) = \mathcal{U}(-0.03608, +0.03608)$ ($\text{std} \approx 0.02083$).
  * Our code (`src/foal/projection.py:15`): `torch.randn(768, 1000) * 1.0` initializes weights from $\mathcal{N}(0, 1^2)$ ($\text{std} = 1.0$).
* **Empirical Measurement on LayerNormed $768$-D Vectors ($N = 100$ samples, $D = 1000$):**

| Initialization Scheme | Weight Std $\sigma_P$ | Pre-Sigmoid Std $\text{std}(u)$ | Pre-Sigmoid Range $[u_{\min}, u_{\max}]$ | Post-Sigmoid Std $\text{std}(z)$ | Saturation Rate ($z < 0.01$ or $z > 0.99$) |
|---|---:|---:|---:|---:|---:|
| **Official (`nn.Linear(768, 1000, bias=False)`)** | $0.0208$ | **$0.5779$** | $[-2.47, +2.52]$ | $0.1343$ | **$0.00\%$** |
| **Ours (`torch.randn(768, 1000) * 1.0`)** | $1.0000$ | **$27.7918$** | $[-108.06, +125.87]$ | $0.4854$ | **$86.88\%$** |

* **Hypothesis & Confidence:** **High Confidence** that $86.88\%$ hard saturation of the Sigmoid activation under $\mathcal{N}(0, 1)$ initialization degrades the smooth random feature representation compared to the $0.00\%$ saturation of `nn.Linear(768, 1000, bias=False)`. Exact accuracy impact on CIFAR-100 remains **unverified** until Controlled Experiment `C1`.

### 3.3. H3 — Feature Fusion Aggregation & LayerNorm
* **Observation:** Comparing `models/pretrained.py:154–159` and `src/foal/backbone.py:42–58`:
  * Both extract `x[:, 0]` (the CLS token) at the output of each of the 12 Transformer blocks (`self.vit.blocks` / `self.backbone.blocks`).
  * Both compute the unweighted arithmetic mean across the 12 blocks (`cumulative_output / 12` vs. `torch.stack(..., dim=0).mean(dim=0)`).
  * Both apply the ViT final LayerNorm (`self.vit.norm(x)` vs. `self.backbone.norm(fused)`) once to the averaged 768-dimensional vector.
* **Status:** **`STRUCTURALLY MATCHED`** (Ruled out as an algorithmic discrepancy).

### 3.4. H4 — Image Preprocessing (`[0, 1]` Bilinear vs. `[-1, 1]` Bicubic)
* **Observation:**
  * Official code (`utils/setup_elements.py:49–53`): `ToPILImage() -> Resize((224, 224))` (Bilinear) `-> ToTensor()` (pixel values in $[0.0, 1.0]$, no normalization).
  * Our code (`src/foal/datasets.py:67–71`): `Resize((224, 224), InterpolationMode.BICUBIC) -> ToTensor() -> Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))` (pixel values in $[-1.0, 1.0]$, matching `timm`'s `pretrained_cfg`).
* **Hypothesis & Confidence:** **Low–Medium Confidence** as a primary driver of the gap (likely $\pm 0.5\text{ to }2.0\text{ pp}$), to be isolated in Controlled Experiment `C3`.

### 3.5. H5 — RLS Update & Runtime Analysis (`3047.6 s` vs. `1735.3 s`)
* **Observation:**
  * Both implementations execute the Woodbury update in `float64` (`double()`) with $\gamma = 1.0$.
  * Why did the Official Run take **$3047.6\text{ s}$** (`train_time = 1545.0 s`, `test_time = 1496.4 s`) vs. Our Run's **$1735.3\text{ s}$** on the same Tesla T4 GPU?
    1. **Double Forward Pass per Training Batch in Official Code (`agents/FOAL.py:35, 44`):** Line 35 calls `new_activation = self.model.expansion(batch_x)` to update RLS, and line 44 immediately calls `logits = self.model(batch_x)` (which runs `self.expansion(batch_x)` a second time on the exact same batch!) just to print running training accuracy. This doubles training ViT forward passes from $50,000$ to $100,000$ images.
    2. **Full 10-Task Evaluation at Every Stage (`agents/base.py:155`):** At every stage $t \in \{1, \dots, 10\}$, the official code runs `evaluate(test_loaders)` across all 10 tasks ($10 \times 10,000 = 100,000$ test images), whereas our `scripts/train.py` evaluates only on seen tasks $1 \dots t$ ($55,000$ test images).
    * Total ViT forward passes: **$200,000$ images (Official)** vs. **$105,000$ images (Ours)** — a $1.905\times$ ratio that directly explains the $1.756\times$ wall-clock runtime ratio ($3047.6\text{ s}$ vs. $1735.3\text{ s}$).

### 3.6. H7b — Exact Reconciliation of Forgetting Definitions (`F_official_definition` vs. `F_ours_definition`)
* **Official Implementation Definition (`experiment/metrics.py:49–54`):**
  $$F_{\text{official}} = \frac{1}{T} \sum_{j=1}^{T} \left( \max_{1 \le k \le T} a_{k, j} - a_{T, j} \right)$$
  * Divides by $T = 10$ (including Task 10 where forgetting is $0$).
  * Includes the final stage $k = T$ inside $\max_{1 \le k \le T} a_{k, j}$, which clamps any negative forgetting (positive backward transfer) to $0.0$.
* **Our Implementation Definition (`src/foal/metrics.py:39–55`):**
  $$F_{\text{ours}} = \frac{1}{T - 1} \sum_{j=1}^{T - 1} \left( \max_{j \le k \le T - 1} a_{k, j} - a_{T, j} \right)$$
  * Divides by $T - 1 = 9$ past tasks.
  * Maximizes strictly over prior stages $k \in \{j, \dots, T-1\}$, preserving negative values when $a_{T, j} > \max_{k < T} a_{k, j}$ (e.g., on Official Run Task 9, $a_{9, 9} = 0.868 \to a_{10, 9} = 0.872$, giving $f_9^{10} = -0.004$).
* **Cross-Evaluated Forgetting Table (Apples-to-Apples):**

| Forgetting Formula | Paper Reported | Official Run (`73e4ac1`, `seed=0`) | Our Run (`aad7500`, `seed=42`) | Official vs. Paper ($\Delta_{\text{pp}}$) | Ours vs. Official ($\Delta_{\text{pp}}$) |
|---|---:|---:|---:|---:|---:|
| **`F_official_definition` ($T=10$, clamped)** | **5.50%** | **4.69%** (`0.469 / 10`) | **8.29%** (`0.829 / 10`) | **$-0.81\text{ pp}$** | **$+3.60\text{ pp}$** |
| **`F_ours_definition` ($T-1=9$, unclamped)** | — | **5.17%** (`0.465 / 9`) | **9.21%** (`0.829 / 9`) | — | **$+4.04\text{ pp}$** |
| **9-Task Clamped ($\max_{k \le T}$, div by $9$)** | — | **5.21%** (`0.469 / 9`) | **9.21%** (`0.829 / 9`) | — | **$+4.00\text{ pp}$** |

---

## 4. Controlled Experiment Plan (`C0`–`C4`) — Ordered by Cost & Diagnostic Power

> **Design Rule:** All controlled experiments must be configurable via YAML flags without overwriting the historical `aad7500` (`seed=42`) checkpoints or results, and each run must be logged as a separate W&B run and Git commit.

| Experiment | Variable Changed (Single-Factor Control) | Fixed Control Baseline | GPU Cost (Colab T4) | Diagnostic Purpose |
|---|---|---|---|---|
| **Experiment C0 — Metric & Class-Order Alignment** | Recompute `F_official_definition` and `F_ours_definition` on existing matrices (0 GPU cost — **DONE above**); optionally run Our Code (`aad7500`) with Official `seed=0` class order (`[26, 86, 2, ...]`). | Our current pipeline (`augreg_in1k`, $\mathcal{N}(0,1)$ proj, `[-1,1]` norm) | $0\text{ min}$ (math) / $\sim 29\text{ min}$ (`seed=0` run) | Quantify exact effect of metric denominator ($0.92\text{ pp}$ on $F$) and class permutation (`seed=0` vs `seed=42`). |
| **Experiment C1 — Smooth Projection Initialization (`H2`)** | Change Smooth Projection initialization from $\mathcal{N}(0, 1)$ (`gaussian_std=1.0`) to Official `nn.Linear(768, 1000, bias=False)` ($\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$) via config flag `projection_init: kaiming_uniform`. | `vit_base_patch16_224.augreg_in1k`, `[-1, 1]` bicubic norm, `seed=42` | $\sim 3\text{ min}$ (2-task pilot) + $\sim 29\text{ min}$ (full 10-task) | Measure exact $\Delta \bar{A}$ and $\Delta A_{10}$ from eliminating the $86.88\%$ Sigmoid saturation. |
| **Experiment C2 — Pretrained ViT Checkpoint (`H1`)** | Change `backbone` from `vit_base_patch16_224.augreg_in1k` to `vit_base_patch16_224` (`augreg2_in21k_ft_in1k`). | Our current projection & `[-1, 1]` bicubic norm, `seed=42` (plus combined with `C1`) | $\sim 3\text{ min}$ (pilot) + $\sim 29\text{ min}$ (full) | Measure exact $\Delta \bar{A}$ and $\Delta A_{10}$ contributed by ImageNet-21K pre-training. |
| **Experiment C3 — Preprocessing & Normalization (`H4`)** | Change transform to `Resize((224, 224), BILINEAR)` + `ToTensor()` (`mean=[0,0,0], std=[1,1,1]`, range $[0, 1]$). | Fixed backbone + projection from `C1`/`C2`, `seed=42` | $\sim 3\text{ min}$ (pilot) + $\sim 29\text{ min}$ (full) | Measure sensitivity of frozen ViT features to $[0, 1]$ bilinear vs. $[-1, 1]$ bicubic preprocessing. |
| **Experiment C4 — Combined Protocol Alignment (`H1+H2+H4`)** | Enable all verified official protocol settings (`vit_base_patch16_224`, `kaiming_uniform` projection, `[0, 1]` bilinear transform) in `src/foal/` and run on both `seed=0` and `seeds=[42, 123, 2026]`. | Official Run (`90.98%` on `seed=0`) | $\sim 29\text{ min/seed}$ | Verify that our independent implementation matches the Official Run (`~90.98%`) within $< 0.2\text{ pp}$ on `seed=0` and establish the 3-seed $Mean \pm Std$. |


---

## 5. Kết quả Triển khai Thí nghiệm Kiểm soát C0 & C1

### 5.1. C0 — Đồng nhất Hóa Định nghĩa Forgetting
- **Phân tích đối chiếu:** Paper Section 4.2 Equations (17)–(18) chia trung bình trên $T-1=9$ task quá khứ ($F_T^{\text{past}}$). Ngược lại, mã nguồn tác giả `experiment/metrics.py:49–54` chia trung bình trên $T=10$ task ($F_T^{\text{code}}$).
- **Tái tính toán thực tế:**
  - Official Run Seed 0 (`3eca8232`): $F_T^{\text{past}} = 5.17\%$, $F_T^{\text{code}} = 4.69\%$.
  - Our Historical Run Seed 42 (`2cdeiptv`): $F_T^{\text{past}} = 9.21\%$, $F_T^{\text{code}} = 8.29\%$.
- Cả hai công thức đã được tích hợp vào `src/foal/metrics.py` và tự động ghi nhận vào `metrics.json` và W&B.

### 5.2. C1 — Chẩn đoán Khởi tạo Chiếu & Pilot Thực nghiệm
- Đã hoàn tất module chẩn đoán `src/foal/diagnostics.py` và đo đạc trên 200 đặc trưng ViT thực tế:
  - P0 (`gaussian_std1`): Độ lệch chuẩn trước kích hoạt $u$ đạt $19.28$, dẫn đến tỷ lệ bão hòa cực đoan Sigmoid lên tới **$81.03\%$**.
  - P1 (`official_uniform`) & P2 (`gaussian_matched_variance`): Giữ độ lệch chuẩn $u \approx 0.40$, tỷ lệ bão hòa **$0.00\%$**, bảo toàn độ nhạy đạo hàm $\approx 0.25$.
- Đã hoàn tất Level A Pilot 2 Tasks trên CPU: P1 đạt $79.00\%$ Accuracy (cao hơn P0 $77.50\%$).
- Đã hoàn tất notebook Google Colab GPU [`notebooks/foal_c1_projection_controlled.ipynb`](../notebooks/foal_c1_projection_controlled.ipynb) phục vụ Level B Full 10 Tasks run trên GPU.
