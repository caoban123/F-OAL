# Kế hoạch Thực nghiệm & Báo cáo Kết quả Tái hiện — F-OAL (NeurIPS 2024)

Tài liệu này quy định giao thức thực nghiệm, phạm vi tái hiện (**Core Scope** đối chiếu với **Full Scope**), bảng đối chiếu chuẩn 3 nguồn (`PAPER_REPORTED`, `ORIGINAL_REPRODUCED`, `OUR_REPRODUCED`) và kết quả thực nghiệm của nhóm cho bài báo:  
**F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning (NeurIPS 2024).**

* **Repository nhóm:** `https://github.com/caoban123/F-OAL`
* **Repository gốc của tác giả:** `https://github.com/liuyuchen-cz/F-OAL`
* **Tài liệu kiểm toán kỹ thuật:** [`docs/FOAL_AUDIT_AND_FIX_PLAN.md`](FOAL_AUDIT_AND_FIX_PLAN.md)

---

## 1. Phạm vi Thực nghiệm: Full Scope vs. Core Scope (Checkpoint 3 Compliance)

### 1.1. Phạm vi Đầy đủ của Bài báo (Full Paper Scope)
Trong bài báo gốc, tác giả thực hiện:
* **6 bộ dữ liệu:** CIFAR-100, CORe50, FGVCAircraft, DTD, Tiny-ImageNet, Country211.
* **Hơn 14 phương pháp so sánh (Table 2):** ER, iCaRL, GDumb, DER++, Co2L, DVC, OCM, OnPro, SCR, MEMO, EASE, SLCA, ACIL, DS-AL, F-OAL (chạy 3 lần cho mỗi phương pháp).
* **Đánh giá tài nguyên (Table 3 & Figure 2):** Thời gian huấn luyện (giây) và mức tiêu thụ VRAM đỉnh (GB).
* **Ablation Study (Table 4 & Section 5.3):** 4 cấu hình kết hợp Feature Fusion (FF) & Smooth Projection (SP); đối chiếu Analytic Classifier (RLS) với Fully Connected Classifier huấn luyện bằng Backpropagation.
* **Additional Analysis:** Khảo sát ảnh hưởng của kích thước mini-batch ($B$), số chiều chiếu ($D$) và hệ số chính quy hóa ($\gamma$).

### 1.2. Đề xuất Phạm vi Cốt lõi của Nhóm (Proposed Core Scope — Chờ Nhóm/GV Xác nhận)
Do giới hạn tài nguyên tính toán trên Google Colab / Kaggle GPU miễn phí (mỗi lượt chạy CIFAR-100 10 tasks trên NVIDIA T4 tốn $\approx 29$ phút), nhóm đề xuất **Core Scope** tập trung vào trọng tâm khoa học của bài báo với đầy đủ **3 trụ cột thực nghiệm**:

1. **Experiment 1 — Headline Comparison & Resource Profiling (Table 2, Table 3, Figure 2 trên CIFAR-100):**
   * Thiết lập: CIFAR-100 chia thành **10 tasks $\times$ 10 classes**, streaming **one-pass** (1 epoch duy nhất, zero-replay), `batch_size = 10`.
   * Đối chiếu 3 nguồn của **F-OAL**:
     * `PAPER_REPORTED`: Kết quả công bố trong Table 2 & Table 3 của bài báo.
     * `ORIGINAL_REPRODUCED` (**A19**): Kết quả chạy trực tiếp repository gốc của tác giả (`liuyuchen-cz/F-OAL`) trên cùng môi trường GPU của nhóm.
     * `OUR_REPRODUCED` (**A10**, **A12**): Kết quả chạy mã nguồn tự hiện thực của nhóm (`src/foal/`) trên 3 random seeds `[42, 123, 2026]` ($Mean \pm Std$).
   * Đối chiếu với các Baseline đại diện trên cùng Frozen ViT backbone và cùng chia tách task:
     * **Baseline 1 (`A15`):** Frozen ViT + Feature Fusion + Smooth Projection + **Linear/FC Classifier huấn luyện bằng Backpropagation (SGD/Adam, 1-pass)** (tái hiện thí nghiệm Section 5.3 của paper).
     * **Baseline 2 (`A16`):** Frozen ViT + **Linear Ridge Analytic Classifier** (không dùng Smooth Projection phi tuyến).
     * **Baseline 3 (`A17`):** Frozen ViT + **Experience Replay (ER)**.
   * Đo đạc đồng thời: **Average Incremental Accuracy ($\bar{A}$)**, **Last Accuracy ($A_{10}$)**, **Forgetting Rate ($F_{10}$)**, **Training Time (giây)** và **Peak GPU Memory (GB)** (**A18**).

2. **Experiment 2 — Ablation Study (Table 4 trên CIFAR-100 & tùy chọn DTD):**
   * Chạy đủ 4 tổ hợp qua `scripts/ablation.py` (**A11**):
     1. **Full F-OAL:** `feature_fusion = True`, `smooth_projection = True`
     2. **w/o Feature Fusion:** `feature_fusion = False` (chỉ lấy CLS token của block thứ 12), `smooth_projection = True`
     3. **w/o Smooth Projection:** `feature_fusion = True`, `smooth_projection = False` ($D = 768$ đưa thẳng vào RLS)
     4. **w/o Both:** `feature_fusion = False`, `smooth_projection = False`

3. **Experiment 3 — Additional & Sensitivity Analysis (Kiểm chứng Lý thuyết & Độ nhạy Siêu tham số):**
   * **Analysis 3A (Batch-Size Invariance):** Chạy F-OAL với $B \in \{5, 10, 25, 50, 100\}$ trên cùng seed để kiểm chứng thực nghiệm tính chất toán học của cập nhật Woodbury RLS: nghiệm giải tích cuối mỗi task không phụ thuộc vào kích thước mini-batch $B$, trong khi thời gian thực thi thay đổi theo $B$.
   * **Analysis 3B (Projection Scaling, Dimension & Regularization Sensitivity):** Khảo sát ảnh hưởng của hệ số tỷ lệ khởi tạo ma trận chiếu ($\sigma_{\text{init}} = 1.0$ vs $1/\sqrt{768}$), số chiều chiếu $D \in \{500, 1000, 2000\}$ và hệ số Ridge regularization $\gamma \in \{0.1, 1.0, 10.0\}$.

---

## 2. Công thức Đánh giá & Quy ước Độ lệch (Metrics & Deviation Formulas)

Giả sử quá trình học tăng dần gồm $T = 10$ giai đoạn (tasks). Gọi $a_{k, j} \in [0, 1]$ là độ chính xác kiểm thử trên tập test của Task $j$ sau khi hoàn tất huấn luyện Task $k$ ($1 \le j \le k \le T$). Ma trận kết quả $R \in \mathbb{R}^{T \times T}$ là ma trận tam giác dưới với $R_{k, j} = a_{k, j}$.

1. **Stage Accuracy tại giai đoạn $k$ ($A_k$):**
   $$A_k = \frac{1}{k} \sum_{j=1}^{k} a_{k, j}$$
2. **Average Incremental Accuracy ($\bar{A}$ hoặc $A_{\text{avg}}$, $\uparrow$ cao hơn là tốt hơn):**
   $$\bar{A} = \frac{1}{T} \sum_{k=1}^{T} A_k$$
3. **Last Task Accuracy ($A_T$ hoặc $A_{\text{last}}$, $\uparrow$ cao hơn là tốt hơn):**
   $$A_T = \frac{1}{T} \sum_{j=1}^{T} a_{T, j}$$
4. **Average Forgetting Rate ($F_T$ hoặc $F$, $\downarrow$ thấp hơn là tốt hơn):**
   $$f_j^T = \max_{\ell \in \{j, \dots, T-1\}} a_{\ell, j} - a_{T, j}, \qquad F_T = \frac{1}{T-1} \sum_{j=1}^{T-1} f_j^T$$
5. **Chênh lệch tuyệt đối ($\Delta_{\text{pp}}$, tính bằng điểm phần trăm — percentage points):**
   $$\Delta_{\text{pp}} = \text{Ours} - \text{Paper}$$
6. **Độ lệch tương đối ($\Delta_{\text{rel}}\%$):**
   $$\Delta_{\text{rel}}(\%) = \frac{\text{Ours} - \text{Paper}}{\text{Paper}} \times 100\%$$
   *(Lưu ý: Đối với $\bar{A}$ và $A_T$, $\Delta < 0$ nghĩa là kết quả tái hiện thấp hơn paper. Đối với Forgetting $F_T$, vì giá trị càng nhỏ càng tốt nên $\Delta > 0$ nghĩa là mô hình bị quên nhiều hơn so với paper).*

---

## 3. Lệnh Thực thi Chuẩn (Reproducible Commands)

```bash
# 1. Chạy kiểm thử đơn vị (33 unit tests trên CPU)
python -m pytest -v

# 2. Chạy Pilot Smoke Test trên CPU (2 tasks, 100 mẫu/task — KHÔNG phải full reproduction)
python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100 --run-name smoke_test_post_fix

# 3. Chạy Full Single-Seed Reproduction trên GPU (10 tasks x 10 classes, 50.000 train, 10.000 test)
python -m scripts.train --config configs/cifar100.yaml --device cuda --seed 42 --wandb --run-name foal_full_cifar100_seed42

# 4. Đánh giá độc lập Checkpoint cuối
python -m scripts.evaluate --checkpoint results/foal_full_cifar100_seed42/task_10.pt --device cuda

# 5. Chạy Multi-Seed (3 seeds: 42, 123, 2026) và tổng hợp Mean +/- Std
python -m scripts.run_seeds --seeds 42 123 2026 --config configs/cifar100.yaml --device cuda --wandb --run-name-prefix foal_cifar100 --skip-completed

# 6. Chạy 4 cấu hình Ablation Study trên GPU
python -m scripts.ablation --config configs/cifar100.yaml --device cuda --wandb

# 7. Tổng hợp toàn bộ kết quả hợp lệ trong thư mục results/
python -m scripts.summarize_runs --results-dir results
```

---

## 4. Kết quả Thực nghiệm Đã Ghi nhận (Recorded Experimental Results)

### 4.1. Kết quả Pilot Smoke Test trên CPU (Kiểm chứng Hạ tầng — 2 Tasks, 100 mẫu/task)
* **Mục đích:** Xác minh luồng dữ liệu thật, trích xuất đặc trưng ViT + Feature Fusion + LayerNorm, cập nhật Woodbury RLS và nạp lại checkpoint trên máy cục bộ (CPU Intel Core Ultra 5 125H). **Không dùng để so sánh với Table 2 của bài báo.**
* **Hiện vật lưu trữ:** `results/smoke_test_post_fix/` (`VERIFIED` trên đĩa cục bộ).
* **Kết quả (`seed = 42`):**
  * Stage 1 Accuracy (Task 1): $78.00\%$
  * Stage 2 Accuracy (Task 1: $69.00\%$, Task 2: $85.00\%$): $77.00\%$
  * $\bar{A} = 77.50\%$, $A_2 = 77.00\%$, $F_2 = 9.00\%$
  * Kiểm chứng `scripts/evaluate.py --checkpoint results/smoke_test_post_fix/task_02.pt`: Khớp chính xác $77.00\%$ (`all_finite = True`).

---

### 4.2. Kết quả Single-Seed Full Reproduction trên GPU (P1.2 — CIFAR-100, 10 Tasks, Seed 42)

> **Ghi chú Sửa lỗi Tài liệu & Trạng thái Bằng chứng (01/11/2026):**
> 1. Các số liệu dưới đây được khôi phục chính xác từ [`results/foal_full_cifar100_seed42/metrics.json`](../results/foal_full_cifar100_seed42/metrics.json), khắc phục lỗi mất chữ số đầu do nội suy biến `$` của PowerShell trong commit `c74b0de`.
> 2. **Trạng thái hiện vật (`VERIFIED` trên đĩa cục bộ & Git):** Gói kết quả [`foal_cifar100_single_seed42_results.zip`](../foal_cifar100_single_seed42_results.zip) ($109.8\text{ MB}$) đã được giải nén và kiểm chứng đầy đủ:
>    * Các tệp kết quả nhẹ ([`metrics.json`](../results/foal_full_cifar100_seed42/metrics.json), [`metadata.json`](../results/foal_full_cifar100_seed42/metadata.json), [`task_accuracy.csv`](../results/foal_full_cifar100_seed42/task_accuracy.csv), [`summary_runs.csv`](../results/summary_runs.csv), [`summary_aggregated.csv`](../results/summary_aggregated.csv)) được lưu trực tiếp trên Git.
>    * 10 tệp checkpoint nhị phân `task_01.pt` .. `task_10.pt` ($\approx 11.5\text{ MB}$/tệp; đã kiểm chứng `task_10.pt` có $\|R - R^\top\|_{\max} = 0.0$, `all_finite = True`) và gói W&B offline `wandb/offline-run-20261101_084136-2cdeiptv/run-2cdeiptv.wandb` ($188\text{ KB}$) được lưu trên đĩa cục bộ (loại khỏi Git để tránh giới hạn $100\text{ MB}$/tệp của GitHub).

#### A. Cấu hình Thực thi Chính xác (`configs/cifar100.yaml`)
* **Thiết bị:** Google Colab GPU — NVIDIA T4 (`device = cuda`).
* **Giao thức:** CIFAR-100, 10 tasks $\times$ 10 classes, `batch_size = 10`, one-pass (50.000 mẫu train, 10.000 mẫu test, không giới hạn số mẫu).
* **Seeds:** `seed = 42` (`class_order_seed = 42`, `projection_seed = 42`, `stream_seed = 42`).
* **Kiến trúc & Siêu tham số:**
  * Backbone: `vit_base_patch16_224.augreg_in1k` (đóng băng 100%, resize bicubic $224 \times 224$, `mean = (0.5, 0.5, 0.5)`, `std = (0.5, 0.5, 0.5)`).
  * Feature Fusion: Trung bình cộng CLS token của 12 transformer blocks + `network.norm` (final LayerNorm).
  * Smooth Projection: Ma trận Gaussian cố định $W_{\text{proj}} \in \mathbb{R}^{768 \times 1000}$ (`projection_dim = 1000`, `gaussian_std = 1.0`) + hàm kích hoạt Sigmoid $\sigma(x W_{\text{proj}})$.
  * Analytic Classifier (Online Ridge RLS): Cập nhật Woodbury trên `rls_dtype = float64` với hệ số chính quy hóa L2 **`gamma = 1.0`** ($R_0 = \gamma^{-1} I = I$). *(Đính chính: không có tham số `lambda = 0.1` trong mã nguồn).*
* **Thời gian huấn luyện (`total_duration_seconds`):** **$1735.3\text{ giây}$ ($\approx 28.92\text{ phút}$)**.

#### B. Bảng So sánh Chuẩn 3 Nguồn trên CIFAR-100 (Headline Comparison Table)

| Phương pháp & Nguồn Kết quả | Loại Nguồn | Số Seeds | $\bar{A}$ ($A_{\text{avg}}$, $\uparrow$) | $A_{10}$ ($A_{\text{last}}$, $\uparrow$) | $F_{10}$ ($F$, $\downarrow$) | $\Delta_{\text{pp}}(\bar{A})$ | $\Delta_{\text{rel}}(\bar{A})$ | $\Delta_{\text{pp}}(A_{10})$ | $\Delta_{\text{rel}}(A_{10})$ | $\Delta_{\text{pp}}(F_{10})$ | $\Delta_{\text{rel}}(F_{10})$ | Thời gian (s) | Peak VRAM (GB) | W&B Run Link & Trạng thái Xác minh |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **F-OAL (Paper Table 2 & 3)** | `PAPER_REPORTED` | 3 runs | **91.10%** | **86.50%** | **5.50%** | $0.00\text{ pp}$ | $0.00\%$ | $0.00\text{ pp}$ | $0.00\%$ | $0.00\text{ pp}$ | $0.00\%$ | $261\text{ s}$ (RTX 4070 Ti) | $1.9\text{ GB}$ | Số liệu công bố trong bài báo NeurIPS 2024. |
| **F-OAL (Author Repo `liuyuchen-cz/F-OAL @ 73e4ac1`)** | `ORIGINAL_REPRODUCED` | `PENDING` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` — Đã chuẩn bị [`notebooks/foal_official_cifar100_reproduction.ipynb`](../notebooks/foal_official_cifar100_reproduction.ipynb) (**A19**). |
| **F-OAL (Ours — Colab T4, code commit `aad7500`)** | `OUR_REPRODUCED` *(1-seed v1)* | `n=1, std=N/A` (`seed=42`) | **79.33%** (`n=1, std=N/A`) | **70.92%** (`n=1, std=N/A`) | **9.21%** (`n=1, std=N/A`) | **$-11.77\text{ pp}$** | **$-12.92\%$** | **$-15.58\text{ pp}$** | **$-18.01\%$** | **$+3.71\text{ pp}$** | **$+67.45\%$** | $1735.3\text{ s}$ (Colab T4) | `NOT LOGGED` | `offline-run-2cdeiptv` (`PENDING_USER_AUTH` để `wandb sync`); hiện vật gốc tại `results/foal_full_cifar100_seed42/` (`VERIFIED`). |
| **F-OAL (Ours — Sau điều tra A20 & 3 Seeds)** | `OUR_REPRODUCED` *(3-seed)* | 3 (`42, 123, 2026`) | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` — Sẽ thực thi tại Stage D (**A12**, **A21**). |

#### C. Ma trận Độ chính xác $10 \times 10$ Đầy đủ (`OUR_REPRODUCED`, Seed 42)

| Giai đoạn (Sau Task $k$) | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 | **Stage Accuracy ($A_k$)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | **89.60%** | — | — | — | — | — | — | — | — | — | **89.60%** |
| **Stage 2** | 84.80% | **91.20%** | — | — | — | — | — | — | — | — | **88.00%** |
| **Stage 3** | 82.10% | 88.50% | **84.60%** | — | — | — | — | — | — | — | **85.07%** |
| **Stage 4** | 78.60% | 87.40% | 78.90% | **81.50%** | — | — | — | — | — | — | **81.60%** |
| **Stage 5** | 76.60% | 84.40% | 76.60% | 80.10% | **77.80%** | — | — | — | — | — | **79.10%** |
| **Stage 6** | 73.10% | 83.70% | 76.70% | 78.80% | 73.90% | **75.50%** | — | — | — | — | **76.95%** |
| **Stage 7** | 70.60% | 82.10% | 75.70% | 77.70% | 73.20% | 73.60% | **77.50%** | — | — | — | **75.77%** |
| **Stage 8** | 67.40% | 80.90% | 75.50% | 74.30% | 72.70% | 71.70% | 75.90% | **73.70%** | — | — | **74.01%** |
| **Stage 9** | 64.70% | 78.60% | 74.80% | 73.80% | 71.30% | 71.10% | 75.60% | 71.60% | **69.30%** | — | **72.31%** |
| **Stage 10 (Final)** | 60.50% | 76.20% | 74.00% | 73.50% | 70.50% | 69.50% | 74.20% | 71.00% | 68.40% | **71.40%** | **70.92%** ($A_{10}$) |
| **Task Forgetting ($f_j^{10}$)** | **29.10%** | **15.00%** | **10.60%** | **8.00%** | **7.30%** | **6.00%** | **3.30%** | **2.70%** | **0.90%** | — | **Mean $F_{10} = 9.21\%$** |

#### D. Nhận định Học thuật & Các Giả thuyết Cần Kiểm chứng về Khoảng cách Kết quả (`79.33%` vs `91.10%`)
* **Tính đúng đắn kỹ thuật đã được xác nhận:** Quá trình chạy đã hoàn tất toàn bộ 10 tasks (50.000 mẫu train, 10.000 mẫu test) mà không phát sinh lỗi `NaN`/`Inf`, đồng thời ma trận $10 \times 10$ thỏa mãn chính xác các đẳng thức của $\bar{A} = 79.33\%$, $A_{10} = 70.92\%$ và $F_{10} = 9.21\%$.
* **Tại sao khoảng cách $-11.77\text{ pp}$ trên $\bar{A}$ và $-15.58\text{ pp}$ trên $A_{10}$ cần được điều tra sâu hơn ngoài thứ tự lớp (`A04`):**
  * Về mặt toán học, tại thời điểm kết thúc Task 10 (khi mô hình đã đi qua toàn bộ 50.000 mẫu huấn luyện của 100 lớp), nghiệm giải tích của bài toán Ridge Regression $W_{10}^* = (\gamma I + H^\top H)^{-1} H^\top Y$ **hoàn toàn bất biến với thứ tự chia lớp (`class_order`) hay thứ tự trộn mẫu (`stream_seed`)**!
  * Do đó, việc độ chính xác cuối cùng $A_{10}$ của nhóm đạt $70.92\%$ so với $86.50\%$ của bài báo ($\Delta_{\text{pp}} = -15.58\text{ pp}$) chứng minh rằng nguyên nhân cốt lõi **nằm ở biểu diễn đặc trưng đưa vào bộ phân loại RLS**, bao gồm các giả thuyết chính sẽ được kiểm chứng trong **A19** và **A20**:
    1. **Giả thuyết H1 (Pretrained ViT Checkpoint — `A01`):** Tác giả có thể sử dụng trọng số ViT-B/16 được pre-train trên ImageNet-21K rồi fine-tune trên ImageNet-1K (`vit_base_patch16_224.augreg_in21k_ft_in1k`) thay vì chỉ `augreg_in1k`.
    2. **Giả thuyết H2 (Sự bão hòa Sigmoid trong Smooth Projection — `A03`):** Trong `src/foal/projection.py`, ma trận chiếu được khởi tạo $W_{\text{proj}} \sim \mathcal{N}(0, 1)^{768 \times 1000}$ (`gaussian_std = 1.0`). Khi nhân với vector đặc trưng $768$ chiều sau LayerNorm, tích vô hướng trước Sigmoid có độ lệch chuẩn $\approx \sqrt{768} \approx 27.7$, làm hàm $\sigma(\cdot)$ bão hòa về $0$ hoặc $1$. Trong code gốc của tác giả, phép chiếu có thể được khởi tạo với độ lệch chuẩn $1/\sqrt{768}$ (như mặc định của `nn.Linear`) hoặc chuẩn hóa L2 vector đặc trưng trước khi chiếu.
    3. **Giả thuyết H3–H5 (Vị trí LayerNorm / Chuẩn hóa L2 trong Feature Fusion, Preprocessing, hoặc quy ước $\gamma$ trong RLS):** Sẽ được đối chiếu từng dòng lệnh với repo gốc `liuyuchen-cz/F-OAL` tại Stage B & Stage C.

---

## 5. Khung Bảng Kết quả cho Ablation Study & Additional Analysis (Sẽ điền tại Stage E)

### 5.1. Bảng Ablation Study (Đối chiếu Table 4 — Feature Fusion & Smooth Projection)

| Cấu hình Ablation | Feature Fusion (FF) | Smooth Projection (SP) | Paper DTD $\bar{A}$ (Tham chiếu Table 4) | Ours CIFAR-100 $\bar{A}$ ($Mean \pm Std$) | Ours CIFAR-100 $A_{10}$ ($Mean \pm Std$) | Ours CIFAR-100 $F_{10}$ ($Mean \pm Std$) | W&B Run Link |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1. Full F-OAL** | Có (`True`) | Có (`True`, $D=1000$) | $82.80\%$ | `79.33%` *(1-seed v1)* / `PENDING 3-seed` | `70.92%` *(1-seed v1)* | `9.21%` *(1-seed v1)* | `NOT YET SYNCED` |
| **2. w/o Feature Fusion** | Không (`False`, last block) | Có (`True`, $D=1000$) | $80.50\%$ | `NOT RUN` *(Stage E)* | `NOT RUN` | `NOT RUN` | `NOT YET SYNCED` |
| **3. w/o Smooth Projection** | Có (`True`) | Không (`False`, $D=768$) | $79.30\%$ | `NOT RUN` *(Stage E)* | `NOT RUN` | `NOT RUN` | `NOT YET SYNCED` |
| **4. w/o Both** | Không (`False`) | Không (`False`, $D=768$) | $71.20\%$ | `NOT RUN` *(Stage E)* | `NOT RUN` | `NOT RUN` | `NOT YET SYNCED` |

### 5.2. Bảng Additional Analysis 3A — Kiểm chứng Tính Bất biến theo Batch Size ($B$)

| Batch Size ($B$) | Số bước cập nhật RLS / Task | $\bar{A}$ ($A_{\text{avg}}$) | $A_{10}$ ($A_{\text{last}}$) | Sai lệch $\|W_{10}^{(B)} - W_{10}^{(10)}\|_F$ | Thời gian chạy (s) | Peak VRAM (GB) | W&B Run Link |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| $B = 5$ | $1000$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |
| $B = 10$ *(Mặc định)* | $500$ | **79.33%** *(v1)* | **70.92%** *(v1)* | $0.0$ (Tham chiếu) | $1735.3\text{ s}$ | `PENDING` | `NOT YET SYNCED` |
| $B = 25$ | $200$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |
| $B = 50$ | $100$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |
| $B = 100$ | $50$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |
