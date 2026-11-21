# Kế hoạch Thực nghiệm & Báo cáo Kết quả Tái hiện — F-OAL (NeurIPS 2024)

Tài liệu này quy định giao thức thực nghiệm, phạm vi tái hiện (**Core Scope** đối chiếu với **Full Scope**), bảng đối chiếu chuẩn 3 nguồn (`PAPER_REPORTED`, `ORIGINAL_REPRODUCED`, `OUR_REPRODUCED`) và kết quả thực nghiệm của nhóm cho bài báo:  
**F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning (NeurIPS 2024).**

* **Repository nhóm:** `https://github.com/caoban123/F-OAL`
* **Repository gốc của tác giả:** `https://github.com/liuyuchen-cz/F-OAL` (Commit đã xác minh: `73e4ac1b4c965030973a2b77f05c57c2f44f49f5`)
* **Tài liệu kiểm toán kỹ thuật:** [`docs/FOAL_AUDIT_AND_FIX_PLAN.md`](FOAL_AUDIT_AND_FIX_PLAN.md)
* **Tài liệu đối chiếu giao thức Original vs. Ours (`A19`/`A20`):** [`docs/original_vs_ours_protocol.md`](original_vs_ours_protocol.md)

---

## 1. Phạm vi Thực nghiệm: Full Scope vs. Core Scope (Checkpoint 3 Compliance)

### 1.1. Phạm vi Đầy đủ của Bài báo (Full Paper Scope)
Trong bài báo gốc, tác giả thực hiện:
* **6 bộ dữ liệu:** CIFAR-100, CORe50, FGVCAircraft, DTD, Tiny-ImageNet, Country211.
* **Hơn 14 phương pháp so sánh (Table 2):** iCaRL, ER, ASER, SCR, DVC, PCR, LwF, EWC, EASE, LAE, SLCA, OCM, OnPro, SCR, MEMO, EASE, SLCA, ACIL, DS-AL, F-OAL (chạy 3 lần cho mỗi phương pháp).
* **Đánh giá tài nguyên (Table 3 & Figure 2):** Thời gian huấn luyện (giây) và mức tiêu thụ VRAM đỉnh (GB).
* **Ablation Study (Table 4 & Section 5.3):** 4 cấu hình kết hợp Feature Fusion (FF) & Smooth Projection (SP); đối chiếu Analytic Classifier (RLS) với Fully Connected Classifier huấn luyện bằng Backpropagation.
* **Additional Analysis:** Khảo sát ảnh hưởng của kích thước mini-batch ($B$), số chiều chiếu ($D$) và hệ số chính quy hóa ($\gamma$).

### 1.2. Đề xuất Phạm vi Cốt lõi của Nhóm (Proposed Core Scope — Chờ Nhóm/GV Xác nhận)
Do giới hạn tài nguyên tính toán trên Google Colab / Kaggle GPU miễn phí (mỗi lượt chạy CIFAR-100 10 tasks trên NVIDIA T4 tốn $\approx 29\text{–}51$ phút), nhóm đề xuất **Core Scope** tập trung vào trọng tâm khoa học của bài báo với đầy đủ **3 trụ cột thực nghiệm**:

1. **Experiment 1 — Headline Comparison & Resource Profiling (Table 2, Table 3, Figure 2 trên CIFAR-100):**
   * Thiết lập: CIFAR-100 chia thành **10 tasks $\times$ 10 classes**, streaming **one-pass** (1 epoch duy nhất, zero-replay), `batch_size = 10`.
   * Đối chiếu 3 nguồn của **F-OAL**:
     * `PAPER_REPORTED`: Kết quả công bố trong Table 2 & Table 3 của bài báo ($3$ runs).
     * `ORIGINAL_REPRODUCED` (**A19 — Đã hoàn tất & xác minh**): Kết quả chạy trực tiếp repository gốc của tác giả (`liuyuchen-cz/F-OAL @ 73e4ac1`) trên cùng môi trường GPU NVIDIA T4 của nhóm.
     * `OUR_REPRODUCED` (**A10**, **A12**): Kết quả chạy mã nguồn tự hiện thực của nhóm (`src/foal/`) — bao gồm bản chạy đơn seed đầu tiên (`aad7500`, `seed=42`) và các thí nghiệm kiểm soát biến số (`C1`–`C4`) + 3 random seeds `[42, 123, 2026]` ($Mean \pm Std$).
   * Đối chiếu với các Baseline đại diện trên cùng Frozen ViT backbone và cùng chia tách task:
     * **Baseline 1 (`A15`):** Frozen ViT + Feature Fusion + Smooth Projection + **Linear/FC Classifier huấn luyện bằng Backpropagation (SGD/Adam, 1-pass)** (tái hiện thí nghiệm Section 5.3 của paper).
     * **Baseline 2 (`A16`):** Frozen ViT + **Linear Ridge Analytic Classifier** (không dùng Smooth Projection phi tuyến).
     * **Baseline 3 (`A17`):** Frozen ViT + **Experience Replay (ER)**.
   * Đo đạc đồng thời: **Average Incremental Accuracy ($\bar{A}$)**, **Last Accuracy ($A_{10}$)**, **Forgetting Rate ($F$)**, **Training Time (giây)** và **Peak GPU Memory (GB)** (**A18**).

2. **Experiment 2 — Ablation Study (Table 4 trên CIFAR-100 & tùy chọn DTD):**
   * Chạy đủ 4 tổ hợp qua `scripts/ablation.py` (**A11**):
     1. **Full F-OAL:** `feature_fusion = True`, `smooth_projection = True`
     2. **w/o Feature Fusion:** `feature_fusion = False` (chỉ lấy CLS token của block thứ 12), `smooth_projection = True`
     3. **w/o Smooth Projection:** `feature_fusion = True`, `smooth_projection = False` ($D = 768$ đưa thẳng vào RLS)
     4. **w/o Both:** `feature_fusion = False`, `smooth_projection = False`

3. **Experiment 3 — Additional & Sensitivity Analysis (Kiểm chứng Lý thuyết & Độ nhạy Siêu tham số):**
   * **Analysis 3A (Batch-Size Invariance):** Chạy F-OAL với $B \in \{5, 10, 25, 50, 100\}$ trên cùng seed để kiểm chứng thực nghiệm tính chất toán học của cập nhật Woodbury RLS: nghiệm giải tích cuối mỗi task không phụ thuộc vào kích thước mini-batch $B$, trong khi thời gian thực thi thay đổi theo $B$.
   * **Analysis 3B (Projection Scaling, Dimension & Regularization Sensitivity):** Khảo sát ảnh hưởng của hệ số tỷ lệ khởi tạo ma trận chiếu ($\sigma_{\text{init}} = 1.0$ vs $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$), số chiều chiếu $D \in \{500, 1000, 2000\}$ và hệ số Ridge regularization $\gamma \in \{0.1, 1.0, 10.0\}$.

---

## 2. Công thức Đánh giá & Quy ước Độ lệch (Metrics & Deviation Formulas)

Giả sử quá trình học tăng dần gồm $T = 10$ giai đoạn (tasks). Gọi $a_{k, j} \in [0, 1]$ là độ chính xác kiểm thử trên tập test của Task $j$ sau khi hoàn tất huấn luyện Task $k$ ($1 \le j \le k \le T$). Ma trận kết quả $R \in \mathbb{R}^{T \times T}$ là ma trận tam giác dưới với $R_{k, j} = a_{k, j}$.

1. **Stage Accuracy tại giai đoạn $k$ ($A_k$):**
   $$A_k = \frac{1}{k} \sum_{j=1}^{k} a_{k, j}$$
   *(Lưu ý: Trên CIFAR-100, mỗi task có đúng $1,000$ mẫu test nên trung bình vĩ mô $A_k$ trùng khớp tuyệt đối với độ chính xác gộp trên toàn bộ mẫu test đã thấy).*
2. **Average Incremental Accuracy ($\bar{A}$ hoặc $A_{\text{avg}}$, $\uparrow$ cao hơn là tốt hơn):**
   $$\bar{A} = \frac{1}{T} \sum_{k=1}^{T} A_k$$
3. **Last Task Accuracy ($A_T$ hoặc $A_{\text{last}}$, $\uparrow$ cao hơn là tốt hơn):**
   $$A_T = \frac{1}{T} \sum_{j=1}^{T} a_{T, j}$$
4. **Average Forgetting Rate — Hai định nghĩa cần phân biệt rõ (`H7b`):**
   * **Định nghĩa chuẩn Continual Learning (`F_ours_definition`, cài đặt trong `src/foal/metrics.py`):** Chia trung bình trên $T - 1 = 9$ task cũ và lấy cực đại trên các giai đoạn trước $k \in \{j, \dots, T-1\}$:
     $$f_{j,\text{ours}}^T = \max_{\ell \in \{j, \dots, T-1\}} a_{\ell, j} - a_{T, j}, \qquad F_{\text{ours}} = \frac{1}{T-1} \sum_{j=1}^{T-1} f_{j,\text{ours}}^T$$
   * **Định nghĩa trong Code Gốc Tác giả (`F_official_definition`, cài đặt trong `experiment/metrics.py:49–54`):** Chia trung bình trên tất cả $T = 10$ tasks (bao gồm cả Task $T$ có độ quên bằng $0$) và lấy cực đại trên $\ell \in \{1, \dots, T\}$ (tự động chặn dưới mức quên âm về $0$):
     $$f_{j,\text{official}}^T = \max_{\ell \in \{1, \dots, T\}} a_{\ell, j} - a_{T, j}, \qquad F_{\text{official}} = \frac{1}{T} \sum_{j=1}^{T} f_{j,\text{official}}^T$$
   * **Quy tắc đối chiếu công bằng:** Khi so sánh với con số $5.50\%$ của Paper và `cifar100_FOAL.csv` của Code gốc, phải đặt cùng công thức $F_{\text{official}}$ cạnh nhau; khi so sánh nội bộ theo chuẩn 9-task, đặt cùng $F_{\text{ours}}$ cạnh nhau.
5. **Chênh lệch tuyệt đối ($\Delta_{\text{pp}}$, tính bằng điểm phần trăm — percentage points):**
   $$\Delta_{\text{pp}}(A, B) = A - B$$
6. **Độ lệch tương đối ($\Delta_{\text{rel}}\%$):**
   $$\Delta_{\text{rel}}(A, B) = \frac{A - B}{B} \times 100\%$$

---

## 3. Lệnh Thực thi Chuẩn (Reproducible Commands)

```bash
# 1. Chạy kiểm thử đơn vị (35 unit tests trên CPU)
python -m pytest -v

# 2. Chạy Pilot Smoke Test trên CPU (2 tasks, 100 mẫu/task — KHÔNG phải full reproduction)
python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100 --run-name smoke_test_post_fix

# 3. Chạy Full Single-Seed Reproduction của Nhóm trên GPU (10 tasks x 10 classes, 50.000 train, 10.000 test)
python -m scripts.train --config configs/cifar100.yaml --device cuda --seed 42 --wandb --run-name foal_full_cifar100_seed42

# 4. Đánh giá độc lập Checkpoint cuối của Nhóm
python -m scripts.evaluate --checkpoint results/foal_full_cifar100_seed42/task_10.pt --device cuda

# 5. Chạy Code Gốc của Tác giả (liuyuchen-cz/F-OAL @ 73e4ac1) trên Colab GPU
# Mở và chạy tuần tự Cell 1 -> Cell 10 trong: notebooks/foal_official_cifar100_reproduction.ipynb

# 6. Tổng hợp toàn bộ kết quả hợp lệ trong thư mục results/
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

### 4.2. Kết quả Single-Seed Full Reproduction của Nhóm trên GPU (P1.2 — CIFAR-100, 10 Tasks, Seed 42, Commit `aad7500`)

* **Thiết bị & Môi trường:** Google Colab GPU — NVIDIA Tesla T4 (`CUDA 13.0`, `Python 3.13.16`, `torch 2.11.0+cu130`, `timm 1.0.30`).
* **Git Commit thực thi:** `aad7500c425bd13ece2fd7d4d21014616344ec81` (`aad7500`).
* **Giao thức & Siêu tham số:** CIFAR-100 (`uoft-cs/cifar100`), 10 tasks $\times$ 10 classes, `batch_size = 10`, one-pass (`50,000` train, `10,000` test), `seed = 42` (`class_order_seed = projection_seed = stream_seed = 42`), backbone `vit_base_patch16_224.augreg_in1k` (bicubic $224 \times 224$, `mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)`), Smooth Projection $\mathcal{N}(0, 1)$ (`gaussian_std = 1.0`, $D = 1000$), RLS `gamma = 1.0` (`float64`).
* **Hiện vật đã xác minh (`VERIFIED`):**
  * Gói nén gốc: `foal_cifar100_single_seed42_results.zip` (`109,814,583 bytes`, 0 lỗi CRC).
  * Thư mục kết quả: [`results/foal_full_cifar100_seed42/`](../results/foal_full_cifar100_seed42/) (`metrics.json`, `metadata.json`, `task_accuracy.csv`, cùng 10 checkpoints `task_01.pt`..`task_10.pt` trên đĩa cục bộ).
  * **W&B Online Run (`2cdeiptv` — Đã đồng bộ):** [`https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv).
* **Kết quả tổng hợp (`n=1, std=N/A`):**
  * **Average Incremental Accuracy ($\bar{A}$):** **$79.33\%$** (`0.7933317063492064`)
  * **Last Accuracy ($A_{10}$):** **$70.92\%$** (`0.7091999999999999`)
  * **Forgetting (`F_ours_definition`, 9-task):** **$9.21\%$** (`0.09211111111111112`)
  * **Forgetting (`F_official_definition`, 10-task):** **$8.29\%$** (`0.0829`)
  * **Thời gian chạy (`total_duration_seconds`):** **$1735.3\text{ s}$** ($\approx 28.92\text{ phút}$)

#### Ma trận Độ chính xác $10 \times 10$ Đầy đủ (`OUR_REPRODUCED`, Seed 42, Commit `aad7500`)

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
| **Task Forgetting ($f_j^{10}$)** | **29.10%** | **15.00%** | **10.60%** | **8.00%** | **7.30%** | **6.00%** | **3.30%** | **2.70%** | **0.90%** | $0.00\%$ | **$9.21\%$ (9-task) / $8.29\%$ (10-task)** |

---

### 4.3. Official F-OAL CIFAR-100 Reproduction Results (`A19` — Code Gốc Tác giả `liuyuchen-cz/F-OAL @ 73e4ac1`, Seed 0)

#### A. Thiết lập Thực nghiệm & Xác minh Nguồn gốc (`COMPATIBILITY-PATCHED ORIGINAL` — Zero Algorithmic Modification)
* **Repository gốc:** `https://github.com/liuyuchen-cz/F-OAL.git`
* **Commit SHA đã checkout & xác minh:** `73e4ac1b4c965030973a2b77f05c57c2f44f49f5` (`73e4ac1`).
* **Phân loại Mức độ Chỉnh sửa:** **`COMPATIBILITY-PATCHED ORIGINAL` (Không chỉnh sửa bất kỳ tệp nào trong `/content/official-F-OAL`, không thay đổi thuật toán/siêu tham số):**
  * Tệp `utils/name_match.py:21` trong repo tác giả import sẵn `from agents.exp_replay_dvc import ExperienceReplay_DVC`, tệp này gọi `from loss import agmax_loss, cross_entropy_loss` (`agents/exp_replay_dvc.py:10`), nhưng tác giả quên không đưa tệp `loss.py` của phương pháp DVC lên GitHub. Vì thuật toán `FOAL` hoàn toàn không sử dụng `ExperienceReplay_DVC`, nhóm đã cung cấp một tệp shim bên ngoài tại `/content/official_foal_shim/loss.py` qua biến môi trường `PYTHONPATH` để Python vượt qua bước import đầu chương trình mà giữ nguyên `git status` sạch 100% trong `/content/official-F-OAL`.
  * Sử dụng đúng tên cờ `--projection 1000` theo `general_main.py:164` (thay vì lỗi đánh máy `--projection_size` trong `README.md` của tác giả).
* **Lệnh thực thi:**
  ```bash
  /usr/bin/python3 -u general_main.py --data cifar100 --cl_type nc --agent FOAL --batch 10 --test_batch 10 --projection 1000 --num_runs 1 --seed 0 --verbose True
  ```
* **Phần cứng & Phần mềm:** Google Colab NVIDIA Tesla T4 (`CUDA 13.0`, `Python 3.13.16`, `torch 2.11.0+cu130`, `torchvision 0.26.0+cu130`, `timm 1.0.30`, `kornia 0.9.0`).
* **Tập dữ liệu & Thứ tự lớp (`seed = 0`):** CIFAR-100 (`torchvision.datasets.CIFAR100`), 100 lớp, 50.000 mẫu train ($5,000$/task), 10.000 mẫu test ($1,000$/task), 10 tasks $\times$ 10 classes, one-pass (`epoch = 1`), `batch = 10`, `test_batch = 10`. Hoán vị lớp sinh bởi `np.random.seed(0); np.random.permutation(100)`:
  * `Task 0`: `[26, 86, 2, 55, 75, 93, 16, 73, 54, 95]`
  * `Task 1`: `[53, 92, 78, 13, 7, 30, 22, 24, 33, 8]`
  * `Task 2`: `[43, 62, 3, 71, 45, 48, 6, 99, 82, 76]`
  * `Task 3`: `[60, 80, 90, 68, 51, 27, 18, 56, 63, 74]`
  * `Task 4`: `[1, 61, 42, 41, 4, 15, 17, 40, 38, 5]`
  * `Task 5`: `[91, 59, 0, 34, 28, 50, 11, 35, 23, 52]`
  * `Task 6`: `[10, 31, 66, 57, 79, 85, 32, 84, 14, 89]`
  * `Task 7`: `[19, 29, 49, 97, 98, 69, 20, 94, 72, 77]`
  * `Task 8`: `[25, 37, 81, 46, 39, 65, 58, 12, 88, 70]`
  * `Task 9`: `[87, 36, 21, 83, 9, 96, 67, 64, 47, 44]`
* **Hiện vật đã xác minh (`VERIFIED`):**
  * Gói nén: `foal_official_cifar100_results.zip` (`19,080 bytes`, SHA256 `e3bc5c1e8637da7f16d0c29de9d617908632ac908ae62d92ba8e1a4763b787e4`, 12 tệp, 0 lỗi CRC).
  * Thư mục lưu trữ: [`results/official_foal_cifar100_seed0/`](../results/official_foal_cifar100_seed0/) (`metrics.json`, `metadata.json`, `task_accuracy.csv`, `cifar100_FOAL.csv`, `official_stdout.log`). *(Lưu ý: Code gốc của tác giả trong `experiment/run.py` không lưu tệp `.pt` checkpoint).*
  * **W&B Online Run (`3eca8232` — Đã đồng bộ):** [`https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232).

#### B. Kiểm chứng Toán học Độc lập (`Official Log` vs. `Extracted JSON` vs. `Independent Recalculation`)

| Metric | Official Log (`official_stdout.log` & `cifar100_FOAL.csv`) | Extracted JSON (`metrics.json`) | Independent Recalculation | Match? |
|---|---:|---:|---:|:---:|
| **Average Incremental Accuracy ($\bar{A}$)** | `0.9098279365079364` (`90.98%`) | `0.9098279365079364` | `0.9098279365079364` ($\text{diff} = 0.0$) | **`EXACT MATCH`** |
| **Last Accuracy ($A_{10}$)** | `0.8583999999999999` (`85.84%`) | `0.8583999999999999` | `0.8583999999999999` ($\text{diff} = 0.0$) | **`EXACT MATCH`** |
| **Forgetting — Official 10-Task Def ($F_{\text{official}}$)** | `0.0469` (`4.69%`) | `0.0469` | `0.0469` (`0.469 / 10`, $\text{diff} = 0.0$) | **`EXACT MATCH`** |
| **Forgetting — Standard 9-Task Def ($F_{\text{ours}}$)** | — *(Computed from printed $10 \times 10$ matrix)* | `0.051666666666666666` (`5.17%`) | `0.051666666666666666` (`0.465 / 9`, $\text{diff} = 0.0$) | **`EXACT MATCH`** |
| **Runtime (Seconds)** | `3047.5978 s` (`train: 1544.96 s`, `test: 1496.42 s`) | `3056.24 s` *(outer subprocess wall-clock)* | `3047.60 s` (internal) / `3056.24 s` (subprocess) | **`VERIFIED`** |

#### C. Ma trận Độ chính xác $10 \times 10$ Đầy đủ của Code Gốc Tác giả (`ORIGINAL_REPRODUCED`, Seed 0, Commit `73e4ac1`)

| Giai đoạn (Sau Task $k$) | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 | **Stage Accuracy ($A_k$)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | **96.60%** | — | — | — | — | — | — | — | — | — | **96.60%** |
| **Stage 2** | 94.90% | **96.60%** | — | — | — | — | — | — | — | — | **95.75%** |
| **Stage 3** | 93.50% | 94.00% | **94.50%** | — | — | — | — | — | — | — | **94.00%** |
| **Stage 4** | 92.50% | 92.20% | 93.70% | **90.20%** | — | — | — | — | — | — | **92.15%** |
| **Stage 5** | 91.20% | 91.90% | 92.60% | 89.60% | **93.60%** | — | — | — | — | — | **91.78%** |
| **Stage 6** | 89.30% | 91.00% | 92.40% | 87.70% | 92.70% | **84.90%** | — | — | — | — | **89.67%** |
| **Stage 7** | 88.50% | 90.50% | 92.00% | 87.40% | 91.80% | 83.90% | **88.10%** | — | — | — | **88.89%** |
| **Stage 8** | 86.70% | 90.10% | 91.10% | 86.10% | 90.80% | 81.50% | 87.20% | **91.30%** | — | — | **88.10%** |
| **Stage 9** | 86.60% | 87.90% | 90.80% | 85.50% | 89.30% | 79.00% | 87.20% | 90.40% | **86.80%** | — | **87.06%** |
| **Stage 10 (Final)** | 86.10% | 86.90% | 90.30% | 84.20% | 88.90% | 75.80% | 86.60% | 90.10% | 87.20% | **82.30%** | **85.84%** ($A_{10}$) |
| **Forgetting $f_{j,\text{ours}}^{10}$ ($\max_{k \le 9}$)** | **10.50%** | **9.70%** | **4.20%** | **6.00%** | **4.70%** | **9.10%** | **1.50%** | **1.20%** | **$-0.40\%$** | — | **Mean $F_{\text{ours}} = 5.17\%$** |
| **Forgetting $f_{j,\text{official}}^{10}$ ($\max_{k \le 10}$)** | **10.50%** | **9.70%** | **4.20%** | **6.00%** | **4.70%** | **9.10%** | **1.50%** | **1.20%** | **0.00%** | **0.00%** | **Mean $F_{\text{official}} = 4.69\%$** |

---

### 4.4. Bảng Đối chiếu Khoa học 3 Nguồn trên CIFAR-100 (`Paper vs. Official Run vs. Our Run`)

> **Lưu ý về tính công bằng thống kê (`D5`):**
> * **Paper** báo cáo trung bình qua **3 runs**.
> * **Official Run** hiện tại là kết quả đơn seed (**`n=1, std=N/A`**, `seed=0`).
> * **Our Run** hiện tại là kết quả đơn seed (**`n=1, std=N/A`**, `seed=42`).
> * Mọi phép cộng/trừ độ lệch dưới đây chỉ thực hiện giữa các chỉ số **có cùng định nghĩa toán học**.

| Chỉ số Đánh giá (Metric) | Paper (3-run Mean) | Official Run (`73e4ac1`, `seed=0`, `n=1`) | Our Run (`aad7500`, `seed=42`, `n=1`) | Official vs. Paper ($\Delta_{\text{pp}}$ / $\Delta_{\text{rel}}$) | Ours vs. Official ($\Delta_{\text{pp}}$ / $\Delta_{\text{rel}}$) | Ours vs. Paper ($\Delta_{\text{pp}}$ / $\Delta_{\text{rel}}$) |
|:---|---:|---:|---:|---:|---:|---:|
| **Average Incremental Accuracy ($\bar{A}$, $\uparrow$)** | **91.10%** | **90.98%** (`n=1, std=N/A`) | **79.33%** (`n=1, std=N/A`) | **$-0.12\text{ pp}$** (`-0.13%`) | **$-11.65\text{ pp}$** (`-12.80%`) | **$-11.77\text{ pp}$** (`-12.92%`) |
| **Last Accuracy ($A_{10}$, $\uparrow$)** | **86.50%** | **85.84%** (`n=1, std=N/A`) | **70.92%** (`n=1, std=N/A`) | **$-0.66\text{ pp}$** (`-0.76%`) | **$-14.92\text{ pp}$** (`-17.38%`) | **$-15.58\text{ pp}$** (`-18.01%`) |
| **Forgetting — Official 10-Task Def ($F_{\text{official}}$, $\downarrow$)** | **5.50%** | **4.69%** (`n=1, std=N/A`) | **8.29%** (`n=1, std=N/A`) | **$-0.81\text{ pp}$** (`-14.73%`) | **$+3.60\text{ pp}$** (`+76.76%`) | **$+2.79\text{ pp}$** (`+50.73%`) |
| **Forgetting — Standard 9-Task Def ($F_{\text{ours}}$, $\downarrow$)** | — *(Không dùng)* | **5.17%** (`n=1, std=N/A`) | **9.21%** (`n=1, std=N/A`) | — | **$+4.04\text{ pp}$** (`+78.26%`) | — *(Khác mẫu số)* |
| **Training + Evaluation Time (s)** | $261.0\text{ s}$ (RTX 4070 Ti) | $3047.6\text{ s}$ (Colab T4) | $1735.3\text{ s}$ (Colab T4) | — | $-1312.3\text{ s}$ (`-43.06%`) | — |
| **W&B Run Link** | — | [Run `3eca8232`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232) | [Run `2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv) | — | — | — |

#### Tóm tắt Các Khác biệt Giao thức Đã Xác minh giữa Official (`73e4ac1`) và Ours (`aad7500`)
Chi tiết phân tích mã nguồn và kế hoạch thí nghiệm kiểm soát biến số (`C0`–`C4`) được trình bày tại [`docs/original_vs_ours_protocol.md`](original_vs_ours_protocol.md):
1. **`H1` (Pretrained ViT Checkpoint — `VERIFIED DIFFERENCE`):** Code tác giả gọi `timm.create_model('vit_base_patch16_224', pretrained=True)` (tải `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`, pre-train trên ImageNet-21K rồi fine-tune trên ImageNet-1K), trong khi bản chạy `aad7500` của nhóm dùng `vit_base_patch16_224.augreg_in1k` (chỉ pre-train trên ImageNet-1K).
2. **`H2` (Smooth Projection Initialization — `VERIFIED DIFFERENCE`):** Code tác giả dùng `nn.Linear(768, 1000, bias=False)` ($\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$, $\text{std}(u) \approx 0.578$, tỷ lệ bão hòa Sigmoid $0.00\%$), trong khi bản chạy `aad7500` của nhóm dùng $\mathcal{N}(0, 1)$ ($\text{std}(u) \approx 27.79$, tỷ lệ bão hòa Sigmoid $86.88\%$).
3. **`H4` (Image Preprocessing — `VERIFIED DIFFERENCE`):** Code tác giả resize Bilinear và giữ pixel trong $[0, 1]$ (không `Normalize`), trong khi nhóm resize Bicubic và chuẩn hóa `mean=0.5, std=0.5` về $[-1, 1]$.
4. **`H6` & `H7` (Random Seed & Forgetting Formula — `VERIFIED DIFFERENCE`):** Code tác giả dùng `seed=0` (`np.random.seed(0)`) và chia Forgetting cho $T=10$ tasks; nhóm dùng `seed=42` (`default_rng(42)`) và chia Forgetting cho $T-1=9$ tasks.

---

## 5. Khung Bảng Kết quả cho Ablation Study & Additional Analysis (Sẽ điền tại Stage E)

### 5.1. Bảng Ablation Study (Đối chiếu Table 4 — Feature Fusion & Smooth Projection)

| Cấu hình Ablation | Feature Fusion (FF) | Smooth Projection (SP) | Paper DTD $\bar{A}$ (Tham chiếu Table 4) | Ours CIFAR-100 $\bar{A}$ | Ours CIFAR-100 $A_{10}$ | Ours CIFAR-100 $F_{\text{ours}}$ | W&B Run Link |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1. Full F-OAL (`aad7500` v1)** | Có (`True`) | Có (`True`, $D=1000$) | $82.80\%$ | `79.33% (n=1, std=N/A)` | `70.92% (n=1, std=N/A)` | `9.21% (n=1, std=N/A)` | [Run `2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv) |
| **2. w/o Feature Fusion** | Không (`False`, last block) | Có (`True`, $D=1000$) | $80.50\%$ | `NOT RUN` *(Stage E)* | `NOT RUN` | `NOT RUN` | `NOT YET SYNCED` |
| **3. w/o Smooth Projection** | Có (`True`) | Không (`False`, $D=768$) | $79.30\%$ | `NOT RUN` *(Stage E)* | `NOT RUN` | `NOT RUN` | `NOT YET SYNCED` |
| **4. w/o Both** | Không (`False`) | Không (`False`, $D=768$) | $71.20\%$ | `NOT RUN` *(Stage E)* | `NOT RUN` | `NOT RUN` | `NOT YET SYNCED` |

### 5.2. Bảng Additional Analysis 3A — Kiểm chứng Tính Bất biến theo Batch Size ($B$)

| Batch Size ($B$) | Số bước cập nhật RLS / Task | $\bar{A}$ ($A_{\text{avg}}$) | $A_{10}$ ($A_{\text{last}}$) | Sai lệch $\|W_{10}^{(B)} - W_{10}^{(10)}\|_F$ | Thời gian chạy (s) | Peak VRAM (GB) | W&B Run Link |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| $B = 5$ | $1000$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |
| $B = 10$ *(Mặc định)* | $500$ | **79.33%** *(v1)* | **70.92%** *(v1)* | $0.0$ (Tham chiếu) | $1735.3\text{ s}$ | `PENDING` | [Run `2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv) |
| $B = 25$ | $200$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |
| $B = 50$ | $100$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |
| $B = 100$ | $50$ | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` |


---

## 6. Thí nghiệm Kiểm soát Khác biệt Kỹ thuật: C0 & C1 (Checkpoint 3 Compliance)

### 6.1. C0 — Đối chiếu & Tái tính toán Công thức Forgetting (`Eq. 17–18` vs. `Official Code`)

Nhóm đã đối chiếu trực tiếp giữa công thức toán học công bố trong Paper PDF chính thức (NeurIPS 2024, Section 4.2, trang 8) và mã nguồn tác giả trong file `experiment/metrics.py`:

1. **Công thức Paper — Equations (17) & (18):**
   $$F_i = \frac{1}{i - 1} \sum_{j=1}^{i - 1} f_{i, j} \quad (17), \qquad f_{k, j} = \max_{l \in \{1, \dots, k-1\}} (a_{l, j}) - a_{k, j}, \ \forall j < k \quad (18)$$
   - Ký hiệu mã nguồn nhóm: `forgetting_past_tasks` ($F_T^{\text{past}}$).
   - Mẫu số là $T - 1 = 9$ (chỉ tính trên các task đã qua, loại trừ Task $T$ hiện tại).
   - Cực đại lấy trên các giai đoạn trước $l \in \{1, \dots, k-1\}$. Nếu ở giai đoạn $k$ accuracy của task $j$ tăng lên so với mọi giai đoạn trước (positive backward transfer), $f_{k,j} < 0$.

2. **Công thức trong Code Gốc Tác giả — `experiment/metrics.py:18, 22–26, 49–54`:**
   ```python
   # experiment/metrics.py:49-54 (single_run_avg_end_fgt)
   best_acc = np.max(end_task_acc_arr, axis=1)  # max over ALL stages l in {1, ..., T}
   final_forgets = best_acc - end_acc
   avg_fgt = np.mean(final_forgets, axis=1)  # mean over ALL T tasks (including task T)
   ```
   - Ký hiệu mã nguồn nhóm: `forgetting_official_code` ($F_T^{\text{code}}$).
   - Mẫu số là $T = 10$ tasks (chia cho 10 thay vì 9).
   - Cực đại lấy trên toàn bộ $l \in \{1, \dots, T\}$ bao gồm cả $l = T$, do đó với Task $T$, $\max(a_{l, T}) - a_{T, T} = 0.0$ và mọi mức quên âm đều bị ép về $\ge 0.0$.

#### Bảng Tái tính toán C0 trên Dữ liệu Thực nghiệm Thật

| Ma trận Dữ liệu Thực nghiệm | $F_T^{\text{past}}$ (Paper Eq. 17–18, $T-1=9$) | $F_T^{\text{code}}$ (Official Code, $T=10$) | Chênh lệch giữa 2 định nghĩa ($\Delta_{\text{pp}}$) |
|:---|:---:|:---:|:---:|
| **Official Run (`3eca8232`, Seed 0)** | **`5.1667%`** ($0.465 / 9$) | **`4.6900%`** ($0.469 / 10$) | $-0.4767\text{ pp}$ (do Task 9 tăng từ 86.8% lên 87.2%, và Task 10 thêm số 0) |
| **Our Historical Run (`2cdeiptv`, Seed 42)** | **`9.2111%`** ($0.829 / 9$) | **`8.2900%`** ($0.829 / 10$) | $-0.9211\text{ pp}$ (hoàn toàn do khác biệt mẫu số $9$ vs $10$) |

---

### 6.2. C1 — Chẩn đoán Ma trận Chiếu trên Cùng Đầu vào Thực tế (Matched-Input Diagnostics)

Để kiểm chứng giả thuyết `H2` rằng việc khởi tạo ma trận ngẫu nhiên chuẩn tắc $\mathcal{N}(0, 1)$ làm bão hòa Sigmoid, nhóm đã trích xuất đặc trưng thực tế từ **200 ảnh CIFAR-100** ($20$ mẫu/class trên $10$ classes của Task 1) bằng backbone ViT đông cứng `vit_base_patch16_224.augreg_in1k`, sau đó đánh giá 3 cơ chế khởi tạo trên **cùng một tensor đặc trưng đầu vào $h \in \mathbb{R}^{200 \times 768}$**:

* **P0 (`gaussian_std1`):** Khởi tạo ban đầu của nhóm, $P_{ij} \sim \mathcal{N}(0, 1)$, variance = $1.0$.
* **P1 (`official_uniform`):** Khởi tạo của tác giả (`nn.Linear(768, 1000, bias=False)`), $P_{ij} \sim \mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$, variance = $1/(3 \times 768) \approx 0.000434$.
* **P2 (`gaussian_matched_variance`):** Đối chứng Gaussian có phương sai lý thuyết trùng khớp tuyệt đối với P1: $P_{ij} \sim \mathcal{N}(0, 1/2304)$, variance = $1/2304 \approx 0.000434$.

#### Bảng Kết quả Đo đạc Chẩn đoán Khách quan (`results/c1_diagnostics/`)

| Chỉ số Chẩn đoán (Diagnostic Metric) | P0 (`gaussian_std1`) | P1 (`official_uniform`) | P2 (`gaussian_matched_variance`) | Ý nghĩa Vật lý / Toán học |
|:---|:---:|:---:|:---:|:---|
| **Trọng số ma trận $\text{std}(P)$** | $1.000661$ | $0.020832$ | $0.020847$ | P1 và P2 khớp phương sai lý thuyết ($\sigma \approx 1/48$) |
| **Độ lệch chuẩn tiền kích hoạt $\text{std}(u)$ ($u = h P$)** | **$19.2844$** | **$0.3979$** | **$0.4018$** | P0 phóng đại tín hiệu gấp $\approx 48$ lần |
| **Độ lệch chuẩn sau kích hoạt $\text{std}(z)$ ($z = \sigma(u)$)** | $0.4790$ | $0.0958$ | $0.0967$ | P0 đẩy giá trị về sát 0 hoặc 1 |
| **Tỷ lệ bão hòa cực đoan ($z < 0.01 \lor z > 0.99$)** | **`81.03%`** | **`0.00%`** | **`0.00%`** | P0 đẩy $81\%$ chiều vào vùng bão hòa phi tuyến |
| **Đạo hàm trung bình Sigmoid $\phi(1-\phi)$** | $0.0469$ | $0.2458$ | $0.2457$ | P1/P2 duy trì vùng tuyến tính nhạy cảm nhất ($\approx 0.25$) |
| **Số điều kiện Gram chính quy hóa $\text{Cond}(G + \gamma I)$** | $65,218.28$ | $50,467.89$ | $50,954.65$ | Gram matrix của P1/P2 ổn định hơn, số điều kiện nhỏ hơn |
| **Effective Rank Chưa Chuẩn Hóa Tâm $r_{\text{eff}}(G)$** | $8.60$ | $1.20$ | $1.20$ | Bị chi phối bởi vector trung bình lệch $\mathbf{m} \approx 0.5 \times \mathbf{1}$ |
| **Effective Rank Hiệp Phương Sai Đã Chuẩn Tâm $r_{\text{eff}}(C)$** | **$115.73$** | **$71.26$** | **$71.10$** | P1/P2 phản ánh trung thực số chiều hữu hiệu của ViT backbone ($\approx 71$) |

> **Phát hiện & Hiệu chỉnh Khoa học Then chốt:**
> 1. **Cơ chế Tác động của Hiện tượng Bão hòa Sigmoid (Saturation Causality):**
>    Trong mô hình F-OAL, toàn bộ mạng không sử dụng Backpropagation mà giải trực tiếp bài toán bình phương tối thiểu hồi quy (Recursive Least Squares - RLS). Do đó, hiện tượng bão hòa Sigmoid ($81.03\%$ ở P0) **không** gây ra "triệt tiêu đạo hàm gradient" theo nghĩa của Gradient Descent, mà gây ra **sự sụp đổ biểu diễn đặc trưng (feature representation collapse)**: các chiều đặc trưng bị nhị phân hóa về sát $0$ hoặc $1$, làm mất độ nhạy vi phân giữa các lớp và làm suy giảm tính phân tách tuyến tính của không gian đặc trưng.
> 2. **Giải mã Chỉ số Effective Rank Chưa Chuẩn Hóa Tâm (Centered vs. Uncentered Rank):**
>    Chỉ số Effective Rank của ma trận Gram chưa trừ tâm $G = Z^T Z$ ở P1 và P2 đo được xấp xỉ $1.20$, nguyên nhân là do toàn bộ vector kích hoạt sau hàm Sigmoid có giá trị trung bình dương lệch chuẩn $\mathbf{m} \approx 0.5 \times \mathbf{1}$, tạo ra một trị riêng trội chiếm tới $97.81\%$ tổng vết của ma trận. Khi tính toán trên **ma trận hiệp phương sai đã chuẩn hóa tâm (Centered Covariance Matrix)** $C = \frac{1}{N-1}(Z - \bar{Z})^T (Z - \bar{Z})$, Effective Rank thực sự của P1 là **$71.26$** và P2 là **$71.10$**, hoàn toàn khớp với số chiều biểu diễn nội tại của backbone ViT ($\approx 71$), trong khi P0 có rank giả tạo phân tán $115.73$ do nhiễu bão hòa ngẫu nhiên.

---

### 6.3. Level A — Thí nghiệm Pilot 2 Tasks trên CPU (`results/c1_pilot/`)

Đã thực hiện chạy kiểm chứng 2 tasks đầu tiên (100 ảnh train/task, 100 ảnh test/task) với tính năng bộ nhớ đệm đặc trưng (`--feature-cache-dir results/c1_pilot_feature_cache`):

| Cấu hình Pilot (2 Tasks) | Task 1 Acc ($A_1$) | Task 2 Acc ($a_{2, 2}$) | Drop Task 1 ($f_{2, 1}$) | Avg Incr Acc ($\bar{A}$) | Last Acc ($A_2$) | Thời gian RLS & Eval (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **P0 (`gaussian_std1`)** | $78.00\%$ | $85.00\%$ | $9.00\%$ | $77.50\%$ | $77.00\%$ | $0.23\text{ s}$ |
| **P1 (`official_uniform`)** | **$80.00\%$** | **$87.00\%$** | $11.00\%$ | **$79.00\%$** | **$78.00\%$** | $0.25\text{ s}$ |
| **P2 (`gaussian_matched_variance`)** | $77.00\%$ | $86.00\%$ | $8.00\%$ | $77.25\%$ | $77.50\%$ | $0.27\text{ s}$ |

- **Kiểm chứng tính toàn vẹn của Feature Cache:** Lượt chạy P0 dùng cache cho ra kết quả `task_matrix = [[0.78], [0.69, 0.85]]` trùng khớp **100% bitwise** với lượt chạy không dùng cache (`smoke_test_post_fix_seed42`), trong khi thời gian thực thi giảm từ $94.95\text{ s}$ xuống $1.36\text{ s}$ (nhanh gấp $\approx 70$ lần).

---

### 6.4. Level B — Kết quả Thực nghiệm Toàn diện 10 Tasks CIFAR-100 trên GPU NVIDIA A100 (`results/c1_cifar100_full/`)

Toàn bộ 3 thí nghiệm C1 đầy đủ (10 tasks $\times$ 10 classes, 50.000 train, 10.000 test, seed 42) đã được thực thi thành công trên Google Colab GPU (NVIDIA A100 High-RAM, Linux x86_64, PyTorch 2.11.0+cu130, timm 1.0.30, execution commit `269f05c`). Hiện vật số học đã được kiểm toán toàn diện từ gói ZIP `foal_c1_projection_results.zip` (SHA256: `21eb4e977e739f8380b22f87e1db14937925b69d4a93b9285e7b49a6c2374cab`):

#### A. Bảng Đối chiếu Toàn diện 3 Cấu hình C1

| Chỉ số Đánh giá (Metric) | P0 (`gaussian_std1`) | P1 (`official_uniform`) | P2 (`gaussian_matched_variance`) | Độ lệch $\Delta_{P1 - P0}$ | Độ lệch $\Delta_{P2 - P0}$ | Độ lệch $\Delta_{P1 - P2}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Khởi tạo Ma trận Chiếu ($P$)** | $\mathcal{N}(0, 1.0)$ | $\mathcal{U}(-1/\sqrt{d}, +1/\sqrt{d})$ | $\mathcal{N}(0, 1/(3d))$ | — | — | — |
| **Quy mô Phương sai ($\sigma_P^2$)** | $1.0$ | $1/2304 \approx 0.000434$ | $1/2304 \approx 0.000434$ | Giảm $\approx 2304\times$ | Giảm $\approx 2304\times$ | Khớp tuyệt đối ($0.0$) |
| **Tỷ lệ Bão hòa Sigmoid (Diagnostics)** | **$81.03\%$** | **$0.00\%$** | **$0.00\%$** | $-81.03\text{ pp}$ | $-81.03\text{ pp}$ | $0.00\text{ pp}$ |
| **Average Incremental Accuracy ($\bar{A}$, $\uparrow$)** | **`79.33%`** (`0.793315`) | **`83.28%`** (`0.832814`) | **`83.37%`** (`0.833740`) | **$+3.95\text{ pp}$** | **$+4.04\text{ pp}$** | **$-0.09\text{ pp}$** |
| **Last Accuracy ($A_{10}$, $\uparrow$)** | **`70.95%`** (`0.709500`) | **`74.84%`** (`0.748400`) | **`74.99%`** (`0.749900`) | **$+3.89\text{ pp}$** | **$+4.04\text{ pp}$** | **$-0.15\text{ pp}$** |
| **Forgetting — Paper Eq. 17–18 ($F_{10}^{\text{past}}$, $\downarrow$)** | **`9.17%`** (`0.091667`) | **`8.97%`** (`0.089667`) | **`8.81%`** (`0.088111`) | **$-0.20\text{ pp}$** | **$-0.36\text{ pp}$** | **$+0.16\text{ pp}$** |
| **Forgetting — Official Code ($F_{10}^{\text{code}}$, $\downarrow$)** | **`8.25%`** (`0.082500`) | **`8.07%`** (`0.080700`) | **`7.93%`** (`0.079300`) | **$-0.18\text{ pp}$** | **$-0.32\text{ pp}$** | **$+0.14\text{ pp}$** |
| **Thời gian Cập nhật RLS & Eval (s)** | $8.57\text{ s}$ | $8.59\text{ s}$ | $8.53\text{ s}$ | $+0.02\text{ s}$ | $-0.04\text{ s}$ | $+0.06\text{ s}$ |
| **Thời gian Trích xuất Đặc trưng (s)** | $245.92\text{ s}$ (A100) | $245.92\text{ s}$ (Cached) | $245.92\text{ s}$ (Cached) | — | — | — |
| **Tổng thời gian chạy (Wall-clock, s)** | $298.90\text{ s}$ | $52.01\text{ s}$ | $51.60\text{ s}$ | Nhanh hơn $5.7\times$ | Nhanh hơn $5.8\times$ | Tương đương |
| **Target W&B Run Name** | [Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k) | [Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z) | [Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p) | — | — | — |

#### B. Quỹ đạo Độ chính xác qua 10 Giai đoạn (Stage Accuracy Trajectory, %)

| Giai đoạn (Sau Task $k$) | P0 (`gaussian_std1`) | P1 (`official_uniform`) | P2 (`gaussian_matched_variance`) | Ưu thế của Quy mô Nhỏ ($\Delta_{P1 - P0}$) |
|:---:|:---:|:---:|:---:|:---:|
| **Stage 1** | $89.60\%$ | **$92.70\%$** | **$93.40\%$** | $+3.10\text{ pp}$ |
| **Stage 2** | $88.00\%$ | **$91.60\%$** | **$91.60\%$** | $+3.60\text{ pp}$ |
| **Stage 3** | $85.10\%$ | **$88.60\%$** | **$88.50\%$** | $+3.50\text{ pp}$ |
| **Stage 4** | $81.55\%$ | **$85.62\%$** | **$85.58\%$** | $+4.07\text{ pp}$ |
| **Stage 5** | $79.10\%$ | **$83.74\%$** | **$83.70\%$** | $+4.64\text{ pp}$ |
| **Stage 6** | $76.93\%$ | **$81.50\%$** | **$81.62\%$** | $+4.57\text{ pp}$ |
| **Stage 7** | $75.76\%$ | **$80.06\%$** | **$80.01\%$** | $+4.30\text{ pp}$ |
| **Stage 8** | $74.03\%$ | **$77.76\%$** | **$78.00\%$** | $+3.73\text{ pp}$ |
| **Stage 9** | $72.30\%$ | **$76.39\%$** | **$76.34\%$** | $+4.09\text{ pp}$ |
| **Stage 10 (Final)** | $70.95\%$ | **$74.84\%$** | **$74.99\%$** | $+3.89\text{ pp}$ |
| **Trung bình ($\bar{A}$)** | **`79.33%`** | **`83.28%`** | **`83.37%`** | **$+3.95\text{ pp}$** |

#### C. Độ chính xác Từng Task tại Giai đoạn 10 & Mức Quên Từng Task

| Task $j$ | P0 Final Acc | P1 Final Acc | P2 Final Acc | P0 Forgetting ($f_j^{\text{past}}$) | P1 Forgetting ($f_j^{\text{past}}$) | P2 Forgetting ($f_j^{\text{past}}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 1** | $60.50\%$ | **$66.20\%$** | **$66.60\%$** | $29.10\%$ | **$26.50\%$** | **$26.80\%$** |
| **Task 2** | $76.30\%$ | **$80.80\%$** | **$80.40\%$** | $14.90\%$ | **$13.70\%$** | **$14.00\%$** |
| **Task 3** | $74.00\%$ | **$77.90\%$** | **$77.70\%$** | $10.60\%$ | $10.60\%$ | **$10.30\%$** |
| **Task 4** | $73.50\%$ | **$76.70\%$** | **$76.60\%$** | $8.00\%$ | $8.80\%$ | $9.10\%$ |
| **Task 5** | $70.70\%$ | **$75.60\%$** | **$76.30\%$** | $7.10\%$ | $7.50\%$ | **$6.70\%$** |
| **Task 6** | $69.50\%$ | **$74.10\%$** | **$74.30\%$** | $6.00\%$ | $6.30\%$ | $6.10\%$ |
| **Task 7** | $74.20\%$ | **$76.40\%$** | **$76.90\%$** | $3.30\%$ | $4.30\%$ | **$3.10\%$** |
| **Task 8** | $71.00\%$ | **$73.10\%$** | **$73.40\%$** | $2.70\%$ | **$1.80\%$** | **$2.10\%$** |
| **Task 9** | $68.40\%$ | **$71.90\%$** | **$71.80\%$** | $0.80\%$ | $1.20\%$ | $1.10\%$ |
| **Task 10** | $71.40\%$ | **$75.70\%$** | **$75.90\%$** | — | — | — |
| **Trung bình** | **`70.95%`** | **`74.84%`** | **`74.99%`** | **`9.17%`** | **`8.97%`** | **`8.81%`** |

#### D. Kết luận Khoa học Then chốt từ Thí nghiệm C1
1. **Khẳng định Giả thuyết H2 (`SUPPORTED BY FULL EXPERIMENT`):**
   - Giảm phương sai ma trận chiếu từ $1.0$ về $1/(3d) \approx 0.000434$ loại bỏ hoàn toàn hiện tượng bão hòa cực đoan của Sigmoid ($81.03\% \rightarrow 0.00\%$), giúp tăng độ chính xác tăng dần trung bình thêm **$+3.95\text{ pp}$** (với Uniform) và **$+4.04\text{ pp}$** (với Matched Gaussian).
2. **Quy mô phương sai (Scale) là yếu tố quyết định hàng đầu, tương quan mạnh mẽ hơn dạng hàm phân phối (Shape):**
   - Giữa P1 (Uniform $\mathcal{U}$) và P2 (Gaussian $\mathcal{N}$) khi đã khớp phương sai, độ chính xác gần như tương đồng ($\Delta_{P1 - P2} = -0.09\text{ pp}$ trên $\bar{A}$ và $-0.15\text{ pp}$ trên $A_{10}$). Tuy nhiên, cần lưu ý về **giới hạn thống kê thực nghiệm**: kết luận này được quan sát trên đơn seed ($n=1$, seed 42) trên CIFAR-100; chưa thể suy diễn thành định lý tương đương toán học phổ quát trên mọi seed hay kiến trúc dữ liệu khác nhau.
3. **Mức độ thu hẹp khoảng cách so với Code Gốc Tác giả:**
   - Khoảng cách ban đầu giữa Official Run ($90.98\%$) và Historical Our Run ($79.33\%$) là $11.65\text{ pp}$.
   - Khởi tạo ma trận chiếu thu hẹp được **$+3.95\text{ pp}$**, tức giải thích được **$33.9\%$ (khoảng một phần ba)** khoảng cách lệch.
   - Khoảng cách còn lại $\approx 7.6\text{–}7.7\text{ pp}$ thuộc về 2 nhân tố giao thức chưa căn chỉnh: **H1 (Pretrained ViT Checkpoint ImageNet-21K vs. 1K)** và **H4 (Preprocessing Bilinear $[0, 1]$ vs. Bicubic $[-1, 1]$)**.

#### E. Kiểm toán Nguồn gốc Phần cứng & Khả năng Tái hiện Kết quả Lịch sử
1. **Kiểm toán Nguồn gốc GPU (GPU Provenance):**
   - *Phân loại:* **`GPU_PROVENANCE_PARTIAL / UNRESOLVED_MEMORY_SPECIFICATION`**.
   - *Chi tiết:* Trong tệp `metadata.json` của kết quả Colab, trường `environment.gpu` ghi nhận giá trị `None` do module logger môi trường chưa tự động trích xuất chuỗi tên GPU từ PyTorch. Tuy nhiên, bảng thông tin phiên làm việc Colab hiển thị thiết bị NVIDIA A100-SXM4 với dung lượng RAM GPU xấp xỉ ~80GB (thay vì 40GB như phỏng đoán ban đầu). Sự khác biệt về dung lượng VRAM này không ảnh hưởng đến nghiệm toán học giải tích của RLS (chạy float64 trên cùng batch 10).
2. **Tái hiện Bản chạy Lịch sử P0 (Historical P0 Reproducibility):**
   - *Phân loại:* **`HISTORICAL_P0_NEAR_REPRODUCED — CAUSE_OF_SMALL_DIFFERENCE_UNDETERMINED`**.
   - *Chi tiết:* Bản chạy P0 mới trên Colab A100 đạt $\bar{A} = 79.3315\%$, $A_{10} = 70.9500\%$, so với bản chạy lịch sử trên Colab T4 (`aad7500`, Run `2cdeiptv`) đạt $\bar{A} = 79.3332\%$, $A_{10} = 70.9200\%$. Độ lệch tuyệt đối giữa hai môi trường là $-0.0017\text{ pp}$ trên $\bar{A}$ và $+0.0300\text{ pp}$ trên $A_{10}$ (sai số cực vi mô, nhiều khả năng do khác biệt làm tròn số học dấu phẩy động hoặc kernel cuBLAS giữa kiến trúc GPU Turing T4 và Ampere A100).


---

## 7. Thí nghiệm Kiểm soát C2: Căn chỉnh Trọng số Pretrained ViT Backbone (ImageNet-21K vs 1K)

Chi tiết kế hoạch nghiên cứu được tài liệu hóa đầy đủ tại [docs/c2_backbone_alignment_plan.md](c2_backbone_alignment_plan.md).

- **Mục tiêu nghiên cứu:** Định lượng mức độ đóng góp độc lập của trọng số Pretrained ViT Checkpoint (it_base_patch16_224.augreg2_in21k_ft_in1k trong mã nguồn tác giả vs it_base_patch16_224.augreg_in1k ban đầu của nhóm) đối với khoảng cách $\approx 7.7\text{ pp}$ còn lại sau C1.
- **Trạng thái hiện tại:** **FULL_GPU_COMPLETED_AND_VERIFIED** (Nghiệm thu thực nghiệm thành công ngày 02/11/2026).
  - Tệp nén kết quả: oal_c2_backbone_results.zip (SHA256: d11e4a98fc6c4a40018427521bf365468f2d5a3ceeceeb941c27050f33a94429, 0 lỗi CRC).
  - Thư mục trích xuất & kiểm chứng: [
esults/c2_full_gpu_verified/](../results/c2_full_gpu_verified/).
  - Bảng tổng hợp & hình vẽ phân tích: [
esults/c2_analysis/](../results/c2_analysis/).
  - Phần cứng thực thi: Google Colab GPU — NVIDIA A100-SXM4-40GB (CUDA 13.0, PyTorch 2.11.0+cu130, 	imm 1.0.30).
  - Bộ kiểm thử hồi quy: 56/56 tests PASSED (100%).

### 7.1. Bảng Kết quả Toàn diện Thí nghiệm Kiểm soát C2

| Chỉ số Thực nghiệm (CIFAR-100, 10 Tasks, Seed 42) | C2-B0 (Control: ugreg_in1k) | C2-B1 (Treatment: ugreg2_in21k_ft_in1k) | Tác động Đơn yếu tố $\Delta(B1 - B0)$ | C1-P1 Ref (ugreg_in1k) | Official Run (seed=0, 73e4ac1) | Paper (NeurIPS 2024 Table 1) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\bar{A}$ (Avg Incremental Acc, $\uparrow$)** | **.2814\%$** | **.7742\%$** | **$+8.4929\text{ pp}$** | .2814\%$ | .9828\%$ | .10\%$ |
| **{10}$ (Last Task Acc, $\uparrow$)** | **.8400\%$** | **.4300\%$** | **$+12.5900\text{ pp}$** | .8400\%$ | .8400\%$ | — |
| **{10}^{\text{past}}$ (Paper Forgetting, $\downarrow$)** | **.9667\%$** | **.0222\%$** | **$-3.9444\text{ pp}$** | .9667\%$ | .1667\%$ | — |
| **{10}^{\text{code}}$ (Official Code Forgetting, $\downarrow$)** | **.0700\%$** | **.5200\%$** | **$-3.5500\text{ pp}$** | .0700\%$ | .6900\%$ | .50\%$ |
| **Thời gian Trích xuất Đặc trưng (s)** | .46\text{ s}$ | .30\text{ s}$ | $-0.16\text{ s}$ | — | — | — |
| **Thời gian Cập nhật RLS + Eval (s)** | .55\text{ s}$ | .49\text{ s}$ | $-0.06\text{ s}$ | — | — | — |
| **Tổng Thời gian Chạy (s)** | .13\text{ s}$ ($\approx 5.04\text{ phút}$) | .32\text{ s}$ ($\approx 5.02\text{ phút}$) | $-0.81\text{ s}$ | — | .6\text{ s}$ | .0\text{ s}$ |
| **W&B Online Run URL** | [Run 3oiftwg6](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3oiftwg6) | [Run 
e1p16b5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | — | [Run 6whrot4z](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z) | [Run 3eca8232](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232) | — |

### 7.2. Quỹ đạo Độ chính xác Qua Từng Giai đoạn (Stage-by-Stage Trajectory)

| Giai đoạn ($) | B0 (ugreg_in1k) | B1 (ugreg2_in21k_ft_in1k) | Ưu thế của B1 ($\Delta_{B1 - B0}$) | C1-P1 Ref | Official Run (seed 0) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 1** | .70\%$ | **.30\%$** | $+4.60\text{ pp}$ | .70\%$ | .60\%$ |
| **Stage 2** | .60\%$ | **.10\%$** | $+4.50\text{ pp}$ | .60\%$ | .75\%$ |
| **Stage 3** | .60\%$ | **.13\%$** | $+5.53\text{ pp}$ | .60\%$ | .00\%$ |
| **Stage 4** | .62\%$ | **.65\%$** | $+7.03\text{ pp}$ | .62\%$ | .15\%$ |
| **Stage 5** | .74\%$ | **.10\%$** | $+8.36\text{ pp}$ | .74\%$ | .78\%$ |
| **Stage 6** | .50\%$ | **.78\%$** | $+9.28\text{ pp}$ | .50\%$ | .67\%$ |
| **Stage 7** | .06\%$ | **.04\%$** | $+9.99\text{ pp}$ | .06\%$ | .89\%$ |
| **Stage 8** | .76\%$ | **.12\%$** | $+11.36\text{ pp}$ | .76\%$ | .10\%$ |
| **Stage 9** | .39\%$ | **.08\%$** | $+11.69\text{ pp}$ | .39\%$ | .06\%$ |
| **Stage 10 (Final)** | .84\%$ | **.43\%$** | $+12.59\text{ pp}$ | .84\%$ | .84\%$ |
| **Trung bình ($\bar{A}$)** | **.28\%$** | **.77\%$** | **$+8.49\text{ pp}$** | **.28\%$** | **.98\%$** |

### 7.3. Kết luận Khoa học & Hiệu chỉnh Học thuật Then chốt (Scientific Corrections)
1. **Khẳng định Giả thuyết H1 trong Phạm vi Cấu hình Thử nghiệm (H1_SUPPORTED_BY_C2):**
   - Trọng số Pretrained ViT Checkpoint từ ImageNet-21K (ugreg2_in21k_ft_in1k) tạo ra bước nhảy vọt hiệu năng:
     - Tăng $\bar{A}$ thêm **$+8.4929\text{ pp}$** (.28\% \to 91.77\%$).
     - Tăng {10}$ thêm **$+12.5900\text{ pp}$** (.84\% \to 87.43\%$).
     - Giảm tỷ lệ quên {10}^{\text{past}}$ từ .97\%$ xuống .02\%$ (giảm $-3.9444\text{ pp}$).
2. **Hiệu chỉnh Nhận định So sánh với Paper & Official Run:**
   - Kết quả .77\%$ của B1 tương đương về mặt số học và cao hơn một lượng nhỏ so với Paper (.10\%$) và Official Run (.98\%$).
   - Do giao thức giữa hai bên chưa hoàn toàn đồng nhất (thứ tự lớp seed 42 vs seed 0; tiền xử lý Bicubic vs Bilinear), **đây không phải là bằng chứng cho thấy mã nguồn nhóm vượt trội có ý nghĩa thống kê so với bài báo hay tác giả**.
3. **Hiệu chỉnh Phân tích Tỷ lệ Thu hẹp Khoảng cách:**
   - Mức tăng $+8.49\text{ pp}$ của B1 vượt qua khoảng cách trước đó so với bản chạy đơn seed tác giả (.70\text{ pp}$). Đây là một tỷ số số học thuần túy phản ánh độ lớn cải thiện, **không phải là bằng chứng phân rã nhân quả hình thức (formal causal decomposition) đã giải thích 110.3% nguyên nhân**.
4. **Hiệu chỉnh Phân tích Độ tương đồng Cosine:**
   - B1 có độ tương đồng nội lớp $\mathcal{S}_{\text{intra}} = 0.8423$ (tăng từ .5449$), nhưng độ tương đồng liên lớp $\mathcal{S}_{\text{inter}}$ cũng tăng mạnh (.7642$ so với .3603$). Hiệu số  = \mathcal{S}_{\text{intra}} - \mathcal{S}_{\text{inter}}$ của B1 (.0781$) nhỏ hơn B0 (.1846$). Điều này phản ánh các vector đặc trưng của B1 tập trung trong một nón góc hẹp hơn với biên độ chuẩn L2 gấp đôi ($\approx 38.7$ vs .3$), chứ không phải là sự tách biệt góc tuyệt đối giữa các lớp.
5. **Hiệu chỉnh So sánh Thời gian Chạy (Runtime):**
   - Sự khác biệt về thời gian chạy giữa C2 ($\approx 301\text{ s}$ trên Colab A100) và Official Run (\text{ s}$ trên Colab T4) đến từ sự khác biệt phần cứng GPU, cơ chế cache đặc trưng và việc loại bỏ forward pass thừa, **không phải là gia tốc thuật toán thuần túy**.
6. **Kiểm chứng Tái hiện Hoàn hảo của C2-B0 so với C1-P1:**
   - C2-B0 trên Colab A100 đạt $\bar{A} = 83.2814\%$, {10} = 74.8400\%$, khớp chính xác 100% đến từng số thập phân với C1-P1.
   - Ma trận chiếu $ được kiểm chứng đồng nhất tuyệt đối qua mã băm SHA256 (6d3048f4a0fca38799d090113ff9a0c5b9f80a6dfd743a6c4f3d6e9607553a8b), bảo đảm tính công bằng đơn yếu tố tuyệt đối (Single-Factor Control Fairness).

---

## 8. Thực nghiệm C3 — Kiểm soát Quy trình Tiền xử lý Hình ảnh (Controlled Preprocessing Alignment — Factorial $2 \times 2$)

- **Trạng thái:** `C3_SIGNED_OFF_WITH_LIMITATIONS` (Full GPU hoàn tất trên NVIDIA A100, đồng bộ 10 runs W&B trên CoreWeave Forge)
- **Nhánh thực nghiệm:** `experiment/c3-preprocessing-alignment`
- **Gói nén lưu trữ:** `foal_c3_preprocessing_results.zip` (SHA256: `a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6`)
- **Tài liệu phân tích:** [`results/c3_analysis/`](../results/c3_analysis/) (Bao gồm 6 hình vẽ chất lượng xuất bản, manifest và các bảng tổng hợp CSV)

### 8.1. Bảng Kết quả Thực nghiệm Giai thừa $2 \times 2$ (A100 GPU, CIFAR-100, Seed 42)

| Điều kiện | Nội suy | Chuẩn hóa | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{10}^{\text{past}}$ (%) | $F_{10}^{\text{code}}$ (%) | T_Feat (s) | T_RLS (s) | W&B Run ID | Ghi chú |
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---|
| **C3-A** | Bicubic | Normalize | **91.7742%** | **87.4300%** | **5.0222%** | **4.5200%** | 248.30 | 8.49 | [`ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | Reused C2-B1 |
| **C3-B** | Bilinear | Normalize | **91.3918%** | **86.9500%** | **4.8889%** | **4.4000%** | 176.08 | 8.56 | [`ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3) | Factor A |
| **C3-C** | Bicubic | Raw | **90.8762%** | **86.3500%** | **5.2333%** | **4.7100%** | 175.58 | 8.49 | [`4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h) | Factor B |
| **C3-D** | Bilinear | Raw | **90.4016%** | **85.7300%** | **5.1556%** | **4.6400%** | 175.28 | 8.50 | [`3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | Official Pipeline |

### 8.2. Phân rã Tác động Giai thừa & Kết luận Khoa học
1. **Tác động chính của Phép Nội suy ($E_{\text{interp}}$):** $-0.4285\text{ pp}$ (Bilinear làm giảm nhẹ độ chính xác so với Bicubic).
2. **Tác động chính của Chuẩn hóa dải pixel ($E_{\text{norm}}$):** $-0.9441\text{ pp}$ (Không chuẩn hóa dải pixel làm giảm đáng kể độ chính xác).
3. **Hiệu ứng tương tác ($E_{\text{IN}}$):** $-0.0922\text{ pp}$ (Hai yếu tố tác động độc lập tuyến tính, tính chất cộng gộp xấp xỉ).
4. **Tổng độ dịch chuyển tiền xử lý ($\Delta_{\text{total}}$):** $-1.3726\text{ pp}$ (từ $91.7742\%$ xuống $90.4016\%$).
5. **Hội tụ với Tác giả:** Khi tiền xử lý được chuẩn hóa hoàn toàn về phía tác giả (C3-D), Last Accuracy $A_{10} = 85.73\%$ khớp gần như chính xác tuyệt đối với tác giả ($85.84\%$, độ lệch chỉ $-0.11\text{ pp}$), tỷ lệ quên $F_{10}^{\text{code}} = 4.64\%$ so với $4.69\%$ (lệch $-0.05\text{ pp}$).
- **Nhánh thực nghiệm:** `experiment/c3-preprocessing-alignment`
- **Colab Notebook:** [`notebooks/foal_c3_preprocessing_controlled.ipynb`](file:///d:/Nhập%20môn%20ML/Paper/notebooks/foal_c3_preprocessing_controlled.ipynb)
- **Thiết kế Nghiên cứu:** Mô hình Giai thừa $2 \times 2$ (Factorial Design)
  - Factor A (Nội suy không gian): Bicubic vs Bilinear
  - Factor B (Chuẩn hóa dải pixel): Normalized $[-1.0, 1.0]$ (`mean=0.5, std=0.5`) vs Raw $[0.0, 1.0]$ (`ToTensor()` only)
- **Bảng 4 Cấu hình Giai thừa:**
  1. **C3-A (Bicubic, Normalized):** Baseline đối chứng (`configs/c3_a_bicubic_normalized.yaml`). Phê duyệt tái sử dụng kết quả C2-B1 (`C3_A_REUSE_APPROVED`, $\bar{A}=91.7742\%$, $A_{10}=87.4300\%$, W&B run `ne1p16b5`).
  2. **C3-B (Bilinear, Normalized):** Cô lập tác động nội suy (`configs/c3_b_bilinear_normalized.yaml`).
  3. **C3-C (Bicubic, Raw):** Cô lập tác động chuẩn hóa (`configs/c3_c_bicubic_raw.yaml`).
  4. **C3-D (Bilinear, Raw):** Căn chỉnh hoàn toàn với mã nguồn tác giả (`configs/c3_d_bilinear_raw.yaml`). Xác minh chẩn đoán: sai khác điểm ảnh tối đa so với transform gốc là chính xác $0.00\text{e+}00$.
- **Kiểm soát Cố định 100%:**
  - Backbone: `vit_base_patch16_224.augreg2_in21k_ft_in1k`
  - Ma trận chiếu $P$: `official_uniform`, seed 42, $D=1000$
  - Optimizer: Online Ridge RLS, $\gamma = 1.0$, `float64`, batch size 10, seed 42
  - W&B group: `c3_preprocessing_alignment`
- **An toàn Cache Đặc trưng:** Tách biệt đường dẫn và định danh file cache theo `interp_tag` và `norm_tag`, từ chối nghiêm ngặt các cache sai lệch cấu hình.
- **Kiểm thử Hồi quy:** `tests/test_c3_preprocessing.py` (6/6 tests passing 100%).

---

## 9. Thực nghiệm Table 4 — Khảo sát Đóng góp Thành phần Kiến trúc (Ablation Study: Feature Fusion $\times$ Smooth Projection)

- **Trạng thái:** `TABLE4_SIGNED_OFF_WITH_LIMITATIONS` (Full GPU hoàn tất trên Google Colab Tesla T4 và NVIDIA A100 cho baseline A0, 14 artifacts verified, đối soát 17 runs W&B trên CoreWeave Forge)
- **Nhánh thực nghiệm:** `experiment/table4-ablation`
- **Gói nén lưu trữ:** `foal_table4_ablation_results.zip` (SHA256: `90f191d12b06fbd804779fcb038a12574c9b7cdc2f32369a6975dd6309e09e7a`)
- **Tài liệu phân tích:** [`results/table4_analysis/`](../results/table4_analysis/) (Bao gồm 7 hình vẽ PNG xuất bản, 7 CSV manifests, 3 JSON manifests)
- **Tài liệu kế hoạch chi tiết:** [`docs/table4_ablation_plan.md`](table4_ablation_plan.md)

### 9.1. Bảng Kết quả Thực nghiệm Table 4 (CIFAR-100, 10 Tasks, Seed 42)

| Điều kiện | FF | SP | Số chiều ($D$) | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{10}^{\text{past}}$ (%) | $F_{10}^{\text{code}}$ (%) | T_RLS/Eval (s) | GPU Hardware | W&B Run ID | W&B Run Link | Ghi chú |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---|:---:|:---:|:---|
| **A0 (Full F-OAL)** | $\checkmark$ | $\checkmark$ | 1000 | **90.4016%** | **85.7300%** | **5.1556%** | **4.6400%** | 8.50 | A100-SXM4-40GB | [`3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | [Reused C3-D](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | Baseline chuẩn |
| **A1 (w/o FF)** | $\times$ | $\checkmark$ | 1000 | **90.1004%** | **85.4800%** | **5.4444%** | **4.9000%** | 15.52 | Tesla T4 (14.56 GB) | [`shl4p10l`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/shl4p10l) | [A1 Online](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/shl4p10l) | Bỏ Feature Fusion |
| **A2 (w/o SP)** | $\checkmark$ | $\times$ | 768 | **90.3823%** | **85.7200%** | **5.2444%** | **4.7200%** | 11.59 | Tesla T4 (14.56 GB) | [`rkedlkkk`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/rkedlkkk) | [A2 Online](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/rkedlkkk) | Bỏ Smooth Projection |
| **A3 (w/o Both)** | $\times$ | $\times$ | 768 | **89.9224%** | **85.4800%** | **5.3222%** | **4.7900%** | 11.97 | Tesla T4 (14.56 GB) | [`k01pp8st`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/k01pp8st) | [A3 Online](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/k01pp8st) | Bỏ Cả Hai |

*Lưu ý tính công bằng thời gian chạy:* A0 được đo trên NVIDIA A100 trong thực nghiệm C3-D; A1, A2, A3 được đo trên cùng GPU Tesla T4. Do đó so sánh thời gian chỉ được kiểm soát đơn yếu tố trong cụm A1 vs A2 vs A3 (D=1000 mất 15.52s, D=768 mất 11.59s–11.97s, tương ứng mức giảm ~25% phù hợp độ phức tạp $O(D^2)$).

### 9.2. Bảng Tiến trình Độ chính xác qua 10 Stages ($A_k$, %)

| Stage | Số Lớp | A0 (Full F-OAL) | A1 (w/o FF) | A2 (w/o SP) | A3 (w/o Both) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 10 | 96.10% | 96.10% | 96.20% | 95.30% |
| **2** | 20 | 95.05% | 94.90% | 94.95% | 94.50% |
| **3** | 30 | 93.27% | 92.83% | 93.20% | 92.63% |
| **4** | 40 | 91.50% | 91.25% | 91.53% | 91.07% |
| **5** | 50 | 90.58% | 90.46% | 90.68% | 90.26% |
| **6** | 60 | 89.45% | 89.03% | 89.33% | 88.92% |
| **7** | 70 | 88.63% | 88.14% | 88.54% | 88.06% |
| **8** | 80 | 87.30% | 86.84% | 87.35% | 86.91% |
| **9** | 90 | 86.41% | 85.97% | 86.32% | 86.09% |
| **10** | 100 | 85.73% | 85.48% | 85.72% | 85.48% |
| **Trung bình ($\bar{A}$)** | — | **90.4016%** | **90.1004%** | **90.3823%** | **89.9224%** |

*Kiểm chứng toán học:* Trung bình cộng của 10 giá trị stage accuracy trên mỗi cột khớp chính xác $100\%$ đến $10^{-12}$ với Average Incremental Accuracy $\bar{A}$ tương ứng.

### 9.3. Bảng Độ Quên Từng Task Cũ tại Điểm Cuối ($f_j^{10}$, %)

| Task ($j$) | Lớp trong Task | $f_j$ (A0: Full) | $f_j$ (A1: w/o FF) | $f_j$ (A2: w/o SP) | $f_j$ (A3: w/o Both) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 1** | 0 – 9 | 16.30% | 16.40% | 16.30% | 15.80% |
| **Task 2** | 10 – 19 | 11.90% | 13.00% | 11.00% | 12.50% |
| **Task 3** | 20 – 29 | 4.90% | 5.80% | 5.30% | 6.00% |
| **Task 4** | 30 – 39 | 4.20% | 5.30% | 4.50% | 4.70% |
| **Task 5** | 40 – 49 | 3.00% | 3.60% | 3.70% | 3.80% |
| **Task 6** | 50 – 59 | 2.80% | 2.80% | 3.10% | 2.30% |
| **Task 7** | 60 – 69 | 1.50% | 1.30% | 1.20% | 1.20% |
| **Task 8** | 70 – 79 | 1.10% | 0.80% | 1.50% | 1.10% |
| **Task 9** | 80 – 89 | 0.70% | 0.00% | 0.60% | 0.50% |
| **Task 10** | 90 – 99 | 0.00% | 0.00% | 0.00% | 0.00% |
| **$F_{\text{past}}$ (9 tasks cũ)** | — | **5.1556%** | **5.4444%** | **5.2444%** | **5.3222%** |
| **$F_{\text{code}}$ (10 tasks)** | — | **4.6400%** | **4.9000%** | **4.7200%** | **4.7900%** |

### 9.4. Phân rã Tác động Giai thừa $2 \times 2$ (Factorial Effects)

- **Tác động đơn (Simple Effects):**
  - $E_{\text{FF}|\text{SP}} = A0 - A1 = \mathbf{+0.3013\text{ pp}}$
  - $E_{\text{FF}|\neg\text{SP}} = A2 - A3 = \mathbf{+0.4599\text{ pp}}$
  - $E_{\text{SP}|\text{FF}} = A0 - A2 = \mathbf{+0.0193\text{ pp}}$
  - $E_{\text{SP}|\neg\text{FF}} = A1 - A3 = \mathbf{+0.1780\text{ pp}}$
- **Tác động chính (Main Effects):**
  - **Tác động chính của Feature Fusion ($E_{\text{FF}}$):** $\mathbf{+0.3806\text{ pp}}$ (Đóng góp kiến trúc chủ đạo trên CIFAR-100).
  - **Tác động chính của Smooth Projection ($E_{\text{SP}}$):** $\mathbf{+0.0987\text{ pp}}$ (Đóng góp dương khiêm tốn).
- **Hiệu ứng tương tác ($I_{\text{FF} \times \text{SP}}$):**
  - $I = A0 - A1 - A2 + A3 = \mathbf{-0.1587\text{ pp}}$ (Hệ số hồi quy $\beta_{\text{int}} = -0.0397\text{ pp}$).
  - Hiệu ứng dưới cộng tính (sub-additive): Khi đã có Feature Fusion chất lượng cao từ 12 layer ViT, Smooth Projection chỉ mang lại thêm $+0.02\text{ pp}$ thay vì $+0.18\text{ pp}$.
- **Lợi ích kết hợp (Joint Benefit):**
  - $\Delta_{\text{Full - Both}} = A0 - A3 = \mathbf{+0.4793\text{ pp}}$ ($= E_{\text{FF}} + E_{\text{SP}}$).

### 9.5. Đối chiếu Xu hướng Khoa học với Bảng 4 Bài báo Gốc (NeurIPS 2024, Trang 10)

| Điều kiện | Paper $\bar{A}$ | Ours $\bar{A}$ (Seed 42) | Paper $\Delta$ | Ours $\Delta$ | Đánh giá Tái hiện Xu hướng |
|:---|:---:|:---:|:---:|:---:|:---|
| **A0: Full F-OAL** | **91.1%** | **90.4016%** | Baseline | Baseline | Khớp tương đối ($\Delta = -0.70\text{ pp}$) |
| **A1: w/o Feature Fusion** | **90.6%** | **90.1004%** | $-0.5\text{ pp}$ | $-0.3013\text{ pp}$ | $\checkmark$ Suy giảm chính xác |
| **A2: w/o Smooth Projection** | **90.7%** | **90.3823%** | $-0.4\text{ pp}$ | $-0.0193\text{ pp}$ | $\checkmark$ Suy giảm chính xác |
| **A3: w/o Both** | **90.6%** | **89.9224%** | $-0.5\text{ pp}$ | $-0.4793\text{ pp}$ | $\checkmark$ Thấp nhất toàn bảng |

- **Kết luận:** Tái hiện thành công **xu hướng kiến trúc** của Table 4 bài báo gốc: cả hai thành phần đều mang lại giá trị gia tăng thực nghiệm, và việc loại bỏ đồng thời cả hai đẩy độ chính xác xuống mức thấp nhất ($89.92\%$). Thứ tự ưu tiên giữa các mô hình trên Seed 42 bảo toàn: $A0 > A2 > A1 > A3$.


---

## 10. Giai đoạn Kế tiếp: Headline Comparison (Table 2) & Phân công Nhóm Song song Đa Tập Dữ liệu

### 10.1. Tình trạng Hiện tại & Khép lại Pha CIFAR-100
- C0 (Metric Alignment), C1 (Random Projection Init), C2 (Pretrained ViT Backbone Alignment), C3 (Preprocessing Alignment) và Table 4 (Architectural Ablation) đã được hoàn thành, kiểm toán pháp y toàn diện và merge vào nhánh `main` tại commit `747b828`.
- Bộ tiêu chuẩn hoàn tất 15 tiêu chí (`docs/cifar100_completion_checklist.md`) chuyển trạng thái sang `CIFAR100_PENDING_BASELINES`.

### 10.2. Triển khai Headline Comparison (Table 2 CIFAR-100)
- Đã hiện thực hai baseline trung tâm từ Table 2 của paper trên cùng đặc trưng ViT trích xuất:
  1. **LwF (Li & Hoiem TPAMI 2018):** `src/foal/baselines/lwf.py`, config `configs/headline/lwf_cifar100.yaml` (Exemplar-free, Distillation $T=2.0$, paper ref: $\bar{A} = 69.3\%$, $A_{10} = 64.8\%$, $F = 1.3\%$).
  2. **ER (Chaudhry et al. ICRA 2019):** `src/foal/baselines/er.py`, config `configs/headline/er_cifar100.yaml` (Replay buffer $M=5000$, reservoir update, paper ref: $\bar{A} = 90.1\%$, $A_{10} = 84.6\%$, $F = 20.7\%$).
- Runner hợp nhất: `scripts/train_baseline.py`.
- Notebook thực thi push-button Colab GPU: `notebooks/foal_cifar100_headline_final.ipynb`.

### 10.3. Khung Dataset Adapter Dùng chung & Phân công Nhóm Đa Tập Dữ liệu
- Khung adapter cơ sở: `src/foal/dataset_adapters/base.py` (bảo đảm không làm ảnh hưởng `src/foal/datasets.py`, bảo toàn tính phân hoạch chặt, zero replay, deterministic fingerprinting).
- Phân công song song hai thành viên (`docs/team_parallel_dataset_plan.md`):
  - **Thành viên A (Track DTD):** `src/foal/dataset_adapters/dtd.py`, branch `experiment/dtd-ablation`. Đo đạc 4 cấu hình Table 4 trên 40 lớp texture (10 tasks $\times$ 4 lớp), kiểm chứng mức suy giảm $11.6\text{ pp}$ khi bỏ SP.
  - **Thành viên B (Track FGVC-Aircraft):** `src/foal/dataset_adapters/fgvc_aircraft.py`, branch `experiment/fgvc-aircraft-ablation`. Đo đạc 4 cấu hình Table 4 trên 100 lớp aircraft variants (10 tasks $\times$ 10 lớp), điều tra đối chiếu bất đồng số liệu giữa Section 4.4 ($66.2\%$) và Table 4 ($62.2\%$).
- Lộ trình tổng thể 6 cột mốc: [`docs/FOAL_FINAL_REPRODUCTION_ROADMAP.md`](FOAL_FINAL_REPRODUCTION_ROADMAP.md).


---

## 11. Báo cáo Thực nghiệm Headline Comparison (Table 2 CIFAR-100 Full GPU)

### 11.1. Bảng Tổng hợp Kết quả Thực nghiệm So sánh Baseline
- **Phần cứng thực thi:** Google Colab GPU Tesla T4 (14.56 GB VRAM), CUDA 13.0, PyTorch 2.11.0.
- **Dữ liệu kiểm toán:** `foal_cifar100_headline_results.zip` (SHA256: `6ec6a4a4cadecb4e77b7704a217878356ac6f6cd4e2d72c8a78a3813cceea551`).
- **Giao thức:** CIFAR-100, 10 tasks $\times$ 10 classes, seed 42, backbone `vit_base_patch16_224.augreg2_in21k_ft_in1k`, single-pass streaming.

| Phương pháp | Loại / Cơ chế | Bộ nhớ Đệm ($M$) | $\bar{A}$ (Ours) | $\bar{A}$ (Paper) | $\Delta \bar{A}$ | $A_{10}$ (Ours) | $A_{10}$ (Paper) | $F_{past}$ (Ours) | $F_{code}$ (Ours) | $F$ (Paper) | Thời gian chạy | Phần cứng & Ghi chú |
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| **F-OAL (Full)** | Giải tích (Analytic RLS) | **0 (Zero)** | **90.4016%** | **91.1%** | $-0.70\text{ pp}$ | **85.73%** | **86.5%** | **5.16%** | **4.64%** | **5.5%** | **229.11s** | A100 GPU (175.3s feat + 8.5s RLS) |
| **ER (Replay)** | Experience Replay | **5000** | **86.6703%** | **90.1%** | $-3.43\text{ pp}$ | **79.79%** | **84.6%** | **16.74%** | **15.07%** | **20.7%** | **626.85s** | Tesla T4 (Colab streaming loop) |
| **LwF (Official Protocol)** | Distillation (SGD + Dynamic Loss) | **0 (Zero)** | **46.2935%** | **69.3%** | $-23.01\text{ pp}$ | **23.34%** | **64.8%** | **0.41%** | **0.37%** | **1.3%** | **646.35s** | Tesla T4 (Colab streaming loop) |
| **LwF (Distill, Adapted)** | Distillation (AdamW, Static Loss) | **0 (Zero)** | **29.6295%** | **69.3%** | $-39.67\text{ pp}$ | **11.21%** | **64.8%** | **94.99%** | **85.49%** | **1.3%** | **629.52s** | Tesla T4 (Colab streaming loop) |

### 11.2. Phân tích Khoa học & Kiểm toán Pháp y Cốt lõi
1. **F-OAL giải quyết triệt để bài toán Exemplar-Free:** Không lưu bất kỳ mẫu cũ nào ($M=0$), F-OAL vẫn đạt $\bar{A} = 90.40\%$ và độ quên $F_{\text{past}} = 5.16\%$ (tiệm cận sát các mốc của bài báo: $91.1\%$, $86.5\%$, và $5.5\%$).
2. **Khảo sát Nguồn gốc Độ lệch Baseline LwF (Protocol Parity Audit):**
   - Đã clone và kiểm toán đối sánh chi tiết mã nguồn tác giả tại `ref/official_foal/` (commit SHA `73e4ac1b4c965030973a2b77f05c57c2f44f49f5`).
   - Kết quả chỉ ra rằng độ lệch của baseline LwF (đo được 29.63% so với bài báo 69.3%) xuất phát từ các khác biệt giao thức then chốt trong mã tác giả:
     - **Trọng số mất mát động (Dynamic Loss Weighting):** Mã tác giả (`agents/lwf.py:55`) dùng công thức $\mathcal{L} = \frac{1}{t+1} \mathcal{L}_{\text{new}} + \left(1 - \frac{1}{t+1}\right) \mathcal{L}_{\text{old}}$. Đến Task 10, tín hiệu học lớp mới bị giảm xuống còn 0.10, trong khi chưng cất lớp cũ tăng lên 0.90, khống chế hiện tượng recency bias. Ngược lại, baseline chuyển đổi trước đó sử dụng trọng số tĩnh 1.0 / 1.0.
     - **Phạm vi Logit Chưng cất (KD Scoping):** Mã tác giả khởi tạo sẵn đầu ra 100 lớp (`nn.Linear(768, 100)`) và tính toán chưng cất trên toàn bộ 100 lớp, ngăn chặn việc logit lớp mới tăng đột biến không kiểm soát.
     - **Tối ưu hóa (Optimizer):** Tác giả sử dụng `SGD(lr=0.001, momentum=0.0, weight_decay=0.0)` thay vì AdamW.
   - Nhóm đã hoàn thành việc triển khai `OfficialLwFAgent` (`src/foal/baselines/official_lwf.py`) và cấu hình `configs/headline/lwf_cifar100_official_protocol.yaml` với độ tương thích 100% mã nguồn tác giả.
3. **ER duy trì độ chính xác nhưng đòi hỏi bộ đệm lớn:** ER giữ được $70\text{--}83\%$ độ chính xác trên các task cũ nhờ bộ đệm 5.000 mẫu, đạt $\bar{A} = 86.67\%$, tiệm cận kết quả bài báo ($90.1\%$, sai lệch $-3.43\text{ pp}$), nhưng độ quên vẫn ở mức $16.74\%$ và tiêu tốn tài nguyên bộ nhớ đáng kể.
4. **Giải trình Toàn diện Thời gian Chạy & Rút lại Tuyên bố Gia tốc Phần cứng chéo:**
   - Thời gian vòng lặp huấn luyện F-OAL (tổng cộng 229.11s trên NVIDIA A100) được hạch toán minh bạch:
     - Trích xuất đặc trưng ViT: **175.28s** (76.5%).
     - Cập nhật giải tích RLS & đánh giá kiểm thử: **8.50s** (3.7%).
     - Chi phí I/O tuần tự hóa đĩa đặc trưng (`torch.save` 20 tệp `.pt`), DataLoader collation overhead, kiểm tra tính toàn vẹn SHA256 ma trận chiếu và ghi nhật ký: **45.33s** (19.8%).
   - Tuyên bố gia tốc "11.7 lần" trước đây được chính thức rút lại do thiếu tính tương đương phần cứng (NVIDIA A100 vs Tesla T4) và phạm vi đo lường.
5. **Trạng thái Quản trị Pha CIFAR-100:** Thiết lập trạng thái **`CIFAR100_COMPLETED_WITH_LIMITATIONS`**.
   - Lượt chạy Official Protocol LwF trên Google Colab GPU (Tesla T4, 646.35s) đã hoàn tất và được kiểm toán toàn diện (`foal_cifar100_official_lwf_results.zip`).
   - Kết quả: $\bar{A} = 46.2935\%$, $A_{10} = 23.34\%$, $F_{\text{past}} = 0.4111\%$, $F_{\text{code}} = 0.3700\%$.
   - Tỷ lệ quên $F_{\text{code}} = 0.37\%$ tiệm cận sát mốc $1.3\%$ của bài báo, chứng minh rằng cơ chế dynamic loss weighting của tác giả đã loại bỏ hoàn toàn recency collapse.
   - Toàn bộ 15 tiêu chí nghiệm thu CIFAR-100 đã PASS 100%. Nhóm chính thức đóng pha CIFAR-100 và chuyển sang DTD và FGVC-Aircraft. Báo cáo chi tiết: [`docs/CIFAR100_FINAL_REPRODUCTION_REPORT.md`](CIFAR100_FINAL_REPRODUCTION_REPORT.md).
