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
* **Hơn 14 phương pháp so sánh (Table 2):** ER, iCaRL, GDumb, DER++, Co2L, DVC, OCM, OnPro, SCR, MEMO, EASE, SLCA, ACIL, DS-AL, F-OAL (chạy 3 lần cho mỗi phương pháp).
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
| **Tỷ lệ bão hòa cực đoan ($z < 0.01 \lor z > 0.99$)** | **`81.03%`** | **`0.00%`** | **`0.00%`** | **81.03% chiều đặc trưng của P0 bị triệt tiêu đạo hàm và mất phân biệt!** |
| **Đạo hàm trung bình Sigmoid $\phi(1-\phi)$** | $0.0469$ | $0.2458$ | $0.2457$ | P1/P2 duy trì vùng tuyến tính nhạy cảm nhất ($\approx 0.25$) |
| **Số điều kiện Gram chính quy hóa $\text{Cond}(G + \gamma I)$** | $65,218.28$ | $50,467.89$ | $50,954.65$ | Gram matrix của P1/P2 ổn định hơn, số điều kiện nhỏ hơn |
| **Effective Rank $r_{\text{eff}}(G)$** | $8.60$ | $1.20$ | $1.20$ | P1 và P2 phân bổ phổ trị riêng tập trung hơn |

> **Phát hiện Khoa học then chốt:** Thí nghiệm chẩn đoán matched-input khẳng định chắc chắn 100% rằng ma trận chiếu $\mathcal{N}(0, 1)$ trong code ban đầu của nhóm đã đẩy $81.03\%$ kích hoạt phi tuyến vào vùng bão hòa chết của hàm Sigmoid. Trong khi đó, khởi tạo $\mathcal{U}(-1/\sqrt{d}, +1/\sqrt{d})$ của tác giả giữ $100\%$ kích hoạt trong vùng tuyến tính mượt mà ($0.00\%$ bão hòa).

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
| **Target W&B Run Name** | `c1_cifar100_gaussian_std1_seed42` | `c1_cifar100_official_uniform_seed42` | `c1_cifar100_gaussian_matched_variance_seed42` | — | — | — |

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
2. **Quy mô phương sai (Scale) là yếu tố quyết định, không phải dạng hàm phân phối (Shape):**
   - Giữa P1 (Uniform $\mathcal{U}$) và P2 (Gaussian $\mathcal{N}$) khi đã khớp phương sai, độ chính xác gần như trùng khớp hoàn toàn ($\Delta_{P1 - P2} = -0.09\text{ pp}$ trên $\bar{A}$ và $-0.15\text{ pp}$ trên $A_{10}$). Điều này khẳng định dạng phân phối không có ý nghĩa phân biệt, mà bản chất là độ lớn phương sai giữ cho đầu vào nằm trong miền dốc tối ưu của hàm Sigmoid.
3. **Mức độ thu hẹp khoảng cách so với Code Gốc Tác giả:**
   - Khoảng cách ban đầu giữa Official Run ($90.98\%$) và Historical Our Run ($79.33\%$) là $11.65\text{ pp}$.
   - Khởi tạo ma trận chiếu thu hẹp được **$+3.95\text{ pp}$**, tức giải thích được **$33.9\%$ (khoảng một phần ba)** khoảng cách lệch.
   - Khoảng cách còn lại $\approx 7.6\text{–}7.7\text{ pp}$ thuộc về 2 nhân tố giao thức chưa căn chỉnh: **H1 (Pretrained ViT Checkpoint ImageNet-21K vs. 1K)** và **H4 (Preprocessing Bilinear $[0, 1]$ vs. Bicubic $[-1, 1]$)**.
