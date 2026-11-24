# BÁO CÁO THỰC NGHIỆM CHỨNG MINH ĐIỂM YẾU & GIẢI PHÁP KHẮC PHỤC (LAB 2: EMPIRICAL DEMONSTRATION)

> **Học phần:** Nhập môn Học máy (*Introduction to Machine Learning*) — Trường ĐHKHTN, ĐHQG-HCM (HCMUS)
> **Liên kết lý thuyết:** Kiểm định thực nghiệm trực tiếp các Giả thuyết bác bỏ được **`H1`**, **`H2a`**, **`H2b`** và hai giải pháp một lượt không lưu mẫu cũ **`FD1` (`BalancedOnlineRidgeRLS`)**, **`FD2` (`Trace-Scaled Adaptive Ridge`)** từ [`docs/CRITICAL_ANALYSIS_REPORT.md`](CRITICAL_ANALYSIS_REPORT.md).
> **Mã nguồn thực thi:** [`src/foal/critical_analysis.py`](../src/foal/critical_analysis.py), [`scripts/run_lab2_longtail.py`](../scripts/run_lab2_longtail.py), [`scripts/run_lab2_corruptions.py`](../scripts/run_lab2_corruptions.py), [`scripts/run_lab2_scaling_sensitivity.py`](../scripts/run_lab2_scaling_sensitivity.py), [`scripts/generate_lab2_figures.py`](../scripts/generate_lab2_figures.py), [`notebooks/foal_lab2_empirical_demonstration.ipynb`](../notebooks/foal_lab2_empirical_demonstration.ipynb).
> **Bộ kiểm thử tự động:** [`tests/test_critical_analysis.py`](../tests/test_critical_analysis.py) (`4 passed`).

---

## 1. Kết Quả Thực Nghiệm E1: Sụp Đổ Dưới Phân Phối Long-Tail & Khôi Phục Bằng `BalancedOnlineRidgeRLS` (Trục Poster 1)

### 1.1. Thiết lập thực nghiệm
* **Dữ liệu & Biểu diễn:** Toàn bộ `50,000` ảnh train và `10,000` ảnh test của **CIFAR-100** qua encoder đóng băng `vit_base_patch16_224.augreg2_in21k_ft_in1k` (`Feature Fusion = True`, `Smooth Projection D = 1000, official_uniform`, $\gamma = 1.0$).
* **Giao thức Long-Tail tách biệt thứ tự thời gian (`permuted_global`):** Áp dụng phân phối mũ $N_{(r)} = \max(1, \text{round}(500 \cdot \rho^{-r/99}))$ gán ngẫu nhiên tất định theo seed lên toàn bộ 100 lớp để mọi task từ Task 1 đến Task 10 đều chứa cả lớp giàu mẫu (Head), trung bình (Medium) và hiếm mẫu (Tail). Nhờ đó **loại trừ hoàn toàn nhiễu giao thoa từ thứ tự xuất hiện task**.
* **Độ lặp thống kê:** **3 seeds độc lập (`42, 123, 2026`)** × 6 mức $\rho \in \{1, 5, 10, 25, 50, 100\}$ × 3 phương pháp = **54 lượt huấn luyện OCIL 10-task đầy đủ**.

### 1.2. Bảng số liệu tổng hợp (`Mean ± Std`, `ddof=1`, $N = 3$ seeds)

| Hệ số lệch $\rho$ | Phương pháp | $\bar{A}$ 10 Tasks (%) | $A_{10}$ Cuối (%) | Head Acc (%) | Tail Acc (%) | Harmonic Mean (%) | Tương quan $r(\|\hat{\mathbf{w}}_c\|_2, N_c)$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1$** *(Cân bằng)* | Vanilla F-OAL *(Paper)* | $91.85 \pm 0.09$ | $87.41 \pm 0.21$ | $86.17 \pm 0.99$ | $87.48 \pm 0.33$ | $87.40 \pm 0.21$ | $+0.000$ |
| **$1$** *(Cân bằng)* | F-OAL + Norm Calib. *(FD1b)* | $91.72 \pm 0.07$ | $87.33 \pm 0.12$ | $86.07 \pm 0.64$ | $87.41 \pm 0.67$ | $87.33 \pm 0.12$ | $+0.000$ |
| **$1$** *(Cân bằng)* | **Balanced Woodbury *(FD1a+b)*** | $91.64 \pm 0.12$ | $87.18 \pm 0.08$ | $86.14 \pm 0.80$ | $86.94 \pm 0.44$ | $87.17 \pm 0.08$ | $+0.000$ |
| **$10$** | Vanilla F-OAL *(Paper)* | $87.30 \pm 0.19$ | $79.84 \pm 0.85$ | $91.37 \pm 0.68$ | $66.13 \pm 1.23$ | $78.29 \pm 0.87$ | **$+0.914 \pm 0.021$** |
| **$10$** | F-OAL + Norm Calib. *(FD1b)* | $89.78 \pm 0.06$ | $84.52 \pm 0.52$ | $90.00 \pm 0.35$ | $78.26 \pm 1.34$ | $84.18 \pm 0.57$ | $+0.914 \pm 0.021$ |
| **$10$** | **Balanced Woodbury *(FD1a+b)*** | **$90.50 \pm 0.06$** | **$85.83 \pm 0.23$** | $88.09 \pm 0.40$ | **$83.07 \pm 1.33$** | **$85.75 \pm 0.25$** | $+0.604 \pm 0.082$ |
| **$25$** | Vanilla F-OAL *(Paper)* | $82.79 \pm 0.23$ | $73.20 \pm 1.03$ | $91.73 \pm 0.70$ | $50.59 \pm 1.97$ | $68.89 \pm 1.48$ | **$+0.909 \pm 0.021$** |
| **$25$** | F-OAL + Norm Calib. *(FD1b)* | $87.83 \pm 0.13$ | $81.72 \pm 0.84$ | $91.01 \pm 0.50$ | $70.82 \pm 1.60$ | $80.84 \pm 0.94$ | $+0.909 \pm 0.021$ |
| **$25$** | **Balanced Woodbury *(FD1a+b)*** | **$89.50 \pm 0.09$** | **$84.62 \pm 0.67$** | $88.90 \pm 0.41$ | **$79.49 \pm 1.92$** | **$84.38 \pm 0.74$** | $+0.722 \pm 0.047$ |
| **$50$** | Vanilla F-OAL *(Paper)* | $77.97 \pm 0.63$ | $67.57 \pm 1.25$ | $91.70 \pm 0.69$ | **$37.93 \pm 2.61$** | $58.50 \pm 2.47$ | **$+0.899 \pm 0.020$** |
| **$50$** | F-OAL + Norm Calib. *(FD1b)* | $85.58 \pm 0.17$ | $78.67 \pm 0.68$ | $91.31 \pm 0.66$ | $63.69 \pm 0.44$ | $76.94 \pm 0.59$ | $+0.899 \pm 0.020$ |
| **$50$** | **Balanced Woodbury *(FD1a+b)*** | **$88.18 \pm 0.04$** | **$82.93 \pm 0.36$** | $89.41 \pm 0.54$ | **$74.79 \pm 1.38$** | **$82.45 \pm 0.44$** | $+0.739 \pm 0.056$ |
| **$100$** | Vanilla F-OAL *(Paper)* | $72.79 \pm 0.99$ | $62.22 \pm 1.23$ | $91.61 \pm 0.77$ | **$26.64 \pm 2.31$** | $45.98 \pm 2.86$ | **$+0.888 \pm 0.019$** |
| **$100$** | F-OAL + Norm Calib. *(FD1b)* | $82.44 \pm 0.23$ | $75.01 \pm 0.86$ | $91.48 \pm 0.72$ | $55.14 \pm 0.97$ | $71.90 \pm 0.96$ | $+0.888 \pm 0.019$ |
| **$100$** | **Balanced Woodbury *(FD1a+b)*** | **$86.11 \pm 0.17$** | **$80.82 \pm 0.58$** | $90.17 \pm 0.53$ | **$69.31 \pm 1.62$** | **$79.89 \pm 0.69$** | $+0.741 \pm 0.066$ |

### 1.3. Kết luận kiểm định Giả thuyết H1
1. **Bác bỏ tính tổng quát của Figure 3 bài báo gốc:** Ngay khi rời khỏi chế độ cân bằng $\rho = 1$, hệ số tương quan Pearson giữa chuẩn cột trọng số $\|\hat{\mathbf{w}}_c\|_2$ và cỡ mẫu $N_c$ của F-OAL gốc vọt lên **$r = +0.89 \text{ đến } +0.91$** (khớp chính xác định luật lý thuyết Eq. L1.2).
2. **Sụp đổ nghiêm trọng lớp hiếm dù không quên đặc trưng:** Ở $\rho = 100$, nhóm lớp giàu mẫu thậm chí tăng nhẹ lên $91.61\%$, nhưng nhóm 1/3 lớp ít mẫu (Tail) sụp đổ từ **$87.48\%$ xuống còn $26.64\%$ (`-60.84 pp`)**!
3. **Hiệu quả vượt trội của Giải pháp FD1 (0-replay, 1-pass):**
   - Chỉ riêng bước chuẩn hóa biên độ cột lúc suy luận (`FD1b`, **0 FLOPs huấn luyện thêm**) đã kéo Tail Accuracy tăng **`+28.50 pp`** (`26.64% -> 55.14%`).
   - Kết hợp cập nhật Woodbury trọng số tần suất trực tuyến nhân quả (`BalancedOnlineRidgeRLS` — `FD1a+b`) khôi phục Tail Accuracy tới **`+42.67 pp`** (`26.64% -> 69.31%`), nâng $A_{10}$ thêm **`+18.60 pp`** (`62.22% -> 80.82%`) và $\bar{A}$ thêm **`+13.32 pp`** (`72.79% -> 86.11%`) mà không lưu bất kỳ mẫu cũ nào!

---

## 2. Kết Quả Thực Nghiệm E3: Quy Luật Phức Tạp Bậc Hai $\Theta(D^2)$ & Bẫy Nội Suy Quá Tham Số Hóa $(D \times \gamma)$ (Trục Poster 2)

### 2.1. Kiểm chứng thực nghiệm quy luật $\Theta(D^2)$ (3 seeds `42, 123, 2026` trên 50,000 mẫu CIFAR-100)

| Chiều chiếu $D$ | Dung lượng ma trận $\mathbf{R}$ (`float64`) | Thời gian cập nhật RLS TB (s) | Độ trễ mỗi batch (`B=100`, ms) | $\bar{A}$ 10 Tasks (%) |
| :---: | :---: | :---: | :---: | :---: |
| `250` | `0.48 MB` | $4.90\text{ s}$ | $9.80\text{ ms}$ | $88.99 \pm 0.10$ |
| `500` | `1.91 MB` | $4.79\text{ s}$ | $9.58\text{ ms}$ | $90.88 \pm 0.09$ |
| `1,000` *(Paper)* | `7.63 MB` | $9.21\text{ s}$ | $18.42\text{ ms}$ | $91.85 \pm 0.09$ |
| `2,000` | `30.52 MB` | $27.80\text{ s}$ | $55.60\text{ ms}$ | $92.19 \pm 0.09$ |
| `4,000` | `122.07 MB` | $204.26\text{ s}$ | $408.53\text{ ms}$ | $92.43 \pm 0.06$ |
| `6,000` | `274.66 MB` | $327.73\text{ s}$ | $655.45\text{ ms}$ | $92.62 \pm 0.08$ |

* **Hệ số mũ thực nghiệm $\alpha$ trong $T_{\text{RLS}} \propto D^\alpha$ (với $D \ge 1{,}000$):** **$\alpha = 2.109$** (khớp sát quy luật lý thuyết bậc hai $\Theta(V D^2)$ với $\alpha = 2.000$ đã chứng minh tại Mục 2.4 [`docs/CRITICAL_ANALYSIS_REPORT.md`](CRITICAL_ANALYSIS_REPORT.md), bác bỏ công thức sai biến Eq. 14 của bài báo). Khi tăng $D$ từ `1,000` lên `6,000`, độ chính xác chỉ tăng thêm `+0.77 pp` nhưng thời gian RLS tăng gấp **`35.6 lần`** và dung lượng $\mathbf{R}$ tăng gấp **`36.0 lần`**.

### 2.2. Phát hiện thực nghiệm Đỉnh Nội Suy Double-Descent tại ngưỡng $D \approx V$ & Hiệu quả tự thích nghi của `FD2`

Đối chiếu trực tiếp chế độ giàu mẫu (**`native`, $V = 50{,}000$**) với chế độ ít mẫu tương đương DTD (**`small_sample_v1600`, $V = 1{,}600$**):

| Chiều chiếu $D$ | $\bar{A}$ tại $\gamma = 10^{-3}$ ($V=1{,}600$) | $A_{10}$ tại $\gamma = 10^{-3}$ ($V=1{,}600$) | Chuẩn $\|\hat{\mathbf{W}}\|_F$ tại $\gamma = 10^{-3}$ | $\bar{A}$ với $\gamma = 1$ cố định *(Paper)* | $\bar{A}$ với **`trace_auto` *(FD2)*** | $\gamma_{\text{auto}}$ được FD2 tự chọn |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `250` ($D \ll V$) | $82.38\%$ | $79.49\%$ | $12.88$ | **$85.25\%$** | $85.08\%$ | $0.28$ |
| `500` ($D < V$) | $80.72\%$ | $81.53\%$ | $22.99$ | **$87.26\%$** | $87.02\%$ | $0.55$ |
| `1,000` ($D \lesssim V$) | **$78.83\%$** | $78.98\%$ | **$85.89$** | $87.71\%$ | **$87.74\%$** | $1.12$ |
| **`2,000` ($D \approx V$)** | $80.22\%$ | **$72.64\%$** *(Sụp đổ!)* | **$154.98$** *(Nổ chuẩn 16x!)* | $87.52\%$ | **$88.05\%$** | **$2.82$** |
| **`4,000` ($D > 2V$)** | $82.65\%$ | $78.47\%$ | $77.81$ | $86.91\%$ | **$88.19\%$** | **$11.25$** |

* **Ý nghĩa khoa học đắt giá cho Poster A2:**
  1. Dữ liệu tái hiện **chính xác hiện tượng đỉnh suy biến Marchenko–Pastur Double-Descent**: khi $D$ băng qua cỡ mẫu $V = 1{,}600$ (tại $D = 2{,}000, \gamma = 10^{-3}$), các trị riêng sát $0$ khiến chuẩn trọng số $\|\hat{\mathbf{W}}\|_F$ nổ vọt lên **`154.98`** (gấp `16 lần` mức bình thường) và kéo $A_{10}$ sập xuống **`72.64%`** (`-10.59 pp`), trong khi ở chế độ $V = 50{,}000$ cùng cấu hình đó hoàn toàn miễn nhiễm ($\bar{A} = 92.15\%$).
  2. Công thức không tham số **`trace_auto` (FD2)** tự động nâng $\gamma_{\text{auto}}$ từ $0.28 \to 1.12 \to 2.82 \to 11.25$ theo chiều $D$ và tỷ số $D/V$, đạt **độ chính xác cao nhất toàn lưới** ở cả $D = 1{,}000, 2{,}000$ và $4{,}000$ (vượt cả $\gamma = 1$ cố định của bài báo tới **`+1.28 pp`** tại $D = 4{,}000$) mà **không cần bất kỳ tập dữ liệu validation nào**!

---

## 3. Bộ Biểu Đồ Chuẩn 300 DPI Đã Xuất Sẵn Cho Poster A2

Toàn bộ biểu đồ nằm tại [`results/lab2_empirical/figures/`](../results/lab2_empirical/figures/):
1. [`fig_e1_cifar100_longtail_curves.png`](../results/lab2_empirical/figures/fig_e1_cifar100_longtail_curves.png) — Đường suy giảm Few-Shot Tail Accuracy & $\bar{A}$ theo $\rho \in [1, 100]$ (`Mean ± Std`, 3 seeds) và mức phục hồi của FD1.
2. [`fig_e1_cifar100_weight_norm_refutation.png`](../results/lab2_empirical/figures/fig_e1_cifar100_weight_norm_refutation.png) — Phản bác trực tiếp Figure 3 bài báo gốc: sự sụp đổ đơn điệu của $\|\hat{\mathbf{w}}_c\|_2$ ở $\rho = 100$ đối chiếu với đường cân bằng phẳng của `BalancedOnlineRidgeRLS`.
3. [`fig_e3_cifar100_quadratic_scaling.png`](../results/lab2_empirical/figures/fig_e3_cifar100_quadratic_scaling.png) — Đường cong thực nghiệm $\alpha = 2.11$ bám khít định luật bậc hai $\Theta(D^2)$ về thời gian và bộ nhớ.
4. [`fig_e3_cifar100_overparam_heatmap.png`](../results/lab2_empirical/figures/fig_e3_cifar100_overparam_heatmap.png) — Đối sánh Heatmap $(D \times \gamma)$ giữa $V = 50{,}000$ và $V = 1{,}600$, hiển thị vùng sụp đổ quá tham số hóa và đường cứu nguy của FD2.

---

## 4. Hướng Dẫn Chạy Mở Rộng Cho DTD, FGVC-Aircraft, Nhiễu Ảnh E2 & Đồng Bộ W&B

Để hoàn tất nốt phần mở rộng trên GPU (Colab / Kaggle Tesla T4, tổng thời gian chạy **~6–8 phút**) và ghi nhận đầy đủ lên W&B với tag `lab2-empirical-demo`:
- Mở trực tiếp notebook [`notebooks/foal_lab2_empirical_demonstration.ipynb`](../notebooks/foal_lab2_empirical_demonstration.ipynb) trên Google Colab / Kaggle GPU, hoặc chạy qua CLI:
  ```bash
  # Đồng bộ E1 & E3 lên W&B (tự động gắn tags: ['lab2-empirical-demo', 'critical-analysis', ...])
  python -m scripts.run_lab2_longtail --dataset dtd --seeds 42 123 2026 --cache-dir ./lab2_cache --wandb
  python -m scripts.run_lab2_longtail --dataset aircraft --seeds 42 123 2026 --cache-dir ./lab2_cache --wandb
  python -m scripts.run_lab2_corruptions --dataset dtd --device cuda --wandb
  python -m scripts.run_lab2_corruptions --dataset aircraft --device cuda --wandb
  ```
