# F-OAL CIFAR-100 Multi-Seed Table 4 Ablation Study (Seeds 42, 123, 2026)

**Status:** `MULTISEED_READY_FOR_GPU` *(Seed 42 verified and reused; Seeds 123 and 2026 pipeline, notebook, statistical analyzer, and tests implemented and ready for Colab GPU execution)*  
**Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL)  
**Branch:** `experiment/cifar100-multiseed-ablation`  
**Target Benchmark:** CIFAR-100 (100 classes, 10 incremental tasks × 10 classes, 1-pass online analytic learning, zero replay)

---

## 1. Research Motivation

Trong giai đoạn tái lập trước đó, nhóm đã hoàn thành nghiên cứu Table 4 Ablation trên CIFAR-100 với một seed duy nhất (`seed=42`). Kết quả cho thấy:
- **Full F-OAL (FF On, SP On):** $\bar{A} = 90.401635\%$
- **Without FF (FF Off, SP On):** $\bar{A} = 90.100369\%$ ($\Delta_{\text{FF}} = +0.3013\text{ pp}$)
- **Without SP (FF On, SP Off):** $\bar{A} = 90.382341\%$ ($\Delta_{\text{SP}} = +0.0193\text{ pp}$)
- **Without FF & SP (FF Off, SP Off):** $\bar{A} = 89.922353\%$ ($\Delta_{\text{Both}} = +0.4793\text{ pp}$)

Tuy nhiên, với $n = 1$ seed, chúng ta chưa đủ cơ sở để kết luận:
1. Liệu độ chênh lệch nhỏ như $+0.0193\text{ pp}$ của Smooth Projection (SP) có nhất quán qua các phân hoạch tác vụ (class order) và ma trận chiếu ngẫu nhiên khác nhau hay chỉ là dao động ngẫu nhiên của `seed=42`.
2. Độ biến thiên giữa các seed ($\sigma_{\text{seed}}$) lớn hay nhỏ hơn so với kích thước hiệu ứng (effect size) của Feature Fusion (FF) và SP.
3. Mức độ ổn định của toàn bộ hệ thống F-OAL (Average Incremental Accuracy $\bar{A}$, Final Accuracy $A_{10}$, và Past Forgetting $F_{\text{past}}$) trên 3 seed độc lập: **42, 123, và 2026**.

---

## 2. Experimental Protocol (Fixed Parity with Verified Seed 42)

Toàn bộ 8 lượt chạy mới (`seed ∈ {123, 2026}` × `{full, no_ff, no_sp, no_ff_no_sp}`) khóa cứng cấu hình đã nghiệm thu của `C3-D / Table 4`:

| Tham số giao thức | Giá trị cố định |
| :--- | :--- |
| **Dataset & Split** | `uoft-cs/cifar100`, 100 lớp, 10 tasks × 10 lớp/task, toàn bộ 50,000 mẫu train & 10,000 mẫu test |
| **Learning Regime** | 1-pass online analytic learning (RLS), `replay_buffer = 0` |
| **Backbone** | `vit_base_patch16_224.augreg2_in21k_ft_in1k` (frozen) |
| **Preprocessing** | Bilinear resize ($224 \times 224$), Raw Tensor $[0, 1]$ (`normalize: false`) — **C3-D** |
| **Feature Fusion (FF)** | Trung bình CLS token qua toàn bộ 12 Transformer blocks + LayerNorm ($D = 768$) khi bật; CLS block 12 khi tắt |
| **Smooth Projection (SP)** | $D_p = 1000$, `official_uniform` $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$ + `sigmoid` khi bật; $D_p = 768$ trực tiếp khi tắt |
| **Online RLS** | `gamma = 1.0`, `rls_dtype = float64`, `batch_size = 10` |

> [!IMPORTANT]
> Tuyệt đối không chuyển sang cấu hình C3-A (Bicubic + Normalized) và không thay đổi backbone, preprocessing, RLS hyperparameter hoặc evaluation protocol trong quá trình mở rộng seed.

---

## 3. Seed Policy & Reproducibility Audit

### 3.1. Kiểm toán nguồn gốc Seed 42 (Đã xác minh tương thích 100%)

Kiểm toán tự động đối với 4 runs `seed=42` hiện có cho thấy toàn bộ 4 cấu hình chia sẻ chính xác cùng `task_split_fingerprint` (`7c461e9300daaa73e894d716782b0cb7d4eb922d4b2b3e35debfa2d61f67ca6f`) và `class_order_sha256` (`c64d29597444deef58361ab932aaacacd44e77b1f237e09ee9ad88521f335f31`):

| Cấu hình | Mã | Thư mục nguồn đã kiểm toán | W&B Run ID | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{\text{past}}$ (%) | Trạng thái |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Full F-OAL** | A0 | `results/c3_full_gpu_verified/.../c3_d_bilinear_raw` | `3v53tmq5` | **90.401635%** | 85.73% | 5.1556% | **Reused (Verified)** |
| **Without FF** | A1 | `results/table4_full_gpu_verified/.../table4_a1_without_ff` | `shl4p10l` | **90.100369%** | 85.48% | 5.4444% | **Reused (Verified)** |
| **Without SP** | A2 | `results/table4_full_gpu_verified/.../table4_a2_without_sp` | `rkedlkkk` | **90.382341%** | 85.72% | 5.2444% | **Reused (Verified)** |
| **Without FF & SP** | A3 | `results/table4_full_gpu_verified/.../table4_a3_without_both` | `k01pp8st` | **89.922353%** | 85.48% | 5.3222% | **Reused (Verified)** |

**Kết luận kiểm toán:** Seed 42 đủ điều kiện tái sử dụng nguyên vẹn, không cần chạy lại và được bảo vệ chống ghi đè trong `scripts/run_cifar100_multiseed_ablation.py`.

### 3.2. Chính sách Seed cho 42, 123, 2026 & Bảo toàn Paired Comparison

Mỗi seed $S \in \{42, 123, 2026\}$ kiểm soát nhất quán 4 nguồn ngẫu nhiên theo đúng cơ chế đã xác minh của `seed=42`:
1. `class_order_seed = S`: Hoán vị 100 lớp CIFAR-100 bằng `np.random.default_rng(S).permutation(100)`.
2. `projection_seed = S`: Khởi tạo ma trận chiếu $P \in \mathbb{R}^{768 \times 1000}$ bằng `torch.Generator('cpu').manual_seed(S)` cô lập.
3. `stream_seed = S`: Thứ tự xáo trộn mini-batch trong mỗi tác vụ `torch.Generator().manual_seed(S + task_id)`.
4. `base_seed = S`: `random.seed(S)`, `np.random.seed(S)`, `torch.manual_seed(S)`.

**Bất biến Paired Comparison:** Trong cùng một seed $S$, cả 4 cấu hình ablation (`full`, `no_ff`, `no_sp`, `no_ff_no_sp`) dùng **cùng `class_order_seed = S` và `stream_seed = S`**, đảm bảo `task_split_fingerprint` giống hệt nhau giữa 4 cấu hình. Mọi phép tính $\Delta\text{FF}(S), \Delta\text{SP}(S), \Delta\text{Both}(S)$ đều là so sánh cặp (paired) trên cùng một thứ tự lớp và luồng dữ liệu.

**An toàn Feature Cache:** Khóa cache trong `scripts/train.py` bao gồm `(backbone, interpolation, normalize, ff/noff, class_order_seed, stream_seed, task_id, limit)` kèm kiểm tra SHA256 của `indices`. Nhờ đó, trong cùng một seed, `full` và `no_sp` dùng chung cache `ff`, `no_ff` và `no_ff_no_sp` dùng chung cache `noff`, còn giữa các seed khác nhau hoàn toàn tách biệt.

---

## 4. Multi-Seed Results *(Pending GPU Execution)*

8 lượt chạy Full GPU cần thực thi trên Colab thông qua [`notebooks/foal_cifar100_multiseed_ablation.ipynb`](../notebooks/foal_cifar100_multiseed_ablation.ipynb):

- `cifar100_full_seed123`, `cifar100_no_ff_seed123`, `cifar100_no_sp_seed123`, `cifar100_no_ff_no_sp_seed123`
- `cifar100_full_seed2026`, `cifar100_no_ff_seed2026`, `cifar100_no_sp_seed2026`, `cifar100_no_ff_no_sp_seed2026`

> [!NOTE]
> Phần số liệu của `seed=123` và `seed=2026` hiện chưa có trong báo cáo vì môi trường hiện tại không có CUDA GPU (`torch.cuda.is_available() == False`). Không có bất kỳ số liệu giả lập hay ước lượng nào được điền vào bảng dưới đây.

### 4.1. Average Incremental Accuracy ($\bar{A}$, %)

| Configuration | Seed 42 (Verified) | Seed 123 | Seed 2026 | Mean ± Std (`ddof=1`) |
| :--- | :---: | :---: | :---: | :---: |
| **Full** | 90.4016% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No FF** | 90.1004% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No SP** | 90.3823% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No FF & SP** | 89.9224% | *Pending GPU* | *Pending GPU* | *Pending* |

### 4.2. Final Accuracy ($A_{10}$, %)

| Configuration | Seed 42 (Verified) | Seed 123 | Seed 2026 | Mean ± Std (`ddof=1`) |
| :--- | :---: | :---: | :---: | :---: |
| **Full** | 85.73% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No FF** | 85.48% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No SP** | 85.72% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No FF & SP** | 85.48% | *Pending GPU* | *Pending GPU* | *Pending* |

### 4.3. Past Forgetting ($F_{\text{past}}$, %)

| Configuration | Seed 42 (Verified) | Seed 123 | Seed 2026 | Mean ± Std (`ddof=1`) |
| :--- | :---: | :---: | :---: | :---: |
| **Full** | 5.1556% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No FF** | 5.4444% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No SP** | 5.2444% | *Pending GPU* | *Pending GPU* | *Pending* |
| **No FF & SP** | 5.3222% | *Pending GPU* | *Pending GPU* | *Pending* |

---

## 5. Mean ± Std Comparison Methodology

Sau khi hoàn thành 8 lượt chạy GPU và chạy `python -m scripts.analyze_cifar100_multiseed`, tất cả chỉ số sẽ được tổng hợp với **sample standard deviation (`ddof=1`)** trên $n = 3$ seeds, ghi rõ đơn vị phần trăm (`%`) cho giá trị tuyệt đối và điểm phần trăm (`pp`) cho các hiệu số.

---

## 6. Paired Component Contribution Analysis

Với mỗi seed $S \in \{42, 123, 2026\}$, các hiệu ứng ghép cặp được tính trực tiếp trên cùng một `task_split_fingerprint`:

$$\Delta\text{FF}(S) = \text{Acc}_{\text{Full}}(S) - \text{Acc}_{\text{NoFF}}(S)$$
$$\Delta\text{SP}(S) = \text{Acc}_{\text{Full}}(S) - \text{Acc}_{\text{NoSP}}(S)$$
$$\Delta\text{Both}(S) = \text{Acc}_{\text{Full}}(S) - \text{Acc}_{\text{NoFFNoSP}}(S)$$

Giá trị đã xác minh tại `seed=42` cho $\bar{A}$:
- $\Delta\text{FF}(42) = +0.3013\text{ pp}$
- $\Delta\text{SP}(42) = +0.0193\text{ pp}$
- $\Delta\text{Both}(42) = +0.4793\text{ pp}$

Các giá trị `Mean(Δ)` và `Std(Δ, ddof=1)`, số seed dương ($k/3$), kiểm tra đảo chiều dấu và 2×2 factorial effects (`E_FF_main`, `E_SP_main`, `E_FF×SP_interaction`) sẽ được sinh tự động vào `results/cifar100_multiseed/analysis/multiseed_summary.json`.

---

## 7. Interpretation of FF and SP Effects

- **Single-seed (Seed 42):** FF đóng góp hiệu ứng chính rõ hơn ($+0.3806\text{ pp}$ main effect), trong khi SP chỉ tăng $+0.0193\text{ pp}$ khi FF đã bật (nhưng $+0.1780\text{ pp}$ khi FF tắt).
- **Multi-seed (Pending):** Chỉ sau khi có `seed=123` và `seed=2026`, ta mới đánh giá được liệu hiệu ứng SP ($+0.0193\text{ pp}$) có dương nhất quán trên cả 3 seed hay dao động quanh 0, và liệu FF có luôn ổn định vượt trên phương sai giữa các seed hay không.

---

## 8. Limitations

1. **Cỡ mẫu nhỏ ($n = 3$ seeds):** Với 3 seed, độ lệch chuẩn mẫu (`ddof=1`) chỉ có 2 bậc tự do ($df = 2$), khoảng tin cậy $t_{0.975, 2} \approx 4.30$ rất rộng; do đó báo cáo không được phép khẳng định statistical significance chỉ từ `mean != 0`.
2. **Nguồn biến thiên theo seed:** Theo đúng cấu hình Seed 42, mỗi seed mới thay đổi đồng thời cả `class_order` và `projection` (khi SP bật). Đây là biến thiên tổng hợp của trình tự tác vụ và phép chiếu ngẫu nhiên.
3. **Trạng thái thực thi hiện tại:** 8 lượt GPU của `seed=123` và `seed=2026` chưa chạy do môi trường local không có GPU.

---

## 9. Comparison with Original Single-Seed Findings & Report Update Plan

- **Phân tách rõ ràng:** Kết quả tái lập single-seed (`90.40%` ở `seed=42` trong Table 2 / Headline Comparison) được giữ nguyên khi đối chiếu với paper single-seed, **không** tự động thay thế bằng trung bình multi-seed.
- **Các mục cần cập nhật trong báo cáo CIFAR-100 hiện tại sau khi hoàn tất GPU:**
  1. [`docs/CIFAR100_COMPLETE_WORK_REPORT.md`](CIFAR100_COMPLETE_WORK_REPORT.md) (Phần Table 4 Ablation & Phụ lục Chỉ mục): Bổ sung bảng 3-seed Mean ± Std và phân tích paired effects làm bằng chứng bổ trợ độ ổn định.
  2. [`docs/CIFAR100_FINAL_REPRODUCTION_REPORT.md`](CIFAR100_FINAL_REPRODUCTION_REPORT.md) (Mục Table 4 & Limitations): Cập nhật ghi chú chuyển từ giới hạn "single-seed only" sang kết quả 3-seed khi archive `foal_cifar100_multiseed_ablation_results.zip` được kiểm toán xong.

---

## 10. Recommendations for Next Steps

1. Mở [`notebooks/foal_cifar100_multiseed_ablation.ipynb`](../notebooks/foal_cifar100_multiseed_ablation.ipynb) trên Google Colab (chọn GPU Runtime T4/L4/A100).
2. Chạy lần lượt **Cell 1 (Preflight)** → **Cell 2 (Pilot)** → **Cell 3 (8 Full GPU Runs)** → **Cell 4 (Validation & Statistical Analysis)** → **Cell 5 (Export ZIP)**.
3. Tải `foal_cifar100_multiseed_ablation_results.zip` về `D:\Nhập môn ML\Paper\`, giải nén kiểm toán và cập nhật các bảng số liệu thực tế + 4 biểu đồ nghiên cứu vào báo cáo này để chuyển trạng thái từ `MULTISEED_READY_FOR_GPU` sang `MULTISEED_COMPLETED`.
