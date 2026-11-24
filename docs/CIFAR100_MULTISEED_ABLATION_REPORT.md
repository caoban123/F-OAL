# F-OAL CIFAR-100 Multi-Seed Table 4 Ablation Study (Seeds 42, 123, 2026)


> **Điều hướng nhanh:** [← Trang chủ `README.md`](../README.md) | [Báo cáo Headline CIFAR-100](CIFAR100_FINAL_REPRODUCTION_REPORT.md) | [W&B Report CIFAR-100](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/reports/F-OAL-T-i-l-p-k-t-qu-v-ph-n-t-ch-th-c-nghi-m-tr-n-b-d-li-u-CIFAR-100--VmlldzoxODI2NDExMA)

**Status:** `MULTISEED_COMPLETED`
**Repository:** [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL)
**Branch:** `experiment/cifar100-multiseed-ablation`
**Archive Executed Commit:** `d7ef0974e381e75bf0175caf11fb3096c37b7c3e`
**Target Benchmark:** CIFAR-100 (100 classes, 10 incremental tasks × 10 classes, 1-pass online analytic learning, zero replay)

---

## 1. Research Motivation

Ở giai đoạn tái lập single-seed trước đây (`seed=42`), nghiên cứu Table 4 Ablation trên CIFAR-100 ghi nhận:
- **Full F-OAL (FF On, SP On):** $\bar{A} = 90.401635\%$
- **Without FF (FF Off, SP On):** $\bar{A} = 90.100369\%$ ($\Delta_{\text{FF}} = +0.3013\text{ pp}$)
- **Without SP (FF On, SP Off):** $\bar{A} = 90.382341\%$ ($\Delta_{\text{SP}} = +0.0193\text{ pp}$)
- **Without FF & SP (FF Off, SP Off):** $\bar{A} = 89.922353\%$ ($\Delta_{\text{Both}} = +0.4793\text{ pp}$)

Với chỉ $n = 1$ seed, không thể kết luận liệu chênh lệch $+0.0193\text{ pp}$ của Smooth Projection (SP) hay $+0.3013\text{ pp}$ của Feature Fusion (FF) có ổn định qua các phân hoạch tác vụ khác nhau hay chỉ là dao động ngẫu nhiên của `seed=42`. Nghiên cứu này mở rộng thực nghiệm sang **3 seed độc lập: 42, 123, và 2026** (tổng cộng 12 Full runs = 4 cấu hình × 3 seeds) để kiểm định tính nhất quán bằng **paired within-seed comparison**.

---

## 2. Experimental Protocol & Reproducibility Provenance

Toàn bộ 12 runs tuân thủ nghiêm ngặt cấu hình **C3-D / Table 4** đã nghiệm thu (không trộn C3-A hay cấu hình khác):

| Tham số giao thức | Giá trị cố định |
| :--- | :--- |
| **Dataset & Split** | `uoft-cs/cifar100`, 100 lớp, 10 tasks × 10 lớp/task, Full 50,000 train & 10,000 test |
| **Learning Regime** | 1-pass online analytic learning (RLS), `replay_buffer = 0` |
| **Backbone** | `vit_base_patch16_224.augreg2_in21k_ft_in1k` (frozen) |
| **Preprocessing** | Bilinear resize ($224 \times 224$), Raw Tensor $[0, 1]$ (`normalize: false`) |
| **Feature Fusion (FF)** | Trung bình CLS token qua 12 Transformer blocks + LayerNorm ($D = 768$) khi bật; CLS block 12 khi tắt |
| **Smooth Projection (SP)** | $D_p = 1000$, `official_uniform` $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$ + `sigmoid` khi bật; $D_p = 768$ trực tiếp khi tắt |
| **Online RLS** | `gamma = 1.0`, `rls_dtype = float64`, `batch_size = 10` |

### Kiểm toán Archive & Seed 42 Provenance
- **Archive thực nghiệm mới:** `foal_cifar100_multiseed_ablation_results.zip` (`1,036,533 bytes`, SHA256: `9db4caf9e39c29cf19db9d79e6324895f4ebc48374ee08129912cb873a20521b`).
- **Tính toàn vẹn:** `CRC Check: ALL_OK`, toàn bộ **39/39 tệp** bên trong khớp `archive_sha256_manifest.json`, thực thi tại commit `d7ef0974e381e75bf0175caf11fb3096c37b7c3e` trên GPU Kaggle Tesla T4 (Python 3.13.16).
- **Seed 42 Reuse:** 4 runs `seed=42` được truy xuất trực tiếp từ artifacts gốc (`c3_d_bilinear_raw`, `table4_a1_without_ff`, `table4_a2_without_sp`, `table4_a3_without_both`), không chạy lại hay ghi đè.

---

## 3. Seed Policy & Paired-Comparison Invariants

Mỗi seed $S \in \{42, 123, 2026\}$ điều khiển đồng bộ `base_seed = S`, `class_order_seed = S`, `projection_seed = S`, và `stream_seed = S`. Kiểm toán độc lập xác nhận:
1. **Trong cùng một seed**, cả 4 cấu hình dùng chung một `task_split_fingerprint` và `class_order_sha256`, đảm bảo phép trừ paired hoàn toàn công bằng:
   - Seed 42: `7c461e9300daaa73...`
   - Seed 123: `6f045ced54415fb5...`
   - Seed 2026: `ed77195430fc8743...`
2. **Giữa các seed khác nhau**, `class_order_sha256` khác nhau hoàn toàn, phản ánh biến thiên thực của trình tự tác vụ và phép chiếu ngẫu nhiên.
3. **Tính toán Forgetting:** Cả 12 runs đều có $f_{10, j}^{\text{past}} \ge 0$ trên mọi tác vụ cũ $j \in \{0, \dots, 8\}$, nên `forgetting_past_tasks` dạng **unclipped** và **clipped** cho giá trị **giống hệt nhau**.

---

## 4. Full 12-Run Audited Results Table

| Seed | Configuration | W&B Run ID | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{\text{past}}$ (%) | $F_{\text{code}}$ (%) | Duration (s) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | Full F-OAL | `3v53tmq5` | 90.401635% | 85.73% | 5.1556% | 4.6400% | 229.11 | Reused (Verified) |
| **42** | Without FF | `shl4p10l` | 90.100369% | 85.48% | 5.4444% | 4.9000% | — | Reused (Verified) |
| **42** | Without SP | `rkedlkkk` | 90.382341% | 85.72% | 5.2444% | 4.7200% | — | Reused (Verified) |
| **42** | Without FF & SP | `k01pp8st` | 89.922353% | 85.48% | 5.3222% | 4.7900% | — | Reused (Verified) |
| **123** | Full F-OAL | `g32wwf9a` | 90.677456% | 85.70% | 6.0333% | 5.4300% | 902.84 | Verified (ZIP + W&B) |
| **123** | Without FF | `4lqkxf9f` | 90.290853% | 85.46% | 5.8556% | 5.2700% | 910.44 | Verified (ZIP + W&B) |
| **123** | Without SP | `q3d9rei2` | 90.630726% | 85.72% | 5.8667% | 5.2800% | 54.32* | Verified (ZIP + W&B) |
| **123** | Without FF & SP | `cvo857q3` | 90.270012% | 85.48% | 5.8778% | 5.2900% | 53.15* | Verified (ZIP + W&B) |
| **2026** | Full F-OAL | `9uinxzzr` | 90.686496% | 85.71% | 5.8556% | 5.2700% | 914.86 | Verified (ZIP + W&B) |
| **2026** | Without FF | `0wotm3e8` | 90.285738% | 85.38% | 5.6667% | 5.1000% | 915.32 | Verified (ZIP + W&B) |
| **2026** | Without SP | `6oxn4ioz` | 90.720476% | 85.72% | 5.9000% | 5.3100% | 59.80* | Verified (ZIP + W&B) |
| **2026** | Without FF & SP | `w0e0h1od` | 90.286528% | 85.48% | 5.3778% | 4.8400% | 59.89* | Verified (ZIP + W&B) |

*\*Các run `no_sp` và `no_ff_no_sp` tái sử dụng hợp lệ feature cache `ff` / `noff` trong cùng seed sau khi kiểm tra SHA256.*

---

## 5. Multi-Seed Mean ± Std Comparison (`ddof=1`, $n=3$)

### 5.1. Average Incremental Accuracy ($\bar{A}$, %)

| Configuration | Seed 42 | Seed 123 | Seed 2026 | Mean ± Std (`ddof=1`) |
| :--- | :---: | :---: | :---: | :---: |
| **Full F-OAL** | 90.4016% | 90.6775% | 90.6865% | **90.5885 ± 0.1619%** |
| **Without FF** | 90.1004% | 90.2909% | 90.2857% | **90.2257 ± 0.1085%** |
| **Without SP** | 90.3823% | 90.6307% | 90.7205% | **90.5778 ± 0.1752%** |
| **Without FF & SP** | 89.9224% | 90.2700% | 90.2865% | **90.1596 ± 0.2057%** |

### 5.2. Final Accuracy ($A_{10}$, %)

| Configuration | Seed 42 | Seed 123 | Seed 2026 | Mean ± Std (`ddof=1`) |
| :--- | :---: | :---: | :---: | :---: |
| **Full F-OAL** | 85.73% | 85.70% | 85.71% | **85.7133 ± 0.0153%** |
| **Without FF** | 85.48% | 85.46% | 85.38% | **85.4400 ± 0.0529%** |
| **Without SP** | 85.72% | 85.72% | 85.72% | **85.7200 ± 0.0000%** |
| **Without FF & SP** | 85.48% | 85.48% | 85.48% | **85.4800 ± 0.0000%** |

*Nhận xét đáng chú ý ở $A_{10}$:* Khi không dùng phép chiếu ngẫu nhiên (`Without SP` và `Without FF & SP`), nghiệm RLS đóng trên đặc trưng ViT cố định hoàn toàn bất biến trước hoán vị thứ tự tác vụ (độ lệch chuẩn = $0.0000\%$ trên cả 3 seeds: $85.72\%$ và $85.48\%$).

### 5.3. Past Forgetting ($F_{\text{past}}$, %)

| Configuration | Seed 42 | Seed 123 | Seed 2026 | Mean ± Std (`ddof=1`) |
| :--- | :---: | :---: | :---: | :---: |
| **Full F-OAL** | 5.1556% | 6.0333% | 5.8556% | **5.6815 ± 0.4641%** |
| **Without FF** | 5.4444% | 5.8556% | 5.6667% | **5.6556 ± 0.2058%** |
| **Without SP** | 5.2444% | 5.8667% | 5.9000% | **5.6704 ± 0.3692%** |
| **Without FF & SP** | 5.3222% | 5.8778% | 5.3778% | **5.5259 ± 0.3060%** |

---

## 6. Paired Effect Analysis & 2×2 Factorial Decomposition

Tất cả hiệu số dưới đây tính theo **điểm phần trăm (pp)** giữa các cấu hình có cùng `task_split_fingerprint`:

| Contrast ($\bar{A}$, pp) | Seed 42 | Seed 123 | Seed 2026 | Mean ± Std (`ddof=1`) | Positive Seeds |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\Delta\text{FF} = \text{Full} - \text{NoFF}$** | $+0.3013$ | $+0.3866$ | $+0.4008$ | **$+0.3629 \pm 0.0538\text{ pp}$** | **3 / 3** |
| **$\Delta\text{SP} = \text{Full} - \text{NoSP}$** | $+0.0193$ | $+0.0467$ | **$-0.0340$** | **$+0.0107 \pm 0.0410\text{ pp}$** | **2 / 3 (Đảo dấu!)** |
| **$\Delta\text{Both} = \text{Full} - \text{NoFFNoSP}$** | $+0.4793$ | $+0.4074$ | $+0.4000$ | **$+0.4289 \pm 0.0438\text{ pp}$** | **3 / 3** |
| **$E_{\text{FF, main}}$ (2×2 Main)** | $+0.3806$ | $+0.3737$ | $+0.4174$ | **$+0.3905 \pm 0.0235\text{ pp}$** | **3 / 3** |
| **$E_{\text{SP, main}}$ (2×2 Main)** | $+0.0987$ | $+0.0338$ | **$-0.0174$** | **$+0.0384 \pm 0.0582\text{ pp}$** | **2 / 3 (Đảo dấu!)** |
| **$E_{\text{FF}\times\text{SP}}$ (Interaction)** | $-0.1587$ | $+0.0259$ | $-0.0332$ | **$-0.0553 \pm 0.0943\text{ pp}$** | 1 / 3 |

### Trả lời các câu hỏi nghiên cứu (RQ1–RQ5)
- **RQ1 (Tính nhất quán của Feature Fusion):** **CÓ.** FF cải thiện $\bar{A}$ ở **3/3 seeds** ($+0.3013, +0.3866, +0.4008\text{ pp}$; $\text{mean}/\text{std} = 6.74\times$) và cải thiện $A_{10}$ ở **3/3 seeds** ($+0.2733 \pm 0.0493\text{ pp}$).
- **RQ2 (Tính nhất quán của Smooth Projection):** **KHÔNG.** Trên CIFAR-100 với ViT-B/16, SP **không** cải thiện nhất quán: Seed 42 ($+0.0193\text{ pp}$) và Seed 123 ($+0.0467\text{ pp}$) dương nhẹ, nhưng **Seed 2026 đảo dấu thành $-0.0340\text{ pp}$** (`Without SP` đạt $90.7205\%$ > `Full` $90.6865\%$). Trung bình $\Delta\text{SP} = +0.0107 \pm 0.0410\text{ pp}$ có độ lệch chuẩn lớn gấp ~4 lần giá trị trung bình, cắt ngang 0. Trên $A_{10}$, SP thậm chí cho trung bình hơi âm ($-0.0067 \pm 0.0153\text{ pp}$).
- **RQ3 (Hiệu ứng kết hợp FF + SP):** Hiệu ứng gộp $\Delta\text{Both} = +0.4289 \pm 0.0438\text{ pp}$ lớn hơn riêng $\Delta\text{FF}$ một chút ($+0.066\text{ pp}$), chủ yếu đến từ Seed 42 nơi SP giúp bù khi FF tắt ($+0.1780\text{ pp}$). Tuy nhiên trên Seed 123 và Seed 2026, $\Delta\text{Both}$ ($+0.4074, +0.4000\text{ pp}$) gần như **trùng khít** với $\Delta\text{FF}$ ($+0.3866, +0.4008\text{ pp}$).
- **RQ4 (Biến thiên giữa các seeds vs Effect size):** Biến thiên giữa các seeds trên giá trị tuyệt đối $\bar{A}$ vào khoảng $0.11\text{--}0.21\text{ pp}$. Khi khử biến thiên theo từng seed bằng paired comparison:
  - Effect của **FF** ($+0.3629 \pm 0.0538\text{ pp}$) vượt rõ rệt biến thiên giữa các seeds.
  - Effect của **SP** ($+0.0107 \pm 0.0410\text{ pp}$) nhỏ hơn nhiều so với biến thiên giữa các seeds ($\sim 0.16\text{ pp}$) và ngay cả biến thiên của chính $\Delta\text{SP}$ ($0.041\text{ pp}$) cũng lấn át mean.
- **RQ5 (So sánh với kết luận single-seed trước đây):**
  - *Kết luận được củng cố:* FF thực sự mang lại cải thiện ổn định $\sim +0.36\text{ pp}$ trên CIFAR-100.
  - *Kết luận cần điều chỉnh:* Nhận định ở `seed=42` rằng *"bỏ SP làm giảm $0.02\text{ pp}$ và bỏ cả hai làm giảm $0.48\text{ pp}$"* không lặp lại ổn định ở SP; ở quy mô 3 seeds, đóng góp của SP trên đặc trưng ViT-B/16 ($D=768 \to D_p=1000$) trên CIFAR-100 là **không phân biệt được với nhiễu ngẫu nhiên giữa các seed**.

---

## 7. Scientific Visualizations

4 biểu đồ chuẩn 300 DPI được sinh trực tiếp từ dữ liệu đã kiểm toán tại `results/cifar100_multiseed/analysis/`:
1. [`fig1_multiseed_ablation_accuracy.png`](../results/cifar100_multiseed/analysis/fig1_multiseed_ablation_accuracy.png) — So sánh $\bar{A}$ và $A_{10}$ (Mean ± Std + các điểm từng seed).
2. [`fig2_paired_component_contributions.png`](../results/cifar100_multiseed/analysis/fig2_paired_component_contributions.png) — Paired Component Contributions ($\Delta\text{FF}, \Delta\text{SP}, \Delta\text{Both}$ theo từng seed và Mean ± Std).
3. [`fig3_multiseed_accuracy_progression.png`](../results/cifar100_multiseed/analysis/fig3_multiseed_accuracy_progression.png) — Đường cong tiến trình qua 10 tasks (Mean ± 1 Std band).
4. [`fig4_accuracy_vs_forgetting.png`](../results/cifar100_multiseed/analysis/fig4_accuracy_vs_forgetting.png) — Biểu đồ đánh đổi Accuracy ($\bar{A}$) vs Past Forgetting ($F_{\text{past}}$).

---

## 8. Limitations & Recommendation on 5 Seeds

1. **Giới hạn cỡ mẫu ($n = 3$):** Với 3 seeds ($df = 2$), chúng ta không dùng $p$-value từ $t$-test để tuyên bố statistical significance; kết luận dựa trên tính nhất quán dấu (sign consistency), độ lớn hiệu ứng paired và tỷ lệ tín hiệu/nhiễu.
2. **Không nên chạy thêm lên 5 seeds cho CIFAR-100 Table 4:**
   - Với **FF** ($\Delta\text{FF} = +0.3629 \pm 0.0538\text{ pp}$, 3/3 dương, SNR $= 6.74$), bằng chứng đã đủ nhất quán.
   - Với **SP** ($\Delta\text{SP} = +0.0107 \pm 0.0410\text{ pp}$, 2 dương / 1 âm), kích thước hiệu ứng trung bình gần 0 (~$0.01\text{ pp}$) trong thiết lập ViT-B/16 + `official_uniform` ($D_p=1000$) trên CIFAR-100. Dù có tăng lên 5 seeds, sự khác biệt $\sim 0.01\text{ pp}$ này không mang ý nghĩa thực tiễn đáng kể trên CIFAR-100; tài nguyên GPU nên được ưu tiên cho đánh giá đa bộ dữ liệu (DTD, FGVC-Aircraft).

---

## 9. Comparison with Single-Seed Reproduction & Report Update Guidance

- **Giữ nguyên mốc đối chiếu Paper:** Giá trị tái lập chính thức tại `seed=42` ($\bar{A} = \mathbf{90.40\%}$, $A_{10} = \mathbf{85.73\%}$, $F_{\text{past}} = \mathbf{5.16\%}$) trong bảng đối chiếu Table 2 với paper (vốn là single-seed hoặc theo giao thức đối chiếu trước đó) **không** bị thay thế tự động bằng trung bình multi-seed ($\bar{A}_{\text{3-seed}} = 90.59 \pm 0.16\%$). Trung bình multi-seed được trình bày ở mục riêng để chứng minh độ ổn định phương pháp.
- **Nội dung bổ sung cho báo cáo tổng hợp & W&B Report:** Bổ sung tóm tắt 3-seed Mean ± Std và Paired Effect ($\Delta\text{FF} = +0.36 \pm 0.05\text{ pp}$, $\Delta\text{SP} = +0.01 \pm 0.04\text{ pp}$) kèm 4 biểu đồ trên vào mục Table 4 của W&B Report và [`CIFAR100_COMPLETE_WORK_REPORT.md`](CIFAR100_COMPLETE_WORK_REPORT.md).
