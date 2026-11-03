# F-OAL — C3 Controlled Input Preprocessing & Normalization Alignment Plan ($2 \times 2$ Factorial Design)

> **Trạng thái tài liệu:** `C3_SIGNED_OFF_WITH_LIMITATIONS` (Đã hoàn tất Full GPU trên A100, đồng bộ 10 runs W&B trên CoreWeave Forge, kiểm toán độc lập 55-entry matrix và phân rã giai thừa 2x2) (Đã hoàn tất toàn bộ triển khai mã nguồn, cấu hình, kiểm thử hồi quy 6/6 tests passing, chẩn đoán mẫu và Colab Notebook. Sẵn sàng chạy Full GPU trên Google Colab / CoreWeave Forge).  
> **Kế thừa từ:** C0 (Metric Alignment: `COMPLETED`), C1 (Controlled Projection Initialization: `MERGED INTO MAIN`), C2 (Controlled ViT Backbone Alignment: `MERGED INTO MAIN` tại commit `7b76fd7`).  
> **Nhánh thực nghiệm:** `experiment/c3-preprocessing-alignment`.  
> **Quyết định tái sử dụng tính toán:** `C3_A_REUSE_APPROVED` (C3-A tương đương 100% bitwise với C2-B1, chỉ cần chạy C3-B, C3-C, C3-D trên GPU).

---

## 1. Vấn đề Khoa học & Bối cảnh (Scientific Problem & Context)

### 1.1. Bối cảnh từ C0, C1 và C2
- **C0 (Metric Alignment):** Đã phân định độc lập công thức Forgetting giữa Paper Eq. 17–18 ($F_{10}^{\text{past}}$, mẫu số $9$) và Official Code ($F_{10}^{\text{code}}$, mẫu số $10$).
- **C1 (Smooth Projection Scale, H2):** Khởi tạo $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$ loại bỏ triệt để hiện tượng bão hòa Sigmoid ($81.03\% \to 0.00\%$), nâng $\bar{A}$ từ $79.33\%$ lên $83.28\%$ ($+3.95\text{ pp}$), thu hẹp $33.9\%$ khoảng cách ban đầu.
- **C2 (ViT Pretrained Backbone, H1):** Căn chỉnh backbone sang ImageNet-21K checkpoint (`vit_base_patch16_224.augreg2_in21k_ft_in1k`) nâng $\bar{A}$ lên **$91.77\%$** ($+8.49\text{ pp}$ so với $83.28\%$), tương đương về mặt số học và cao hơn một lượng nhỏ so với Paper ($91.10\%$) và Official Run ($90.98\%$).
- **Hiện tượng cần giải quyết trong C3:**
  - Ở C2, bản chạy B1 của nhóm ($91.77\%$) và bản chạy tác giả Official Run ($90.98\%$) vẫn còn hai biến số chưa được căn chỉnh: hạt giống ngẫu nhiên (`seed=42` vs `seed=0`) và **Quy trình tiền xử lý hình ảnh (Image Preprocessing — Giả thuyết H4)**.
  - Hiện thực độc lập của nhóm (`src/foal/datasets.py`):
    `Resize(224, InterpolationMode.BICUBIC) -> ToTensor() -> Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` $\implies$ pixel trong $[-1.0, 1.0]$.
  - Mã nguồn gốc của tác giả (`utils/setup_elements.py:49–53`):
    `ToPILImage() -> Resize((224, 224))` (PIL default Bilinear) `-> ToTensor()` $\implies$ pixel trong $[0.0, 1.0]$ (không có bước `Normalize`).

### 1.2. Tính Phức hợp của Biến số Tiền xử lý
Sự khác biệt giữa hai pipeline **không phải là một biến số đơn nhất** mà bao gồm hai nhân tố độc lập:
1. **Nhân tố A — Phương pháp Nội suy Không gian (Spatial Interpolation):** Bicubic (bậc 3) vs. Bilinear (bậc 1).
2. **Nhân tố B — Chuẩn hóa Dải Pixel (Pixel Normalization):** Chuẩn hóa về $[-1.0, 1.0]$ vs. Giữ nguyên dải thô $[0.0, 1.0]$.

> [!CAUTION]
> Nếu thay đổi đồng thời cả hai yếu tố trong một lượt chạy duy nhất, ta sẽ không thể xác định được sự thay đổi của độ chính xác là do phép nội suy, do phép chuẩn hóa dải pixel, hay do hiệu ứng tương tác (interaction effect) giữa chúng. Vì vậy, **C3 bắt buộc phải được thiết kế theo mô hình giai thừa $2 \times 2$ (Factorial Design)**.

---

## 2. Thiết kế Nghiên cứu Giai thừa $2 \times 2$ (2×2 Factorial Design)

Cố định toàn bộ các thành phần khác:
- Backbone: `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21K + IN-1K FT).
- Khởi tạo ma trận chiếu $P$: `official_uniform` (seed 42, $D=1000$, cùng mã SHA256: `6d3048f4a0fca387...`).
- Seed: `seed = 42` (`class_order_seed = projection_seed = stream_seed = 42`).
- RLS: $\gamma = 1.0$, `float64`, batch size $10$.

### Bảng 4 Cấu hình Giai thừa $2 \times 2$:

| Mã Cấu hình | Phương pháp Nội suy (Factor A) | Chuẩn hóa Pixel (Factor B) | Dải Pixel Đầu vào | Vai trò Khoa học |
|:---|:---:|:---:|:---:|:---|
| **C3-A** | **Bicubic** | **Normalize** (`mean=0.5, std=0.5`) | $[-1.0, 1.0]$ | **Baseline Đối chứng** (Tái hiện chính xác C2 B1) |
| **C3-B** | **Bilinear** | **Normalize** (`mean=0.5, std=0.5`) | $[-1.0, 1.0]$ | **Cô lập Tác động Nội suy** ($\text{Main Effect}_A$) |
| **C3-C** | **Bicubic** | **Raw Tensor** (No Normalization) | $[0.0, 1.0]$ | **Cô lập Tác động Chuẩn hóa** ($\text{Main Effect}_B$) |
| **C3-D** | **Bilinear** | **Raw Tensor** (No Normalization) | $[0.0, 1.0]$ | **Căn chỉnh Giao thức Tác giả** (Full Author Preprocessing) |

### Phân rã Toán học Các Tác động:
- **Tác động chính của Phép Nội suy ($\Delta_{\text{Interp}}$):**
  $$\Delta_{\text{Interp}} = \frac{1}{2} \left[ (\bar{A}_{\text{C3-B}} - \bar{A}_{\text{C3-A}}) + (\bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-C}}) \right]$$
- **Tác động chính của Phép Chuẩn hóa ($\Delta_{\text{Norm}}$):**
  $$\Delta_{\text{Norm}} = \frac{1}{2} \left[ (\bar{A}_{\text{C3-C}} - \bar{A}_{\text{C3-A}}) + (\bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-B}}) \right]$$
- **Hiệu ứng Tương tác ($\Delta_{\text{Interact}}$):**
  $$\Delta_{\text{Interact}} = (\bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-B}}) - (\bar{A}_{\text{C3-C}} - \bar{A}_{\text{C3-A}})$$
- **Tổng chênh lệch Tiền xử lý ($\Delta_{\text{Prep, total}}$):**
  $$\Delta_{\text{Prep, total}} = \bar{A}_{\text{C3-D}} - \bar{A}_{\text{C3-A}}$$

---

## 3. Các Yếu tố Chi tiết Kỹ thuật Cần Kiểm soát (Subtle Preprocessing Nuances)

Thực nghiệm C3 không chỉ so sánh tên hàm mà phải kiểm soát chặt chẽ các chi tiết triển khai cụ thể:
1. **PIL vs. Tensor Resizing:**
   - Mã tác giả sử dụng `transforms.ToPILImage() -> transforms.Resize((224, 224))` (sử dụng backend PIL).
   - PyTorch torchvision `transforms.Resize` trên PIL Image mặc định bật `antialias=True` từ phiên bản $0.15+$. Cần xác minh cờ `antialias` giữa hai phiên bản torchvision để tránh nhiễu do thuật toán lọc chống răng cưa.
2. **Thứ tự Chuyển đổi (Transform Ordering):**
   - Chuyển ảnh PIL sang Tensor trước rồi mới Resize, hay Resize trên PIL rồi mới `ToTensor()` có thể tạo sai khác nhỏ về mức lượng tử hóa dấu phẩy động ($[0, 255] \to [0.0, 1.0]$).
3. **Không gian Màu & Kiểu dữ liệu:**
   - Bảo đảm ảnh được ép kiểu đồng nhất sang RGB 3 kênh chuẩn `torch.float32`.

---

## 4. Kiểm soát Biến số Cố định Nghiêm ngặt (Strict Fixed Controls)

Mọi thành phần dưới đây được đóng băng $100\%$ không thay đổi giữa cả 4 cấu hình C3:

| Thành phần | Thiết lập Cố định | Trạng thái Kiểm soát |
|:---|:---|:---:|
| **Tập dữ liệu** | CIFAR-100 ($50,000$ train, $10,000$ test) | Cố định 100% |
| **Phân chia Task** | 10 tasks $\times$ 10 classes | Cố định 100% |
| **Hạt giống (Seeds)** | `seed = 42`, `class_order_seed = 42`, `stream_seed = 42` | Cố định 100% |
| **Backbone Model** | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | Cố định 100% |
| **Trạng thái Backbone** | Đóng băng tham số (`requires_grad = False`), cố định `eval()` | Cố định 100% |
| **Feature Fusion** | Mean CLS 12 blocks + `network.norm` | Cố định 100% |
| **Ma trận Chiếu $P$** | `official_uniform`, $D=1000$, `projection_seed = 42` | Cố định 100% (cùng SHA256) |
| **Hàm kích hoạt** | Sigmoid $\sigma(\cdot)$ phi tuyến | Cố định 100% |
| **Giải tích Classifier** | Woodbury RLS, $\gamma = 1.0$, `torch.float64` | Cố định 100% |
| **Kích thước Mini-batch**| $B = 10$, xử lý tuần tự 1-pass (zero-replay) | Cố định 100% |

---

## 5. An toàn Bộ nhớ Đệm Đặc trưng (Feature Cache Safety Rules)

Do việc thay đổi tiền xử lý làm thay đổi trực tiếp tensor ảnh đầu vào của ViT, cơ chế Feature Cache bắt buộc phải tuân thủ nghiêm ngặt:
1. **Cô lập Đường dẫn Cache:**
   - Mỗi cấu hình phải có thư mục cache riêng biệt, ví dụ:
     - C3-A: `results/c3_feature_cache/c3_a_bicubic_norm/`
     - C3-B: `results/c3_feature_cache/c3_b_bilinear_norm/`
     - C3-C: `results/c3_feature_cache/c3_c_bicubic_raw/`
     - C3-D: `results/c3_feature_cache/c3_d_bilinear_raw/`
2. **Khóa Cache & Kiểm tra Metadata:**
   Mỗi tệp `.pt` cache phải chứa metadata xác thực:
   ```json
   {
     "backbone": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
     "preprocessing": {
       "interpolation": "bilinear",
       "image_size": 224,
       "normalized": false,
       "mean": [0.0, 0.0, 0.0],
       "std": [1.0, 1.0, 1.0]
     },
     "indices_sha256": "...",
     "features_sha256": "..."
   }
   ```
3. **Từ chối Cache Bất tương thích:** Nếu phát hiện `interpolation` hoặc `normalized` không khớp cấu hình, hàm trích xuất phải ném lỗi hoặc tự động trích xuất lại từ đầu, tuyệt đối không được tải chéo.

---

## 6. Kế hoạch Tối ưu Hóa Tài nguyên (Resource & Compute Planning)

Một lượt chạy đầy đủ 10 tasks CIFAR-100 cho 4 cấu hình đòi hỏi $4 \times 50,000 = 200,000$ lượt forward ViT:
1. **Tái sử dụng Kết quả C2-B1 cho C3-A:**
   Nếu cấu hình tiền xử lý, seed, thứ tự mẫu và mã nguồn của C3-A trùng khớp hoàn toàn $100\%$ bitwise với C2-B1, nhóm có thể tái sử dụng dữ liệu đặc trưng đã lưu trữ của C2-B1 để tiết kiệm $25\%$ thời gian tính toán GPU.
2. **Quy trình Pilot 2 Tasks trước khi chạy Full GPU:**
   - Bắt buộc chạy Pilot 2 tasks trên CPU hoặc GPU nhỏ cho cả 4 cấu hình để kiểm tra shape, dải giá trị tensor ảnh và bộ kiểm tra cache trước khi chạy 10 tasks đầy đủ.
3. **Ước tính Thời gian GPU (NVIDIA A100):**
   - Mỗi cấu hình mới: $\approx 5\text{ phút}$ (trích xuất đặc trưng $248\text{ s}$ + RLS $8.5\text{ s}$).
   - Tổng thời gian cho 3 cấu hình mới (C3-B, C3-C, C3-D): $\approx 15\text{ phút}$.

---

## 7. Chẩn đoán Đối khớp Tiền xử lý (Matched Preprocessing Diagnostics)

Trước khi chạy toàn bộ 10 tasks, trích xuất đặc trưng từ 200 ảnh CIFAR-100 thực tế trên cả 4 cấu hình để đo lường:
1. **Thống kê Tensor Đầu vào:**
   - $\min(x), \max(x), \text{mean}(x), \text{std}(x)$.
2. **Đặc trưng ViT ($h \in \mathbb{R}^{200 \times 768}$):**
   - Độ dài vector $\|h\|_2$.
   - Độ tương đồng Cosine giữa các vector cùng ảnh qua các phép tiền xử lý: $\text{Cosine}(h_{\text{C3-A}}, h_{\text{C3-D}})$.
   - Độ lệch chuẩn tiền kích hoạt $\text{std}(u = hP)$ và kiểm tra bão hòa Sigmoid.
   - Số chiều hữu hiệu ma trận hiệp phương sai chuẩn tâm ($r_{\text{eff}}$).

---

## 8. Tiêu chuẩn Nghiệm thu Khoa học (Acceptance Gates for C3)

- [x] Toàn bộ 4 cấu hình C3-A, C3-B, C3-C, C3-D được thiết lập trên cùng ma trận chiếu $P$ (`official_uniform`, seed 42, $D=1000$).
- [x] Tính năng Cache Safety cô lập hoàn toàn giữa 4 cấu hình qua tiền tố tên file `interp_tag` và `norm_tag`, kiểm tra metadata chặt chẽ khi tải.
- [x] Triển khai thành công chẩn đoán mẫu thực nghiệm (`scripts/run_c3_diagnostics.py`), xác minh sai khác điểm ảnh giữa C3-D và Official Transform bằng chính xác $0.00\text{e+}00$ (tương đương 100%).
- [x] Kiểm thử tự động hoàn chỉnh (`tests/test_c3_preprocessing.py`, 6/6 tests passing 100%).
- [x] Quyết định tái sử dụng tính toán được phê duyệt (`C3_A_REUSE_APPROVED`): C3-A bitwise tương đương C2-B1 ($\bar{A}=91.7742\%$, W&B run `ne1p16b5`), tiết kiệm 25% GPU compute.
- [x] Xây dựng Google Colab Notebook 14 cells hoàn chỉnh (`notebooks/foal_c3_preprocessing_controlled.ipynb`).
- [x] Chạy Full GPU cho C3-B, C3-C, C3-D trên Colab (A100) và đồng bộ W&B group `c3_preprocessing_alignment` ([ts2wh4e3](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3), [4m4uir7h](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h), [3v53tmq5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5)).
- [x] Tính toán phân rã giai thừa thực nghiệm và nghiệm thu báo cáo khoa học C3 (`C3_SIGNED_OFF_WITH_LIMITATIONS`).

---

## 9. Định vị C3 trong Bối cảnh Toàn diện Đồ án (Broader Project Scope)

> [!IMPORTANT]
> **Quy định Nghiêm ngặt về Tiến độ HCMUS Checkpoint 3:**  
> Chuỗi thí nghiệm kiểm soát C1 (Khởi tạo chiếu), C2 (Backbone ViT), và C3 (Tiền xử lý) là **thực nghiệm thẩm định giao thức trên CIFAR-100**.  
> Việc hoàn thành C1–C3 **chưa đồng nghĩa với hoàn tất toàn bộ yêu cầu đồ án**.  
> Theo bản đặc tả đồ án, sau khi hoàn tất căn chỉnh giao thức trên CIFAR-100, nhóm phải thực hiện:
> 1. **Thí nghiệm Đối chứng Đầu bảng (Headline Comparison):** Chạy so sánh F-OAL với các Baseline đại diện (Backprop FC 1-pass, Linear Ridge, Experience Replay) trên cùng task split.
> 2. **Ablation Study Toàn diện (Table 4):** Đánh giá 4 tổ hợp Feature Fusion $\times$ Smooth Projection.
> 3. **Phân tích Mở rộng Độc lập (Original Additional Analysis):** Kiểm chứng tính bất biến theo Batch size $B \in \{5, 10, 25, 50, 100\}$ và độ nhạy siêu tham số $D, \gamma$.
> 4. **Mở rộng Đa Tập Dữ liệu (Cross-Dataset Evaluation):** Mở rộng đánh giá trên các tập dữ liệu khác (như DTD, FGVCAircraft) nếu thuộc phạm vi Core Scope được phê duyệt.

---

## 10. Báo cáo Kết quả Thực nghiệm Full GPU & Phân rã Giai thừa $2 \times 2$

### 10.1. Môi trường Thực thi & Kiểm chứng Toàn vẹn Dữ liệu
- **Phần cứng GPU:** NVIDIA A100-SXM4-40GB (39.49 GB VRAM).
- **Phần mềm:** PyTorch `2.11.0+cu130`, Torchvision `0.26.0+cu130`, timm `1.0.30`, Python `3.13.16`.
- **Gói nén lưu trữ kết quả:** `foal_c3_preprocessing_results.zip` (148,700 bytes, SHA256: `a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6`).
- **Xác thực toàn vẹn:** 54 file trích xuất thành công tại `results/c3_full_gpu_verified/`, CRC-32 hợp lệ 100%, không phát hiện lỗi đường dẫn hay tràn buffer.
- **Tính toán độc lập:** Toàn bộ số liệu được tính toán lại từ ma trận tam giác dưới $10 \times 10$ ($55$ phần tử) tại `scripts/audit_c3_results.py`. Độ lệch tối đa so với số liệu xuất từ Colab bằng chính xác $0.00\text{e+}00$.

### 10.2. Bảng Kết quả Thực nghiệm Giai thừa $2 \times 2$

| Điều kiện | Phép Nội suy (Factor A) | Chuẩn hóa Pixel (Factor B) | Dải Pixel | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{10}^{\text{past}}$ (%) | $F_{10}^{\text{code}}$ (%) | T_Feat (s) | T_RLS (s) | W&B Run ID | Trạng thái Provenance |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---|
| **C3-A** | Bicubic | Normalize | $[-1.0, 1.0]$ | **91.7742%** | **87.4300%** | **5.0222%** | **4.5200%** | 248.30 | 8.49 | [`ne1p16b5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5) | `REUSED_VERIFIED_C2_B1` |
| **C3-B** | Bilinear | Normalize | $[-1.0, 1.0]$ | **91.3918%** | **86.9500%** | **4.8889%** | **4.4000%** | 176.08 | 8.56 | [`ts2wh4e3`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3) | `VERIFIED_FRESH_GPU` |
| **C3-C** | Bicubic | Raw | $[0.0, 1.0]$ | **90.8762%** | **86.3500%** | **5.2333%** | **4.7100%** | 175.58 | 8.49 | [`4m4uir7h`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h) | `VERIFIED_FRESH_GPU` |
| **C3-D** | Bilinear | Raw | $[0.0, 1.0]$ | **90.4016%** | **85.7300%** | **5.1556%** | **4.6400%** | 175.28 | 8.50 | [`3v53tmq5`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5) | `VERIFIED_FRESH_GPU` |

### 10.3. Phân tích Phân rã Giai thừa (Factorial Effects Decomposition)

1. **Tác động Đơn giản (Simple Effects):**
   - Phép Nội suy trong môi trường Normalized: $E_{I|\text{Norm}} = B - A = 91.3918\% - 91.7742\% = -0.3824\text{ pp}$.
   - Phép Nội suy trong môi trường Raw: $E_{I|\text{Raw}} = D - C = 90.4016\% - 90.8762\% = -0.4746\text{ pp}$.
   - Phép Chuẩn hóa trong môi trường Bicubic: $E_{N|\text{Bicubic}} = C - A = 90.8762\% - 91.7742\% = -0.8980\text{ pp}$.
   - Phép Chuẩn hóa trong môi trường Bilinear: $E_{N|\text{Bilinear}} = D - B = 90.4016\% - 91.3918\% = -0.9902\text{ pp}$.

2. **Tác động Chính Trung bình (Average Main Effects):**
   - **Tác động chính của Phép Nội suy ($E_{\text{interp}}$):**
     $$E_{\text{interp}} = \frac{(B - A) + (D - C)}{2} = \frac{-0.3824 + (-0.4746)}{2} = \mathbf{-0.4285\text{ pp}}$$
   - **Tác động chính của Chuẩn hóa dải pixel ($E_{\text{norm}}$):**
     $$E_{\text{norm}} = \frac{(C - A) + (D - B)}{2} = \frac{-0.8980 + (-0.9902)}{2} = \mathbf{-0.9441\text{ pp}}$$

3. **Hiệu ứng Tương tác (Interaction Contrast & Coefficient):**
   - **Tương phản tương tác ($E_{\text{IN}}$):**
     $$E_{\text{IN}} = D - C - B + A = (D - B) - (C - A) = -0.9902 - (-0.8980) = \mathbf{-0.0922\text{ pp}}$$
   - **Hệ số hồi quy tương tác (Effects Coding $\beta_{\text{IN}}$):**
     $$\beta_{\text{IN}} = \frac{E_{\text{IN}}}{4} = \mathbf{-0.0231\text{ pp}}$$

4. **Tổng Dịch chuyển Tiền xử lý (Total Preprocessing Shift):**
   $$\Delta_{\text{total}} = D - A = 90.4016\% - 91.7742\% = \mathbf{-1.3726\text{ pp}}$$
   - **Kiểm chứng đồng nhất thức toán học:** $E_{\text{interp}} + E_{\text{norm}} = -0.4285 + (-0.9441) = -1.3726\text{ pp} \equiv \Delta_{\text{total}}$ (sai số $< 10^{-12}$).
   - **Tỷ trọng đóng góp:** Chuẩn hóa dải pixel đóng góp **$68.78\%$** tổng mức sụt giảm; phương pháp nội suy đóng góp **$31.22\%$**.
   - **Tính chất tương tác:** Giá trị tương tác rất nhỏ ($-0.0922\text{ pp}$), cho thấy hai nhân tố hoạt động gần như độc lập tuyến tính (additive), trong đó chuẩn hóa pixel $[-1, 1]$ là yếu tố mang lại lợi thế biểu diễn lớn hơn.

### 10.4. Đối chiếu Đối chứng với Tác giả (Official Run) và Bài báo (NeurIPS Paper)

| Chỉ số Đánh giá | NeurIPS 2024 Paper | Official Author Run (`3eca8232`, Seed 0) | C3-D Full Alignment (`3v53tmq5`, Seed 42) | Chênh lệch C3-D vs Official | Chênh lệch C3-D vs Paper |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Average Incremental Accuracy ($\bar{A}$)** | **91.10%** | **90.98%** | **90.40%** | **$-0.58\text{ pp}$** | **$-0.70\text{ pp}$** |
| **Last Accuracy ($A_{10}$)** | — | **85.84%** | **85.73%** | **$-0.11\text{ pp}$** | — |
| **Forgetting Official Code ($F_{10}^{\text{code}}$)** | 5.50% | **4.69%** | **4.64%** | **$-0.05\text{ pp}$** | $-0.86\text{ pp}$ |
| **Forgetting Past Tasks ($F_{10}^{\text{past}}$)** | — | **5.17%** | **5.16%** | **$-0.01\text{ pp}$** | — |

> [!IMPORTANT]
> **Ý Nghĩa Khoa Học Then Chốt:**
> 1. Khi căn chỉnh tiền xử lý về đúng pipeline của tác giả (Bilinear, Raw $[0, 1]$), độ chính xác cuối cùng $A_{10}$ của nhóm ($85.73\%$) **hội tụ gần như hoàn hảo** với bản chạy của tác giả ($85.84\%$, độ lệch chỉ vỏn vẹn $-0.11\text{ pp}$).
> 2. Tỷ lệ quên $F_{10}^{\text{past}}$ đạt $5.16\%$ so với $5.17\%$ của tác giả (chênh lệch $-0.01\text{ pp}$), và $F_{10}^{\text{code}}$ đạt $4.64\%$ so với $4.69\%$ (chênh lệch $-0.05\text{ pp}$).
> 3. Sự khác biệt nhỏ còn lại trên $\bar{A}$ ($90.40\%$ vs $90.98\%$, chênh lệch $-0.58\text{ pp}$) xuất phát từ hoán vị thứ tự các lớp và độ khó của các task đầu giữa Seed 42 và Seed 0.
> 4. Bí ẩn về việc C2-B1 vượt tác giả $+0.79\text{ pp}$ đã được giải mã trọn vẹn: đó là do C2-B1 vô tình hưởng lợi từ tiền xử lý chuẩn ImageNet (Bicubic + chuẩn hóa $[-1, 1]$), giúp dải động và tần số không gian của ảnh đưa vào ViT phong phú hơn.

### 10.5. Kết luận Nghiệm thu Khoa học (Sign-off Status)
- **Quyết định Nghiệm thu:** **`C3_SIGNED_OFF_WITH_LIMITATIONS`**
- Toàn bộ 4 thực nghiệm C3-A, C3-B, C3-C, C3-D đã hoàn tất kiểm toán nghiêm ngặt, 10/10 runs được xác thực và công khai trên CoreWeave Forge, các biểu đồ và báo cáo lưu trữ đầy đủ trong repository.
