# F-OAL — C2 Controlled Pretrained ViT Backbone Alignment Plan

## 1. Research Question & Objective

**Research Question:**  
*Trong điều kiện kiểm soát đơn yếu tố (single-factor control) nghiêm ngặt trên tập dữ liệu CIFAR-100, việc sử dụng trọng số Pretrained Vision Transformer (ViT) được tiền huấn luyện trên ImageNet-21K rồi fine-tune trên ImageNet-1K (`vit_base_patch16_224.augreg2_in21k_ft_in1k`, cấu hình mặc định trong mã nguồn gốc tác giả F-OAL) đóng góp bao nhiêu phần trăm trong khoảng cách hiệu năng còn lại ($\approx 7.6\text{–}7.7\text{ pp}$) so với trọng số chỉ tiền huấn luyện trên ImageNet-1K (`vit_base_patch16_224.augreg_in1k`)?*

### Bối cảnh Kế thừa từ C0 & C1
- **Kết quả C0 (Metric Alignment):** Đã phân định và kiểm chứng toán học độc lập giữa hai công thức tính Forgetting: $F_{10}^{\text{past}}$ (Paper Eq. 17–18, mẫu số $T-1=9$) và $F_{10}^{\text{code}}$ (Official Code, mẫu số $T=10$).
- **Kết quả C1 (Controlled Smooth Projection):** Đã giải quyết giả thuyết **H2**, chứng minh việc thu nhỏ phương sai ma trận ngẫu nhiên về $1/(3d) \approx 0.000434$ loại bỏ hoàn toàn hiện tượng bão hòa Sigmoid ($81.03\% \to 0.00\%$), nâng Average Incremental Accuracy từ $79.33\%$ lên $83.28\%$ (Uniform) và $83.37\%$ (Matched Gaussian), thu hẹp được **$33.9\%$** khoảng cách so với mã nguồn tác giả ($90.98\%$).
- **Khoảng cách còn lại:** Sau C1, vẫn còn khoảng cách $\Delta = 90.98\% - 83.28\% = 7.70\text{ pp}$ giữa bản chạy độc lập của nhóm và kết quả tái hiện mã gốc tác giả.
- **Mục tiêu của C2:** Đo lường chính xác tác động độc lập của **Pretrained ViT Checkpoint (Giả thuyết H1)** mà **không thay đổi bất kỳ biến số nào khác** trong hệ thống.

---

## 2. Giả thuyết Khoa học (Scientific Hypothesis)

* **Giả thuyết H1:** Không gian biểu diễn đặc trưng của Vision Transformer phụ thuộc sâu sắc vào quy mô tập dữ liệu tiền huấn luyện. Một mô hình được tiền huấn luyện trên ImageNet-21K ($\approx 14$ triệu ảnh, 21.841 lớp) tích lũy được các bộ lọc không gian và biểu diễn ngữ nghĩa đa dạng, có độ phân tách tuyến tính và tính khái quát hóa cao hơn đáng kể so với mô hình chỉ được huấn luyện trên ImageNet-1K ($\approx 1.28$ triệu ảnh, 1.000 lớp).
* **Cơ chế tác động trong F-OAL:** F-OAL là phương pháp học giải tích truyền thẳng (Forward-only RLS) với backbone bị đóng băng hoàn toàn. Mọi quyết định phân loại tuyến tính đều dựa trên chất lượng của vector đặc trưng được trích xuất. Khi backbone có tính phân tách lớp cao hơn, ma trận hiệp phương sai của các lớp sẽ có độ trực giao tốt hơn, làm giảm hiện tượng nhiễu giao thoa giữa các task tăng dần trong bài toán bình phương tối thiểu.
* **Kỳ vọng Khoa học Khách quan:**
  - Checkpoint ImageNet-21K được kỳ vọng sẽ cải thiện độ chính xác tăng dần trung bình ($\bar{A}$) và giảm mức độ quên ($F_{10}$).
  - *Lưu ý liêm chính học thuật:* Không đưa ra bất kỳ dự đoán định lượng suy diễn trước (như "chắc chắn tăng 4–5 pp") khi chưa có kết quả thực nghiệm GPU thực tế.

---

## 3. Thiết kế Thực nghiệm Kiểm soát Đơn Yếu tố (Single-Factor Control Design)

Thực nghiệm C2 so sánh đối đầu giữa 2 cấu hình trên cùng một môi trường và điều kiện cố định:

### 3.1. Hai Cấu hình Thực nghiệm
1. **B0 — C1 P1 Baseline Reference (Kiểm soát Đối chứng):**
   - **Backbone Identifier:** `vit_base_patch16_224.augreg_in1k`
   - **Pretrained Dataset:** ImageNet-1K (300 epochs, RandAugment/AugReg)
   - **Timm Hub ID:** `timm/vit_base_patch16_224.augreg_in1k`
   - **Timm Pretrained Tag:** `augreg_in1k`
   - **Projection:** `official_uniform` ($\mathcal{U}(-1/\sqrt{d}, +1/\sqrt{d})$)
2. **B1 — Official Backbone Alignment (Xử lý Thực nghiệm):**
   - **Backbone Identifier:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (tương đương model alias `vit_base_patch16_224` trong `timm`)
   - **Pretrained Dataset:** ImageNet-21K tiền huấn luyện, sau đó fine-tune trên ImageNet-1K
   - **Timm Hub ID:** `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`
   - **Timm Pretrained Tag:** `augreg2_in21k_ft_in1k`
   - **Projection:** `official_uniform` ($\mathcal{U}(-1/\sqrt{d}, +1/\sqrt{d})$)

### 3.2. Bảng Kiểm soát Biến số Nghiêm ngặt (Strict Controls)

| Thành phần Thực nghiệm | Cấu hình B0 (Baseline) | Cấu hình B1 (Treatment) | Trạng thái Kiểm soát |
|:---|:---|:---|:---:|
| **Pretrained Checkpoint** | `vit_base_patch16_224.augreg_in1k` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | **Biến số thử nghiệm duy nhất ($\Delta$)** |
| **Tập dữ liệu** | CIFAR-100 (50k train, 10k test) | CIFAR-100 (50k train, 10k test) | **Cố định 100%** |
| **Phân chia Task** | 10 tasks $\times$ 10 classes | 10 tasks $\times$ 10 classes | **Cố định 100%** |
| **Thứ tự lớp (Class Order)** | Hoán vị giả ngẫu nhiên `seed=42` | Hoán vị giả ngẫu nhiên `seed=42` | **Cố định 100%** |
| **Thứ tự mẫu stream** | `stream_seed=42` | `stream_seed=42` | **Cố định 100%** |
| **Độ phân giải & Tiền xử lý** | $224 \times 224$, Bicubic, $[-1, 1]$ | $224 \times 224$, Bicubic, $[-1, 1]$ | **Cố định 100%** (Không đổi sang Bilinear $[0, 1]$ của C3) |
| **Feature Fusion** | Trung bình CLS qua 12 blocks + LayerNorm | Trung bình CLS qua 12 blocks + LayerNorm | **Cố định 100%** |
| **Kích thước đầu ra Encoder** | $d = 768$ | $d = 768$ | **Cố định 100%** |
| **Khởi tạo Ma trận Chiếu $P$** | `official_uniform` | `official_uniform` | **Cố định 100%** |
| **Số chiều Chiếu $D$** | $D = 1000$ | $D = 1000$ | **Cố định 100%** |
| **Hạt giống Ma trận Chiếu** | `projection_seed = 42` | `projection_seed = 42` | **Cố định 100%** |
| **Tính Đồng nhất của $P$ (Projection Equality)**| Cùng ma trận $P$ ngẫu nhiên, cùng SHA256 fingerprint | Cùng ma trận $P$ ngẫu nhiên, cùng SHA256 fingerprint | **Đồng nhất số học tuyệt đối** |
| **Hàm kích hoạt phi tuyến** | Sigmoid $\sigma(\cdot)$ | Sigmoid $\sigma(\cdot)$ | **Cố định 100%** |
| **Thuật toán Phân loại** | Online Recursive Least Squares (RLS) | Online Recursive Least Squares (RLS) | **Cố định 100%** |
| **Hệ số chính quy hóa $\gamma$**| $\gamma = 1.0$ | $\gamma = 1.0$ | **Cố định 100%** |
| **Batch Size ($B$)** | $B = 10$ | $B = 10$ | **Cố định 100%** |
| **Kiểu dữ liệu RLS** | `torch.float64` | `torch.float64` | **Cố định 100%** |

---

## 4. Thủ tục Xác minh Checkpoint ViT (Checkpoint Verification Procedure)

Để loại trừ nguy cơ tải nhầm mô hình hoặc fallback ngầm về khởi tạo ngẫu nhiên, thủ tục xác minh trước thực nghiệm phải kiểm tra:
1. **Kiểm tra Pretrained Config:** Truy vấn `timm.models.get_pretrained_cfg(model_name)`:
   - B0 phải trả về `tag == 'augreg_in1k'`.
   - B1 phải trả về `tag == 'augreg2_in21k_ft_in1k'`.
2. **Kiểm tra Cấu trúc Mạng:**
   - Số lượng Transformer Blocks: Đúng $12$ blocks (`len(model.network.blocks) == 12`).
   - Embedding Dimension: Đúng $768$ (`model.network.num_features == 768`).
   - Patch Size: $16 \times 16$, Image Size: $224 \times 224$.
3. **Kiểm tra Trọng số Khởi tạo:**
   - Trọng số không được chứa `NaN` hoặc `Inf`.
   - Độ lệch chuẩn trọng số các lớp Attention và MLP phải nằm trong khoảng đặc trưng của checkpoint đã hội tụ (không bằng 0, không trùng với phân phối Kaiming/Xavier ngẫu nhiên).
4. **Kiểm tra Tính Đóng băng (Strict Frozen Enforcement):**
   - Mọi tham số phải thỏa mãn `param.requires_grad == False`.
   - `model.encoder.training == False` tại mọi thời điểm.

---

## 5. An toàn Bộ nhớ Đệm Đặc trưng (Feature Cache Safety)

Khi tính năng lưu cache đặc trưng ViT được kích hoạt để tăng tốc, hệ thống phải tuân thủ nghiêm ngặt nguyên tắc **cô lập cache theo backbone**:
- Tên tệp cache và khóa cache **bắt buộc** phải chứa định danh của backbone (ví dụ `b0_augreg_in1k` hoặc `b1_augreg2_in21k_ft_in1k`).
- Tệp lưu cache phải ghi nhận trường metadata:
  ```python
  {
      "backbone_name": model.backbone_name,
      "pretrained_tag": getattr(model.encoder, "pretrained_tag", "unknown"),
      "feature_fusion": True,
      "indices_sha256": indices_sha,
      "features_sha256": features_sha,
  }
  ```
- Nếu phát hiện tệp cache tồn tại nhưng `backbone_name` hoặc `pretrained_tag` không khớp, chương trình phải tự động bỏ qua (invalidation) và trích xuất lại từ đầu, tuyệt đối không được tải chéo đặc trưng của B0 cho B1.

---

## 6. Chẩn đoán So sánh Đặc trưng Đối khớp (Matched-Feature Diagnostics)

Trước khi chạy toàn bộ 10 tasks trên GPU, thực hiện chẩn đoán trực tiếp trên cùng một tập $N=200$ ảnh CIFAR-100 thực tế (20 ảnh/lớp trên 10 lớp đầu tiên) cho cả B0 và B1:
1. **Thống kê Cơ bản:**
   - Giá trị trung bình và độ lệch chuẩn của vector đặc trưng ViT: $\text{mean}(h), \text{std}(h)$.
   - Phân bố chuẩn độ dài vector (L2 norm): $\|h\|_2$.
2. **Độ tương đồng Cosine (Cosine Similarity):**
   - Độ tương đồng trung bình giữa các mẫu cùng lớp (Intra-class Cosine Similarity): $\mathcal{S}_{\text{intra}}$.
   - Độ tương đồng trung bình giữa các mẫu khác lớp (Inter-class Cosine Similarity): $\mathcal{S}_{\text{inter}}$.
   - Tỷ số phân tách lớp (Class Separation Ratio): $\mathcal{R}_{\text{sep}} = \mathcal{S}_{\text{intra}} / \mathcal{S}_{\text{inter}}$ (tỷ số càng cao, tính phân tách tuyến tính càng tốt).
3. **Số chiều Hữu hiệu (Centered Covariance Effective Rank):**
   - $r_{\text{eff}}(C) = \exp\left(-\sum p_i \ln p_i\right)$ trên ma trận hiệp phương sai đã chuẩn tâm $C = \frac{1}{N-1}(H - \bar{H})^T (H - \bar{H})$.
4. **Phân bố Kích hoạt Sau Chiếu (Post-Projection Diagnostics):**
   - Độ lệch chuẩn tiền kích hoạt $u = hP$.
   - Tỷ lệ bão hòa Sigmoid: $\% (z < 0.01 \lor z > 0.99)$.
   - Số điều kiện ma trận Gram chính quy hóa: $\text{Cond}(G + \gamma I)$.

---

## 7. Quy trình Thực thi Thực nghiệm C2

### 7.1. Cấp độ A — Pilot Thực nghiệm 2 Tasks trên CPU/GPU
- Chạy 2 tasks đầu tiên (100 ảnh train/task, 100 ảnh test/task) trên cả B0 và B1.
- Mục tiêu: Phát hiện sớm lỗi shape mismatch, kiểm tra quy trình tải trọng số, xác nhận tính toán số học ổn định và kiểm thử luồng lưu checkpoint `.pt`.

### 7.2. Cấp độ B — Thực nghiệm Toàn diện 10 Tasks CIFAR-100 trên GPU
- Chạy toàn bộ 10 tasks $\times$ 10 classes (50.000 train, 10.000 test) trên Google Colab GPU (NVIDIA A100 High-RAM).
- Cấu hình chạy:
  - Lượt B0: `configs/c2_b0_augreg_in1k.yaml`
  - Lượt B1: `configs/c2_b1_augreg2_in21k_ft_in1k.yaml`
- Thu thập hiện vật: Toàn bộ ma trận accuracy $10 \times 10$, 10 checkpoints `.pt` mỗi cấu hình, log thực thi, metadata môi trường, và đồng bộ W&B.
- Đóng gói xuất xưởng: `foal_c2_backbone_results.zip`.

---

## 8. Chỉ số Đánh giá & Tiêu chuẩn Nghiệm thu (Acceptance Criteria)

### 8.1. Các Chỉ số Bắt buộc Đo lường
1. **Average Incremental Accuracy ($\bar{A}$, $\uparrow$):** $\bar{A} = \frac{1}{T}\sum_{k=1}^T A_k$.
2. **Last Task Accuracy ($A_{10}$, $\uparrow$):** Độ chính xác tại giai đoạn 10 trên toàn bộ 100 lớp.
3. **Forgetting theo Paper Eq. (17–18) ($F_{10}^{\text{past}}$, $\downarrow$):** Chia trung bình cho $T-1=9$ task quá khứ.
4. **Forgetting theo Code tác giả ($F_{10}^{\text{code}}$, $\downarrow$):** Chia trung bình cho $T=10$ task, kẹp dưới tại 0.
5. **Chênh lệch Thực nghiệm giữa B1 và B0:**
   $$\Delta_{B1 - B0}(\bar{A}) = \bar{A}_{B1} - \bar{A}_{B0}, \qquad \Delta_{B1 - B0}(A_{10}) = A_{10, B1} - A_{10, B0}$$
6. **Mức độ giải thích khoảng cách:**
   $$\text{Gap Explained (\%)} = \frac{\Delta_{B1 - B0}(\bar{A})}{90.98\% - 83.28\%} \times 100\%$$

### 8.2. Tiêu chí Chấp thuận Nghiệm thu (Acceptance Gates)
- [ ] Trọng số của cả B0 và B1 được tải thành công từ các nguồn được xác minh (`timm`).
- [ ] Ma trận chiếu $P$ được kiểm chứng đồng nhất tuyệt đối giữa B0 và B1 (cùng seed 42, cùng fingerprint SHA256).
- [ ] Không có lỗi `NaN` hoặc `Inf` trong toàn bộ quá trình cập nhật RLS.
- [ ] Toàn bộ 10 checkpoints `.pt` của mỗi cấu hình có thể tải lại và đánh giá độc lập.
- [ ] Báo cáo kết quả tuân thủ nghiêm ngặt nguyên tắc thống kê đơn seed ($n=1$, `seed=42`).

---

## 9. Phạm vi & Khả năng Khái quát hóa (Scope & Generalization)

- **Phạm vi Hiện tại:** Thí nghiệm C2 được thực hiện trên **CIFAR-100** nhằm thẩm định giao thức và giải mã khoảng cách kỹ thuật.
- **Giới hạn Khái quát hóa:** Kết quả thu được từ C2 trên CIFAR-100 **không được tự động suy diễn** cho các bộ dữ liệu khác như DTD, FGVCAircraft, Tiny-ImageNet hay Country211. Mỗi bộ dữ liệu có đặc trưng miền riêng và sẽ được mở rộng độc lập trong các giai đoạn tiếp theo.
- C2 là bước thẩm định giao thức thiết yếu, đóng góp vào mục tiêu hoàn thành toàn diện **Core Scope** theo yêu cầu của đồ án HCMUS.
