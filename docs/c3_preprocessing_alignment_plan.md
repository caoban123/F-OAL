# F-OAL — C3 Controlled Input Preprocessing & Normalization Alignment Plan

> **Trạng thái tài liệu:** `PLANNING_ONLY` (Chỉ lập kế hoạch nghiên cứu — **TUYỆT ĐỐI CHƯA TRIỂN KHAI HOẶC CHẠY TRONG GIAI ĐOẠN NÀY**).  
> **Kế thừa từ:** C0 (Metric Alignment: `COMPLETED`), C1 (Controlled Projection Initialization: `MERGED INTO MAIN`), C2 (Controlled ViT Backbone Alignment: `VERIFIED & SIGNED OFF`).

---

## 1. Câu hỏi Nghiên cứu & Bối cảnh Khoa học (Research Question)

### 1.1. Câu hỏi Nghiên cứu
*Trong điều kiện kiểm soát đơn yếu tố (single-factor control) nghiêm ngặt trên tập dữ liệu CIFAR-100 với backbone `vit_base_patch16_224.augreg2_in21k_ft_in1k`, việc sử dụng tiền xử lý hình ảnh của mã nguồn gốc tác giả F-OAL (phóng to nội suy Song tuyến tính **Bilinear** và dải pixel **$[0.0, 1.0]$** không chuẩn hóa `Normalize`) so với tiền xử lý theo chuẩn thiết kế của `timm` (nội suy Song nhị phân **Bicubic** và chuẩn hóa `mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)` về dải **$[-1.0, 1.0]$**) tác động định lượng như thế nào lên biểu diễn đặc trưng ViT và hiệu năng phân loại tăng dần của F-OAL?*

### 1.2. Bối cảnh Kế thừa từ C0, C1 & C2
- **C0 (Metric Alignment):** Đã phân định độc lập $F_{10}^{\text{past}}$ (Paper Eq. 17–18, mẫu số $9$) và $F_{10}^{\text{code}}$ (Official Code, mẫu số $10$).
- **C1 (Smooth Projection Scale, Giả thuyết H2):** Khởi tạo $\mathcal{U}(-1/\sqrt{768}, +1/\sqrt{768})$ loại bỏ triệt để hiện tượng bão hòa Sigmoid ($81.03\% \to 0.00\%$), nâng $\bar{A}$ từ $79.33\%$ lên $83.28\%$ ($+3.95\text{ pp}$), giải thích $33.9\%$ khoảng cách ban đầu.
- **C2 (ViT Pretrained Backbone, Giả thuyết H1):** Trọng số ImageNet-21K (`augreg2_in21k_ft_in1k`) tạo bước nhảy vọt $+8.49\text{ pp}$ trên $\bar{A}$ ($83.28\% \to 91.77\%$) và $+12.59\text{ pp}$ trên $A_{10}$ ($74.84\% \to 87.43\%$), chính thức vượt qua Paper ($91.10\%$) và Official Run ($90.98\%$).
- **Hiện tượng cần giải thích của C3:**
  - Ở C2, bản chạy B1 của nhóm đạt $\bar{A} = 91.77\%$, cao hơn bản chạy mã tác giả Official Run ($90.98\%$) khoảng $+0.79\text{ pp}$.
  - Bên cạnh sự khác biệt về hạt giống ngẫu nhiên (`seed=42` vs `seed=0`), biến số giao thức kỹ thuật cuối cùng chưa được cô lập là **Tiền xử lý hình ảnh (Giả thuyết H4)**:
    - Nhóm: `Resize(224, Bicubic) -> ToTensor() -> Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])` $\implies$ $[-1, 1]$.
    - Tác giả: `Resize(224, Bilinear) -> ToTensor()` $\implies$ $[0, 1]$ (không có hàm `Normalize`).

---

## 2. Giả thuyết Khoa học (Scientific Hypothesis H4)

* **Giả thuyết H4a (Phân phối dữ liệu đầu vào của ViT):** Kiến trúc Vision Transformer (`vit_base_patch16_224.augreg2_in21k_ft_in1k`) được tiền huấn luyện trong dự án Google AugReg với chuẩn hóa ảnh về $[-1.0, 1.0]$ (`mean=0.5, std=0.5`). Khi đưa ảnh ở dải $[0.0, 1.0]$ không chuẩn hóa vào mạng, phân phối đầu vào bị dịch chuyển kỳ vọng ($\Delta \mu = +0.5$), có thể làm thay đổi đáp ứng của tầng Patch Embedding đầu tiên ($W_{\text{patch}} x + b$). Tuy nhiên, cơ chế LayerNorm ngay sau Patch Embedding và trong từng block Transformer có khả năng bù trừ một phần sự dịch chuyển mức độ sáng này.
* **Giả thuyết H4b (Phương pháp Nội suy Hình ảnh):** CIFAR-100 có độ phân giải gốc rất nhỏ ($32 \times 32$), phải phóng to gấp $7$ lần lên $224 \times 224$. Thuật toán nội suy **Bicubic** (bậc 3, bảo toàn cạnh và độ mượt tốt hơn) có thể tạo ra các chi tiết bề mặt và cạnh biên sắc nét hơn so với **Bilinear** (bậc 1), giúp các attention head của ViT trích xuất đặc trưng có độ phân tách cao hơn một lượng nhỏ ($0.3\text{–}1.0\text{ pp}$).
* **Kỳ vọng Khoa học Khách quan:** Tác động của tiền xử lý được dự đoán là một hiệu ứng thứ cấp ($\approx \pm 0.5\text{–}1.2\text{ pp}$), nhỏ hơn nhiều so với tác động của Backbone (H1, $+8.49\text{ pp}$) và Khởi tạo Chiếu (H2, $+3.95\text{ pp}$).

---

## 3. Thiết kế Thực nghiệm Kiểm soát Đơn Yếu tố (Single-Factor Control Design)

Thực nghiệm C3 so sánh đối đầu giữa 2 cấu hình trên cùng tập dữ liệu CIFAR-100:

### 3.1. Hai Cấu hình Đối đầu
1. **P_ctrl — C2 B1 Control Baseline (Chuẩn timm Bicubic $[-1, 1]$):**
   - **Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k`
   - **Nội suy:** `InterpolationMode.BICUBIC`
   - **Dải pixel:** $[-1.0, 1.0]$ (`mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)`)
   - **Ma trận Chiếu $P$:** `official_uniform` (seed 42, cùng mã SHA256)
2. **P_treat — Official Preprocessing Alignment (Chuẩn Tác giả Bilinear $[0, 1]$):**
   - **Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k`
   - **Nội suy:** `InterpolationMode.BILINEAR`
   - **Dải pixel:** $[0.0, 1.0]$ (chỉ gọi `transforms.ToTensor()`, không `Normalize`)
   - **Ma trận Chiếu $P$:** `official_uniform` (seed 42, cùng mã SHA256)

### 3.2. Bảng Kiểm soát Biến số Nghiêm ngặt (Strict Single-Factor Controls)

| Thành phần Thực nghiệm | Cấu hình P_ctrl (Baseline) | Cấu hình P_treat (Treatment) | Trạng thái Kiểm soát |
|:---|:---|:---|:---:|
| **Tiền xử lý & Chuẩn hóa** | Bicubic, $224 \times 224$, $[-1, 1]$ | Bilinear, $224 \times 224$, $[0, 1]$ | **Biến số thử nghiệm duy nhất ($\Delta$)** |
| **Pretrained Checkpoint** | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | **Cố định 100%** |
| **Tập dữ liệu** | CIFAR-100 (50k train, 10k test) | CIFAR-100 (50k train, 10k test) | **Cố định 100%** |
| **Phân chia Task** | 10 tasks $\times$ 10 classes | 10 tasks $\times$ 10 classes | **Cố định 100%** |
| **Thứ tự lớp (Class Order)** | Hoán vị giả ngẫu nhiên `seed=42` | Hoán vị giả ngẫu nhiên `seed=42` | **Cố định 100%** |
| **Thứ tự mẫu stream** | `stream_seed=42` | `stream_seed=42` | **Cố định 100%** |
| **Feature Fusion** | Trung bình CLS 12 blocks + LayerNorm | Trung bình CLS 12 blocks + LayerNorm | **Cố định 100%** |
| **Khởi tạo Ma trận Chiếu $P$** | `official_uniform` | `official_uniform` | **Cố định 100%** |
| **Số chiều Chiếu $D$** | $D = 1000$ | $D = 1000$ | **Cố định 100%** |
| **Hạt giống Ma trận Chiếu** | `projection_seed = 42` | `projection_seed = 42` | **Cố định 100%** |
| **Tính Đồng nhất Ma trận $P$** | Cùng mã băm SHA256 của $P$ | Cùng mã băm SHA256 của $P$ | **Đồng nhất số học tuyệt đối** |
| **Hàm kích hoạt** | Sigmoid $\sigma(\cdot)$ | Sigmoid $\sigma(\cdot)$ | **Cố định 100%** |
| **Hệ số chính quy hóa $\gamma$**| $\gamma = 1.0$ | $\gamma = 1.0$ | **Cố định 100%** |
| **Batch Size ($B$)** | $B = 10$ | $B = 10$ | **Cố định 100%** |
| **Kiểu dữ liệu RLS** | `torch.float64` | `torch.float64` | **Cố định 100%** |

---

## 4. An toàn Bộ nhớ Đệm Đặc trưng (Feature Cache Safety Rules)

Do đầu vào của ViT bị thay đổi hoàn toàn bởi hàm biến đổi hình ảnh, cơ chế Feature Cache bắt buộc phải tuân thủ các quy tắc an toàn:
1. **Khóa Cache riêng biệt:** Tên tệp cache và khóa cache bắt buộc phải phân biệt tiền xử lý (ví dụ: `prep_bicubic_norm` vs `prep_bilinear_raw`).
2. **Kiểm tra Metadata trong Cache:** Tệp cache phải lưu trường:
   ```json
   {
     "preprocessing": {
       "image_size": 224,
       "interpolation": "bilinear",
       "mean": [0.0, 0.0, 0.0],
       "std": [1.0, 1.0, 1.0]
     }
   }
   ```
3. **Từ chối Nạp Cache Sai Lệch (Rejection):** Nếu phát hiện tiền xử lý không khớp, chương trình phải tự động từ chối cache và trích xuất lại từ đầu, ngăn ngừa hoàn toàn việc tái sử dụng đặc trưng Bicubic cho Bilinear.

---

## 5. Chẩn đoán So sánh Đặc trưng Đối khớp (Matched-Preprocessing Diagnostics)

Trước khi thực thi GPU đầy đủ, thực hiện chẩn đoán trên cùng 200 ảnh CIFAR-100:
1. **Thống kê tensor ảnh đầu vào:** Giá trị cực tiểu, cực đại, trung bình và độ lệch chuẩn của pixel ảnh sau biến đổi.
2. **Thống kê đặc trưng ViT sau encoder:** $\|h\|_2$ norm, độ lệch chuẩn đặc trưng, khoảng cách góc giữa vector đặc trưng cùng ảnh sinh ra từ 2 cách tiền xử lý ($\text{Cosine}(h_{\text{bicubic}}, h_{\text{bilinear}})$).
3. **Chỉ số phân tách lớp:** $\mathcal{S}_{\text{intra}}$, $\mathcal{S}_{\text{inter}}$, tỷ số $\mathcal{R}_{\text{sep}}$.

---

## 6. Chỉ số Đánh giá & Tiêu chuẩn Nghiệm thu (Acceptance Criteria)

### 6.1. Chỉ số Đo lường
1. **Average Incremental Accuracy ($\bar{A}$, $\uparrow$)**: Tính trung bình trên 10 giai đoạn.
2. **Last Task Accuracy ($A_{10}$, $\uparrow$)**: Độ chính xác tại giai đoạn 10.
3. **Past Forgetting ($F_{10}^{\text{past}}$, $\downarrow$)**: Định nghĩa Paper (9 tasks).
4. **Code Forgetting ($F_{10}^{\text{code}}$, $\downarrow$)**: Định nghĩa Official Code (10 tasks).
5. **Độ chênh lệch thực nghiệm:**
   $$\Delta_{\text{prep}}(\bar{A}) = \bar{A}_{\text{treat}} - \bar{A}_{\text{ctrl}}$$

### 6.2. Tiêu chuẩn Nghiệm thu (Acceptance Gates)
- [ ] Không có bất kỳ thay đổi nào trong kiến trúc mô hình, mã nguồn RLS hay seed ngẫu nhiên.
- [ ] Ma trận chiếu $P$ được kiểm chứng đồng nhất bitwise (cùng SHA256: `6d3048f4a0fca38799d090113ff9a0c5b9f80a6dfd743a6c4f3d6e9607553a8b`).
- [ ] Cơ chế Cache Safety ngăn chặn thành công việc tải chéo đặc trưng giữa hai chế độ tiền xử lý.
- [ ] Toàn bộ kết quả được đóng gói độc lập, không ghi đè kết quả C0, C1 hay C2.

---

## 7. Ranh giới Triển khai & Cam kết

> [!IMPORTANT]
> **Cam kết Nghiêm ngặt:**  
> Kế hoạch C3 này chỉ mang tính chất chuẩn bị tài liệu thiết kế nghiên cứu cho các giai đoạn tiếp theo.  
> **Không tạo nhánh Git mới, không sửa mã nguồn huấn luyện và không chạy bất kỳ thực nghiệm nào cho C3 trong phiên làm việc hiện tại.**
