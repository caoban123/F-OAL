"""Generate the comprehensive master scientific report for DTD dataset.

Mirrors the exact 14-part structure, depth, academic rigor, and style of CIFAR100_COMPLETE_WORK_REPORT.md.
"""
from pathlib import Path

report_content = r"""# BÁO CÁO TỔNG KẾT TOÀN DIỆN VÀ KIỂM TOÁN TÁI HIỆN KHOA HỌC: QUÁ TRÌNH NGHIÊN CỨU F-OAL TRÊN TẬP DỮ LIỆU DESCRIPTABLE TEXTURES DATASET (DTD)

**Dự án:** Tái hiện bài báo *"F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning"* (NeurIPS 2024 Main Conference Track, Rank A* ICORE)  
**Tác giả bài báo:** Huiping Zhuang, Yuchen Liu, Run He, Kai Tong, Ziqian Zeng, Cen Chen, Yi Wang, Lap-Pui Chau  
**Cơ sở đào tạo:** Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS) — Khoa Công nghệ Thông tin — Bộ môn Khoa học Máy tính  
**Môn học:** Nhập môn Học máy (Introduction to Machine Learning)  
**Tập dữ liệu nghiên cứu:** **Describable Textures Dataset (DTD)** — 40 Lớp chọn lọc ($10\text{ tasks} \times 4\text{ lớp/task}$)  
**Nhánh Git thực thi chính:** `experiment/dtd-ablation` (lượt chạy chính thức: [`ca42e672`](https://github.com/caoban123/F-OAL); ghi chú đối chiếu suite CPU cache lịch sử: `747b828`)  
**Kho lưu trữ mã nguồn:** [https://github.com/caoban123/F-OAL](https://github.com/caoban123/F-OAL)  
**Weights & Biases Official Group:** [https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction) (`table4_ablation_dtd`; legacy CPU runs at `747b828`: `phongphanthe70-afsb`)  
**Trạng thái đóng hồ sơ:** **`DTD_REPORT_READY_VERIFIED_SIGN_OFF`**  
**Tập tin đính kèm:**  
- Báo cáo Phân tích Chính thức (`results/dtd/`): [`docs/dtd_ablation_report.md`](dtd_ablation_report.md)  
- Báo cáo PDF Chuyên khảo A4: [`docs/DTD_COMPLETE_WORK_REPORT.pdf`](DTD_COMPLETE_WORK_REPORT.pdf)  
- Notebook Tái hiện GPU Độc lập: [`notebooks/foal_dtd_ablation_colab.ipynb`](../notebooks/foal_dtd_ablation_colab.ipynb)  
- Thư mục Kết quả & Biểu đồ: [`results/dtd/`](../results/dtd/), [`results/dtd_table4_analysis/`](../results/dtd_table4_analysis/), [`results/dtd_c1_analysis/`](../results/dtd_c1_analysis/), [`results/dtd_c2_analysis/`](../results/dtd_c2_analysis/), [`results/dtd_c3_analysis/`](../results/dtd_c3_analysis/)  

---

# MỤC LỤC TỔNG QUAN

- [PHẦN I — TỔNG QUAN DỰ ÁN VÀ BIÊN NIÊN SỬ TIẾN TRÌNH NGHIÊN CỨU](#phần-i--tổng-quan-dự-án-và-biên-niên-sử-tiến-trình-nghiên-cứu)
  - [1. Bối cảnh Dự án và Đặc thù Miền Dữ liệu DTD](#1-bối-cảnh-dự-án-và-đặc-thù-miền-dữ-liệu-dtd)
  - [2. Biên niên sử Nghiên cứu và Giám định Pháp y Khoa học](#2-biên-niên-sử-nghiên-cứu-và-giám-định-pháp-y-khoa-học)
- [PHẦN II — CƠ SỞ LÝ THUYẾT VÀ NỀN TẢNG TOÁN HỌC CỦA F-OAL](#phần-ii--cơ-sở-lý-thuyết-và-nền-tảng-toán-học-của-f-oal)
  - [3. Phát biểu Bài toán: Học Tăng Cường Trực Tuyến Theo Lớp (OCIL) trên DTD](#3-phát-biểu-bài-toán-học-tăng-cường-trực-tuyến-theo-lớp-ocil-trên-dtd)
  - [4. Kiến trúc F-OAL và Luồng Xử lý Giải tích Tuyến Xuôi](#4-kiến-trúc-f-oal-và-luồng-xử-lý-giải-tích-tuyến-xuôi)
  - [5. Thuật toán Cập nhật Đệ quy RLS Woodbury](#5-thuật-toán-cập-nhật-đệ-quy-rls-woodbury)
- [PHẦN III — MÔI TRƯỜNG THỰC NGHIỆM VÀ HẠ TẦNG KỸ THUẬT](#phần-iii--môi-trường-thực-nghiệm-và-hạ-tầng-kỹ-thuật)
  - [6. Cấu hình Phần mềm, Thư viện và Phần cứng Thực thi](#6-cấu-hình-phần-mềm-thư-viện-và-phần-cứng-thực-thi)
  - [7. Đột phá Tối ưu Hóa: Single-Pass Dual-Extractor Caching](#7-đột-phá-tối-ưu-hóa-single-pass-dual-extractor-caching)
- [PHẦN IV — THỰC NGHIỆM C0: CĂN CHỈNH TOÁN HỌC CHỈ SỐ ĐÁNH GIÁ](#phần-iv--thực-nghiệm-c0-căn-chỉnh-toán-học-chỉ-số-đánh-giá)
  - [8. Đối soát và Thống nhất Định nghĩa Độ Quên ($F_{\text{past}}$ vs $F_{\text{code}}$)](#8-đối-soát-và-thống-nhất-định-nghĩa-độ-quên-f_textpast-vs-f_textcode)
- [PHẦN V — THỰC NGHIỆM C1: KHẢO SÁT PHÂN PHỐI CHIẾU NGẪU NHIÊN](#phần-v--thực-nghiệm-c1-khảo-sát-phân-phối-chiếu-ngẫu-nhiên)
  - [9. Hiện tượng Bão hòa Sigmoid và Căn chỉnh Phương sai Ma trận Chiếu](#9-hiện-tượng-bão-hòa-sigmoid-và-căn-chỉnh-phương-sai-ma-trận-chiếu)
- [PHẦN VI — THỰC NGHIỆM C2: CĂN CHỈNH CHECKPOINT PRETRAINED BACKBONE](#phần-vi--thực-nghiệm-c2-căn-chỉnh-checkpoint-pretrained-backbone)
  - [10. Sức mạnh Biểu diễn Đa dạng của ImageNet-21K Đối với Miền Kết cấu Bề mặt](#10-sức-mạnh-biểu-diễn-đa-dạng-của-imagenet-21k-đối-với-miền-kết-cấu-bề-mặt)
- [PHẦN VII — THỰC NGHIỆM C3: NGHIÊN CỨU GIAI THỪA TIỀN XỬ LÝ HÌNH ẢNH](#phần-vii--thực-nghiệm-c3-nghiên-cứu-giai-thừa-tiền-xử-lý-hình-ảnh)
  - [11. Thiết kế Giai thừa 2x2 Interpolation và Normalization trên DTD](#11-thiết-kế-giai-thừa-2x2-interpolation-và-normalization-trên-dtd)
- [PHẦN VIII — THỰC NGHIỆM TABLE 4: ABLATION STUDY THÀNH PHẦN KIẾN TRÚC](#phần-viii--thực-nghiệm-table-4-ablation-study-thành-phần-kiến-trúc)
  - [12. Tái hiện Cú Sụt Giảm Lớn và Phân Rã Hiệu Ứng Bù Trừ Tương Tác Âm](#12-tái-hiện-cú-sụt-giảm-lớn-và-phân-rã-hiệu-ứng-bù-trừ-tương-tác-âm)
- [PHẦN IX — THỰC NGHIỆM TABLE 2 & TABLE 3: SO SÁNH VỚI CÁC BASELINE HÀNG ĐẦU](#phần-ix--thực-nghiệm-table-2--table-3-so-sánh-với-các-baseline-hàng-đầu)
  - [13. Đối Sánh Hiệu Năng và Tốc Độ Huấn Luyện với EASE, SLCA, ER, iCaRL, LwF, EWC](#13-đối-sánh-hiệu-năng-và-tốc-độ-huấn-luyện-với-ease-slca-er-icarl-lwf-ewc)
- [PHẦN X — KIỂM ĐỊNH TÍNH ỔN ĐỊNH TOÁN HỌC ĐA SEED](#phần-x--kiểm-định-tính-ổn-định-toán-học-đa-seed)
  - [14. Phân Tích Phương Sai và Khoảng Tin Cậy Qua 3 Hạt Giống Ngẫu Nhiên](#14-phân-tích-phương-sai-và-khoảng-tin-cậy-qua-3-hạt-giống-ngẫu-nhiên)
- [PHẦN XI — HƯỚNG DẪN CẤU HÌNH BẢNG ĐIỀU KHIỂN VÀ BÁO CÁO TRÊN W&B](#phần-xi--hướng-dẫn-cấu-hình-bảng-điều-khiển-và-báo-cáo-trên-wb)
  - [15. Bố cục 7 Panels Trực quan và Hướng dẫn Đồng bộ Lệnh CLI](#15-bố-cục-7-panels-trực-quan-và-hướng-dẫn-đồng-bộ-lệnh-cli)
- [PHẦN XII — GIẢI ĐÁP 10 CÂU HỎI HỌC THUẬT CHUYÊN SÂU](#phần-xii--giải-đáp-10-câu-hỏi-học-thuật-chuyên-sâu)
- [PHẦN XIII — ĐÁNH GIÁ YÊU CẦU HCMUS VÀ KÝ DUYỆT ĐÓNG HỒ SƠ](#phần-xiii--đánh-giá-yêu-cầu-hcmus-và-ký-duyệt-đóng-hồ-sơ)
- [PHẦN XIV — CÁC PHỤ LỤC TOÁN HỌC VÀ CHỈ MỤC BẰNG CHỨNG](#phần-xiv--các-phụ-lục-toán-học-và-chỉ-mục-bằng-chứng)

---

# PHẦN I — TỔNG QUAN DỰ ÁN VÀ BIÊN NIÊN SỬ TIẾN TRÌNH NGHIÊN CỨU

## 1. Bối cảnh Dự án và Đặc thù Miền Dữ liệu DTD

Trong các nghiên cứu về Học tăng cường theo lớp trực tuyến (Online Class-Incremental Learning — OCIL), tập dữ liệu **Describable Textures Dataset (DTD)** (Cimpoi et al., CVPR 2014) giữ một vị trí mang tính bước ngoặt khoa học. Nếu như CIFAR-100 hay ImageNet chủ yếu bao gồm các thực thể tự nhiên (natural objects) được nhận dạng dựa trên hình thái tổng thể (global shape), thì DTD bao gồm 47 danh mục kết cấu bề mặt mô tả theo thuộc tính tri giác của con người (ví dụ: *banded, bubbly, cracked, fibrous, honeycombed, meshed, striated, zigzagged*).

Sự khác biệt về miền dữ liệu này đặt ra thách thức cực đại cho các mô hình Continual Learning đóng băng encoder:
1. **Độ dịch chuyển phân phối rất lớn (Extreme Domain Shift):** Các patch ViT-B/16 được tiền huấn luyện trên ImageNet nhằm phân loại đối tượng tổng thể thường bị mất mát các tín hiệu không gian tần số cao (local spatial frequencies) ở các tầng transformer cuối cùng.
2. **Kích thước mẫu hạn chế:** Khác với CIFAR-100 có 500 ảnh huấn luyện mỗi lớp, DTD chỉ cung cấp 40 mẫu huấn luyện và 40 mẫu kiểm thử cho mỗi lớp trong giao thức chuẩn của bài báo. Tổng số mẫu huấn luyện cho 10 tasks chỉ là $1.600$ ảnh (ít hơn CIFAR-100 tới **31.25 lần**).
3. **Cú sụt giảm thảm họa trong Table 4 bài báo gốc:** Trong Table 4 của bài báo NeurIPS 2024, tác giả công bố rằng trên CIFAR-100, việc tắt bỏ Feature Fusion (FF) và Smooth Projection (SP) chỉ gây sụt giảm nhẹ $-0.50\text{ pp}$. Tuy nhiên, trên DTD, việc loại bỏ đồng thời hai khối này khiến độ chính xác sụp đổ từ $82.80\%$ xuống $71.20\%$ (sụt giảm thảm họa **$11.60\text{ điểm phần trăm}$**).

Mục tiêu tối thượng của báo cáo này là tái hiện độc lập, kiểm chứng toán học và giải thích cơ chế của hiện tượng sụt giảm $11.60\text{ pp}$ này trên DTD.

---

## 2. Biên niên sử Nghiên cứu và Giám định Pháp y Khoa học

| Giai đoạn / Mốc | Git Commit | Hành động Kỹ thuật & Thực thi | Phát hiện Khoa học & Bằng chứng Số liệu | Trạng thái Pháp y |
|:---|:---:|:---|:---|:---:|
| **P0: Thiết lập Hạ tầng DTD & Đồng nhất Adapter** | `ca42e672` / `cd0843e+` | Chuẩn hóa `src/foal/dtd_datasets.py`, đồng bộ 47 nhãn thật & `samples_per_class=40` sang `src/foal/dataset_adapters/dtd.py`, thêm fingerprint cho `train_dtd.py`. | Kiểm thử đơn vị bảo đảm 40 lớp chọn lọc (1.600 train / 1.600 test), Zero-Replay; công bố minh bạch 7 cặp ảnh trùng byte thô trong `tanganke/dtd`. | `VERIFIED` |
| **P1: Dual Feature Caching** | `ca42e672` | Trích xuất đặc trưng ViT-B/16 cho A0 (FF=True) và A1 (FF=False) trên Tesla T4 (`38.99–42.17s`), tái sử dụng cache cho A2/A3 cùng seed (`0.97–1.27s`). | Loại bỏ sai số trích xuất lại; RLS+Eval chạy dưới $1.0\text{ s}$ (`0.79–1.00s`). | `VERIFIED` |
| **P2: Table 4 Ablation Độc lập (A0..A3 × 3 Seeds)** | `ca42e672` | Chạy 12 cấu hình tách biến độc lập (`Seeds 42, 123, 2026`) lưu tại `results/dtd/`. | **A0: 85.85 ± 0.92%**, **A1: 82.93 ± 0.95%**, **A2: 79.32 ± 1.97%**, **A3: 75.24 ± 1.26%** (s42: `85.08 / 82.08 / 77.81 / 74.58%`). Mức sụt **-10.61 ± 0.50 pp** (Paper: `-11.60 pp`). | `VERIFIED` |
| **P3: Khảo sát Phép Chiếu C1 (P0..P2)** | `d1b4701a` | Đối chứng P0 ($\mathcal{N}(0, 1)$), P1 ($\mathcal{U}$ tác giả), P2 ($\mathcal{N}$ khớp phương sai) tại `results/dtd_c1_full/` (`augreg_in1k`, `torchvision`). | P0 sụp đổ về **61.45%** (bão hòa Sigmoid). P1 đạt **77.93%**, P2 đạt **77.70%** (lệch `-0.24 pp`), chứng minh vai trò quyết định của phương sai ma trận chiếu. | `VERIFIED` |
| **P4: Khảo sát Backbone C2 (B0..B1)** | `d1b4701a` | Đối chứng B0 (`augreg_in1k`) và B1 (`augreg2_in21k_ft_in1k`) tại `results/dtd_c2_full/` (`tanganke/dtd`, `bicubic+norm`). | B1 (**85.12%**) vượt trội B0 (**77.97%**) tới **+7.15 pp**, chứng minh sức mạnh ImageNet-21K là nguyên nhân chính vượt mốc paper `82.8%`. | `VERIFIED` |
| **P5: Khảo sát Tiền xử lý C3 (A–D)** | `d1b4701a` / `44910a1f` | Đối chứng đủ 2×2 Bilinear/Bicubic × Raw/Normalized tại `results/dtd_c3_full/`. | C3-A `85.12%`, C3-B `85.13%`, C3-C `85.13%`, C3-D `85.08%` (biên độ `< 0.05 pp` nhờ LayerNorm của ViT). | `VERIFIED` |
| **P6: Kiểm định Đa Seed Tách biến (42, 123, 2026)** | `ca42e672` | Thử nghiệm 3 bộ `class_order`/`dataset`/`projection`/`stream` độc lập hoàn toàn trên GPU Tesla T4. | A0 đạt **85.85 ± 0.92%** ($\text{CI}_{95\%} = [83.57\%, 88.14\%]$, $p = 0.029 < 0.05$ vs `82.8%`). *(Suite CPU cache cũ ở `747b828` khóa `class_order=42` đạt `85.13 ± 0.11%`.)* | `VERIFIED` |
| **P7: Xuất Bản Phẩm & Kiểm thử Pháp y** | `cd0843e+` | Tạo bảng/biểu đồ chính thức tại `results/dtd_table4_analysis/`, `charts/`, cập nhật LaTeX/Markdown/PDF và bộ pytest `test_dtd_forensic_verification.py`. | Khớp toán học $<10^{-12}$ trên toàn bộ 12 lượt chạy chính thức; kiểm thử đơn vị và kiểm chứng số liệu đồng bộ 100%. | `SIGN-OFF` |

---

# PHẦN II — CƠ SỞ LÝ THUYẾT VÀ NỀN TẢNG TOÁN HỌC CỦA F-OAL

## 3. Phát biểu Bài toán: Học Tăng Cường Trực Tuyến Theo Lớp (OCIL) trên DTD

Bài toán OCIL trên DTD tuân thủ nghiêm ngặt các điều kiện:
- Chuỗi $T = 10$ nhiệm vụ tuần tự $\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_{10}$.
- Mỗi nhiệm vụ $\mathcal{T}_k$ giới thiệu $C_k = 4$ lớp texture mới chưa từng xuất hiện, nâng tổng số lớp tích lũy lên $\sum_{i=1}^k C_i = 4k$ lớp.
- Dữ liệu huấn luyện đến dưới dạng luồng trực tuyến mini-batch $B = 10$, chỉ duyệt qua đúng một lượt duy nhất (Single-pass streaming, $1\text{ epoch}$).
- Tuyệt đối không lưu trữ mẫu dữ liệu cũ vào bộ nhớ đệm (Zero-Replay, Exemplar-Free, $\mathcal{M} = 0$).
- Tại pha suy luận, mô hình không được cung cấp Task-ID và phải dự đoán chính xác nhãn trong toàn bộ $4k$ lớp đã quan sát.

---

## 4. Kiến trúc F-OAL và Luồng Xử lý Giải tích Tuyến Xuôi

```mermaid
flowchart TD
    In["Ảnh Đầu Vào 3x224x224"] --> Enc["Frozen ViT-B/16 Backbone (augreg2_in21k_ft_in1k)"]
    Enc -->|"CLS tokens layers 1..12"| FF["Feature Fusion (FF): h = LayerNorm((1/12) sum h^(l)) in R^768"]
    FF --> SP["Smooth Projection (SP): z = sigma(h * P) in (0, 1)^1000"]
    SP --> RLS["Online Analytic Classifier (Woodbury RLS): min ||ZW - Y||^2 + gamma ||W||^2"]
    RLS --> Out["Dự đoán Nhãn Tuyến Tính: y_hat = argmax (z * W)"]
```

1. **Frozen ViT Backbone:** Toàn bộ $85.8\text{ triệu}$ tham số của ViT-B/16 được đóng băng hoàn toàn ($\nabla_\theta = 0$), loại bỏ hoàn toàn chi phí tính gradient và lưu trữ computation graph.
2. **Feature Fusion (FF):** Thay vì chỉ sử dụng token `[CLS]` của layer 12 (vốn đã bị trừu tượng hóa cao độ), F-OAL lấy trung bình số học token `[CLS]` của toàn bộ 12 khối transformer rồi đưa qua LayerNorm:
   $$\mathbf{h} = \text{LayerNorm}\left(\frac{1}{12} \sum_{l=1}^{12} \mathbf{h}^{(l)}\right) \in \mathbb{R}^{768}$$
3. **Smooth Projection (SP):** Ánh xạ phi tuyến trơn vector đặc trưng $\mathbf{h}$ sang không gian $D = 1000$ chiều thông qua ma trận chiếu ngẫu nhiên $\mathbf{P} \in \mathbb{R}^{768 \times 1000}$ và hàm kích hoạt Sigmoid:
   $$\mathbf{z} = \sigma(\mathbf{h} \mathbf{P}) \in (0, 1)^{1000}, \quad P_{ij} \sim \mathcal{U}\left(-\frac{1}{\sqrt{768}}, +\frac{1}{\sqrt{768}}\right)$$
4. **Analytic Ridge RLS Classifier:** Bộ phân loại phân tích nghiệm đóng tối ưu hàm mất mát bình phương tối thiểu có chính quy hóa $L_2$ ($\gamma = 1.0$):
   $$\mathbf{W}^* = \arg\min_{\mathbf{W}} \frac{1}{2} \|\mathbf{Z} \mathbf{W} - \mathbf{Y}\|_F^2 + \frac{\gamma}{2} \|\mathbf{W}\|_F^2 = (\mathbf{Z}^T \mathbf{Z} + \gamma \mathbf{I})^{-1} \mathbf{Z}^T \mathbf{Y}$$

---

## 5. Thuật toán Cập nhật Đệ quy RLS Woodbury

Với mỗi mini-batch gồm $B = 10$ mẫu $(\mathbf{Z}_t \in \mathbb{R}^{10 \times 1000}, \mathbf{Y}_t \in \mathbb{R}^{10 \times C_t})$, ma trận hiệp phương sai nghịch đảo $\mathbf{R}_t \in \mathbb{R}^{1000 \times 1000}$ và trọng số phân loại $\mathbf{W}_t \in \mathbb{R}^{1000 \times C_t}$ được cập nhật đệ quy:

$$\begin{aligned}
\mathbf{K} &= \mathbf{R}_{t-1} \mathbf{Z}_t^T \in \mathbb{R}^{1000 \times 10} \\
\mathbf{A} &= \mathbf{I}_{10} + \mathbf{Z}_t \mathbf{K} \in \mathbb{R}^{10 \times 10} \\
\mathbf{G} &= \mathbf{A}^{-1} \mathbf{K}^T \in \mathbb{R}^{10 \times 1000} \\
\mathbf{W}_t &= \mathbf{W}_{t-1} + \mathbf{G}^T (\mathbf{Y}_t - \mathbf{Z}_t \mathbf{W}_{t-1}) \\
\mathbf{R}_t &= \mathbf{R}_{t-1} - \mathbf{G}^T \mathbf{K}^T
\end{aligned}$$

Do $B = 10$ rất nhỏ, phép nghịch đảo ma trận $\mathbf{A} \in \mathbb{R}^{10 \times 10}$ diễn ra tức thời trong chưa đầy $0.05\text{ ms}$, trong khi vẫn đảm bảo nghiệm hội tụ toán học tuyệt đối về nghiệm Ridge Regression toàn cục.

---

# PHẦN III — MÔI TRƯỜNG THỰC NGHIỆM VÀ HẠ TẦNG KỸ THUẬT

## 6. Cấu hình Phần mềm, Thư viện và Phần cứng Thực thi

- **Môi trường thực thi chính thức (`results/dtd/`, Google Colab GPU):** Linux 6.6.159+, Python 3.13.16, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, timm 1.0.31, datasets 4.8.5, wandb 0.31.0 trên **NVIDIA Tesla T4** (14.56 GB VRAM, CUDA 13.0). Commit thực thi: `ca42e672` (nhánh `experiment/dtd-ablation`; lưu ý 11/12 metadata ghi `git_dirty=true` do các thư mục `results/` chưa theo dõi tồn tại trước bản vá `9f955ca`).
- **Môi trường phát triển / kiểm chứng CPU cục bộ:** Windows 11, Python 3.14, PyTorch CPU (cũng là môi trường chạy suite feature-cache lịch sử ở commit `747b828` với `85.26%` / `85.13 ± 0.11%`).

---

## 7. Đột phá Tối ưu Hóa: Caching Đặc trưng ViT

Do đặc trưng của ViT-B/16 chiếm >95% tổng chi phí tính toán, việc kích hoạt bộ đệm đặc trưng (`feature_cache_enabled = true`) cho phép:
- Trên Google Colab Tesla T4 (`results/dtd/`), pha trích xuất đặc trưng ViT của `A0` và `A1` tốn **$38.99\text{--}42.17\text{ giây}$** cho $3.200$ lượt forward ảnh (1.600 train + 1.600 test). Các điều kiện `A2`/`A3` cùng hạt giống tái sử dụng bộ đệm đặc trưng từ `A0`/`A1`, rút ngắn wall-clock xuống còn **$0.97\text{--}1.27\text{ giây}$**.
- Pha giải tích RLS và đánh giá (`rls_and_eval_seconds`) trên cả 4 cấu hình chỉ tiêu tốn **$0.79\text{--}1.00\text{ giây}$** toàn chuỗi 10 tác vụ.
- Trên CPU cục bộ ở nhánh phát triển trước đó (`747b828`), bộ `DualFeatureExtractor` hoàn thành lập cache 3.200 ảnh trong 8.6 phút và chạy RLS trong `1.1–1.4 giây`.

---

# PHẦN IV — THỰC NGHIỆM C0: CĂN CHỈNH TOÁN HỌC CHỈ SỐ ĐÁNH GIÁ

## 8. Đối soát và Thống nhất Định nghĩa Độ Quên ($F_{\text{past}}$ vs $F_{\text{code}}$)

Một đóng góp quan trọng của dự án là phân định rạch ròi hai định nghĩa độ quên đang được dùng song song trong tài liệu và mã nguồn thực tế:

1. **Định nghĩa Chuẩn Văn hiến có dấu ($F_{\text{past}}$ — 9 tác vụ quá khứ):**
   Chia trung bình trên $T - 1 = 9$ nhiệm vụ quá khứ sử dụng đỉnh lịch sử $\ell \in \{j, \dots, T-1\}$ (tức trước giai đoạn $T$, cho phép giá trị âm nếu có backward transfer):
   $$F_{\text{past}} = \frac{1}{T-1} \sum_{j=1}^{T-1} \left( \max_{\ell \in \{j, \dots, T-1\}} R_{\ell, j} - R_{T, j} \right)$$
   Trên `results/dtd/`, Full F-OAL đạt $F_{\text{past}} = \mathbf{8.22 \pm 0.36\%}$ (Seed 42: $\mathbf{8.61\%}$).

2. **Định nghĩa trong Code Gốc Tác giả ($F_{\text{code}}$ — 10 tác vụ có chặn dưới 0):**
   Lấy max bao gồm cả giai đoạn cuối $\ell \in \{j, \dots, T\}$ và chia cho $T = 10$:
   $$F_{\text{code}} = \frac{1}{T} \sum_{j=1}^{T} \left( \max_{\ell \in \{j, \dots, T\}} R_{\ell, j} - R_{T, j} \right)$$
   Khi không có backward transfer dương ở giai đoạn cuối ($R_{T,j} \le \max_{\ell < T} R_{\ell,j}$), đẳng thức $F_{\text{code}} = \frac{9}{10} F_{\text{past}}$ đúng tuyệt đối (ví dụ A0 Seed 42: $7.75\% = 0.9 \times 8.6111\%$). Trung bình 3 seeds đạt $F_{\text{code}} = \mathbf{7.46 \pm 0.25\%}$.

---

# PHẦN V — THỰC NGHIỆM C1: KHẢO SÁT PHÂN PHỐI CHIẾU NGẪU NHIÊN

## 9. Hiện tượng Bão hòa Sigmoid và Căn chỉnh Phương sai Ma trận Chiếu

Kết quả đo đạc trên GPU Tesla T4 (`results/dtd_c1_full/`, commit `d1b4701a`, dùng nguồn `torchvision` fallback, `augreg_in1k`, `bicubic+norm`, Seed 42):

| Cấu hình | Phân phối Ma trận $\mathbf{P}$ | Phương sai Phần tử | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{\text{past}}$ (%) | Hiện tượng Khoa học |
|:---|:---|:---:|---:|---:|---:|:---|
| **P0** (`gaussian_std1`) | $\mathcal{N}(0, 1.0)$ | $1.0$ | **61.45%** | **53.19%** | **10.42%** | Bão hòa Sigmoid nghiêm trọng, sụt giảm **$-16.48\text{ pp}$** so với P1 |
| **P1** (`official_uniform`) | $\mathcal{U}[-1/\sqrt{768}, +1/\sqrt{768}]$ | $\frac{1}{3 \times 768} \approx 0.000434$ | **77.93%** | **68.69%** | **10.00%** | Tối ưu vùng tuyến tính quanh 0 của Sigmoid ($\sigma'(0)=0.25$) |
| **P2** (`gaussian_matched_variance`) | $\mathcal{N}(0, \frac{1}{3 \times 768})$ | $\frac{1}{3 \times 768} \approx 0.000434$ | **77.70%** | **68.69%** | **9.51%** | Khớp phương sai lý thuyết với P1; độ lệch chỉ $-0.24\text{ pp}$ |

> **Phát hiện Cốt lõi C1:** Chuẩn độ dài của vector đặc trưng ViT $\|\mathbf{h}\|_2 \approx \sqrt{768} \approx 27.7$. Khi nhân với $\mathbf{P} \sim \mathcal{N}(0, 1)$, tích vô hướng đẩy đầu vào Sigmoid vào vùng bão hòa, triệt tiêu độ nhạy biểu diễn ($-16.48\text{ pp}$). Chỉ cần khớp đúng phương sai ($\sigma = 1/48$), khởi tạo Gaussian P2 khôi phục hiệu năng gần như trùng khớp P1 (`77.70%` vs `77.93%`). *(Suite CPU cache trên `tanganke/dtd` với `augreg2` cũng ghi nhận quy luật tương tự: `70.82% / 85.26% / 85.30%`).*

---

# PHẦN VI — THỰC NGHIỆM C2: CĂN CHỈNH CHECKPOINT PRETRAINED BACKBONE

## 10. Sức mạnh Biểu diễn Đa dạng của ImageNet-21K Đối với Miền Kết cấu Bề mặt

Kết quả đo đạc trên GPU Tesla T4 (`results/dtd_c2_full/`, commit `d1b4701a`, `tanganke/dtd`, `bicubic+norm`, Seed 42):

| Checkpoint Backbone | Dữ liệu Tiền Huấn Luyện | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{\text{past}}$ (%) | Đánh giá Biểu diễn |
|:---|:---|---:|---:|---:|:---|
| **B0** (`augreg_in1k`) | ImageNet-1K (1.2 triệu ảnh, 1.000 lớp) | **77.97%** | **69.19%** | **9.93%** | Thiếu biểu diễn kết cấu tần số cao; độ quên cao |
| **B1** (`augreg2_in21k_ft_in1k`) | ImageNet-21K (14.2 triệu ảnh, 21.841 lớp) | **85.12%** | **77.19%** | **8.26%** | **Vượt trội +7.15 pp**, độ quên giảm $1.67\text{ pp}$ |

Kho tri thức bao quát từ 14.2 triệu ảnh của ImageNet-21K là động lực chính tạo nên mức chênh lệch tích cực $+2.28\text{--}+3.05\text{ pp}$ của F-OAL hiện hành so với mốc `82.8%` trong bài báo gốc.

---

# PHẦN VII — THỰC NGHIỆM C3: NGHIÊN CỨU GIAI THỪA TIỀN XỬ LÝ HÌNH ẢNH

## 11. Thiết kế Giai thừa $2 \times 2$ Interpolation và Normalization trên DTD (`results/dtd_c3_full/`)

| Điều kiện | Phép Nội Suy | Khoảng Giá Trị Điểm Ảnh | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{\text{past}}$ (%) |
|:---|:---:|:---:|---:|---:|---:|
| **C3-A** | Bicubic | Chuẩn hóa Mean/Std $[-1, 1]$ | **85.12%** | **77.19%** | **8.26%** |
| **C3-B** | Bilinear | Chuẩn hóa Mean/Std $[-1, 1]$ | **85.13%** | **76.88%** | **8.47%** |
| **C3-C** | Bicubic | Giá trị gốc $[0, 1]$ | **85.13%** | **76.62%** | **8.54%** |
| **C3-D** (= A0 s42) | Bilinear | Giá trị gốc $[0, 1]$ (Cấu hình Table 4) | **85.08%** | **76.06%** | **8.61%** |

Khác biệt giữa cả 4 điều kiện tiền xử lý chỉ dao động trong biên độ **$0.05\text{ pp}$** (`85.08%–85.13%`), khẳng định cơ chế LayerNorm của ViT triệt tiêu gần như hoàn toàn sai khác từ nội suy ảnh và chuẩn hóa tuyến tính trên tập DTD. *(Ghi chú: Thư mục `c3_a_bicubic_normalized` chỉ lưu `metadata.json`/`metrics.json` mà không xuất `matrix.json`/`summary.json`/`task_accuracy.csv` như C3-B/C/D.)*

---

# PHẦN VIII — THỰC NGHIỆM TABLE 4: ABLATION STUDY THÀNH PHẦN KIẾN TRÚC

## 12. Tái hiện Cú Sụt Giảm Lớn và Phân Rã Hiệu Ứng Giai thừa $2 \times 2$ (`results/dtd/`)

### 12.1. Bảng Số Liệu Chính Thức 4 Cấu Hình qua 3 Hạt giống Tách biến Độc lập (`42, 123, 2026`)

| Điều kiện | FF | SP | $D$ | Seed 42 | Seed 123 | Seed 2026 | **Mean $\pm$ Std $\bar{A}$** | Mean $A_{10}$ | Mean $F_{\text{past}}$ | $T_{\text{RLS+Eval}}$ | Paper Table 4 |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A0 (Full F-OAL)** | $\checkmark$ | $\checkmark$ | 1000 | 85.08% | 86.87% | 85.60% | **85.85 ± 0.92%** | **77.31%** | 8.22% | 0.88–0.89s | 82.8% |
| **A1 (w/o FF)** | $\times$ | $\checkmark$ | 1000 | 82.08% | 83.95% | 82.75% | **82.93 ± 0.95%** | **75.56%** | 7.69% | 0.90–0.94s | 80.5% |
| **A2 (w/o SP)** | $\checkmark$ | $\times$ | 768 | 77.81% | 81.54% | 78.60% | **79.32 ± 1.97%** | **74.48%** | 5.88% | 0.79–1.00s | 79.3% |
| **A3 (w/o Both)** | $\times$ | $\times$ | 768 | 74.58% | 76.70% | 74.45% | **75.24 ± 1.26%** | **72.88%** | 4.54% | 0.82–1.00s | 71.2% |

*(Lưu ý đối chiếu: Suite CPU cache lịch sử ở commit `747b828` cho kết quả `85.26 / 82.55 / 78.64 / 72.97%` trên Seed 42; thử nghiệm 80 ảnh/lớp chưa lấy mẫu con trước `acea57e` đạt `87.20 / 85.01 / 84.35 / 81.10%` và đã được loại bỏ khỏi báo cáo chính thức.)*

### 12.2. Phân Rã Hiệu Ứng Giai Thừa $2 \times 2$ & Kiểm định $t$ Ghép cặp ($n=3, \text{df}=2$)

$$\begin{aligned}
\Delta_{\text{Joint}} &= A0 - A3 = \mathbf{+10.61 \pm 0.50\text{ pp}} \quad (t_2 = 36.83,\, p = 0.00074;\; \text{Seed 42: }+10.50\text{ pp};\; \text{Paper: }+11.60\text{ pp}) \\
E_{\text{FF}|\text{SP}} &= A0 - A1 = \mathbf{+2.93 \pm 0.07\text{ pp}} \quad (t_2 = 68.56,\, p = 0.00021;\; \text{Seed 42: }+3.00\text{ pp};\; \text{Paper: }+2.30\text{ pp}) \\
E_{\text{FF}|\neg\text{SP}} &= A2 - A3 = \mathbf{+4.07 \pm 0.81\text{ pp}} \quad (t_2 = 8.71,\, p = 0.01292;\; \text{Seed 42: }+3.23\text{ pp};\; \text{Paper: }+8.10\text{ pp}) \\
E_{\text{SP}|\text{FF}} &= A0 - A2 = \mathbf{+6.53 \pm 1.05\text{ pp}} \quad (t_2 = 10.74,\, p = 0.00855;\; \text{Seed 42: }+7.28\text{ pp};\; \text{Paper: }+3.50\text{ pp}) \\
E_{\text{SP}|\neg\text{FF}} &= A1 - A3 = \mathbf{+7.68 \pm 0.55\text{ pp}} \quad (t_2 = 24.30,\, p = 0.00169;\; \text{Seed 42: }+7.50\text{ pp};\; \text{Paper: }+9.30\text{ pp})
\end{aligned}$$

- **Tác động chính trung bình (Main Effects):**
  $$E_{\text{FF}} = \mathbf{+3.50 \pm 0.38\text{ pp}} \quad (t_2 = 15.80,\, p = 0.0040), \qquad E_{\text{SP}} = \mathbf{+7.11 \pm 0.72\text{ pp}} \quad (t_2 = 17.05,\, p = 0.0034)$$
- **Độ tương phản tương tác (Interaction Contrast):**
  $$I_{\text{DiD}} = (A0 - A1) - (A2 - A3) = \mathbf{-1.15 \pm 0.86\text{ pp}} \quad (I_{\text{half}} = -0.57 \pm 0.43\text{ pp},\; t_2 = -2.32,\, p = 0.1458)$$

> **Biện Giải Cơ Chế:** Cả hai tác động chính của SP ($+7.11\text{ pp}$) và FF ($+3.50\text{ pp}$) đều có ý nghĩa thống kê vững chắc trên cả 3 hạt giống độc lập ($p < 0.005$), trong khi tương tác DiD âm nhẹ ($-1.15\text{ pp}$) không đạt ngưỡng ý nghĩa ($p = 0.146$), cho thấy đóng góp của hai khối chủ yếu mang tính cộng hưởng bổ sung cho nhau và sụp đổ nặng ($-10.61\text{ pp}$) khi thiếu cả hai.

---

# PHẦN IX — THỰC NGHIỆM TABLE 2 & TABLE 3: SO SÁNH VỚI CÁC BASELINE HÀNG ĐẦU

## 13. Đối Sánh Hiệu Năng và Tốc Độ Huấn Luyện với EASE, SLCA, ER, iCaRL, LwF, EWC

| Phương Pháp | Họ Thuật Toán | Bộ Nhớ Replay $\mathcal{M}$ | $\bar{A}$ (%) | $A_{\text{last}}$ (%) | Forgetting (%) | Thời Gian Huấn Luyện (Table 3) |
|:---|:---|:---:|---:|---:|---:|---:|
| **EWC** (PNAS 2017) | Regularization | 0 | 41.2% | 21.5% | 58.4% | — |
| **LwF** (TPAMI 2018) | Distillation | 0 | 55.8% | 48.2% | 12.6% | — |
| **ER** (ICRA 2019) | Experience Replay | 5.000 mẫu | 73.2% | 68.1% | 18.5% | 21 giây |
| **iCaRL** (CVPR 2017) | Replay + NCM | 5.000 mẫu | 74.5% | 69.8% | 14.2% | — |
| **SLCA** (ICCV 2023) | Alignment | 0 | 72.8% | 67.5% | 9.8% | — |
| **EASE** (CVPR 2024) | Subspace Ensemble | 0 | 76.0% | 71.4% | 8.1% | 139 giây |
| **F-OAL (Paper Báo cáo)** | Forward-only Analytic | 0 | **82.8%** | **78.2%** | **5.2%** | **8 giây** |
| **F-OAL (`results/dtd/`, 3 seeds)** | Forward-only Analytic | 0 | **85.85 ± 0.92%** | **77.31 ± 1.17%** | **8.22 ± 0.36%** | **0.88s RLS** *(+40s ViT)* |

- **Vượt trội baseline tốt nhất:** F-OAL (`results/dtd/` trung bình 3 seeds) vượt qua baseline Exemplar-Free tốt nhất (**EASE**, `76.0%`) tới **$+9.85\text{ điểm phần trăm}$** trên DTD (`+9.08 pp` ở Seed 42).
- **Tốc độ:** Pha cập nhật RLS + đánh giá chạy dưới 1 giây (`0.79–1.00s` trên Tesla T4; `40.4–43.6s` tổng nếu tính trích xuất ViT từ đầu), nhanh hơn đáng kể so với phương pháp fine-tuning lặp lại của EASE (`139 giây`).

---

# PHẦN X — KIỂM ĐỊNH ĐA HẠT GIỐNG TÁCH BIẾN ĐỘC LẬP (`results/dtd/`)

## 14. Phân Tích Phương Sai và Khoảng Tin Cậy Qua 3 Bộ Seeds Độc Lập (`42, 123, 2026`)

| Điều kiện / Seed | $\bar{A}$ (%) | $A_{10}$ (%) | $F_{\text{past}}$ (%) | $F_{\text{code}}$ (%) |
|:---|---:|---:|---:|---:|
| **A0 — Seed 42** (`dtd_a0_full`) | 85.08% | 76.06% | 8.61% | 7.75% |
| **A0 — Seed 123** (`a0_seed123`) | 86.87% | 78.38% | 7.92% | 7.31% |
| **A0 — Seed 2026** (`a0_seed2026`) | 85.60% | 77.50% | 8.12% | 7.31% |
| **A0 — Tổng hợp ($N=3$)** | **85.85 ± 0.92%** | **77.31 ± 1.17%** | **8.22 ± 0.36%** | **7.46 ± 0.25%** |
| **A1 / A2 / A3 — Tổng hợp ($N=3$)** | `82.93 / 79.32 / 75.24%` | `75.56 / 74.48 / 72.88%` | `7.69 / 5.88 / 4.54%` | `6.98 / 5.37 / 4.65%` |

- **Khoảng tin cậy 95% của A0** ($t_{2, 0.025} = 4.3027$): $[83.57\%, 88.14\%]$, kiểm định một mẫu vượt mốc `82.8%` cho $t_2 = 5.75, p = 0.0290 < 0.05$.
- *(Đối chiếu minh bạch: Bộ suite CPU cache lịch sử ở `results/dtd_multiseed_1600_summary.json` chỉ thay đổi `projection`/`stream` trên cùng `class_order=42` nên có độ lệch chuẩn thấp hơn giả định: `85.13 ± 0.11%`.)*

---

# PHẦN XI — HƯỚNG DẪN CẤU HÌNH BẢNG ĐIỀU KHIỂN VÀ BÁO CÁO TRÊN W&B

## 15. Bố cục 7 Panels Trực quan và Hướng dẫn Đồng bộ Lệnh CLI

### 15.1. Bố Cục Panels Đề Xuất Cho Báo Cáo W&B

```
┌────────────────────────────────────────────────────────────────────────┐
│ W&B REPORT: F-OAL DTD REPRODUCTION & EXPERIMENTAL ANALYSIS             │
├──────────────────────────────────┬─────────────────────────────────────┤
│ Panel 1: Stage Accuracy Trajectory│ Panel 2: Table 4 Ablation Bar Chart │
│ - Chart Type: Line Plot           │ - Chart Type: Grouped Bar Plot      │
│ - X: stage (1..10)                │ - X: Condition (A0, A1, A2, A3)     │
│ - Y: accuracy                     │ - Y: average_incremental_accuracy   │
│ - Runs: A0, A1, A2, A3            │ - Group: Paper vs Ours              │
│ [File: fig5_stage_accuracy...png] │ [File: fig1_table4_accuracy...png]  │
├──────────────────────────────────┼─────────────────────────────────────┤
│ Panel 3: Baseline Comparison      │ Panel 4: Test Accuracy Heatmap      │
│ - Chart Type: Bar Chart           │ - Chart Type: 2D Matrix Heatmap     │
│ - Methods: EWC, LwF, ER, EASE,    │ - Matrix: 10x10 Stage vs Task       │
│   SLCA, F-OAL Paper, F-OAL Ours   │ - Run: A0_Full_FOAL                 │
├──────────────────────────────────┼─────────────────────────────────────┤
│ Panel 5: C1 Projection Init       │ Panel 6: C2 Backbone Comparison     │
│ - Chart Type: Dual Bar Chart      │ - Chart Type: Bar Comparison        │
│ - Metrics: A_avg & F_past         │ - C2: augreg_1k vs augreg2_21k      │
│ - Runs: P0, P1, P2                │                                     │
├──────────────────────────────────┴─────────────────────────────────────┤
│ Panel 7: Multi-Seed Stability (N=3 Error Bars)                         │
│ - Chart Type: Scatter / Bar with Error Margin                          │
│ - X: Seed (42, 123, 2026), Line: Mean = 85.85% (Std = 0.92%)           │
└────────────────────────────────────────────────────────────────────────┘
```

### 15.2. Hướng Dẫn Đồng Bộ Offline Lên W&B Server
Khi thực thi với cờ `--wandb-offline`, toàn bộ dữ liệu lưu tại thư mục `wandb/`. Để đồng bộ lên máy chủ:
```bash
# 1. Đăng nhập W&B CoreWeave Forge
wandb login --host=https://forge.coreweave.com <YOUR_API_KEY>

# 2. Đồng bộ tất cả các run DTD
wandb sync wandb/offline-run-*
```

---

# PHẦN XII — GIẢI ĐÁP 10 CÂU HỎI HỌC THUẬT CHUYÊN SÂU

1. **Tại sao Feature Fusion lại đóng vai trò nhất quán trên DTD ($+3.50 \pm 0.38\text{ pp}$)?**  
   *Trả lời:* Các lớp vân bề mặt DTD đòi hỏi các biểu diễn cục bộ ở tầng thấp và giữa (biên cạnh, tần số lặp lại, sọc, vân). Layer 12 của ViT có xu hướng trừu tượng hóa quá mức khiến mất đi các chi tiết này. Feature Fusion trung bình hóa 12 layers giúp phục hồi các tín hiệu texture đó ($+2.93\text{ pp}$ khi có SP; $+4.07\text{ pp}$ khi không có SP).
2. **Tại sao Smooth Projection lại có tác động lớn nhất trên DTD ($+7.11 \pm 0.72\text{ pp}$)?**  
   *Trả lời:* Không gian 768 chiều chưa đủ phân tách tuyến tính đối với các lớp texture phức tạp. Chiếu phi tuyến qua Sigmoid mở rộng không gian lên 1000 chiều và nén các giá trị cực trị, tạo ra biên phân cách trơn tru giúp bộ phân loại Ridge RLS tìm được siêu phẳng tối ưu ($+6.53\text{ pp}$ khi có FF; $+7.68\text{ pp}$ khi không có FF).
3. **Cú sụt giảm $-10.61\text{ pp}$ khi tắt cả hai thành phần có ý nghĩa lý thuyết gì?**  
   *Trả lời:* Cả FF và SP đều cần thiết để tách biệt các kết cấu ngoài phân phối (out-of-distribution): khi loại bỏ đồng thời, độ chính xác sụt từ `85.85%` về `75.24%` (`-10.61 ± 0.50 pp`, tái hiện sát mốc `-11.60 pp` của paper), với tương tác DiD âm nhẹ không đáng kể (`-1.15 ± 0.86 pp`, $p = 0.146$).
4. **Tại sao ViT tiền huấn luyện trên ImageNet-21K lại vượt trội ImageNet-1K tới $+7.15\text{ pp}$ trên DTD?**  
   *Trả lời:* ImageNet-21K bao gồm 14.2 triệu ảnh thuộc 21,841 lớp, bao quát hàng ngàn loại vật liệu và kết cấu tự nhiên. Trong kiểm chứng C2 (`results/dtd_c2_full/`), `augreg2_in21k_ft_in1k` đạt `85.12%` so với `77.97%` của `augreg_in1k` (`+7.15 pp`).
5. **Hiện tượng bão hòa Sigmoid ở khởi tạo Gaussian $\sigma=1.0$ (C1-P0) diễn ra như thế nào?**  
   *Trả lời:* Chuẩn độ lớn của vector đặc trưng khiến tích vô hướng $h P$ có phương sai quá lớn. Khi đưa qua hàm Sigmoid $\sigma(x)$, các giá trị bị đẩy vào vùng bão hòa, làm mất độ phân tách và khiến độ chính xác giảm `16.48 pp` trong kiểm chứng C1 trên `torchvision` (`61.45%` vs `77.93%`).
6. **Vì sao pha RLS của F-OAL chạy dưới 1 giây trên GPU?**  
   *Trả lời:* EASE phải liên tục xây dựng các không gian con (subspaces) và tối ưu lặp bằng gradient descent qua nhiều epoch (`139s`). F-OAL chuyển hoàn toàn việc học sang cập nhật giải tích Woodbury không cần gradient, mỗi mini-batch chỉ giải nghịch đảo ma trận $10 \times 10$, giúp pha `rls_and_eval_seconds` chỉ mất `0.79–1.00 giây` trên Tesla T4 sau `39–42 giây` trích xuất đặc trưng ViT.
7. **Bộ nhớ RAM/VRAM của F-OAL trên DTD bị chi phối bởi thành phần nào?**  
   *Trả lời:* Bộ nhớ chủ yếu dành cho tham số ViT-B/16 (~330 MB) và activation của batch 10 ảnh (~1.5 GB). Ma trận hiệp phương sai RLS $R \in \mathbb{R}^{1000 \times 1000}$ kiểu `float64` chỉ chiếm đúng 8 MB, hoàn toàn miễn nhiễm với bộ nhớ lưu trữ replay buffer khổng lồ của ER hay iCaRL.
8. **Hiện tượng Quên Lãng (Catastrophic Forgetting) được kiểm soát như thế nào trong F-OAL?**  
   *Trả lời:* Ma trận $R_t$ tích lũy chính xác thông tin thống kê $Z^T Z$ của tất cả các mẫu trong quá khứ mà không bị nén mất mát. Bổ đề Woodbury đảm bảo nghiệm tại mọi thời điểm tương đương với việc huấn luyện Ridge Regression trên toàn bộ dữ liệu lịch sử.
9. **Tại sao F-OAL trên DTD có độ ổn định rất cao trước tiền xử lý hình ảnh (C3)?**  
   *Trả lời:* ViT có tầng LayerNorm ở đầu mỗi Transformer block và ở đầu ra, tự động chuẩn hóa kỳ vọng và phương sai của các patch token, triệt tiêu phần lớn ảnh hưởng của độ sáng trung bình giữa dải $[0, 1]$ và $[-1, 1]$.
10. **Giới hạn khoa học lớn nhất của F-OAL khi triển khai thực tế là gì?**  
    *Trả lời:* Mô hình phụ thuộc 100% vào chất lượng không gian đặc trưng của encoder đóng băng. Nếu miền dữ liệu mới hoàn toàn xa lạ với dữ liệu tiền huấn luyện, việc không fine-tune backbone sẽ trở thành rào cản hiệu năng.

---

# PHẦN XIII — ĐÁNH GIÁ YÊU CẦU HCMUS VÀ KÝ DUYỆT ĐÓNG HỒ SƠ

| Yêu Cầu Học Thuật HCMUS | Tiêu Chuẩn Giám Định Pháp Y | Hiện Vật Chứng Thực Trong Repo | Kết Quả Thẩm Định |
|:---|:---|:---|:---:|
| **Tái hiện Độc lập (Clean-room)** | Tự xây dựng mã nguồn dựa trên toán học bài báo | `src/foal/` (backbone, projection, rls, dtd_datasets) | **PASS (100% Độc lập)** |
| **Headline Benchmark (Table 2)** | Đối sánh toàn diện với các baseline hàng đầu | Table 2 đối sánh 7 phương pháp (EWC, LwF, ER, EASE, v.v.) | **PASS (Vượt EASE +9.85 pp)** |
| **Ablation Study (Table 4)** | Phân rã đầy đủ ma trận giai thừa 2x2 (FF x SP) | `results/dtd/` & `results/dtd_table4_analysis/` (12 runs) | **PASS (Sụt -10.61 ± 0.50 pp)** |
| **Nghiên cứu Kiểm soát (C0–C3)** | Khảo sát mọi giả thuyết độ lệch kỹ thuật | `results/dtd_c1_full/`, `dtd_c2_full/`, `dtd_c3_full/` | **PASS (100% Số liệu đo đạc thực tế)** |
| **Độ Ổn Định Đa Seed** | Kiểm định tính tái lặp trên 3 bộ seed độc lập | `results/dtd/` (`42, 123, 2026`, $N=3$) | **PASS ($85.85 \pm 0.92\%$)** |
| **Tính Khả Lặp Pháp Y** | Lưu vết Git SHA, seed, fingerprint và metadata | `metadata.json`, `metrics.json`, `matrix.json`, `task_accuracy.csv` | **PASS (Truy vết & đối chiếu <1e-12)** |
| **Báo Cáo Ấn Phẩm PDF & Markdown** | Báo cáo tổng hợp + chuyên khảo A4 tiếng Việt | `docs/DTD_COMPLETE_WORK_REPORT.pdf` & `docs/dtd_ablation_report.md` | **PASS (Đồng bộ 100%)** |

**KÝ DUYỆT ĐÓNG HỒ SƠ THỰC NGHIỆM DTD:**  
*Trạng thái chính thức: `DTD_REPORT_READY_VERIFIED_SIGN_OFF`*  
*Toàn bộ mục tiêu nghiên cứu và tái hiện trên tập dữ liệu DTD đã đồng bộ 100% với bộ dữ liệu chính thức `results/dtd/`.*

---

# PHẦN XIV — CÁC PHỤ LỤC TOÁN HỌC VÀ CHỈ MỤC BẰNG CHỨNG

## Phụ lục A: Ma trận Độ chính xác Kiểm thử $10 \times 10$ của Full F-OAL Seed 42 (`results/dtd/dtd_a0_full/matrix.json`)

| Giai đoạn ($k$) | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 | $A_k$ (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| **Stage 1** | **98.12%** | — | — | — | — | — | — | — | — | — | **98.12%** |
| **Stage 2** | 93.75% | **88.75%** | — | — | — | — | — | — | — | — | **91.25%** |
| **Stage 3** | 93.75% | 87.50% | **95.62%** | — | — | — | — | — | — | — | **92.29%** |
| **Stage 4** | 94.38% | 82.50% | 90.00% | **88.12%** | — | — | — | — | — | — | **88.75%** |
| **Stage 5** | 93.12% | 82.50% | 88.12% | 85.62% | **75.00%** | — | — | — | — | — | **84.88%** |
| **Stage 6** | 91.88% | 82.50% | 88.75% | 83.12% | 77.50% | **73.75%** | — | — | — | — | **82.92%** |
| **Stage 7** | 88.75% | 80.00% | 80.62% | 83.75% | 75.00% | 72.50% | **80.62%** | — | — | — | **80.18%** |
| **Stage 8** | 80.62% | 80.00% | 81.88% | 81.25% | 70.00% | 71.88% | 80.00% | **84.38%** | — | — | **78.75%** |
| **Stage 9** | 80.62% | 78.75% | 80.00% | 80.00% | 69.38% | 70.00% | 78.75% | 83.12% | **78.12%** | — | **77.64%** |
| **Stage 10** | 79.38% | 77.50% | 75.00% | 76.88% | 68.12% | 68.75% | 80.00% | 84.38% | 77.50% | **73.12%** | **76.06%** ($A_{10}$) |
| **Độ quên $f_j^{10}$** | 18.75% | 11.25% | 20.62% | 11.25% | 9.38% | 5.00% | 0.62% | 0.00% | 0.62% | 0.00% | **$F_{\text{past}} = 8.61\%$ / $F_{\text{code}} = 7.75\%$** |

## Phụ lục B: Phân định Minh bạch Nhóm W&B Chính thức (`caoban170106...`) vs Legacy CPU (`747b828`)

- **Bộ kết quả chính thức 12 lượt chạy (`results/dtd/`, Colab Tesla T4, commit `ca42e672`):** ghi cấu hình `logging.entity = caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm`, `project = hcmus-foal-reproduction`, `group = table4_ablation_dtd` (lưu ý `metadata.json` không lưu riêng `wandb_run_id`).
- **Bộ chạy CPU feature cache lịch sử (commit `747b828`, W&B `phongphanthe70-afsb`):** thu được `A0–A3 = 85.26 / 82.55 / 78.64 / 72.97%` (Seed 42) và `85.13 ± 0.11%` (chỉ thay đổi `projection`/`stream` trên cache Seed 42 trong `results/dtd_multiseed_1600_summary.json`). Được lưu giữ nguyên trạng làm chứng từ phát triển cục bộ; các số liệu chính thức toàn dự án sử dụng `results/dtd/`.

## Phụ lục C: Tổng hợp Kiểm toán Pháp y (Plasticity–Forgetting, Tách Biến Đa Seed & Provenance)

### C.1. Nghịch lý Plasticity–Forgetting khi bỏ Smooth Projection (Seed 42: `dtd_a0_full` vs `dtd_a2_without_sp`)

Khi loại bỏ SP (`A0 -> A2`), $\bar{A}$ giảm từ `85.08%` xuống `77.81%` (`-7.28 pp`) nhưng $F_{\text{past}}$ lại giảm từ `8.61%` xuống `6.32%` (`-2.29 pp`). Phân rã từng tác vụ quá khứ (Tasks 1–9) chứng minh đỉnh học ban đầu của A2 tụt mạnh hơn điểm cuối (**"Underlearning Masks Forgetting"**):

| Tác vụ $j$ | Đỉnh A0 ($\text{Peak}_j^{\text{A0}}$) | Cuối A0 ($\text{Final}_j^{\text{A0}}$) | Quên A0 ($f_j^{\text{A0}}$) | Đỉnh A2 ($\text{Peak}_j^{\text{A2}}$) | Cuối A2 ($\text{Final}_j^{\text{A2}}$) | Quên A2 ($f_j^{\text{A2}}$) | $\Delta\text{Peak}_j$ | $\Delta\text{Final}_j$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Task 1 | 98.12% | 79.38% | 18.75% | 97.50% | 75.00% | 22.50% | +0.62 pp | +4.38 pp |
| Task 2 | 88.75% | 77.50% | 11.25% | 86.25% | 69.38% | 16.88% | +2.50 pp | +8.12 pp |
| Task 3 | 95.62% | 75.00% | 20.62% | 88.75% | 77.50% | 11.25% | +6.88 pp | -2.50 pp |
| Task 4 | 88.12% | 76.88% | 11.25% | 76.25% | 75.00% | 1.25% | +11.88 pp | +1.88 pp |
| Task 5 | 77.50% | 68.12% | 9.38% | 64.38% | 61.25% | 3.12% | +13.12 pp | +6.88 pp |
| Task 6 | 73.75% | 68.75% | 5.00% | 68.75% | 68.12% | 0.62% | +5.00 pp | +0.62 pp |
| Task 7 | 80.62% | 80.00% | 0.62% | 76.88% | 76.88% | 0.00% | +3.75 pp | +3.12 pp |
| Task 8 | 84.38% | 84.38% | 0.00% | 80.00% | 80.00% | 0.00% | +4.38 pp | +4.38 pp |
| Task 9 | 78.12% | 77.50% | 0.62% | 75.62% | 74.38% | 1.25% | +2.50 pp | +3.12 pp |
| **Mean (1–9)** | **85.00%** | **76.39%** | **8.61%** | **79.38%** | **73.06%** | **6.32%** | **+5.62 pp** | **+3.33 pp** |

- **Đẳng thức sai biệt độ quên:** $\Delta F_{\text{past}} = \Delta\overline{\text{Peak}} - \Delta\overline{\text{Final}} = (85.0000\% - 79.3750\%) - (76.3889\% - 73.0556\%) = +5.6250 - 3.3333 = \mathbf{+2.2917\text{ pp}}$.
- **Tỷ lệ duy trì kiến thức tương đối ($\text{RKR} = \overline{\text{Final}}/\overline{\text{Peak}}$ trên Seed 42):** `A0 = 89.87%` (`76.39 / 85.00`), `A1 = 90.52%` (`74.24 / 82.01`), `A2 = 92.04%` (`73.06 / 79.38`), `A3 = 92.28%` (`71.39 / 77.36`). Trên trung bình 3 seeds, đúng thứ bậc nghịch đảo lặp lại đơn điệu (`A0..A3`: $\bar{A}$ `85.85 > 82.93 > 79.32 > 75.24%`; $F_{\text{past}}$ `8.22 > 7.69 > 5.88 > 4.54%`).

### C.2. Giải quyết Khớp nối Đa Seed (`class_order`, `dataset`, `projection`, `stream`)
- Ở giai đoạn trước `ca42e672`, file cấu hình YAML đặt cứng các seed con bằng `42` nên `--seed 123/2026` chỉ đổi nhãn `base` mà không đổi ma trận chiếu hay thứ tự lớp.
- Sau khi chuyển các hạt giống con trong `configs/dtd/dtd_a*.yaml` về `null` và truyền tách biến đồng bộ qua `resolve_seed()` + `make_dtd_task_specs(seed=class_order_seed)` (đã bổ sung tương ứng vào `scripts/train_dtd.py`), cả 12 lượt chạy tại `results/dtd/` xác nhận `base = dataset = class_order = projection = stream` trong từng seed `{42, 123, 2026}`, hash `task_split_fingerprint.sha256` trùng khít trong cùng một seed và khác biệt hoàn toàn giữa 3 seeds (`42b13803...`, `45da4afe...`, `29e80afd...`).

### C.3. Kiểm toán `git.dirty = true` ở `ca42e672` & Phân tách Thời gian Tesla T4
- **Nguyên nhân `git.dirty = true` trên 11/12 runs:** Lượt chạy đầu tiên (`dtd_a0_full`) chạy trên worktree sạch (`dirty = false`). Khi kết quả `dtd_a0_full` được ghi vào thư mục `results/dtd/` chưa track, hàm `get_git_info()` ở thời điểm `ca42e672` chạy `git status --porcelain` (không có `--untracked-files=no`), khiến 11 lượt chạy kế tiếp ghi `git.dirty = true` (đã vá cho các lần chạy sau tại `9f955ca`). Diff tại thời điểm chạy không được lưu riêng; các file thô trong `results/dtd/` được giữ nguyên không sửa đổi.
- **Vi mô thời gian thực thi (Tesla T4):** 6 lượt chạy lạnh `A0/A1` mất `38.99–42.17s` trích xuất đặc trưng ViT (`40.40–43.60s` tổng); 6 lượt chạy dùng cache `A2/A3` ghi `feature_extraction_seconds` kế thừa từ cache nguồn nhưng wall-clock thực tế chỉ `0.97–1.27s`; pha giải tích `rls_and_eval_seconds` trên toàn bộ 12 lượt chạy chỉ mất `0.79–1.00s`.
"""

p = Path("docs/DTD_COMPLETE_WORK_REPORT.md")
p.write_text(report_content.strip() + "\n", encoding="utf-8")
print(f"Generated comprehensive master report at {p.resolve()}")
