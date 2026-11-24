# Bảng Đối chiếu Tuân thủ Toàn diện Đề bài HCMUS (Lab 1 + Milestone 3 + Lab 2 + Poster A2)

> **Điều hướng nhanh:** [← Trang chủ `README.md`](../README.md) | [Ma trận Phạm vi `hcmus_core_scope_matrix.md`](hcmus_core_scope_matrix.md) | [Hướng dẫn Thực thi `README_IMPLEMENTATION.md`](../README_IMPLEMENTATION.md)

Tài liệu này đối chiếu trực tiếp nhánh **`main`** của kho lưu trữ [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL) với toàn bộ yêu cầu chính thức của môn **Nhập môn Học máy (HCMUS)**:
1. **Project Statement Overview:** [`https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/`](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)
2. **Lab 1 — Experimental Reproduction:** [`https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/`](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/)
3. **Milestone 3 & Lab 2 — Critical Analysis, Empirical Demonstration & Oral Poster Workshop:** [`https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/`](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)

---

## 1. Ma trận Tuân thủ Toàn diện (Verified on `main`)

| Hạng mục Đánh giá | Quy định gốc (HCMUS Lab Book) | Trạng thái | Minh chứng Trực tiếp trong Repository & W&B |
| :--- | :--- | :---: | :--- |
| **1. Kho GitHub Công khai & Lịch sử Commit Rõ ràng** | Project Overview — Logistics | `PASS` | [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL), toàn bộ nhánh thực nghiệm đã tích hợp sạch vào `main`, GitHub Actions CI xanh (`ci.yml`). |
| **2. Lựa chọn Phạm vi (Core Scope) + Mở rộng Đa Tập Dữ Liệu** | Reproduction — Choosing the scope | `PASS` | [`README.md`](../README.md), [`docs/hcmus_core_scope_matrix.md`](hcmus_core_scope_matrix.md): hoàn tất Core Scope trên **CIFAR-100** và mở rộng đầy đủ sang **DTD** & **FGVC-Aircraft**. |
| **3. Tự Hiện thực Thuật toán Lõi từ Bài báo (Clean-Room)** | Reproduction — Rules of implementation | `PASS` | [`src/foal/`](../src/foal/) (`backbone.py`, `feature_fusion.py`, `projection.py`, `rls.py`, `model.py`); mã nguồn tác giả chỉ dùng chạy đối chứng độc lập. |
| **4. Minh bạch Thư viện Phụ trợ Tái sử dụng** | Reproduction — Rules of implementation | `PASS` | [`README.md`](../README.md) (Mục 1.3): tách bạch rõ `torch`, `torchvision`, `timm`, `datasets`, `wandb`, `PyYAML`, `numpy`, `pytest` với thuật toán tự viết. |
| **5. Python >= 3.9, Docstrings & Chuẩn Kỹ thuật Phần mềm** | Reproduction — Rules of implementation | `PASS` | [`pyproject.toml`](../pyproject.toml) (`>= 3.10`), [`requirements.txt`](../requirements.txt), đầy đủ docstrings và bộ kiểm thử tự động trong [`tests/`](../tests/). |
| **6. Cố định Random Seeds & Khả năng Lặp lại Đa Seed** | Reproduction — Rules of implementation | `PASS` | Kiểm soát đồng bộ 4 hạt giống (`base`, `class_order`, `projection`, `stream`) qua 3 seeds `42, 123, 2026` trên cả CIFAR-100, DTD, Aircraft và Lab 2. |
| **7. Nạp Dữ liệu Tự động bằng Code (Programmatic Loading)** | Reproduction — Running & tracking | `PASS` | [`src/foal/datasets.py`](../src/foal/datasets.py) & [`src/foal/dataset_adapters/`](../src/foal/dataset_adapters/): tự động tải `uoft-cs/cifar100`, `tanganke/dtd`, `Multimodal-Fatima/FGVC_Aircraft_*` từ Hugging Face. |
| **8. Một W&B Project Công khai Duy nhất + 4 Reports + Bảng Ánh xạ** | Reproduction & Lab 2 — Tracking | `PASS` | Project [`hcmus-foal-reproduction`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction) với **4 W&B Reports** (CIFAR-100, DTD, Aircraft, Lab 2) và toàn bộ runs hiển thị đầy đủ biểu đồ. |
| **9. Đối chiếu 3 Nguồn (Paper vs. Official Code vs. Ours) & Giải mã Sai lệch** | Reproduction — What results must contain | `PASS` | [`docs/original_vs_ours_protocol.md`](original_vs_ours_protocol.md), [`docs/CIFAR100_FINAL_REPRODUCTION_REPORT.md`](CIFAR100_FINAL_REPRODUCTION_REPORT.md): chuỗi chẩn đoán `C0–C3` giải thích triệt để khoảng cách `-11.65 pp` đến khớp bitwise (`Max abs diff = 0.0`). |
| **10. Kiểm toán Pháp y Nguồn gốc Dữ liệu (Provenance Audit)** | Scientific Integrity | `PASS` | [`docs/WANDB_PROVENANCE_AUDIT.md`](WANDB_PROVENANCE_AUDIT.md), [`docs/WANDB_PROVENANCE_REMEDIATION_REPORT.md`](WANDB_PROVENANCE_REMEDIATION_REPORT.md): phân loại trung thực `ONLINE_TRAINING`, `POST_HOC_IMPORT` và `ANALYSIS_ONLY` kèm SHA256. |
| **11. Milestone 3 — Phân tích Phản biện Lý thuyết (Critical Analysis)** | Critical Analysis (`>=3` Strengths, `>=2` Limitations + Remedies) | `PASS` | [`docs/CRITICAL_ANALYSIS_REPORT.md`](CRITICAL_ANALYSIS_REPORT.md): xác lập `S1–S3`, kiểm toán 5 lỗi nội tại của bài báo (Eq. 1, Eq. 14, Aircraft), chứng minh `L1–L3` gắn liền với 3 giải pháp `FD1–FD3`. |
| **12. Lab 2 — Thực nghiệm Chứng minh Điểm yếu & Giải pháp 0-Replay** | Empirical Demonstration (Lab 2) | `PASS` | [`src/foal/critical_analysis.py`](../src/foal/critical_analysis.py), [`docs/LAB2_EMPIRICAL_DEMONSTRATION_REPORT.md`](LAB2_EMPIRICAL_DEMONSTRATION_REPORT.md): 54 runs `E1` Long-Tail (`+42.67 pp` Tail Recall) + 98 runs `E3` Scaling & Double-Descent (`FD2` đạt `88.19%`), 6 runs chuẩn gắn nhãn `lab2-empirical-demo` trên W&B. |
| **13. Poster Khổ Dọc `Portrait A2` (`.pptx`, 1 Slide) & Mô tả Điểm yếu Phụ ở `README`** | Poster Workshop Deliverable | `PASS` | [`poster/F-OAL_Critical_Analysis_Poster_A2.pptx`](../poster/F-OAL_Critical_Analysis_Poster_A2.pptx) (`1` slide, `420.0 × 594.0 mm`), điểm yếu `L3` ghi nhận đầy đủ tại `README.md` Mục 5.4, kèm kịch bản bảo vệ [`docs/POSTER_WORKSHOP_DEFENSE_GUIDE.md`](POSTER_WORKSHOP_DEFENSE_GUIDE.md). |

---

## 2. Phân biệt Minh bạch 3 Chế độ Thực thi trong Dự án

1. **Chạy hoàn toàn từ đầu (`From Scratch`):**
   - `python -m scripts.reproduce_cifar100 --mode full --config configs/c3_d_bilinear_raw.yaml --seed 42` (hoặc `scripts.train`).
   - Tự tải dataset từ Hugging Face, trích xuất đặc trưng ViT-B/16 từ đầu, cập nhật Woodbury RLS và xuất ma trận đánh giá 10 tasks.
2. **Chạy từ Bộ nhớ đệm Đặc trưng Đã Xác thực SHA256 (`From Verified Feature Cache`):**
   - Sử dụng trong các phép quét đa cấu hình (`Table 4 Multi-Seed`, `Lab 2 E1 Long-Tail`, `Lab 2 E3 Scaling`) khi encoder đóng băng hoàn toàn không đổi.
   - Chỉ nạp cache khi khớp tuyệt đối `indices_sha256`, `backbone_name`, `pretrained_tag`, `feature_fusion`, `dataset_name` và `preprocessing`.
3. **Tái tạo Bảng biểu, Đồ họa & Poster từ Kết quả Đã Lưu (`Post-hoc Analysis & Poster Build`):**
   - Các script `scripts/analyze_*`, `scripts/generate_*`, `scripts/build_poster_assets.py` và `scripts/build_poster_a2_pptx.py` đọc trực tiếp từ các tệp JSON/CSV đã kiểm định trong `results/` để xuất hình ảnh và file `.pptx`.
