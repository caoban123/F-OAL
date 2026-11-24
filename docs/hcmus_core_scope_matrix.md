# HCMUS Term Project Scope & Reproducibility Matrix

> **Điều hướng nhanh:** [← Trang chủ `README.md`](../README.md) | [Checklist Giảng viên `TEACHER_COMPLIANCE_CHECKLIST.md`](TEACHER_COMPLIANCE_CHECKLIST.md) | [Cấu trúc Thư mục `repository_structure.md`](repository_structure.md)

## 1. Executive Summary & Audit Mandate

Tài liệu này xác lập ma trận phạm vi và mức độ hoàn thành trên nhánh `main` cho toàn bộ đồ án nghiên cứu bài báo:
**"F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning" (NeurIPS 2024)**.

Dự án tuân thủ đầy đủ quy chuẩn tại **HCMUS Intro to ML Lab Book**:
- [Project Statement Overview](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)
- [Lab 1: Experimental Reproduction](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/)
- [Milestone 3 & Lab 2: Critical Analysis, Empirical Demonstration & Poster Workshop](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)

Mọi hạng mục trong dự án được phân loại minh bạch thành 3 nhóm:
- `REQUIRED_BY_HCMUS`: Yêu cầu bắt buộc theo quy định môn học.
- `CHOSEN_BY_TEAM`: Lựa chọn phương pháp luận và chuỗi thực nghiệm chẩn đoán do nhóm thiết kế để giải quyết triệt để yêu cầu Core Scope & Critical Analysis.
- `OPTIONAL_RESEARCH_EXTENSION`: Mở rộng đa tập dữ liệu (DTD, FGVC-Aircraft) và đa hạt giống vượt mức yêu cầu tối thiểu.

---

## 2. Direct Source Verification from HCMUS Rubric

Trích dẫn nguyên văn từ Lab Book chính thức của môn học:
> - **Core scope:** *"the two most important experiments at minimum, including the headline comparison of the paper, with the rest replaced by one additional analysis the team designs itself, such as a sensitivity sweep over the configuration the paper varies."*
> - **The rules of implementation:** *"Reimplement from the paper's description only... Python 3.9 or newer... Fix random seeds, document them, keep runs repeatable... Every important function carries a docstring... Every experiment lands in one public Weights & Biases project (W&B)... Datasets load programmatically from hosted sources."*
> - **Critical Analysis & Empirical Demonstration:** *"Identify at least 3 evidenced strengths, at least 2 theoretical limitations paired with concrete future directions... Empirically demonstrate the limitations and showcase 2 primary axes on a single-slide Portrait A2 `.pptx` poster while documenting remaining proven weaknesses in `README.md`."*

---

## 3. Formal Scope & Completion Matrix (Verified on `main`)

| Hạng mục / Thực nghiệm | Phân loại | Mô tả & Lý do Khoa học | Trạng thái & Vị trí trên `main` |
| :--- | :--- | :--- | :--- |
| **1. Clean-Room Core Re-Implementation** | `REQUIRED_BY_HCMUS` | Tự hiện thực hoàn toàn từ mô tả toán học của bài báo (`FrozenViT`, `fuse_cls_features`, `SmoothProjection`, `OnlineRidgeRLS`, `FOALModel`), không sao chép mã tác giả. | **Hoàn tất** ([`src/foal/`](../src/foal/)) |
| **2. Official Code Execution & Parity Audit** | `REQUIRED_BY_HCMUS` | Chạy nguyên bản mã nguồn `liuyuchen-cz/F-OAL @ 73e4ac1` trên CIFAR-100 (`3eca8232`), DTD (`c9yyo08r`) và FGVC-Aircraft (`9ko84fxj`); đối chiếu sai lệch từng thành phần. | **Hoàn tất** ([`docs/original_vs_ours_protocol.md`](original_vs_ours_protocol.md)) |
| **3. Public W&B Tracking & 4 Reports** | `REQUIRED_BY_HCMUS` | Toàn bộ runs và 4 trang W&B Report công khai (CIFAR-100, DTD, Aircraft, Lab 2) nằm chung trong project `hcmus-foal-reproduction`. | **Hoàn tất** ([`README.md#6`](../README.md)) |
| **4. Programmatic Dataset Loading** | `REQUIRED_BY_HCMUS` | Tự động tải `uoft-cs/cifar100`, `tanganke/dtd`, `Multimodal-Fatima/FGVC_Aircraft_train` / `_test` qua Hugging Face `datasets` (kèm fallback `torchvision` cho CIFAR-100/DTD). | **Hoàn tất** ([`src/foal/datasets.py`](../src/foal/datasets.py), [`src/foal/dataset_adapters/`](../src/foal/dataset_adapters/)) |
| **5. Main Exp 1: Headline Comparison (CIFAR-100)** | `REQUIRED_BY_HCMUS` *(Core Scope)* | Đối sánh F-OAL (`90.40%`, 0-replay) với Experience Replay `B=5000` (`86.67%`), Official-Protocol LwF (`46.29%`) và Adapted LwF (`29.63%`). | **Hoàn tất** ([`docs/CIFAR100_FINAL_REPRODUCTION_REPORT.md`](CIFAR100_FINAL_REPRODUCTION_REPORT.md)) |
| **6. Main Exp 2: Multi-Seed Table 4 Ablation (CIFAR-100)** | `CHOSEN_BY_TEAM` *(Core Scope)* | Thiết kế giai thừa $2 \times 2$ (`A0–A3`) qua **3 seeds độc lập (`42, 123, 2026`, 12 runs)** cô lập vai trò của Feature Fusion và Smooth Projection. | **Hoàn tất** ([`docs/CIFAR100_MULTISEED_ABLATION_REPORT.md`](CIFAR100_MULTISEED_ABLATION_REPORT.md)) |
| **7. Controlled Diagnostic Suite (`C0–C3`)** | `CHOSEN_BY_TEAM` *(Core Scope)* | Phân tích nguyên nhân gốc giải thích khoảng cách $-11.65\text{ pp}$ ban đầu: công thức độ quên (`C0`), bão hòa Sigmoid do phương sai $\mathbf{P}$ (`C1`), checkpoint `IN-21K` vs `IN-1K` (`C2`), và nội suy ảnh `[0,1]` (`C3`, khớp bitwise `Max abs diff = 0.0`). | **Hoàn tất** ([`docs/CIFAR100_PREPROCESSING_AUDIT.md`](CIFAR100_PREPROCESSING_AUDIT.md)) |
| **8. Cross-Dataset Extension: DTD & FGVC-Aircraft** | `OPTIONAL_RESEARCH_EXTENSION` | Mở rộng đầy đủ Table 4 (3 seeds × 4 cấu hình = 24 runs), chuỗi kiểm soát `C1–C3`, Official Reference và Baselines sang miền kết cấu bề mặt (**DTD**) và phân loại tinh (**FGVC-Aircraft**). | **Hoàn tất** ([`docs/dtd_ablation_report.md`](dtd_ablation_report.md), [`docs/aircraft_ablation_report_vi.md`](aircraft_ablation_report_vi.md)) |
| **9. Milestone 3: Critical Analysis** | `REQUIRED_BY_HCMUS` | Kiểm toán 5 giả định ngầm, đính chính lỗi đại số Eq. (1) & Eq. (14) của bài báo, xác lập 3 điểm mạnh (`S1–S3`), 3 giới hạn lý thuyết (`L1–L3`) và 3 hướng khắc phục (`FD1–FD3`). | **Hoàn tất** ([`docs/CRITICAL_ANALYSIS_REPORT.md`](CRITICAL_ANALYSIS_REPORT.md)) |
| **10. Lab 2: Empirical Demonstration & Remedies** | `REQUIRED_BY_HCMUS` | Triển khai `BalancedOnlineRidgeRLS` (`FD1`) và `compute_trace_scaled_gamma` (`FD2`); chạy 54 runs Long-Tail OCIL (`E1`) + 98 runs Scaling & Double-Descent (`E3`), đồng bộ 6 runs chuẩn lên W&B (`tag: lab2-empirical-demo`). | **Hoàn tất** ([`src/foal/critical_analysis.py`](../src/foal/critical_analysis.py), [`docs/LAB2_EMPIRICAL_DEMONSTRATION_REPORT.md`](LAB2_EMPIRICAL_DEMONSTRATION_REPORT.md)) |
| **11. Portrait A2 Scientific Poster & Defense Guide** | `REQUIRED_BY_HCMUS` | Thiết kế Poster chuẩn hội nghị 1 slide khổ dọc `Portrait A2 (420 mm × 594 mm)` định dạng `.pptx` kèm kịch bản thuyết trình 3.5 phút & 12 câu hỏi phản biện. | **Hoàn tất** ([`poster/F-OAL_Critical_Analysis_Poster_A2.pptx`](../poster/F-OAL_Critical_Analysis_Poster_A2.pptx), [`docs/POSTER_WORKSHOP_DEFENSE_GUIDE.md`](POSTER_WORKSHOP_DEFENSE_GUIDE.md)) |
