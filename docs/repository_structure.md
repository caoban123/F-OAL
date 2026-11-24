# Cấu trúc Thư mục Hệ thống Dự án F-OAL (NeurIPS 2024)

> **Điều hướng nhanh:** [← Trang chủ `README.md`](../README.md) | [Hướng dẫn Thực thi `README_IMPLEMENTATION.md`](../README_IMPLEMENTATION.md) | [Checklist Giảng viên](TEACHER_COMPLIANCE_CHECKLIST.md)

Tài liệu này mô tả cấu trúc thực tế trên nhánh `main` của kho mã nguồn [`caoban123/F-OAL`](https://github.com/caoban123/F-OAL), bao quát cả hai chặng: **Giai đoạn I — Tái lập Thực nghiệm Đa Tập Dữ Liệu (CIFAR-100, DTD, FGVC-Aircraft)** và **Giai đoạn II — Phân tích Phản biện Lý thuyết, Thực nghiệm Lab 2 & Poster Khổ Dọc A2**.

```text
F-OAL/
├── README.md                                     # Cổng thông tin tổng quan, lý thuyết, kết quả & bảng ánh xạ W&B
├── README_IMPLEMENTATION.md                      # Hướng dẫn kỹ thuật chi tiết: cài đặt, CLI & tái tạo biểu đồ
├── pyproject.toml                                # Khai báo gói Python (>= 3.10) và cấu hình pytest
├── requirements.txt / requirements-lock.txt      # Danh sách thư viện phụ thuộc chuẩn & bản khóa phiên bản
├── .github/workflows/ci.yml                      # GitHub Actions CI tự động kiểm tra định dạng & CPU unit tests
│
├── src/foal/                                     # [LÕI THUẬT TOÁN TỰ TRIỂN KHAI TỪ BÀI BÁO — KHÔNG SAO CHÉP CODE GỐC]
│   ├── backbone.py                               # Frozen ViT-B/16 (timm) + trích xuất CLS qua 12 Transformer blocks
│   ├── feature_fusion.py                         # Trung bình cộng 12-block CLS tokens + LayerNorm (d_in = 768)
│   ├── projection.py                             # Smooth Projection cố định + Sigmoid (P0 / P1 / P2 variance modes)
│   ├── rls.py                                    # OnlineRidgeRLS cập nhật đệ quy Woodbury chính xác ở float64
│   ├── model.py                                  # Pipeline FOALModel hoàn chỉnh, mở rộng cột phân loại động
│   ├── critical_analysis.py                      # [LAB 2] BalancedOnlineRidgeRLS (FD1), Trace-Scaled Ridge (FD2), Long-Tail & Corruptions
│   ├── datasets.py                               # Bộ nạp luồng OCIL 1-pass (0-replay), kiểm soát 4 hạt giống độc lập
│   ├── dataset_adapters/                         # Bộ nạp tự động từ Hugging Face cho DTD (tanganke/dtd) & FGVC-Aircraft
│   ├── baselines/                                # Các baseline đối chứng: Experience Replay (er.py), LwF (lwf.py, official_lwf.py)
│   ├── metrics.py                                # Chuẩn hóa A_bar, A_last, F_past (9-task) & F_code (10-task) + gộp đa seed
│   └── environment.py                            # Thu thập thông tin phần cứng, phiên bản thư viện & Git SHA
│
├── configs/                                      # Cấu hình thực nghiệm YAML minh bạch, tách theo từng tập dữ liệu
│   ├── c3_d_bilinear_raw.yaml                    # Cấu hình chuẩn hóa CIFAR-100 khớp giao thức gốc (A0 Full F-OAL)
│   ├── c1_*.yaml / c2_*.yaml / c3_*.yaml         # Cấu hình kiểm soát biến số C1, C2, C3 trên CIFAR-100
│   ├── headline/                                 # Cấu hình ER (B=5000), Official LwF và Adapted LwF
│   ├── dtd/                                      # Cấu hình Table 4 (A0–A3) & C1–C3 trên DTD
│   └── aircraft/                                 # Cấu hình Table 4 (A0–A3) & C1–C3 trên FGVC-Aircraft
│
├── scripts/                                      # Các điểm vào thực thi (CLI Entry Points), kiểm toán & vẽ hình
│   ├── reproduce_cifar100.py                     # Launcher một lệnh cho người chấm (--mode smoke / --mode full)
│   ├── train.py / train_baseline.py / evaluate.py# Huấn luyện F-OAL, huấn luyện Baselines & đánh giá checkpoint độc lập
│   ├── ablation.py / run_cifar100_multiseed_ablation.py / run_aircraft_ablation.py
│   ├── run_c1_*.py / run_c2_*.py / run_c3_*.py   # Chuỗi thực nghiệm chẩn đoán C1–C3 trên CIFAR-100, DTD, Aircraft
│   ├── run_lab2_longtail.py                      # [LAB 2 — E1] Stress-test Long-Tail OCIL & kiểm chứng FD1
│   ├── run_lab2_scaling_sensitivity.py           # [LAB 2 — E3] Quét quy luật bậc hai Θ(D²), Double-Descent & kiểm chứng FD2
│   ├── run_lab2_corruptions.py                   # [LAB 2 — E2] Đánh giá độ bền vững dưới nhiễu cảm biến (Noise/Blur/JPEG/Contrast)
│   ├── sync_lab2_to_wandb.py / harmonize_wandb_workspace_and_runs.py
│   └── build_poster_assets.py / build_poster_a2_pptx.py  # Xuất đồ họa cao phân giải & đóng gói Poster Portrait A2 (.pptx)
│
├── notebooks/                                    # Bộ Jupyter Notebooks chạy thực tế trên Google Colab / Kaggle GPU
│   ├── foal_cifar100_full_reproduction.ipynb / foal_cifar100_multiseed_ablation.ipynb / foal_cifar100_headline_final.ipynb
│   ├── foal_dtd_ablation_colab.ipynb / foal_dtd_headline_final.ipynb
│   ├── foal_official_cifar100_reproduction.ipynb / foal_official_dtd_reproduction.ipynb
│   └── foal_lab2_empirical_demonstration.ipynb   # [LAB 2] Notebook tái lập đầu-cuối toàn bộ thực nghiệm Phản biện & Giải pháp
│
├── poster/                                       # Học liệu Poster Khoa học Khổ dọc Portrait A2 (420 mm × 594 mm)
│   ├── F-OAL_Critical_Analysis_Poster_A2.pptx    # File trình chiếu .pptx 1 slide duy nhất nộp Moodle
│   ├── F-OAL_Critical_Analysis_Poster_A2_preview.png
│   └── assets/                                   # Sơ đồ kiến trúc, bảng công thức LaTeX & biểu đồ đa panel chuẩn mù màu
│
├── results/                                      # Kết quả JSON/CSV/PNG đã xác minh (không commit file trọng số .pt/.zip lớn)
│   ├── README.md                                 # Chỉ mục chi tiết cấu trúc kết quả thực nghiệm
│   ├── cifar100_multiseed/ · table4_analysis/ · headline_analysis/ · c1_analysis/ · c2_analysis/ · c3_analysis/
│   ├── dtd/ · dtd_table4_analysis/ · headline_dtd_analysis/ · aircraft_ablation/
│   └── lab2_empirical/                           # [LAB 2] Tổng hợp 54 runs E1, 98 runs E3, biểu đồ & manifest W&B
│
├── tests/                                        # Hệ thống kiểm thử tự động (CPU-safe trên CI + kiểm thử đối chiếu lịch sử)
│   ├── test_rls.py / test_projection.py / test_model.py / test_metrics.py
│   ├── test_critical_analysis.py                 # [LAB 2] Kiểm thử nghiệm đóng Weighted Woodbury, Long-Tail, Corruptions & GCV
│   └── test_*_ablation*.py / test_*_dtd*.py / test_backfill_wandb_provenance_backup.py
│
└── docs/                                         # Hệ thống báo cáo khoa học, kiểm toán nguồn gốc & kịch bản bảo vệ
    ├── TEACHER_COMPLIANCE_CHECKLIST.md           # Bảng đối chiếu toàn bộ yêu cầu môn học HCMUS
    ├── hcmus_core_scope_matrix.md                # Ma trận phân định phạm vi bắt buộc / tự thiết kế / mở rộng
    ├── CIFAR100_FINAL_REPRODUCTION_REPORT.md / CIFAR100_MULTISEED_ABLATION_REPORT.md
    ├── dtd_ablation_report.md / aircraft_ablation_report_vi.md
    ├── WANDB_PROVENANCE_AUDIT.md / WANDB_PROVENANCE_REMEDIATION_REPORT.md
    ├── CRITICAL_ANALYSIS_REPORT.md               # [MILESTONE 3] Phản biện lý thuyết (S1–S3, L1–L3, FD1–FD3)
    ├── LAB2_EMPIRICAL_DEMONSTRATION_REPORT.md    # [LAB 2] Báo cáo thực nghiệm 152 runs chứng minh điểm yếu & giải pháp
    └── POSTER_WORKSHOP_DEFENSE_GUIDE.md          # Kịch bản thuyết trình 3.5 phút & 12 câu hỏi phản biện sâu
```

## Quy ước Kiến trúc & Liêm chính Học thuật

1. **Tách biệt tuyệt đối phần tự hiện thực và mã tham chiếu:** Toàn bộ module trong `src/foal/` được nhóm tự viết từ công thức toán học của bài báo NeurIPS 2024; mã nguồn tác giả (`liuyuchen-cz/F-OAL`) chỉ dùng để chạy đối chứng độc lập và nằm ngoài phần lõi (`ref/` bị loại trừ trong `.gitignore`).
2. **Không chỉnh sửa kết quả lịch sử:** Module mở rộng của Lab 2 (`src/foal/critical_analysis.py`) kế thừa/đối chiếu độc lập mà không sửa đổi `src/foal/rls.py`, `src/foal/model.py`, `src/foal/projection.py` hay dữ liệu Lab 1.
3. **Kiểm thử an toàn trên môi trường sạch (Clean Checkout CI):** Mọi commit lên `main` đều được GitHub Actions kiểm tra định dạng (`git diff-tree --check`) và chạy bộ kiểm thử CPU-safe độc lập với dữ liệu GPU cục bộ.
