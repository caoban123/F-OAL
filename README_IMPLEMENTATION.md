# F-OAL Technical Implementation & Execution Guide

> **Điều hướng nhanh:** [← Trang chủ `README.md`](README.md) | [Cấu trúc Thư mục `docs/repository_structure.md`](docs/repository_structure.md) | [Checklist Giảng viên `docs/TEACHER_COMPLIANCE_CHECKLIST.md`](docs/TEACHER_COMPLIANCE_CHECKLIST.md)

Tài liệu kỹ thuật hướng dẫn chi tiết cách cài đặt môi trường, thực thi các thí nghiệm tái lập (CIFAR-100, DTD, FGVC-Aircraft), chạy lại bộ kiểm định phản biện lý thuyết & giải pháp giải tích (Lab 2), và xuất lại Poster khoa học khổ dọc `Portrait A2`.

---

## 1. Yêu cầu Hệ thống & Cài đặt Môi trường

* **Python:** `3.10+` (đã xác minh trên Python `3.10.5` của GitHub Actions CI và Python `3.10–3.13` trên Google Colab / Kaggle).
* **Phần cứng:**
  * **CPU:** Chạy đầy đủ bộ unit tests, smoke tests 2 tasks, chẩn đoán C1 không cần huấn luyện, tái tạo bảng/biểu đồ và xuất Poster A2 `.pptx`.
  * **CUDA GPU (T4 / P100 / A100):** Khuyến nghị khi trích xuất đặc trưng ViT-B/16 từ đầu trên toàn bộ 60,000 ảnh CIFAR-100, DTD hoặc FGVC-Aircraft.
* **Tự động nạp dữ liệu & trọng số (Programmatic Loading):**
  * CIFAR-100: Hugging Face [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100) (dự phòng `torchvision.datasets.CIFAR100`).
  * DTD: Hugging Face [`tanganke/dtd`](https://huggingface.co/datasets/tanganke/dtd) (dự phòng `torchvision.datasets.DTD`).
  * FGVC-Aircraft: Hugging Face [`Multimodal-Fatima/FGVC_Aircraft_train`](https://huggingface.co/datasets/Multimodal-Fatima/FGVC_Aircraft_train) & [`_test`](https://huggingface.co/datasets/Multimodal-Fatima/FGVC_Aircraft_test).
  * Pre-trained ViT-B/16: tải tự động qua `timm` (`vit_base_patch16_224.augreg2_in21k_ft_in1k` cho cấu hình chuẩn hóa gốc và `vit_base_patch16_224.augreg_in1k` cho đối chứng C2).

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate

# 2. Cài đặt thư viện và gói nội bộ
pip install -r requirements.txt
pip install -e .

# 3. Chạy bộ kiểm thử đơn vị CPU-safe (tương đương GitHub Actions CI)
python -m pytest -q \
  --ignore=tests/test_checkpoint_roundtrip.py \
  --ignore=tests/test_rls_real_features.py \
  --ignore=tests/test_vit_transforms.py
```

*(Phiên bản khóa phụ thuộc chi tiết được lưu tại [`requirements-lock.txt`](requirements-lock.txt)).*

---

## 2. Năm Chế độ Thực thi Chính

### 2.1. Chế độ 1 — Smoke Test Nhanh trên CPU (Không cần GPU, Không cần W&B API Key)
Kiểm tra trọn vẹn luồng nạp dữ liệu $\to$ Frozen ViT + Feature Fusion $\to$ Smooth Projection $\to$ cập nhật Woodbury RLS $\to$ xuất `metrics.json` & `task_accuracy.csv` trong ~30–60 giây (lưu tách biệt tại `results/smoke_cifar100_seed42/`, tuyệt đối không ghi đè kết quả chính thức):

```bash
python -m scripts.reproduce_cifar100 --mode smoke
```

### 2.2. Chế độ 2 — Tái lập Đầy đủ Giai đoạn I trên CIFAR-100 (Core Scope)

```bash
# 1. Full F-OAL chuẩn hóa giao thức gốc (C3-D: Bilinear + Raw [0,1] + IN-21K ft IN-1K + Uniform P)
python -m scripts.reproduce_cifar100 --mode full --config configs/c3_d_bilinear_raw.yaml --seed 42

# 2. Table 4 Multi-Seed Factorial Ablation (4 cấu hình A0–A3 × 3 seeds = 12 runs)
python -m scripts.run_cifar100_multiseed_ablation --seeds 42 123 2026

# 3. Ba phương pháp Baseline đối chứng (Experience Replay B=5000, Official LwF, Adapted LwF)
python -m scripts.train_baseline --config configs/headline/er_cifar100.yaml --device cuda
python -m scripts.train_baseline --config configs/headline/lwf_cifar100_official_protocol.yaml --device cuda
python -m scripts.train_baseline --config configs/headline/lwf_cifar100.yaml --device cuda

# 4. Chuỗi chẩn đoán cô lập nguyên nhân C1 (Projection), C2 (Backbone), C3 (Preprocessing)
python -m scripts.run_c1_diagnostics --output-dir results/c1_diagnostics --device cpu
python -m scripts.run_c2_diagnostics --output-dir results/c2_analysis --device cpu
python -m scripts.run_c3_diagnostics --output-dir results/c3_analysis --device cpu
```

### 2.3. Chế độ 3 — Mở rộng Đa Miền: DTD & FGVC-Aircraft

```bash
# DTD — Full F-OAL (A0) & Baselines
python -m scripts.train --config configs/dtd/dtd_a0_full.yaml --seed 42 --device cuda
python -m scripts.train_baseline --config configs/dtd/er_dtd.yaml --device cuda

# FGVC-Aircraft — Table 4 Ablation 3 seeds (42, 123, 2026) & C1 Diagnostics
python -m scripts.run_aircraft_ablation --seeds 42 123 2026 --device cuda
python -m scripts.run_c1_aircraft_diagnostics --device cpu
```

*(Các Jupyter Notebooks chạy thực tế trên Google Colab / Kaggle GPU được lưu đầy đủ trong thư mục [`notebooks/`](notebooks/)).*

### 2.4. Chế độ 4 — Thực nghiệm Phản biện Lý thuyết & Giải pháp Giải tích 0-Replay (Lab 2)

Toàn bộ các thuật toán mới của Giai đoạn II nằm độc lập tại [`src/foal/critical_analysis.py`](src/foal/critical_analysis.py) (kiểm thử bởi [`tests/test_critical_analysis.py`](tests/test_critical_analysis.py)):

```bash
# 1. Thực nghiệm E1: Long-Tail OCIL (rho in {1, 5, 10, 25, 50, 100}, 3 seeds = 54 runs)
#    So sánh Vanilla F-OAL vs. Norm Calibration (FD1b) vs. BalancedOnlineRidgeRLS (FD1a+b)
python -m scripts.run_lab2_longtail --dataset cifar100 --seeds 42 123 2026

# 2. Thực nghiệm E3: Quy luật chi phí bậc hai Theta(D^2), Đỉnh Double-Descent & FD2 (98 runs)
python -m scripts.run_lab2_scaling_sensitivity --dataset cifar100 --seeds 42 123 2026

# 3. Thực nghiệm E2: Độ bền vững dưới nhiễu cảm biến (Gaussian Noise / Blur / JPEG / Contrast)
python -m scripts.run_lab2_corruptions --dataset dtd --seed 42

# 4. Vẽ lại toàn bộ biểu đồ phân tích Lab 2
python -m scripts.generate_lab2_figures
```

### 2.5. Chế độ 5 — Tái xuất Đồ họa & Đóng gói Poster Khoa học `Portrait A2` (`.pptx`)

```bash
# 1. Sinh sơ đồ kiến trúc, bảng công thức LaTeX và biểu đồ đa panel chuẩn mù màu (Okabe-Ito)
python -m scripts.build_poster_assets

# 2. Đóng gói thành đúng 1 slide khổ dọc Portrait A2 (420.0 mm x 594.0 mm)
python -m scripts.build_poster_a2_pptx
```

---

## 3. Ánh xạ Kiến trúc Mã nguồn (`src/foal/`)

| Module | Vai trò Toán học & Kỹ thuật | Tương ứng trong Bài báo / Lab 2 |
| :--- | :--- | :--- |
| [`src/foal/backbone.py`](src/foal/backbone.py) | Nạp ViT-B/16 đóng băng ($\nabla_\theta = 0$), trích xuất CLS tokens qua 12 Transformer blocks. | Section 3.1 (Eq. 1–2) |
| [`src/foal/feature_fusion.py`](src/foal/feature_fusion.py) | Trung bình cộng 12 vector CLS + `LayerNorm` tạo đặc trưng $\mathbf{h} \in \mathbb{R}^{768}$. | Section 3.1 (Eq. 2) |
| [`src/foal/projection.py`](src/foal/projection.py) | Phép chiếu ngẫu nhiên cố định $\boldsymbol{\phi}(x) = \sigma(\mathbf{h}\mathbf{P}) \in (0,1)^D$, hỗ trợ `P0` / `P1` / `P2`. | Section 3.1 (Eq. 3) & Chẩn đoán `C1` / `L3` |
| [`src/foal/rls.py`](src/foal/rls.py) | `OnlineRidgeRLS`: cập nhật đệ quy Woodbury ma trận $\mathbf{R}_t \in \mathbb{R}^{D \times D}$ và $\mathbf{W}_t \in \mathbb{R}^{D \times C}$ ở `float64`. | Section 3.2 (Eq. 10–13) |
| [`src/foal/model.py`](src/foal/model.py) | Ghép nối toàn tuyến F-OAL, quản lý mở rộng số lớp động qua từng task. | Toàn bộ Pipeline F-OAL |
| [`src/foal/critical_analysis.py`](src/foal/critical_analysis.py) | `BalancedOnlineRidgeRLS` (`FD1`), tự hiệu chỉnh `compute_trace_scaled_gamma` (`FD2`), tạo luồng Long-Tail & nhiễu ảnh. | Milestone 3 & Lab 2 (`L1–L3`, `FD1–FD2`) |
| [`src/foal/metrics.py`](src/foal/metrics.py) | Tính ma trận tam giác dưới $R_{k,j}$, $\bar{A}$, $A_T$, $F_{\text{past}}$ (9-task) và $F_{\text{code}}$ (10-task). | Section 4.1 & Chẩn đoán `C0` |
| [`src/foal/datasets.py`](src/foal/datasets.py) & [`dataset_adapters/`](src/foal/dataset_adapters/) | Nạp tự động CIFAR-100, DTD, FGVC-Aircraft từ Hugging Face; chia luồng 1-pass không lặp mẫu. | Section 4.1 |
