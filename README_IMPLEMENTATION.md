# F-OAL Implementation & Experimentation Guide

Hướng dẫn triển khai và thực thi các thí nghiệm cho dự án tái hiện **F-OAL (NeurIPS 2024)**.

---

## 1. Yêu cầu & Cài đặt môi trường

* Python **3.10+** (đã kiểm thử và xác minh trên Python 3.10.5).
* Hỗ trợ cả môi trường **CPU** (kiểm thử nhanh, unit tests) và **CUDA GPU** (Colab, Kaggle, server).
* Tự động tải CIFAR-100 từ Hugging Face (`uoft-cs/cifar100`) và pre-trained ViT weights (`timm/vit_base_patch16_224.augreg_in1k`).

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / WSL:        source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
pip install -e .

# Chạy toàn bộ 33 unit tests xác minh
python -m pytest -v
```

Phiên bản kiểm thử cố định được lưu tại `requirements-lock.txt`.

---

## 2. Hướng dẫn thực thi thí nghiệm

### 2.0. Full Reproduction Notebook trên Kaggle / Google Colab (GPU)
Để chạy toàn bộ benchmark CIFAR-100 (10 tasks $\times$ 10 classes, one-pass stream, batch size 10) trên môi trường GPU (T4, P100, V100, A100), sử dụng notebook:
* [`notebooks/foal_cifar100_full_reproduction.ipynb`](notebooks/foal_cifar100_full_reproduction.ipynb)

Notebook này bao gồm đầy đủ các bước kiểm tra phần cứng GPU, chạy preflight tests, chạy Single-Seed (Seed 42), đánh giá checkpoint độc lập, xuất bảng tổng hợp và điều phối Multi-Seed.

### 2.1. Pilot Smoke Test (Kiểm thử chức năng nhanh)
Chạy trên 2 tasks với 100 mẫu train/task để kiểm tra luồng dữ liệu, ViT encoding và cập nhật RLS:
```bash
python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100 --run-name smoke_test_p0
```

### 2.2. Multi-Seed Experiment Orchestration (A12)
Chạy tuần tự nhiều random seeds và tự động tổng hợp $Mean \pm Std$ (hỗ trợ bỏ qua các run đã hoàn thành qua `--skip-completed`):
```bash
# Chạy 3 seeds [42, 123, 2026] và tổng hợp:
python -m scripts.run_seeds --seeds 42 123 2026 --config configs/cifar100.yaml --run-name-prefix foal_3seeds --skip-completed

# Pilot multi-seed (tiết kiệm thời gian trên CPU):
python -m scripts.run_seeds --seeds 42 123 --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100 --run-name-prefix pilot_multiseed
```

### 2.3. Tự động tổng hợp kết quả (A14)
Quét toàn bộ thư mục `results/`, đọc `metadata.json` + `metrics.json`, nhóm theo phương pháp và xuất báo cáo CSV:
```bash
python -m scripts.summarize_runs --results-dir results
```
Tập tin xuất:
* `results/summary_runs.csv`: Thông tin chi tiết từng run.
* `results/summary_aggregated.csv`: Tổng hợp $Mean \pm Std$ cho từng phương pháp / loại thí nghiệm.
* `results/summary_ablation.csv`: Tổng hợp chuyên biệt cho các biến thể Ablation.

### 2.4. Theo dõi thực nghiệm qua Weights & Biases (A13)
```bash
# Chạy có logging W&B trực tuyến:
python -m scripts.train --config configs/cifar100.yaml --wandb --run-name foal_full_seed42

# Chạy W&B ở chế độ offline (ghi log cục bộ, không cần mạng):
python -m scripts.train --config configs/cifar100.yaml --wandb --wandb-offline --run-name foal_offline_seed42
```
*Tất cả metadata cấu hình, accuracy matrix, `metrics.json` và `task_accuracy.csv` được tự động upload thành artifact.*

### 2.5. Chạy 4 cấu hình Ablation Study (A11)
```bash
python -m scripts.ablation --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100
```

### 2.6. Đánh giá Checkpoint độc lập
```bash
python -m scripts.evaluate --checkpoint results/smoke_test_post_fix/task_02.pt --device cpu --limit-per-task 100
```

---

## 3. Kiến trúc mã nguồn

* `src/foal/backbone.py`: Frozen ViT (`vit_base_patch16_224.augreg_in1k`) với Feature Fusion averaging 12 blocks + final LayerNorm.
* `src/foal/projection.py`: Smooth Projection (Gaussian random matrix + Sigmoid).
* `src/foal/rls.py`: Recursive Least Squares trực tiếp bằng Woodbury formula và `torch.linalg.solve`.
* `src/foal/model.py`: Mô hình F-OAL tích hợp, quản lý dynamic class expansion.
* `src/foal/datasets.py`: Tải CIFAR-100, chia 10 tasks, ánh xạ nhãn song ánh, zero-replay.
* `src/foal/metrics.py`: Định nghĩa chuẩn Average Incremental Accuracy, Last Accuracy, Forgetting Rate và Multi-seed aggregation.
* `src/foal/environment.py`: Tự động thu thập thông số môi trường, thư viện, GPU, OS, Git commit.
* `scripts/run_seeds.py`: Điều phối chạy nhiều seeds và tổng hợp kết quả.
* `scripts/summarize_runs.py`: Tự động quét và lập bảng tổng kết các thí nghiệm.

---

## 4. Lệnh sẵn sàng chạy trên Kaggle / Google Colab (GPU) cho Phase P1.2/P1.3

```bash
# 1. Clone repository
git clone https://github.com/caoban123/F-OAL.git
cd F-OAL

# 2. Cài đặt thư viện
pip install -r requirements.txt
pip install -e .

# 3. Chạy 33 unit tests
python -m pytest -v

# 4. Huấn luyện Full CIFAR-100 trên GPU
python -m scripts.train --config configs/cifar100.yaml --device cuda --wandb --run-name foal_full_cifar100_seed42
```

---

## 5. Tài liệu Kiểm toán & Báo cáo Thực nghiệm (Checkpoint 3 Compliance)

* **Báo cáo Kiểm toán Kỹ thuật & Kế hoạch Khắc phục Toàn diện (`A01`–`A22`):** [`docs/FOAL_AUDIT_AND_FIX_PLAN.md`](docs/FOAL_AUDIT_AND_FIX_PLAN.md)
* **Kế hoạch Thực nghiệm, Core Scope & Bảng Đối chiếu Kết quả (`Paper vs Original vs Ours`):** [`docs/experiments.md`](docs/experiments.md)
* **Đối chiếu Giao thức Code Gốc Tác giả (`liuyuchen-cz/F-OAL @ 73e4ac1`) vs. Code Nhóm (`src/foal @ aad7500`):** [`docs/original_vs_ours_protocol.md`](docs/original_vs_ours_protocol.md)
* **Notebook Chạy Code Gốc Tác giả trên GPU (`A19`):** [`notebooks/foal_official_cifar100_reproduction.ipynb`](notebooks/foal_official_cifar100_reproduction.ipynb)

---

## 6. Bảng Ánh xạ Bằng chứng Thực nghiệm (README Evidence Mapping)

| Experiment | Implementation | Dataset | Seed | $\bar{A}$ | $A_{10}$ | $F$ ($F_{\text{official}}$ / $F_{\text{ours}}$) | Git SHA | Local Artifact | W&B Run | Status |
|---|---|---|---:|---:|---:|---:|---|---|---|---|
| **F-OAL Paper** | Paper (Table 2 & 3) | CIFAR-100 | 3 runs | 91.10% | 86.50% | 5.50% / — | — | — | — | `PAPER_REPORTED` |
| **F-OAL Official** | Original (`liuyuchen-cz/F-OAL`) | CIFAR-100 | `0` (`n=1, std=N/A`) | **90.98%** | **85.84%** | **4.69%** / **5.17%** | `73e4ac1` | [`results/official_foal_cifar100_seed0/`](results/official_foal_cifar100_seed0/) | [Run `3eca8232`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232) | `VERIFIED` (`A19 PASS`) |
| **F-OAL Full** | Ours (`src/foal`) | CIFAR-100 | `42` (`n=1, std=N/A`) | **79.33%** | **70.92%** | **8.29%** / **9.21%** | `aad7500` | [`results/foal_full_cifar100_seed42/`](results/foal_full_cifar100_seed42/) | [Run `2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv) | `VERIFIED` (`P1.2 PASS`) |
| **F-OAL Controlled (`C1`–`C4`)** | Ours (`src/foal`) | CIFAR-100 | `42` / `0` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | [`docs/original_vs_ours_protocol.md`](docs/original_vs_ours_protocol.md) | `NOT YET SYNCED` | `IN PROGRESS` (`A20`) |
| **F-OAL Multi-Seed** | Ours (`src/foal`) | CIFAR-100 | `42, 123, 2026` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `NOT YET SYNCED` | `PENDING` (`A12`) |
| **Ablation Study (4 configs)** | Ours (`scripts/ablation.py`) | CIFAR-100 | `42` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `NOT YET SYNCED` | `PENDING` (`A11`) |

### Đường dẫn W&B Đã Đồng bộ (`hcmus-foal-reproduction`)
* **Our Run (`2cdeiptv`, Seed 42, Commit `aad7500`):** [https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv)
* **Official Run (`3eca8232`, Seed 0, Commit `73e4ac1`):** [https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232)
