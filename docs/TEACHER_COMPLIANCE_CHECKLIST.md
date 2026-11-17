# Teacher Requirements Compliance Checklist (CIFAR-100 Track)

Đối chiếu trực tiếp nhánh `experiment/cifar100-multiseed-ablation` của repository [`https://github.com/caoban123/F-OAL`](https://github.com/caoban123/F-OAL) với hai tài liệu chính thức của giảng viên HCMUS:

1. **Project Overview:** [`https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/`](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)
2. **Experimental Reproduction:** [`https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/`](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/)

---

## 1. Compliance Matrix

| Requirement | Official Source Section | Status | Evidence in Repository | Remaining Notes / Limitations |
| :--- | :--- | :---: | :--- | :--- |
| **1. Public GitHub Repository & Commit History** | [Project Overview — Logistics](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/#logistics) | `PASS` | `https://github.com/caoban123/F-OAL`, commit history từ `main` (`747b828`) đến `experiment/cifar100-multiseed-ablation` (`86ed263`) | Nhánh hiện tại chưa merge vào `main` (chờ review trước khi commit/merge) |
| **2. Core Scope Selection & Reasoning** | [Experimental Reproduction — Choosing the scope](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#choosing-the-scope) | `PASS` | `README.md` (Mục 14.1), `docs/hcmus_core_scope_matrix.md` | Tập trung Headline Comparison + Table 4 Ablation (3 seeds) trên CIFAR-100, kèm chẩn đoán C0–C3 |
| **3. Reimplement Core Method from Paper Only** | [Experimental Reproduction — The rules of implementation](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#the-rules-of-implementation) | `PASS` | `src/foal/` (`backbone.py`, `feature_fusion.py`, `projection.py`, `rls.py`, `model.py`); mã tác giả tách riêng ở `ref/official_foal/` (git-ignored) | Không sao chép từ `liuyuchen-cz/F-OAL` |
| **4. Reuse of Auxiliary Libraries Documented & Cited** | [Experimental Reproduction — The rules of implementation](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#the-rules-of-implementation) | `PASS` | `README.md` (Mục 14.3 — `torch`, `torchvision`, `timm`, `datasets`, `wandb`, `PyYAML`, `numpy`, `pytest`) | Phân tách rõ phần tái sử dụng với phần thuật toán tự triển khai |
| **5. Python >= 3.9 & Standard Scientific Stack** | [Experimental Reproduction — The rules of implementation](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#the-rules-of-implementation) | `PASS` | `pyproject.toml` (`requires-python >= 3.10`), `requirements.txt`, `requirements-lock.txt` | Kiểm thử trên Python `3.10.5` |
| **6. Docstrings on Important Functions** | [Experimental Reproduction — The rules of implementation](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#the-rules-of-implementation) | `PASS` | `src/foal/{model,backbone,projection,rls,datasets,environment}.py`, `scripts/reproduce_cifar100.py` | Đã bổ sung docstrings cho các class/hàm lõi CIFAR-100 |
| **7. Fixed Random Seeds & Repeatable Runs** | [Experimental Reproduction — The rules of implementation](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#the-rules-of-implementation) | `PASS` | `scripts/train.py`, `scripts/reproduce_cifar100.py`, `results/cifar100_multiseed/seed_manifest.json` | Ghi nhận `base_seed`, `class_order_seed`, `projection_seed`, `stream_seed` |
| **8. README Setup & End-to-End Execution Guide** | [Experimental Reproduction — The rules of implementation](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#the-rules-of-implementation) | `PASS` | `README.md` (Quick Start + Mục 14.2), `README_IMPLEMENTATION.md`, `scripts/reproduce_cifar100.py` | Hỗ trợ `--mode smoke` (không cần W&B key) và `--mode full` |
| **9. Programmatic Dataset Loading (HF / Kaggle / torchvision)** | [Experimental Reproduction — Running and tracking](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#running-and-tracking-the-experiments) | `PASS` | `src/foal/datasets.py` (`load_cifar100` dùng Hugging Face `uoft-cs/cifar100`, fallback `torchvision`) | Không cần tải hay giải nén thủ công |
| **10. Public W&B Project & README Mapping Table** | [Experimental Reproduction — Running and tracking](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#running-and-tracking-the-experiments) | `PASS` | `README.md` (Mục 14.4 & Mục 15), W&B `hcmus-foal-reproduction`, `docs/WANDB_PROVENANCE_AUDIT.md`, `docs/WANDB_PROVENANCE_REMEDIATION_REPORT.md` | Phân biệt rõ `ONLINE_TRAINING`, `POST_HOC_IMPORT` và `ANALYSIS_ONLY` |
| **11. Paper vs. Team Comparison & Deviation Analysis** | [Experimental Reproduction — What the results must contain](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#what-the-results-must-contain) | `PASS` | `docs/CIFAR100_FINAL_REPRODUCTION_REPORT.md`, `docs/CIFAR100_MULTISEED_ABLATION_REPORT.md`, `docs/CIFAR100_PREPROCESSING_AUDIT.md` | Có đủ số liệu paper, nhóm, relative deviation và phân tích C1–C3 |
| **12. Original-Code Comparison & Bug Localization** | [Experimental Reproduction — What the results must contain](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#what-the-results-must-contain) | `PARTIAL` | `docs/original_vs_ours_protocol.md`, `docs/CIFAR100_PREPROCESSING_AUDIT.md`, W&B `3eca8232` | Đã chạy code gốc (`seed=0`) và đối chiếu tensor (`Max abs diff = 0.0`); còn chênh seed (`0` vs `42`) |
| **13. Slide Deck Deliverable** | [Project Overview — Grading](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/#grading) & [Reproduction — Where it lands](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/#where-it-lands) | `NOT_VERIFIED` | Chưa có file slide deck (`.pptx`/`.pdf`) lưu trong repo này | Nhóm nộp slide qua Moodle / Google Slides ngoài repo |

---

## 2. Phân biệt 3 Chế độ Thực thi trong Dự án

1. **Chạy hoàn toàn từ đầu (`From Scratch`):**
   - `python -m scripts.reproduce_cifar100 --mode full --config configs/c3_d_bilinear_raw.yaml --seed 42` (hoặc `scripts.train`).
   - Tự tải dataset, trích xuất đặc trưng ViT từ đầu, chạy RLS và đánh giá 10 tasks.
2. **Chạy lại từ Feature Cache hợp lệ (`From Verified Feature Cache`):**
   - Dùng `--feature-cache-dir` (trong `scripts/run_cifar100_multiseed_ablation.py` / `scripts/train.py`).
   - Chỉ tái sử dụng tensor đặc trưng khi khớp toàn bộ `indices_sha256`, `backbone_name`, `pretrained_tag`, `feature_fusion`, `dataset_name`, và `preprocessing`.
3. **Tái tạo báo cáo từ kết quả đã lưu (`Post-hoc Analysis / Visualization`):**
   - `scripts/analyze_cifar100_multiseed.py`, `scripts/generate_headline_final_analysis.py`, `scripts/generate_table4_paper_vs_reproduction.py`.
   - Chỉ đọc `summary.json`/`metrics.json`/`task_matrix.csv` đã lưu để tạo bảng CSV và biểu đồ PNG; **không** phải lượt huấn luyện mới.
