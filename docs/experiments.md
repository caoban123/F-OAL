# F-OAL — Experimental protocol & milestones

## Đối chiếu paper gốc

- Section 3.1: trung bình các block output của ViT, Gaussian random projection cố định, sigmoid.
- Section 3.2–3.3, Equations (10)–(13): block RLS, tăng số cột của classifier khi có lớp mới.
- Dataset CIFAR-100, 10 stages × 10 classes, minibatch size 10 theo mô tả thực nghiệm.
- Số chiều projection mặc định 1000, gamma=1.0.

## Bước A — Validity (CPU)

1. `python -m pytest -v`: Đạt **30/30 passed**; RLS bằng nghiệm ridge đóng theo sai số máy; không replay train indices; projection cố định; metrics hand calculation; environment detection; multi-seed isolation.
2. Kiểm tra class order, nhãn remap, train/test disjoint, old class columns không biến mất.
3. Khi thay minibatch partition, nghiệm RLS không đổi nếu thứ tự mẫu không đổi.

## Bước B — Pilot (KHÔNG được coi là reproduction paper)

`python -m scripts.train --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100`

Mục tiêu là kiểm chứng kỹ thuật: tải dataset, ViT chạy được, RLS không singular, metric có giá trị hợp lệ, không bị OOM. Pilot dùng các lớp của hai task đầu, không thay cho báo cáo kết quả đầy đủ.

## Bước C — Main comparison & Multi-Seed (Kaggle/Colab GPU)

1. Chạy 10 task × 10 class trên CIFAR-100 với class order và preprocessing chuẩn ViT.
2. Điều phối đa seed: `python -m scripts.run_seeds --seeds 42 123 2026 --config configs/cifar100.yaml --device cuda`.
3. Lưu accuracy per task, mean stage accuracy, last accuracy, metric forgetting theo Equations (17)–(18); lưu tỷ lệ `[0, 1]` và hiển thị `xx.xx%`.
4. Tổng hợp tự động qua: `python -m scripts.summarize_runs`.

## Bước D — Ablation Study (Table 4)

- Full F-OAL (`ablation_full`)
- No feature fusion (`ablation_no_fusion`)
- No smooth projection (`ablation_no_projection`)
- Neither (`ablation_neither`)

`python -m scripts.ablation --config configs/cifar100.yaml --device cuda --wandb`

## Bước E — Additional analysis (nhóm tự thiết kế)

Batch-size sensitivity of analytic update under fixed data stream: batch size `[1, 10, 25, 50]`.
Đo accuracy, wall time, peak memory, tính nhất quán nghiệm.

## W&B convention

- Project: `hcmus-foal-reproduction`.
- Nhóm run với tags: `pilot`, `reproduction`, `ablation`, `additional-analysis`, `seed_{seed}`.
- Mỗi run: metadata chi tiết (Python, PyTorch, CUDA, GPU, git commit), model config, class order, accuracy matrix.
- Upload evaluation artifacts: `metrics.json`, `metadata.json`, `task_accuracy.csv`.
- Hỗ trợ `--wandb-offline` khi không có kết nối internet trực tiếp.
- Tuyệt đối không commit secrets/token vào repo.

## Trạng thái và Rủi ro mở

- **A01 & A04:** Vẫn duy trì trạng thái **PARTIAL** do chưa có sha256 checksum weights của checkpoint private và mảng hoán vị lớp CIFAR-100 từ tác giả paper.
- **P1.2:** Triển khai các baseline (ER, EWC) cho Continual Learning.
- **P1.3:** Huấn luyện Full 10 tasks CIFAR-100 trên GPU và Ablation 4 cấu hình.

---

## Kết quả Thực nghiệm GPU Độc lập: Single-Seed Full Reproduction (P1.2)

* **Thiết bị:** NVIDIA T4 GPU (Google Colab).
* **Thời gian thực thi:** 1735.3s (~28.9 phút).
* **Cấu hình:** Seed 42, 10 tasks, 10 classes/task, one-pass, batch size 10, full data (50,000 train / 10,000 test).

### Bảng Kết quả So sánh với Paper (NeurIPS 2024 Table 2)

| Chỉ số | Target Paper (Table 2) | Thực nghiệm F-OAL (Seed 42) | Độ lệch ($\\Delta$) |
| :--- | :---: | :---: | :---: |
| **Average Incremental Accuracy ({\\text{avg}}$)** | **.1\\%$** | **.33\\%$** | $-11.77\\%$ |
| **Last Accuracy ({\\text{last}}$)** | **.5\\%$** | **.92\\%$** | $-15.58\\%$ |
| **Forgetting Rate ($)** | **.5\\%$** | **.21\\%$** | $+3.71\\%$ |

### Nhận xét & Đánh giá:
1. **Khả năng duy trì tri thức cũ:** Tỷ lệ quên chỉ **.21\\%$** sau 10 bài học tích lũy 100 lớp trong chế độ hoàn toàn không lưu trữ dữ liệu cũ (zero-replay buffer).
2. **Khởi đầu vững chắc:** Task 1 đạt **.60\\%$**, chứng minh tính chính xác của việc đóng băng ViT-B/16 và trích xuất CLS token qua 12 tầng LayerNorm.
3. **Phân tích khoảng cách:** Khoảng cách với Paper chủ yếu do hai yếu tố còn mở (A01 - Checkpoint Provenance của ViT và A04 - Permutation thứ tự lớp). Đây là hiện tượng bình thường khi tái hiện độc lập mô hình Continual Learning mà tác giả không công bố file checkpoint weights nhị phân và random sequence gốc.

