# F-OAL — Experimental protocol & milestones

## Đối chiếu paper gốc

- Section 3.1: trung bình các block output của ViT, Gaussian random projection cố định, sigmoid.
- Section 3.2–3.3, Equations (10)–(13): block RLS, tăng số cột của classifier khi có lớp mới.
- Dataset CIFAR-100, 10 stages × 10 classes, minibatch size 10 theo mô tả thực nghiệm.
- Số chiều projection mặc định 1000, gamma=1.0 (kiểm tra lại cấu hình tác giả khi chạy chính thức).

## Bước A — Validity (CPU)

1. `pytest -q`: RLS bằng nghiệm ridge đóng theo sai số máy; không replay train indices; projection cố định.
2. Kiểm tra class order, nhãn remap, train/test disjoint, old class columns không biến mất.
3. Khi thay minibatch partition, nghiệm RLS gần như không đổi nếu thứ tự mẫu không đổi.

## Bước B — Pilot (KHÔNG được coi là reproduction paper)

`python -m scripts.train --max-tasks 2 --limit-train-per-task 100`

Mục tiêu là kiểm chứng kỹ thuật: tải dataset, ViT chạy được, RLS không singular, metric có giá trị hợp lệ, không bị OOM. Pilot có thể dùng các lớp của hai task đầu, không thay cho báo cáo kết quả đầy đủ.

## Bước C — Main comparison

1. Chạy 10 task × 10 class trên CIFAR-100 với class order và preprocessing ghi rõ.
2. Lưu accuracy per task, mean stage accuracy, last accuracy; metric forgetting theo Equations (17)–(18); kiểm chứng bằng ví dụ thủ công.
3. Chạy ít nhất một baseline phù hợp. **Chưa có baseline trong starter hiện tại** — phải tự bổ sung và kiểm chứng protocol tương thích, hoặc báo cáo rõ scope.
4. Chạy các seed theo protocol của paper, ghi median/mean và độ phân tán. Không tuyên bố khớp số liệu gốc nếu backbone/weight/data khác.

## Bước D — Ablation

- Full F-OAL
- No feature fusion
- No smooth projection
- Neither

`python -m scripts.ablation --wandb`

**Ghi chú:** Ablation này là bố cục thực nghiệm phù hợp Eq. 1 nhưng cần đối chiếu Table 4 và cách bỏ thành phần trong repo tác giả trước khi dùng là strict reproduction.

## Bước E — Additional analysis (nhóm tự thiết kế)

Gợi ý: sensitivity theo batch size (1/10/25/50); với RLS lý tưởng, batch grouping không ảnh hưởng nghiệm trên cùng stream nhưng độ ổn định số và thời gian có thể thay đổi. Đo accuracy, wall time, peak memory, tính nhất quán nghiệm; đây là hypothesis, chưa có kết quả.

## W&B convention

- Project: `hcmus-foal-reproduction` (thay theo nhóm).
- Nhóm run với tags `pilot`, `reproduction`, `ablation`, `additional-analysis`.
- Mỗi run: config, git commit, seed, model checkpoint info, class order, accuracy matrix.
- Không commit secrets/token vào repo; wandb login trên máy hoặc môi trường Colab riêng.

## Rủi ro chưa giải quyết

- Chưa xác minh weights ViT và cách chuẩn hóa ảnh khớp repo tác giả; checkpoint chỉ lưu R/W và projection, ViT được tải lại bằng tên mô hình.
- `metrics.forgetting` dùng maximum historical task accuracy trừ final task accuracy, tương ứng Equations (17)–(18). Cần kiểm tra scaling 0–1 so với % và quy ước lấy trung bình của các seeds.
- Replay baselines, multi-seed driver, precision/hardware benchmarks, profiling GPU memory chưa triển khai.
- Chưa có end-to-end evaluation trên CIFAR-100 thực tế tại thời điểm đóng gói.
