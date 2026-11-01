# F-OAL Starter — Hướng dẫn xây dựng

**Giữ nguyên `README.md` đã có trên GitHub nhóm.** Bộ starter này thêm code độc lập từ paper, chưa tự động commit/push và chưa tạo số liệu reproduction.

## Yêu cầu

- Python **3.10+** và thư viện PyTorch tương thích CUDA nếu chạy GPU.
- Google Colab/Kaggle có GPU được khuyến nghị cho ViT-B/16; CPU phù hợp để kiểm thử RLS.
- Internet để tải CIFAR-100 qua Hugging Face Datasets và weights của ViT qua timm (lần đầu).

## Cài đặt (từ thư mục gốc repo)

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux / WSL: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Nếu `torch`/`torchvision` chưa tương thích với CUDA của môi trường, cài chúng theo hướng dẫn PyTorch trước, rồi cài các dependency còn lại.

## Smoke test (pilot, KHÔNG dùng như kết quả paper)

```bash
python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100
```

Chạy full 10 tasks (không có sample cap):

```bash
python -m scripts.train --config configs/cifar100.yaml --wandb --run-name foal_full_seed42
```

Chạy 4 cấu hình ablation:

```bash
python -m scripts.ablation --config configs/cifar100.yaml --wandb
```

Đánh giá checkpoint tại stage 2:

```bash
python -m scripts.evaluate --checkpoint results/foal_seed42_ff_sp/task_02.pt --device cpu
```

## Những gì đã triển khai

- `datasets.py`: automatic CIFAR-100 retrieval, 10-task class-incremental split, deterministic label remap, no train replay.
- `backbone.py` + `feature_fusion.py`: frozen timm ViT; average block CLS tokens.
- `projection.py`: seeded Gaussian random weights, fixed and sigmoid activation.
- `rls.py`: block Woodbury update, class expansion across tasks; no backpropagation.
- `model.py`: end-to-end model; `metrics.py`: task summary theo Equations (16)–(18) (đầu ra tỷ lệ 0–1; paper báo cáo phần trăm).
- `train.py`: full/pilot training pipeline, per-task evaluation, checkpoint, CSV/JSON, optional W&B.
- `ablation.py`, `evaluate.py`: orchestration and checkpoint evaluation.
- `tests`: analytic correctness and data stream split tests.

## Phần chưa hoàn thành để báo cáo giữa kỳ

- Reproduce các baseline trong Table 2; multi-seed runs; kiểm tra class order/cifar transform/weight variant đúng tác giả.
- Kiểm tra đối chiếu đầy đủ metric theo Eq. (16)–(18), đổi tỷ lệ 0–1 ra %, đánh giá trung bình 3 seeds như paper.
- Chạy ablation chính thức, additional experiment, ghi W&B runs và đối chiếu số liệu.
- Chuẩn bị slides 20–40 trang và speaker notes. Không dùng kết quả pilot làm headline comparison.

## Quan trọng về README

Hãy thêm link từ README hiện tại sang `[Hướng dẫn triển khai](README_IMPLEMENTATION.md)` thay vì ghi đè nội dung phân tích paper ban đầu.

Thông tin thêm: `docs/repository_structure.md`, `docs/project_context.md`, `docs/experiments.md`.

Lưu ý: checkpoint mỗi task chỉ chứa **R, W và ma trận projection**, không lưu lặp hàng trăm MB ViT weights; `evaluate.py` sẽ tải lại weights pretrained theo tên model. Cần pin model revision/version chính xác để tái lập nghiêm ngặt.
