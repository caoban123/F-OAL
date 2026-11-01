# Cấu trúc thư mục dự án F-OAL

Đây là cấu trúc đã thống nhất ở hội thoại, được lưu để nhóm không mất định hướng khi chia việc. Có thêm `pyproject.toml`, `README_IMPLEMENTATION.md` để hỗ trợ chạy từ thư mục gốc và tránh ghi đè `README.md` hiện có.

```text
F-OAL/
├── README.md                         # Giữ nguyên README phân tích paper đang có trên GitHub
├── README_IMPLEMENTATION.md          # Hướng dẫn chạy starter
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── cifar100.yaml
├── src/
│   └── foal/
│       ├── __init__.py
│       ├── datasets.py
│       ├── backbone.py
│       ├── feature_fusion.py
│       ├── projection.py
│       ├── rls.py
│       ├── model.py
│       └── metrics.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── ablation.py
├── tests/
│   ├── test_dataset.py
│   ├── test_rls.py
│   ├── test_projection.py
│   └── test_metrics.py
├── notebooks/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── docs/
    ├── repository_structure.md
    ├── project_context.md
    └── experiments.md
```

## Quy ước quan trọng

- `src/foal/` chỉ chứa thuật toán và xử lý dữ liệu; `scripts/` chứa entry points.
- `tests/` kiểm tra từng thành phần độc lập, đặc biệt RLS với nghiệm ridge đóng.
- `configs/` giữ các thiết lập thí nghiệm. Mọi đổi khác so với paper phải ghi rõ.
- `results/` giữ kết quả thực nghiệm cục bộ nhưng không commit checkpoint/file dữ liệu dung lượng lớn.
- `docs/` lưu quyết định nghiên cứu, quy định project và kế hoạch thực nghiệm.
- W&B ghi lại cấu hình, metrics, seed, links, thực nghiệm phải có thể truy lại.

## Liêm chính học thuật

Mã nguồn này được xây dựng từ các công thức và mô tả trong paper F-OAL, không copy implementation của tác giả. Repo tác giả chỉ dùng để đối chiếu khi cần. Không tuyên bố reproduced paper cho đến khi có run thực tế và so sánh đúng protocol.
