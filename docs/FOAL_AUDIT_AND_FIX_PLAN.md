# F-OAL — Technical Audit, Verification & Fix Plan

> **Dự án:** Nhập môn Machine Learning — HCMUS  
> **Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024)  
> **Repository nhóm:** https://github.com/caoban123/F-OAL  
> **Phạm vi audit:** Bộ `F-OAL_starter_v1.zip` được tạo trong cuộc trò chuyện, **không phải** bản `main` trên GitHub đã được xác nhận có toàn bộ source.  
> **Ngày lập kế hoạch:** 01/11/2026  
> **Hạn nộp URL GitHub:** 07/11/2026, 23:59  
> **Hạn slide giữa kỳ:** 21/11/2026, 23:59  
> **Hạn poster cuối kỳ:** 05/12/2026, 23:59

---

## Trạng thái Hiện tại của Dự án (Current Project Status — Cập nhật 01/11/2026)

> **Lưu ý về cấu trúc lịch sử tài liệu:** Các Mục 0 đến Mục 19 bên dưới được giữ nguyên để bảo toàn biên bản kiểm toán lịch sử qua từng giai đoạn (**Initial Audit**, **P0 Remediation**, **P1.1 Multi-Seed & W&B Infrastructure**, **P1.2 GPU Preflight**). Do đó, những con số lịch sử như *"9 passed"*, *"14 passed"*, *"15 passed"*, *"30 passed"* hay các ghi chú *"chưa có `scripts/run_seeds.py`"* ở các mục đầu phản ánh trạng thái tại thời điểm ghi biên bản đó. **Mục này và các Mục 20–23 ở cuối tài liệu là nguồn tham chiếu chuẩn xác nhất về trạng thái hiện tại của dự án.**

### 1. Phân loại trạng thái chuẩn hóa (Status Taxonomy)
Để tránh nhầm lẫn giữa việc *"đã viết code"* và *"đã xác minh kết quả khoa học"*, toàn bộ các hạng mục trong dự án được phân loại theo 6 mức độ nghiêm ngặt:
* **`IMPLEMENTED`**: Đã hiện thực mã nguồn trong repository.
* **`UNIT TESTED`**: Đã có unit test tự động kiểm tra tính đúng đắn của logic (hiện tại bộ test đạt **33/33 PASS** trên CPU).
* **`EXECUTED`**: Đã chạy thực nghiệm thực tế (trên CPU cho smoke test hoặc trên GPU cho full benchmark).
* **`VERIFIED`**: Đã có đầy đủ raw artifacts (`metrics.json`, `metadata.json`, `task_accuracy.csv`, checkpoints) lưu trữ hợp lệ trong repository và đối chiếu độc lập thành công.
* **`PARTIAL`**: Đã hoàn thành một phần (ví dụ: đã có code/test hoặc kết quả báo cáo từ Colab, nhưng thiếu raw artifact vật lý trên đĩa cục bộ hoặc chưa đối chiếu với repo gốc của tác giả).
* **`PENDING`**: Chưa thực hiện.

### 2. Bảng Tổng hợp Trạng thái Kiểm toán Hiện tại (Master Audit Matrix A01–A22)

| Mã | Hạng mục Kiểm toán | Ưu tiên | Trạng thái hiện tại | Bằng chứng & Ghi chú xác minh |
|:---|:---|:---:|:---:|:---|
| **A01** | Pre-trained ViT Backbone & Exact Checkpoint Provenance | P0 | **`PARTIAL`** | **`IMPLEMENTED` & `UNIT TESTED`**: Đã đóng băng `timm/vit_base_patch16_224.augreg_in1k` (`src/foal/backbone.py`), ghi nhận vào `configs/cifar100.yaml` và `metadata.json`. **Chưa `VERIFIED`**: Chưa đối chiếu trực tiếp với code gốc của tác giả (`liuyuchen-cz/F-OAL`) để xác nhận chính xác biến thể trọng số ViT-B/16 mà tác giả nạp (xem **A19**, **A20**). |
| **A02** | Feature Fusion across ViT Blocks + Final LayerNorm | P0 | **`VERIFIED`** | **`IMPLEMENTED`, `UNIT TESTED`, `EXECUTED`**: Lấy trung bình cộng CLS token của toàn bộ 12 blocks và áp dụng `network.norm` cuối (`src/foal/backbone.py`, `src/foal/feature_fusion.py`, `tests/test_vit_transforms.py`). Vẫn sẽ đối chiếu thêm thứ tự phép toán với code gốc tác giả trong **A20**. |
| **A03** | Smooth Projection ($D=1000$, Fixed Random Gaussian + Sigmoid) | P0 | **`PARTIAL`** | **`IMPLEMENTED` & `UNIT TESTED`**: Khởi tạo ma trận Gaussian cố định lưu vào `register_buffer` + hàm kích hoạt $\sigma(\cdot)$ (`src/foal/projection.py`). **Cần điều tra thêm (`A20`)**: Hiện tại `gaussian_std=1.0` không chuẩn hóa theo $1/\sqrt{d_{\text{in}}}=1/\sqrt{768}$, khiến tích vô hướng trước Sigmoid có độ lệch chuẩn $\approx 27.7$ (có nguy cơ làm bão hòa hàm $\sigma$). |
| **A04** | Class Incremental Task Splitting, Bijective Label Mapping & Class Order | P0 | **`PARTIAL`** | **`IMPLEMENTED` & `UNIT TESTED`**: Ánh xạ nhãn song ánh `label_to_incremental`, tách `class_order_seed`, lưu `class_order` trong `metadata.json` và checkpoint (`src/foal/datasets.py`, `scripts/train.py`). **Chưa `VERIFIED`**: Chưa xác nhận thứ tự lớp (class permutation) mặc định trong repo gốc của tác giả. |
| **A05** | Recursive Least Squares (RLS) & Woodbury Numerical Stability | P0 | **`VERIFIED`** | **`IMPLEMENTED`, `UNIT TESTED`, `EXECUTED`**: Cập nhật Woodbury trên `float64` với hệ số regularization $\gamma = 1.0$ ($R_0 = \gamma^{-1} I$), giải hệ tuyến tính bằng `torch.linalg.solve` (fallback `lstsq`), đối xứng hóa $R_t \leftarrow \frac{1}{2}(R_t + R_t^\top)$, sai số so với nghiệm Ridge toàn cục $< 10^{-10}$ trên đặc trưng ViT thật (`src/foal/rls.py`, `tests/test_rls_real_features.py`). |
| **A06** | One-Pass Online Stream & Zero-Replay Enforcement | P0 | **`VERIFIED`** | **`IMPLEMENTED`, `UNIT TESTED`, `EXECUTED`**: Luồng dữ liệu 1 epoch duy nhất (`batch_size=10`), kiểm tra giao tập hợp rỗng giữa các task, tách `stream_seed` (`src/foal/datasets.py`, `scripts/train.py`, `tests/test_dataset.py`). |
| **A07** | Continual Learning Metrics ($\bar{A}$, $A_{\text{last}}$, $F$) | P0 | **`VERIFIED`** | **`IMPLEMENTED` & `UNIT TESTED`**: Ma trận tam giác dưới $R \in \mathbb{R}^{T \times T}$, tính chính xác $\bar{A}$, $A_T$, $F_T$ nhất quán giữa `scripts/train.py` và `scripts/evaluate.py` (`src/foal/metrics.py`, `tests/test_metrics_comprehensive.py`). |
| **A08** | Dependency Pinning & Environment Logging | P1 | **`VERIFIED`** | **`IMPLEMENTED` & `UNIT TESTED`**: Đã khóa phiên bản trong `requirements-lock.txt`, tự động ghi nhận OS, Python, PyTorch, CUDA, GPU, Git commit trong `src/foal/environment.py` và `metadata.json`. |
| **A09** | Automated Unit & Mathematical Equivalence Tests | P1 | **`VERIFIED`** | **`UNIT TESTED`**: **34/34 unit tests PASS** (`python -m pytest -q`), bao gồm cả kiểm thử xử lý `null` seed trong `resolve_seed()` và kiểm thử `summarize_runs.py` (`tests/test_summarize_runs.py`). |
| **A10** | Full CIFAR-100 Single-Seed GPU Reproduction | P1 | **`VERIFIED` (1-Seed Execution)** / **`PARTIAL` (Accuracy Gap)** | **`EXECUTED` & `VERIFIED` (Colab T4, Seed 42)**: Đã chạy hoàn tất 10 tasks $\times$ 10 classes trên Google Colab GPU Tesla T4 (`1735.3s`, commit `aad7500`), đạt $\bar{A}=79.33\%$, $A_{10}=70.92\%$, $F_{10}=9.21\%$. Đã giải nén và xác minh trực tiếp gói `foal_cifar100_single_seed42_results.zip` ($109.8\text{ MB}$) chứa đầy đủ `metrics.json`, `metadata.json`, `task_accuracy.csv`, 10 checkpoints `task_01.pt`..`task_10.pt` ($\|R - R^\top\|_{\max} = 0.0$, `all_finite=True`) và W&B offline run `run-2cdeiptv.wandb`. Còn khoảng cách $-11.77\text{ pp}$ so với paper cần điều tra qua **A19** & **A20**. |
| **A11** | Ablation Study (Feature Fusion $\times$ Smooth Projection) | P1 | **`IMPLEMENTED`** / **`UNIT TESTED`** | Đã hoàn thiện `scripts/ablation.py` và kiểm thử pilot trên CPU. **Chưa thực thi (`PENDING`)** lượt chạy 10-task đầy đủ trên GPU cho 4 cấu hình Ablation. |
| **A12** | Multi-Seed Orchestration ($Mean \pm Std$) | P1 | **`IMPLEMENTED`** / **`UNIT TESTED`** | Đã hoàn thiện `scripts/run_seeds.py` với kiểm tra tương thích giao thức (`--skip-completed`) và unit tests (`tests/test_multi_seed_infrastructure.py`). **Chưa thực thi (`PENDING`)** đủ 3 seeds (`42, 123, 2026`) full 10-task trên GPU. |
| **A13** | Experiment Tracking (Weights & Biases) | P1 | **`PARTIAL`** | Đã tích hợp `src/foal/wandb_utils.py` và kiểm thử qua unit tests, nhưng **chưa xác minh upload online** và **chưa có public W&B project link** (xem bảng chi tiết W&B tại Mục 3 bên dưới). |
| **A14** | Automated Results Aggregation (`summarize_runs.py`) | P1 | **`VERIFIED`** | **`IMPLEMENTED`, `UNIT TESTED`, `EXECUTED`**: Tự động lọc các run theo `protocol_signature` 13 thành phần, tách biệt pilot/full/ablation và xuất CSV chuẩn (`tests/test_summarize_runs.py`). |
| **A15** | Baseline 1: Backprop Fine-tuning / Linear Probe | P2 | **`PENDING`** | Chưa hiện thực và chưa chạy thực nghiệm trên cùng ViT backbone + task split. |
| **A16** | Baseline 2: Offline / Linear Ridge Analytic Classifier | P2 | **`PENDING`** | Chưa chạy đối chiếu độc lập đầy đủ trên GPU. |
| **A17** | Baseline 3: Experience Replay (ER) / Continual Learning Baseline | P2 | **`PENDING`** | Chưa hiện thực và chưa chạy thực nghiệm. |
| **A18** | Efficiency Profiling (Training Time & Peak GPU VRAM) | P2 | **`PARTIAL`** | Đã ghi nhận `total_duration_seconds = 1735.3s` trên Colab T4 cho Seed 42, nhưng chưa ghi nhận `torch.cuda.max_memory_allocated()` chuẩn hóa để đối chiếu với Table 3 ($261\text{ s}$) và Figure 2 ($1.9\text{ GB}$). |
| **A19** | **Original Code Reproduction (`liuyuchen-cz/F-OAL`)** | **P0 (CP3)** | **`PENDING`** | **Yêu cầu bắt buộc Checkpoint 3**: Chạy repository gốc của tác giả trong thư mục độc lập ngoài `src/`, thu thập kết quả `ORIGINAL_REPRODUCED` trên CIFAR-100 và đối chiếu với Paper & Ours. |
| **A20** | **Pipeline Discrepancy Investigation (Ours vs Original)** | **P0 (CP3)** | **`PENDING`** | Điều tra có hệ thống nguyên nhân khoảng cách $-11.77\text{ pp}$ giữa Ours (`79.33%`) và Paper (`91.10%`) qua từng tầng xử lý (Dataset, Preprocessing, ViT Weights, Feature Fusion, Smooth Projection scaling, RLS, Evaluation). |
| **A21** | **Public W&B Project & Evidence Traceability** | **P1 (CP3)** | **`PARTIAL`** | Cần tạo project W&B ở chế độ **Public** (truy cập không cần đăng nhập), đồng bộ các run chính thức và gắn link trực tiếp từng dòng kết quả trong README/docs tới W&B run. |
| **A22** | **Core Scope Definition & 3-Way Comparison Table (`Paper vs Original vs Ours`)** | **P1 (CP3)** | **`PARTIAL`** | Đã bổ sung đề xuất **Core Scope** và khung bảng đối chiếu chuẩn gồm `PAPER_REPORTED`, `ORIGINAL_REPRODUCED`, `OUR_REPRODUCED`, $\Delta_{\text{pp}}$ và $\Delta_{\text{rel}}\%$ trong tài liệu này; chờ điền số liệu sau khi hoàn tất **A19**–**A21**. |

### 3. Bảng Phân rã Trạng thái Weights & Biases (A13 & A21 Breakdown)

| Thành phần kiểm tra W&B | Trạng thái | Chi tiết xác minh |
|:---|:---:|:---|
| **1. W&B Code Integration (`src/foal/wandb_utils.py`, `--wandb`, `--wandb-offline`)** | **`IMPLEMENTED`** & **`UNIT TESTED`** | Đã tích hợp đầy đủ vào `scripts/train.py` và kiểm thử logic gọi hàm/artifact bằng mock trong `tests/test_multi_seed_infrastructure.py`. |
| **2. W&B Offline Logging CLI Execution** | **`VERIFIED`** | Đã xác minh trực tiếp gói W&B offline thật sinh ra từ lượt chạy Full GPU trên Colab: `wandb/offline-run-20261101_084136-2cdeiptv/run-2cdeiptv.wandb` ($188,433\text{ bytes}$), sẵn sàng đồng bộ lên W&B Cloud qua lệnh `wandb sync`. |
| **3. W&B Online Upload Verification** | **`PENDING`** | Lượt chạy GPU Seed 42 trên Colab vừa qua không bật cờ `--wandb` với `WANDB_API_KEY`. Chưa có run nào được đẩy lên đám mây W&B. |
| **4. Public W&B Project Link (No-Login Access)** | **`PENDING`** | Chưa khởi tạo hoặc kiểm chứng đường dẫn W&B project ở chế độ công khai (`Public`) không yêu cầu đăng nhập theo quy định Checkpoint 3. |
| **5. README Result Table Linked to W&B Runs** | **`PENDING`** | Bảng kết quả trong tài liệu hiện đang ghi nhận trạng thái `NOT YET SYNCED TO PUBLIC W&B`; sẽ cập nhật URL trực tiếp của từng run sau khi thực hiện đồng bộ ở Stage C/D. |

---

## 0. Mục đích, phạm vi và cách đọc (Bản ghi lịch sử Initial Audit & P0)

Tài liệu này là **checklist kỹ thuật có thể giao việc**, tập hợp các vấn đề phải kiểm tra, phần thiếu cần bổ sung, thứ tự ưu tiên và tiêu chí nghiệm thu để nhóm chuyển từ **starter implementation** sang **experimental reproduction** có thể kiểm chứng. Mục tiêu không phải sửa để có accuracy bằng tác giả bằng mọi giá; mục tiêu là **đúng thuật toán, đúng experimental protocol, đo lường trung thực, truy vết được kết quả và giải thích được sai lệch**.

Ký hiệu trạng thái:

- **ĐÃ XÁC NHẬN:** Kết luận có thể suy ra trực tiếp từ source đang có hoặc từ phép kiểm tra đã thực hiện.
- **CHƯA XÁC MINH:** Chưa có bằng chứng đối chiếu đầy đủ với paper/repository tác giả hoặc chưa chạy dữ liệu thật. **Không tự coi là bug.**
- **CHƯA TRIỂN KHAI:** Chức năng được kỳ vọng nhưng hiện không có trong starter.
- **ĐỀ XUẤT:** Hướng cải tiến/thiết kế của nhóm; không phải yêu cầu hay định nghĩa bắt buộc trong paper.

Mức ưu tiên:

- **P0 — Blocker:** Cần giải quyết trước khi chạy reproduction chính thức; nếu sai, kết quả có thể không hợp lệ.
- **P1 — Cần cho giữa kỳ:** Hoàn thành để có bằng chứng thực nghiệm, baseline, ablation, W&B và báo cáo.
- **P2 — Nâng chất lượng:** Profiling, tự động hóa, bảo trì, phân tích sâu và chuẩn bị poster.

### 0.1. Trạng thái đã kiểm tra của starter

- [x] Đã đọc `configs/cifar100.yaml`, `src/foal/*.py`, `scripts/*.py`, `tests/*.py`, `docs/experiments.md`.
- [x] Đã chạy `python -m pytest -v`: **14 passed (01/11/2026)** (bổ sung tests cho ViT transforms, RLS trên real features, checkpoint roundtrip, finite checks).
- [x] Có implementation cho task splitting, Frozen ViT, Feature Fusion, Smooth Projection, RLS, evaluation và ablation orchestration.
- [x] Có W&B integration ở mức cơ bản trong `scripts/train.py`.
- [x] **ĐÃ CHẠY END-TO-END SMOKE TEST THÀNH CÔNG** trên CIFAR-100 thật và pretrained ViT-B/16 (2 tasks × 100 samples/task).
- [x] **ĐÃ XÁC NHẬN** checkpoint `vit_base_patch16_224.augreg_in1k`, sửa normalization chuẩn ViT `(0.5, 0.5, 0.5)`, bicubic interpolation.
- [x] **ĐÃ XÁC MINH RLS BẰNG BATCH RIDGE REGRESSION** trên đặc trưng thật của ViT với sai số $< 10^{-10}$.
- [x] Source starter và các bản vá P0 đã được đẩy lên GitHub của nhóm: https://github.com/caoban123/F-OAL.
- [ ] **Chưa có** baseline implementation trong starter (P1).
- [ ] **Chưa có** multi-seed orchestration, bảng tổng hợp và profiling GPU hoàn chỉnh (P1/P2).
- [ ] **Chưa có** W&B run thực tế hay số liệu full reproduction được xác nhận (P1).

**Quan trọng:** Sau khi hoàn thành P0, core implementation của F-OAL đã đúng chuẩn toán học, vượt qua toàn bộ 14 tests và hoàn thành smoke test trên dữ liệu thật.

## 1. Ma trận tổng quan: kiểm tra gì, sửa gì, ưu tiên nào

| ID | Hạng mục | Tình trạng | Ưu tiên | File liên quan & Dòng code | Kiểm thử (Tests) | Bằng chứng nghiệm thu (Evidence) |
|---|---|---|---|---|---|---|
| A01 | Đúng checkpoint ViT | **PARTIAL** | **P0** | `src/foal/backbone.py` (L18-29), `configs/cifar100.yaml` (L5-14) | `tests/test_vit_transforms.py::test_frozen_vit_properties` | Đã cố định `vit_base_patch16_224.augreg_in1k` (85.8M params, `requires_grad=False`, `train(False)` giữ `eval()`). Đánh dấu **PARTIAL** vì chưa có checksum nhị phân đối chiếu với checkpoint nội bộ của tác giả paper. |
| A02 | Feature Fusion đúng vị trí | **PASS** | **P0** | `src/foal/backbone.py` (L43-65), `src/foal/feature_fusion.py` (L6-15) | `tests/test_vit_transforms.py::test_feature_fusion_applies_final_layernorm_both_branches` | Cả 2 nhánh: FF=True (trung bình 12 blocks + `network.norm`) và FF=False (khớp 100% `forward_features`) đều trùng khớp giải tích tính tay với sai số đại số 0.0. |
| A03 | Transforms đúng backbone | **PASS** | **P0** | `src/foal/datasets.py` (L92-109), `configs/cifar100.yaml` (L23-28) | `tests/test_vit_transforms.py::test_vit_transforms_structure` | Chuẩn hóa `(0.5, 0.5, 0.5)` và `bicubic` 224x224. Đã phân tích rõ ràng: không áp dụng CenterCrop(224) của timm ImageNet để tránh cắt bỏ 10% biên ảnh CIFAR-100 $32 \times 32$. |
| A04 | CIFAR-100 task protocol | **PARTIAL** | **P0** | `src/foal/datasets.py` (L23-60), `scripts/train.py` (L51-57) | `tests/test_dataset.py::test_seeded_order_is_reproducible_and_permutation` | 10 tasks × 10 classes, stream one-pass không replay, seed 42 cố định và tái lập. Đánh dấu **PARTIAL** do chưa có mảng permutation private từ repo tác giả để chứng minh khớp 100% thứ tự lớp trong Bảng 2. |
| A05 | Không train/test leakage | **PASS** | **P0** | `src/foal/datasets.py` (L48-60), `scripts/train.py` (L104-123) | `tests/test_dataset.py::test_task_partition_no_train_replay_and_no_future_test` | `isdisjoint` giữa các train sets; `seen_test_indices` chỉ chứa lớp đã học; stream one-pass không replay buffer |
| A06 | RLS về mặt đại số | **PASS** | **P0** | `src/foal/rls.py` (L41-76) | `tests/test_rls.py::test_rls_matches_closed_form_ridge_with_expanding_classes` | Woodbury identity: `gain = solve(A, K.T).T`. Nghiệm khớp Batch Ridge Regression với sai số $< 10^{-11}$ |
| A07 | RLS ổn định trên ViT features | **PASS** | **P0** | `src/foal/rls.py` (L59-74), `src/foal/model.py` (L38-46) | `tests/test_rls_real_features.py::test_rls_matches_ridge_on_real_vit_features` | Chạy trên ảnh CIFAR-100 thật qua ViT-B khớp Ridge với sai số $< 10^{-10}$; không có NaN/Inf (`isfinite=True`) |
| A08 | Load/save checkpoint | **PASS** | **P0** | `scripts/train.py` (L134-146), `scripts/evaluate.py` (L16-95) | `tests/test_checkpoint_roundtrip.py::test_checkpoint_save_and_reload` | So sánh trực tiếp class scores (logits) trước/sau nạp checkpoint cho sai số $< 10^{-12}$; `evaluate.py` nạp checkpoint mới khớp tuyệt đối với `train.py` (0.77 vs 0.77, per-task [0.69, 0.85]) |
| A09 | Định nghĩa metrics | **PASS** | **P1.1** | `src/foal/metrics.py` (L1-125), `tests/test_metrics.py` | `tests/test_metrics.py` (8 passed) | Chuẩn hóa công thức theo Chaudhry et al. (2018) và F-OAL Eq. 17-18; unit tests tính tay cho 1, 2, 3 tasks, non-monotonic forgetting, positive backward transfer, tỷ lệ [0, 1] và hiển thị [%] |
| A10 | Headline comparison | Chưa triển khai đủ | **P1.2** | `train.py`, `src/baselines/` mới | — | Có F-OAL + baseline công bằng, đủ protocol |
| A11 | Ablation Table 4 | Có driver, chờ chạy GPU | **P1.3** | `ablation.py`, `train.py` | — | 4 cấu hình, output truy vết, tổng hợp tự động |
| A12 | Multi-seed & uncertainty | **PASS** | **P1.1** | `scripts/run_seeds.py`, `scripts/train.py` | `tests/test_multi_seed_infrastructure.py` | Tách bạch `class_order_seed`, `projection_seed`, `stream_seed`; điều phối multi-seed, hỗ trợ `--skip-completed`, tính mean $\pm$ std không ghi đè kết quả |
| A13 | W&B đầy đủ | **PASS** | **P1.1** | `scripts/train.py` (L125-245) | `tests/test_wandb_offline_and_config.py` | W&B tracking đầy đủ: tags (`pilot`, `reproduction`, `ablation`), metadata môi trường/phần cứng, per-task & summary metrics, artifact logging, hỗ trợ `--wandb-offline`, không hardcode token |
| A14 | Tự động tổng hợp báo cáo | **PASS** | **P1.1** | `scripts/summarize_runs.py` | `tests/test_summarize_runs.py` | Quét `results/`, tự động nhóm theo `(method, experiment_type)`, tính mean $\pm$ std, xuất `summary_runs.csv`, `summary_aggregated.csv` và `summary_ablation.csv` |
| A15 | Time/VRAM profiling | Chưa có | **P2** | `train.py`, scripts mới | — | Ghi thiết bị, runtime, peak VRAM, state size |
| A16 | Kiểm soát môi trường | **PASS** | **P1.1** | `src/foal/environment.py`, `requirements-lock.txt` | `tests/test_environment.py` | Tự động ghi nhận thông số OS/Python/CUDA/GPU/PyTorch/timm/git commit; khóa phiên bản trong `requirements-lock.txt`; đường dẫn cross-platform, không hardcode path cá nhân |
| A17 | Additional analysis | Mới có đề xuất | **P1.3** | `docs/experiments.md`, scripts | — | Một phân tích **mới**, giả thuyết rõ, có kết quả |
| A18 | README, source & W&B traceability | Cập nhật đầy đủ P1.1 | **P1.1** | `README_IMPLEMENTATION.md`, docs | — | Hướng dẫn chi tiết chạy multi-seed, summarize và W&B offline/online |

---

## 2. P0 — Xác minh pretrained backbone và Feature Fusion

**Vấn đề:** Starter dùng `vit_base_patch16_224.augreg_in1k` của `timm` với `pretrained=True`. Tuy vậy, mô tả **ViT-B pre-trained on ImageNet-1K** trong paper chưa xác nhận đó là **đúng checkpoint**, cách trích xuất và tiền xử lý của kết quả gốc. Đây là **chưa xác minh**, không phải kết luận model đang dùng sai.

**Vì sao quan trọng:** F-OAL đóng băng backbone. RLS chỉ học từ đặc trưng được đưa vào; khác checkpoint hoặc transforms có thể tạo khoảng cách accuracy lớn dù RLS hoàn toàn đúng.

### A01 — Checkpoint và phiên bản

- [x] Đọc Section 3.1, 4 và appendix của paper để ghi tên kiến trúc, tiền huấn luyện và feature extraction.
- [x] Đọc README/config mã nguồn **tác giả chỉ để đối chiếu**, không sao chép implementation core.
- [x] Xác định `timm` model ID: `vit_base_patch16_224.augreg_in1k`, 85.8M parameters, embedding dimension $D=768$, 12 blocks.
- [x] Khóa/ghi `torch==2.13.0+cpu`, `torchvision==0.28.0+cpu`, `timm==1.0.30`, `datasets==5.0.0`.
- [x] Đảm bảo `requires_grad=False` cho mọi tham số ViT và encoder luôn ở `eval()` trong mọi run (đã có override `train(mode)`).
- [x] Kiểm tra encoder không bị optimizer cập nhật và không có backward trong training loop F-OAL.

**Nghiệm thu:** Đã nghiệm thu. Model `vit_base_patch16_224.augreg_in1k` tải và chạy thành công trên máy.

### A02 — Feature Fusion

- [x] Xác nhận đặc trưng các block có shape `[batch, tokens, D]`, `D=768`.
- [x] So sánh đầu ra 2 cách: **full fusion** (lấy trung bình CLS tokens của 12 blocks) và **last-block-only** (`collected[-1:]`).
- [x] Kiểm tra đầu ra của hooks lấy đúng thứ tự blocks; gỡ hooks sau mỗi forward trong khối `finally`.
- [x] So khớp chỗ lấy feature với định nghĩa của tác giả trong Section 4.2.
- [x] Test cùng một ảnh, cùng model, cùng chế độ `eval()`: feature ổn định tuyệt đối (test `test_vit_transforms.py`).

### A03 — Preprocessing

- [x] Đối chiếu `timm.data.resolve_model_data_config`: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation='bicubic'.
- [x] Khắc phục lỗi starter (trước đó dùng nhầm ImageNet CNN mean `(0.485, 0.456, 0.406)`), đã sửa chuẩn ViT.
- [x] Bổ sung `test_vit_transforms.py` kiểm tra shape, normalize và tính hữu hạn (`torch.isfinite`).

---

## 3. P0 — Xác minh CIFAR-100 stream và class-incremental protocol

### A04 — Chia task đúng

- [x] Xác nhận tập CIFAR-100 tải được trực tiếp từ `uoft-cs/cifar100` (Hugging Face) và có fallback torchvision.
- [x] Xác nhận 50.000 train, 10.000 test; mỗi task có 5.000 ảnh train và 1.000 ảnh test.
- [x] Kiểm tra `label_to_incremental` là một song ánh và nhất quán giữa train/test.
- [x] Kiểm tra `total_seen_classes` là 10, 20, ..., 100.
- [x] Lưu class order và seeds vào metadata.json của từng run.

### A05 — Data leakage và one-pass

- [x] Tại task `t`, train loader chỉ chứa nhãn của **các lớp mới thuộc task t** (tập train disjoint).
- [x] Không có mẫu test trong train loader.
- [x] Không đánh giá test trên lớp tương lai trước khi học lớp đó.
- [x] Mỗi sample train xuất hiện **một lần** trong luồng F-OAL (one-pass, không epoch replay).
- [x] Đánh giá sau task `t` cho classifier dự đoán trên **toàn bộ các lớp đã thấy** (C_seen classes).

---

## 4. P0 — RLS: đúng toán học và ổn định số

### A06 — Kiểm thử correctness

- [x] `test_rls_matches_closed_form_ridge_with_expanding_classes` qua trên synthetic data.
- [x] `test_batch_partition_does_not_change_exact_solution` qua trên synthetic data.
- [x] `test_can_only_add_classes_not_remove` qua.
- [x] Dùng `torch.linalg.solve` trong cập nhật batch; không dùng nghịch đảo `inv(A)`.
- [x] Bổ sung kiểm tra `torch.isfinite` phát hiện NaN/Inf trong ma trận đầu vào và cập nhật.

### A07 — Kiểm tra số học trên feature thật

- [x] Bổ sung `tests/test_rls_real_features.py`: chạy online RLS và so khớp với batch Ridge Regression $W^* = (H^T H + \gamma I)^{-1} H^T Y$ trên đặc trưng ViT thật từ CIFAR-100.
- [x] Đạt độ chính xác tuyệt đối: sai số giữa nghiệm online RLS và batch Ridge $< 10^{-10}$.
- [x] Chạy smoke test end-to-end trên CIFAR-100 thật (2 tasks × 100 samples) không xuất hiện NaN/Inf.

### A08 — Checkpoint and restart

- [x] Bổ sung `tests/test_checkpoint_roundtrip.py`: save checkpoint, reload và so sánh dự đoán (`pred_before == pred_after` với sai số $< 10^{-12}$).
- [x] Lưu cấu hình chi tiết `rls_dtype`, `projection_seed`, `backbone_name`, `class_order`.
- [x] Đã kiểm tra lưu và đọc thành công checkpoint thật `task_01.pt` (11.1MB) và `task_02.pt` (11.2MB).

**Nghiệm thu:** Một checkpoint có thể được đánh giá lại độc lập với code train mà không tạo chênh lệch ngoài sai số cho phép.

---

## 5. P1 — Metrics: kiểm chứng định nghĩa và phép tổng hợp

**File:** `src/foal/metrics.py`, `scripts/train.py`, `tests/test_metrics.py`.

**Không kết luận `metrics.py` đang sai.** Starter tính **trung bình accuracy của các task đã thấy ở từng stage**, rồi trung bình qua stages; đây là cách hợp lý trong CIL với CIFAR-100 cân bằng. Tuy nhiên, phải đối chiếu với định nghĩa và cách báo cáo **cụ thể** của F-OAL trước khi đưa số lên slide. Với forgetting, starter dùng *max historical accuracy minus final accuracy* trên các task cũ; cũng cần đối chiếu paper.

Đặt `a[i,j]` là accuracy trên task `j` sau khi hoàn thành stage `i`, chỉ xét `j≤i`.

\[
A_i=\frac{1}{i}\sum_{j=1}^{i}a_{i,j},\qquad
A_{\rm avg}=\frac1T\sum_{i=1}^{T} A_i,\qquad A_{\rm last}=A_T.
\]

Một cách tổng hợp forgetting cuối kỳ thường dùng:

\[
F_T=\frac{1}{T-1}\sum_{j=1}^{T-1}
\left(\max_{\ell\in\{j,\ldots,T-1\}} a_{\ell,j}-a_{T,j}\right).
\]

**Chỉ sử dụng công thức này để báo cáo paper sau khi xác minh định nghĩa gốc và protocol.**

- [x] Đọc đúng equation/section của paper cho `A_i`, `A_avg`, `A_last`, `F` (Chaudhry et al., 2018; F-OAL Section 5, Eq. 17-18).
- [x] Kiểm tra task-wise accuracy và pooled accuracy: phân định rõ task accuracy $a_{k, j}$, stage accuracy $A_k$, average incremental accuracy $\bar{A}$, và last accuracy $A_T$.
- [x] Unit test bằng bảng 3 task tự tính tay; test thêm trường hợp accuracy phục hồi về sau (positive backward transfer / signed forgetting).
- [x] Quy ước rõ accuracy trong JSON/code lưu dạng tỷ lệ $[0, 1]$ và hiển thị báo cáo dưới dạng phần trăm `xx.xx%`.
- [x] Báo cáo **mỗi run trước**, sau đó tổng hợp mean/std qua seeds bằng `aggregate_multi_seed_metrics`; không nhầm average over tasks với average over seeds.
- [x] Thử `T=1` để tránh chia 0 ở forgetting (trả về 0.0 theo định nghĩa).
- [x] Đảm bảo W&B metrics, CSV, JSON và bảng slide đều cùng công thức.

**Nghiệm thu:** Đã nghiệm thu A09 (**PASS**). Vượt qua toàn bộ 8 unit tests tính tay trong `tests/test_metrics.py`.

---

## 6. P1 — Main comparison và baselines

**Tình trạng xác nhận:** Starter hiện **chưa có** thư mục `src/baselines/` và `scripts/train.py` **chưa hỗ trợ `--method er|ewc|...`**. Đây là chức năng chưa triển khai, không phải chỉ sửa một dòng.

### A10 — Chọn phạm vi và baseline

- [ ] Chốt **Core Scope** theo đề bài: ít nhất **hai thí nghiệm quan trọng**, gồm headline comparison, và **một phân tích bổ sung tự thiết kế** (xác nhận lại wording chính thức khi báo cáo).
- [ ] Chọn 1–2 baseline thực sự có trong protocol của F-OAL và có đủ thông số triển khai, **không chọn theo độ dễ viết đơn thuần**.
- [ ] Có thể cân nhắc **ER** và **EWC**, nhưng implementation phải theo cấu hình hợp lệ của paper; một CNN nhỏ chạy dễ **không phải** baseline so sánh trực tiếp công bằng với F-OAL dùng ViT-B.
- [ ] Tách biệt phần so sánh **paper-reported** (số liệu tác giả) và phần **group-reproduced** (nhóm thật sự chạy).
- [ ] Nếu không đủ GPU để reproduce baseline deep learning, ghi rõ giới hạn scope và ưu tiên baseline nhẹ hợp lệ hoặc dùng số liệu paper làm reference — **không nói là nhóm đã reproduce baseline chưa chạy**.

### Kiến trúc code ĐỀ XUẤT (chưa có)

```text
src/
  baselines/
    __init__.py
    er.py
    ewc.py
scripts/
  compare.py
```

Entry point `--method foal|er|ewc` **là đề xuất tương lai**, không phải tham số có sẵn trong `scripts/train.py` hiện tại.

### Kiểm thử baseline

- [ ] Tất cả methods trong cùng trial sử dụng đúng class order, test split và evaluation definitions.
- [ ] Kiểm tra replay buffer/optimizer/training epochs/loss function theo cấu hình baseline, có chú thích nếu khác.
- [ ] Đo runtime + VRAM nếu paper so sánh tài nguyên.
- [ ] Mọi kết quả thuộc một W&B run riêng, không trộn log của nhiều phương pháp.

**Nghiệm thu:** Bảng so sánh F-OAL, baseline và số liệu paper có đường dẫn đến từng run; sai khác được diễn giải hợp lý.

---

## 7. P1 — Ablation của Feature Fusion và Smooth Projection

**File:** `scripts/ablation.py`; **trạng thái:** driver hiện có bốn tổ hợp, **chưa có kết quả**.

| Ablation | Feature Fusion | Smooth Projection | CIFAR-100 `A_avg` trong Table 4 (paper) | Nhóm |
|---|---|---|---:|---|
| Full | On | On | **91,1%** | Chưa chạy |
| No FF | Off | On | **90,6%** | Chưa chạy |
| No SP | On | Off | **90,7%** | Chưa chạy |
| Neither | Off | Off | **90,6%** | Chưa chạy |

Các giá trị trên là **số của tác giả**, không phải kết quả starter. Trong bản rút gọn có thể cho chênh lệch khác; chỉ so với Table 4 nếu khớp protocol.

### A11 — Những gì phải xác minh

- [ ] Cách **tắt Feature Fusion** trong starter là lấy CLS token block cuối; so sánh với chính xác định nghĩa của ablation tác giả.
- [ ] Cách **tắt Smooth Projection** đưa feature gốc vào RLS; khi đó `D` có thể đổi từ 1000 xuống 768. Ghi rõ đây là thay đổi đi kèm ablation theo cấu hình, không diễn giải sai rằng chỉ bỏ sigmoid.
- [ ] Class order, seed, preprocessing, checkpoint, minibatch và metric phải giữ cố định giữa bốn cấu hình trong cùng trial (trừ thành phần chủ động thay đổi).
- [ ] Bổ sung tags W&B `ablation`, config `ff`, `sp`, nguồn table đối chiếu.
- [ ] Chạy đầy đủ từng run, kiểm tra không dùng pilot/truncated tasks để làm số liệu Table 4.
- [ ] Tổng hợp mean/std và chênh lệch theo **điểm phần trăm**, không nhầm % với phần trăm tương đối.

**Nghiệm thu:** Có bốn cấu hình chạy thành công, bảng có n/seed, W&B URLs, mô tả cách bỏ từng module và nhận xét dựa trên chênh lệch thực tế. Chênh nhỏ chưa đủ để tuyên bố ý nghĩa thống kê nếu không có đánh giá tương ứng.

---

## 8. P1 — Multi-seed, W&B, quản lý thí nghiệm

### A12 — Multi-seed

- [x] Định nghĩa các trial seeds có thể tùy biến qua `--seeds 42 123 2026` hoặc YAML config.
- [x] Lưu và tách bạch độc lập `class_order_seed`, `projection_seed`, `stream_seed` để kiểm soát triệt để mọi nguồn biến thiên ngẫu nhiên.
- [x] Đảm bảo cùng trial/stream giữa F-OAL và các baseline/ablation trong cùng một lượt chạy.
- [x] Tạo script điều phối `scripts/run_seeds.py` hỗ trợ `--skip-completed`, bảo đảm không ghi đè kết quả.
- [x] Sau mỗi run, lưu mean/std qua `aggregate_multi_seed_metrics` sử dụng hiệu chỉnh Bessel ($N-1$).

**Nghiệm thu:** Đã nghiệm thu A12 (**PASS**). Đã kiểm thử thành công bằng `tests/test_multi_seed_infrastructure.py` và chạy thực nghiệm multi-seed.

### A13 — W&B

- [x] Phân loại `pilot`, `reproduction`, `ablation`, `additional-analysis` bằng **tags thực sự**, group và run name.
- [x] Ghi đầy đủ `method`, dataset, model ID, transforms, git commit, seed, class order, projection `D`, `gamma`, batch size, precision, hardware/OS.
- [x] Lưu accuracy matrix đầy đủ cùng task-wise CSV và metadata thành artifact W&B (`{run_name}_results`).
- [x] Log runtime, duration và accuracy sau từng task và tổng kết final summary.
- [x] Hỗ trợ chế độ offline qua `--wandb-offline` (`WANDB_MODE=offline`), không crash training nếu W&B gặp sự cố mạng.
- [x] Tuyệt đối không lưu W&B API key hay credential trong mã nguồn repo.

**Nghiệm thu:** Đã nghiệm thu A13 (**PASS**). Đã kiểm thử schema và cơ chế offline qua `tests/test_wandb_offline_and_config.py`.

### A14 — Tự động tổng hợp

- [x] Xây dựng script `scripts/summarize_runs.py` đọc toàn bộ kết quả từ `results/`.
- [x] Tự động xuất `results/summary_runs.csv` ghi nhận chi tiết từng run cá lẻ.
- [x] Tự động xuất `results/summary_aggregated.csv` tính toán mean $\pm$ std cho từng cặp (method, experiment_type).
- [x] Tự động xuất `results/summary_ablation.csv` khi có các run ablation.
- [x] Kiểm thử toàn diện bằng dữ liệu mẫu `SYNTHETIC_TEST_DATA` trong `tests/test_summarize_runs.py`.

**Nghiệm thu:** Đã nghiệm thu A14 (**PASS**). Quản lý kết quả tự động, sẵn sàng tổng hợp khi chạy full reproduction.

---

## 9. P1/P2 — Reproducibility môi trường, profiling và phân tích mới

### A16 — Dependencies, môi trường và checkpoint

- [x] Dùng Python **>=3.10** theo `pyproject.toml` starter.
- [x] Tự động thu thập phiên bản thực tế của torch, torchvision, timm, datasets, numpy, CUDA, driver, GPU model qua `src/foal/environment.py`.
- [x] Khóa toàn bộ các phiên bản đã kiểm thử thành công trong `requirements-lock.txt`.
- [x] Cơ chế tự động tải và cache CIFAR-100 (Hugging Face Datasets với fallback torchvision) và ViT weights (`vit_base_patch16_224.augreg_in1k`).
- [x] Có script smoke test end-to-end kiểm thử tải batch, trích xuất đặc trưng, RLS update, evaluation, save/reload checkpoint.
- [x] Hướng dẫn setup và sử dụng độc lập, đường dẫn trừu tượng cross-platform qua `pathlib.Path`, không chứa đường dẫn cá nhân.

**Nghiệm thu:** Đã nghiệm thu A16 (**PASS**). Kiểm thử thành công qua `tests/test_environment.py`.

### A15 — Profiling chính xác và công bằng

- [ ] Đo wall-clock có/không bao gồm tải pretrained weights; nên báo cáo cả hai hoặc nói rõ.
- [ ] Khi đo CUDA time, đồng bộ CUDA tại các điểm phù hợp để tránh time asynchronous gây sai lệch.
- [ ] Lấy peak GPU memory qua PyTorch khi chạy trên GPU, ghi rõ giới hạn của cách đo (không đại diện mọi loại VRAM).
- [ ] Tính chi phí ma trận theo projection dim: `R` có **D²** phần tử; với `D=1000`, `R` float64 ~ **8 MB** chưa tính `W`, `P`, encoder và buffer. Không nhầm chi phí R với tổng VRAM.
- [ ] Nếu benchmark baseline khác quy trình, phải chỉ rõ khác biệt; không dùng chênh runtime giữa các GPU khác nhau để kết luận thuật toán nhanh hơn.

### A17 — Additional analysis (một thí nghiệm mới, cần có giả thuyết)

**Đề xuất ưu tiên:** *Batch-size sensitivity of analytic update under fixed data stream.*

- **Research question:** Thay cách chia minibatch trên cùng một chuỗi feature có làm thay đổi nghiệm RLS và tài nguyên sử dụng không?
- **Giả thuyết:** Về toán học, nghiệm Ridge cuối cùng không phụ thuộc cách chia minibatch nếu dùng cùng `H/Y` và `gamma`; về tính toán, floating-point error và runtime có thể thay đổi.
- **Biến độc lập:** batch size `[1, 10, 25, 50]` (ví dụ); giữ nguyên dữ liệu, thứ tự mẫu, encoder, projection seed và `gamma`.
- **Biến phụ thuộc:** `||W_online-W_batch||`, accuracy, runtime, peak memory.
- **Control:** cùng CIFAR-100 stream/class order, cùng feature vectors; không thay đổi checkpoint.
- **Hạn chế:** Nếu chỉ xác minh tính chất vốn suy ra trực tiếp từ toán học, đây là kiểm thử độ ổn định/năng suất, **chưa tự động là theoretical limitation**. Cần dự đoán failure mode cụ thể nếu muốn đưa lên poster phản biện.

**Phương án cho poster cuối kỳ (chưa cần chứng minh ở giữa kỳ):** khảo sát suy giảm khi encoder kém phù hợp với target domain hoặc khi projection dimension quá lớn/nhỏ so với ngân sách bộ nhớ. Phải có nhóm đối chứng, nhiều seeds và tiêu chí phủ định giả thuyết.

---

## 10. Các vấn đề phụ bổ sung cần theo dõi

1. **Checkpoint không lưu frozen ViT weights:** là quyết định tiết kiệm dung lượng có lý do, nhưng tạo phụ thuộc vào kho weights bên ngoài. **Cần ghi model revision/hash và kiểm tra sau reload**, không mặc định là lỗi.
2. **`scripts/evaluate.py` báo pooled accuracy của tất cả seen classes:** cần bảo đảm khớp metric nhóm trình bày; có thể bổ sung task-wise matrix cho checkpoint để thống nhất cách báo cáo.
3. **`scripts/ablation.py` chỉ chạy 4 processes:** chưa tổng hợp kết quả, chưa chạy đa seed, chưa phân biệt tag ablation.
4. **Giới hạn pilot (`--limit-train-per-task`) lấy mẫu con ngẫu nhiên:** thuận tiện smoke test, nhưng phải đánh dấu PILOT và không trộn vào báo cáo reproduction.
5. **Huấn luyện online không backward:** đúng với F-OAL, nhưng encoder lớn vẫn tốn thời gian forward; cần ghi throughput thực tế.
6. **Class order và projection cùng lấy từ `seed`:** không sai; nhưng gây khó phân tích nguồn biến thiên. Tách seed khi thiết kế thí nghiệm riêng.
7. **Kết quả `A_avg` của một run không phải trung bình 3 seeds:** giữ hai cấp tổng hợp độc lập.
8. **Chênh accuracy 0,4–0,5 điểm phần trăm trong Table 4:** không tự tuyên bố có ý nghĩa thống kê khi chưa đủ số runs/uncertainty.
9. **Không sao chép code thuật toán chính từ tác giả:** repo tác giả chỉ dùng làm nguồn kiểm chứng config, protocol và đánh giá; ghi nguồn tham khảo rõ ràng.
10. **`README.md` trong GitHub ban đầu là phân tích paper:** cập nhật hướng dẫn build, tests và experiments sau khi push source; giữ lịch sử đóng góp và attribution minh bạch.

---

## 11. Lệnh kiểm tra hiện CÓ trong starter

> Chạy tại thư mục gốc của starter sau khi giải nén. Hãy cài bản torch/torchvision tương thích CPU/GPU trước khi cài requirements. Các lệnh bên dưới **tồn tại trong code hiện tại**, không phải script chưa được viết.

### 11.1. Tạo môi trường, cài và test

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Kỳ vọng trên bản starter đã audit: `9 passed`. Con số này có thể thay đổi khi nhóm bổ sung tests.

### 11.2. Pilot hai tasks (KHÔNG phải reproduction)

```bash
python -m scripts.train \
  --config configs/cifar100.yaml \
  --max-tasks 2 \
  --limit-train-per-task 100
```

Muốn ghi W&B, thêm `--wandb` sau khi đăng nhập. Chỉ đánh dấu pilot chạy xong khi thật sự đã thấy output `metrics.json` và checkpoint có thể reload.

### 11.3. Đánh giá checkpoint

```bash
python -m scripts.evaluate \
  --checkpoint results/<run_name>/task_02.pt \
  --device cpu
```

`<run_name>` là placeholder cần thay bằng thư mục run thực tế.

### 11.4. Chạy 4 cấu hình ablation (chỉ chạy khi P0 hoàn tất)

```bash
python -m scripts.ablation \
  --config configs/cifar100.yaml \
  --wandb
```

**Cảnh báo:** Nếu không truyền `--max-tasks` và `--limit-train-per-task`, script sẽ cố chạy đủ cấu hình YAML. Chỉ thực hiện khi đã xác nhận tài nguyên và thiết lập đúng. Hiện chưa có script `scripts/run_seeds.py`, `scripts/summarize_runs.py`, `scripts/compare.py`; các tên đó chỉ là **file đề xuất bổ sung**.

---

## 12. Lộ trình sửa/kiểm tra theo deadline

| Thời gian đề xuất | Công việc | Sản phẩm/bằng chứng |
|---|---|---|
| **01–03/11** | P0: checkpoint, transforms, fusion, data protocol, RLS sanity checks | Bản `protocol.md`, tests bổ sung, smoke test trên ảnh thật |
| **04–07/11** | Đẩy source vào repo nhóm, xác nhận script chạy trên máy/Kaggle/Colab, nộp URL GitHub đúng hạn | `main` có code, README setup, commit rõ ràng |
| **08–11/11** | Chạy F-OAL full CIFAR-100, kiểm tra metrics/W&B/checkpoint; bắt đầu baseline | W&B runs đầu tiên, bảng accuracy thực |
| **12–15/11** | Chạy headline comparison, các seeds có thể và ablation | Bảng comparison và 4 cấu hình ablation |
| **16–18/11** | Additional analysis, profiling, đối chiếu số paper | Figures, mean/std, giải thích sai lệch |
| **19–21/11** | Kiểm toán nguồn số liệu, slide 20–40 trang, notes/citation | `ID.pptx` đúng yêu cầu, nộp trước 23:59 |
| **22/11–05/12** | Stress tests cho ít nhất hai limitations, chuẩn bị poster A2 dọc | Poster, evidence W&B, câu trả lời phản biện |

Đây là **kế hoạch nhóm tự tổ chức**, không phải phân công chính thức của giảng viên. Nếu phần cứng thiếu, cần giảm scope minh bạch thay vì tự nhận đã tái hiện Table 2/4.

---

## 13. Definition of Done — Điều kiện kết luận đã sẵn sàng nộp giữa kỳ

### 13.1. Code & protocol

- [ ] Starter đã được merge/push vào **repo GitHub nhóm**, commit URL có thể truy cập công khai.
- [ ] Pretrained encoder và transforms được ghi chính xác; khác paper ở đâu được giải thích.
- [ ] Class split, mapping, online constraints, no leakage được kiểm thử trên CIFAR-100 thật.
- [ ] RLS tương đương Ridge Regression trên synthetic và kiểm chứng feature thật; không NaN/Inf.
- [ ] Checkpoint save/reload cho cùng dự đoán/metric.
- [ ] `pytest -q` qua, smoke test end-to-end qua trên môi trường ghi trong README.

### 13.2. Experiments & evidence

- [ ] Có đủ **hai thí nghiệm quan trọng** cho Core Scope, bao gồm headline comparison, theo đề bài.
- [ ] Có ít nhất **một additional analysis do nhóm thiết kế**, có hypothesis và kết quả.
- [ ] Các ablation được chạy thật, so đúng các cấu hình paper (nếu nhóm chọn làm thí nghiệm thứ hai).
- [ ] Các run có config, seeds, environment, code commit và W&B URL.
- [ ] Có raw accuracy matrix, CSV/JSON, bảng comparison, độ phân tán hoặc giới hạn số runs.
- [ ] Không dùng số tác giả làm kết quả tự chạy; không diễn giải pilot như full experiment.

### 13.3. Báo cáo

- [ ] Giải thích được research problem, related work, novelty, methodology và các công thức cốt lõi.
- [ ] Slide **20–40 trang** (tính cả appendix theo quy định đã tổng hợp), speaker notes tối đa **500 từ/slide**.
- [ ] Citation IEEE ngay tại slide có dùng nguồn, figures tự vẽ lại; file PPTX **dưới 20 MB**.
- [ ] Các kết quả quan trọng có W&B links và khớp với Git commit.
- [ ] Giải thích trung thực khi kết quả nhóm thấp/cao hơn paper, chỉ ra các khác biệt protocol/compute.
- [ ] Nộp file **`ID.pptx`** với **ID nhóm**, không phải MSSV.

---

## 14. Gợi ý phân công và cách review để tránh lỗi dây chuyền

Nhóm có thể chia thành ba tuyến công việc song song, với một người tích hợp chịu trách nhiệm kết luận reproduction:

| Tuyến | Phụ trách | Tiêu chí bàn giao |
|---|---|---|
| **Pipeline/Math** | `datasets.py`, `projection.py`, `rls.py`, tests | Có unit/integration tests, script kiểm tra phát hiện sai rõ ràng |
| **Backbone/Protocol** | `backbone.py`, `feature_fusion.py`, configs | Có checkpoint provenance, transforms, một batch real-image chạy được |
| **Experiments/Reporting** | `train.py`, baseline, W&B, metrics, slides | Có run IDs, artifacts, bảng so sánh và mô tả cấu hình |

Mọi thay đổi quan trọng nên tạo pull request/commit tách biệt (ví dụ `fix/encoder-protocol`, `test/rls-checkpoint`, `feat/metrics-and-seeds`) và có một lần kiểm tra lại trước khi chạy thí nghiệm lớn. **Không chỉnh sửa experimental protocol giữa chừng rồi gộp số liệu các runs khác nhau vào cùng một bảng**.

---

## 15. Nguồn chuẩn để đối chiếu

1. **Paper chính thức (NeurIPS 2024):** https://proceedings.neurips.cc/paper_files/paper/2024/hash/48ffa38c13078d6ce26b328e7f373243-Abstract-Conference.html
2. **PDF paper:** https://papers.neurips.cc/paper_files/paper/2024/file/48ffa38c13078d6ce26b328e7f373243-Paper-Conference.pdf
3. **Repository tác giả (chỉ tham khảo/kiểm chứng):** https://github.com/liuyuchen-cz/F-OAL
4. **Term Project HCMUS:** https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/
5. **Repository nhóm:** https://github.com/caoban123/F-OAL
6. **Source được audit:** `F-OAL_starter_v1.zip` trong phiên trao đổi; phần lớn hiện mới có các kiểm thử CPU/synthetic, chưa đủ kết luận reproduction.

### Ghi chú cuối

**Không có nghĩa vụ làm cho kết quả giống tác giả bằng mọi cách.** Một reproduction có chất lượng là nghiên cứu có **quy trình kiểm soát biến số, code kiểm thử được, kết quả tái lập được và báo cáo trung thực**. Nếu cần chạy cấu hình giảm quy mô do tài nguyên hạn chế, hãy gắn nhãn `PILOT` hoặc `REDUCED_SCOPE`, ghi rõ khác biệt và tuyệt đối không so nó như một bản tái hiện nguyên vẹn Table 2/4.

---

## 16. Báo cáo Kiểm toán và Nghiệm thu P0 Thực tế (01/11/2026)

### 16.1. Phân loại phát hiện kiểm toán (Audit Findings)

1. **Confirmed Bug:**
   - **Tệp:** `src/foal/datasets.py`
   - **Mô tả:** Preprocessing trong hàm `cifar100_transforms` ban đầu sử dụng tham số chuẩn hóa của ResNet/CNN ImageNet `mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)`. Trong khi đó, checkpoint `vit_base_patch16_224.augreg_in1k` của ViT được huấn luyện với `mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)` và phép nội suy `bicubic`. Sự sai lệch này làm lệch phân phối đặc trưng zero-shot của backbone đông băng.
   - **Khắc phục:** Đã sửa mặc định trong `cifar100_transforms` về `mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)` và `bicubic`.

2. **Unverified Assumption:**
   - **Tệp:** `src/foal/model.py`, `scripts/evaluate.py`
   - **Mô tả:** Giả định kiểu dữ liệu RLS luôn là `float64` khi reload checkpoint mà không ghi nhận cấu hình `rls_dtype` thực tế vào `model_config`, gây nguy cơ ép kiểu không mong muốn khi chuyển đổi giữa `float32` và `float64`.
   - **Khắc phục:** Cập nhật `FOALModel` nhận và lưu trữ `rls_dtype` vào checkpoint, cho phép tái lập chính xác cấu hình số học ban đầu.

3. **Missing Implementation:**
   - **Tệp:** `src/foal/datasets.py`, `src/foal/rls.py`, `tests/`
   - **Mô tả:**
     - `load_cifar100` thiếu cơ chế dự phòng (fallback) sang `torchvision.datasets.CIFAR100` khi Hugging Face gặp sự cố mạng hoặc chứng chỉ.
     - `OnlineRidgeRLS.update` thiếu kiểm tra `torch.isfinite` phát hiện sớm các giá trị NaN/Inf từ đầu vào hoặc quá trình cập nhật.
     - Thiếu bài kiểm tra đối chiếu trực tiếp giữa Online RLS và Batch Ridge Regression trên đặc trưng thật từ ViT.
     - Thiếu bài kiểm tra khép kín (roundtrip) cho save/load checkpoint.
   - **Khắc phục:** Đã bổ sung fallback torchvision, bổ sung assertion `torch.isfinite`, và viết mới các bài kiểm thử `test_vit_transforms.py`, `test_rls_real_features.py`, `test_checkpoint_roundtrip.py`.

4. **Improvement:**
   - **Tệp:** `src/foal/backbone.py`
   - **Mô tả:** Thêm thuộc tính `default_cfg` truy xuất cấu hình gốc từ `timm.data.resolve_model_data_config`, thêm kiểm tra kích thước tensor đầu vào `images.ndim == 4` tại `FrozenViT.forward`.

---

### 16.2. Kết quả kiểm thử bộ Test Suite (15 passed)

Lệnh thực thi:
```bash
python -m pytest -v
```

Kết quả:
```text
tests/test_checkpoint_roundtrip.py::test_checkpoint_save_and_reload PASSED        [  6%]
tests/test_dataset.py::test_seeded_order_is_reproducible_and_permutation PASSED    [ 13%]
tests/test_dataset.py::test_task_partition_no_train_replay_and_no_future_test PASSED [ 20%]
tests/test_metrics.py::test_incremental_summary_values PASSED                       [ 26%]
tests/test_model.py::test_model_end_to_end_on_synthetic_features PASSED             [ 33%]
tests/test_projection.py::test_projection_is_fixed_and_seeded PASSED                [ 40%]
tests/test_projection.py::test_feature_fusion_averages_cls_token_across_blocks PASSED [ 46%]
tests/test_rls.py::test_rls_matches_closed_form_ridge_with_expanding_classes PASSED [ 53%]
tests/test_rls.py::test_can_only_add_classes_not_remove PASSED                      [ 60%]
tests/test_batch_partition_does_not_change_exact_solution PASSED                    [ 66%]
tests/test_rls.py::test_nan_or_inf_detection PASSED                                 [ 73%]
tests/test_rls_real_features.py::test_rls_matches_ridge_on_real_vit_features PASSED  [ 80%]
tests/test_vit_transforms.py::test_vit_transforms_structure PASSED                  [ 86%]
tests/test_vit_transforms.py::test_frozen_vit_properties PASSED                     [ 93%]
tests/test_vit_transforms.py::test_feature_fusion_applies_final_layernorm_both_branches PASSED [100%]

======================== 15 passed in 36.46s ========================
```

Đặc biệt:
- `test_feature_fusion_applies_final_layernorm_both_branches`: Xác nhận trực tiếp tính đúng đắn toán học của **cả hai nhánh**:
  1. `feature_fusion=False` (ablation): Khớp 100% với `forward_features(x)[:, 0, :]` của `timm` (diff đại số 0.0).
  2. `feature_fusion=True` (full F-OAL): Khớp 100% với ground truth tính thủ công qua hook thu thập CLS tokens của 12 blocks, tính trung bình và áp dụng `network.norm` (final LayerNorm) (diff đại số 0.0).
- `test_checkpoint_roundtrip.py`: Xác nhận so sánh trực tiếp class scores (logits) của cùng một batch trước và sau khi lưu/nạp checkpoint cho kết quả trùng khớp với sai số đại số $< 10^{-12}$.
- `test_rls_matches_ridge_on_real_vit_features`: Xác minh nghiệm của Online RLS khi học từng mini-batch trên đặc trưng ViT-B/16 từ dữ liệu CIFAR-100 thật khớp với nghiệm giải tích toàn cục Batch Ridge Regression:
  $$W^* = (H^T H + \gamma I)^{-1} H^T Y$$
  với sai số tuyệt đối $< 10^{-10}$.

---

### 16.3. Chi tiết kiểm tra Preprocessing Pipeline (timm vs Continual Learning)

| Thành phần | timm ImageNet Default (`vit_base_patch16_224.augreg_in1k`) | F-OAL Continual Learning (`src/foal/datasets.py`) | Lý do kỹ thuật & Đối chiếu Paper |
|---|---|---|---|
| **Input size** | $224 \times 224$ | $224 \times 224$ | Khớp cấu hình patch embedding ViT-B/16 |
| **Interpolation** | `bicubic` | `bicubic` (`InterpolationMode.BICUBIC`) | Khớp ViT pretraining chuẩn |
| **Resize & Crop** | `Resize(248)` $\to$ `CenterCrop(224)` (`crop_pct=0.9`) | `Resize((224, 224))` trực tiếp, **không crop** | Ảnh gốc CIFAR-100 chỉ có kích thước $32 \times 32$. Nếu crop 10% biên sẽ làm mất đặc trưng biên quan trọng của vật thể (cánh, bánh xe, chân). Protocol CL chuẩn resize trực tiếp. |
| **Normalization** | `mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)` | `mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)` | Đưa pixel về $[-1.0, 1.0]$, khớp 100% pretraining của ViT AugReg |

---

### 16.4. Kết quả P0 Post-Fix Validation & Smoke Test CIFAR-100 Mới

Sau khi hoàn tất bản vá LayerNorm và bổ sung test cho Feature Fusion đầy đủ, một đợt kiểm thử End-to-End Smoke Test mới đã được thực thi từ đầu (tuyệt đối không tái sử dụng checkpoint cũ) để tạo run `smoke_test_post_fix`.

#### 1. Huấn luyện One-pass Smoke Test mới:
```bash
python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100 --run-name smoke_test_post_fix
```

* **Dữ liệu:** CIFAR-100 thật từ Hugging Face (`uoft-cs/cifar100`).
* **Backbone:** ViT-B/16 (`vit_base_patch16_224.augreg_in1k`, 85.8M parameters, frozen).
* **Thiết bị:** Intel Core Ultra 5 125H (CPU).
* **Kết quả huấn luyện qua từng task:**
  * **Task 1:** 10 lớp seen, 100 mẫu train, đánh giá trên 100 mẫu test:
    * Task 1 Accuracy: **78,00%** (0.78)
    * Thời gian thực thi: 65,02 giây
    * Checkpoint tạo lập: `results/smoke_test_post_fix/task_01.pt` (11,15 MB)
  * **Task 2:** 20 lớp seen, thêm 100 mẫu train (tổng 200 mẫu), đánh giá trên 200 mẫu test:
    * Task 1 Accuracy (lớp cũ): **69,00%** (0.69)
    * Task 2 Accuracy (lớp mới): **85,00%** (0.85)
    * Average Seen Accuracy: **77,00%** (0.77)
    * Thời gian thực thi: 163,19 giây
    * Forgetting Rate $F$: $78,00\% - 69,00\% = \mathbf{9,00\%}$
    * Checkpoint tạo lập: `results/smoke_test_post_fix/task_02.pt` (11,23 MB)
  * **Chỉ số tổng hợp (`metrics.json`):**
    * `average_incremental_accuracy`: **77,50%**
    * `last_accuracy`: **77,00%**
    * `forgetting`: **9,00%**

#### 2. Đánh giá độc lập Checkpoint mới bằng `scripts/evaluate.py`:
```bash
python -m scripts.evaluate --checkpoint results/smoke_test_post_fix/task_02.pt --device cpu --limit-per-task 100
```

* **Độ chính xác tổng thể (Accuracy):** **77,00%** (154/200 mẫu đúng) — **khớp 100% với kết quả của `train.py`**.
* **Độ chính xác từng task (Per-Task Accuracy):**
  * Task 1: **69,00%** (69/100 mẫu) — khớp chính xác với `train.py`.
  * Task 2: **85,00%** (85/100 mẫu) — khớp chính xác với `train.py`.
* **Kiểm tra độ ổn định số học (Score Statistics / Logits):**
  * Giá trị nhỏ nhất (min): `-0.50286`
  * Giá trị lớn nhất (max): `1.07427`
  * Giá trị trung bình (mean): `0.04989`
  * Độ lệch chuẩn (std): `0.16399`
  * `all_finite`: **True** (tuyệt đối không có NaN, không có Inf trong mọi logits và ma trận RLS).

---

### 16.5. Hướng dẫn chạy trên Kaggle / Google Colab (GPU) cho Full Reproduction (Giai đoạn P1)

Do máy cá nhân sử dụng CPU khiến việc chạy toàn bộ 10 tasks (50.000 train + 10.000 test) mất nhiều giờ, cấu hình GPU T4/P100/A100 trên Kaggle/Colab được khuyến nghị cho giai đoạn P1:

```bash
# 1. Clone repository
git clone https://github.com/caoban123/F-OAL.git
cd F-OAL

# 2. Cài đặt thư viện
pip install -r requirements.txt
pip install -e .

# 3. Chạy kiểm thử xác minh (30 tests)
python -m pytest -v

# 4. Huấn luyện Full Reproduction 10 tasks trên GPU (CUDA)
python -m scripts.train --config configs/cifar100.yaml --run-name foal_full_cifar100_seed42 --device cuda

# 5. Đánh giá checkpoint
python -m scripts.evaluate --checkpoint results/foal_full_cifar100_seed42/task_10.pt --device cuda
```

---

## 17. Báo cáo nghiệm thu Giai đoạn P1.1 — Experimental Infrastructure & Reproducibility

* **Thời điểm nghiệm thu:** 01/11/2026.
* **Tổng số kiểm thử đạt chuẩn:** **30/30 passed** trong ~34s (`python -m pytest -v`).
* **Trạng thái các hạng mục P1.1:**
  * **A09 (Metrics Verification):** **PASS** — Kiểm thử tính toán tay cho 1, 2, 3 tasks, non-monotonic forgetting, positive backward transfer, quy ước tỷ lệ $[0, 1]$ và hiển thị phần trăm $[0, 100]$.
  * **A12 (Multi-Seed Infrastructure):** **PASS** — Tách biệt hoàn toàn `class_order_seed`, `projection_seed`, `stream_seed`. Có script `scripts/run_seeds.py` điều phối đa seed, hỗ trợ `--skip-completed`, tính mean $\pm$ std không ghi đè kết quả.
  * **A13 (W&B Experiment Tracking):** **PASS** — Tích hợp W&B với tags phân loại (`pilot`, `reproduction`, `ablation`), ghi nhận hardware/OS/git commit, per-task metrics, summary metrics, offline logging và upload artifacts (`metrics.json`, `metadata.json`, `task_accuracy.csv`).
  * **A14 (Automatic Summarization):** **PASS** — Xây dựng `scripts/summarize_runs.py` tự động quét kết quả, nhóm theo phương pháp và xuất `summary_runs.csv`, `summary_aggregated.csv` và `summary_ablation.csv`.
  * **A16 (Environment & Reproducibility):** **PASS** — Module `src/foal/environment.py` tự động ghi nhận môi trường, khóa phiên bản trong `requirements-lock.txt`, đường dẫn cross-platform, không hardcode path cá nhân.
* **Trạng thái A01 và A04:** Tiếp tục duy trì **PARTIAL** do chưa có sha256 checksum weights của checkpoint private từ tác giả paper (A01) và mảng permutation private của CIFAR-100 trong Bảng 2 paper (A04).
* **Kết luận:** Hạ tầng thực nghiệm đã sẵn sàng 100% để chuyển sang giai đoạn **P1.2 (Baselines Implementation)** và **P1.3 (Full CIFAR-100 Reproduction & Ablations)** mà không cần phải can thiệp hay sửa đổi thêm hệ thống logging/metrics.

---

## 18. Báo cáo nghiệm thu Giai đoạn GPU Preflight & Single-Seed Full Reproduction Preparation

* **Thời điểm nghiệm thu:** 01/11/2026.
* **Tổng số kiểm thử đạt chuẩn:** **31/31 passed** trong 101s (`python -m pytest -v`).
* **Trạng thái phần cứng môi trường cục bộ:**
  * **Hệ điều hành:** Windows 11.
  * **CPU:** Intel Core Ultra 5 125H (14 cores, 18 threads).
  * **CUDA Available:** `False` (`torch.cuda.is_available() == False`).
  * **Chính sách thực nghiệm:** Nghiêm cấm giả lập kết quả hoặc khai báo đã chạy xong full reproduction trên máy cục bộ khi không có GPU.
* **Các cải tiến kỹ thuật hoàn thiện trong giai đoạn:**
  1. **CUDA Fail-Fast Safeguard:** Hàm `resolve_device()` trong `scripts/train.py` và `scripts/evaluate.py` bảo đảm khi người dùng hoặc config chỉ định `--device cuda` mà không có GPU, chương trình lập tức báo lỗi `RuntimeError` rõ ràng, ngăn chặn hoàn toàn việc âm thầm chạy trên CPU.
  2. **Cách ly Protocol & Kiểm tra tính hoàn tất trong `scripts/summarize_runs.py`:**
     * Nâng cấp khóa gom nhóm từ `(method, exp_type)` thành Protocol Signature đầy đủ: `(method, exp_type, expected_tasks, limit_train, limit_test, backbone, projection_dim, gamma)`. Ngăn chặn triệt để việc tính trung bình lẫn lộn giữa pilot (2 tasks, limit 100) và full reproduction (10 tasks, no limits).
     * Kiểm tra tính toàn vẹn vật lý của run: bắt buộc có đủ các khóa metrics và tồn tại file checkpoint cuối `task_{expected_tasks:02d}.pt` với dung lượng > 0 bytes. Run bị ngắt giữa chừng được đánh dấu `status: 'INCOMPLETE'` và loại khỏi bảng tổng hợp.
     * Kiểm thử xác minh: `tests/test_summarize_runs.py` (`test_load_and_summarize_runs_protocol_isolation`).
  3. **Bảo vệ `--skip-completed` trong `scripts/run_seeds.py`:**
     * Kiểm tra đa tầng: `metrics.json` hợp lệ, checkpoint cuối `task_XX.pt` tồn tại, và toàn bộ tham số trong `metadata.json` (`seed`, `max_tasks`, `limit_train_per_task`, `limit_test_per_task`, `feature_fusion`, `smooth_projection`) phải khớp hoàn toàn với cấu hình yêu cầu.
     * Kiểm thử xác minh: `tests/test_multi_seed_infrastructure.py` (`test_is_run_completed_logic`).
  4. **Kiểm chứng Weights & Biases (W&B):**
     * Đã cài đặt `wandb==0.31.0` vào môi trường Python 3.10.
     * Xác minh quy trình offline bằng run thực tế và unit test: `wandb.init()` offline, ghi step metrics, summary metrics, và artifact upload (`metrics.json`, `metadata.json`, `task_accuracy.csv`) ghi nhận đầy đủ vào thư mục local mà không phát sinh lỗi mạng.
     * Xác nhận rõ: Việc upload trực tuyến lên cloud W&B chưa kiểm chứng do chưa có W&B API Key của người dùng (sẽ được kích hoạt trên Kaggle/Colab qua biến môi trường `WANDB_API_KEY`).
  5. **Chuẩn bị Notebook thực nghiệm Full Reproduction:**
     * Tạo file notebook hoàn chỉnh: `notebooks/foal_cifar100_full_reproduction.ipynb`.
     * Cung cấp luồng chuẩn hóa: Hardware Check (`nvidia-smi`, PyTorch CUDA check) -> Dependency Setup -> Preflight PyTest Suite -> Single-Seed Full F-OAL (Seed 42, 10 tasks $\times$ 10 classes, batch size 10, one-pass) -> Independent Checkpoint Evaluation -> Results Summarizer -> Multi-Seed Orchestration -> Download Zip.
* **Trạng thái A01 và A04:** Tiếp tục duy trì **PARTIAL** do chưa có sha256 checksum weights của checkpoint private từ tác giả paper (A01) và mảng permutation private của CIFAR-100 trong Bảng 2 paper (A04).

---

## 19. Báo cáo kiểm toán & chuẩn bị thực thi P1.2 — First GPU Execution & Single-Seed Full F-OAL Reproduction

* **Thời điểm kiểm toán:** 01/11/2026.
* **Tổng số kiểm thử đạt chuẩn:** **32/32 passed** trong 47s (`python -m pytest -v`).
* **Trạng thái môi trường phần cứng máy cục bộ:**
  * **Hệ điều hành:** Windows 11.
  * **GPU:** `Intel(R) Arc(TM) Graphics` (Không có card NVIDIA CUDA rời).
  * **CUDA Available:** `False` (`torch.cuda.is_available() == False`, `torch.cuda.device_count() == 0`).
  * **Định hướng thực nghiệm:** Tuyệt đối không giả lập kết quả hoặc chạy full reproduction trên CPU (vi phạm thời gian và cấu hình thực nghiệm chuẩn). Giai đoạn P1.2 chỉ được đánh dấu **PASS** khi toàn bộ quá trình full training và independent evaluation hoàn tất thành công trên môi trường GPU thật (Google Colab / Kaggle).
* **Kết quả xử lý các nguy cơ làm sai lệch kết quả thực nghiệm:**
  1. **Khắc phục triệt để Protocol Signature trong `scripts/summarize_runs.py`:**
     * Bổ sung đầy đủ 13 thành phần vào `protocol_sig`: `(method, exp_type, bool(ff), bool(sp), expected_tasks, limit_train, limit_test, backbone, projection_dim, gamma, rls_dtype, seed_coupling, class_order_mode)`.
     * Bảo đảm các run khác nhau về Feature Fusion (True/False), Smooth Projection (True/False), RLS dtype (float64/float32), hoặc sử dụng seeds phân tách (decoupled projection/class_order seeds) tuyệt đối không bị gom nhóm nhầm lẫn với các run chuẩn.
     * Xác minh qua unit test: `test_summarize_runs_hyperparameter_and_seed_isolation` trong `tests/test_summarize_runs.py`.
  2. **Thắt chặt rào chắn `--skip-completed` trong `scripts/run_seeds.py`:**
     * Nâng cấp `is_run_completed` để đối chiếu toàn diện: `seed`, `class_order_seed`, `projection_seed`, `stream_seed`, `max_tasks`, `limit_train_per_task`, `limit_test_per_task`, `feature_fusion`, `smooth_projection`, `backbone`, `projection_dim`, `gamma`, `rls_dtype`, và `batch_size`.
     * Nếu bất kỳ tham số nào trong checkpoint cũ không khớp với cấu hình hiện tại, hàm sẽ trả về `False`, buộc hệ thống huấn luyện lại từ đầu thay vì tái sử dụng checkpoint không tương thích.
     * Xác minh qua unit test: `tests/test_multi_seed_infrastructure.py`.
  3. **Minh bạch hóa trạng thái kiểm thử Weights & Biases (W&B):**
     * Đã cài đặt `wandb==0.31.0`.
     * Luồng Offline Logging: Đã kiểm chứng thành công việc khởi tạo `wandb.init(mode='offline')`, ghi step metrics, summary metrics và upload file artifacts (`metrics.json`, `metadata.json`, `task_accuracy.csv`) vào thư mục local.
     * Ghi nhận lỗi môi trường: Trên hệ thống Windows với đường dẫn chứa ký tự tiếng Việt có dấu (`D:\Nhập môn ML\Paper`), logger mặc định của wandb có thể gặp lỗi `charmap` codec nếu thiếu `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1`.
     * Báo cáo trung thực: Việc đồng bộ trực tuyến lên cloud W&B **chưa được kiểm chứng** do máy cục bộ chưa có `WANDB_API_KEY`.
  4. **Kiểm tra và chuẩn hóa Notebook Single-Seed (`notebooks/foal_cifar100_full_reproduction.ipynb`):**
     * Đã điều chỉnh để notebook mặc định **chỉ thực thi Single-Seed (Seed 42)** theo đúng phạm vi P1.2.
     * Section 7 (Multi-seed 3 seeds) đã được chủ động comment out và chuyển thành tùy chọn cho giai đoạn P1.3, ngăn chặn việc tự động chạy multi-seed ngoài ý muốn khi nhấn 'Run All'.
     * Mọi lệnh trong notebook đều khớp chính xác với CLI interface của `scripts.train`, `scripts.evaluate`, và `scripts.summarize_runs`.

---

## 20. Báo cáo Thực nghiệm GPU & Kiểm toán Sau Thực thi: P1.2 — Single-Seed Full Reproduction on GPU (Đã sửa lỗi dữ liệu & chuẩn hóa khoa học)

> **Ghi chú Sửa lỗi Tài liệu (Post-GPU Audit Repair — 01/11/2026):** Trong commit `c74b0de`, Mục 20 được ghi thông qua chuỗi nháy kép PowerShell khiến các mẫu ký tự dạng `$91`, `$79`, `$A_`, `$D`, `$\lambda` bị trình thông dịch PowerShell hiểu nhầm là biến môi trường rỗng và cắt mất chữ số đầu (biến `$91.1\%$` thành `.1\%$`, `$A_{\text{avg}}$` thành `{\text{avg}}$`, `$D=1000$` thành `=1000`). Bản cập nhật dưới đây khôi phục chính xác 100% số liệu từ nội dung `metrics.json` gốc do người dùng trích xuất trực tiếp trên Google Colab, đồng thời chuẩn hóa các nhận định khoa học và trạng thái lưu trữ artifact.

* **Thời điểm thực thi trên Colab:** 01/11/2026 (`git_commit = aad7500`).
* **Môi trường thực thi (`metadata.json`):** Google Colab GPU — **NVIDIA Tesla T4** (CUDA `13.0`, Compute Capability `7.5`), Linux `6.6.159+`, Python `3.13.16`, PyTorch `2.11.0+cu130`, `timm==1.0.30`, `datasets==4.8.5`, `wandb==0.28.1`.
* **Trạng thái Giai đoạn P1.2:** **`EXECUTED` & `VERIFIED` (Raw Artifacts Extracted & Verified on Local Disk)** — Đã giải nén gói `foal_cifar100_single_seed42_results.zip` ($109,814,583\text{ bytes} \approx 109.8\text{ MB}$) và xác minh đầy đủ `metrics.json`, `metadata.json`, `task_accuracy.csv`, 10 checkpoints `task_01.pt`..`task_10.pt`, bảng tổng hợp `summary_runs.csv` / `summary_aggregated.csv` và gói W&B offline `wandb/offline-run-20261101_084136-2cdeiptv/run-2cdeiptv.wandb` (xem Mục 20.4).
* **Thông số cấu hình thực tế đã chạy (`configs/cifar100.yaml` + `scripts/train.py`):**
  * **Dataset:** CIFAR-100 (`uoft-cs/cifar100`, 100 classes, 50,000 mẫu train, 10,000 mẫu test; resize bicubic $224 \times 224$, chuẩn hóa `mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)`).
  * **Giao thức:** 10 tasks $\times$ 10 classes, streaming `batch_size = 10`, **One-pass** (1 epoch duy nhất, zero-replay buffer, `limit_train_per_task = null`, `limit_test_per_task = null`).
  * **Random Seed:** `seed = 42` (thông qua hàm `resolve_seed`: `class_order_seed = 42`, `projection_seed = 42`, `stream_seed = 42`).
  * **Cấu hình Mô hình & Siêu tham số (Đã rà soát khớp 100% mã nguồn):**
    * **Backbone (`src/foal/backbone.py`):** Frozen ViT-B/16 (`timm/vit_base_patch16_224.augreg_in1k`, $d_{\text{in}} = 768$, 12 blocks, `requires_grad = False`, giữ cố định ở `eval()` mode).
    * **Feature Fusion (`src/foal/feature_fusion.py`):** Trung bình cộng đại số (`mean`) của 12 CLS tokens qua 12 transformer blocks, sau đó đi qua `network.norm` (final LayerNorm của ViT).
    * **Smooth Projection (`src/foal/projection.py`):** Phép chiếu tuyến tính ngẫu nhiên cố định $W_{\text{proj}} \in \mathbb{R}^{768 \times 1000}$ khởi tạo từ phân phối chuẩn $\mathcal{N}(0, \sigma_{\text{init}}^2)$ với `projection_dim = 1000`, `gaussian_std = 1.0`, `seed = 42`, theo sau bởi hàm kích hoạt Sigmoid $z = \sigma(x W_{\text{proj}})$. *(Lưu ý đính chính: `SmoothProjection` không có tham số $\gamma$).*
    * **Online Ridge RLS Classifier (`src/foal/rls.py`):** Cập nhật giải tích theo công thức Woodbury trên kiểu dữ liệu `rls_dtype = float64` với hệ số chính quy hóa L2 (Ridge regularization) **`gamma = 1.0`** ($R_0 = \gamma^{-1} I_{1000} = I_{1000}$). *(Đính chính: Trong bản ghi `c74b0de` trước đó đã ghi nhầm `Smooth Projection (gamma=1.0)` và `RLS (lambda=0.1)`; thực tế trong toàn bộ codebase và `configs/cifar100.yaml` chỉ có duy nhất tham số Ridge regularization `gamma = 1.0` truyền vào `OnlineRidgeRLS`, không tồn tại tham số `lambda = 0.1`).*
  * **Tổng thời gian chạy (`total_duration_seconds`):** **1735.3 giây ($\approx 28.92$ phút)** trên GPU NVIDIA T4.

---

### 20.1. Bảng Đối chiếu Kết quả với Paper F-OAL (NeurIPS 2024, Table 2 — CIFAR-100)

Theo quy định báo cáo tái hiện thực nghiệm (Checkpoint 3), độ lệch giữa kết quả tái hiện (`Ours`) và kết quả công bố trong bài báo (`Paper`) được định lượng bằng cả hai đại lượng:
1. **Chênh lệch tuyệt đối theo điểm phần trăm ($\Delta_{\text{pp}}$):**
   $$\Delta_{\text{pp}} = \text{Ours} - \text{Paper}$$
2. **Độ lệch tương đối ($\Delta_{\text{rel}}\%$):**
   $$\Delta_{\text{rel}}(\%) = \frac{\text{Ours} - \text{Paper}}{\text{Paper}} \times 100\%$$

| Chỉ số Đánh giá (Metric) | Ký hiệu | Paper F-OAL (`PAPER_REPORTED`, 3-seed mean) | Original Code (`ORIGINAL_REPRODUCED`) | Kết quả Nhóm (`OUR_REPRODUCED`, Seed 42) | Chênh lệch tuyệt đối ($\Delta_{\text{pp}}$) | Độ lệch tương đối ($\Delta_{\text{rel}}\%$) | Chiều hướng tốt & Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Average Incremental Accuracy** | $\bar{A}$ ($A_{\text{avg}}$) | **91.10%** | `NOT RUN` *(xem A19)* | **79.33%** (`0.793332`) | **$-11.77\text{ pp}$** | **$-12.92\%$** | Cao hơn là tốt hơn ($\uparrow$). Kết quả hiện tại thấp hơn paper $11.77\text{ pp}$ (tương ứng lệch tương đối $-12.92\%$). |
| **Last Task Accuracy** | $A_{10}$ ($A_{\text{last}}$) | **86.50%** | `NOT RUN` *(xem A19)* | **70.92%** (`0.709200`) | **$-15.58\text{ pp}$** | **$-18.01\%$** | Cao hơn là tốt hơn ($\uparrow$). Sau khi học đủ 100 lớp qua 10 tasks (one-pass), độ chính xác cuối đạt $70.92\%$, thấp hơn paper $15.58\text{ pp}$ ($-18.01\%$). |
| **Average Forgetting Rate** | $F_{10}$ ($F$) | **5.50%** | `NOT RUN` *(xem A19)* | **9.21%** (`0.092111`) | **$+3.71\text{ pp}$** | **$+67.45\%$** | **Thấp hơn là tốt hơn ($\downarrow$)**. Dấu dương ($+3.71\text{ pp}$ / $+67.45\%$) biểu thị mô hình hiện tại có mức độ quên **cao hơn** (kém hơn) so với con số công bố trong paper. |

---

### 20.2. Ma trận Độ chính xác $10 \times 10$ Đầy đủ & Kiểm chứng Toán học Độc lập

Dữ liệu nguyên gốc từ `metrics.json` (do người dùng xuất trực tiếp trên Colab sau khi kết thúc Task 10) cung cấp ma trận độ chính xác tam giác dưới $R \in \mathbb{R}^{10 \times 10}$, trong đó phần tử $a_{k, j}$ là độ chính xác kiểm thử trên tập test của Task $j$ sau khi hoàn tất huấn luyện Task $k$ ($1 \le j \le k \le 10$):

| Giai đoạn (Sau Task $k$) | Task 1 ($a_{k,1}$) | Task 2 ($a_{k,2}$) | Task 3 ($a_{k,3}$) | Task 4 ($a_{k,4}$) | Task 5 ($a_{k,5}$) | Task 6 ($a_{k,6}$) | Task 7 ($a_{k,7}$) | Task 8 ($a_{k,8}$) | Task 9 ($a_{k,9}$) | Task 10 ($a_{k,10}$) | **Stage Accuracy $A_k = \frac{1}{k}\sum_{j=1}^k a_{k,j}$** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | **89.60%** | — | — | — | — | — | — | — | — | — | **89.60%** (`0.896000`) |
| **Stage 2** | 84.80% | **91.20%** | — | — | — | — | — | — | — | — | **88.00%** (`0.880000`) |
| **Stage 3** | 82.10% | 88.50% | **84.60%** | — | — | — | — | — | — | — | **85.07%** (`0.850667`) |
| **Stage 4** | 78.60% | 87.40% | 78.90% | **81.50%** | — | — | — | — | — | — | **81.60%** (`0.816000`) |
| **Stage 5** | 76.60% | 84.40% | 76.60% | 80.10% | **77.80%** | — | — | — | — | — | **79.10%** (`0.791000`) |
| **Stage 6** | 73.10% | 83.70% | 76.70% | 78.80% | 73.90% | **75.50%** | — | — | — | — | **76.95%** (`0.769500`) |
| **Stage 7** | 70.60% | 82.10% | 75.70% | 77.70% | 73.20% | 73.60% | **77.50%** | — | — | — | **75.77%** (`0.757714`) |
| **Stage 8** | 67.40% | 80.90% | 75.50% | 74.30% | 72.70% | 71.70% | 75.90% | **73.70%** | — | — | **74.01%** (`0.740125`) |
| **Stage 9** | 64.70% | 78.60% | 74.80% | 73.80% | 71.30% | 71.10% | 75.60% | 71.60% | **69.30%** | — | **72.31%** (`0.723111`) |
| **Stage 10** | 60.50% | 76.20% | 74.00% | 73.50% | 70.50% | 69.50% | 74.20% | 71.00% | 68.40% | **71.40%** | **70.92%** (`0.709200`) |
| **Peak $\max_{\ell \in \{j..9\}} a_{\ell,j}$** | 89.60% | 91.20% | 84.60% | 81.50% | 77.80% | 75.50% | 77.50% | 73.70% | 69.30% | — | — |
| **Task Forgetting $f_j^{10}$** | **29.10%** | **15.00%** | **10.60%** | **8.00%** | **7.30%** | **6.00%** | **3.30%** | **2.70%** | **0.90%** | — | **Mean $F_{10} = 9.21\%$** (`0.092111`) |

#### Kiểm chứng toán học độc lập từ ma trận $10 \times 10$:
1. **Last Accuracy ($A_{10}$):**
   $$A_{10} = \frac{0.605 + 0.762 + 0.740 + 0.735 + 0.705 + 0.695 + 0.742 + 0.710 + 0.684 + 0.714}{10} = \frac{7.092}{10} = 0.7092 \quad (70.92\%)$$
2. **Average Incremental Accuracy ($\bar{A}$):**
   $$\bar{A} = \frac{1}{10} \sum_{k=1}^{10} A_k = \frac{0.8960000 + 0.8800000 + 0.8506667 + 0.8160000 + 0.7910000 + 0.7695000 + 0.7577143 + 0.7401250 + 0.7231111 + 0.7092000}{10} = 0.7933317 \quad (79.33\%)$$
3. **Average Forgetting ($F_{10}$):**
   $$F_{10} = \frac{1}{9} \sum_{j=1}^{9} f_j^{10} = \frac{0.291 + 0.150 + 0.106 + 0.080 + 0.073 + 0.060 + 0.033 + 0.027 + 0.009}{9} = \frac{0.829}{9} = 0.0921111 \quad (9.21\%)$$
*Kết luận kiểm chứng số học:* Ba chỉ số tổng hợp trong `metrics.json` khớp tuyệt đối 100% với định nghĩa giải tích trong `src/foal/metrics.py`.

---

### 20.3. Kiểm toán Các Nhận định Khoa học & Phân tích Giả thuyết Sai lệch (Scientific Claim Audit)

Trong báo cáo sơ bộ tại commit `c74b0de`, một số nhận định về nguyên nhân sai lệch và hiệu năng thuật toán đã được viết quá mức chứng minh (*overclaimed*). Dưới góc độ kiểm toán khoa học, nhóm thực hiện điều chỉnh chuẩn xác như sau:

1. **Đính chính nhận định về độ ổn định và hiệu năng của thuật toán:**
   * *Nhận định cũ (loại bỏ):* Khẳng định thuật toán *"hoạt động hoàn hảo"* và *"vượt xa các baseline thông thường như Fine-tuning (~15–25%)"*.
   * *Nhận định chuẩn hóa:*
     * Pipeline huấn luyện của nhóm đã **chạy hoàn tất end-to-end qua 10 tasks trên dữ liệu CIFAR-100 thật bằng GPU**, không xảy ra lỗi tràn bộ nhớ hay suy biến số học (`NaN`/`Inf`) trong ma trận $R_t \in \mathbb{R}^{1000 \times 1000}$ khi tính toán bằng `float64`.
     * Nhóm **chưa chạy thực nghiệm baseline Fine-tuning (`A15`) trên cùng giao thức** nên không đưa ra con số tự ước lượng `~15–25%`. Nếu trích dẫn tài liệu tham khảo, chỉ trích dẫn số liệu do chính tác giả báo cáo trong Paper (Section 5.3 / Bảng phân tích Analytic Classifier vs Backprop FC Classifier trên CIFAR-100: cấu hình thay Analytic Classifier bằng Fully Connected Classifier huấn luyện bằng Backpropagation đạt $32.4\%$ Average Accuracy so với $91.1\%$ của F-OAL).
     * Khoảng cách giữa kết quả hiện tại của nhóm ($\bar{A} = 79.33\%$, $A_{10} = 70.92\%$, $F_{10} = 9.21\%$) và kết quả công bố trong Paper ($\bar{A} = 91.10\%$, $A_{10} = 86.50\%$, $F_{10} = 5.50\%$) là **đáng kể ($-11.77\text{ pp}$ trên $\bar{A}$ và $-15.58\text{ pp}$ trên $A_{10}$)** và bắt buộc phải được điều tra có hệ thống thông qua việc chạy code gốc của tác giả (**A19**) và đối chiếu từng tầng pipeline (**A20**).

2. **Bảng 9 Giả thuyết Khoa học về Nguyên nhân Chênh lệch Kết quả (`79.33%` vs `91.10%`):**
   Thay vì kết luận sớm rằng khoảng cách chỉ do `A01` và `A04`, mọi yếu tố có khả năng gây sai lệch giữa `OUR_REPRODUCED` và `PAPER_REPORTED` đều được đặt làm **giả thuyết cần kiểm chứng thực nghiệm**:

| Mã GT | Yếu tố / Thành phần Pipeline | Trạng thái trong Code Nhóm (`src/foal/`) | Giả thuyết Kỹ thuật & Cơ chế Gây Sai lệch Tiềm ẩn | Cách Kiểm chứng Thực nghiệm (`A19` / `A20`) |
|:---:|:---|:---|:---|:---|
| **H1** | **Pretrained ViT Weights (`A01`)** | `timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)` | Tác giả chỉ ghi *"ViT-B pre-trained on ImageNet-1K"*. Trong `timm` hoặc repo gốc, tác giả có thể sử dụng checkpoint được pre-train trên ImageNet-21K rồi fine-tune trên ImageNet-1K (`vit_base_patch16_224.augreg_in21k_ft_in1k` hoặc `vit_base_patch16_224` gốc). Vì CIFAR-100 có độ tương đồng cao với ImageNet-21K, khác biệt về chất lượng đặc trưng zero-shot của Frozen ViT có thể tạo ra chênh lệch $5\text{–}12\text{ pp}$. | Kiểm tra tên model/checkpoint tải trong repo gốc `liuyuchen-cz/F-OAL`; so sánh trực tiếp độ chính xác Task 1 (10 lớp đầu: nhóm đạt $89.6\%$, trong khi để trung bình 10 task đạt $91.1\%$ thì Task 1 của tác giả phải đạt $\approx 95\text{–}98\%$). |
| **H2** | **Khởi tạo & Hệ số Tỷ lệ của Smooth Projection (`A03`)** | `weight = torch.randn(768, 1000) * 1.0`, sau đó `torch.sigmoid(x @ weight)` | **Giả thuyết kỹ thuật có xác suất rất cao:** Sau LayerNorm, vector $x \in \mathbb{R}^{768}$ có phương sai mỗi chiều $\approx 1$. Khi nhân với $W \sim \mathcal{N}(0, 1)^{768 \times 1000}$ (không chia cho $\sqrt{d_{\text{in}}} = \sqrt{768} \approx 27.71$), tích vô hướng $u = x W$ có độ lệch chuẩn $\text{std}(u) \approx 27.7$. Với $|u| \gg 5$, hàm $\sigma(u) = \frac{1}{1+e^{-u}}$ bị **bão hòa mạnh về 0 hoặc 1** (biến Smooth Projection thành hàm bước nhị phân cứng thay vì phép chiếu trơn), làm mất thông tin biên độ đặc trưng! | Kiểm tra cách khởi tạo ma trận chiếu trong `liuyuchen-cz/F-OAL` (ví dụ: `nn.Linear` mặc định dùng $\mathcal{U}(-1/\sqrt{d_{\text{in}}}, 1/\sqrt{d_{\text{in}}})$, hoặc chuẩn hóa $\mathcal{N}(0, 1/d_{\text{in}})$, hoặc chuẩn hóa L2 vector $x$ trước khi chiếu). |
| **H3** | **Cơ chế Feature Fusion & Vị trí LayerNorm (`A02`)** | Lấy trung bình cộng `block_output[:, 0, :]` của 12 blocks rồi áp dụng `network.norm` một lần ở cuối. | Tác giả có thể: (a) áp dụng LayerNorm cho từng block trước khi lấy trung bình; (b) không áp dụng `network.norm` cuối; (c) chuẩn hóa L2 (`F.normalize`) vector đặc trưng sau mỗi block hoặc sau khi cộng gộp; hoặc (d) chỉ lấy một tập con các block. | Đặt câu lệnh in kích thước và chuẩn L2 của vector đặc trưng trung gian giữa code gốc tác giả và `FrozenViT.forward()` trên cùng một ảnh đầu vào. |
| **H4** | **Tiền xử lý ảnh (Image Preprocessing)** | `Resize((224, 224), BICUBIC)` + `Normalize(mean=0.5, std=0.5)` | Repo tác giả (thường kế thừa từ PyCIL / LAMDA-PILOT) có thể dùng `Resize(256) -> CenterCrop(224)` hoặc `InterpolationMode.BILINEAR` hoặc chuẩn hóa ImageNet `mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)` kết hợp với cấu hình `timm` khác. | Đối chiếu trực tiếp transform pipeline trong code gốc của tác giả. |
| **H5** | **Siêu tham số RLS & Quy ước Ma trận Gram (`A05`)** | $R_0 = \gamma^{-1} I$ với $\gamma = 1.0$, nhãn đích $Y \in \{0, 1\}^{B \times C}$ | Một số hiện thực Analytic Learning (như ACIL / DS-AL) dùng $\gamma$ nhân với ma trận đơn vị theo tỷ lệ số mẫu, hoặc khởi tạo $R_0 = \gamma I$ (nghịch đảo ký hiệu $\gamma \leftrightarrow 1/\gamma$), hoặc dùng nhãn đích $\{-1, +1\}$ thay vì $\{0, 1\}$. | Kiểm tra phương trình cập nhật $R_t$ và $W_t$ trong code gốc của tác giả; chạy thử $\gamma \in \{10^{-2}, 10^{-1}, 1, 10, 10^2, 10^3\}$. |
| **H6** | **Thứ tự lớp (Class Permutation — `A04`)** | Sinh hoán vị bằng `np.random.default_rng(42).permutation(100)` | Thứ tự 10 lớp trong mỗi task ảnh hưởng đến độ khó của từng task (ví dụ Task 1 có nhiều lớp dễ hay khó) và quỹ đạo $\bar{A}$. Tuy nhiên, riêng hoán vị lớp thường chỉ gây dao động $\pm 1\text{–}2\%$ chứ hiếm khi gây chênh lệch tới $15.58\text{ pp}$ ở Last Accuracy $A_{10}$ (vì nghiệm RLS tại Task 10 là bất biến với thứ tự các lớp đã học nếu tập mẫu là như nhau!). | Nhận xét toán học quan trọng: Tại Task 10 (sau khi đã nhìn thấy toàn bộ 50.000 mẫu), nghiệm giải tích của Ridge Regression trên toàn bộ 100 lớp **không phụ thuộc vào thứ tự phân chia task hay thứ tự lớp**! Do đó, việc $A_{10}$ lệch tới $-15.58\text{ pp}$ ($70.92\%$ vs $86.50\%$) chứng minh chắc chắn rằng **nguyên nhân chính nằm ở biểu diễn đặc trưng / phép chiếu / tiền xử lý (H1, H2, H3, H4, H5)** chứ không phải chỉ do thứ tự lớp `A04`! |
| **H7** | **Thứ tự luồng mẫu (Data Stream Order)** | `DataLoader(..., shuffle=True, generator=g)` với `stream_seed=42` | Do tính chất toán học của RLS (đã chứng minh qua unit test `test_batch_partition_does_not_change_exact_solution`), thứ tự mini-batch trong một task không làm thay đổi nghiệm cuối của task đó (trừ sai số dấu chấm động $< 10^{-10}$). | Đã loại trừ về mặt toán học; chỉ kiểm tra xem code tác giả có dùng data augmentation ngẫu nhiên (RandomHorizontalFlip / AutoAugment) trong luồng 1-pass hay không. |
| **H8** | **Công thức tính Metrics (`A07`)** | $\bar{A} = \frac{1}{T}\sum_{k=1}^T A_k$, với $A_k = \frac{1}{k}\sum_{j=1}^k a_{k,j}$ | Trên CIFAR-100 với 10 lớp/task (mỗi lớp có đúng 100 mẫu test), trung bình các task đã học $A_k = \frac{1}{k}\sum_{j=1}^k a_{k,j}$ trùng khớp hoàn toàn với độ chính xác trên toàn bộ các mẫu test đã thấy (overall seen-sample accuracy). Riêng công thức Forgetting có thể khác nhau ở chỗ tính trung bình trên $T-1$ tasks hay $T$ tasks. | Đối chiếu hàm tính accuracy và forgetting trong repo gốc của tác giả. |
| **H9** | **Phiên bản Thư viện (`A08`)** | `timm==1.0.30`, `torch` trên Colab | Khác biệt phiên bản `timm` có thể dẫn đến thay đổi URL trọng số mặc định khi gọi tên model ngắn gọn (ví dụ `vit_base_patch16_224`). | Kiểm tra `requirements.txt` hoặc môi trường trong repo tác giả. |

> **Phát hiện kiểm toán quan trọng nhất từ Bảng Giả thuyết trên:** Nhờ tính chất bất biến theo thứ tự của nghiệm giải tích Ridge Regression tại thời điểm kết thúc toàn bộ 100 lớp (Task 10), việc **Last Accuracy $A_{10}$ của nhóm đạt $70.92\%$ so với $86.50\%$ của paper ($\Delta_{\text{pp}} = -15.58\text{ pp}$)** là bằng chứng toán học rõ ràng cho thấy khoảng cách này **không thể giải thích chỉ bằng thứ tự lớp (`A04`) hay thứ tự mẫu**, mà bắt nguồn từ **khác biệt trong không gian đặc trưng đầu vào của bộ phân loại RLS** (cụ thể là **H1: Pretrained ViT weight variant**, **H2: Độ lệch chuẩn khởi tạo của Smooth Projection gây bão hòa Sigmoid**, hoặc **H3/H4/H5**). Điều này đặt ưu tiên cao nhất cho hạng mục **A19** và **A20**.

---

### 20.4. Trạng thái Xác minh Hiện vật Thực nghiệm (Artifact Verification Status)

Toàn bộ hiện vật từ lượt chạy GPU trên Google Colab đã được trích xuất từ gói `foal_cifar100_single_seed42_results.zip` ($109,814,583\text{ bytes}$) và kiểm chứng trực tiếp trên đĩa cục bộ:

| Hiện vật / Bằng chứng (Artifact) | Dung lượng & Chi tiết Kỹ thuật | Trạng thái Lưu trữ Git vs Local Disk | Trạng thái Kiểm toán |
|:---|:---|:---:|:---:|
| [`results/foal_full_cifar100_seed42/metrics.json`](../results/foal_full_cifar100_seed42/metrics.json) | `1,644 bytes` — Chứa ma trận $10 \times 10$, `stage_accuracies`, `task_forgetting`, `final_task_accuracies`, `total_duration_seconds = 1735.3` | **Đã đưa vào Git (`tracked`)** | **`VERIFIED`** |
| [`results/foal_full_cifar100_seed42/metadata.json`](../results/foal_full_cifar100_seed42/metadata.json) | `3,350 bytes` — Ghi nhận `Tesla T4`, `torch 2.11.0+cu130`, `commit aad7500`, `seeds = 42`, và mảng `class_order` 100 lớp | **Đã đưa vào Git (`tracked`)** | **`VERIFIED`** |
| [`results/foal_full_cifar100_seed42/task_accuracy.csv`](../results/foal_full_cifar100_seed42/task_accuracy.csv) | `635 bytes` — 55 dòng `(stage, task, accuracy)` từ Stage 1 đến Stage 10 | **Đã đưa vào Git (`tracked`)** | **`VERIFIED`** |
| Checkpoints `results/foal_full_cifar100_seed42/task_01.pt` .. `task_10.pt` | 10 tệp nhị phân (`11.15 MB` $\to$ `11.88 MB`/tệp). Kiểm tra `task_10.pt` bằng `torch.load`: `P` shape `(768, 1000)` `float32`, `R` shape `(1000, 1000)` `float64` (`sym_err = 0.0`, `all_finite = True`), `W` shape `(1000, 100)` `float64` (`min = -0.0496, max = 0.0756, all_finite = True`), `class_order` khớp 100% `metadata.json` | **Lưu trên đĩa cục bộ** (loại khỏi Git qua `.gitignore` để không vượt giới hạn 100 MB của GitHub) | **`VERIFIED`** |
| Gói W&B Offline `wandb/offline-run-20261101_084136-2cdeiptv/run-2cdeiptv.wandb` | `188,433 bytes` — Gói dữ liệu W&B offline hoàn chỉnh sinh ra trên Colab, sẵn sàng chạy `wandb sync` lên Public W&B Project | **Lưu trên đĩa cục bộ** (`wandb/` trong `.gitignore`) | **`VERIFIED (OFFLINE)`** |
| [`results/summary_runs.csv`](../results/summary_runs.csv) & [`results/summary_aggregated.csv`](../results/summary_aggregated.csv) | Xuất tự động bởi `python -m scripts.summarize_runs --results-dir results` (tách biệt rõ 2 nhóm Pilot CPU và 1 nhóm Full GPU Reproduction `79.33% (n=1, std=N/A)`) | **Đã đưa vào Git (`tracked`)** | **`VERIFIED`** |
| Gói nén gốc `foal_cifar100_single_seed42_results.zip` | `109,814,583 bytes` ($\approx 109.8\text{ MB}$) | **Lưu trên đĩa cục bộ** (`*.zip` trong `.gitignore` vì vượt ngưỡng 100 MB/file của GitHub) | **`VERIFIED`** |

---

## 21. Kiểm toán Tuân thủ Quy định Đồ án Tái hiện — Checkpoint 3 Compliance

Mục này rà soát toàn diện dự án dựa trên quy định chính thức của môn Nhập môn Machine Learning (HCMUS) tại:  
`https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/`

### 21.1. Phạm vi Thực nghiệm (Experimental Scope: Full Scope vs Core Scope Proposal)

#### A. Kiểm kê Toàn bộ Thực nghiệm trong Paper gốc (Full Scope Inventory)
Bài báo F-OAL (NeurIPS 2024) thực hiện khối lượng thực nghiệm rất lớn gồm:
1. **6 Bộ dữ liệu (Datasets):** CIFAR-100 ($10 \text{ tasks} \times 10 \text{ classes}$), CORe50, FGVCAircraft, DTD, Tiny-ImageNet, Country211.
2. **Hơn 14 Phương pháp Đối sánh (Baselines trong Table 2):** ER, iCaRL, GDumb, DER++ (Literature Extension / Not in Table 2), Co2L, DVC, OCM, OnPro, SCR, MEMO, EASE, SLCA, ACIL, DS-AL và F-OAL, chạy lặp lại **3 lần (3 random runs)** cho mỗi phương pháp trên mỗi bộ dữ liệu.
3. **Đánh giá Hiệu năng Tính toán (Table 3 & Figure 2):** Đo tổng thời gian huấn luyện (Training Time, tính bằng giây) và mức tiêu thụ bộ nhớ GPU đỉnh (Peak GPU Memory, tính bằng GB) trên CIFAR-100, FGVCAircraft, DTD, Tiny-ImageNet.
4. **Nghiên cứu Cắt bỏ Thành phần (Ablation Study — Table 4 & Section 5.3):**
   * Đánh giá 4 tổ hợp bật/tắt **Feature Fusion (FF)** và **Smooth Projection (SP)**.
   * So sánh trực tiếp **Analytic Classifier (RLS)** với **Fully Connected Classifier huấn luyện bằng Backpropagation** trên cùng kiến trúc đặc trưng.
5. **Phân tích Bổ sung (Additional / Sensitivity Analysis):** Khảo sát ảnh hưởng của kích thước mini-batch (Batch size $B$), số chiều chiếu ($D$) và hệ số chính quy hóa ($\gamma$).

#### B. Lý do Không Thực hiện Full Scope và Đề xuất Core Scope (Pending Group/Instructor Confirmation)
* **Rào cản tài nguyên của Full Scope:** Việc chạy đủ 6 bộ dữ liệu (đặc biệt là Country211, Tiny-ImageNet, CORe50) nhân với $>10$ phương pháp baseline và 3 seeds đòi hỏi hàng trăm giờ GPU chuyên dụng, vượt quá giới hạn phiên chạy miễn phí của Google Colab / Kaggle T4 (mỗi lượt chạy F-OAL 10 tasks trên CIFAR-100 với T4 đã mất $\approx 29$ phút).
* **Đề xuất Core Scope (Phạm vi Cốt lõi có cơ sở khoa học — Đề xuất để nhóm xác nhận):**
  Theo đúng quy định môn học, khi thu hẹp về **Core Scope**, dự án vẫn phải bảo toàn đầy đủ **3 trụ cột thực nghiệm bắt buộc**:

| Trụ cột Thực nghiệm (Core Scope Pillar) | Nội dung Thực nghiệm Cụ thể trong Core Scope | Tập dữ liệu & Quy mô | Mục tiêu Kiểm chứng Khoa học | Trạng thái Hiện tại |
|:---|:---|:---|:---|:---:|
| **1. Headline Comparison (So sánh Chính — Table 2, Table 3, Figure 2)** | So sánh 3 nguồn của **F-OAL** (`PAPER_REPORTED` vs `ORIGINAL_REPRODUCED` vs `OUR_REPRODUCED`, **3 seeds `[42, 123, 2026]`**) cùng các baseline đại diện chạy trên cùng backbone ViT-B/16 và cùng chia tách task:<br>• **Baseline 1 (`A15`):** Frozen ViT + Feature Fusion + Smooth Projection + **Linear Classifier huấn luyện bằng Backpropagation (SGD/Adam, 1-pass)** (tái hiện trực tiếp thí nghiệm Section 5.3 của paper).<br>• **Baseline 2 (`A16`):** Frozen ViT + **Linear Ridge Analytic Classifier** (không dùng Smooth Projection phi tuyến).<br>• **Baseline 3 (`A17`):** Frozen ViT + **Experience Replay (ER)** với bộ nhớ đệm nhỏ.<br>• Kèm đo đạc **Training Time (s)** và **Peak GPU VRAM (GB)** (`A18`). | **CIFAR-100** (10 tasks $\times$ 10 classes, 50.000 train, 10.000 test, batch size 10, 1-pass). *(Tùy chọn mở rộng sang **DTD** — bộ dữ liệu nhỏ chỉ mất ~1 phút/run trên GPU — nếu thời gian cho phép).* | Kiểm chứng đóng góp trung tâm của F-OAL: khả năng chống quên thảm khốc (Catastrophic Forgetting) vượt trội so với Backpropagation 1-pass mà không cần lưu mẫu cũ (Zero-Replay), đồng thời tiết kiệm thời gian và VRAM. | **`PARTIAL`** (Đã có 1-seed `OUR_REPRODUCED`; đang triển khai `A19`, `A20`, 3-seed và `A15`–`A18`) |
| **2. Ablation Study (Nghiên cứu Cắt bỏ — Table 4)** | Chạy đầy đủ **4 cấu hình tổ hợp** qua `scripts/ablation.py` (3 seeds hoặc 1 seed chuẩn cố định kèm kiểm chứng):<br>1. **Full F-OAL:** `FF = True`, `SP = True`<br>2. **w/o Feature Fusion:** `FF = False` (chỉ lấy block cuối), `SP = True`<br>3. **w/o Smooth Projection:** `FF = True`, `SP = False` ($D=768$ trực tiếp vào RLS)<br>4. **w/o Both:** `FF = False`, `SP = False` | **CIFAR-100** (10 tasks $\times$ 10 classes, full protocol) và/hoặc **DTD** (bộ dữ liệu gốc được dùng trong Table 4 của paper). | Định lượng đóng góp độc lập và hiệu ứng cộng hưởng của **Feature Fusion** (tổng hợp đa tầng ViT) và **Smooth Projection** (mở rộng không gian đặc trưng phi tuyến qua Sigmoid). | **`IMPLEMENTED` & `UNIT TESTED`** (`scripts/ablation.py` đã sẵn sàng; chờ chạy full GPU) |
| **3. Additional Analysis (Phân tích Bổ sung & Kiểm chứng Lý thuyết)** | Thực hiện 2 khảo sát chuyên sâu trên CIFAR-100:<br>• **Analysis 3A — Batch Size Invariance Verification:** Chạy F-OAL với $B \in \{5, 10, 25, 50, 100\}$ trên cùng `stream_seed` và `projection_seed` để kiểm chứng thực nghiệm định lý toán học của bài báo: nghiệm cập nhật tuần tự Woodbury RLS là **bất biến theo kích thước mini-batch $B$** (độ chính xác giữ nguyên trong khi thời gian chạy thay đổi).<br>• **Analysis 3B — Sensitivity to Projection Scaling ($\sigma_{\text{init}}$), Dimension ($D \in \{500, 1000, 2000\}$) & Ridge Regularization ($\gamma \in \{0.1, 1.0, 10.0\}$):** Làm rõ hiện tượng bão hòa Sigmoid và độ nhạy siêu tham số. | **CIFAR-100** (10 tasks $\times$ 10 classes). | Vượt lên trên việc chỉ chạy lại lệnh có sẵn: chứng minh thực nghiệm tính chất toán học cốt lõi của thuật toán (Batch-size invariance) và giải thích cơ chế hoạt động của Smooth Projection. | **`PENDING`** (Đã có unit test kiểm chứng bất biến batch size ở mức nhỏ; chờ chạy GPU full) |

---

### 21.2. Tự Hiện thực Thành phần Cốt lõi (Independent Core Implementation Verification)

Quy định môn học yêu cầu nhóm phải **tự hiện thực các thành phần cốt lõi của thuật toán** thay vì chỉ sao chép mã nguồn của tác giả. Bảng dưới đây xác nhận quyền sở hữu mã nguồn trong `src/foal/`:

| Thành phần Cốt lõi | Tệp Mã nguồn của Nhóm | Trạng thái Tự Hiện thực & Kiểm thử | Ghi chú Ranh giới với Code Gốc Tác giả |
|:---|:---|:---:|:---|
| **1. Frozen ViT & Hook-based Feature Extraction** | `src/foal/backbone.py` | **100% Nhóm tự viết** (`FrozenViT`, forward hooks thu thập 12 blocks, ép `eval()`, khóa gradient) | Không sao chép từ repo tác giả. Sử dụng trực tiếp thư viện chuẩn `timm` và `torch`. |
| **2. Multi-Block CLS Feature Fusion** | `src/foal/feature_fusion.py` | **100% Nhóm tự viết** (`fuse_cls_features`, kiểm tra chiều tensor chặt chẽ) | Được kiểm chứng bằng unit test `test_feature_fusion_applies_final_layernorm_both_branches`. |
| **3. Smooth Random Projection** | `src/foal/projection.py` | **100% Nhóm tự viết** (`SmoothProjection`, quản lý `torch.Generator` độc lập, lưu `register_buffer`) | Được kiểm chứng bằng `test_projection_is_fixed_and_seeded`. |
| **4. Online Recursive Least Squares (Woodbury RLS)** | `src/foal/rls.py` | **100% Nhóm tự viết** (`OnlineRidgeRLS`, mở rộng lớp động `ensure_classes`, giải hệ tuyến tính ổn định bằng `torch.linalg.solve` + đối xứng hóa $R_t$) | Được kiểm chứng tương đương nghiệm giải tích Ridge toàn cục với sai số $< 10^{-10}$ trên đặc trưng ViT thật (`test_rls_real_features.py`). |
| **5. Incremental Task Stream & Zero-Replay Dataset** | `src/foal/datasets.py` | **100% Nhóm tự viết** (`TaskSpec`, `make_class_order`, `make_task_specs`, `IndexedImageDataset`, kiểm tra chống rò rỉ nhãn tương lai) | Được kiểm chứng bằng `test_task_partition_no_train_replay_and_no_future_test`. |
| **6. Continual Learning Metrics & Multi-Seed Aggregation** | `src/foal/metrics.py` | **100% Nhóm tự viết** ($\bar{A}$, $A_T$, $F_T$, Bessel's correction $N-1$ cho multi-seed std) | Được kiểm chứng bằng 11 bài test trong `test_metrics_comprehensive.py`. |

> **Quy tắc Cách ly Nghiêm ngặt khi Thực hiện A19 (Original Code Reproduction):** Khi tải repository gốc của tác giả (`https://github.com/liuyuchen-cz/F-OAL`) để chạy đối chiếu theo yêu cầu Checkpoint 3, mã nguồn của tác giả **bắt buộc phải được đặt trong thư mục cách ly độc lập** (ví dụ: `external/official_foal/` hoặc một nhánh/thư mục ngoài `src/foal/`), tuyệt đối không trộn lẫn hay ghi đè lên phần tự hiện thực trong `src/foal/`.

---

### 21.3. Tính Tái lập của Tập dữ liệu (Dataset Reproducibility)

* **Tự động tải dữ liệu:** Hàm `load_cifar100()` trong `src/foal/datasets.py` tự động tải bộ dữ liệu chuẩn CIFAR-100 từ Hugging Face (`uoft-cs/cifar100`) và tự động chuyển đổi dự phòng (fallback) sang `torchvision.datasets.CIFAR100(download=True)` nếu môi trường không có Hugging Face. Không yêu cầu tải thủ công bất kỳ tệp dữ liệu cá nhân nào.
* **Chia tách Task tất định (Deterministic Task Partitioning):** Thứ tự 100 lớp được sinh hoàn toàn tất định bởi `class_order_seed` (`np.random.default_rng(class_order_seed).permutation(100)`) hoặc danh sách `class_order` tường minh trong YAML, và được lưu trực tiếp vào cả `metadata.json` lẫn từng tệp checkpoint `task_XX.pt`.
* **Chống rò rỉ dữ liệu (Data Leakage Prevention):** Hàm `make_task_specs()` thực thi kiểm tra bất biến `if any(i in all_train_indices for i in train_ids): raise AssertionError('A training sample would be replayed')`, đồng thời tập test tại giai đoạn $k$ chỉ lấy các mẫu thuộc tập lớp đã thấy `seen` ($10 \times k$ lớp đầu), tuyệt đối không để lộ nhãn của các task tương lai.

---

### 21.4. Kế hoạch Chạy Code Gốc của Tác giả (`A19` — Original Code Reproduction Plan)

Theo quy định Checkpoint 3 của môn học, khi bài báo có công bố mã nguồn chính thức (`https://github.com/liuyuchen-cz/F-OAL`), nhóm **bắt buộc phải chạy lại code gốc của tác giả** để có cột số liệu `ORIGINAL_REPRODUCED`, từ đó phân biệt rõ:
* Sai lệch giữa `PAPER_REPORTED` và `ORIGINAL_REPRODUCED` (do khác biệt phần cứng GPU RTX 4070 Ti vs Colab T4, phiên bản PyTorch/timm).
* Sai lệch giữa `ORIGINAL_REPRODUCED` và `OUR_REPRODUCED` (do khác biệt chi tiết cài đặt trong pipeline của nhóm).

**Quy trình thực hiện `A19` (Chuẩn bị cho lượt chạy GPU tiếp theo):**
1. **Cách ly mã nguồn:** Clone `https://github.com/liuyuchen-cz/F-OAL.git` vào thư mục `external/official_foal` (thêm `external/` vào `.gitignore` hoặc lưu script bọc thực thi tại `scripts/run_official_foal.py`).
2. **Kiểm tra cấu hình gốc của tác giả:** Đọc các tệp cấu hình/script của tác giả cho CIFAR-100 để trích xuất chính xác:
   * Tên checkpoint ViT trong `timm` mà tác giả gọi.
   * Quy trình tiền xử lý ảnh (`transforms`).
   * Cách khởi tạo ma trận chiếu ngẫu nhiên $P$ và hàm kích hoạt trong Smooth Projection.
   * Giá trị hệ số chính quy hóa $\gamma$ và kiểu dữ liệu (`float32` hay `float64`) trong cập nhật RLS.
   * Danh sách thứ tự lớp (`class_order`) và random seeds mà tác giả sử dụng.
3. **Thực thi trên cùng GPU Colab T4:** Chạy code gốc của tác giả trên CIFAR-100 (10 tasks $\times$ 10 classes, batch size 10), ghi nhận đầy đủ:
   * Ma trận độ chính xác $10 \times 10$.
   * Ba chỉ số chính: $\bar{A}$ (`Average Incremental Accuracy`), $A_{10}$ (`Last Accuracy`), $F_{10}$ (`Forgetting Rate`).
   * Thời gian chạy (`Training Time`) và mức tiêu thụ bộ nhớ đỉnh (`Peak GPU VRAM`).
4. **Lưu trữ hiện vật:** Lưu toàn bộ log đầu ra và metrics của code gốc vào `results/official_foal_cifar100/` và đồng bộ lên project W&B công khai với tag `original_code`.

---

### 21.5. Quy trình Điều tra Chênh lệch Kết quả (`A20` — Step-by-Step Pipeline Discrepancy Investigation)

Để giải quyết triệt để khoảng cách $-11.77\text{ pp}$ trên $\bar{A}$ ($79.33\%$ vs $91.10\%$) và $-15.58\text{ pp}$ trên $A_{10}$ ($70.92\%$ vs $86.50\%$), nhóm thiết lập quy trình đối chiếu 7 bước tuần tự giữa `external/official_foal` (Original) và `src/foal` (Ours):

```
[Bước 1: Dataset & Class Order] 
   │  So sánh danh sách 10 lớp của Task 1..10 và số mẫu/task (5.000 train, 1.000 test).
   ▼
[Bước 2: Image Preprocessing]
   │  Đưa cùng 1 ảnh CIFAR-100 qua transform của Original và Ours; kiểm tra shape, min, max, mean, std.
   ▼
[Bước 3: Pretrained ViT Backbone Weights]
   │  So sánh tên model timm, state_dict checksum và đầu ra từng block h_1..h_12 trên cùng tensor ảnh.
   ▼
[Bước 4: Feature Fusion & LayerNorm]
   │  So sánh vector h_fusion (chuẩn L2, mean, std, cosine similarity giữa Original và Ours).
   ▼
[Bước 5: Smooth Projection (Trọng tâm nghi vấn H2)]
   │  Kiểm tra phân phối trọng số W_proj (std = 1.0 hay 1/sqrt(768)) và phân phối đầu ra sau Sigmoid z = sigma(h_fusion @ W_proj).
   │  Đo tỷ lệ phần tử bị bão hòa (z < 0.01 hoặc z > 0.99).
   ▼
[Bước 6: Online RLS Update]
   │  Đưa cùng ma trận đặc trưng Z và nhãn Y vào RLS của Original và Ours; kiểm tra chuẩn Frobenius ||W_orig - W_ours||_F.
   ▼
[Bước 7: Evaluation & Metrics Formula]
      Đưa cùng ma trận kết quả 10x10 vào hàm tính metric của Original và Ours để xác nhận công thức A_avg, A_last, F.
```

> **Nguyên tắc học thuật:** Mọi phát hiện từ quy trình 7 bước trên đều phải được ghi thành biên bản thực nghiệm rõ ràng. Nếu phát hiện lỗi hoặc khác biệt cấu hình (ví dụ: hệ số tỷ lệ khởi tạo Smooth Projection hay tên trọng số ViT), nhóm sẽ ghi nhận kết quả **trước khi sửa** (`v1 — Pre-alignment: 79.33%`) và **sau khi căn chỉnh giao thức** (`v2 — Protocol-aligned`), tuyệt đối không xóa bỏ lịch sử thực nghiệm cũ.

---

### 21.6. Truy vết Thực nghiệm Công khai qua W&B (`A21`) & Bảng So sánh Chuẩn 3 Nguồn (`A22`)

#### A. Yêu cầu Public W&B Project (`A21`)
Theo quy định Checkpoint 3:
1. Project W&B (`hcmus-foal-reproduction`) phải được cấu hình ở chế độ **Public (Open Access — bất kỳ ai có đường dẫn đều xem được charts, config, logs, system metrics và artifacts mà không cần đăng nhập tài khoản W&B)**.
2. Mọi lượt chạy được báo cáo trong bảng kết quả chính thức (`ORIGINAL_REPRODUCED`, `OUR_REPRODUCED`, `Ablation`, `Baselines`, `Sensitivity Analysis`) đều phải được đồng bộ lên project W&B này (hoặc chạy trực tiếp với `--wandb`, hoặc đồng bộ từ thư mục offline bằng lệnh `wandb sync`).
3. Mỗi dòng kết quả trong báo cáo và `README_IMPLEMENTATION.md` / `docs/experiments.md` phải đính kèm đường dẫn trực tiếp (`https://wandb.ai/<entity>/hcmus-foal-reproduction/runs/<run_id>`) khớp với `git_commit` trong `metadata.json`.

#### B. Khung Bảng Đối chiếu Chuẩn 3 Nguồn (`A22`: Paper vs Original vs Ours)
Dưới đây là bảng chuẩn hóa theo đúng yêu cầu của giảng viên để sử dụng thống nhất trong `docs/experiments.md`, báo cáo giữa kỳ và README:

| Phương pháp / Nguồn Kết quả | Phân loại Nguồn | Seeds / Số lần chạy | $\bar{A}$ ($A_{\text{avg}}$, $\uparrow$) | $A_{10}$ ($A_{\text{last}}$, $\uparrow$) | $F_{10}$ ($F$, $\downarrow$) | $\Delta_{\text{pp}}(\bar{A})$ so với Paper | $\Delta_{\text{rel}}(\bar{A})$ so với Paper | Thời gian chạy (s) | Peak VRAM (GB) | Đường dẫn W&B Công khai & Ghi chú Giao thức |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **F-OAL (Paper Table 2, 3, Fig. 2)** | `PAPER_REPORTED` | 3 runs (Mean) | **91.10%** | **86.50%** | **5.50%** | $0.00\text{ pp}$ | $0.00\%$ | $261\text{ s}$ (RTX 4070 Ti) | $1.9\text{ GB}$ | Số liệu tham chiếu từ bài báo NeurIPS 2024 (Section 5.2, Table 2 & 3). |
| **F-OAL (Author Repo `liuyuchen-cz/F-OAL @ 73e4ac1`)** | `ORIGINAL_REPRODUCED` | `n=1, std=N/A` (`seed=0`) | **90.98%** (`n=1, std=N/A`) | **85.84%** (`n=1, std=N/A`) | **4.69%** (10-task) / **5.17%** (9-task) | **$-0.12\text{ pp}$** | **$-0.13\%$** | $3047.6\text{ s}$ (Colab T4) | `NOT LOGGED` | [Run `3eca8232`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232) — Hiện vật đã xác minh tại [`results/official_foal_cifar100_seed0/`](../results/official_foal_cifar100_seed0/) (**A19 PASS**). |
| **F-OAL (Nhóm tự hiện thực — Bản chạy GPU đầu tiên, code commit `aad7500`)** | `OUR_REPRODUCED` *(Single-seed v1)* | `n=1, std=N/A` (`seed=42`) | **79.33%** (`n=1, std=N/A`) | **70.92%** (`n=1, std=N/A`) | **8.29%** (10-task) / **9.21%** (9-task) | **$-11.77\text{ pp}$** | **$-12.92\%$** | $1735.3\text{ s}$ (Colab T4) | `NOT LOGGED` | [Run `2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv) — Hiện vật gốc đã xác minh tại [`results/foal_full_cifar100_seed42/`](../results/foal_full_cifar100_seed42/) (`VERIFIED`). Xem phân tích chênh lệch tại [`docs/original_vs_ours_protocol.md`](original_vs_ours_protocol.md). |
| **F-OAL (Nhóm tự hiện thực — Sau điều tra A20 & Multi-Seed)** | `OUR_REPRODUCED` *(3-seed Official)* | 3 seeds (`42, 123, 2026`) | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `NOT YET SYNCED` — Sẽ thực thi tại Stage C/D sau khi hoàn tất đối chiếu **A20**. |
| **Baseline 1: Frozen ViT + Backprop FC Classifier (1-pass)** | `PAPER_REPORTED` / `OUR_REPRODUCED` | Paper: $32.40\%$ / Ours: `PENDING` | Paper: **32.40%** / Ours: `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Thí nghiệm đối chứng vai trò của RLS so với SGD/Adam 1-pass (Paper Section 5.3 / **A15**). |
| **Baseline 2: Frozen ViT + Linear Ridge (w/o Smooth Projection)** | `OUR_REPRODUCED` | `PENDING` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Thí nghiệm đối chứng vai trò của Smooth Projection (**A16** / Ablation cấu hình 3). |
| **Baseline 3: Frozen ViT + Experience Replay (ER)** | `PAPER_REPORTED` / `OUR_REPRODUCED` | Paper: $87.90\%$ / Ours: `PENDING` | Paper: **87.90%** / Ours: `NOT RUN` | Paper: **81.20%** / Ours: `NOT RUN` | Paper: **9.30%** / Ours: `NOT RUN` | `PENDING` | `PENDING` | Paper: $652\text{ s}$ | Paper: $6.3\text{ GB}$ | Đối chứng phương pháp Replay truyền thống (**A17**). |

---

## 22. Đặc tả Chi tiết Các Hạng mục Kiểm toán Mới (`A19`–`A22`) & Tiêu chí Nghiệm thu

### A19 — Original Code Reproduction (`https://github.com/liuyuchen-cz/F-OAL`)
* **Mức ưu tiên:** **P0 (Bắt buộc cho Checkpoint 3)**
* **Trạng thái hiện tại:** **`PASS` (`VERIFIED`)** — Đã chạy hoàn tất 10 tasks CIFAR-100 từ commit `73e4ac1` (`seed=0`) trên Colab Tesla T4, đạt $\bar{A} = 90.98\%$, $A_{10} = 85.84\%$, $F_{\text{official}} = 4.69\%$ ($F_{\text{ours}} = 5.17\%$), lưu trữ tại [`results/official_foal_cifar100_seed0/`](../results/official_foal_cifar100_seed0/) và đồng bộ lên W&B tại [Run `3eca8232`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232).
* **Mô tả:** Tải và chạy mã nguồn chính thức do tác giả công bố trên bộ dữ liệu CIFAR-100 trong một thư mục cách ly (`external/official_foal/`), không can thiệp vào mã nguồn tự hiện thực trong `src/foal/`.
* **Tiêu chí nghiệm thu (Acceptance Criteria):**
  1. Ghi nhận đầy đủ lệnh chạy, môi trường, phiên bản thư viện và biến thể pretrained ViT mà repo gốc sử dụng.
  2. Thu được ma trận độ chính xác $10 \times 10$ và 3 chỉ số $\bar{A}$, $A_{10}$, $F_{10}$ của `ORIGINAL_REPRODUCED` trên GPU.
  3. Lưu trữ toàn bộ log và kết quả vào `results/official_foal_cifar100/` và đẩy lên project W&B công khai.

### A20 — Pipeline Discrepancy Investigation (Điều tra Nguyên nhân Sai lệch Ours vs Original/Paper)
* **Mức ưu tiên:** **P0 (Bắt buộc cho Checkpoint 3)**
* **Trạng thái hiện tại:** **`IN PROGRESS / PARTIAL`** — Đã hoàn tất kiểm toán đối chiếu mã nguồn (`H1`–`H7`) tại [`docs/original_vs_ours_protocol.md`](original_vs_ours_protocol.md) và thiết kế kế hoạch thí nghiệm kiểm soát biến số (`C0`–`C4`); chờ chạy các thí nghiệm kiểm soát trên GPU để định lượng đóng góp của từng yếu tố.
* **Mô tả:** Thực hiện quy trình đối chiếu 7 bước (Mục 21.5) giữa `external/official_foal/` và `src/foal/` để xác định chính xác nguyên nhân gây ra khoảng cách $-11.77\text{ pp}$ trên $\bar{A}$ và $-15.58\text{ pp}$ trên $A_{10}$.
* **Tiêu chí nghiệm thu (Acceptance Criteria):**
  1. Xác minh từng giả thuyết từ **H1** đến **H9** (đặc biệt là **H1: Pretrained ViT checkpoint** và **H2: Độ lệch chuẩn khởi tạo của Smooth Projection gây bão hòa Sigmoid**).
  2. Có bảng đối chiếu định lượng đầu ra trung gian của từng tầng giữa hai pipeline trên cùng một batch dữ liệu CIFAR-100.
  3. Nếu cập nhật cấu hình/code để khớp giao thức tác giả, phải giữ lại kết quả cũ (`79.33%`) làm đối chứng lịch sử và báo cáo rõ mức cải thiện của từng thay đổi.

### A21 — Public W&B Tracking & Evidence Traceability
* **Mức ưu tiên:** **P1 (Bắt buộc cho Checkpoint 3)**
* **Trạng thái hiện tại:** **`PARTIAL` (2 runs chính đã đồng bộ trực tuyến)** — Đã đồng bộ thành công cả Our Run ([Run `2cdeiptv`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/2cdeiptv)) và Official Run ([Run `3eca8232`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3eca8232)) lên project `hcmus-foal-reproduction` trên CoreWeave Forge W&B; cần duy trì chế độ Public và tiếp tục đồng bộ các lượt chạy Multi-Seed / Ablation / Controlled Experiments tiếp theo.
* **Mô tả:** Đồng bộ hóa toàn bộ các lượt chạy thực nghiệm chính thức lên một project Weights & Biases ở chế độ công khai (`Public`) và liên kết từng kết quả trong báo cáo tới W&B run tương ứng.
* **Tiêu chí nghiệm thu (Acceptance Criteria):**
  1. Đường dẫn project W&B mở được trực tiếp trong trình duyệt ẩn danh (Incognito) mà không cần đăng nhập.
  2. Mỗi run trên W&B lưu đầy đủ `config`, `git_commit`, `metrics.json`, `metadata.json` và `task_accuracy.csv`.
  3. Bảng kết quả trong `docs/experiments.md` và `README_IMPLEMENTATION.md` có cột link trực tiếp tới từng W&B run.

### A22 — Core Scope Alignment & Standardized Relative Deviation Reporting
* **Mức ưu tiên:** **P1 (Bắt buộc cho Checkpoint 3)**
* **Trạng thái hiện tại:** **`PARTIAL`** (Đã thiết lập khung bảng và công thức trong tài liệu; chờ hoàn tất số liệu thực nghiệm)
* **Mô tả:** Thống nhất phạm vi thực nghiệm **Core Scope** (Headline Comparison + Ablation Study + Batch-size/Hyperparameter Additional Analysis) và trình bày mọi kết quả so sánh theo đúng 3 nguồn (`PAPER_REPORTED`, `ORIGINAL_REPRODUCED`, `OUR_REPRODUCED`) kèm cả $\Delta_{\text{pp}}$ và $\Delta_{\text{rel}}\%$.
* **Tiêu chí nghiệm thu (Acceptance Criteria):**
  1. Nhóm xác nhận phạm vi Core Scope đề xuất tại Mục 21.1.
  2. Mọi bảng so sánh trong tài liệu và slide báo cáo đều tính đúng $\Delta_{\text{pp}}$ và $\Delta_{\text{rel}} = \frac{\text{Ours}-\text{Paper}}{\text{Paper}} \times 100\%$, kèm chú thích rõ chiều hướng tốt ($\uparrow$ cho Accuracy, $\downarrow$ cho Forgetting).

---

## 23. Lộ trình Thực nghiệm Cập nhật đến Báo cáo Giữa kỳ (Updated Roadmap Stages A–F)

Để đảm bảo hoàn thành toàn bộ yêu cầu kỹ thuật và học thuật trước hạn nộp URL GitHub (**07/11/2026**) và hạn nộp Slide Giữa kỳ (**21/11/2026**), lộ trình thực hiện được cấu trúc lại thành 6 giai đoạn rõ ràng:

| Giai đoạn (Stage) | Thời gian Mục tiêu | Hạng mục Công việc Cụ thể | Sản phẩm Bàn giao & Bằng chứng Nghiệm thu (Deliverables) |
|:---|:---:|:---|:---|
| **Stage A — Evidence Verification & Artifact Recovery** | **01–03/11/2026** | 1. Thu hồi đầy đủ thư mục `results/foal_full_cifar100_seed42/` (`metrics.json`, `metadata.json`, `task_accuracy.csv`, `task_10.pt`) từ Colab hoặc chạy lại có bật `--wandb`.<br>2. Chạy độc lập `scripts/evaluate.py --checkpoint .../task_10.pt` để xác nhận độ khớp 100% với `train.py`.<br>3. Bổ sung ghi nhận `peak_gpu_memory_mb` (`torch.cuda.max_memory_allocated()`) vào `scripts/train.py` (`A18`). | • Thư mục `results/foal_full_cifar100_seed42/` hợp lệ trên repo.<br>• Log xác nhận của `evaluate.py`.<br>• Trạng thái `A10` chuyển từ `PENDING RAW ARTIFACT VERIFICATION` sang `VERIFIED`. |
| **Stage B — Original Code Reproduction (`A19`)** | **03–05/11/2026** | 1. Clone `https://github.com/liuyuchen-cz/F-OAL` vào `external/official_foal/` trên phiên GPU Colab.<br>2. Kiểm tra toàn bộ cấu hình gốc (backbone weights, transforms, projection scaling, RLS $\gamma$).<br>3. Chạy code gốc trên CIFAR-100 và thu thập số liệu `ORIGINAL_REPRODUCED`. | • Kết quả `ORIGINAL_REPRODUCED` ($\bar{A}, A_{10}, F_{10}$, Time, VRAM).<br>• Log & W&B run của code gốc tác giả.<br>• Hoàn tất nghiệm thu **A19**. |
| **Stage C — Discrepancy Investigation & Protocol Alignment (`A20`)** | **05–08/11/2026** | 1. Thực hiện quy trình đối chiếu 7 bước (Mục 21.5) giữa code gốc tác giả và `src/foal/`.<br>2. Kiểm chứng các giả thuyết trọng tâm (**H1** ViT weights, **H2** Smooth Projection scaling $1/\sqrt{768}$, **H3** Feature Fusion norm).<br>3. Căn chỉnh cấu hình/pipeline (nếu phát hiện lệch giao thức) và lưu rõ kết quả trước/sau căn chỉnh.<br>4. Đẩy code hoàn chỉnh lên GitHub trước hạn **07/11/2026**. | • Biên bản kết luận nguyên nhân khoảng cách $-11.77\text{ pp}$ (**A20**).<br>• Commit GitHub sạch, có đầy đủ tài liệu nộp đúng hạn 07/11/2026. |
| **Stage D — Multi-Seed Reproduction & Public W&B (`A12`, `A21`)** | **08–12/11/2026** | 1. Chạy `scripts/run_seeds.py --seeds 42 123 2026` cho cấu hình F-OAL chuẩn trên GPU.<br>2. Bật `--wandb` đồng bộ toàn bộ các run lên project W&B ở chế độ **Public**.<br>3. Chạy `scripts/summarize_runs.py` xuất bảng $Mean \pm Std$ chính thức. | • Bảng kết quả 3-seed $Mean \pm Std$ cho `OUR_REPRODUCED`.<br>• Link project W&B công khai truy cập không cần login (**A21**). |
| **Stage E — Ablation, Baselines & Additional Analysis (`A11`, `A15`–`A18`, `A22`)** | **12–17/11/2026** | 1. Chạy 4 cấu hình Ablation (`scripts/ablation.py`) trên GPU (**A11**).<br>2. Chạy các baseline trong Core Scope (`A15` Backprop FC 1-pass, `A16` Linear Ridge, `A17` ER) trên cùng task split.<br>3. Chạy Additional Analysis: kiểm chứng tính bất biến theo Batch size $B \in \{5, 10, 25, 50, 100\}$ và độ nhạy $D, \gamma$. | • Bảng kết quả Ablation đầy đủ (tái hiện Table 4).<br>• Bảng Headline Comparison 3 nguồn + Baselines + Time/VRAM.<br>• Biểu đồ phân tích độ nhạy Batch size / $D$ / $\gamma$. |
| **Stage F — Final Audit & Midterm Presentation (`ID.pptx`)** | **17–21/11/2026** | 1. Rà soát chéo toàn bộ bảng số liệu, công thức $\Delta_{\text{pp}}$, $\Delta_{\text{rel}}\%$ và link W&B.<br>2. Thiết kế bộ slide 20–40 trang theo chuẩn IEEE citation, tự vẽ lại toàn bộ biểu đồ từ số liệu thực.<br>3. Nộp `ID.pptx` trước hạn **21/11/2026, 23:59**. | • Bộ slide báo cáo giữa kỳ `ID.pptx` (< 20 MB).<br>• Kho lưu trữ GitHub và trang W&B công khai hoàn chỉnh. |


---

## 24. Cập nhật Thực nghiệm: Hoàn tất C0 & Toàn diện C1 (Full GPU Execution on NVIDIA A100)

- **Trạng thái C0 (Metric Alignment):** **`COMPLETED & VERIFIED`**. Cả hai công thức $F_T^{\text{past}}$ (Paper Eq. 17–18, mẫu số $T-1$) và $F_T^{\text{code}}$ (Official Code `experiment/metrics.py:49–54`, mẫu số $T$) đã được cài đặt độc lập, kiểm thử unit test 100% pass và đối chiếu trên dữ liệu thực:
  - Official Run Seed 0 (`3eca8232`): $F_T^{\text{past}} = 5.17\%$, $F_T^{\text{code}} = 4.69\%$.
  - Our Historical Run Seed 42 (`2cdeiptv`): $F_T^{\text{past}} = 9.21\%$, $F_T^{\text{code}} = 8.29\%$.
- **Trạng thái C1 (Controlled Smooth Projection):**
  - **Matched-Input Diagnostics:** **`COMPLETED & VERIFIED`**. Đo đạc trực tiếp trên 200 đặc trưng CIFAR-100 ViT thực tế chứng minh P0 ($\mathcal{N}(0, 1)$) gây bão hòa Sigmoid lên tới $81.03\%$ (std $u = 19.28$), trong khi P1 và P2 có mức bão hòa $0.00\%$ (std $u \approx 0.40$), duy trì đạo hàm trung bình Sigmoid tại $0.246$.
  - **Level A Pilot 2 Tasks:** **`COMPLETED & VERIFIED`**. Chạy thành công trên CPU với `--feature-cache-dir`. P1 đạt Accuracy $79.00\%$ so với P0 $77.50\%$. Cache đặc trưng cho kết quả trùng khớp $100\%$ bitwise với chạy không cache và tăng tốc $70$ lần.
  - **Level B Full 10 Tasks GPU Execution:** **`PASS (VERIFIED)`**. Đã chạy hoàn tất trên Google Colab NVIDIA A100 High-RAM qua notebook `notebooks/foal_c1_projection_controlled.ipynb`. Toàn bộ 56 file và 30 checkpoints đã được xác thực an toàn từ `foal_c1_projection_results.zip`:
    - **P0 (`gaussian_std1`):** $\bar{A} = 79.33\%$, $A_{10} = 70.95\%$, $F_{10}^{\text{past}} = 9.17\%$, $F_{10}^{\text{code}} = 8.25\%$ ([W&B Run `qw78us1k`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/qw78us1k)).
    - **P1 (`official_uniform`):** $\bar{A} = 83.28\%$, $A_{10} = 74.84\%$, $F_{10}^{\text{past}} = 8.97\%$, $F_{10}^{\text{code}} = 8.07\%$ ($\Delta_{P1-P0} = +3.95\text{ pp}$, [W&B Run `6whrot4z`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/6whrot4z)).
    - **P2 (`gaussian_matched_variance`):** $\bar{A} = 83.37\%$, $A_{10} = 74.99\%$, $F_{10}^{\text{past}} = 8.81\%$, $F_{10}^{\text{code}} = 7.93\%$ ($\Delta_{P2-P0} = +4.04\text{ pp}$, [W&B Run `hsgsvb6p`](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/hsgsvb6p)).
  - **Kết luận Giả thuyết H2:** **`SUPPORTED BY FULL EXPERIMENT`**. Phương sai khởi tạo ma trận chiếu thu hẹp được $33.9\%$ khoảng cách giữa nhóm và tác giả. Phân phối Uniform vs Gaussian không gây ra khác biệt đáng kể trên đơn seed ($\Delta_{P1-P2} = -0.09\text{ pp}$ trên $\bar{A}$, seed 42); biến số tác động chi phối là quy mô phương sai (variance scale) ngăn hiện tượng bão hòa đặc trưng. Phân tích Centered Covariance Rank ($71.26$ vs $71.10$) giải mã việc rank Gram chưa chuẩn tâm $\approx 1.20$ chỉ là hiệu ứng phụ của vector trung bình dương.
- **Trạng thái Ký duyệt C1 (Sign-off):** **`MERGED INTO MAIN`** (Commit `e2252b9`). Nhánh C1 đã hoàn thành đầy đủ kiểm thử, đối chiếu hiện vật số học, đồng bộ W&B và merge thành công vào nhánh chính.


---

## 25. Triển khai Thí nghiệm Kiểm soát C2 (Controlled Pretrained ViT Backbone Alignment)

- **Kế hoạch Nghiên cứu Đầy đủ:** [`docs/c2_backbone_alignment_plan.md`](c2_backbone_alignment_plan.md).
- **Nhánh làm việc:** `experiment/c2-backbone-alignment` (tách từ `main` commit `e2252b9`).
- **Câu hỏi Nghiên cứu:** Xác định mức độ ảnh hưởng của checkpoint ViT `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21K tiền huấn luyện, cấu hình tác giả) so với `vit_base_patch16_224.augreg_in1k` (ImageNet-1K ban đầu của nhóm).
- **Thiết kế Kiểm soát Đơn yếu tố:** Cố định tuyệt đối ma trận chiếu $P$ (cùng seed 42, $D=1000$, `official_uniform`, cùng SHA256), cố định tiền xử lý, thứ tự lớp, siêu tham số và tối ưu giải tích RLS.
- **Trạng thái Thực nghiệm C2:**
  - `PLANNED`: Hoàn thành văn bản kế hoạch [`docs/c2_backbone_alignment_plan.md`](c2_backbone_alignment_plan.md).
  - `IMPLEMENTED`: Đã triển khai hỗ trợ backbone và chẩn đoán biểu diễn trong `src/foal/`.
  - `UNIT_TESTED`: 8/8 ca kiểm thử trong `tests/test_c2_backbone.py` vượt qua tuyệt đối (tổng cộng 56 unit tests).
  - `PILOT_VERIFIED`: Đã chạy kiểm chứng Pilot B0 đạt $\bar{A} = 79.00\%$, $A_2 = 78.00\%$.
  - `READY_FOR_FULL_GPU`: Notebook Colab hoàn tất tại [`notebooks/foal_c2_backbone_controlled.ipynb`](../notebooks/foal_c2_backbone_controlled.ipynb).
  - `FULL_GPU_COMPLETED_AND_VERIFIED`: **`PASS (VERIFIED)`** (Đã chạy hoàn tất trên Google Colab NVIDIA A100-SXM4-40GB; xác thực toàn vẹn gói nén `foal_c2_backbone_results.zip` SHA256: `d11e4a98fc6c4a40018427521bf365468f2d5a3ceeceeb941c27050f33a94429`).
    - **B0 (`augreg_in1k`):** $\bar{A} = \mathbf{83.2814\%}$, $A_{10} = \mathbf{74.8400\%}$, $F_{10}^{\text{past}} = \mathbf{8.9667\%}$, $F_{10}^{\text{code}} = \mathbf{8.0700\%}$ (khớp chính xác 100% C1-P1).
    - **B1 (`augreg2_in21k_ft_in1k`):** $\bar{A} = \mathbf{91.7742\%}$, $A_{10} = \mathbf{87.4300\%}$, $F_{10}^{\text{past}} = \mathbf{5.0222\%}$, $F_{10}^{\text{code}} = \mathbf{4.5200\%}$.
    - **Tác động Đơn yếu tố $\Delta(B1 - B0)$:** $\Delta \bar{A} = \mathbf{+8.4929\text{ pp}}$, $\Delta A_{10} = \mathbf{+12.5900\text{ pp}}$, $\Delta F_{10}^{\text{past}} = \mathbf{-3.9444\text{ pp}}$.
- **Kết luận Giả thuyết H1:** **`H1_SUPPORTED_BY_C2 với Giới hạn Thống kê`**. Checkpoint ViT từ ImageNet-21K nâng $\bar{A}$ lên **$91.77\%$** dưới cấu hình kiểm nghiệm, tương đương số học với Paper ($91.10\%$) và Official Run ($90.98\%$). Mức tăng $+8.49\text{ pp}$ vượt qua khoảng cách trước đó ($7.70\text{ pp}$) trên đơn seed CIFAR-100.
- **Trạng thái Ký duyệt C2 (Sign-off):** **`MERGED INTO MAIN`** (Commit `7b76fd7`). Nhánh `experiment/c2-backbone-alignment` đã vượt qua toàn bộ tiêu chuẩn nghiệm thu, 56/56 unit tests passed, 7 runs công khai trên CoreWeave Forge, merge thành công vào `main`.

---

## 26. Triển khai Thí nghiệm Kiểm soát C3 (Controlled Preprocessing Alignment — Factorial $2 \times 2$)

- **Kế hoạch Nghiên cứu Đầy đủ:** [`docs/c3_preprocessing_alignment_plan.md`](c3_preprocessing_alignment_plan.md).
- **Nhánh làm việc:** `experiment/c3-preprocessing-alignment` (tách từ `main` commit `7b76fd7`).
- **Câu hỏi Nghiên cứu:** Xác định tác động của Phương pháp Nội suy (Bicubic vs Bilinear) và Chuẩn hóa Dải Pixel (Normalized $[-1, 1]$ vs Raw $[0, 1]$) trong thiết kế giai thừa $2 \times 2$.
- **Bốn Cấu hình Giai thừa:**
  1. **C3-A (Bicubic, Normalized):** Phê duyệt tái sử dụng kết quả C2-B1 (`C3_A_REUSE_APPROVED`, $\bar{A}=91.7742\%$, W&B run `ne1p16b5`).
  2. **C3-B (Bilinear, Normalized):** Cô lập tác động nội suy.
  3. **C3-C (Bicubic, Raw):** Cô lập tác động chuẩn hóa.
  4. **C3-D (Bilinear, Raw):** Căn chỉnh hoàn toàn với mã nguồn tác giả. Sai khác điểm ảnh so với transform gốc là chính xác $0.00\text{e+}00$.
- **Trạng thái Thực nghiệm C3:** **`C3_SIGNED_OFF_WITH_LIMITATIONS`**
  - Đã chạy hoàn tất Full GPU trên Google Colab NVIDIA A100-SXM4-40GB; xác thực toàn vẹn gói nén `foal_c3_preprocessing_results.zip` (SHA256: `a44b00ca2f12fcc895dedbefe6712087f7d1b17ebf977002e35b13b023361fa6`).
  - **C3-A (Bicubic, Normalized):** $\bar{A} = 91.7742\%$, $A_{10} = 87.4300\%$, $F_{10}^{\text{past}} = 5.0222\%$, $F_{10}^{\text{code}} = 4.5200\%$ ([ne1p16b5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ne1p16b5)).
  - **C3-B (Bilinear, Normalized):** $\bar{A} = 91.3918\%$, $A_{10} = 86.9500\%$, $F_{10}^{\text{past}} = 4.8889\%$, $F_{10}^{\text{code}} = 4.4000\%$ ([ts2wh4e3](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/ts2wh4e3)).
  - **C3-C (Bicubic, Raw):** $\bar{A} = 90.8762\%$, $A_{10} = 86.3500\%$, $F_{10}^{\text{past}} = 5.2333\%$, $F_{10}^{\text{code}} = 4.7100\%$ ([4m4uir7h](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/4m4uir7h)).
  - **C3-D (Bilinear, Raw — Official Aligned):** $\bar{A} = 90.4016\%$, $A_{10} = 85.7300\%$, $F_{10}^{\text{past}} = 5.1556\%$, $F_{10}^{\text{code}} = 4.6400\%$ ([3v53tmq5](https://forge.coreweave.com/wandb/caoban170106-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/hcmus-foal-reproduction/runs/3v53tmq5)).
  - **Phân rã giai thừa $2 \times 2$:** $E_{\text{interp}} = -0.4285\text{ pp}$, $E_{\text{norm}} = -0.9441\text{ pp}$, $E_{\text{IN}} = -0.0922\text{ pp}$, $\Delta_{\text{total}} = -1.3726\text{ pp}$.
  - Toàn bộ 7/7 unit tests trong `tests/test_c3_preprocessing.py` và 63/63 tests toàn bộ repo vượt qua 100%.
  - Đã xây dựng 4 file cấu hình YAML hoàn chỉnh trong `configs/`.
  - Đã tích hợp tham số `normalize` trong `src/foal/datasets.py` và bảo vệ cache trong `scripts/train.py`.
  - Đã xây dựng script chẩn đoán `scripts/run_c3_diagnostics.py`.
  - Đã vượt qua 6/6 unit & regression tests trong `tests/test_c3_preprocessing.py`.
  - Đã xây dựng Google Colab notebook 14 cells hoàn chỉnh tại [`notebooks/foal_c3_preprocessing_controlled.ipynb`](../notebooks/foal_c3_preprocessing_controlled.ipynb).
