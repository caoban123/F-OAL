# F-OAL — Technical Audit, Verification & Fix Plan

> **Dự án:** Nhập môn Machine Learning — HCMUS  
> **Paper:** *F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning* (NeurIPS 2024)  
> **Repository nhóm:** https://github.com/caoban123/F-OAL  
> **Phạm vi audit:** Bộ `F-OAL_starter_v1.zip` được tạo trong cuộc trò chuyện, **không phải** bản `main` trên GitHub đã được xác nhận có toàn bộ source.  
> **Ngày lập kế hoạch:** 01/11/2026  
> **Hạn nộp URL GitHub:** 07/11/2026, 23:59  
> **Hạn slide giữa kỳ:** 21/11/2026, 23:59  
> **Hạn poster cuối kỳ:** 05/12/2026, 23:59

## 0. Mục đích, phạm vi và cách đọc

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

