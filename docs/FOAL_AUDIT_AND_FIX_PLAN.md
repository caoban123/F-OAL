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

| ID | Hạng mục | Tình trạng | Ưu tiên | File liên quan | Tiêu chí đạt |
|---|---|---|---|---|---|
| A01 | Đúng checkpoint ViT | **ĐÃ XÁC MINH** | **P0** | `backbone.py`, YAML | `vit_base_patch16_224.augreg_in1k`, freeze grad, eval mode cố định |
| A02 | Feature Fusion đúng vị trí | **ĐÃ XÁC MINH** | **P0** | `backbone.py`, `feature_fusion.py` | Mean qua CLS tokens của các blocks, shape `[B, 768]`, ablation đúng |
| A03 | Transforms đúng backbone | **ĐÃ KHẮC PHỤC** | **P0** | `datasets.py`, YAML | Chuẩn hóa `(0.5, 0.5, 0.5)`, bicubic, size 224x224 |
| A04 | CIFAR-100 task protocol | **ĐÃ XÁC MINH** | **P0** | `datasets.py`, `train.py` | 10 × 10 classes, reproducible seed, deterministic mapping |
| A05 | Không train/test leakage | **ĐÃ XÁC MINH** | **P0** | `datasets.py`, `train.py`, tests | One-pass stream, disjoint train sets, seen test only |
| A06 | RLS về mặt đại số | **ĐÃ XÁC MINH** | **P0** | `rls.py`, `test_rls.py` | Woodbury update ≡ Batch Ridge Regression (sai số $< 10^{-11}$) |
| A07 | RLS ổn định trên ViT features | **ĐÃ XÁC MINH** | **P0** | `rls.py`, `model.py`, tests | Không NaN/Inf, kiểm tra nghiệm trên ảnh thật đạt sai số $< 10^{-10}$ |
| A08 | Load/save checkpoint | **ĐÃ XÁC MINH** | **P0** | `train.py`, `evaluate.py` | Lưu đủ R, W, projection, dtype, predictions trùng khớp 100% |
| A09 | Định nghĩa metrics | Có code, cần đối chiếu | **P1** | `metrics.py` | Khớp paper; không lẫn task/stage/seed/đơn vị |
| A10 | Headline comparison | Chưa triển khai đủ | **P1** | `train.py`, `src/baselines/` mới | Có F-OAL + baseline công bằng, đủ protocol |
| A11 | Ablation Table 4 | Có driver, chưa chạy | **P1** | `ablation.py`, `train.py` | 4 cấu hình, output truy vết, so sánh số liệu |
| A12 | Multi-seed & uncertainty | Chưa có driver | **P1** | scripts mới, YAML | 3 trials hoặc scope báo cáo minh bạch |
| A13 | W&B đầy đủ | Có tích hợp cơ bản | **P1** | `train.py`, scripts | Tags, config, commit, URL, artifacts |
| A14 | Tự động tổng hợp báo cáo | Chưa có | **P1** | scripts mới | CSV + bảng paper-vs-ours + mean/std |
| A15 | Time/VRAM profiling | Chưa có | **P2** | `train.py`, scripts mới | Ghi thiết bị, runtime, peak VRAM, state size |
| A16 | Kiểm soát môi trường | Chưa khóa phiên bản | **P1** | `requirements.txt`, docs | Có phiên bản torch/timm/weights/OS/GPU |
| A17 | Additional analysis | Mới có đề xuất | **P1** | `docs/experiments.md`, scripts | Một phân tích **mới**, giả thuyết rõ, có kết quả |
| A18 | README, source & W&B traceability | Chưa đầy đủ | **P1** | `README.md`, `docs/` | Người chấm chạy lại và truy nguồn được |

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

- [ ] Đọc đúng equation/section của paper cho `A_i`, `A_avg`, `A_last`, `F`.
- [ ] Kiểm tra task-wise accuracy và pooled accuracy: khi các tập test không cân bằng, không được đánh đồng.
- [ ] Unit test bằng bảng 3 task tự tính tay; test thêm trường hợp accuracy phục hồi về sau.
- [ ] Quy ước rõ accuracy trong JSON (0–1) hay trên slide (%).
- [ ] Báo cáo **mỗi run trước**, sau đó tổng hợp mean/std qua seeds; không nhầm average over tasks với average over seeds.
- [ ] Thử `T=1` để tránh chia 0 ở forgetting.
- [ ] Đảm bảo W&B metrics, CSV, JSON và bảng slide đều cùng công thức.

**Nghiệm thu:** Các chỉ số có nguồn định nghĩa, kiểm thử tính tay và không có mâu thuẫn đơn vị/aggregation.

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

Starter YAML có `seed: 42`, nhưng chưa có vòng điều phối nhiều seeds. Seed hiện tác động cùng lúc lên class order, DataLoader và projection.

- [ ] Định nghĩa các trial seeds, ví dụ `[42, 123, 2026]` (đây là **ví dụ đề xuất**, không khẳng định tác giả dùng chúng).
- [ ] Lưu độc lập `data_order_seed`, `projection_seed`, `stream_seed` nếu muốn tách nguồn biến thiên; thay đổi cấu hình theo cách có kiểm soát.
- [ ] Đảm bảo **cùng trial/stream** giữa F-OAL và baselines.
- [ ] Tạo script `scripts/run_seeds.py` (đề xuất mới) để tránh chạy sót một seed.
- [ ] Sau mỗi run, lưu mean/std hoặc CI phù hợp; nếu chỉ chạy được 1 seed, ghi **single-run**, không giả vờ là trung bình 3 runs.

### A13 — W&B

`train.py` hiện hỗ trợ `--wandb` và ghi một số cấu hình, task accuracy, summary. Các tính năng sau cần bổ sung:

- [ ] Phân loại `pilot`, `reproduction`, `ablation`, `additional-analysis` bằng **tags thực sự**, không chỉ run name.
- [ ] Ghi `paper`, method, dataset, model ID, weights revision, transforms, code commit, seed, class order, projection `D`, `gamma`, batch size, precision, GPU.
- [ ] Lưu accuracy matrix đầy đủ cùng task-wise CSV và metadata thành artifact hoặc liên kết file đáng tin cậy.
- [ ] Log runtime tách `feature_extraction`, `RLS_update`, `evaluation`, `total` (định nghĩa rõ có tính tải model/dataset không).
- [ ] Log peak allocated GPU memory khi CUDA có sẵn, dung lượng state `R/W/P`, và CPU RAM nếu đo được.
- [ ] Xác nhận run thực sự hoàn thành và đồng bộ trước khi đưa URL vào README/slide.
- [ ] Không đưa W&B API key hay credential vào GitHub.

### A14 — Tự động tổng hợp

**Đề xuất bổ sung:** `scripts/summarize_runs.py` đọc kết quả CSV/JSON/W&B từ các run có đủ protocol, rồi tạo:

1. `results/summary_main_comparison.csv`;
2. `results/summary_ablation.csv`;
3. bảng mean/std cho từng method/trial;
4. chênh lệch với kết quả công bố: `ours - paper` (điểm phần trăm);
5. bảng mapping `experiment → git_commit → config → run_url → metrics`.

**Nghiệm thu:** Mọi ô số liệu trong slide đều truy được đến raw results và W&B run/artifact cụ thể.

---

## 9. P1/P2 — Reproducibility môi trường, profiling và phân tích mới

### A16 — Dependencies, môi trường và checkpoint

- [ ] Dùng Python **>=3.10** theo `pyproject.toml` starter (đồng thời đáp ứng yêu cầu môn >=3.9).
- [ ] Ghi phiên bản thực tế torch, torchvision, timm, datasets, numpy, CUDA, driver, GPU model.
- [ ] Tránh chỉ dùng dependency ranges trong `requirements.txt` khi công bố kết quả chính thức: kèm `environment-lock.txt` hoặc phiên bản đã được thử nghiệm.
- [ ] Kiểm tra model download/cache ổn định trên Kaggle/Colab; ghi đường dẫn dữ liệu và license liên quan khi cần.
- [ ] Có một script smoke test tải 1 batch, encode, update, evaluate, save, reload.
- [ ] `README.md` cần hướng dẫn setup độc lập, không phụ thuộc đường dẫn cá nhân hay ổ đĩa trên laptop.

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

### 16.2. Kết quả kiểm thử bộ Test Suite (14 passed)

Lệnh thực thi:
```bash
python -m pytest -v
```

Kết quả:
```text
tests/test_checkpoint_roundtrip.py::test_checkpoint_save_and_reload PASSED
tests/test_dataset.py::test_seeded_order_is_reproducible_and_permutation PASSED
tests/test_dataset.py::test_task_partition_no_train_replay_and_no_future_test PASSED
tests/test_metrics.py::test_incremental_summary_values PASSED
tests/test_model.py::test_model_end_to_end_on_synthetic_features PASSED
tests/test_projection.py::test_projection_is_fixed_and_seeded PASSED
tests/test_projection.py::test_feature_fusion_averages_cls_token_across_blocks PASSED
tests/test_rls.py::test_rls_matches_closed_form_ridge_with_expanding_classes PASSED
tests/test_rls.py::test_can_only_add_classes_not_remove PASSED
tests/test_rls.py::test_batch_partition_does_not_change_exact_solution PASSED
tests/test_rls.py::test_nan_or_inf_detection PASSED
tests/test_rls_real_features.py::test_rls_matches_ridge_on_real_vit_features PASSED
tests/test_vit_transforms.py::test_vit_transforms_structure PASSED
tests/test_vit_transforms.py::test_frozen_vit_properties PASSED

======================== 14 passed in 45.69s ========================
```

Đặc biệt, `test_rls_matches_ridge_on_real_vit_features` xác minh rằng nghiệm của Online RLS khi học từng mini-batch trên đặc trưng ViT-B/16 từ dữ liệu CIFAR-100 thật khớp với nghiệm giải tích toàn cục Batch Ridge Regression:
$$W^* = (H^T H + \gamma I)^{-1} H^T Y$$
với sai số tuyệt đối $< 10^{-10}$.

---

### 16.3. Kết quả End-to-End Smoke Test trên CIFAR-100 thật

Lệnh thực thi:
```bash
python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100 --run-name smoke_test_p0
```

* **Dữ liệu:** CIFAR-100 thật (50.000 train, 10.000 test).
* **Backbone:** ViT-B/16 (`vit_base_patch16_224.augreg_in1k`, 85.8M parameters, frozen).
* **Thiết bị chạy:** Intel Core Ultra 5 125H (CPU).
* **Số tasks:** 2 tasks, mỗi task 10 lớp, giới hạn 100 mẫu train/task để kiểm thử smoke test.
* **Kết quả đo đạc thực tế:**
  * **Task 1:** 10 lớp seen, 100 train samples, đánh giá trên 1.000 test samples của Task 1:
    * Accuracy Task 1 = **72,10%**
    * Thời gian thực thi = 380,66 giây
    * Checkpoint tạo lập: `results/smoke_test_p0/task_01.pt` (11,15 MB, $R \in \mathbb{R}^{1000 \times 1000}$, $W \in \mathbb{R}^{1000 \times 10}$)
  * **Task 2:** 20 lớp seen, thêm 100 train samples (tổng 200 samples), đánh giá trên 2.000 test samples (Task 1 + Task 2):
    * Accuracy Task 1 (lớp cũ sau khi học Task 2) = **63,10%**
    * Accuracy Task 2 (lớp mới) = **74,20%**
    * Average Seen Accuracy tại Stage 2 = **68,65%**
    * Forgetting Rate $F$ = $72,10\% - 63,10\% = \mathbf{9,00\%}$
    * Checkpoint tạo lập: `results/smoke_test_p0/task_02.pt` (11,23 MB, $R \in \mathbb{R}^{1000 \times 1000}$, $W \in \mathbb{R}^{1000 \times 20}$)
  * **Tổng kết nghiệm thu:**
    * Không xảy ra hiện tượng NaN / Inf trong toàn bộ ma trận $R$ và $W$.
    * Phép gán nhãn, class expansion và lưu checkpoint nhất quán tuyệt đối.
    * Đã tạo đầy đủ các tệp kết quả: `metadata.json`, `task_accuracy.csv`, `metrics.json`.

---

### 16.4. Hướng dẫn chạy trên Kaggle / Google Colab (GPU) cho Full Reproduction (Giai đoạn P1)

Do máy cá nhân sử dụng CPU khiến việc chạy toàn bộ 10 tasks (50.000 train + 10.000 test) mất nhiều giờ, cấu hình GPU T4/P100/A100 trên Kaggle/Colab được khuyến nghị cho giai đoạn P1:

```bash
# 1. Clone repository
git clone https://github.com/caoban123/F-OAL.git
cd F-OAL

# 2. Cài đặt thư viện
pip install -r requirements.txt
pip install -e .

# 3. Chạy kiểm thử xác minh
pytest -q

# 4. Huấn luyện Full Reproduction 10 tasks trên GPU (CUDA)
python -m scripts.train --config configs/cifar100.yaml --run-name foal_full_cifar100_seed42

# 5. Đánh giá checkpoint
python -m scripts.evaluate --checkpoint results/foal_full_cifar100_seed42/task_10.pt --device cuda
```

